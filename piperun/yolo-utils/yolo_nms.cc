#include "yolo_nms.h"

namespace pipeline::yolo_utils {

void non_max_suppression(std::vector<YoloBox> *out,
                         nndeploy::device::Tensor *tensor_prediction,
                         int nc, float conf_thres, float iou_thres,
                         const std::vector<int> &classes, int max_det) {
  auto hh = tensor_prediction->getShapeIndex(1);
  auto ww = tensor_prediction->getShapeIndex(2);
  cv::Mat prediction(hh, ww, CV_32FC1, tensor_prediction->getData());
  int merge = prediction.rows;
  int nm = merge - nc - 4;
  int mi = 4 + nc;
  cv::Mat pred = prediction.t();

  std::vector<YoloBox> ret;

  std::vector<cv::Rect2d> boxes;
  std::vector<float> scores;
  for (int i = 0; i < pred.rows; ++i) {
    cv::Mat row = pred.row(i);
    cv::Mat box = row.colRange(0, 4);
    cv::Mat conf = row.colRange(4, mi);
    cv::Mat mask = row.colRange(mi, merge);

    double max_val;
    cv::Point max_loc;
    cv::minMaxLoc(conf, nullptr, &max_val, nullptr, &max_loc);
    if (max_val > conf_thres) {
      auto center_x = box.at<float>(0, 0);
      auto center_y = box.at<float>(0, 1);
      auto width = box.at<float>(0, 2);
      auto height = box.at<float>(0, 3);
      auto left = center_x - width / 2;
      auto top = center_y - height / 2;
      cv::Rect2d rect(left, top, width, height);
      auto conf_val = static_cast<float>(max_val);
      auto cls_id = max_loc.y;
      boxes.push_back(rect);
      scores.push_back(conf_val);
      ret.push_back({rect, conf_val, cls_id, mask.clone()});
    }
  }

  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, scores, 0.0, iou_thres, indices);

  std::transform(indices.begin(), indices.end(), std::back_inserter(*out),
                 [&ret](int idx) { return ret[idx]; });
}

std::array<int, 3> process_index(int value) {
  int cur_h, cur_w, stride;
  if (value < 6400) {
    cur_h = value / 80;
    cur_w = value % 80;
    stride = 8;
  } else if (value < 8000) {
    value -= 6400;
    cur_h = value / 40;
    cur_w = value % 40;
    stride = 16;
  } else {
    value -= 8000;
    cur_h = value / 20;
    cur_w = value % 20;
    stride = 32;
  }
  return {cur_h, cur_w, stride};
}

struct CutBox {
  int index;
  int cur_h;
  int cur_w;
  int stride;
  float conf;
  int cls;
  cv::Mat box64;
  std::array<float, 4> xyxy;
  std::array<float, 4> xywh;
};

float sigmod(float x) { return 1 / (1 + std::exp(-x)); }

void cut_non_max_suppression(std::vector<YoloBox> *out,
            nndeploy::device::Tensor* tensor_prediction,
                             int nc,
                             float conf_thres, float iou_thres,
                             const std::vector<int> &classes, int max_det) {

  auto hh = tensor_prediction->getShapeIndex(1);
  auto ww = tensor_prediction->getShapeIndex(2);
  cv::Mat prediction(hh, ww, CV_32FC1, tensor_prediction->getData());
  cv::Mat tensor_t = prediction.t();
  auto cut_result = new std::vector<CutBox>();
  auto conf_rows = tensor_t.colRange(64, 64 + nc);
  for (int j = 0; j < 8400; j++) {
    auto conf_row = conf_rows.row(j);
    double max_val;
    cv::Point max_loc;
    cv::minMaxLoc(conf_row, nullptr, &max_val, nullptr, &max_loc);
    if (max_val > -12) {
      std::cout << j << " " << max_val << std::endl;
      auto [cur_h, cur_w, stride] = process_index(j);
      auto conf = static_cast<float>(max_val);
      auto cls_id = max_loc.y;
      cv::Mat box64 = cv::Mat(4, 16, CV_32FC1, tensor_t.ptr<float>(j)).clone();
      cut_result->push_back({j, cur_h, cur_w, stride, conf, cls_id, box64});
    }
  }
  std::cout << "cut_result size is " << cut_result->size() << std::endl;
  std::vector<cv::Rect2d> boxes;
  std::vector<float> scores;

  for (auto &cut_box : *cut_result) {
    auto box64 = cut_box.box64;
    static cv::Mat data = (cv::Mat_<float>(16, 1) << 0, 1, 2, 3, 4, 5, 6, 7, 8,
                           9, 10, 11, 12, 13, 14, 15);
    cv::Mat rowmax;
    cv::reduce(box64, rowmax, 1, cv::ReduceTypes::REDUCE_MAX);
    auto minus_max = box64 - cv::repeat(rowmax, 1, 16);
    for (int i = 0; i < 4; i++) {
      cv::Mat rowi = minus_max.row(i);
      cv::Mat rowi_exp;
      cv::exp(rowi, rowi_exp);
      auto sum_val = cv::sum(rowi_exp)[0];
      box64.row(i) = rowi_exp / sum_val;
    }
    cv::Mat dfl = box64 * data;
    auto cur_h = cut_box.cur_h;
    auto cur_w = cut_box.cur_w;
    auto stride = cut_box.stride;
    float x1 = (cur_w + 0.5 - dfl.ptr<float>(0)[0]) * stride;
    float y1 = (cur_h + 0.5 - dfl.ptr<float>(0)[1]) * stride;
    float x2 = (cur_w + 0.5 + dfl.ptr<float>(0)[2]) * stride;
    float y2 = (cur_h + 0.5 + dfl.ptr<float>(0)[3]) * stride;
    float w = x2 - x1;
    float h = y2 - y1;
    cut_box.xyxy = {x1, y1, x2, y2};
    cut_box.xywh = {x1, y1, w, h};

    cut_box.conf = sigmod(cut_box.conf);
    boxes.push_back(cv::Rect2d(x1, y1, w, h));
    scores.push_back(cut_box.conf);
  }

  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, scores, 0.0, iou_thres, indices);
  std::transform(indices.begin(), indices.end(), std::back_inserter(*out),
                 [&cut_result](int idx) {
                   auto &cut_box = (*cut_result)[idx];
                   auto [x, y, w, h] = cut_box.xywh;
                   return YoloBox{cv::Rect2d(x, y, w, h),
                                  cut_box.conf,
                                  cut_box.cls,
                                  {},
                                  {},
                                  cut_box.cur_h,
                                  cut_box.cur_w,
                                  cut_box.stride,
                                  cut_box.index};
                 });
}
}  // namespace pipeline::yolo_utils