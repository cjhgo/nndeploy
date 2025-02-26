#include <iostream>

#include "nndeploy/inference/rknn/rknn_inference_param.h"
#include "segp_post.h"

namespace pipeline::segp_post {

using nndeploy::inference::RknnInferenceParam;
void rk_custom(nndeploy::base::Param *param) {
  RknnInferenceParam *rknn_inference_param =
      dynamic_cast<RknnInferenceParam *>(param);
  rknn_inference_param->input_data_type_ = RKNN_TENSOR_UINT8;

  std::cout << "rk_custom done" << std::endl;
}
void quanrk_custom(nndeploy::base::Param *param) {
  RknnInferenceParam *rknn_inference_param =
      dynamic_cast<RknnInferenceParam *>(param);
  rknn_inference_param->input_data_type_ = RKNN_TENSOR_UINT8;
  rknn_inference_param->output_data_types_ = {
      RKNN_TENSOR_INT8, RKNN_TENSOR_INT8, RKNN_TENSOR_INT8,
      RKNN_TENSOR_FLOAT32};

  std::cout << "rk_custom done" << std::endl;
}

using namespace pipeline::yolo_utils;
using nndeploy::device::Tensor;

std::tuple<int32_t, float> get_zpscale(YoloCtx &ctx, int index) {
  RknnInferenceParam *rknn_inference_param =
      dynamic_cast<RknnInferenceParam *>(ctx.infer->getParam());
  auto zp = rknn_inference_param->output_zero_points_[index];
  auto scale = rknn_inference_param->output_scales_[index];
  return {zp, scale};
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
  inline float area() const { return xywh[2] * xywh[3]; }
};
float sigmod(float x) { return 1 / (1 + std::exp(-x)); }

int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale) {
  float dst_val = (f32 / scale) + zp;
  return std::clamp((int8_t)dst_val, (int8_t)-128, (int8_t)127);
}
float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) {
  return ((float)qnt - (float)zp) * scale;
}

void find_nms_box(YoloCtx &ctx, float iou_thresh = 0.7f);
void fill_cut_seg(YoloCtx &ctx);
void fill_cut_pose(YoloCtx &ctx);

void yolo_quan8_segp_post(YoloCtx &ctx) {


  find_nms_box(ctx);
  if(ctx.boxes.size() > 0){
    fill_cut_seg(ctx);
    fill_cut_pose(ctx);
  }

}

// 索引处理函数（保持不变）
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

// 核心处理函数
void find_nms_box(YoloCtx &ctx, float iou_thresh) {
  // 获取量化参数
  auto [box_zp, box_scale] = get_zpscale(ctx, 1);
  cv::Mat box_mat(65, 8400, CV_8SC1, ctx.output_tensors[1]->getData());
  std::vector<CutBox> cut_result;
  auto conf_thresh = (int8_t)box_zp;
  for (int j = 0; j < 8400; j++) {
    auto conf = box_mat.at<int8_t>(64, j);
    if (conf > conf_thresh) {
      auto [cur_h, cur_w, stride] = process_index(j);
      float colj[65];
      for (int k = 0; k < 65; k++) {
        colj[k] = deqnt_affine_to_f32(box_mat.at<int8_t>(k, j), box_zp,
                                       box_scale);
      }
      auto conf = colj[64];
      auto cls_id = 0;
      std::cout << "at j is conf : " << j << " " << conf << std::endl;
      cv::Mat box64 = cv::Mat(4, 16, CV_32FC1, colj).clone();
      cut_result.push_back({j, cur_h, cur_w, stride, conf, cls_id, box64});
    }
  }

  std::vector<cv::Rect2d> boxes;
  std::vector<float> scores;

  for (auto &cut_box : cut_result) {
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
    x1 = std::max(0.0f, x1);
    y1 = std::max(0.0f, y1);
    w = std::min(w, 640.0f - x1);
    h = std::min(h, 640.0f - y1);
    cut_box.xywh = {x1, y1, w, h};

    cut_box.conf = sigmod(cut_box.conf);
    boxes.push_back(cv::Rect2d(x1, y1, w, h));
    scores.push_back(cut_box.conf);
  }
  std::vector<int> indices;
  _NMSBoxes(boxes, scores, 0.0, iou_thresh, indices);
  std::sort(indices.begin(), indices.end(), [&cut_result](int a, int b) {
    return cut_result[a].area() > cut_result[b].area();
  });

  std::transform(indices.begin(), indices.end(), std::back_inserter(ctx.boxes),
                 [&cut_result](int idx) {
                   auto &cut_box = cut_result[idx];
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

void fill_cut_seg(YoloCtx &ctx){
  cv::Mat proto(32, 160 * 160, CV_32FC1, ctx.output_tensors[3]->getData());
  auto seg_mat = cv::Mat(32, 8400, CV_8SC1, ctx.output_tensors[2]->getData());//!!! index 2
  auto[seg_zp, seg_scale] = get_zpscale(ctx, 0);
  cv::Mat seg_all;
  auto& detects = ctx.boxes;
  for (auto &detect : detects) {
    cv::Mat seg_i(1, 32, CV_32FC1);
    auto index = detect.index;
    for (int i = 0; i < 32; i++) {
      seg_i.ptr<float>(0)[i] = deqnt_affine_to_f32(seg_mat.ptr<int8_t>(i)[index], seg_zp, seg_scale);
    }
    seg_all.push_back(seg_i);
  }
  cv::Mat mat_mul = (seg_all * proto) > 0.5;
  for (int i = 0; i < detects.size(); i++) {
    cv::Mat sega(160, 160, CV_8UC1, mat_mul.ptr<uchar>(i));
    cv::Mat maski; cv::resize(sega, maski, cv::Size(640, 640), cv::INTER_LINEAR);
    detects[i].mask = cv::Mat(640, 640, CV_8UC1, cv::Scalar(0));
    auto roi = cv::Rect(detects[i].rect.x, detects[i].rect.y, detects[i].rect.width, detects[i].rect.height);
    maski(roi).copyTo(detects[i].mask(roi));
  }
}
void fill_cut_pose(YoloCtx &ctx){
  ctx.output_tensors[0]->getDesc().print();
  auto pose_mat = cv::Mat(51, 8400, CV_8SC1, ctx.output_tensors[0]->getData());
  auto[pose_zp, pose_scale] = get_zpscale(ctx, 2);
  std::cout << "pose zp scale " << pose_zp << " " << pose_scale << std::endl;

  auto& detects = ctx.boxes;
  for(auto& box : detects){
    auto index = box.index;
    box.kpts.resize(17);
    cv::Mat kpts_mat = cv::Mat(17, 3, CV_32FC1);

    std::cout << " at ? get ? " << " " << index << std::endl;
    for(int i = 0; i < 17; i++){
      for(int j = 0; j < 3; j++){
        auto value = pose_mat.ptr<int8_t>(i*3+j)[index];
        kpts_mat.ptr<float>(i)[j] = deqnt_affine_to_f32(value, pose_zp, pose_scale);
      }
    }
    std::cout << kpts_mat.ptr<float>(0)[0] << std::endl;
    set_cut_pose(box, kpts_mat);
  }
}

}  // namespace pipeline::segp_post