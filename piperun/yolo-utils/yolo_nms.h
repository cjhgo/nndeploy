#pragma once
#include <opencv2/opencv.hpp>

#include "nntaskflow/nndep.h"

namespace pipeline::yolo_utils {
struct YoloBox {
  cv::Rect2d rect;
  float conf;
  int cls;
  cv::Mat mask;
  std::vector<std::array<float, 3> > kpts;
  int cur_h;
  int cur_w;
  int stride;
  int index;
};
// cv::Mat crop_seg_byproto(YoloBox& box, cv::Mat proto);

void _NMSBoxes(const std::vector<cv::Rect2d>& boxes,
               const std::vector<float>& scores, float score_threshold,
               float iou_threshold, std::vector<int>& indices);

// 实现非极大值抑制（NMS）
void non_max_suppression(std::vector<YoloBox>* out,
                         nndeploy::device::Tensor* prediction, int nc = 80,
                         float conf_thres = 0.25, float iou_thres = 0.7,
                         const std::vector<int>& classes = {},
                         int max_det = 300);

void cut_non_max_suppression(std::vector<YoloBox>* out,
                             nndeploy::device::Tensor* prediction, int nc = 80,
                             float conf_thres = 0.25, float iou_thres = 0.7,
                             const std::vector<int>& classes = {},
                             int max_det = 300);

std::vector<cv::Mat> draw_yolobox(cv::Mat input, std::vector<YoloBox>& results);

}  // namespace pipeline::yolo_utils