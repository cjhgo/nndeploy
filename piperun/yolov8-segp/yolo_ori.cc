#include "segp_post.h"
#include <iostream>
namespace pipeline::segp_post{

using namespace pipeline::yolo_utils;

void yolo_det_post(YoloCtx& ctx){
  non_max_suppression(&ctx.boxes, ctx.output_tensors[0].get(), ctx.nc);
}

cv::Mat crop_seg_byproto(YoloBox& box, cv::Mat proto){
    cv::Mat mask = box.mask;
    std::cout << mask.size();

    cv::Mat mask_mul = mask * proto;
    cv::Mat mask_160(160, 160, CV_32FC1, mask_mul.data);
    cv::Mat mask_640;
    cv::resize(mask_160, mask_640, cv::Size(640, 640));

    mask_640.setTo(0, mask_640 < 0.5);
    mask_640.setTo(255, mask_640 >= 0.5);

    cv::Mat zeros(640, 640, CV_32FC1, cv::Scalar(0));
    auto rect = box.rect;
    zeros(rect).setTo(1);

    cv::Mat crop_mask;
    cv::multiply(mask_640, zeros, crop_mask);
    return crop_mask;
}

void yolo_seg_post(YoloCtx& ctx){
  yolo_det_post(ctx);
  auto results_ = &ctx.boxes;
  cv::Mat proto(32, 160*160, CV_32FC1, ctx.output_tensors[1]->getData());
  for(int index = 0; index < results_->size(); ++index) {
    (*results_)[index].mask = crop_seg_byproto((*results_)[index], proto);
  }

}

void set_pose(YoloBox& box, cv::Mat kpts_mat){
    for(int index = 0; index < kpts_mat.rows; ++index) {
      cv::Mat row = kpts_mat.row(index);
      auto x = row.at<float>(0, 0);
      auto y = row.at<float>(0, 1);
      auto score = row.at<float>(0, 2);
      box.kpts[index]={x,y,score};
    }
}
void yolo_pose_post(YoloCtx& ctx){
  yolo_det_post(ctx);
  auto results_ = &ctx.boxes;
  auto nkpt_ = ctx.nkpts;
  for(int index = 0; index < results_->size(); ++index) {
    cv::Mat mask = (*results_)[index].mask;
    cv::Mat kpts_mat(nkpt_, 3, CV_32FC1, mask.data);
    auto& box = (*results_)[index];
    box.kpts.resize(nkpt_);
    set_pose(box, kpts_mat);
  }
}
void yolo_segp_post(YoloCtx& ctx){
  yolo_det_post(ctx);
  cv::Mat proto(32, 160*160, CV_32FC1, ctx.output_tensors[1]->getData());
  auto results_ = &ctx.boxes;
  auto nkpt_ = ctx.nkpts;
  for(int index = 0; index < results_->size(); ++index) {
    auto& box = (*results_)[index];
    box.kpts.resize(nkpt_);
    float* mask_ptr = (float*)box.mask.data;
    cv::Mat kpts_mat=cv::Mat(nkpt_, 3, CV_32FC1, mask_ptr+32).clone();
    box.mask = cv::Mat(1, 32, CV_32FC1, mask_ptr).clone();
    box.mask = crop_seg_byproto(box,proto);
    set_pose(box, kpts_mat.clone());
  }

}
}//namespace pipeline::segp_post