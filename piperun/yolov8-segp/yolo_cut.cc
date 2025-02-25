#include "segp_post.h"
#include <iostream>
namespace pipeline::segp_post{

using namespace pipeline::yolo_utils;

void yolo_cut_det_post(YoloCtx& ctx){
  cut_non_max_suppression(&ctx.boxes, ctx.output_tensors[0].get(), ctx.nc);
}

void yolo_cut_seg_post(YoloCtx& ctx){
  yolo_cut_det_post(ctx);

  auto& seg_tensor = ctx.output_tensors[3];//?!!!
  cv::Mat seg_mask(32, 8400, CV_32FC1, seg_tensor->getData());
  seg_mask = seg_mask.t();
  auto& proto_tensor = ctx.output_tensors[2];
  cv::Mat proto(32, 160 * 160, CV_32FC1, proto_tensor->getData());
  auto results_ = &ctx.boxes;

  for (int index = 0; index < results_->size(); ++index) {
    auto g_index = (*results_)[index].index;
    std::cout << "g_index:" << g_index << std::endl;
    (*results_)[index].mask =
        cv::Mat(1, 32, CV_32FC1, seg_mask.ptr<float>(g_index)).clone();
    (*results_)[index].mask = crop_seg_byproto((*results_)[index], proto);
  }
}


void set_cut_pose(YoloBox &box, cv::Mat kpts_mat) {
  auto cur_h = box.cur_h;
  auto cur_w = box.cur_w;
  auto stride = box.stride;
  for (int index = 0; index < kpts_mat.rows; ++index) {
    cv::Mat row = kpts_mat.row(index);
    auto x = row.at<float>(0, 0);
    auto y = row.at<float>(0, 1);
    auto score = row.at<float>(0, 2);
    x = (x * 2 + cur_w) * stride;
    y = (y * 2 + cur_h) * stride;
    box.kpts[index] = {x, y, score};
  }
}

void yolo_cut_pose_post(YoloCtx& ctx){
  yolo_cut_det_post(ctx);
  auto nkpt_ = ctx.nkpts;
  auto& pose_tensor = ctx.output_tensors[1];
  cv::Mat pose_data(nkpt_ * 3, 8400, CV_32FC1, pose_tensor->getData());
  pose_data = pose_data.t();

  auto results_ = &ctx.boxes;
  for (int index = 0; index < results_->size(); ++index) {
    auto &box = (*results_)[index];
    auto g_index = box.index;
    cv::Mat kpts_mat =
        cv::Mat(nkpt_, 3, CV_32FC1, pose_data.ptr<float>(g_index)).clone();
    box.kpts.resize(nkpt_);
    set_cut_pose(box, kpts_mat);
  }
}

void yolo_cut_segp_post(YoloCtx& ctx){
  yolo_cut_det_post(ctx);

  auto& seg_tensor = ctx.output_tensors[3];//?!!!
  cv::Mat seg_mask(32, 8400, CV_32FC1, seg_tensor->getData());
  seg_mask = seg_mask.t();

  auto& proto_tensor = ctx.output_tensors[2];
  cv::Mat proto(32, 160 * 160, CV_32FC1, proto_tensor->getData());

  auto nkpt_ = ctx.nkpts;
  auto& pose_tensor = ctx.output_tensors[1];
  cv::Mat pose_data(nkpt_ * 3, 8400, CV_32FC1, pose_tensor->getData());
  pose_data = pose_data.t();

  auto results_ = &ctx.boxes;

  for (int index = 0; index < results_->size(); ++index) {
    auto &box = (*results_)[index];
    auto g_index = box.index;
    box.kpts.resize(nkpt_);

    box.mask = cv::Mat(1, 32, CV_32FC1, seg_mask.ptr<float>(g_index)).clone();
    box.mask = crop_seg_byproto(box, proto);

    cv::Mat kpts_mat =
        cv::Mat(nkpt_, 3, CV_32FC1, pose_data.ptr<float>(g_index)).clone();
    set_cut_pose(box, kpts_mat);
  }
}

}//namespace pipeline::segp_post