#pragma once

#include "piperun.h"
namespace pipeline::segp_post{

void yolo_det_post(YoloCtx& ctx);
void yolo_seg_post(YoloCtx& ctx);
void yolo_pose_post(YoloCtx& ctx);
void yolo_segp_post(YoloCtx& ctx);


using pipeline::yolo_utils::YoloBox;
cv::Mat crop_seg_byproto(YoloBox& box, cv::Mat proto);

void yolo_cut_det_post(YoloCtx& ctx);
void yolo_cut_seg_post(YoloCtx& ctx);
void yolo_cut_pose_post(YoloCtx& ctx);
void yolo_cut_segp_post(YoloCtx& ctx);

void yolo_quan8_segp_post(YoloCtx& ctx);


}// namespace pipeline::segp_post