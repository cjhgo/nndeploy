#pragma once

#include "piperun.h"

namespace pipeline::segp_post{

void yolo_det_post(YoloCtx& ctx);
void yolo_seg_post(YoloCtx& ctx);
void yolo_pose_post(YoloCtx& ctx);
void yolo_segp_post(YoloCtx& ctx);


using pipeline::yolo_utils::YoloBox;
cv::Mat crop_seg_byproto(YoloBox& box, cv::Mat proto);
void set_cut_pose(YoloBox &box, cv::Mat kpts_mat);

void yolo_cut_det_post(YoloCtx& ctx);
void yolo_cut_seg_post(YoloCtx& ctx);
void yolo_cut_pose_post(YoloCtx& ctx);
void yolo_cut_segp_post(YoloCtx& ctx);

void yolo_quan8_segp_post(YoloCtx& ctx);
void rk_custom(nndeploy::base::Param* param);
void quanrk_custom(nndeploy::base::Param* param);

#if !(__linux__ && __aarch64__)
inline void yolo_quan8_segp_post(YoloCtx& ctx) {}
inline void rk_custom(nndeploy::base::Param* param) {}
inline void quanrk_custom(nndeploy::base::Param* param) {}
#endif


}// namespace pipeline::segp_post