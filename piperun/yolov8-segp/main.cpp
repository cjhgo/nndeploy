#include "piperun.h"
#include "segp_post.h"
#include "yolo_ori.cc"
#include "yolo_cut.cc"
#include <opencv2/opencv.hpp>
#include <map>
#include <functional>

using namespace pipeline;

// 修改点1：在配置结构体中添加 nkpts 字段
struct TaskConfig {
    std::string infer_name;
    std::function<void(YoloCtx&)> post_func;
    int nc;
    int nkpts;
};

void generic_run(YoloCtx& ctx, const TaskConfig& config) {
    tf::Executor executor;
    tf::Taskflow taskflow("segp");
    ctx.infer = pipeline::create_inference(config.infer_name);
    ctx.nc = config.nc;
    ctx.nkpts = config.nkpts;  // 新增关键点配置注入

    create_nn_pipe<YoloCtx>(taskflow, ctx, 
                          yolo_utils::yolo_pre_process<YoloCtx>,
                          config.post_func);
    executor.run(taskflow).get();
    yolo_utils::draw_yolobox(ctx.input, ctx.boxes);
}

// 修改点3：在任务映射中添加关键点配置
std::map<std::string, TaskConfig> TASK_MAP = {
    {"yolo_det",   {"y11", segp_post::yolo_det_post,   80,  0}},  // 无关键点
    {"yolo_seg",   {"seg",  segp_post::yolo_seg_post,  80,  0}},
    {"yolo_pose",  {"pose", segp_post::yolo_pose_post,  1, 17}},  // COCO 17关键点
    {"yolo_segp",  {"segp", segp_post::yolo_segp_post,  1,  17}},

    {"cut_det",  {"segp_cut", segp_post::yolo_cut_det_post,  1,  0}},
    {"cut_seg",  {"segp_cut", segp_post::yolo_cut_seg_post,  1,  0}},
    {"cut_pose",  {"segp_cut", segp_post::yolo_cut_pose_post,  1,  17}},
    {"cut_segp",  {"segp_cut", segp_post::yolo_cut_segp_post,  1,  17}},
};

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <task_type>\n"
                  << "Available tasks: yolo_det | yolo_seg | yolo_pose | yolo_segp \n";
        return 1;
    }

    std::string task_type = argv[1];
    if (auto it = TASK_MAP.find(task_type); it != TASK_MAP.end()) {
        auto img = cv::imread("test.jpg", -1);
        if (img.empty()) {
            std::cerr << "Failed to load image\n";
            return 1;
        }

        YoloCtx ctx;
        ctx.input = img;
        generic_run(ctx, it->second);
    } 
    else {
        std::cerr << "Invalid task type: " << task_type << std::endl;
        return 1;
    }

    return 0;
}
