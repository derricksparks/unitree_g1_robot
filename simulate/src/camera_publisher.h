// ~/unitree_rl_mjlab/simulate/src/camera_publisher.h
#pragma once

#include <mujoco/mujoco.h>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/idl/go2/AudioData_.hpp>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <iostream>
#include <condition_variable>

class CameraPublisher {
public:
    CameraPublisher(mjModel* model, mjData* data, int width = 640, int height = 480)
        : mj_model_(model), mj_data_(data), width_(width), height_(height) {

        std::cout << "[CameraPublisher] Initializing..." << std::endl;

        // Allocate buffers
        rgb_buffer_ = new unsigned char[width * height * 3];
        depth_buffer_ = new float[width * height];

        // Find camera
        camera_id_ = mj_name2id(model, mjOBJ_CAMERA, "d435i_rgb");
        if (camera_id_ < 0) {
            std::cout << "[CameraPublisher] Warning: 'd435i_rgb' not found, using free camera" << std::endl;
            camera_id_ = -1;
        } else {
            std::cout << "[CameraPublisher] Using camera: d435i_rgb (id=" << camera_id_ << ")" << std::endl;
        }

        // Initialize DDS publisher
        rgb_pub_ = std::make_shared<unitree::robot::ChannelPublisher<unitree_go::msg::dds_::AudioData_>>("rt/camera/rgb");

        std::cout << "[CameraPublisher] Initialized " << width << "x" << height << std::endl;
    }

    ~CameraPublisher() {
        stop();
        delete[] rgb_buffer_;
        delete[] depth_buffer_;
    }

    void start(int fps = 30) {
        running_ = true;
        fps_ = fps;

        thread_ = std::make_unique<std::thread>([this]() {
            this->renderLoop();
        });

        std::cout << "[CameraPublisher] Started at " << fps << " FPS" << std::endl;
    }

    void stop() {
        running_ = false;
        if (thread_ && thread_->joinable()) {
            thread_->join();
        }
        std::cout << "[CameraPublisher] Stopped" << std::endl;
    }

private:
    mjModel* mj_model_;
    mjData* mj_data_;
    int width_;
    int height_;
    int camera_id_ = -1;
    int fps_ = 30;

    unsigned char* rgb_buffer_ = nullptr;
    float* depth_buffer_ = nullptr;

    std::shared_ptr<unitree::robot::ChannelPublisher<unitree_go::msg::dds_::AudioData_>> rgb_pub_;
    std::unique_ptr<std::thread> thread_;
    std::atomic<bool> running_{false};

    void renderLoop() {
        // Wait for main window to initialize
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        // Setup MuJoCo rendering
        mjrContext con;
        mjvScene scn;
        mjvOption opt;
        mjvCamera cam;

        mjv_defaultOption(&opt);
        mjv_defaultScene(&scn);
        mjv_defaultCamera(&cam);

        if (camera_id_ >= 0) {
            cam.type = mjCAMERA_FIXED;
            cam.fixedcamid = camera_id_;
        } else {
            cam.type = mjCAMERA_FREE;
            mjv_defaultFreeCamera(mj_model_, &cam);
        }

        // Create offscreen context
        mjr_makeContext(mj_model_, &con, 200);
        mjv_makeScene(mj_model_, &scn, 1000);
        mjr_setBuffer(mjFB_OFFSCREEN, &con);

        std::cout << "[CameraPublisher] Render loop started" << std::endl;

        int frame_count = 0;
        auto frame_time = std::chrono::milliseconds(1000 / fps_);

        while (running_) {
            auto start = std::chrono::steady_clock::now();

            // Render frame
            mjv_updateScene(mj_model_, mj_data_, &opt, NULL, &cam, mjCAT_ALL, &scn);

            mjrRect viewport = {0, 0, width_, height_};
            mjr_render(viewport, &scn, &con);
            mjr_readPixels(rgb_buffer_, depth_buffer_, viewport, &con);

            // Publish RGB
            unitree_go::msg::dds_::AudioData_ msg;
            msg.data().assign(rgb_buffer_, rgb_buffer_ + width_ * height_ * 3);
            rgb_pub_->Write(msg);

            frame_count++;
            if (frame_count % 100 == 0) {
                std::cout << "[CameraPublisher] Published " << frame_count << " frames" << std::endl;
            }

            // Maintain FPS
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start);
            if (elapsed < frame_time) {
                std::this_thread::sleep_for(frame_time - elapsed);
            }
        }

        mjr_freeContext(&con);
        mjv_freeScene(&scn);
        std::cout << "[CameraPublisher] Render loop stopped" << std::endl;
    }
};