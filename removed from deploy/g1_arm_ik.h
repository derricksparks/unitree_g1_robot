#pragma once

#include <vector>
#include <cmath>
#include <algorithm>

struct Vec3 {
    float x, y, z;
    Vec3() : x(0), y(0), z(0) {}
    Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    Vec3 operator-(const Vec3& other) const {
        return Vec3(x - other.x, y - other.y, z - other.z);
    }
    float norm() const {
        return std::sqrt(x*x + y*y + z*z);
    }
};

class G1ArmIK {
public:
    const float L1 = 0.15f;
    const float L2 = 0.25f;
    
    const float LIMITS[7][2] = {
        {-3.0892f, 2.6704f}, {-1.5882f, 2.2515f}, {-2.618f, 2.618f},
        {-1.0472f, 2.0944f}, {-1.9722f, 1.9722f}, {-1.6144f, 1.6144f}, {-1.6144f, 1.6144f}
    };
    
    const float HOME[7] = {0.3f, 0.0f, 0.0f, -0.5f, -1.2f, 0.0f, 0.0f};
    
    static float clamp(float val, float min_val, float max_val) {
        return std::max(min_val, std::min(max_val, val));
    }
    
    std::vector<float> solveIK(
        const Vec3& target_world,
        const Vec3& shoulder_pos,
        const std::vector<float>& current_joints
    ) {
        std::vector<float> joints(7, 0.0f);
        
        if (current_joints.size() >= 7) {
            for (int i = 0; i < 7; i++) joints[i] = current_joints[i];
        } else {
            for (int i = 0; i < 7; i++) joints[i] = HOME[i];
        }
        
        Vec3 target(target_world.x - shoulder_pos.x,
                    target_world.y - shoulder_pos.y,
                    target_world.z - shoulder_pos.z);
        
        float distance = target.norm();
        float max_reach = L1 + L2 - 0.02f;
        float min_reach = std::abs(L1 - L2) + 0.02f;
        
        if (distance > max_reach) {
            float scale = max_reach / distance;
            target.x *= scale; target.y *= scale; target.z *= scale;
            distance = max_reach;
        } else if (distance < min_reach) {
            float scale = min_reach / distance;
            target.x *= scale; target.y *= scale; target.z *= scale;
            distance = min_reach;
        }
        
        float horizontal_dist = std::sqrt(target.x*target.x + target.y*target.y);
        joints[0] = std::atan2(-target.z, horizontal_dist);
        joints[2] = std::atan2(target.y, target.x);
        
        float cos_elbow = (distance*distance - L1*L1 - L2*L2) / (2.0f * L1 * L2);
        cos_elbow = clamp(cos_elbow, -1.0f, 1.0f);
        joints[3] = std::acos(cos_elbow);
        
        joints[0] = clamp(joints[0], LIMITS[0][0], LIMITS[0][1]);
        joints[1] = clamp(joints[1], LIMITS[1][0], LIMITS[1][1]);
        joints[2] = clamp(joints[2], LIMITS[2][0], LIMITS[2][1]);
        joints[3] = clamp(joints[3], LIMITS[3][0], LIMITS[3][1]);
        
        for (int i = 4; i < 7; i++) {
            joints[i] = clamp(joints[i], LIMITS[i][0], LIMITS[i][1]);
        }
        
        return joints;
    }
    
    Vec3 forwardKinematics(const std::vector<float>& joints) {
        float s1 = std::sin(joints[0]), c1 = std::cos(joints[0]);
        float s2 = std::sin(joints[2]), c2 = std::cos(joints[2]);
        float s3 = std::sin(joints[3]), c3 = std::cos(joints[3]);
        
        float x = L1 * c1 * c2 + L2 * (c1 * c2 * c3 - c1 * s2 * s3);
        float y = L1 * c1 * s2 + L2 * (c1 * s2 * c3 + c1 * c2 * s3);
        float z = -L1 * s1 - L2 * s1 * c3;
        
        return Vec3(x, y, z);
    }
    
    std::vector<float> getHomeJoints() {
        return std::vector<float>(HOME, HOME + 7);
    }
};
