// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "onnxruntime_cxx_api.h"
#include <iostream>
#include <mutex>

namespace isaaclab
{

class Algorithms
{
public:
    virtual std::vector<float> act(std::unordered_map<std::string, std::vector<float>> obs) = 0;

    std::vector<float> get_action()
    {
        std::lock_guard<std::mutex> lock(act_mtx_);
        return action;
    }
    
    std::vector<float> action;
protected:
    std::mutex act_mtx_;
};

class OrtRunner : public Algorithms
{
public:
    OrtRunner(std::string model_path)
    : model_path_(std::move(model_path))
    {
        // Init Model
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "onnx_model");
        session_options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);

        session = std::make_unique<Ort::Session>(env, model_path_.c_str(), session_options);

        for (size_t i = 0; i < session->GetInputCount(); ++i) {
            Ort::TypeInfo input_type = session->GetInputTypeInfo(i);
            input_shapes.push_back(input_type.GetTensorTypeAndShapeInfo().GetShape());
            auto input_name = session->GetInputNameAllocated(i, allocator);
            input_names.push_back(input_name.release());
        }

        // Get output name (shape can be dynamic, so we size action lazily on first act()).
        auto output_name = session->GetOutputNameAllocated(0, allocator);
        output_names.push_back(output_name.release());
    }

    std::vector<float> act(std::unordered_map<std::string, std::vector<float>> obs)
    {
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

        // make sure all input names are in obs (with small compatibility aliases)
        //
        // Historically, some exported policies use input name "observation" while this
        // deploy stack produces the single concatenated observation under key "obs".
        // If the model expects a single input and obs has a single entry, we also allow
        // that entry to be used.
        for (const auto& name_c : input_names) {
            const std::string name(name_c);
            if (obs.find(name) != obs.end()) {
                continue;
            }

            // Common alias: observation <-> obs
            if (name == "observation" && obs.find("obs") != obs.end()) {
                obs[name] = obs["obs"];
                continue;
            }
            if (name == "obs" && obs.find("observation") != obs.end()) {
                obs[name] = obs["observation"];
                continue;
            }

            // Fallback: if there is exactly one observation vector, use it.
            if (obs.size() == 1) {
                obs[name] = obs.begin()->second;
                continue;
            }

            std::string available;
            for (const auto& kv : obs) {
                if (!available.empty()) available += ", ";
                available += kv.first;
            }
            throw std::runtime_error(
                "Input name " + name + " not found in observations. Available keys: [" + available + "]"
            );
        }

        // Create input tensors (resolve dynamic shapes like -1 using actual input length).
        // Some exported policies expect inputs shaped [-1, K] (e.g. K=24). If the provided
        // observation length is off by a few elements due to packing differences, we pad/truncate
        // to the nearest multiple of K to avoid hard-crashing during deployment.
        std::vector<Ort::Value> input_tensors;
        std::vector<std::vector<float>> owned_inputs;  // keeps padded buffers alive for Run()
        for(int i(0); i<input_names.size(); ++i)
        {
            const std::string name_str(input_names[i]);
            auto& input_data_ref = obs.at(name_str);

            // Resolve tensor shape: ONNX may use -1 for dynamic dims (e.g. [1, -1]).
            std::vector<int64_t> resolved_shape = input_shapes[i];
            int64_t known_prod = 1;
            int dyn_count = 0;
            int dyn_index = -1;
            for (int j = 0; j < static_cast<int>(resolved_shape.size()); ++j) {
                if (resolved_shape[j] < 0) {
                    dyn_count += 1;
                    dyn_index = j;
                } else if (resolved_shape[j] == 0) {
                    // Treat 0 as dynamic as well (rare, but used in some exports).
                    dyn_count += 1;
                    dyn_index = j;
                    resolved_shape[j] = -1;
                } else {
                    known_prod *= resolved_shape[j];
                }
            }
            if (dyn_count > 1) {
                throw std::runtime_error("ONNX input has multiple dynamic dims; unsupported in this deploy runner.");
            }

            // Default: use the original buffer.
            const float* input_ptr = input_data_ref.data();
            size_t input_size = input_data_ref.size();

            if (dyn_count == 1) {
                if (known_prod <= 0) {
                    throw std::runtime_error("Invalid ONNX input shape product while resolving dynamic dim.");
                }
                int64_t total = static_cast<int64_t>(input_size);
                if (total % known_prod != 0) {
                    // Heuristic padding/truncation: align to nearest multiple of known_prod.
                    const int64_t up = ((total + known_prod - 1) / known_prod) * known_prod;
                    const int64_t down = (total / known_prod) * known_prod;
                    const int64_t target = (down > 0 && (total - down) <= (up - total)) ? down : up;

                    owned_inputs.emplace_back(input_data_ref.begin(), input_data_ref.end());
                    auto& buf = owned_inputs.back();
                    if (static_cast<int64_t>(buf.size()) < target) {
                        buf.resize(static_cast<size_t>(target), 0.0f);
                    } else if (static_cast<int64_t>(buf.size()) > target) {
                        buf.resize(static_cast<size_t>(target));
                    }
                    input_ptr = buf.data();
                    input_size = buf.size();
                    total = static_cast<int64_t>(input_size);
                }

                if (total % known_prod != 0) {
                    std::string shape_str;
                    for (size_t k = 0; k < input_shapes[i].size(); ++k) {
                        shape_str += std::to_string(input_shapes[i][k]);
                        if (k + 1 < input_shapes[i].size()) shape_str += ",";
                    }
                    throw std::runtime_error(
                        "Cannot resolve dynamic input shape: obs len not divisible by known dims."
                        " model=" + model_path_ +
                        " input=" + name_str +
                        " onnx_shape=[" + shape_str + "]" +
                        " obs_len=" + std::to_string(total) +
                        " known_prod=" + std::to_string(known_prod)
                    );
                }
                resolved_shape[dyn_index] = total / known_prod;
            }

            auto input_tensor = Ort::Value::CreateTensor<float>(
                memory_info,
                const_cast<float*>(input_ptr),
                input_size,
                resolved_shape.data(),
                resolved_shape.size()
            );
            input_tensors.push_back(std::move(input_tensor));
        }

        // Run the model
        auto output_tensor = session->Run(Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(), input_tensors.size(), output_names.data(), 1);

        // Copy output data
        auto & out0 = output_tensor.front();
        auto floatarr = out0.GetTensorMutableData<float>();
        const auto out_info = out0.GetTensorTypeAndShapeInfo();
        const size_t out_elems = out_info.GetElementCount();
        std::lock_guard<std::mutex> lock(act_mtx_);
        if (action.size() != out_elems) {
            action.resize(out_elems);
        }
        std::memcpy(action.data(), floatarr, out_elems * sizeof(float));
        return action;
    }

private:
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;

    std::string model_path_;
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;

    std::vector<std::vector<int64_t>> input_shapes;
};
};