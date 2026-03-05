#pragma once

#include <deeplearningcore/nn/module.hpp>
#include <string>
#include <chrono>
#include <iostream>

namespace deeplearningcore {

class Serialization {
public:
    static void save_weights(Module& module, const std::string& filepath);
    static void load_weights(Module& module, const std::string& filepath);
};

class Timer {
public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}

    void reset() {
        start_ = std::chrono::high_resolution_clock::now();
    }

    [[nodiscard]] double elapsed_ms() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

} // namespace deeplearningcore
