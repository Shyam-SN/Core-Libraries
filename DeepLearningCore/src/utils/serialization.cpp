#include <deeplearningcore/utils/serialization.hpp>
#include <fstream>
#include <stdexcept>

namespace deeplearningcore {

void Serialization::save_weights(Module& module, const std::string& filepath) {
    std::ofstream out(filepath, std::ios::binary);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open file for saving weights: " + filepath);
    }

    auto params = module.parameters();
    uint32_t num_params = static_cast<uint32_t>(params.size());
    out.write(reinterpret_cast<const char*>(&num_params), sizeof(num_params));

    for (auto& p : params) {
        uint32_t numel = static_cast<uint32_t>(p->numel());
        out.write(reinterpret_cast<const char*>(&numel), sizeof(numel));
        out.write(reinterpret_cast<const char*>(p->data()), numel * sizeof(float));
    }
}

void Serialization::load_weights(Module& module, const std::string& filepath) {
    std::ifstream in(filepath, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open file for loading weights: " + filepath);
    }

    auto params = module.parameters();
    uint32_t num_params = 0;
    in.read(reinterpret_cast<char*>(&num_params), sizeof(num_params));

    if (num_params != params.size()) {
        throw std::runtime_error("Parameter count mismatch in model checkpoint");
    }

    for (auto& p : params) {
        uint32_t numel = 0;
        in.read(reinterpret_cast<char*>(&numel), sizeof(numel));
        if (numel != p->numel()) {
            throw std::runtime_error("Parameter numel mismatch in checkpoint");
        }
        in.read(reinterpret_cast<char*>(p->data()), numel * sizeof(float));
    }
}

} // namespace deeplearningcore
