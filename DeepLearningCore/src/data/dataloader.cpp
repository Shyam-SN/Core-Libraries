#include <deeplearningcore/data/dataloader.hpp>
#include <numeric>
#include <algorithm>
#include <random>

namespace deeplearningcore {

TensorDataset::TensorDataset(Tensor inputs, Tensor targets)
    : inputs_(std::move(inputs)), targets_(std::move(targets)) {
    if (inputs_.shape()[0] != targets_.shape()[0]) {
        throw std::invalid_argument("Inputs and targets must have same batch size dimension 0");
    }
}

size_t TensorDataset::size() const {
    return inputs_.shape()[0];
}

std::pair<Tensor, Tensor> TensorDataset::get_item(size_t index) const {
    size_t in_features = inputs_.numel() / inputs_.shape()[0];
    size_t out_features = targets_.numel() / targets_.shape()[0];

    std::vector<size_t> in_dims = inputs_.shape().dims();
    in_dims[0] = 1;

    std::vector<size_t> target_dims = targets_.shape().dims();
    target_dims[0] = 1;

    Tensor item_in(Shape(in_dims), false);
    for (size_t i = 0; i < in_features; ++i) {
        item_in[i] = inputs_[index * in_features + i];
    }

    Tensor item_target(Shape(target_dims), false);
    for (size_t i = 0; i < out_features; ++i) {
        item_target[i] = targets_[index * out_features + i];
    }

    return {item_in, item_target};
}

DataLoader::DataLoader(const Dataset& dataset, size_t batch_size, bool shuffle)
    : dataset_(dataset), batch_size_(batch_size), shuffle_(shuffle) {
    indices_.resize(dataset_.size());
    std::iota(indices_.begin(), indices_.end(), 0);
    if (shuffle_) {
        static std::mt19937 gen(42);
        std::shuffle(indices_.begin(), indices_.end(), gen);
    }
}

DataLoader::Iterator::Iterator(const DataLoader& loader, size_t index)
    : loader_(loader), index_(index) {}

std::pair<Tensor, Tensor> DataLoader::Iterator::operator*() const {
    size_t end_idx = std::min(index_ + loader_.batch_size_, loader_.indices_.size());
    size_t cur_batch = end_idx - index_;

    auto first_pair = loader_.dataset_.get_item(loader_.indices_[index_]);
    size_t in_feat = first_pair.first.numel();
    size_t out_feat = first_pair.second.numel();

    std::vector<size_t> in_dims = first_pair.first.shape().dims();
    in_dims[0] = cur_batch;

    std::vector<size_t> target_dims = first_pair.second.shape().dims();
    target_dims[0] = cur_batch;

    Tensor batch_in(Shape(in_dims), false);
    Tensor batch_target(Shape(target_dims), false);

    for (size_t b = 0; b < cur_batch; ++b) {
        auto pair = loader_.dataset_.get_item(loader_.indices_[index_ + b]);
        for (size_t i = 0; i < in_feat; ++i) {
            batch_in[b * in_feat + i] = pair.first[i];
        }
        for (size_t i = 0; i < out_feat; ++i) {
            batch_target[b * out_feat + i] = pair.second[i];
        }
    }

    return {batch_in, batch_target};
}

DataLoader::Iterator& DataLoader::Iterator::operator++() {
    index_ += loader_.batch_size_;
    return *this;
}

bool DataLoader::Iterator::operator!=(const Iterator& other) const {
    return index_ < other.index_;
}

} // namespace deeplearningcore
