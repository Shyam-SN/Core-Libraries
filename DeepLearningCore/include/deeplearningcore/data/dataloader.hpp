#pragma once

#include <deeplearningcore/core/tensor.hpp>
#include <vector>
#include <utility>
#include <cstddef>

namespace deeplearningcore {

class Dataset {
public:
    virtual ~Dataset() = default;
    [[nodiscard]] virtual size_t size() const = 0;
    [[nodiscard]] virtual std::pair<Tensor, Tensor> get_item(size_t index) const = 0;
};

class TensorDataset : public Dataset {
public:
    TensorDataset(Tensor inputs, Tensor targets);

    [[nodiscard]] size_t size() const override;
    [[nodiscard]] std::pair<Tensor, Tensor> get_item(size_t index) const override;

private:
    Tensor inputs_;
    Tensor targets_;
};

class DataLoader {
public:
    DataLoader(const Dataset& dataset, size_t batch_size, bool shuffle = true);

    class Iterator {
    public:
        Iterator(const DataLoader& loader, size_t index);
        std::pair<Tensor, Tensor> operator*() const;
        Iterator& operator++();
        bool operator!=(const Iterator& other) const;

    private:
        const DataLoader& loader_;
        size_t index_;
    };

    Iterator begin() const { return Iterator(*this, 0); }
    Iterator end() const { return Iterator(*this, indices_.size()); }

    [[nodiscard]] size_t size() const { return indices_.size() / batch_size_ + (indices_.size() % batch_size_ != 0 ? 1 : 0); }

private:
    const Dataset& dataset_;
    size_t batch_size_;
    bool shuffle_;
    std::vector<size_t> indices_;
};

} // namespace deeplearningcore
