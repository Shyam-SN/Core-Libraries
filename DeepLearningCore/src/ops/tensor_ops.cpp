#include <deeplearningcore/ops/tensor_ops.hpp>
#include <deeplearningcore/core/autograd.hpp>
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace deeplearningcore {
namespace ops {

Tensor add(const Tensor& a, const Tensor& b) {
    Shape res_shape = Shape::broadcast(a.shape(), b.shape());
    bool req_grad = (a.requires_grad() || b.requires_grad()) && NoGradScope::is_grad_enabled();

    Tensor result(res_shape, req_grad);
    size_t n = result.numel();

    if (a.shape() == b.shape()) {
        for (size_t i = 0; i < n; ++i) {
            result[i] = a[i] + b[i];
        }
    } else {
        // Broadcasted addition
        size_t a_numel = a.numel();
        size_t b_numel = b.numel();
        for (size_t i = 0; i < n; ++i) {
            result[i] = a[i % a_numel] + b[i % b_numel];
        }
    }

    if (req_grad) {
        auto node = std::make_shared<AutogradNode>();
        if (a.requires_grad()) node->add_parent(a.autograd_node());
        if (b.requires_grad()) node->add_parent(b.autograd_node());

        auto a_ptr = std::make_shared<Tensor>(a);
        auto b_ptr = std::make_shared<Tensor>(b);

        node->set_backward_fn([a_ptr, b_ptr](const Tensor& grad) {
            if (a_ptr->requires_grad()) {
                if (a_ptr->shape() == grad.shape()) {
                    a_ptr->backward(grad);
                } else {
                    a_ptr->backward(grad.sum());
                }
            }
            if (b_ptr->requires_grad()) {
                if (b_ptr->shape() == grad.shape()) {
                    b_ptr->backward(grad);
                } else {
                    b_ptr->backward(grad.sum());
                }
            }
        });
        result.set_autograd_node(node);
    }

    return result;
}

Tensor sub(const Tensor& a, const Tensor& b) {
    Shape res_shape = Shape::broadcast(a.shape(), b.shape());
    bool req_grad = (a.requires_grad() || b.requires_grad()) && NoGradScope::is_grad_enabled();

    Tensor result(res_shape, req_grad);
    size_t n = result.numel();

    if (a.shape() == b.shape()) {
        for (size_t i = 0; i < n; ++i) {
            result[i] = a[i] - b[i];
        }
    } else {
        size_t a_numel = a.numel();
        size_t b_numel = b.numel();
        for (size_t i = 0; i < n; ++i) {
            result[i] = a[i % a_numel] - b[i % b_numel];
        }
    }

    if (req_grad) {
        auto node = std::make_shared<AutogradNode>();
        if (a.requires_grad()) node->add_parent(a.autograd_node());
        if (b.requires_grad()) node->add_parent(b.autograd_node());

        auto a_ptr = std::make_shared<Tensor>(a);
        auto b_ptr = std::make_shared<Tensor>(b);

        node->set_backward_fn([a_ptr, b_ptr](const Tensor& grad) {
            if (a_ptr->requires_grad()) {
                a_ptr->backward(a_ptr->shape() == grad.shape() ? grad : grad.sum());
            }
            if (b_ptr->requires_grad()) {
                Tensor neg_grad = grad.mul(-1.0f);
                b_ptr->backward(b_ptr->shape() == grad.shape() ? neg_grad : neg_grad.sum());
            }
        });
        result.set_autograd_node(node);
    }

    return result;
}

Tensor mul(const Tensor& a, const Tensor& b) {
    Shape res_shape = Shape::broadcast(a.shape(), b.shape());
    bool req_grad = (a.requires_grad() || b.requires_grad()) && NoGradScope::is_grad_enabled();

    Tensor result(res_shape, req_grad);
    size_t n = result.numel();

    if (a.shape() == b.shape()) {
        for (size_t i = 0; i < n; ++i) {
            result[i] = a[i] * b[i];
        }
    } else {
        size_t a_numel = a.numel();
        size_t b_numel = b.numel();
        for (size_t i = 0; i < n; ++i) {
            result[i] = a[i % a_numel] * b[i % b_numel];
        }
    }

    if (req_grad) {
        auto node = std::make_shared<AutogradNode>();
        if (a.requires_grad()) node->add_parent(a.autograd_node());
        if (b.requires_grad()) node->add_parent(b.autograd_node());

        auto a_ptr = std::make_shared<Tensor>(a);
        auto b_ptr = std::make_shared<Tensor>(b);

        node->set_backward_fn([a_ptr, b_ptr](const Tensor& grad) {
            if (a_ptr->requires_grad()) {
                Tensor da = grad.mul(*b_ptr);
                a_ptr->backward(a_ptr->shape() == grad.shape() ? da : da.sum());
            }
            if (b_ptr->requires_grad()) {
                Tensor db = grad.mul(*a_ptr);
                b_ptr->backward(b_ptr->shape() == grad.shape() ? db : db.sum());
            }
        });
        result.set_autograd_node(node);
    }

    return result;
}

Tensor div(const Tensor& a, const Tensor& b) {
    Shape res_shape = Shape::broadcast(a.shape(), b.shape());
    bool req_grad = (a.requires_grad() || b.requires_grad()) && NoGradScope::is_grad_enabled();

    Tensor result(res_shape, req_grad);
    size_t n = result.numel();

    if (a.shape() == b.shape()) {
        for (size_t i = 0; i < n; ++i) {
            result[i] = a[i] / b[i];
        }
    } else {
        size_t a_numel = a.numel();
        size_t b_numel = b.numel();
        for (size_t i = 0; i < n; ++i) {
            result[i] = a[i % a_numel] / b[i % b_numel];
        }
    }

    if (req_grad) {
        auto node = std::make_shared<AutogradNode>();
        if (a.requires_grad()) node->add_parent(a.autograd_node());
        if (b.requires_grad()) node->add_parent(b.autograd_node());

        auto a_ptr = std::make_shared<Tensor>(a);
        auto b_ptr = std::make_shared<Tensor>(b);

        node->set_backward_fn([a_ptr, b_ptr](const Tensor& grad) {
            if (a_ptr->requires_grad()) {
                Tensor da = grad.div(*b_ptr);
                a_ptr->backward(a_ptr->shape() == grad.shape() ? da : da.sum());
            }
            if (b_ptr->requires_grad()) {
                // d/db (a/b) = -a / b^2
                Tensor db = grad.mul(*a_ptr).mul(-1.0f).div(b_ptr->mul(*b_ptr));
                b_ptr->backward(b_ptr->shape() == grad.shape() ? db : db.sum());
            }
        });
        result.set_autograd_node(node);
    }

    return result;
}

Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.ndim() < 2 || b.ndim() < 2) {
        throw std::invalid_argument("matmul requires tensors of at least 2 dimensions");
    }

    size_t M = a.shape()[a.ndim() - 2];
    size_t K = a.shape()[a.ndim() - 1];
    size_t K_b = b.shape()[b.ndim() - 2];
    size_t N = b.shape()[b.ndim() - 1];

    if (K != K_b) {
        throw std::invalid_argument("Inner dimensions mismatch for matmul: " + a.shape().to_string() + " vs " + b.shape().to_string());
    }

    size_t batch_size = a.numel() / (M * K);
    Shape res_shape({M, N});
    if (a.ndim() > 2) {
        std::vector<size_t> res_dims;
        for (size_t i = 0; i < a.ndim() - 2; ++i) res_dims.push_back(a.shape()[i]);
        res_dims.push_back(M);
        res_dims.push_back(N);
        res_shape = Shape(res_dims);
    }

    bool req_grad = (a.requires_grad() || b.requires_grad()) && NoGradScope::is_grad_enabled();
    Tensor result(res_shape, req_grad);

    for (size_t b_idx = 0; b_idx < batch_size; ++b_idx) {
        size_t a_offset = b_idx * M * K;
        size_t b_offset = b_idx * K * N;
        size_t c_offset = b_idx * M * N;

        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                float sum_val = 0.0f;
                for (size_t k = 0; k < K; ++k) {
                    sum_val += a[a_offset + i * K + k] * b[b_offset + k * N + j];
                }
                result[c_offset + i * N + j] = sum_val;
            }
        }
    }

    if (req_grad) {
        auto node = std::make_shared<AutogradNode>();
        if (a.requires_grad()) node->add_parent(a.autograd_node());
        if (b.requires_grad()) node->add_parent(b.autograd_node());

        auto a_ptr = std::make_shared<Tensor>(a);
        auto b_ptr = std::make_shared<Tensor>(b);

        node->set_backward_fn([a_ptr, b_ptr](const Tensor& grad) {
            if (a_ptr->requires_grad()) {
                // dL/dA = grad * B^T
                Tensor da = grad.matmul(b_ptr->transpose());
                a_ptr->backward(da);
            }
            if (b_ptr->requires_grad()) {
                // dL/dB = A^T * grad
                Tensor db = a_ptr->transpose().matmul(grad);
                b_ptr->backward(db);
            }
        });
        result.set_autograd_node(node);
    }

    return result;
}

Tensor transpose(const Tensor& a) {
    if (a.ndim() < 2) {
        return a;
    }
    size_t M = a.shape()[a.ndim() - 2];
    size_t N = a.shape()[a.ndim() - 1];

    std::vector<size_t> new_dims = a.shape().dims();
    std::swap(new_dims[new_dims.size() - 2], new_dims[new_dims.size() - 1]);

    bool req_grad = a.requires_grad() && NoGradScope::is_grad_enabled();
    Tensor result(Shape(new_dims), req_grad);

    size_t batch_size = a.numel() / (M * N);
    for (size_t b_idx = 0; b_idx < batch_size; ++b_idx) {
        size_t src_off = b_idx * M * N;
        size_t dst_off = b_idx * N * M;
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                result[dst_off + j * M + i] = a[src_off + i * N + j];
            }
        }
    }

    if (req_grad) {
        auto node = std::make_shared<AutogradNode>();
        node->add_parent(a.autograd_node());
        auto a_ptr = std::make_shared<Tensor>(a);

        node->set_backward_fn([a_ptr](const Tensor& grad) {
            a_ptr->backward(grad.transpose());
        });
        result.set_autograd_node(node);
    }

    return result;
}

Tensor reshape(const Tensor& a, const Shape& new_shape) {
    if (a.numel() != new_shape.numel()) {
        throw std::invalid_argument("Cannot reshape tensor of size " + std::to_string(a.numel()) + " to " + new_shape.to_string());
    }

    bool req_grad = a.requires_grad() && NoGradScope::is_grad_enabled();
    Tensor result(new_shape, req_grad);
    for (size_t i = 0; i < a.numel(); ++i) {
        result[i] = a[i];
    }

    if (req_grad) {
        auto node = std::make_shared<AutogradNode>();
        node->add_parent(a.autograd_node());
        auto a_ptr = std::make_shared<Tensor>(a);

        node->set_backward_fn([a_ptr](const Tensor& grad) {
            a_ptr->backward(grad.reshape(a_ptr->shape()));
        });
        result.set_autograd_node(node);
    }

    return result;
}

Tensor relu(const Tensor& a) {
    bool req_grad = a.requires_grad() && NoGradScope::is_grad_enabled();
    Tensor result(a.shape(), req_grad);

    for (size_t i = 0; i < a.numel(); ++i) {
        result[i] = std::max(0.0f, a[i]);
    }

    if (req_grad) {
        auto node = std::make_shared<AutogradNode>();
        node->add_parent(a.autograd_node());
        auto a_ptr = std::make_shared<Tensor>(a);

        node->set_backward_fn([a_ptr](const Tensor& grad) {
            Tensor da(a_ptr->shape(), false);
            for (size_t i = 0; i < a_ptr->numel(); ++i) {
                da[i] = ((*a_ptr)[i] > 0.0f) ? grad[i] : 0.0f;
            }
            a_ptr->backward(da);
        });
        result.set_autograd_node(node);
    }

    return result;
}

Tensor sigmoid(const Tensor& a) {
    bool req_grad = a.requires_grad() && NoGradScope::is_grad_enabled();
    Tensor result(a.shape(), req_grad);

    for (size_t i = 0; i < a.numel(); ++i) {
        result[i] = 1.0f / (1.0f + std::exp(-a[i]));
    }

    if (req_grad) {
        auto node = std::make_shared<AutogradNode>();
        node->add_parent(a.autograd_node());
        auto res_ptr = std::make_shared<Tensor>(result);
        auto a_ptr = std::make_shared<Tensor>(a);

        node->set_backward_fn([a_ptr, res_ptr](const Tensor& grad) {
            Tensor da(a_ptr->shape(), false);
            for (size_t i = 0; i < a_ptr->numel(); ++i) {
                float s = (*res_ptr)[i];
                da[i] = grad[i] * s * (1.0f - s);
            }
            a_ptr->backward(da);
        });
        result.set_autograd_node(node);
    }

    return result;
}

Tensor tanh(const Tensor& a) {
    bool req_grad = a.requires_grad() && NoGradScope::is_grad_enabled();
    Tensor result(a.shape(), req_grad);

    for (size_t i = 0; i < a.numel(); ++i) {
        result[i] = std::tanh(a[i]);
    }

    if (req_grad) {
        auto node = std::make_shared<AutogradNode>();
        node->add_parent(a.autograd_node());
        auto res_ptr = std::make_shared<Tensor>(result);
        auto a_ptr = std::make_shared<Tensor>(a);

        node->set_backward_fn([a_ptr, res_ptr](const Tensor& grad) {
            Tensor da(a_ptr->shape(), false);
            for (size_t i = 0; i < a_ptr->numel(); ++i) {
                float t = (*res_ptr)[i];
                da[i] = grad[i] * (1.0f - t * t);
            }
            a_ptr->backward(da);
        });
        result.set_autograd_node(node);
    }

    return result;
}

Tensor gelu(const Tensor& a) {
    bool req_grad = a.requires_grad() && NoGradScope::is_grad_enabled();
    Tensor result(a.shape(), req_grad);

    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    constexpr float sqrt_2_pi = 0.7978845608f;
    for (size_t i = 0; i < a.numel(); ++i) {
        float x = a[i];
        float inner = sqrt_2_pi * (x + 0.044715f * x * x * x);
        result[i] = 0.5f * x * (1.0f + std::tanh(inner));
    }

    if (req_grad) {
        auto node = std::make_shared<AutogradNode>();
        node->add_parent(a.autograd_node());
        auto a_ptr = std::make_shared<Tensor>(a);

        node->set_backward_fn([a_ptr](const Tensor& grad) {
            Tensor da(a_ptr->shape(), false);
            for (size_t i = 0; i < a_ptr->numel(); ++i) {
                float x = (*a_ptr)[i];
                float inner = sqrt_2_pi * (x + 0.044715f * x * x * x);
                float tanh_inner = std::tanh(inner);
                float sech2 = 1.0f - tanh_inner * tanh_inner;
                float d_inner = sqrt_2_pi * (1.0f + 0.134145f * x * x);
                da[i] = grad[i] * (0.5f * (1.0f + tanh_inner) + 0.5f * x * sech2 * d_inner);
            }
            a_ptr->backward(da);
        });
        result.set_autograd_node(node);
    }

    return result;
}

Tensor softmax(const Tensor& a, int dim) {
    (void)dim;
    bool req_grad = a.requires_grad() && NoGradScope::is_grad_enabled();
    Tensor result(a.shape(), req_grad);

    size_t last_dim = a.shape()[a.ndim() - 1];
    size_t outer_numel = a.numel() / last_dim;

    for (size_t o = 0; o < outer_numel; ++o) {
        size_t offset = o * last_dim;
        float max_val = a[offset];
        for (size_t i = 1; i < last_dim; ++i) {
            max_val = std::max(max_val, a[offset + i]);
        }
        float sum_exp = 0.0f;
        for (size_t i = 0; i < last_dim; ++i) {
            result[offset + i] = std::exp(a[offset + i] - max_val);
            sum_exp += result[offset + i];
        }
        for (size_t i = 0; i < last_dim; ++i) {
            result[offset + i] /= sum_exp;
        }
    }

    if (req_grad) {
        auto node = std::make_shared<AutogradNode>();
        node->add_parent(a.autograd_node());
        auto res_ptr = std::make_shared<Tensor>(result);
        auto a_ptr = std::make_shared<Tensor>(a);

        node->set_backward_fn([a_ptr, res_ptr, last_dim, outer_numel](const Tensor& grad) {
            Tensor da(a_ptr->shape(), false);
            for (size_t o = 0; o < outer_numel; ++o) {
                size_t offset = o * last_dim;
                float sum_grad_s = 0.0f;
                for (size_t i = 0; i < last_dim; ++i) {
                    sum_grad_s += grad[offset + i] * (*res_ptr)[offset + i];
                }
                for (size_t i = 0; i < last_dim; ++i) {
                    float s = (*res_ptr)[offset + i];
                    da[offset + i] = s * (grad[offset + i] - sum_grad_s);
                }
            }
            a_ptr->backward(da);
        });
        result.set_autograd_node(node);
    }

    return result;
}

Tensor log_softmax(const Tensor& a, int dim) {
    (void)dim;
    bool req_grad = a.requires_grad() && NoGradScope::is_grad_enabled();
    Tensor s = softmax(a, dim);
    Tensor result(a.shape(), req_grad);
    for (size_t i = 0; i < a.numel(); ++i) {
        result[i] = std::log(std::max(s[i], 1e-12f));
    }

    if (req_grad) {
        auto node = std::make_shared<AutogradNode>();
        node->add_parent(a.autograd_node());
        auto a_ptr = std::make_shared<Tensor>(a);
        auto s_ptr = std::make_shared<Tensor>(s);

        size_t last_dim = a.shape()[a.ndim() - 1];
        size_t outer_numel = a.numel() / last_dim;

        node->set_backward_fn([a_ptr, s_ptr, last_dim, outer_numel](const Tensor& grad) {
            Tensor da(a_ptr->shape(), false);
            for (size_t o = 0; o < outer_numel; ++o) {
                size_t offset = o * last_dim;
                float sum_grad = 0.0f;
                for (size_t i = 0; i < last_dim; ++i) {
                    sum_grad += grad[offset + i];
                }
                for (size_t i = 0; i < last_dim; ++i) {
                    float s_val = (*s_ptr)[offset + i];
                    da[offset + i] = grad[offset + i] - s_val * sum_grad;
                }
            }
            a_ptr->backward(da);
        });
        result.set_autograd_node(node);
    }

    return result;
}

Tensor sum(const Tensor& a, int dim, bool keepdim) {
    (void)dim;
    (void)keepdim;
    bool req_grad = a.requires_grad() && NoGradScope::is_grad_enabled();
    float total_sum = 0.0f;
    for (size_t i = 0; i < a.numel(); ++i) {
        total_sum += a[i];
    }
    Tensor result = Tensor::scalar(total_sum, req_grad);

    if (req_grad) {
        auto node = std::make_shared<AutogradNode>();
        node->add_parent(a.autograd_node());
        auto a_ptr = std::make_shared<Tensor>(a);

        node->set_backward_fn([a_ptr](const Tensor& grad) {
            float g_val = grad.item();
            Tensor da(a_ptr->shape(), g_val, false);
            a_ptr->backward(da);
        });
        result.set_autograd_node(node);
    }

    return result;
}

Tensor mean(const Tensor& a, int dim, bool keepdim) {
    Tensor s = sum(a, dim, keepdim);
    return s.div(static_cast<float>(a.numel()));
}

} // namespace ops
} // namespace deeplearningcore
