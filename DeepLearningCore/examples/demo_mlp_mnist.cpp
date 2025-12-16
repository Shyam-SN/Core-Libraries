#include <deeplearningcore/deeplearningcore.hpp>
#include <iostream>
#include <iomanip>
#include <memory>

using namespace deeplearningcore;

class MLPClassifier : public Module {
public:
    MLPClassifier(size_t in_features, size_t hidden_features, size_t num_classes) {
        fc1_ = std::make_shared<Linear>(in_features, hidden_features);
        bn1_ = std::make_shared<BatchNorm1D>(hidden_features);
        fc2_ = std::make_shared<Linear>(hidden_features, num_classes);

        register_module("fc1", fc1_);
        register_module("bn1", bn1_);
        register_module("fc2", fc2_);
    }

    Tensor forward(const Tensor& input) override {
        Tensor x = (*fc1_)(input);
        x = (*bn1_)(x);
        x = x.relu();
        x = (*fc2_)(x);
        return x;
    }

private:
    std::shared_ptr<Linear> fc1_;
    std::shared_ptr<BatchNorm1D> bn1_;
    std::shared_ptr<Linear> fc2_;
};

int main() {
    std::cout << "=========================================================\n";
    std::cout << " DeepLearningCore: Multi-Layer Perceptron Training Demo  \n";
    std::cout << "=========================================================\n\n";

    constexpr size_t num_samples = 128;
    constexpr size_t in_features = 8;
    constexpr size_t num_classes = 3;

    // Create synthetic classification dataset
    Tensor X = Tensor::randn(Shape{num_samples, in_features}, 0.0f, 1.0f, false);
    Tensor Y_onehot = Tensor::zeros(Shape{num_samples, num_classes}, false);

    for (size_t i = 0; i < num_samples; ++i) {
        size_t cls = i % num_classes;
        Y_onehot.set_at({i, cls}, 1.0f);
    }

    TensorDataset dataset(X, Y_onehot);
    DataLoader dataloader(dataset, 32, true);

    auto model = std::make_shared<MLPClassifier>(in_features, 16, num_classes);
    Adam optimizer(model->parameters(), 0.02f);
    CrossEntropyLoss loss_fn;

    std::cout << "Training MLP Classifier on synthetic dataset...\n\n";
    Timer timer;

    for (int epoch = 1; epoch <= 15; ++epoch) {
        float epoch_loss = 0.0f;
        size_t batch_count = 0;

        for (auto [x_batch, y_batch] : dataloader) {
            optimizer.zero_grad();

            Tensor preds = (*model)(x_batch);
            Tensor loss = loss_fn(preds, y_batch);

            loss.backward();
            optimizer.step();

            epoch_loss += loss.item();
            batch_count++;
        }

        float avg_loss = epoch_loss / static_cast<float>(batch_count);
        std::cout << "Epoch [" << std::setw(2) << epoch << "/15] - Loss: "
                  << std::fixed << std::setprecision(4) << avg_loss << "\n";
    }

    std::cout << "\nTraining finished in " << timer.elapsed_ms() << " ms.\n";
    std::cout << "Demonstrating Model Weights Checkpoint Serialization...\n";
    Serialization::save_weights(*model, "mlp_checkpoint.bin");
    std::cout << "Model checkpoint saved to 'mlp_checkpoint.bin' successfully.\n\n";

    std::cout << "=========================================================\n";
    std::cout << " DeepLearningCore Demo Completed Successfully!           \n";
    std::cout << "=========================================================\n";

    return 0;
}
