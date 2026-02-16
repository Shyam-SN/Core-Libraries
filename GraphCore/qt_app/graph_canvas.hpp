#pragma once

#include <QWidget>
#include <graphcore/graph/graph.hpp>
#include <graphcore/visualization/animation_engine.hpp>
#include <vector>
#include <utility>

namespace graphcore {

class GraphCanvas : public QWidget {
    Q_OBJECT
public:
    explicit GraphCanvas(QWidget* parent = nullptr);

    void set_graph(Graph g);
    const Graph& graph() const noexcept { return graph_; }

    void set_positions(std::vector<std::pair<double, double>> pos);
    void apply_animation_step(const AnimationStep* step);

protected:
    void paintEvent(QPaintEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;

private:
    Graph graph_{0, false};
    std::vector<std::pair<double, double>> node_pos_;
    const AnimationStep* current_step_{nullptr};
};

} // namespace graphcore
