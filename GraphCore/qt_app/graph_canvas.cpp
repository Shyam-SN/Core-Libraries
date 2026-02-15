#include "graph_canvas.hpp"
#include <QPainter>
#include <QMouseEvent>
#include <cmath>

namespace graphcore {

GraphCanvas::GraphCanvas(QWidget* parent) : QWidget(parent) {
    setMinimumSize(600, 400);
}

void GraphCanvas::set_graph(Graph g) {
    graph_ = std::move(g);
    node_pos_.resize(graph_.num_vertices());
    for (size_t i = 0; i < graph_.num_vertices(); ++i) {
        node_pos_[i] = {100.0 + (i % 5) * 100.0, 100.0 + (i / 5) * 100.0};
    }
    update();
}

void GraphCanvas::set_positions(std::vector<std::pair<double, double>> pos) {
    node_pos_ = std::move(pos);
    update();
}

void GraphCanvas::apply_animation_step(const AnimationStep* step) {
    current_step_ = step;
    update();
}

void GraphCanvas::paintEvent(QPaintEvent* event) {
    Q_UNUSED(event);
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);

    // Draw background
    painter.fillRect(rect(), QColor(245, 247, 250));

    // Draw Edges
    painter.setPen(QPen(QColor(180, 190, 200), 2));
    for (const auto& edge : graph_.edges()) {
        if (edge.source < node_pos_.size() && edge.target < node_pos_.size()) {
            QPointF p1(node_pos_[edge.source].first, node_pos_[edge.source].second);
            QPointF p2(node_pos_[edge.target].first, node_pos_[edge.target].second);
            painter.drawLine(p1, p2);
        }
    }

    // Draw Nodes
    double radius = 18.0;
    for (size_t i = 0; i < graph_.num_vertices(); ++i) {
        if (i >= node_pos_.size()) break;
        QPointF p(node_pos_[i].first, node_pos_[i].second);

        QColor fill_color(70, 130, 180); // Default Steel Blue
        if (current_step_) {
            if (current_step_->active_nodes.count(i)) {
                fill_color = QColor(230, 80, 80); // Active node Red
            } else if (current_step_->highlighted_nodes.count(i)) {
                fill_color = QColor(80, 200, 120); // Visited / Highlighted Green
            }
        }

        painter.setBrush(fill_color);
        painter.setPen(QPen(Qt::white, 2));
        painter.drawEllipse(p, radius, radius);

        // Node index text
        painter.setPen(Qt::white);
        painter.drawText(QRectF(p.x() - radius, p.y() - radius, radius * 2, radius * 2), Qt::AlignCenter, QString::number(i));
    }
}

void GraphCanvas::mousePressEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton) {
        size_t new_node = graph_.add_vertex();
        if (node_pos_.size() <= new_node) node_pos_.resize(new_node + 1);
        node_pos_[new_node] = {static_cast<double>(event->position().x()), static_cast<double>(event->position().y())};
        update();
    }
}

} // namespace graphcore
