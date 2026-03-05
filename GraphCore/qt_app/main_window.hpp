#pragma once

#include <QMainWindow>
#include <QComboBox>
#include <QPushButton>
#include <QLabel>
#include <QTextEdit>
#include <QSlider>
#include <graphcore/graphcore.hpp>
#include "graph_canvas.hpp"

namespace graphcore {

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(QWidget* parent = nullptr);

    void show_tour_guide();

private slots:
    void generate_random_graph();
    void run_selected_algorithm();
    void step_next();
    void step_prev();

private:
    GraphCanvas* canvas_{nullptr};
    QComboBox* combo_algo_{nullptr};
    QPushButton* btn_run_{nullptr};
    QPushButton* btn_next_{nullptr};
    QPushButton* btn_prev_{nullptr};
    QPushButton* btn_tour_{nullptr};
    QTextEdit* text_log_{nullptr};
    QLabel* status_label_{nullptr};

    Graph current_graph_{0, false};
    AnimationEngine animation_;

    void setup_ui();
};

} // namespace graphcore
