#include "main_window.hpp"
#include "tour_guide_dialog.hpp"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QSplitter>
#include <QGroupBox>
#include <QMenuBar>
#include <QStatusBar>

namespace graphcore {

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent) {
    setWindowTitle("GraphCore: Research-Grade Graph Processing & Qt Visualizer");
    resize(1100, 750);
    setup_ui();
    generate_random_graph();
}

void MainWindow::setup_ui() {
    auto* central_widget = new QWidget(this);
    setCentralWidget(central_widget);

    auto* main_layout = new QHBoxLayout(central_widget);

    auto* splitter = new QSplitter(Qt::Horizontal, central_widget);

    // Left Control Panel
    auto* left_panel = new QWidget(this);
    auto* left_layout = new QVBoxLayout(left_panel);

    auto* group_control = new QGroupBox("Algorithm Controls", left_panel);
    auto* control_layout = new QVBoxLayout(group_control);

    combo_algo_ = new QComboBox(group_control);
    combo_algo_->addItem("BFS Traversal");
    combo_algo_->addItem("DFS Traversal");
    combo_algo_->addItem("Dijkstra Shortest Path");
    combo_algo_->addItem("A* Search");
    combo_algo_->addItem("Kruskal MST");
    combo_algo_->addItem("Tarjan SCC");
    combo_algo_->addItem("PageRank Centrality");

    btn_run_ = new QPushButton("Run Algorithm", group_control);
    auto* btn_gen = new QPushButton("Generate Random Graph", group_control);
    btn_tour_ = new QPushButton("Launch Tour Guide", group_control);

    connect(btn_gen, &QPushButton::clicked, this, &MainWindow::generate_random_graph);
    connect(btn_run_, &QPushButton::clicked, this, &MainWindow::run_selected_algorithm);
    connect(btn_tour_, &QPushButton::clicked, this, &MainWindow::show_tour_guide);

    control_layout->addWidget(new QLabel("Select Algorithm:", group_control));
    control_layout->addWidget(combo_algo_);
    control_layout->addWidget(btn_run_);
    control_layout->addWidget(btn_gen);
    control_layout->addSpacing(10);
    control_layout->addWidget(btn_tour_);

    // Animation Controls
    auto* group_anim = new QGroupBox("Step Visualizer", left_panel);
    auto* anim_layout = new QHBoxLayout(group_anim);
    btn_prev_ = new QPushButton("Step Prev", group_anim);
    btn_next_ = new QPushButton("Step Next", group_anim);

    connect(btn_prev_, &QPushButton::clicked, this, &MainWindow::step_prev);
    connect(btn_next_, &QPushButton::clicked, this, &MainWindow::step_next);

    anim_layout->addWidget(btn_prev_);
    anim_layout->addWidget(btn_next_);

    // Step Log & Execution Panel
    text_log_ = new QTextEdit(left_panel);
    text_log_->setReadOnly(true);

    left_layout->addWidget(group_control);
    left_layout->addWidget(group_anim);
    left_layout->addWidget(new QLabel("Step-by-Step Execution Log:"));
    left_layout->addWidget(text_log_);

    // Right Canvas Panel
    canvas_ = new GraphCanvas(this);

    splitter->addWidget(left_panel);
    splitter->addWidget(canvas_);
    splitter->setStretchFactor(1, 2);

    main_layout->addWidget(splitter);

    statusBar()->showMessage("GraphCore Ready. Click 'Launch Tour Guide' for quick tutorial.");
}

void MainWindow::show_tour_guide() {
    TourGuideDialog dialog(this);
    dialog.exec();
}

void MainWindow::generate_random_graph() {
    current_graph_ = GraphGenerators::erdos_renyi(10, 0.25, false);
    auto pos = ForceDirectedLayout::compute_layout(current_graph_, 100, 600, 500);
    canvas_->set_graph(current_graph_);
    canvas_->set_positions(pos);
    text_log_->setText("[INFO] Generated random graph with 10 vertices.");
}

void MainWindow::run_selected_algorithm() {
    animation_.clear();
    int idx = combo_algo_->currentIndex();

    if (idx == 0) { // BFS
        auto order = Traversal::bfs(current_graph_, 0);
        std::unordered_set<size_t> visited;
        for (size_t u : order) {
            visited.insert(u);
            animation_.add_step(QString("Visited node %1").arg(u).toStdString(), visited, {u});
        }
    } else {
        auto order = Traversal::dfs(current_graph_, 0);
        std::unordered_set<size_t> visited;
        for (size_t u : order) {
            visited.insert(u);
            animation_.add_step(QString("DFS Visited node %1").arg(u).toStdString(), visited, {u});
        }
    }

    if (animation_.total_steps() > 0) {
        canvas_->apply_animation_step(animation_.get_current_step());
        text_log_->setText(QString::fromStdString(animation_.get_current_step()->description));
    }
}

void MainWindow::step_next() {
    if (animation_.step_next()) {
        const auto* step = animation_.get_current_step();
        canvas_->apply_animation_step(step);
        text_log_->append(QString::fromStdString(step->description));
    }
}

void MainWindow::step_prev() {
    if (animation_.step_prev()) {
        const auto* step = animation_.get_current_step();
        canvas_->apply_animation_step(step);
        text_log_->append(QString("Reverted to step %1").arg(animation_.current_step_index()));
    }
}

} // namespace graphcore
