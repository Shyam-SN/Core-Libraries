#include "tour_guide_dialog.hpp"

namespace graphcore {

TourGuideDialog::TourGuideDialog(QWidget* parent) : QDialog(parent) {
    setWindowTitle("Welcome to GraphCore Interactive Studio - Tour Guide");
    resize(550, 400);
    setup_ui();
}

void TourGuideDialog::setup_ui() {
    auto* main_layout = new QVBoxLayout(this);

    stack_widget_ = new QStackedWidget(this);

    // Page 1: Canvas Navigation & Node Editing
    stack_widget_->addWidget(create_page(
        "1. Interactive Graph Canvas & Editing",
        "Create, drag, and connect nodes in real time.",
        "• Click on the canvas to add a new graph node.\n"
        "• Drag between two nodes to construct a weighted directed/undirected edge.\n"
        "• Drag nodes around to manually position them or pan/zoom the infinite canvas."
    ));

    // Page 2: Algorithm Selection
    stack_widget_->addWidget(create_page(
        "2. Comprehensive Graph Algorithm Suite",
        "Select from 30+ graph algorithms.",
        "• Shortest Paths: Dijkstra, A*, Bellman-Ford, Floyd-Warshall.\n"
        "• Network Flow & MST: Dinic Max Flow, $s$-$t$ Min Cut, Kruskal & Prim MST.\n"
        "• Connectivity & Topology: Tarjan SCC, Topological Sort, Bron-Kerbosch Cliques.\n"
        "• Centrality & Community: PageRank, Betweenness, Louvain Community Detection."
    ));

    // Page 3: Step-by-Step Animation Engine
    stack_widget_->addWidget(create_page(
        "3. Step-by-Step Visualizer & Animation Engine",
        "Inspect execution state at every step.",
        "• Use Play/Pause/Next/Previous buttons to step through graph algorithms.\n"
        "• Inspect the Queue/Stack/Priority Queue state panel in real time.\n"
        "• View active nodes, visited frontiers, and shortest path tree highlights."
    ));

    // Page 4: Force-Directed Layouts & Benchmarking
    stack_widget_->addWidget(create_page(
        "4. Force-Directed Layouts & Performance Dashboard",
        "Automated layout generation and benchmarking.",
        "• Click 'Simulate Force Layout' to auto-organize random graphs via physics repulsion.\n"
        "• Compare Dijkstra vs A* execution time side-by-side.\n"
        "• Generate random Erdos-Renyi, Grid, or Scale-Free graphs in one click."
    ));

    main_layout->addWidget(stack_widget_);

    // Footer controls
    auto* footer = new QHBoxLayout();
    label_step_indicator_ = new QLabel("Step 1 of 4", this);

    btn_prev_ = new QPushButton("Previous", this);
    btn_next_ = new QPushButton("Next", this);
    btn_skip_ = new QPushButton("Skip Tour", this);

    btn_prev_->setEnabled(false);

    connect(btn_prev_, &QPushButton::clicked, this, &TourGuideDialog::prev_step);
    connect(btn_next_, &QPushButton::clicked, this, &TourGuideDialog::next_step);
    connect(btn_skip_, &QPushButton::clicked, this, &QDialog::accept);

    footer->addWidget(label_step_indicator_);
    footer->addStretch();
    footer->addWidget(btn_prev_);
    footer->addWidget(btn_next_);
    footer->addWidget(btn_skip_);

    main_layout->addLayout(footer);
}

QWidget* TourGuideDialog::create_page(const QString& title, const QString& description, const QString& details) {
    auto* page = new QWidget(this);
    auto* layout = new QVBoxLayout(page);

    auto* lbl_title = new QLabel(QString("<h2>%1</h2>").arg(title), page);
    auto* lbl_desc = new QLabel(QString("<i>%1</i>").arg(description), page);
    auto* lbl_details = new QLabel(details, page);
    lbl_details->setWordWrap(true);

    layout->addWidget(lbl_title);
    layout->addWidget(lbl_desc);
    layout->addSpacing(10);
    layout->addWidget(lbl_details);
    layout->addStretch();

    return page;
}

void TourGuideDialog::next_step() {
    int idx = stack_widget_->currentIndex();
    if (idx + 1 < stack_widget_->count()) {
        stack_widget_->setCurrentIndex(idx + 1);
        label_step_indicator_->setText(QString("Step %1 of %2").arg(idx + 2).arg(stack_widget_->count()));
        btn_prev_->setEnabled(true);
        if (idx + 2 == stack_widget_->count()) {
            btn_next_->setText("Finish & Launch");
        }
    } else {
        accept();
    }
}

void TourGuideDialog::prev_step() {
    int idx = stack_widget_->currentIndex();
    if (idx > 0) {
        stack_widget_->setCurrentIndex(idx - 1);
        label_step_indicator_->setText(QString("Step %1 of %2").arg(idx).arg(stack_widget_->count()));
        btn_next_->setText("Next");
        if (idx - 1 == 0) {
            btn_prev_->setEnabled(false);
        }
    }
}

} // namespace graphcore
