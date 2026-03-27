#pragma once

#include <QDialog>
#include <QLabel>
#include <QPushButton>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QStackedWidget>

namespace graphcore {

class TourGuideDialog : public QDialog {
    Q_OBJECT
public:
    explicit TourGuideDialog(QWidget* parent = nullptr);

private slots:
    void next_step();
    void prev_step();

private:
    QStackedWidget* stack_widget_{nullptr};
    QPushButton* btn_prev_{nullptr};
    QPushButton* btn_next_{nullptr};
    QPushButton* btn_skip_{nullptr};
    QLabel* label_step_indicator_{nullptr};

    void setup_ui();
    QWidget* create_page(const QString& title, const QString& description, const QString& details);
};

} // namespace graphcore
