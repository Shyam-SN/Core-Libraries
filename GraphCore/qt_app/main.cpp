#include <QApplication>
#include "main_window.hpp"
#include "tour_guide_dialog.hpp"

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);

    graphcore::MainWindow window;
    window.show();

    // Automatically trigger Onboarding Tour Guide on launch
    graphcore::TourGuideDialog tour(&window);
    tour.exec();

    return app.exec();
}
