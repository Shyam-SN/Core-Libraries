find_package(Qt6 COMPONENTS Widgets QUIET)
if(NOT Qt6_FOUND)
    find_package(Qt5 COMPONENTS Widgets QUIET)
endif()

if(Qt6_FOUND OR Qt5_FOUND)
    set(CMAKE_AUTOMOC ON)
    set(CMAKE_AUTOUIC ON)
    set(CMAKE_AUTORCC ON)

    add_executable(graphcore_qt_app
        main.cpp
        main_window.cpp
        graph_canvas.cpp
        tour_guide_dialog.cpp
    )

    target_link_libraries(graphcore_qt_app PRIVATE graphcore)

    if(Qt6_FOUND)
        target_link_libraries(graphcore_qt_app PRIVATE Qt6::Widgets)
    else()
        target_link_libraries(graphcore_qt_app PRIVATE Qt5::Widgets)
    endif()
    message(STATUS "Qt Interactive Studio enabled for GraphCore")
else()
    message(STATUS "Qt not found. Skipping GraphCore Qt GUI desktop application.")
endif()
