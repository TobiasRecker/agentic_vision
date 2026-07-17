cmake_minimum_required(VERSION 3.16)
project(oak4_fullres_capture)

find_package(ament_cmake REQUIRED)
find_package(depthai_v3 REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_srvs REQUIRED)

add_executable(oak4_fullres_capture_node src/oak4_fullres_capture_node.cpp)
target_compile_features(oak4_fullres_capture_node PRIVATE cxx_std_17)
target_compile_options(oak4_fullres_capture_node PRIVATE -Wall -Wextra -Wpedantic)
ament_target_dependencies(oak4_fullres_capture_node rclcpp sensor_msgs std_srvs)
target_link_libraries(oak4_fullres_capture_node depthai_v3::core ${OpenCV_LIBS})

install(TARGETS oak4_fullres_capture_node DESTINATION lib/${PROJECT_NAME})

ament_package()
