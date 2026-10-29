cmake_minimum_required(VERSION 3.10.0)
project(ceres_learn VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Ceres REQUIRED)

add_executable(base01 base01.cpp)
target_link_libraries(base01 PRIVATE Ceres::ceres)

add_executable(base02_curve_fitting base02_curve_fitting.cpp)
target_link_libraries(base02_curve_fitting PRIVATE Ceres::ceres)

add_executable(base03_pose_graph base03_pose_graph.cpp)
target_link_libraries(base03_pose_graph PRIVATE Ceres::ceres)

add_executable(base04_ba base04_ba.cpp)
target_link_libraries(base04_ba PRIVATE Ceres::ceres)

add_executable(base05_imu base05_imu.cpp)
target_link_libraries(base05_imu PRIVATE Ceres::ceres)

add_executable(base06_marginalization base06_marginalization.cpp)
target_link_libraries(base06_marginalization PRIVATE Ceres::ceres)

add_executable(base07_engineering base07_engineering.cpp)
target_link_libraries(base07_engineering PRIVATE Ceres::ceres)

# 课后作业参考答案
add_executable(hw01_basics assignments/hw01_basics.cpp)
target_link_libraries(hw01_basics PRIVATE Ceres::ceres)

add_executable(hw02_robust assignments/hw02_robust.cpp)
target_link_libraries(hw02_robust PRIVATE Ceres::ceres)

add_executable(hw03_pose_graph assignments/hw03_pose_graph.cpp)
target_link_libraries(hw03_pose_graph PRIVATE Ceres::ceres)

add_executable(hw04_ba assignments/hw04_ba.cpp)
target_link_libraries(hw04_ba PRIVATE Ceres::ceres)

add_executable(hw05_imu assignments/hw05_imu.cpp)
target_link_libraries(hw05_imu PRIVATE Ceres::ceres)

add_executable(hw06_marginalization assignments/hw06_marginalization.cpp)
target_link_libraries(hw06_marginalization PRIVATE Ceres::ceres)

add_executable(hw07_engineering assignments/hw07_engineering.cpp)
target_link_libraries(hw07_engineering PRIVATE Ceres::ceres)

#课后练习
add_executable(my_hwo1 assignments/my_hw01.cpp)
target_link_libraries(my_hwo1 PRIVATE Ceres::ceres)

add_executable(my_hwo2 assignments/my_hw02.cpp)
target_link_libraries(my_hwo2 PRIVATE Ceres::ceres)
