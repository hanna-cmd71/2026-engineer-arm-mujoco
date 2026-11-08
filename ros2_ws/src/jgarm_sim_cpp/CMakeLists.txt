cmake_minimum_required(VERSION 3.24)
project(jgarm_sim_cpp)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(tf2_ros REQUIRED)

if(DEFINED ENV{MUJOCO_ROOT} AND NOT "$ENV{MUJOCO_ROOT}" STREQUAL "")
  set(_jgarm_default_mujoco_root "$ENV{MUJOCO_ROOT}")
else()
  set(_jgarm_default_mujoco_root "/sim-workspaces/mujoco-3.17.0")
endif()
set(
  MUJOCO_ROOT
  "${_jgarm_default_mujoco_root}"
  CACHE PATH
  "Path to the MuJoCo C/C++ SDK")
set(MUJOCO_INCLUDE_DIR "${MUJOCO_ROOT}/include")
set(MUJOCO_LIBRARY "${MUJOCO_ROOT}/lib/libmujoco.so.3.17.0")

if(NOT EXISTS "${MUJOCO_INCLUDE_DIR}/mujoco/mujoco.h")
  message(FATAL_ERROR "MuJoCo headers not found under ${MUJOCO_ROOT}")
endif()
if(NOT EXISTS "${MUJOCO_LIBRARY}")
  message(FATAL_ERROR "MuJoCo library not found at ${MUJOCO_LIBRARY}")
endif()

add_executable(mujoco_sim_node src/mujoco_sim_node.cpp)
target_include_directories(
  mujoco_sim_node
  PRIVATE
    "${MUJOCO_INCLUDE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(mujoco_sim_node "${MUJOCO_LIBRARY}")
set_target_properties(
  mujoco_sim_node
  PROPERTIES
    BUILD_RPATH "${MUJOCO_ROOT}/lib"
    INSTALL_RPATH "${MUJOCO_ROOT}/lib")

ament_target_dependencies(
  mujoco_sim_node
  rclcpp
  geometry_msgs
  nav_msgs
  sensor_msgs
  std_msgs
  std_srvs
  tf2_ros)

install(TARGETS mujoco_sim_node DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(
    test_hardware_profile
    test/test_hardware_profile.cpp)
  target_include_directories(
    test_hardware_profile
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
endif()

ament_package()
