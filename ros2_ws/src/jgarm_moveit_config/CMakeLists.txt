cmake_minimum_required(VERSION 3.24)
project(jgarm_moveit_config)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY config launch
  DESTINATION "share/${PROJECT_NAME}")
install(
  FILES "${CMAKE_CURRENT_SOURCE_DIR}/../jgarm_description/urdf/JGarm.ros.urdf"
  DESTINATION "share/${PROJECT_NAME}/config"
  RENAME "JGarm.urdf")

ament_package()
