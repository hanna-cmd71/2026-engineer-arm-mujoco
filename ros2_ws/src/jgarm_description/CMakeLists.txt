cmake_minimum_required(VERSION 3.24)
project(jgarm_description)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../../../model/"
  DESTINATION "share/${PROJECT_NAME}/model")
install(
  FILES "${CMAKE_CURRENT_SOURCE_DIR}/../../../JGarm.urdf"
  DESTINATION "share/${PROJECT_NAME}/urdf")

ament_package()
