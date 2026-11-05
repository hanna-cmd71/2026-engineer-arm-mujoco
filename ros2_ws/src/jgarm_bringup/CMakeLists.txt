cmake_minimum_required(VERSION 3.24)
project(jgarm_bringup)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY launch config
  DESTINATION "share/${PROJECT_NAME}")

ament_package()
