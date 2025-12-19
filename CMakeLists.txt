cmake_minimum_required(VERSION 3.8)
project(gr2_mimic)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 1. Find Dependencies
find_package(ament_cmake REQUIRED)
# Note: rclpy, std_msgs, etc. are usually runtime dependencies for Python
# and don't strictly need to be listed here unless you are compiling C++ nodes.

# 2. Install Directories (CRITICAL for Gazebo to find your meshes)
# This copies your folders to the 'install' directory
install(DIRECTORY launch urdf meshes config
  DESTINATION share/${PROJECT_NAME}
)

# 3. Install Python Scripts (Optional)
# If you create a "scripts" folder with python nodes later, uncomment below:
# install(PROGRAMS
#   scripts/your_python_script.py
#   DESTINATION lib/${PROJECT_NAME}
# )

# 4. Build the Package
ament_package()