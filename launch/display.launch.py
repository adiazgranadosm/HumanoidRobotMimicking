import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
import xacro

def generate_launch_description():
    # ========================================================================
    # 1. CONFIGURATION
    # ========================================================================
    pkg_name = 'gr2_mimic'
    file_name = 'GR2v2.1.1_fourier_hand_6dof.urdf' # <--- CONFIRM THIS NAME
    
    # Get paths
    pkg_share = get_package_share_directory(pkg_name)
    urdf_path = os.path.join(pkg_share, 'urdf', file_name)

    # ========================================================================
    # 2. FIX GAZEBO MODEL PATH (Fixes invisible meshes)
    # ========================================================================
    # We point to the PARENT of the package share directory.
    # This allows Gazebo to resolve "package://gr2_mimic/meshes/..."
    gazebo_models_path = os.path.dirname(pkg_share)
    
    if 'GAZEBO_MODEL_PATH' in os.environ:
        os.environ['GAZEBO_MODEL_PATH'] += ":" + gazebo_models_path
    else:
        os.environ['GAZEBO_MODEL_PATH'] = gazebo_models_path

    # ========================================================================
    # 3. LOAD ROBOT DESCRIPTION
    # ========================================================================
    # Process the URDF file
    doc = xacro.process_file(urdf_path)
    robot_desc = doc.toxml()

    # Node: Robot State Publisher
    # Publishes the URDF to the /robot_description topic so Gazebo/RViz can read it
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_desc}]
    )

    # ========================================================================
    # 4. LAUNCH GAZEBO
    # ========================================================================
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')]),
    )

    # ========================================================================
    # 5. SPAWN ROBOT
    # ========================================================================
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-topic', 'robot_description',
                   '-entity', 'gr2_robot',
                   '-x', '0.0',
                   '-y', '0.0',
                   '-z', '1.05'], # Start slightly in the air to prevent clipping
        output='screen'
    )

    # ========================================================================
    # 6. LOAD CONTROLLERS (Fixes "Weak/Limp" Robot)
    # ========================================================================
    
    # 1. Joint State Broadcaster (Publishes joint angles)
    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster"],
    )

    # 2. Position Controller (Holds the joints stiff)
    robot_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["forward_position_controller"],
    )

    # ========================================================================
    # 7. ORCHESTRATE
    # ========================================================================
    # We use event handlers to ensure controllers start ONLY after the robot spawns.
    
    return LaunchDescription([
        # Start Gazebo and RSP
        gazebo,
        robot_state_publisher, 
        spawn_entity,

        # Wait for Spawn to finish, then start Joint State Broadcaster
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=spawn_entity,
                on_exit=[joint_state_broadcaster_spawner],
            )
        ),

        # Wait for Joint State Broadcaster to start, then start Position Controller
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=joint_state_broadcaster_spawner,
                on_exit=[robot_controller_spawner],
            )
        ),
    ])