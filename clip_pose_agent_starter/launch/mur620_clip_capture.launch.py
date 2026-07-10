from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    params_file = LaunchConfiguration("params_file")

    capture_node = Node(
        package="clip_pose_agent_starter",
        executable="clip_object_capture_session",
        name="clip_object_capture_session",
        output="screen",
        emulate_tty=True,
        parameters=[
            params_file,
            {
                "image_topic": LaunchConfiguration("image_topic"),
                "image_compressed": LaunchConfiguration("image_compressed"),
                "camera_info_topic": LaunchConfiguration("camera_info_topic"),
                "pointcloud_topic": LaunchConfiguration("pointcloud_topic"),
                "output_root": LaunchConfiguration("output_root"),
                "session_name": LaunchConfiguration("session_name"),
                "robot_base_frame": LaunchConfiguration("robot_base_frame"),
                "robot_tcp_frame": LaunchConfiguration("robot_tcp_frame"),
                "camera_frame": LaunchConfiguration("camera_frame"),
                "extra_tf_topics": LaunchConfiguration("extra_tf_topics"),
                "extra_tf_static_topics": LaunchConfiguration("extra_tf_static_topics"),
                "use_configured_tcp_to_camera": LaunchConfiguration("use_configured_tcp_to_camera"),
                "tcp_to_camera_translation_xyz": LaunchConfiguration("tcp_to_camera_translation_xyz"),
                "tcp_to_camera_quaternion_xyzw": LaunchConfiguration("tcp_to_camera_quaternion_xyzw"),
                "planning_frame": LaunchConfiguration("planning_frame"),
                "action_name": LaunchConfiguration("action_name"),
                "move_enabled": LaunchConfiguration("move_enabled"),
                "keyboard_jog_enabled": LaunchConfiguration("keyboard_jog_enabled"),
                "jog_twist_topic": LaunchConfiguration("jog_twist_topic"),
                "jog_frame": LaunchConfiguration("jog_frame"),
                "samples": LaunchConfiguration("samples"),
                "allow_2d_center_fallback": LaunchConfiguration("allow_2d_center_fallback"),
                "fallback_center_depth_m": LaunchConfiguration("fallback_center_depth_m"),
            },
        ],
    )

    fullres_node = Node(
        package="oak4_fullres_capture",
        executable="oak4_fullres_capture_node",
        name="oak_fullres_capture",
        output="screen",
        emulate_tty=True,
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("clip_pose_agent_starter"),
                        "config",
                        "clip_capture.yaml",
                    ]
                ),
            ),
            DeclareLaunchArgument("image_topic", default_value="/oak/rgb/image_raw/compressed"),
            DeclareLaunchArgument("image_compressed", default_value="true"),
            DeclareLaunchArgument("camera_info_topic", default_value="/oak/rgb/camera_info"),
            DeclareLaunchArgument("pointcloud_topic", default_value="/oak/rgbd/points"),
            DeclareLaunchArgument("output_root", default_value="~/clip_pose_sessions"),
            DeclareLaunchArgument("session_name", default_value=""),
            DeclareLaunchArgument("robot_base_frame", default_value="mur620d/UR10_r/base_link"),
            DeclareLaunchArgument("robot_tcp_frame", default_value="mur620d/UR10_r/tool0"),
            DeclareLaunchArgument("camera_frame", default_value=""),
            DeclareLaunchArgument("extra_tf_topics", default_value=""),
            DeclareLaunchArgument("extra_tf_static_topics", default_value=""),
            DeclareLaunchArgument("use_configured_tcp_to_camera", default_value="true"),
            DeclareLaunchArgument("tcp_to_camera_translation_xyz", default_value="0.0068564203,-0.0892312561,0.1018930213"),
            DeclareLaunchArgument(
                "tcp_to_camera_quaternion_xyzw",
                default_value="0.0241307793,-0.0030488269,-0.0062149980,0.9996848423",
            ),
            DeclareLaunchArgument("planning_frame", default_value="mur620d/UR10_r/base_link"),
            DeclareLaunchArgument("action_name", default_value="/mur620d/jparse_move_r"),
            DeclareLaunchArgument("move_enabled", default_value="false"),
            DeclareLaunchArgument("keyboard_jog_enabled", default_value="false"),
            DeclareLaunchArgument(
                "jog_twist_topic",
                default_value="/mur620d/jparse_velocity_controller_r/twist_cmd",
            ),
            DeclareLaunchArgument("jog_frame", default_value="UR10_r/base_link"),
            DeclareLaunchArgument("samples", default_value="18"),
            DeclareLaunchArgument("allow_2d_center_fallback", default_value="true"),
            DeclareLaunchArgument("fallback_center_depth_m", default_value="0.45"),
            fullres_node,
            capture_node,
        ]
    )
