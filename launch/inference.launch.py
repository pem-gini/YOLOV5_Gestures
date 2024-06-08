from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg = get_package_share_directory('YOLOV5_Gestures')
    return LaunchDescription([
        DeclareLaunchArgument('checkpoint', description="Checkpoint path for model used", default_value=os.path.join(pkg, "runs", "train", "exp20", "weights", "best.pt")),
        Node(
            package='YOLOV5_Gestures',
            executable='gesture_recognition_inference_node',
            name='gesture_recognition_inference_node',
            output='screen',
            emulate_tty=True,
            parameters=[{
                "model_checkpoint": LaunchConfiguration('checkpoint'),
                # "rgb_topic": "/oak/rgb/image_raw",
                # "depth_topic": "/oak/stereo/image_raw",
                "visualize": True,
            }]
        ),
    ])

