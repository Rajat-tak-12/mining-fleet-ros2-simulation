# Autonomous Mining Fleet Simulation (ROS 2)

Smart India Hackathon 2026 – Problem Statement 26007 (cleared internal round).

A ROS 2 simulation of a 6-vehicle automated haul-truck fleet, inspired by the
NMDC Bailadila iron-ore mines. Each vehicle publishes live sensor data and the
whole fleet is monitored from a single teleoperation dashboard.

## Features
- 6 dumper trucks with independent motion, braking, turn signals and payload state
- Per-vehicle sensor topics: LiDAR point cloud, IMU, GPS, camera, and TF transforms
- 6-channel live cockpit dashboard showing each vehicle's POV with range and speed telemetry
- RViz markers for the mine scene
- QoS reliability profiles matched between publishers and subscribers for instant streaming

## Tech stack
Python 3 · ROS 2 · RViz · OpenCV · NumPy · tf2 · cv_bridge

## Setup
- ROS 2 (Ubuntu) with `rclpy`, `tf2_ros`, `cv_bridge`
- Python packages: `opencv-python`, `numpy`

The main node is `mining_vehicle/rviz_markers.py` (class `BailadilaProductionEngine`).
To run it, place it inside a ROS 2 Python package named `mining_vehicle`, register it as an
entry point in `setup.py`, then build and run:
```bash
colcon build --packages-select mining_vehicle
source install/setup.bash
ros2 run mining_vehicle <your_entry_point_name>
```
In RViz, add the `/mining_scene_markers` display and the `/fleet/cockpit_monitor` image topic.

> Package files (`setup.py`, `package.xml`, launch file) are not yet included in this repo.

## Topics
| Topic | Type |
|---|---|
| `/dumper_{1-6}/camera/image_raw` | sensor_msgs/Image |
| `/dumper_{1-6}/lidar/points` | sensor_msgs/PointCloud2 |
| `/dumper_{1-6}/imu/data` | sensor_msgs/Imu |
| `/dumper_{1-6}/gps/fix` | sensor_msgs/NavSatFix |
| `/fleet/cockpit_monitor` | sensor_msgs/Image |
| `/mining_scene_markers` | visualization_msgs/MarkerArray |

## Author
Rajat Tak – B.Tech Electronics Engineering, NIELIT Aurangabad
