import math
import random
import time
import threading
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import Image, PointCloud2, PointField, Imu, NavSatFix
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, TransformStamped
from tf2_ros import TransformBroadcaster
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

class BailadilaProductionEngine(Node):
    def __init__(self):
        super().__init__('bailadila_production_engine')
        
        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE
        )

        self.pub_markers = self.create_publisher(MarkerArray, '/mining_scene_markers', 1)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.bridge = CvBridge()

        self.pub_cams = {i: self.create_publisher(Image, f'/dumper_{i}/camera/image_raw', sensor_qos) for i in range(1, 7)}
        self.pub_lidar = {i: self.create_publisher(PointCloud2, f'/dumper_{i}/lidar/points', sensor_qos) for i in range(1, 7)}
        self.pub_imu = {i: self.create_publisher(Imu, f'/dumper_{i}/imu/data', sensor_qos) for i in range(1, 7)}
        self.pub_gps = {i: self.create_publisher(NavSatFix, f'/dumper_{i}/gps/fix', sensor_qos) for i in range(1, 7)}
        
        self.pub_cockpit = self.create_publisher(Image, '/fleet/cockpit_monitor', sensor_qos)

        self.w, self.h = 320, 220
        self.horizon_y = 105
        self.lane_right_x = -7.5
        self.lane_left_x = -11.5
        self.blue_lane_x = 0.0
        self.theta = 0.139626
        self.cos_t = math.cos(self.theta)

        self.sal_trees = [
            (-18.0, -22.0), (-18.0, 36.0), (12.0, -18.0), (14.0, 28.0),
            (-22.0, -5.0), (-22.0, 18.0), (26.0, -12.0), (28.0, 20.0),
            (-32.0, 2.0), (-32.0, 25.0), (32.0, -25.0), (35.0, 10.0), (14.0, 42.0), (-18.0, 48.0)
        ]

        random.seed(42)
        self.dust_particles = []
        for i in range(24):
            self.dust_particles.append({
                'truck': random.choice([3, 4, 6]),
                'dx': random.uniform(-3.5, -1.2),
                'dy': random.uniform(-0.8, 0.8),
                'dz': random.uniform(0.1, 1.2),
                'size': random.uniform(0.8, 1.8),
                'alpha': random.uniform(0.25, 0.55)
            })

        self.reset_simulation()
        
        self.timer = self.create_timer(0.04, self.tick)
        self.render_thread = threading.Thread(target=self.render_loop, daemon=True)
        self.render_thread.start()

        self.get_logger().info("Advanced Fleet Engine Active.")

    def reset_simulation(self):
        self.sim_time = 0.0
        self.state = {
            1: [0.0, -18.0, 1.5708],   # Orange
            2: [18.0, 2.0, 3.14159],   # Blue
            3: [-7.5, -8.0, 1.5708],   # Yellow
            4: [-7.5, -18.0, 1.5708],  # Green Ego
            5: [0.0, 22.0, -1.5708],   # Red
            6: [-11.5, 14.0, -1.5708]  # Purple
        }
        self.speeds = {1: 0.45, 2: 0.50, 3: 0.22, 4: 0.58, 5: 0.55, 6: 0.48}
        self.braking = {1: False, 2: False, 3: False, 4: False, 5: False, 6: False}
        self.turn_signal = {1: "NONE", 2: "NONE", 3: "NONE", 4: "NONE", 5: "NONE", 6: "NONE"}
        self.payload_status = {1: "HAULING_ORE", 2: "EMPTY_RETURNING", 3: "HAULING_ORE", 4: "HAULING_ORE", 5: "EMPTY_RETURNING", 6: "EMPTY_RETURNING"}
        self.t1_phase = "NORTHBOUND"
        self.t1_arc = 0.0
        self.t1_cleared = False
        self.t5_phase = "APPROACHING_NORTH_BAY"
        self.t2_phase = "APPROACHING_STOP_BAY"
        self.t2_arc = 0.0
        self.t4_phase = "APPROACH"
        self.t4_progress = 0.0
        self.t4_merge_progress = 0.0

    def create_header(self, frame_id="odom"):
        h = Header()
        h.frame_id = frame_id
        h.stamp = self.get_clock().now().to_msg()
        return h

    def calc_z(self, truck_id):
        curr_y = self.state[truck_id][1]
        if truck_id in [3, 4, 6]:
            ramp_z = 1.2 - (curr_y - 12.0) * math.tan(self.theta) + 0.45
            if truck_id in [3, 4] and curr_y > 16.0:
                blend = min(1.0, max(0.0, (curr_y - 16.0) / 12.0))
                return (1.0 - blend) * ramp_z + blend * 0.45
            return ramp_z
        return 0.45

    def publish_sensors_and_transforms(self, truck_id):
        pos = self.state[truck_id]
        z_elev = self.calc_z(truck_id)
        yaw = pos[2]
        pitch = -self.theta if truck_id in [3, 4] and pos[1] <= 18.0 else (self.theta if truck_id == 6 else 0.0)
        stamp = self.get_clock().now().to_msg()

        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = "odom"
        t.child_frame_id = f"dumper_{truck_id}_base_link"
        t.transform.translation.x = float(pos[0])
        t.transform.translation.y = float(pos[1])
        t.transform.translation.z = float(z_elev)
        
        cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
        cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
        t.transform.rotation.x = float(sp * cy)
        t.transform.rotation.y = float(-sp * sy)
        t.transform.rotation.z = float(sy * cp)
        t.transform.rotation.w = float(cp * cy)
        self.tf_broadcaster.sendTransform(t)

        t_lidar = TransformStamped()
        t_lidar.header.stamp = stamp
        t_lidar.header.frame_id = f"dumper_{truck_id}_base_link"
        t_lidar.child_frame_id = f"dumper_{truck_id}_lidar_link"
        t_lidar.transform.translation.z = 1.2
        t_lidar.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(t_lidar)

        imu_msg = Imu()
        imu_msg.header.stamp = stamp
        imu_msg.header.frame_id = f"dumper_{truck_id}_base_link"
        imu_msg.orientation.z = sy
        imu_msg.orientation.w = cy
        imu_msg.linear_acceleration.z = 9.81
        self.pub_imu[truck_id].publish(imu_msg)

        gps_msg = NavSatFix()
        gps_msg.header.stamp = stamp
        gps_msg.header.frame_id = f"dumper_{truck_id}_base_link"
        gps_msg.latitude = 18.69 + (pos[1] * 0.00001)
        gps_msg.longitude = 81.25 + (pos[0] * 0.00001)
        gps_msg.altitude = 350.0 + z_elev
        self.pub_gps[truck_id].publish(gps_msg)

        pc_header = Header(stamp=stamp, frame_id=f"dumper_{truck_id}_lidar_link")
        points = []
        azimuth_steps = np.linspace(0, 2 * math.pi, 36, endpoint=False)
        elevation_channels = np.radians(np.linspace(-15.0, 15.0, 6))

        for az in azimuth_steps:
            for el in elevation_channels:
                dist = random.uniform(5.0, 28.0)
                intensity = 150.0
                px = dist * math.cos(el) * math.cos(az)
                py = dist * math.cos(el) * math.sin(az)
                pz = dist * math.sin(el)
                points.append([float(px), float(py), float(pz), float(intensity)])

        pc_data = np.array(points, dtype=np.float32).tobytes()
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1)
        ]
        pc_msg = PointCloud2(
            header=pc_header, height=1, width=len(points),
            fields=fields, is_bigendian=False, point_step=16,
            row_step=16 * len(points), data=pc_data
        )
        self.pub_lidar[truck_id].publish(pc_msg)

    def add_sal_forest_tree(self, arr, base_id, x, y):
        trunk = Marker(header=self.create_header(), ns="sal_forest", id=base_id, type=Marker.CYLINDER, action=Marker.ADD)
        trunk.pose.position.x, trunk.pose.position.y, trunk.pose.position.z = float(x), float(y), 1.4
        trunk.pose.orientation.w = 1.0
        trunk.scale.x, trunk.scale.y, trunk.scale.z = 0.50, 0.50, 2.8
        trunk.color.r, trunk.color.g, trunk.color.b, trunk.color.a = 0.28, 0.16, 0.08, 1.0
        arr.markers.append(trunk)

        f1 = Marker(header=self.create_header(), ns="sal_forest", id=base_id+1, type=Marker.SPHERE, action=Marker.ADD)
        f1.pose.position.x, f1.pose.position.y, f1.pose.position.z = float(x), float(y), 3.2
        f1.pose.orientation.w = 1.0
        f1.scale.x, f1.scale.y, f1.scale.z = 3.6, 3.6, 2.2
        f1.color.r, f1.color.g, f1.color.b, f1.color.a = 0.10, 0.34, 0.12, 1.0
        arr.markers.append(f1)

        f2 = Marker(header=self.create_header(), ns="sal_forest", id=base_id+2, type=Marker.SPHERE, action=Marker.ADD)
        f2.pose.position.x, f2.pose.position.y, f2.pose.position.z = float(x), float(y), 4.4
        f2.pose.orientation.w = 1.0
        f2.scale.x, f2.scale.y, f2.scale.z = 2.4, 2.4, 1.8
        f2.color.r, f2.color.g, f2.color.b, f2.color.a = 0.14, 0.42, 0.16, 1.0
        arr.markers.append(f2)

    def add_hematite_highwall(self, arr, base_id, x, y, z, sx, sy, sz):
        h = Marker(header=self.create_header(), ns="highwalls", id=base_id, type=Marker.CUBE, action=Marker.ADD)
        h.pose.position.x, h.pose.position.y, h.pose.position.z = float(x), float(y), float(z)
        h.pose.orientation.w = 1.0
        h.scale.x, h.scale.y, h.scale.z = float(sx), float(sy), float(sz)
        h.color.r, h.color.g, h.color.b, h.color.a = 0.48, 0.16, 0.10, 1.0
        arr.markers.append(h)

    def add_nmdc_detailed_truck(self, arr, truck_id, base_id, pos, yaw, pitch, name, col):
        z_elev = float(pos[2])
        speed = self.speeds[truck_id]
        payload = self.payload_status[truck_id]
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
        cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
        qw, qx, qy, qz = cp * cy, sp * cy, -sp * sy, sy * cp

        ring = Marker(header=self.create_header(), ns="fleet_rings", id=base_id+1, type=Marker.CYLINDER, action=Marker.ADD)
        ring.pose.position.x, ring.pose.position.y, ring.pose.position.z = float(pos[0]), float(pos[1]), z_elev - 0.40
        ring.pose.orientation.w = 1.0
        ring.scale.x, ring.scale.y, ring.scale.z = 4.6, 4.6, 0.02
        ring.color.r, ring.color.g, ring.color.b, ring.color.a = col[0], col[1], col[2], 0.25
        arr.markers.append(ring)

        chassis = Marker(header=self.create_header(), ns="fleet_body", id=base_id+2, type=Marker.CUBE, action=Marker.ADD)
        chassis.pose.position.x, chassis.pose.position.y, chassis.pose.position.z = float(pos[0]), float(pos[1]), z_elev
        chassis.pose.orientation.x, chassis.pose.orientation.y, chassis.pose.orientation.z, chassis.pose.orientation.w = qx, qy, qz, qw
        chassis.scale.x, chassis.scale.y, chassis.scale.z = 2.9, 1.50, 0.48
        chassis.color.r, chassis.color.g, chassis.color.b, chassis.color.a = 0.16, 0.16, 0.16, 1.0
        arr.markers.append(chassis)

        bx, by = float(pos[0] - 0.38 * cos_y), float(pos[1] - 0.38 * sin_y)
        bed = Marker(header=self.create_header(), ns="fleet_body", id=base_id+3, type=Marker.CUBE, action=Marker.ADD)
        bed.pose.position.x, bed.pose.position.y, bed.pose.position.z = bx, by, z_elev + 0.48
        bed.pose.orientation.x, bed.pose.orientation.y, bed.pose.orientation.z, bed.pose.orientation.w = qx, qy, qz, qw
        bed.scale.x, bed.scale.y, bed.scale.z = 1.85, 1.40, 0.70
        bed.color.r, bed.color.g, bed.color.b, bed.color.a = col[0], col[1], col[2], 1.0
        arr.markers.append(bed)

        cx, cy_p = float(pos[0] + 0.58 * cos_y), float(pos[1] + 0.58 * sin_y)
        canopy = Marker(header=self.create_header(), ns="fleet_body", id=base_id+4, type=Marker.CUBE, action=Marker.ADD)
        canopy.pose.position.x, canopy.pose.position.y, canopy.pose.position.z = cx, cy_p, z_elev + 0.88
        canopy.pose.orientation.x, canopy.pose.orientation.y, canopy.pose.orientation.z, canopy.pose.orientation.w = qx, qy, qz, qw
        canopy.scale.x, canopy.scale.y, canopy.scale.z = 1.05, 1.40, 0.12
        canopy.color.r, canopy.color.g, canopy.color.b, canopy.color.a = col[0]*0.85, col[1]*0.85, col[2]*0.85, 1.0
        arr.markers.append(canopy)

        if payload == "HAULING_ORE":
            ore = Marker(header=self.create_header(), ns="fleet_body", id=base_id+5, type=Marker.SPHERE, action=Marker.ADD)
            ore.pose.position.x, ore.pose.position.y, ore.pose.position.z = bx, by, z_elev + 0.82
            ore.pose.orientation.w = 1.0
            ore.scale.x, ore.scale.y, ore.scale.z = 1.6, 1.20, 0.52
            ore.color.r, ore.color.g, ore.color.b, ore.color.a = 0.55, 0.15, 0.08, 1.0
            arr.markers.append(ore)

        cab_x = float(pos[0] + 0.72 * cos_y - 0.38 * sin_y)
        cab_y = float(pos[1] + 0.72 * sin_y + 0.38 * cos_y)
        cab = Marker(header=self.create_header(), ns="fleet_body", id=base_id+6, type=Marker.CUBE, action=Marker.ADD)
        cab.pose.position.x, cab.pose.position.y, cab.pose.position.z = cab_x, cab_y, z_elev + 0.50
        cab.pose.orientation.x, cab.pose.orientation.y, cab.pose.orientation.z, cab.pose.orientation.w = qx, qy, qz, qw
        cab.scale.x, cab.scale.y, cab.scale.z = 0.90, 0.60, 0.62
        cab.color.r, cab.color.g, cab.color.b, cab.color.a = 0.95, 0.95, 0.95, 1.0
        arr.markers.append(cab)

        for w_idx, (wx_rel, wy_rel) in enumerate([(0.88, 0.80), (0.88, -0.80), (-0.88, 0.80), (-0.88, -0.80)]):
            wh = Marker(header=self.create_header(), ns="fleet_wheels", id=base_id+10+w_idx, type=Marker.CYLINDER, action=Marker.ADD)
            wh.pose.position.x = float(pos[0] + wx_rel * cos_y - wy_rel * sin_y)
            wh.pose.position.y = float(pos[1] + wx_rel * sin_y + wy_rel * cos_y)
            wh.pose.position.z = z_elev - 0.12
            wh.pose.orientation.x, wh.pose.orientation.y, wh.pose.orientation.z, wh.pose.orientation.w = qx, qy, qz, qw
            wh.scale.x, wh.scale.y, wh.scale.z = 0.76, 0.76, 0.34
            wh.color.r, wh.color.g, wh.color.b, wh.color.a = 0.08, 0.08, 0.08, 1.0
            arr.markers.append(wh)

        strobe_on = (int(time.time() * 4) % 2 == 0)
        strobe = Marker(header=self.create_header(), ns="fleet_lights", id=base_id+20, type=Marker.CYLINDER, action=Marker.ADD)
        strobe.pose.position.x, strobe.pose.position.y, strobe.pose.position.z = cab_x, cab_y, z_elev + 0.95
        strobe.pose.orientation.w = 1.0
        strobe.scale.x, strobe.scale.y, strobe.scale.z = 0.20, 0.20, 0.16
        strobe.color.r, strobe.color.g, strobe.color.b = 1.0, 0.72, 0.05
        strobe.color.a = 1.0 if strobe_on else 0.35
        arr.markers.append(strobe)

        tag = Marker(header=self.create_header(), ns="fleet_hud", id=base_id+40, type=Marker.TEXT_VIEW_FACING, action=Marker.ADD)
        tag.pose.position.x, tag.pose.position.y, tag.pose.position.z = float(pos[0]), float(pos[1]), z_elev + 2.1
        tag.pose.orientation.w = 1.0
        tag.scale.z = 0.40
        tag.text = f"{name}\n{speed*3.6:.1f} km/h | {payload}"
        tag.color.r, tag.color.g, tag.color.b, tag.color.a = 1.0, 1.0, 1.0, 1.0
        arr.markers.append(tag)

    def tick(self):
        dt = 0.04
        self.sim_time += dt

        if self.sim_time > 110.0 or self.state[4][1] > 65.0:
            self.reset_simulation()

        R_turn = 2.4
        if self.t1_phase == "NORTHBOUND":
            self.turn_signal[1] = "RIGHT"
            if self.state[1][1] >= -4.8:
                self.t1_phase = "TURNING_EAST"
                self.t1_arc = 0.0
            else:
                self.state[1][1] += 0.45 * dt
                self.speeds[1] = 0.45
        elif self.t1_phase == "TURNING_EAST":
            self.t1_arc += (0.35 / R_turn) * dt
            theta = min(math.pi / 2.0, self.t1_arc)
            self.state[1][0] = 2.4 - R_turn * math.cos(theta)
            self.state[1][1] = -4.8 + R_turn * math.sin(theta)
            self.state[1][2] = (math.pi / 2.0) - theta
            self.speeds[1] = 0.35
            if theta >= (math.pi / 2.0):
                self.state[1][1] = -2.4
                self.state[1][2] = 0.0
                self.payload_status[1] = "DUMPING"
                self.t1_phase = "EASTBOUND"
        elif self.t1_phase == "EASTBOUND":
            self.turn_signal[1] = "NONE"
            self.state[1][0] += 0.50 * dt
            self.speeds[1] = 0.50
            if self.state[1][0] >= 5.0:
                self.payload_status[1] = "EMPTY_RETURNING"
                self.t1_cleared = True

        if self.t5_phase == "APPROACHING_NORTH_BAY":
            if not self.t1_cleared:
                if self.state[5][1] > 9.5:
                    self.state[5][1] -= 0.50 * dt
                    self.speeds[5] = 0.50
                    self.braking[5] = False
                else:
                    self.state[5][1] = 9.5
                    self.speeds[5] = 0.0
                    self.braking[5] = True
                    self.payload_status[5] = "LOADING"
                    self.t5_phase = "HOLDING_AT_NORTH_BAY"
            else:
                self.state[5][1] -= 0.55 * dt
                self.speeds[5] = 0.55
        elif self.t5_phase == "HOLDING_AT_NORTH_BAY":
            self.speeds[5] = 0.0
            self.braking[5] = True
            if self.t1_cleared:
                self.payload_status[5] = "HAULING_ORE"
                self.t5_phase = "TRANSITING_SOUTH"
                self.braking[5] = False
        elif self.t5_phase == "TRANSITING_SOUTH":
            self.state[5][1] -= 0.55 * dt
            self.speeds[5] = 0.55
            self.braking[5] = False

        red_cleared = (self.state[5][1] <= -6.0)
        R_blue = 2.2
        if self.t2_phase == "APPROACHING_STOP_BAY":
            if self.state[2][0] > 8.5:
                self.state[2][0] -= 0.48 * dt
                self.speeds[2] = 0.48
            else:
                self.state[2][0] = 8.5
                self.speeds[2] = 0.0
                self.braking[2] = True
                self.payload_status[2] = "LOADING"
                self.t2_phase = "HOLDING_AT_SAFE_BAY"
        elif self.t2_phase == "HOLDING_AT_SAFE_BAY":
            self.speeds[2] = 0.0
            self.braking[2] = True
            if self.t1_cleared and red_cleared:
                self.payload_status[2] = "HAULING_ORE"
                self.t2_phase = "LEAVING_BAY_WESTBOUND"
                self.braking[2] = False
        elif self.t2_phase == "LEAVING_BAY_WESTBOUND":
            self.turn_signal[2] = "RIGHT"
            if self.state[2][0] > 2.2:
                self.state[2][0] -= 0.45 * dt
                self.speeds[2] = 0.45
            else:
                self.t2_phase = "TURN_NORTH"
                self.t2_arc = 0.0
        elif self.t2_phase == "TURN_NORTH":
            self.t2_arc += (0.40 / R_blue) * dt
            phi = min(math.pi / 2.0, self.t2_arc)
            self.state[2][0] = 2.2 - R_blue * math.sin(phi)
            self.state[2][1] = 4.2 - R_blue * math.cos(phi)
            self.state[2][2] = math.pi - phi
            self.speeds[2] = 0.40
            if phi >= (math.pi / 2.0):
                self.state[2][0] = 0.0
                self.state[2][2] = 1.5708
                self.turn_signal[2] = "NONE"
                self.t2_phase = "NORTHBOUND"
        elif self.t2_phase == "NORTHBOUND":
            self.state[2][1] += 0.58 * dt
            self.speeds[2] = 0.58

        self.state[6][1] -= 0.46 * dt
        self.speeds[6] = 0.46
        self.state[3][1] += 0.22 * dt
        self.speeds[3] = 0.22

        purple_cleared = (self.state[4][1] - self.state[6][1]) >= 4.0
        lead_gap = self.state[3][1] - self.state[4][1]

        if self.t4_phase == "APPROACH":
            self.turn_signal[4] = "NONE"
            if lead_gap <= 5.5 and purple_cleared:
                self.t4_phase = "SHIFT_LEFT"
                self.t4_progress = 0.0
            elif lead_gap <= 5.5 and not purple_cleared:
                self.speeds[4] = 0.22
                self.braking[4] = True
                self.state[4][1] += (0.22 * self.cos_t) * dt
            else:
                self.speeds[4] = 0.56
                self.braking[4] = False
                self.state[4][1] += (0.56 * self.cos_t) * dt
        elif self.t4_phase == "SHIFT_LEFT":
            self.turn_signal[4] = "LEFT"
            self.braking[4] = False
            self.t4_progress += dt * 0.65
            tau = min(1.0, self.t4_progress)
            s = 10 * (tau**3) - 15 * (tau**4) + 6 * (tau**5)
            self.state[4][0] = self.lane_right_x + (self.lane_left_x - self.lane_right_x) * s
            self.state[4][1] += (0.50 * self.cos_t) * dt
            self.speeds[4] = 0.50
            if self.t4_progress >= 1.0:
                self.state[4][0] = self.lane_left_x
                self.t4_phase = "PASSING_SPRINT"
        elif self.t4_phase == "PASSING_SPRINT":
            self.turn_signal[4] = "NONE"
            self.speeds[4] = 0.74
            self.state[4][1] += (0.74 * self.cos_t) * dt
            if self.state[4][1] >= self.state[3][1] + 5.5:
                self.t4_phase = "SHIFT_RIGHT"
                self.t4_progress = 0.0
        elif self.t4_phase == "SHIFT_RIGHT":
            self.turn_signal[4] = "RIGHT"
            self.t4_progress += dt * 0.65
            tau = min(1.0, self.t4_progress)
            s = 10 * (tau**3) - 15 * (tau**4) + 6 * (tau**5)
            self.state[4][0] = self.lane_left_x + (self.lane_right_x - self.lane_right_x) * s
            self.state[4][1] += (0.50 * self.cos_t) * dt
            self.speeds[4] = 0.50
            if self.t4_progress >= 1.0:
                self.state[4][0] = self.lane_right_x
                self.t4_phase = "CRUISE_TO_SUMMIT"
        elif self.t4_phase == "CRUISE_TO_SUMMIT":
            self.turn_signal[4] = "NONE"
            self.speeds[4] = 0.58
            self.state[4][1] += (0.58 * self.cos_t) * dt
            if self.state[4][1] >= 18.0:
                self.t4_phase = "SUMMIT_MERGE"
                self.t4_merge_progress = 0.0
        elif self.t4_phase == "SUMMIT_MERGE":
            self.turn_signal[4] = "RIGHT"
            self.t4_merge_progress += dt * 0.20
            tau_m = min(1.0, self.t4_merge_progress)
            s_m = 10 * (tau_m**3) - 15 * (tau_m**4) + 6 * (tau_m**5)
            self.state[4][0] = self.lane_right_x + (self.blue_lane_x - self.lane_right_x) * s_m
            self.state[4][1] += 0.52 * dt
            self.speeds[4] = 0.52
            if self.t4_merge_progress >= 1.0:
                self.state[4][0] = self.blue_lane_x
                self.t4_phase = "HIGHWAY_CRUISE"
        elif self.t4_phase == "HIGHWAY_CRUISE":
            self.turn_signal[4] = "NONE"
            self.state[4][1] += 0.55 * dt
            self.speeds[4] = 0.55

        for t_id in range(1, 7):
            self.publish_sensors_and_transforms(t_id)

        arr = MarkerArray()
        ground = Marker(header=self.create_header(), ns="terrain", id=1, type=Marker.CUBE, action=Marker.ADD)
        ground.pose.position.x, ground.pose.position.y, ground.pose.position.z = 0.0, 15.0, -0.3
        ground.pose.orientation.w = 1.0
        ground.scale.x, ground.scale.y, ground.scale.z = 210.0, 210.0, 0.5
        ground.color.r, ground.color.g, ground.color.b, ground.color.a = 0.45, 0.16, 0.09, 1.0
        arr.markers.append(ground)

        self.add_hematite_highwall(arr, 2, -26.0, 15.0, 2.5, 7.0, 150.0, 5.0)
        self.add_hematite_highwall(arr, 3, -33.0, 15.0, 6.0, 7.0, 150.0, 7.0)
        self.add_hematite_highwall(arr, 4, 40.0, 15.0, 3.5, 9.0, 150.0, 7.0)
        self.add_hematite_highwall(arr, 300, 4.8, 7.8, 1.8, 5.0, 5.0, 3.8)
        self.add_hematite_highwall(arr, 301, 4.8, -7.8, 1.8, 5.0, 5.0, 3.8)

        r1 = Marker(header=self.create_header(), ns="roads", id=10, type=Marker.CUBE, action=Marker.ADD)
        r1.pose.position.x, r1.pose.position.y, r1.pose.position.z = 0.0, 18.0, 0.02
        r1.pose.orientation.w = 1.0
        r1.scale.x, r1.scale.y, r1.scale.z = 5.4, 150.0, 0.04
        r1.color.r, r1.color.g, r1.color.b, r1.color.a = 0.22, 0.08, 0.05, 1.0
        arr.markers.append(r1)

        r1_c = Marker(header=self.create_header(), ns="roads", id=11, type=Marker.CUBE, action=Marker.ADD)
        r1_c.pose.position.x, r1_c.pose.position.y, r1_c.pose.position.z = 0.0, 18.0, 0.05
        r1_c.pose.orientation.w = 1.0
        r1_c.scale.x, r1_c.scale.y, r1_c.scale.z = 0.18, 148.0, 0.02
        r1_c.color.r, r1_c.color.g, r1_c.color.b, r1_c.color.a = 0.95, 0.85, 0.10, 1.0
        arr.markers.append(r1_c)

        r2 = Marker(header=self.create_header(), ns="roads", id=12, type=Marker.CUBE, action=Marker.ADD)
        r2.pose.position.x, r2.pose.position.y, r2.pose.position.z = 24.0, -0.2, 0.02
        r2.pose.orientation.w = 1.0
        r2.scale.x, r2.scale.y, r2.scale.z = 52.0, 9.2, 0.04
        r2.color.r, r2.color.g, r2.color.b, r2.color.a = 0.22, 0.08, 0.05, 1.0
        arr.markers.append(r2)

        sp, cp = math.sin(-self.theta * 0.5), math.cos(-self.theta * 0.5)
        ramp = Marker(header=self.create_header(), ns="roads", id=14, type=Marker.CUBE, action=Marker.ADD)
        ramp.pose.position.x, ramp.pose.position.y, ramp.pose.position.z = -9.5, 0.0, 2.05
        ramp.pose.orientation.x, ramp.pose.orientation.w = sp, cp
        ramp.scale.x, ramp.scale.y, ramp.scale.z = 9.6, 68.0, 0.22
        ramp.color.r, ramp.color.g, ramp.color.b, ramp.color.a = 0.26, 0.10, 0.06, 1.0
        arr.markers.append(ramp)

        ramp_line = Marker(header=self.create_header(), ns="roads", id=15, type=Marker.CUBE, action=Marker.ADD)
        ramp_line.pose.position.x, ramp_line.pose.position.y, ramp_line.pose.position.z = -9.5, 0.0, 2.18
        ramp_line.pose.orientation.x, ramp_line.pose.orientation.w = sp, cp
        ramp_line.scale.x, ramp_line.scale.y, ramp_line.scale.z = 0.20, 66.0, 0.02
        ramp_line.color.r, ramp_line.color.g, ramp_line.color.b, ramp_line.color.a = 0.95, 0.95, 0.95, 1.0
        arr.markers.append(ramp_line)

        apron = Marker(header=self.create_header(), ns="roads", id=16, type=Marker.TRIANGLE_LIST, action=Marker.ADD)
        apron.pose.position.z = 0.04
        apron.pose.orientation.w = 1.0
        apron.scale.x, apron.scale.y, apron.scale.z = 1.0, 1.0, 1.0
        apron.color.r, apron.color.g, apron.color.b, apron.color.a = 0.24, 0.09, 0.05, 1.0
        p_bl = Point(x=-14.3, y=16.0, z=0.55)
        p_br = Point(x=-4.7, y=16.0, z=0.55)
        p_tr = Point(x=2.7, y=38.0, z=0.02)
        p_tl = Point(x=-2.7, y=38.0, z=0.02)
        apron.points = [p_bl, p_br, p_tr, p_bl, p_tr, p_tl]
        arr.markers.append(apron)

        m_stripe = Marker(header=self.create_header(), ns="roads", id=17, type=Marker.LINE_STRIP, action=Marker.ADD)
        m_stripe.pose.orientation.w = 1.0
        m_stripe.scale.x = 0.20
        m_stripe.points = [Point(x=-9.5, y=16.0, z=0.62), Point(x=0.0, y=38.0, z=0.08)]
        m_stripe.color.r, m_stripe.color.g, m_stripe.color.b, m_stripe.color.a = 0.95, 0.85, 0.10, 0.95
        arr.markers.append(m_stripe)

        stop_bay_e = Marker(header=self.create_header(), ns="bays", id=20, type=Marker.CUBE, action=Marker.ADD)
        stop_bay_e.pose.position.x, stop_bay_e.pose.position.y, stop_bay_e.pose.position.z = 8.5, 2.2, 0.03
        stop_bay_e.pose.orientation.w = 1.0
        stop_bay_e.scale.x, stop_bay_e.scale.y, stop_bay_e.scale.z = 5.2, 3.8, 0.04
        stop_bay_e.color.r, stop_bay_e.color.g, stop_bay_e.color.b, stop_bay_e.color.a = 0.18, 0.07, 0.04, 1.0
        arr.markers.append(stop_bay_e)

        for s_idx in range(5):
            h_stripe = Marker(header=self.create_header(), ns="bays", id=21+s_idx, type=Marker.CUBE, action=Marker.ADD)
            h_stripe.pose.position.x = 6.4 + s_idx * 1.0
            h_stripe.pose.position.y = 2.2
            h_stripe.pose.position.z = 0.06
            h_stripe.pose.orientation.z = 0.38268
            h_stripe.pose.orientation.w = 0.92388
            h_stripe.scale.x, h_stripe.scale.y, h_stripe.scale.z = 0.20, 3.2, 0.01
            h_stripe.color.r, h_stripe.color.g, h_stripe.color.b, h_stripe.color.a = 0.95, 0.85, 0.10, 0.85
            arr.markers.append(h_stripe)

        stop_line_e = Marker(header=self.create_header(), ns="bays", id=28, type=Marker.CUBE, action=Marker.ADD)
        stop_line_e.pose.position.x, stop_line_e.pose.position.y, stop_line_e.pose.position.z = 6.2, 2.0, 0.07
        stop_line_e.pose.orientation.w = 1.0
        stop_line_e.scale.x, stop_line_e.scale.y, stop_line_e.scale.z = 0.40, 4.0, 0.02
        stop_line_e.color.r, stop_line_e.color.g, stop_line_e.color.b, stop_line_e.color.a = 1.0, 1.0, 1.0, 1.0
        arr.markers.append(stop_line_e)

        stop_bay_n = Marker(header=self.create_header(), ns="bays", id=120, type=Marker.CUBE, action=Marker.ADD)
        stop_bay_n.pose.position.x, stop_bay_n.pose.position.y, stop_bay_n.pose.position.z = 0.0, 9.5, 0.03
        stop_bay_n.pose.orientation.w = 1.0
        stop_bay_n.scale.x, stop_bay_n.scale.y, stop_bay_n.scale.z = 4.8, 5.0, 0.04
        stop_bay_n.color.r, stop_bay_n.color.g, stop_bay_n.color.b, stop_bay_n.color.a = 0.18, 0.07, 0.04, 1.0
        arr.markers.append(stop_bay_n)

        for s_idx in range(5):
            h_stripe_n = Marker(header=self.create_header(), ns="bays", id=121+s_idx, type=Marker.CUBE, action=Marker.ADD)
            h_stripe_n.pose.position.x = 0.0
            h_stripe_n.pose.position.y = 7.5 + s_idx * 1.0
            h_stripe_n.pose.position.z = 0.06
            h_stripe_n.pose.orientation.z = 0.38268
            h_stripe_n.pose.orientation.w = 0.92388
            h_stripe_n.scale.x, h_stripe_n.scale.y, h_stripe_n.scale.z = 3.6, 0.20, 0.01
            h_stripe_n.color.r, h_stripe_n.color.g, h_stripe_n.color.b, h_stripe_n.color.a = 0.95, 0.85, 0.10, 0.85
            arr.markers.append(h_stripe_n)

        stop_line_n = Marker(header=self.create_header(), ns="bays", id=128, type=Marker.CUBE, action=Marker.ADD)
        stop_line_n.pose.position.x, stop_line_n.pose.position.y, stop_line_n.pose.position.z = 0.0, 7.2, 0.07
        stop_line_n.pose.orientation.w = 1.0
        stop_line_n.scale.x, stop_line_n.scale.y, stop_line_n.scale.z = 4.8, 0.40, 0.02
        stop_line_n.color.r, stop_line_n.color.g, stop_line_n.color.b, stop_line_n.color.a = 1.0, 1.0, 1.0, 1.0
        arr.markers.append(stop_line_n)

        berm_l = Marker(header=self.create_header(), ns="berms", id=50, type=Marker.CUBE, action=Marker.ADD)
        berm_l.pose.position.x, berm_l.pose.position.y, berm_l.pose.position.z = -14.8, 12.0, 2.2
        berm_l.pose.orientation.x, berm_l.pose.orientation.w = sp, cp
        berm_l.scale.x, berm_l.scale.y, berm_l.scale.z = 1.3, 110.0, 2.2
        berm_l.color.r, berm_l.color.g, berm_l.color.b, berm_l.color.a = 0.50, 0.18, 0.10, 1.0
        arr.markers.append(berm_l)

        berm_r = Marker(header=self.create_header(), ns="berms", id=51, type=Marker.CUBE, action=Marker.ADD)
        berm_r.pose.position.x, berm_r.pose.position.y, berm_r.pose.position.z = -4.2, -8.0, 2.2
        berm_r.pose.orientation.x, berm_r.pose.orientation.w = sp, cp
        berm_r.scale.x, berm_r.scale.y, berm_r.scale.z = 1.2, 60.0, 2.2
        berm_r.color.r, berm_r.color.g, berm_r.color.b, berm_r.color.a = 0.50, 0.18, 0.10, 1.0
        arr.markers.append(berm_r)

        tower = Marker(header=self.create_header(), ns="infrastructure", id=70, type=Marker.CYLINDER, action=Marker.ADD)
        tower.pose.position.x, tower.pose.position.y, tower.pose.position.z = -17.5, 12.0, 7.0
        tower.pose.orientation.w = 1.0
        tower.scale.x, tower.scale.y, tower.scale.z = 0.60, 0.60, 14.0
        tower.color.r, tower.color.g, tower.color.b, tower.color.a = 0.82, 0.82, 0.86, 1.0
        arr.markers.append(tower)

        dish = Marker(header=self.create_header(), ns="infrastructure", id=71, type=Marker.CYLINDER, action=Marker.ADD)
        dish.pose.position.x, dish.pose.position.y, dish.pose.position.z = -17.5, 12.4, 13.0
        dish.pose.orientation.w = 1.0
        dish.scale.x, dish.scale.y, dish.scale.z = 1.8, 1.8, 0.22
        dish.color.r, dish.color.g, dish.color.b, dish.color.a = 0.95, 0.95, 0.95, 1.0
        arr.markers.append(dish)

        for t_idx, (tx, ty) in enumerate(self.sal_trees):
            self.add_sal_forest_tree(arr, 200 + t_idx*3, tx, ty)

        for p_idx, p in enumerate(self.dust_particles):
            t_id = p['truck']
            t_speed = self.speeds[t_id]
            t_pos = self.state[t_id]
            t_yaw = t_pos[2]
            t_z = self.calc_z(t_id)

            d_mark = Marker(header=self.create_header(), ns="dust_particles", id=500+p_idx, type=Marker.SPHERE, action=Marker.ADD)
            d_mark.pose.position.x = float(t_pos[0] + (p['dx'] * math.cos(t_yaw) - p['dy'] * math.sin(t_yaw)))
            d_mark.pose.position.y = float(t_pos[1] + (p['dx'] * math.sin(t_yaw) + p['dy'] * math.cos(t_yaw)))
            d_mark.pose.position.z = float(t_z - 0.2 + p['dz'] * min(1.0, t_speed * 1.8))
            d_mark.pose.orientation.w = 1.0
            scale = p['size'] * (0.4 + min(1.0, t_speed * 1.5))
            d_mark.scale.x, d_mark.scale.y, d_mark.scale.z = scale, scale, scale * 0.75
            d_mark.color.r, d_mark.color.g, d_mark.color.b = 0.58, 0.20, 0.12
            d_mark.color.a = p['alpha'] * min(1.0, t_speed * 2.2)
            arr.markers.append(d_mark)

        fleet = [
            (1, 1000, 0.0, "BEML-01 (ORANGE)", (1.0, 0.5, 0.0)),
            (2, 2000, 0.0, "CAT-02 (BLUE)", (0.0, 0.65, 1.0)),
            (3, 3000, -self.theta if self.state[3][1] <= 18.0 else 0.0, "BEML-03 (YELLOW LEAD)", (0.95, 0.85, 0.1)),
            (4, 4000, -self.theta if self.state[4][1] <= 18.0 else 0.0, "CAT-04 (GREEN EGO)", (0.1, 0.85, 0.2)),
            (5, 5000, 0.0, "BEML-05 (RED)", (0.9, 0.15, 0.15)),
            (6, 6000, self.theta, "BEML-06 (PURPLE)", (0.6, 0.2, 0.85))
        ]

        for t_id, base_id, p_pitch, name, col in fleet:
            p = [self.state[t_id][0], self.state[t_id][1], self.calc_z(t_id)]
            self.add_nmdc_detailed_truck(arr, t_id, base_id, p, self.state[t_id][2], p_pitch, name, col)

        self.pub_markers.publish(arr)

    def draw_hauler_model(self, canvas, cx, cy, scale, col, is_oncoming=False):
        bw, bh = int(68 * scale), int(50 * scale)
        cw, ch = int(30 * scale), int(22 * scale)
        top_left = (cx - bw // 2, cy - bh)
        bot_right = (cx + bw // 2, cy)

        cv2.rectangle(canvas, top_left, bot_right, col, -1)
        cv2.rectangle(canvas, top_left, bot_right, (20, 20, 20), 1)
        cab_x = cx - int(10 * scale) if not is_oncoming else cx + int(10 * scale)
        cv2.rectangle(canvas, (cab_x - cw // 2, top_left[1] - ch), (cab_x + cw // 2, top_left[1]), (210, 210, 210), -1)
        cv2.rectangle(canvas, (cab_x - cw // 2, top_left[1] - ch), (cab_x + cw // 2, top_left[1]), (20, 20, 20), 1)

        if int(time.time() * 4) % 2 == 0:
            cv2.circle(canvas, (cab_x, top_left[1] - ch - 3), max(2, int(3 * scale)), (0, 180, 255), -1)

        tw, th = int(12 * scale), int(16 * scale)
        cv2.rectangle(canvas, (top_left[0] - tw // 2, cy - th), (top_left[0] + tw // 2, cy), (15, 15, 15), -1)
        cv2.rectangle(canvas, (bot_right[0] - tw // 2, cy - th), (bot_right[0], cy), (15, 15, 15), -1)

        if is_oncoming:
            cv2.circle(canvas, (cx - int(18 * scale), cy - int(12 * scale)), max(2, int(4 * scale)), (220, 255, 255), -1)
            cv2.circle(canvas, (cx + int(18 * scale), cy - int(12 * scale)), max(2, int(4 * scale)), (220, 255, 255), -1)
        else:
            cv2.rectangle(canvas, (top_left[0] + 2, cy - 6), (top_left[0] + int(8 * scale), cy - 2), (0, 0, 255), -1)
            cv2.rectangle(canvas, (bot_right[0] - int(8 * scale), cy - 6), (bot_right[0] - 2, cy - 2), (0, 0, 255), -1)

    def render_detailed_terrain(self, canvas, ego_id, ego_pitch, ego_y):
        h_offset = int(ego_pitch * 280.0)
        eff_horizon = int(np.clip(self.horizon_y + h_offset, 60, 160))
        
        for y in range(eff_horizon):
            blend = y / max(1.0, float(eff_horizon))
            col = (int(130 - blend * 25), int(150 - blend * 35), int(180 - blend * 45))
            cv2.line(canvas, (0, y), (self.w, y), col, 1)

        pts_left_slope = np.array([[0, eff_horizon], [120, eff_horizon], [50, self.h], [0, self.h]], np.int32)
        cv2.fillPoly(canvas, [pts_left_slope], (75, 42, 105))
        pts_right_slope = np.array([[self.w, eff_horizon], [self.w - 120, eff_horizon], [self.w - 50, self.h], [self.w, self.h]], np.int32)
        cv2.fillPoly(canvas, [pts_right_slope], (75, 42, 105))

        pts_road = np.array([
            [50, self.h], [self.w - 50, self.h],
            [self.w - 120, eff_horizon], [120, eff_horizon]
        ], np.int32)
        cv2.fillPoly(canvas, [pts_road], (45, 40, 50))

        road_center_x = self.w // 2
        lane_dash_y = int((time.time() * 40) % 36)
        cv2.line(canvas, (120, eff_horizon), (50, self.h), (220, 220, 220), 2)
        cv2.line(canvas, (self.w - 120, eff_horizon), (self.w - 50, self.h), (220, 220, 220), 2)

        for y_dash in range(eff_horizon + lane_dash_y, self.h, 36):
            scale_d = (y_dash - eff_horizon) / max(1.0, float(self.h - eff_horizon))
            dash_len = int(18 * scale_d)
            cv2.line(canvas, (road_center_x, y_dash), (road_center_x, min(self.h, y_dash + dash_len)), (255, 235, 90), int(max(1, 3 * scale_d)))

        return eff_horizon

    def render_loop(self):
        fleet_meta = {
            1: ("BEML-01 [ORANGE]", (15, 120, 240)),
            2: ("CAT-02 [BLUE]", (235, 165, 10)),
            3: ("BEML-03 [YELLOW]", (15, 190, 230)),
            4: ("CAT-04 [GREEN EGO]", (30, 210, 50)),
            5: ("BEML-05 [RED]", (30, 30, 230)),
            6: ("BEML-06 [PURPLE]", (180, 50, 150))
        }

        tile_w, tile_h = 320, 220
        grid_positions = {1: (0, 0), 2: (0, 1), 3: (0, 2), 4: (1, 0), 5: (1, 1), 6: (1, 2)}

        while rclpy.ok():
            time.sleep(0.04)
            
            for tid in range(1, 7):
                self.publish_sensors_and_transforms(tid)

            min_fleet_dist = 999.0
            warning_active = False
            for id_a in self.state:
                for id_b in self.state:
                    if id_a >= id_b:
                        continue
                    d = math.hypot(self.state[id_a][0] - self.state[id_b][0], self.state[id_a][1] - self.state[id_b][1])
                    if d < min_fleet_dist:
                        min_fleet_dist = d

            if min_fleet_dist < 6.0:
                warning_active = True

            dashboard = np.zeros((tile_h * 2 + 50, tile_w * 3, 3), dtype=np.uint8)
            dashboard[:40, :] = (20, 22, 28)
            
            if warning_active and (int(time.time() * 6) % 2 == 0):
                dashboard[:40, :] = (0, 0, 180)
                cv2.putText(dashboard, f"WARNING: PROXIMITY ALERT! MIN FLEET DISTANCE: {min_fleet_dist:.1f}m", 
                            (20, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)
            else:
                cv2.putText(dashboard, f"NMDC BAILADILA MINES | FLEET TELEOPERATION | MIN GAP: {min_fleet_dist:.1f}m", 
                            (20, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 220, 255), 2)

            for ego_id, (ego_name, ego_bgr) in fleet_meta.items():
                canvas = np.zeros((self.h, self.w, 3), dtype=np.uint8)
                ego_x, ego_y, ego_yaw = self.state[ego_id]
                ego_z = self.calc_z(ego_id)

                ego_pitch = 0.0
                if ego_id in [3, 4, 6] or abs(ego_x - (-9.5)) < 6.0:
                    ego_pitch = -self.theta if ego_y <= 18.0 else 0.0

                eff_horizon = self.render_detailed_terrain(canvas, ego_id, ego_pitch, ego_y)

                closest_dist = 999.0
                target_seen = False
                visible_targets = []

                for target_id, (target_name, target_bgr) in fleet_meta.items():
                    if target_id == ego_id:
                        continue

                    tgt_x, tgt_y, tgt_yaw = self.state[target_id]
                    tgt_z = self.calc_z(target_id)
                    dx = tgt_x - ego_x
                    dy = tgt_y - ego_y
                    dist = math.hypot(dx, dy)

                    bearing_world = math.atan2(dy, dx)
                    rel_angle = bearing_world - ego_yaw
                    while rel_angle > math.pi: rel_angle -= 2 * math.pi
                    while rel_angle < -math.pi: rel_angle += 2 * math.pi

                    if abs(rel_angle) < math.radians(75) and 1.0 < dist < 45.0:
                        heading_diff = abs(tgt_yaw - ego_yaw)
                        while heading_diff > math.pi: heading_diff -= 2 * math.pi
                        is_oncoming = abs(heading_diff) > math.radians(90)

                        visible_targets.append({
                            'id': target_id, 'name': target_name, 'bgr': target_bgr,
                            'dist': dist, 'rel_angle': rel_angle, 'dz': tgt_z - ego_z,
                            'is_oncoming': is_oncoming
                        })

                visible_targets.sort(key=lambda t: t['dist'], reverse=True)

                for tgt in visible_targets:
                    target_seen = True
                    dist = tgt['dist']
                    rel_angle = tgt['rel_angle']
                    dz = tgt['dz']
                    closest_dist = min(closest_dist, dist)

                    scale = max(0.18, min(2.5, 9.5 / dist))
                    cx = int(self.w // 2 + math.tan(rel_angle) * 190.0)
                    v_shift = int((dz / dist) * 140.0)
                    cy = int(eff_horizon + int(30.0 * scale) - v_shift)

                    if 15 < cx < self.w - 15 and 20 < cy < self.h:
                        self.draw_hauler_model(canvas, cx, cy, scale, tgt['bgr'], tgt['is_oncoming'])

                        if dist < 12.0:
                            overlay = canvas.copy()
                            dust_r = int(48 * scale)
                            cv2.circle(overlay, (cx, cy - int(8 * scale)), dust_r, (45, 30, 20), -1)
                            alpha = min(0.60, max(0.15, 1.0 - (dist / 12.0)))
                            cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)

                        bw, bh = int(74 * scale), int(60 * scale)
                        box_col = (0, 0, 255) if tgt['is_oncoming'] and dist < 12.0 else (0, 255, 255)
                        cv2.rectangle(canvas, (cx - bw // 2, cy - bh), (cx + bw // 2, cy), box_col, 1)
                        label = f"{tgt['name'].split()[0]}: {dist:.1f}m"
                        cv2.putText(canvas, label, (cx - bw // 2, cy - bh - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, box_col, 1)

                cv2.putText(canvas, f"POV: {ego_name}", (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)
                
                payload_str = self.payload_status[ego_id]
                dist_label = f"RANGE: {closest_dist:.1f}m" if target_seen else "CLEAR: >45m"
                col_hud = (0, 230, 255) if (not target_seen or closest_dist > 6.0) else (0, 0, 255)
                speed_kmh = self.speeds[ego_id] * 3.6
                
                cv2.putText(canvas, f"{dist_label} | {speed_kmh:.1f} km/h", (8, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col_hud, 1)
                cv2.putText(canvas, f"PAYLOAD: {payload_str}", (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 255, 180), 1)

                msg = self.bridge.cv2_to_imgmsg(canvas, encoding="bgr8")
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = f"dumper_{ego_id}_cam"
                self.pub_cams[ego_id].publish(msg)

                row, col = grid_positions[ego_id]
                y_start = 45 + row * tile_h
                y_end = y_start + tile_h
                x_start = col * tile_w
                x_end = x_start + tile_w
                dashboard[y_start:y_end, x_start:x_end] = canvas

            self.pub_cockpit.publish(self.bridge.cv2_to_imgmsg(dashboard, encoding="bgr8"))

def main(args=None):
    rclpy.init(args=args)
    node = BailadilaProductionEngine()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
