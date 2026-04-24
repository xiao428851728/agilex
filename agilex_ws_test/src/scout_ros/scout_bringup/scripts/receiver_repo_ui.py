#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import socket
import struct
import threading
import time

import cv2
import numpy as np
import rospy
import tf
from actionlib_msgs.msg import GoalID
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import Image
from std_msgs.msg import Empty

# =========================
# 配置区域
# =========================
WINDOWS_IP = "192.168.31.142"
WINDOWS_PORT = 8888

MAP_FRAME = "/map"
BASE_FRAME_PRIMARY = "/base_link"          # 与原仓库 receiver.py 一致
BASE_FRAME_FALLBACK = "/base_footprint"    # 仅做兼容兜底
SEND_HZ = 10

ROTATE_SPEED = 0.3
MAX_LINEAR = 0.35
MAX_ANGULAR = 1.00

MANUAL_MODE_ID = 99
MANUAL_NONE = 0
MANUAL_DRIVE = 1
MANUAL_STOP_HOLD = 2

bridge = CvBridge()
conn = None
# global publishers/listener
cmd_pub = None
goal_pub = None
cancel_pub = None
reset_pub = None
tf_listener = None

latest_rgb = None
latest_depth = None
latest_map_bgr = None
latest_map_meta = None
map_lock = threading.Lock()


def recv_all(sock, size):
    data = b""
    while len(data) < size:
        try:
            chunk = sock.recv(size - len(data))
            if not chunk:
                return None
            data += chunk
        except socket.timeout:
            return None
        except Exception:
            return None
    return data


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def connect_to_windows():
    global conn
    while not rospy.is_shutdown():
        try:
            rospy.loginfo(f"Connecting to Windows {WINDOWS_IP}:{WINDOWS_PORT} ...")
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10)
            sock.connect((WINDOWS_IP, WINDOWS_PORT))
            conn = sock
            conn.settimeout(None)
            rospy.loginfo("Connected to Windows successfully!")
            return
        except Exception as e:
            rospy.logerr(f"Connection failed: {e}. Retrying in 2s...")
            time.sleep(2)


def rgb_cb(msg):
    global latest_rgb
    try:
        latest_rgb = bridge.imgmsg_to_cv2(msg, "bgr8")
    except Exception:
        pass


def depth_cb(msg):
    global latest_depth
    try:
        latest_depth = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
    except Exception:
        pass


def map_cb(msg):
    global latest_map_bgr, latest_map_meta

    h = msg.info.height
    w = msg.info.width
    if h <= 0 or w <= 0:
        return

    grid = np.array(msg.data, dtype=np.int16).reshape(h, w)
    vis = np.full((h, w, 3), 160, dtype=np.uint8)
    vis[grid == 0] = (255, 255, 255)
    vis[grid >= 50] = (0, 0, 0)
    vis = cv2.flip(vis, 0)

    meta = {
        "resolution": float(msg.info.resolution),
        "origin_x": float(msg.info.origin.position.x),
        "origin_y": float(msg.info.origin.position.y),
        "width": int(w),
        "height": int(h),
    }

    with map_lock:
        latest_map_bgr = vis
        latest_map_meta = meta


def overlay_robot_on_map(map_bgr, meta, x, y, yaw):
    out = map_bgr.copy()
    res = meta["resolution"]
    ox = meta["origin_x"]
    oy = meta["origin_y"]
    w = meta["width"]
    h = meta["height"]

    if res <= 1e-9:
        return out

    px = int((x - ox) / res)
    py = int((y - oy) / res)
    py = h - 1 - py

    if 0 <= px < w and 0 <= py < h:
        cv2.circle(out, (px, py), 4, (0, 0, 255), -1)
        arrow_len = 14
        ex = int(px + arrow_len * math.cos(yaw))
        ey = int(py - arrow_len * math.sin(yaw))
        cv2.arrowedLine(out, (px, py), (ex, ey), (0, 0, 255), 2, tipLength=0.3)

    return out


def lookup_pose():
    last_err = None
    for base_frame in (BASE_FRAME_PRIMARY, BASE_FRAME_FALLBACK):
        try:
            tf_listener.waitForTransform(MAP_FRAME, base_frame, rospy.Time(), rospy.Duration(0.5))
            (trans, rot) = tf_listener.lookupTransform(MAP_FRAME, base_frame, rospy.Time(0))
            (_, _, yaw) = tf.transformations.euler_from_quaternion(rot)
            return float(trans[0]), float(trans[1]), float(yaw)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            last_err = e
    raise last_err if last_err is not None else RuntimeError("TF lookup failed")


def cancel_auto_nav():
    try:
        cancel_pub.publish(GoalID())
    except Exception:
        pass
    try:
        reset_pub.publish(Empty())
    except Exception:
        pass


def publish_auto_goal(target_x, target_y):
    goal_msg = PoseStamped()
    goal_msg.header.stamp = rospy.Time.now()
    goal_msg.header.frame_id = "map"
    goal_msg.pose.position.x = float(target_x)
    goal_msg.pose.position.y = float(target_y)
    goal_msg.pose.position.z = 0.0
    goal_msg.pose.orientation.w = 1.0
    goal_pub.publish(goal_msg)


def main_sync_loop():
    global conn, latest_rgb, latest_depth

    rate = rospy.Rate(SEND_HZ)
    rospy.loginfo("Starting repo-consistent navigation loop with richer UI...")

    # 原仓库的初始化旋转逻辑
    initial_spin_done = False
    total_yaw_rotated = 0.0
    last_yaw_capture = None
    prev_manual_state = MANUAL_NONE

    rospy.sleep(1.0)

    while not rospy.is_shutdown():
        if conn is None:
            connect_to_windows()
            continue

        if latest_rgb is None or latest_depth is None:
            rospy.logwarn_throttle(2, "Waiting for camera data...")
            rate.sleep()
            continue

        try:
            current_x, current_y, current_yaw = lookup_pose()
        except Exception as e:
            rospy.logwarn_throttle(1, f"TF lookup failed: {e}. Waiting for TF tree...")
            rate.sleep()
            continue

        try:
            # =========================
            # A. 发给 Windows：RGB + Depth + Pose + SLAM图 + 元数据
            # =========================
            ok, rgb_encoded = cv2.imencode(".jpg", latest_rgb)
            if not ok:
                raise RuntimeError("Failed to encode RGB image")
            rgb_bytes = rgb_encoded.tobytes()

            if latest_depth.dtype == np.uint16:
                depth_mm = latest_depth
            else:
                depth_temp = latest_depth * 1000.0
                depth_temp = np.nan_to_num(depth_temp, nan=0.0, posinf=65535.0, neginf=0.0)
                depth_mm = np.clip(depth_temp, 0, 65535).astype(np.uint16)

            ok, depth_encoded = cv2.imencode(".png", depth_mm)
            if not ok:
                raise RuntimeError("Failed to encode depth image")
            depth_bytes = depth_encoded.tobytes()

            map_bytes = b""
            slam_meta = {}
            with map_lock:
                if latest_map_bgr is not None and latest_map_meta is not None:
                    map_vis = overlay_robot_on_map(latest_map_bgr, latest_map_meta, current_x, current_y, current_yaw)
                    ok, map_encoded = cv2.imencode(".png", map_vis)
                    if ok:
                        map_bytes = map_encoded.tobytes()
                    slam_meta = latest_map_meta.copy()
            slam_meta_json = json.dumps(slam_meta, ensure_ascii=False).encode("utf-8") if slam_meta else b"{}"

            conn.sendall(struct.pack("I", len(rgb_bytes)))
            conn.sendall(rgb_bytes)
            conn.sendall(struct.pack("I", len(depth_bytes)))
            conn.sendall(depth_bytes)
            conn.sendall(struct.pack("fff", current_x, current_y, current_yaw))
            conn.sendall(struct.pack("I", len(map_bytes)))
            if map_bytes:
                conn.sendall(map_bytes)
            conn.sendall(struct.pack("I", len(slam_meta_json)))
            if slam_meta_json:
                conn.sendall(slam_meta_json)

            # =========================
            # B. 收 Windows 融合回包
            # 协议: [goal_packet 10B][manual_packet 10B]
            # goal_packet   = <BBff> = [action_id][has_goal][goal_x][goal_y]
            # manual_packet = <BBff> = [mode_id][manual_state][linear_x][angular_z]
            # =========================
            combo = recv_all(conn, 20)
            if not combo:
                raise BrokenPipeError("Windows bridge closed")

            action_id, has_goal, target_x, target_y, mode_id, manual_state, linear_x, angular_z = struct.unpack("<BBffBBff", combo)
            if mode_id != MANUAL_MODE_ID:
                manual_state = MANUAL_NONE

            # =========================
            # C. 手动接管层（仅覆盖控制，不改自动导航主干）
            # =========================
            if manual_state in (MANUAL_DRIVE, MANUAL_STOP_HOLD):
                if prev_manual_state == MANUAL_NONE:
                    rospy.loginfo("Manual override entered. Canceling move_base and clearing cached goal.")
                    cancel_auto_nav()

                twist = Twist()
                if manual_state == MANUAL_DRIVE:
                    twist.linear.x = clamp(linear_x, -MAX_LINEAR, MAX_LINEAR)
                    twist.angular.z = clamp(angular_z, -MAX_ANGULAR, MAX_ANGULAR)
                else:
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0
                cmd_pub.publish(twist)
                prev_manual_state = manual_state
                rate.sleep()
                continue

            if prev_manual_state in (MANUAL_DRIVE, MANUAL_STOP_HOLD):
                cmd_pub.publish(Twist())
                cancel_auto_nav()
                rospy.loginfo("Manual override released. Returning to repo auto-navigation chain.")
            prev_manual_state = MANUAL_NONE

            # =========================
            # D. 原仓库自动导航逻辑
            # =========================
            if not initial_spin_done:
                if last_yaw_capture is None:
                    last_yaw_capture = current_yaw

                delta_yaw = current_yaw - last_yaw_capture
                if delta_yaw < -math.pi:
                    delta_yaw += 2 * math.pi
                elif delta_yaw > math.pi:
                    delta_yaw -= 2 * math.pi

                total_yaw_rotated += abs(delta_yaw)
                last_yaw_capture = current_yaw

                if total_yaw_rotated < 6.3:
                    twist = Twist()
                    twist.angular.z = ROTATE_SPEED
                    cmd_pub.publish(twist)
                    rospy.loginfo_throttle(1.0, f"Initial Scanning... Rotated: {math.degrees(total_yaw_rotated):.1f}/360 deg")
                    rate.sleep()
                    continue
                else:
                    initial_spin_done = True
                    cmd_pub.publish(Twist())
                    rospy.loginfo("Initial Scan Completed! Switching to Auto Navigation.")

            if has_goal:
                publish_auto_goal(target_x, target_y)
            else:
                twist = Twist()
                if action_id == 2:
                    twist.angular.z = ROTATE_SPEED
                    cmd_pub.publish(twist)
                    rospy.loginfo_throttle(1.0, "Searching for Frontier (Spinning)...")
                else:
                    cmd_pub.publish(Twist())

        except socket.timeout:
            rospy.logerr("Timeout waiting for Windows packet")
            try:
                cmd_pub.publish(Twist())
            except Exception:
                pass
            try:
                cancel_auto_nav()
            except Exception:
                pass
            if conn:
                conn.close()
            conn = None
        except BrokenPipeError:
            rospy.logerr("Broken pipe to Windows bridge")
            try:
                cmd_pub.publish(Twist())
            except Exception:
                pass
            try:
                cancel_auto_nav()
            except Exception:
                pass
            if conn:
                conn.close()
            conn = None
        except Exception as e:
            rospy.logerr(f"Main loop error: {e}")
            try:
                cmd_pub.publish(Twist())
            except Exception:
                pass
            if conn:
                conn.close()
            conn = None
            time.sleep(1)

        rate.sleep()


if __name__ == "__main__":
    rospy.init_node("robot_vlfm_repo_ui_agent", anonymous=True)
    tf_listener = tf.TransformListener()

    rospy.Subscriber("/camera/color/image_raw", Image, rgb_cb)
    rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, depth_cb)
    rospy.Subscriber("/map", OccupancyGrid, map_cb, queue_size=1)

    cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
    goal_pub = rospy.Publisher("/target_goal", PoseStamped, queue_size=1)
    cancel_pub = rospy.Publisher("/move_base/cancel", GoalID, queue_size=1)
    reset_pub = rospy.Publisher("/auto_nav/reset_goal_cache", Empty, queue_size=1)

    try:
        main_sync_loop()
    except rospy.ROSInterruptException:
        pass
