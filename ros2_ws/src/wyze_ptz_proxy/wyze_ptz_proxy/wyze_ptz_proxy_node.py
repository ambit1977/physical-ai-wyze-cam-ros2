#!/usr/bin/env python3
"""
ROS2 Proxy Node for Wyze Cam Pan v3 (`HL_PAN3`) via docker-wyze-bridge.

Subscribes to:
  1. `/wyze/cmd_ptz` (std_msgs/String JSON):
     e.g. {"pan": 0.0, "tilt": 0.0, "mode": "degree"}
     or   {"action": "left"}
  2. `/cmd_vel` (geometry_msgs/Twist):
     Roomba互換インターフェース！
     angular.z (rad/s) -> Pan 回転角度に変換
     linear.x  (m/s)   -> Tilt 上下角度に変換

Publishes:
  - `/wyze/ptz_state` (std_msgs/String JSON): 現在のPan/Tilt状態と最終送信ステータス
"""

import json
import os
import time
import requests
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist


class WyzePTZProxyNode(Node):
    def __init__(self):
        super().__init__('wyze_ptz_proxy_node')

        self.bridge_host = os.environ.get('WYZE_BRIDGE_HOST', 'http://wyze-bridge:5000').rstrip('/')
        self.cam_name = os.environ.get('WYZE_CAM_NAME', 'wyze')
        self.ptz_method = os.environ.get('WYZE_PTZ_METHOD', 'degree')  # 'degree' (ptz_position) or 'action'
        self.dry_run = os.environ.get('WYZE_DRY_RUN', 'false').lower() in ('1', 'true', 'yes')

        self.current_pan = 0.0
        self.current_tilt = 0.0
        self.last_cmd_time = 0.0
        self.min_cmd_interval = 0.35
        self.deadband_deg = 2.5

        self.sub_ptz = self.create_subscription(
            String,
            '/wyze/cmd_ptz',
            self.on_cmd_ptz,
            10,
        )
        self.sub_twist = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.on_cmd_vel,
            10,
        )
        self.pub_state = self.create_publisher(String, '/wyze/ptz_state', 10)

        self.get_logger().info(
            f'WyzePTZProxyNode started: bridge={self.bridge_host}, '
            f'cam={self.cam_name}, method={self.ptz_method}, dry_run={self.dry_run}'
        )

    def on_cmd_ptz(self, msg: String):
        try:
            payload = json.loads(msg.data)
        except Exception as e:
            self.get_logger().warning(f'Invalid JSON on /wyze/cmd_ptz: {e}')
            return

        if 'action' in payload and payload['action']:
            action = str(payload['action']).lower()
            self.send_rotary_action(action)
            return

        pan = float(payload.get('pan', self.current_pan))
        tilt = float(payload.get('tilt', self.current_tilt))
        mode = payload.get('mode', self.ptz_method)

        if mode == 'relative':
            target_pan = self.current_pan + pan
            target_tilt = self.current_tilt + tilt
        else:
            target_pan = pan
            target_tilt = tilt

        self.move_to_angles(target_pan, target_tilt)

    def on_cmd_vel(self, msg: Twist):
        delta_pan = float(msg.angular.z) * 30.0
        delta_tilt = float(msg.linear.x) * 30.0

        if abs(delta_pan) < 1e-2 and abs(delta_tilt) < 1e-2:
            return

        target_pan = self.current_pan + delta_pan
        target_tilt = self.current_tilt + delta_tilt
        self.move_to_angles(target_pan, target_tilt)

    def move_to_angles(self, pan_deg: float, tilt_deg: float):
        now = time.time()
        if now - self.last_cmd_time < self.min_cmd_interval:
            return

        pan_deg = max(-170.0, min(170.0, pan_deg))
        tilt_deg = max(-45.0, min(90.0, tilt_deg))

        err_pan = pan_deg - self.current_pan
        err_tilt = tilt_deg - self.current_tilt
        if abs(err_pan) < self.deadband_deg and abs(err_tilt) < self.deadband_deg:
            return

        self.last_cmd_time = now

        if self.ptz_method == 'action':
            h_code = 2 if err_pan >= self.deadband_deg else (1 if err_pan <= -self.deadband_deg else 0)
            v_code = 1 if err_tilt >= self.deadband_deg else (2 if err_tilt <= -self.deadband_deg else 0)
            ok = self.send_rotary_action(f'{h_code},{v_code}')
            if ok:
                self.current_pan = pan_deg
                self.current_tilt = tilt_deg
        else:
            # Wyze Cam Pan v3 (HL_PAN3): /api/<cam>/ptz_position/<vertical>,<horizontal>
            wyze_horiz = int(round(max(0.0, min(350.0, 180.0 + pan_deg))))
            wyze_vert = int(round(max(0.0, min(180.0, 90.0 + tilt_deg))))
            ok = self.send_ptz_position(wyze_vert, wyze_horiz)
            if ok:
                self.current_pan = pan_deg
                self.current_tilt = tilt_deg

        self.publish_state()

    def send_ptz_position(self, wyze_vert: int, wyze_horiz: int) -> bool:
        url = f'{self.bridge_host}/api/{self.cam_name}/ptz_position/{wyze_vert},{wyze_horiz}'
        if self.dry_run:
            self.get_logger().info(f'[DRY-RUN] GET {url}')
            return True
        try:
            resp = requests.get(url, timeout=2.5)
            self.get_logger().info(f'Sent ptz_position(vert={wyze_vert}, horiz={wyze_horiz}) -> {resp.status_code}')
            return resp.ok
        except Exception as e:
            self.get_logger().warning(f'Bridge request failed ({url}): {e}')
            return False

    def send_rotary_action(self, action: str) -> bool:
        action_map = {
            'left': '1,0',
            'right': '2,0',
            'up': '0,1',
            'down': '0,2',
            'stop': '0,0',
        }
        payload = action_map.get(action, action)
        url = f'{self.bridge_host}/api/{self.cam_name}/rotary_action/{payload}'
        if self.dry_run:
            self.get_logger().info(f'[DRY-RUN] GET {url}')
            return True
        try:
            resp = requests.get(url, timeout=2.5)
            self.get_logger().info(f'Sent rotary_action({payload}) -> {resp.status_code}')
            return resp.ok
        except Exception as e:
            self.get_logger().warning(f'Bridge action failed ({url}): {e}')
            return False

    def publish_state(self):
        msg = String()
        msg.data = json.dumps({
            'pan': round(self.current_pan, 2),
            'tilt': round(self.current_tilt, 2),
            'timestamp': time.time(),
            'cam_name': self.cam_name,
        })
        self.pub_state.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = WyzePTZProxyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down WyzePTZProxyNode')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
