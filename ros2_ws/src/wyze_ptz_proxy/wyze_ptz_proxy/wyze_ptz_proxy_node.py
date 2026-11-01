#!/usr/bin/env python3
"""
ROS2 Proxy Node for Wyze Cam Pan v3 (via docker-wyze-bridge).

Subscribes to:
  1. `/wyze/cmd_ptz` (std_msgs/String JSON):
     e.g. {"pan": 180.0, "tilt": 0.0, "mode": "degree"}
     or   {"action": "left"}
  2. `/cmd_vel` (geometry_msgs/Twist):
     Roomba互換インターフェース！
     angular.z (rad/s) -> Pan 回転角度/アクションに変換
     linear.x  (m/s)   -> Tilt 上下角度/アクションに変換

Publishes:
  - `/wyze/ptz_state` (std_msgs/String JSON): 現在の推定Pan/Tilt状態と最終送信ステータス
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

        self.bridge_host = os.environ.get('WYZE_BRIDGE_HOST', 'http://localhost:5000').rstrip('/')
        self.cam_name = os.environ.get('WYZE_CAM_NAME', 'wyze-cam-pan')
        self.ptz_method = os.environ.get('WYZE_PTZ_METHOD', 'degree')  # 'degree' or 'action'
        self.dry_run = os.environ.get('WYZE_DRY_RUN', 'false').lower() in ('1', 'true', 'yes')

        # 現在のカメラ角度状態 (Pan: -180〜+180 deg, Tilt: -45〜+90 deg)
        self.current_pan = 0.0
        self.current_tilt = 0.0
        self.last_cmd_time = 0.0
        self.min_cmd_interval = 0.40  # WyzeクラウドAPI負荷軽減のための最小送信間隔(秒)
        self.deadband_deg = 2.5       # 微小揺れ無視閾値(度)

        # Subscribers
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

        # Publisher
        self.pub_state = self.create_publisher(String, '/wyze/ptz_state', 10)

        self.get_logger().info(
            f'WyzePTZProxyNode started: bridge={self.bridge_host}, '
            f'cam={self.cam_name}, method={self.ptz_method}, dry_run={self.dry_run}'
        )

    def on_cmd_ptz(self, msg: String):
        """JSON形式のPTZ指令を処理"""
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
        """ルンバ互換 `/cmd_vel` (geometry_msgs/Twist) からPTZへのプロキシ変換"""
        # angular.z (左回転が正) -> pan差分 (deg), linear.x (前進が正) -> tilt差分 (deg)
        delta_pan = float(msg.angular.z) * 30.0
        delta_tilt = float(msg.linear.x) * 30.0

        if abs(delta_pan) < 1e-2 and abs(delta_tilt) < 1e-2:
            return

        target_pan = self.current_pan + delta_pan
        target_tilt = self.current_tilt + delta_tilt
        self.move_to_angles(target_pan, target_tilt)

    def move_to_angles(self, pan_deg: float, tilt_deg: float):
        """目標Pan/Tilt角度に制限・デッドバンドを適用してdocker-wyze-bridgeへ送信"""
        now = time.time()
        if now - self.last_cmd_time < self.min_cmd_interval:
            return

        # 角度クランプ (Pan: -180..180, Tilt: -45..90)
        pan_deg = max(-180.0, min(180.0, pan_deg))
        tilt_deg = max(-45.0, min(90.0, tilt_deg))

        err_pan = pan_deg - self.current_pan
        err_tilt = tilt_deg - self.current_tilt
        if abs(err_pan) < self.deadband_deg and abs(err_tilt) < self.deadband_deg:
            return

        self.last_cmd_time = now

        if self.ptz_method == 'action':
            # ステップ移動モード: 誤差の大きい軸から優先してアクション発行
            if abs(err_pan) >= abs(err_tilt):
                action = 'right' if err_pan > 0 else 'left'
                self.send_rotary_action(action)
                self.current_pan += 10.0 if err_pan > 0 else -10.0
            else:
                action = 'up' if err_tilt > 0 else 'down'
                self.send_rotary_action(action)
                self.current_tilt += 8.0 if err_tilt > 0 else -8.0
        else:
            # 絶対角度指定モード (docker-wyze-bridge: /api/<cam>/rotary_degree/<pan>,<tilt>)
            # Wyze内部表現(0..360, 0..90)に合わせてシフト
            wyze_pan = int(round((pan_deg + 180.0) % 360.0))
            wyze_tilt = int(round(max(0.0, min(90.0, tilt_deg + 45.0))))
            ok = self.send_rotary_degree(wyze_pan, wyze_tilt)
            if ok:
                self.current_pan = pan_deg
                self.current_tilt = tilt_deg

        self.publish_state()

    def send_rotary_degree(self, wyze_pan: int, wyze_tilt: int) -> bool:
        url = f'{self.bridge_host}/api/{self.cam_name}/rotary_degree/{wyze_pan},{wyze_tilt}'
        if self.dry_run:
            self.get_logger().info(f'[DRY-RUN] POST {url}')
            return True
        try:
            resp = requests.post(url, timeout=2.5)
            if resp.status_code >= 400:
                resp = requests.get(url, timeout=2.5)
            self.get_logger().info(f'Sent rotary_degree({wyze_pan}, {wyze_tilt}) -> {resp.status_code}')
            return resp.ok
        except Exception as e:
            self.get_logger().warning(f'Bridge request failed ({url}): {e}')
            return False

    def send_rotary_action(self, action: str) -> bool:
        url = f'{self.bridge_host}/api/{self.cam_name}/rotary_action/{action}'
        if self.dry_run:
            self.get_logger().info(f'[DRY-RUN] POST {url}')
            return True
        try:
            resp = requests.post(url, timeout=2.5)
            if resp.status_code >= 400:
                resp = requests.get(url, timeout=2.5)
            self.get_logger().info(f'Sent rotary_action({action}) -> {resp.status_code}')
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
