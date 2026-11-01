#!/usr/bin/env python3
"""
Wyze Cam Pan v3 実機PTZ疎通確認スクリプト。
docker-wyze-bridge が起動している状態で、コマンドラインまたはキー操作で
rotary_degree / rotary_action が実機で動くかを即座に検証できます。
"""

import argparse
import time
from src.wyze_bridge_client import WyzeBridgeClient


def main():
    parser = argparse.ArgumentParser(description="Verify Wyze Cam Pan v3 PTZ control via docker-wyze-bridge")
    parser.add_argument("--host", type=str, default="http://localhost:5000")
    parser.add_argument("--cam", type=str, default="wyze-cam-pan")
    parser.add_argument("--method", type=str, default="degree", choices=["degree", "action"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    client = WyzeBridgeClient(
        bridge_host=args.host,
        cam_name=args.cam,
        method=args.method,
        dry_run=args.dry_run,
        min_interval=0.2,
    )

    print(f"Testing Wyze Cam Pan v3 ({args.cam}) at {args.host} [method={args.method}]...")
    test_sequence = [
        ("Left (-45 deg)", -45.0, 0.0),
        ("Right (+45 deg)", 45.0, 0.0),
        ("Up (+25 deg)", 0.0, 25.0),
        ("Center (0, 0)", 0.0, 0.0),
    ]
    for label, pan, tilt in test_sequence:
        print(f" -> Moving to {label}: pan={pan}, tilt={tilt}")
        ok = client.set_angles(pan, tilt, force=True)
        print(f"    Result: {'OK' if ok else 'FAILED'}")
        time.sleep(2.0)


if __name__ == "__main__":
    main()
