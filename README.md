# Wyze Cam Pan v3 × ROS2 × Meta Quest Physical AI 制御システム

東京大学松尾研究室 Physical AI 講座 2026（応用編1）最終課題リポジトリです。
市販のスマートPTZカメラ **Wyze Cam Pan v3 (`HL_PAN3`)** を 2自由度（Pan/Tilt）のアクティブビジョン・ロボットとして扱い、**ROS2 Humble（`/wyze/cmd_ptz` およびルンバ Create 3 互換 `/cmd_vel`）** を介して実機制御するシステムを実装しています。

📄 **詳細な提出レポート（概要・目的・工夫点・今後の発展課題）**: **[REPORT.md](./REPORT.md)**

---

## 主な実装機能

1. **【メイン構成】Meta Quest HMD / 6DoF コントローラ × ROS2 実機テレオペレーション (`src/node_quest_wyze_teleop.py`)**
   - PCモニターの前で顔を大きく上下左右に向けると画面が見えなくなり、HMDを被ると顔が隠れて顔検出ができなくなるというジレンマを解決するため、**Meta Quest HMD の視界内に 16:9 広角ライブ映像を表示しながら、HMDの角度（Yaw/Pitch）や 6DoF コントローラ姿勢で実機カメラを ROS2 制御**します。
   - **Quest Browser (WebXR `immersive-ar` MRパススルー / `immersive-vr`)** と **Open Teach APK (ZeroMQ `TCP 8087 / 10505` + `UDP 8125`)** の2方式に同時対応しています。
2. **多軸ロボットアーム動作（6DoF/7DoF）→ カメラPT軸・視線ベクトル変換 (`src/models/arm_to_ptz_retargeter.py`)**
   - 多軸アームの手先変位・回転 $[\Delta x, \Delta y, \Delta z, \Delta \text{roll}, \Delta \text{pitch}, \Delta \text{yaw}, \text{gripper}]^\top$ から 3次元単位視線ベクトル $\boldsymbol{v}_{\text{gaze}} = (g_x, g_y, g_z)$ を算出し、球面ヤコビアン投影によってカメラの Pan/Tilt 軸と注視ズームへ変換します。
3. **VLM / Open-Vocabulary 視覚認識による自律探索と Step-and-Settle 視覚サーボ捕捉 (`src/models/vlm_target_detector.py`)**
   - **`YOLO-World` (CLIP Open-Vocabulary) + `YOLOv8n` (COCO)**、および Hugging Face **`OWLv2` (`google/owlv2-base-patch16-ensemble`) / `Moondream2`** をサポート。
   - **視界に対象物がある場合**: RTSP映像遅延（約0.6〜0.8秒）による積分ワインドアップ暴走を防ぐ **Step-and-Settle（0.72秒整定待ち）制御** により、対象物（カップ、ペットボトル等）を画面中心へ滑らかに捕捉・ロックオンします。
   - **視界に対象物がない場合**: 人間が周囲を見回すように 3D 作業空間の9つのウェイポイントを段階的に巡回（各視点で1.45秒静止して観察）し、対象物を発見次第すぐさま中心捕捉へ遷移します。
4. **顔角度ミラーリング追従 (`src/node_face_mirror.py`)**
   - MediaPipe 3D 顔姿勢推定と PyTorch MLP (`FaceToPTZNet`) を用い、カメラ間の位置オフセットを補正して操作者の顔の角度にカメラを同期させます。
5. **発展機能（Few-Shot Visual FT / Online RL Bandit / Quest 模倣学習 BC）**
   - ドメインギャップのある未知物体を1クリック（`[f]` キー）で視覚プロトタイプ登録する `VisualPrototypeFT`、検出ヘッド重みをオンライン調整する `OnlineRLBanditHeadSelector`、Quest での探索デモ（`[d]` キー記録、`[l]` キー学習）からニューラル方策を学習する `QuestDemoBCManager` (`src/models/bc_search_policy.py`) を実装しています。

---

## システム構成図

```
[Meta Quest HMD / 6DoF Controller]          [VLM / Open-Vocab 認識エンジン]
 (WebXR HTTPS 8443 / OpenTeach ZMQ 8087)     (YOLO-World + YOLOv8 / OWLv2 / Moondream2)
                 │                                           │
                 ▼                                           ▼
        頭部角度 (Yaw/Pitch)                  対象物の2D位置偏差 (cx_norm, cy_norm)
        & 6DoF 手先変位 (dx,dy,dz)            & 視界外時の3D探索ウェイポイント生成
                 │                                           │
                 └──────────────► [ArmToPTZGazeRetargeter] ◄─┘
                                 ・多軸アーム動作 → 3D視線ベクトル v_gaze 変換
                                 ・Step-and-Settle 暴走防止視覚サーボ制御
                                             │
                                             ▼
                                  [ROS2WyzeBus (roslibpy)]
                                 Topic: /wyze/cmd_ptz (std_msgs/String JSON)
                                 Topic: /cmd_vel (geometry_msgs/Twist ルンバ互換)
                                             │ WebSocket (ws://localhost:9092)
                                             ▼
                        [ROS2 Humble Container: wyze_ptz_proxy_node]
                                             │ HTTP REST API
                                             ▼
                           [docker-wyze-bridge (Port 5050 / RTSP 8554)]
                           GET /api/wyze/ptz_position/<vert>,<horiz>
                                             │
                                             ▼
                              [Wyze Cam Pan v3 実機 (HL_PAN3)]
```

---

## セットアップと実行手順

### 1. 環境変数 `.env` の設定と Docker コンテナ（`wyze-bridge` + `ros2-wyze-proxy`）の起動

```bash
cp .env.example .env
# .env に Wyze の認証情報 (WYZE_EMAIL, WYZE_PASSWORD, API_ID, API_KEY) を記入
docker compose up -d --build
```

- `http://localhost:5050`: `docker-wyze-bridge` Web UI
- `rtsp://localhost:8554/wyze`: Wyze Cam Pan v3 ライブ RTSP ストリーム
- `ws://localhost:9092`: ROS2 Humble `rosbridge_websocket` サーバー（`wyze_ptz_proxy_node` 同梱）

### 2. Quest HMD × ROS2 × AI 探索・捕捉ノードの起動（メインデモ）

```bash
./.venv/bin/python -m src.node_quest_wyze_teleop --record demo_quest_wyze.mp4
```

- **Quest ブラウザ接続**: Quest で `https://<MacのIP>:8443` を開き、「🟢 MRパススルー開始」または「VRモード開始」をタップ。
- **キー操作 / UI操作**:
  - `[a]` キー（または Quest Bボタン / ブラウザの🎯ボタン）: AI 自律見回し探索 ＆ 対象物の中央ロックオン ON/OFF
  - `[t]` キー: ターゲット切替（カップ / ペットボトル / スマホ / 牛乳パック / 本 / リモコン）
  - `[r]` キー（または Quest 右トリガー）: 正面リセット（`(0°, 0°)`）
  - `[f]` キー: 画面中央の物体に対する 1クリック Few-Shot Visual FT 登録（発展機能）
  - `[d]` / `[l]` キー: Quest デモ記録の開始・停止 `/ 模倣学習（BC Policy）の実行（発展機能）
  - `[q]` キー: 終了

### 3. 単独モードでの起動（オプション）

```bash
# VLM / Open-Vocabulary 指定ターゲット指向ノード単独起動
./.venv/bin/python -m src.node_target_director --target "cup" --backend yolo

# 静止カメラからの顔角度ミラーリング追従ノード起動
./.venv/bin/python -m src.node_face_mirror --cam 2 --model data/face_to_ptz.pt
```
