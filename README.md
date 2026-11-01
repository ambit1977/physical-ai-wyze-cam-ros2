# Wyze Cam Pan v3 × ROS2 Physical AI 制御システム

ルンバ（Create 3）のROS2インターフェース（`/cmd_vel`）と互換性を持つ **Wyze Cam Pan v3 用 ROS2 PTZプロキシ**、および以下2つのAI制御モードを実装したパッケージです。

1. **【モード0（推奨・メイン構成）】Meta Quest HMD 視界固定ライブスクリーン × 頭部角度追従 × AIターゲットロックオン (`src/node_quest_wyze_teleop.py`)**
   - PC画面の前で顔を大きく上下左右に向けるとPC画面が見えなくなり、HMDを被るとカメラから顔が隠れるという課題を解決するため、**Meta Quest HMD を被り、VR空間の視界正面に常時固定（Head-Locked HUD）された大型スクリーンで Wyze Cam Pan v3 のライブ映像を見ながら、HMD（またはコントローラー）の向いている角度（Yaw/Pitch）で実機カメラをROS2制御**します。
   - さらに Quest の **Bボタン1つで AI 自動ターゲットロックオン（YOLOv8 / Hugging Face OWLv2 / Moondream2）** に切り替わり、画面内の対象物を自動追尾します。
   - **WebXR 方式（`https://<MacのIP>:8443`）** と、既存のサイドロード済み **Open Teach APK 方式（ZeroMQ `TCP 8087 / 10505` + `UDP 8125` 自動探索）** の両方に同時対応しています。
2. **【モード1】顔角度ミラーリング追従 (`src/node_face_mirror.py`)**
   - 静止カメラで撮影した自分の顔画像から、3D顔姿勢（Yaw / Pitch / Roll）とランドマーク特徴量を抽出。
   - PyTorchのニューラルキャリブレーションモデル（`FaceToPTZNet`）でカメラ間の位置オフセット・非線形性を補正し、ROS2（`/wyze/cmd_ptz`）経由で **Wyze Cam Pan v3 を自分の顔と同じ角度にリアルタイム同期** させます。
3. **【モード2】Hugging Face VLM / Zero-Shot 指定ターゲット指向 (`src/node_target_director.py`)**
   - Hugging Face のオープンボキャブラリ視覚モデル（**`google/owlv2-base-patch16-ensemble`** または軽量VLM **`vikhyatk/moondream2`**、高速な **`YOLOv8`**）を使用。
   - 自然言語テキスト（例: `"coffee mug"`, `"black bottle"`, `"cell phone"`, `"remote control"`）で指定した対象物を探索・検出し、画面中央に捉えるように **Wyze Cam Pan v3 の向きを自動で制御（探索スキャン＋ロックオン追従）** します。

---

## システム構成図

```
[ATOM Cam 2 / USBカメラ] (静止カメラ)        [自然言語ターゲット指示 ("coffee mug" 等)]
         │                                               │
         ▼                                               ▼
  Mode 1: FacePoseEstimator                     Mode 2: Hugging Face VLM / OWLv2
  + PyTorch FaceToPTZNet                        (Zero-Shot Grounding & Visual Servo)
         │                                               │
         └───────────────► ROS2 (rosbridge) ◄────────────┘
                           Topic: /wyze/cmd_ptz (JSON)
                           Topic: /cmd_vel (geometry_msgs/Twist 互換)
                                   │
                                   ▼
                        wyze_ptz_proxy_node (ROS2 Humble)
                                   │ HTTP REST API
                                   ▼
                           docker-wyze-bridge
                     (/api/<cam>/rotary_degree or rotary_action)
                                   │
                                   ▼
                         Wyze Cam Pan v3 (実機PTZ駆動)
```

---

## クイックスタート手順

### Step 1: Wyze API キーの取得と `.env` の設定

1. [Wyze Developer Portal](https://developer-api-console.wyze.com/) にWyzeアカウントでログインし、`API Key` を発行して **`API_ID` (Key ID)** と **`API_KEY`** を控えます。
2. `.env.example` をコピーして `.env` を作成し、認証情報を記入します。

```bash
cd wyze_cam_ros2
cp .env.example .env
# .env を編集して WYZE_EMAIL, WYZE_PASSWORD, API_ID, API_KEY, WYZE_CAM_NAME を設定
```

> **補足（カメラ名の確認）**: Wyzeアプリでカメラに付けた名前が `Wyze Cam Pan` の場合、`WYZE_CAM_NAME=wyze-cam-pan`（小文字・スペースをハイフンに変換）になります。`docker-wyze-bridge` 起動後に `http://localhost:5000` をブラウザで開くと正確なカメラ名（URIスラグ）と映像が確認できます。

### Step 2: Docker で `wyze-bridge` と `ROS2 プロキシ` を起動

```bash
docker compose up -d --build
```

- `http://localhost:5000`: `docker-wyze-bridge` のWeb UI（カメラの接続状態確認）
- `ws://localhost:9090`: ROS2 `rosbridge_server`（`wyze_ptz_proxy_node` が同梱起動）

### Step 3: 実機 PTZ 動作の疎通テスト

```bash
python3 -m src.verify_wyze_ptz --cam wyze-cam-pan --method degree
```
- もし Pan v3 のファームウェアで `rotary_degree`（絶対角度指定）が効かない場合は、`--method action`（および `.env` の `WYZE_PTZ_METHOD=action`）に切り替えると `rotary_action/left|right|up|down` で確実に動きます。

---

## 実行方法（2つのAIモード）

必要なPythonパッケージをインストールします（既存の `venv` または新規 `venv` 推奨）：

```bash
pip install -r requirements.txt
# Hugging Face VLM (OWLv2 / Moondream2) を使う場合:
pip install transformers pillow accelerate timm einops
```

### 【モード1】ATOM Cam 2（またはMacカメラ）の顔角度に合わせて Wyze Cam を動かす

```bash
# 1. カメラ間の非線形角度補正モデル (FaceToPTZNet) の事前学習 (約3秒)
python3 -m src.train_face_to_ptz --out data/face_to_ptz.pt

# (任意) 実機カメラの前で5方向を向いてインタラクティブにキャリブレーション学習する場合:
python3 -m src.train_face_to_ptz --interactive --cam 0 --out data/face_to_ptz.pt

# 2. 顔角度ミラーリング追従ノードの起動 (提出用動画を同時に録画する場合は --record を付与)
python3 -m src.node_face_mirror --cam 0 --model data/face_to_ptz.pt --record data/demo_face_mirror.mp4
```
- **ATOM Cam 2 をRTSPで使う場合**: `--cam "rtsp://user:pass@<ATOMのIP>:8554/live"` を指定
- **ATOM Cam 2 をUSBカメラモードで繋ぐ場合**: `--cam 0` または `--cam 1` を指定
- **実機接続前に画面上でテストする場合**: `--dry-run` を付けると、右下の仮想PTZレーダーで動きを確認できます。

### 【モード2】Hugging Face VLM / Zero-Shot で指定したものに Wyze Cam の向きを向ける

```bash
# 1. Hugging Face OWLv2 (任意の英語テキストで指定した物体を探してロックオン)
python3 -m src.node_target_director --target "coffee mug" --backend owlv2 --record data/demo_target_owlv2.mp4

# 2. Hugging Face Moondream2 (2B 軽量VLM による Pointing / Grounding)
python3 -m src.node_target_director --target "black bottle" --backend moondream

# 3. YOLOv8 (高速リアルタイム追従: cup, bottle, cell phone, person, remote 等)
python3 -m src.node_target_director --target "cup" --backend yolo
```
- 実行中にウィンドウ上で **`1`〜`6` キー** を押すとターゲット（`cup`, `bottle`, `cell phone`, `person`, `book`, `remote`）をリアルタイムに切り替えられます。
- **`n` キー** を押すとターミナルから自由なプロンプト（例: `"red pen"`, `"eyeglasses"`）を入力して追跡対象を変更できます。
