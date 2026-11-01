# Physical AI 講座 2026 応用編1 最終課題 説明資料・レポート

- **課題タイトル**: Meta Quest HMD × ROS2 × VLM/Open-Vocabulary 視覚認識による実機 PTZ カメラ（Wyze Cam Pan v3）の視線同期テレオペレーションと自律探索・対象捕捉システム
- **GitHub リポジトリ (Public)**: https://github.com/ambit1977/physical-ai-wyze-cam-ros2
- **デモ動画 URL (YouTube 公開)**: https://youtu.be/b3hHwuHKy7I （チャンネル: https://www.youtube.com/@ericapapapapapa / 2倍速・`2x Speed` 明記、1分37秒）

---

## 1. 概要（Overview）

本プロジェクトでは、市販のパン・チルト（PTZ）スマートカメラである **Wyze Cam Pan v3（型番: `HL_PAN3`）** を 2自由度（Pan: 水平回転、Tilt: 上下傾き）のアクティブビジョン・ロボットとして扱い、**ROS2 Humble** を介してリアルタイムに制御する Physical AI システムを設計・実装した。

本システムは以下の **3つの統合モード・機能** を備えている：
1. **Meta Quest HMD / 6DoF コントローラによる視線同期テレオペレーション**:
   HMD 内の視界正面に 16:9 広角ライブ映像を表示しながら、操作者の頭部角度（Yaw/Pitch）または 6DoF アームコントローラの姿勢を ROS2 トピック（`/wyze/cmd_ptz` およびルンバ互換 `/cmd_vel`）へ変換し、実機 PTZ カメラを直感的に遠隔操作する。
2. **多軸ロボットアーム動作からカメラ PT 軸・視線ベクトルへの変換（`ArmToPTZGazeRetargeter`）**:
   6軸/7軸ロボットアームの行動表現 $[\Delta x, \Delta y, \Delta z, \Delta \text{roll}, \Delta \text{pitch}, \Delta \text{yaw}, \text{gripper}]^\top$ から、カメラ光学中心を起点とする **3次元単位視線ベクトル（Gaze Line-of-Sight）** $\boldsymbol{v}_{\text{gaze}} = (g_x, g_y, g_z)$ を算出し、球面座標ヤコビアン投影によってカメラの Pan/Tilt 軸角度へと変換するクロスエンボディメント制御器を構築した。
3. **VLM / オープンボキャブラリ物体認識による自律探索と視覚サーボ捕捉（Active Search & Visual Servoing）**:
   指定されたターゲット（カップ、ペットボトル等）が視界に入っていない場合は人間のように周囲を段階的に見回して探索し、視界に対象物が入った瞬間にその対象物を画面中央クロスヘアへ捕捉（センタリング・ロックオン）する自律制御を実現した。

---

## 2. 背景と目的（Background & Purpose）

### 2.1 開発の経緯と発見した課題（PC画面凝視と頭部姿勢追従のジレンマ）
当初、PC（MacBook Air）のカメラで操作者の顔を撮影し、MediaPipe Face Mesh（468点3Dランドマーク）と PyTorch MLP（`FaceToPTZNet`）を用いて「操作者の顔の向きと同じ角度に Wyze Cam Pan v3 を動かす顔角度ミラーリング（Mode 1）」を実装・実機検証した。

しかし、実機を動かして操作する中で**ヒューマンインターフェース上の決定的な課題**が明らかになった：
- **課題A**: PC画面の前で顔を大きく上や下に向けると、操作者自身の目がPCモニターから外れてしまい、**Wyze Cam が映しているライブ映像を見ることができなくなる**。
- **課題B**: 映像を常に見ようとしてヘッドマウントディスプレイ（HMD）を装着すると、**HMDで顔が隠れるためカメラから顔の角度を検出できなくなる**。

### 2.2 解決アプローチ
上記のジレンマを根本から解決するため、**Meta Quest HMD を用いてカメラのライブ映像を視界内で常時確認しながら、HMD自身の向いている角度（または 6DoF コントローラ姿勢）を取得して ROS2 で実機カメラを制御する方式** へと発展させた。さらに、人間による遠隔操作だけでなく、AI（VLM / オープンボキャブラリ視覚モデル）が自ら周囲を見回して対象物を発見・中心捕捉する自律モードを融合させることで、実用的なアクティブビジョン・システムを目指した。

---

## 3. システム構成と実装（System Architecture）

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

### 主要コンポーネント
| モジュール | ファイルパス | 役割・技術要素 |
|---|---|---|
| **ROS2 PTZ プロキシノード** | `ros2_ws/src/wyze_ptz_proxy/.../wyze_ptz_proxy_node.py` | ROS2 Humble ノード。`/wyze/cmd_ptz` と `/cmd_vel` (Twist) を購読し、レート制限・デッドバンド処理を経て `docker-wyze-bridge` へ PTZ コマンドを送信、`/wyze/ptz_state` を Publish する。 |
| **Quest × ROS2 × AI 統合ノード** | `src/node_quest_wyze_teleop.py` | WebXR HMD (HTTPS 8443) と OpenTeach 6DoF APK (ZeroMQ TCP 8087/10505, UDP 8125) の両方式に対応し、ROS2 PTZ 制御と AI 自律探索・追尾を統合。 |
| **多軸アーム→PT視線変換・探索器** | `src/models/arm_to_ptz_retargeter.py` | 7DoF アーム動作を 3D 視線単位ベクトル $\boldsymbol{v}_{\text{gaze}}$ とカメラ Pan/Tilt 角度へ変換。RTSP遅延対策の **Step-and-Settle 視覚サーボ** と **人間らしい3D空間見回し探索** を実行。 |
| **VLM / Open-Vocab 認識器** | `src/models/vlm_target_detector.py` | `YOLO-World (CLIP)` + `YOLOv8n (COCO)` デュアル認識、および Hugging Face `google/owlv2-base-patch16-ensemble` / `vikhyatk/moondream2` に対応。 |
| **顔角度ミラーリング (Mode 1)** | `src/node_face_mirror.py`, `src/models/face_to_ptz_net.py` | MediaPipe 3D 顔姿勢推定 ＋ カメラ間視差を補正する PyTorch MLP (`FaceToPTZNet`) による顔角度同期ノード。 |

---

## 4. 工夫点と技術的考察（Key Innovations & Troubleshooting）

### 工夫1：Wyze Cam Pan v3 (`HL_PAN3`) 実機プロトコルの解析と絶対角度制御の確立
ドキュメント上では `docker-wyze-bridge` の角度制御として `rotary_degree` が紹介されていたが、実機 Pan v3（FW `4.50.3.4240`）で検証したところ `rotary_degree`（`K11000`）は `None` を返し動作しなかった。そこでコンテナ内の `tutk` プロトコル実装を調査し、Pan v3 専用の絶対角度エンドポイント **`/api/wyze/ptz_position/<vertical>,<horizontal>`（`K11018SetPTZPosition`、水平 `0〜350°`［正面 `180°`］、垂直 `0〜180°`［水平 `90°`］）** を特定した。これを ROS2 プロキシノードに組み込むことで、任意の Pan/Tilt 角度への高速かつ再現性の高いダイレクト位置制御を実現した。

### 工夫2：RTSP映像遅延（約0.6〜0.8秒）による「PTZ暴走（積分ワインドアップ）」の原因解明と Step-and-Settle 制御
AIターゲット追従モードの初期実験において、**「ターゲットモードに切り替えると、対象物ではない方向にカメラのPTZがぐるっと一周暴走してしまう」** という重要な問題に直面した。ログと制御周期を分析した結果、以下の複合原因を突き止めた：
1. **RTSP遅延に対する30Hz積分ワインドアップ**:
   メインループが約30Hz（33ms周期）で回る一方、Wyze Camのモーター移動からRTSP映像（`rtsp://localhost:8554/wyze`）に反映されるまで約0.6〜0.8秒（約20フレーム）の遅延が存在する。毎フレーム `pan += cx_norm * Kp` を加算し続けると、**カメラが動いた映像が届く前に同じ偏差が20回連続で加算（`10° × 20回 = 200°`）され、0.6秒で可動限界（`±170°`）まで振り切れていた**。
2. **低閾値での背景誤検出**:
   オープンボキャブラリ検出の閾値が低い（`0.20`）と、壁や机の角を誤検出して画面端へ回転し続けていた。

**【解決策：Step-and-Settle（Move-Wait-Observe）視覚サーボ ＋ 3D見回し探索ステートマシン】**
- **2フレーム連続検出確認** と **面積フィルタ（画面の45%以上を占める背景枠を除外）** を導入し、単発ノイズによる誤動作を遮断した。
- **視界に対象物が入っている時**: 1ステップの最大回転角を `Pan ±11° / Tilt ±7.5°` に制限し、**1回コマンドを送ったら `0.72秒`（モーター移動とRTSPフレームが整定する時間）待機してから次のフレームで再評価する Step-and-Settle 制御** を実装した。これにより暴走が完全に解消され、カップやペットボトルを1〜3ステップで滑らかに画面中央（`|cx_norm| < 0.08`）へ捉えてピタッと停止できるようになった。
- **視界に対象物が入っていない時**: 人間が周囲を見回して物を探す動作を模し、3D作業空間内の9つのウェイポイント（正面→左前・机上→左側→左上→正面やや上→右上→右側→右前・机上→手前・机上）へ視線ベクトルを段階的に向け、**各視点で `1.45秒` 立ち止まって周囲を観察する探索アルゴリズム** を実装した。探索中に視界に対象物が入った瞬間に探索を中断し、上記の中央捕捉モードへシームレスに遷移する。

### 工夫3：多軸アーム動作からカメラPT軸・向かう視線への変換モデル（Cross-Embodiment Retargeting）
ロボットアーム向けのモデルやテレオペ入力（6DoF/7DoF の手先位置変位 $\Delta x, \Delta y, \Delta z$ と手首回転 $\Delta \text{roll}, \Delta \text{pitch}, \Delta \text{yaw}$、グリッパ開閉 $g$）しか存在しない場合でもアクティブビジョン制御に活用できるよう、球面座標ヤコビアン投影に基づく変換則を設計・実装した：
$$\Delta \theta_{\text{pan}} = \text{atan2}(\Delta x, Z_0 + \Delta z)\frac{180}{\pi} + w_{\text{rot}}\Delta \text{yaw}, \quad \Delta \theta_{\text{tilt}} = \text{atan2}\!\left(\Delta y, \sqrt{\Delta x^2 + (Z_0 + \Delta z)^2}\right)\frac{180}{\pi} + w_{\text{rot}}\Delta \text{pitch}$$
さらにアームの奥行リーチ（$\Delta z > 0$）とグリッパ閉動作を「対象物への注視ズーム（Foveation Zoom）」へマッピングすることで、手先が対象に近づく動作をカメラの精密中心ロックとデジタルズームへ直結させた。

---

## 5. 実機検証結果と今後の発展課題（Results & Future Work）

### 5.1 実機検証結果
- **ROS2 PTZ制御 & Quest テレオペレーション**:
  Docker 上の `wyze-bridge` と `ros2-wyze-proxy` を連携させ、Meta Quest の頭部姿勢・コントローラ操作から ROS2 トピック `/wyze/cmd_ptz` を経由して実機 Wyze Cam Pan v3 が滑らかに追従することを確認した（上下感度のチューニング・デッドゾーン `1.5°`・EMAフィルタ適用）。また、RTSP ストリームを本来の 16:9 広角フル画角（`960x540`）かつ TCP トランスポート直結（`rtsp_transport;tcp`）に最適化し、広視野角・低遅延での視認性を確保した。
- **認識と VLM / Open-Vocabulary による対象補足**:
  机上および周囲に配置した **「カップ（`cup`）」** や **「ペットボトル（`plastic bottle`）」** に対して、視界外からの自動見回し探索 → 発見 → 画面中央クロスヘアへの視覚サーボ捕捉（ロックオン）が一連の自律動作として安定して機能することを実機で確認した。

### 5.2 考察と今後の発展課題（Fine-Tuning / Reinforcement Learning への展開）
実機実験を通じて、カップやペットボトルは事前学習モデル（`YOLOv8` / `YOLO-World`）で高い精度で認識・捕捉できた一方、**日本の屋根型「牛乳パック（`milk carton`）」** や **机に平置きした「スマホ（`cell phone`）」** は、欧米中心の事前学習データセット（ガロンボトル等）との**ドメインギャップ**や画角・角度による見え方の違いから、ゼロショット推論のみでは認識率が低下する場合があるという知見を得た。

この実世界ならではの課題を解決するため、本リポジトリには基盤コード（`VisualPrototypeFT` / `OnlineRLBanditHeadSelector` / `QuestDemoBCManager`）を実装済みであり、今後の発展課題として以下の **FT（ファインチューニング）/ RL（強化学習）/ 模倣学習** に取り組んでいく：
1. **Few-Shot Visual Prototype Fine-Tuning（少数ショット視覚ファインチューニング）の発展**:
   目の前に置いた実物の牛乳パックや特定の私物をカメラ中央に映して1クリック（`[f]` キー）で視覚特徴プロトタイプ（色空間分布＋構造特徴・VLM埋め込み）を登録し、事前学習モデルが知らないローカル物体でも即座に検出・追尾可能にするオンライン適応の高度化。
2. **オンライン強化学習（Contextual Bandit / RL）による検出ヘッド・サーボゲインの自律適応**:
   複数の視覚モデル（COCOネイティブ、Open-Vocabulary VLM、Few-Shot FT）の信頼度重みや Step-and-Settle サーボのステップゲインを、「画面中央への安定捕捉成功（正の報酬）」と「見失い・振動（ペナルティ）」からオンライン強化学習で環境ごとに自動最適化する仕組みの本格検証。
3. **Meta Quest テレオペレーションからの模倣学習（Behavior Cloning）**:
   人間が様々な初期角度から Quest を用いて対象物を探し出し中央に捉えるデモンストレーション軌道（`[d]` キー記録、`[l]` キー学習）を蓄積し、死角の少ない効率的な探索行動そのものをニューラル方策（`BCSearchNet`）として獲得させる取り組み。
