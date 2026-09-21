# 既存実装の評価メモ（2026-09-21）

「既存を先に一通り動かしてから、kobae を新規に作るか fork するかを判断する」ための記録。
チェックアウトは `eval/`（git 管理外）。

## 対象と一次情報

| 実装 | データ | 神経エンジン | 体 | 表示 | ライセンス |
|---|---|---|---|---|---|
| [gfly](https://github.com/SimonSaysGiveMeSmile/gfly) | MaleCNS v1.0、≥5 シナプスで 6,236,426 結合に削減（164,740 細胞） | JS 単スレッド、LIF dt 1 ms + 独自補正（常時脱分極、−75 mV クランプ、入次数正規化、分割抑制） | flybody 由来の運動学リグ（68 体、102 関節）、歩行/羽ばたきアニメ、飛行操作 | ブラウザ。139,631 体細胞点群 + 脳殻メッシュ、視葉カラム表示 | コード MIT 相当、データ CC-BY |
| [webgpu-fly](https://github.com/abgnydn/webgpu-fly) | FlyWire FAFB v783（メス脳、139,255 細胞、~1,500 万結合）+ MANC（腹髄 23,188 細胞） | WebGPU compute、LIF + 2 状態 α シナプス、1 カーネル/step | flybody を MuJoCo/WASM で物理シミュレーション、手書き三脚歩行を脊髄出力でスケール | ブラウザ three.js、replay-as-URL | MIT |
| [FLYBOARD](https://github.com/NullLabTests/flybrain) | MaleCNS 全結合（166,700 / 25.58M） | numpy/scipy/numba、**20 ms tick** の leaky integrator | なし | pygame 3 ペイン、3D 点群、6 ボタン電流注入 | MIT |
| [flyverse-core](https://github.com/tel-0s/flyverse-core) | MaleCNS 全結合 + 伝達物質 | PyTorch CUDA（CPU は「遅い」）、Shiu 型 + 視葉 graded | 体あり（詳細未確認） | コンソール（体カメラ、網膜モザイク、集団読み出し） | MIT |
| [Fly.exe](https://github.com/Ibtisam-Mohammad/Fly.exe) | MaleCNS Traced のみ（165,122 / 25.56M） | GeNN/**CUDA**、LIF、15 ms 結合間隔 × 150 step | NeuroMechFly v2 (flygym) + MuJoCo、42 関節駆動、揚力ゼロ | Web アリーナ（preview = 神経なし） | GPL-2.0+ |
| [DoomFly](https://github.com/nftechie/doomfly) | MaleCNS 全結合（166,700 / 25,582,938） | C++ 単スレッド、Shiu 型 dt 0.1 ms、イベント駆動 | なし | DOOM 画面のみ | MIT |

## 実行結果

（下に追記）

### FLYBOARD（開発機、CPU 16 スレッドのうち numba 単スレッド + numpy）
- `eval/flybrain`。uv 3.13 環境。`neuroglancer>=2.42` は PyPI に無く 2.41 に緩和。soma shard の取得先が 1 階層深く保存されるバグあり（手で移動）
- 全 25,582,938 結合、166,700 細胞。dt **20 ms**、τ 100 ms、列正規化した重み、しきい値 1.0 の toy モデル
- 体細胞座標: soma-points shard から 142,782 点 → 140,638 細胞に割当（84 %）、残り 16 % は fallback 位置
- ヘッドレス実測（`eval/flyboard_bench.py`）:

| 条件 | 実時間比 | 発火 |
|---|---|---|
| 無入力 | 9.1× | ~0 |
| FOOD（味覚 2,799 細胞） | 2.9× | 438k spikes/s |
| PAIN（257 細胞） | 6.6× | 57k spikes/s |
| MATE（fru/dsx 3,165 細胞） | 1.2× | 937k spikes/s |

- UI: pygame、3 ペイン（左: 網膜入力、中: 3D 点群、右: 6 ボタン + DRIVE スライダ + 集団メータ + 「誰が光ったか」上位型）。点群は回転/ズーム可、クリックで細胞型表示
- 流用できそうなもの: **presets.yaml のボタン定義**（type_glob / receptorType / fruDsx / 神経伝達物質 glob / 出入神経 でセレクタを書く仕組み）と、soma shard の読み方（neuroglancer + tensorstore）。神経モデルは粗すぎて流用しない

### DoomFly 参照カーネル（開発機、C++ 単スレッド、DOOM なし）
- `eval/doomfly`。uv 3.11 環境（numpy 1.24.4 / numba 0.61.2 / pyarrow 20）。正規化 25 s、prepare + カーネルビルド数十秒
- グラフは kobae の `graph.py` と同一（166,700 / 25,582,938 / 124,177,617 contacts、網膜 3,335、糖 23）
- 実測（`eval/doomfly_bench.py`、dt 0.1 ms、18 ms 刻みで step、2 s 分）:

| 条件 | 実時間比 | 発火 | 活動細胞 |
|---|---|---|---|
| 無入力 | 22.7× | 0 | 0 |
| 糖 LB3c 23 細胞に 30 mV | **0.14×** | 1.09M spikes/s | 33,511 |
| 全 R1-R6 に輝度 1.0 + ラミナ 12 mV | **0.68×** | 642k spikes/s | 14,108 |

- 糖刺激の方が視覚より重いのは、たった 23 細胞から 33k 細胞が持続発火する（1 細胞あたり 33 Hz）ため。Shiu 型 + 全結合 + 曖昧符号 +1 でどれだけ活動が広がるかの基準値として重要（Eon の FlyWire 版は活動細胞 ~450 だった）
- これが kobae GPU 版の**検算相手**であり、**CPU の基準値**（Codex 指摘の通り、GPU 側の勝ち筋は scatter の帯域）

### Fly.exe の体（開発機、flygym 2.1.0 + MuJoCo 3.9.0、神経なし）
- `eval/Fly.exe`。uv 3.13 環境（`flygym==2.1.0 mujoco>=3.9,<3.10`、`flygym_demo` は flygym の wheel に同梱）
- `scripts/body_controller_preview.py --headless`（EGL）: NeuroMechFly（メスの体を population prior として使用と明記）を flygym_demo の HybridTurningController（CPG + 脚接地の反射）で歩かせる。2 s / dt 0.5 ms = 4,000 step + 60 fps 相当の描画で **8.7 s**、前進 26 mm。動画 `eval/out/flygym-walk.mp4`
- Fly.exe の web preview モードは MuJoCo ではなく 2D の「運動学 + 衝突円盤」プロキシなので、体を見るならこのスクリプト
- **飛行はない**（README: "wing command in a body with no air"、揚力ゼロ）。NeuroMechFly / flygym は歩行・身繕い用の体
- 流用できるもの: flygym の Simulation / renderer / 追尾カメラの使い方（CPU で十分速い）、Fly.exe の `engines/flygym.py`（下行ニューロンの発火レート → 歩行コントローラの前進・旋回指令、GPL-2.0 なので写経ではなく参考）

### flybody（Google DeepMind / Janelia、Apache-2.0）— 飛行の唯一の既存資産
- figshare 25309105: `trained-fly-policies.zip` 6.5 MB（歩行 + 飛行 + 視覚誘導飛行の学習済みポリシー、TF SavedModel）、`flight-controller-reuse-checkpoints.zip` 32 MB、`datasets_flight-imitation.zip` 13 MB。`data/flybody/` に取得中（figshare は 202 を返してから配信開始するので遅い）
- webgpu-fly が歩行ポリシーで実証した手順（SavedModel → 重みを抽出 → 自前の MLP forward pass、`tools/extract_walking_policy.py`）を飛行ポリシーで繰り返せば、TensorFlow 無しで飛行制御が動く。ポリシーは obs[741] → 512×4 (LayerNorm+tanh) → 59 actions の小さな MLP
