# kobae 設計メモ

MaleCNS v1.0（オスショウジョウバエ中枢神経系の全結合図）を、**全ニューロン 166,700・全有向結合 25,582,938** のまま
LIF スパイキングモデルとして Vulkan (wgpu-py / WGSL) 上で回し、ブラウザで対話的に観察するためのシミュレータ。
DOOM や外部ゲームは接続しない。

## 決定事項（ユーザーと議論して確定）

### 1. ニューロンモデル
- **段階 1: Shiu et al. 2024 / DoomFly 型 LIF を全細胞に適用**（検証可能性の土台）
  - dt 0.1 ms、静止電位 −52 mV、閾値 −45 mV、リセット −52 mV
  - τm 20 ms、τg 5 ms（指数シナプス）、g の結合係数 (av − ag)/3
  - シナプス遅延 1.8 ms（18 step）、不応期 2.2 ms（22 step）
  - 重み = シナプス数 × 伝達物質符号 × 0.275 mV
  - 1 step の順序: 全細胞を積分・閾値判定 → t−18 のスパイク到着を g に加算（不応期中は捨てる）→ この step の発火細胞をリセット
- **段階 2: 視葉 (ol_intrinsic 89,403 / ol_sensory 6,098 / visual_projection 9,201 ≈ 全体の 6 割) を graded rate ユニット化**
  - 実際の光受容体・ラミナ・メダラの多くはスパイクせず段階的電位で信号を送るため
  - 段階 1 の検証を通してから、細胞集合の指定だけで切り替えられる構造にする（カーネル構造は共通）

### 2. ニューロン集合・重み
- ノード: superclass が付いていて status ≠ Glia の全行（DoomFly と同じ、166,700）。bodyId 昇順 = ノード index
- エッジ: 両端が retained な全リリース済みエッジ。シナプス数の閾値なし、自己結合も保持
- 伝達物質符号: ACh → +1、GABA / glutamate / histamine → −1
- **曖昧な 3,718 細胞（共放出・予測不能・修飾物質のみ）は設定で切り替え、既定 +1**（DoomFly / Shiu 流。検算 1:1 のため）

### 3. 検証基準
- 同じグラフ・同じ刺激で CPU 参照（DoomFly `doom/engine.py` の numba `advance` の忠実移植）と比較
- **初期 20 ms は発火集合（細胞 × step）が完全一致**
- **500 ms では統計一致**: 総発火数 ±2 %、細胞ごとの発火数の相関 > 0.99、発火した細胞集合の Jaccard > 0.95
- GPU 側は int32 固定小数点の整数加算で決定論的（実行ごとにビット一致）。CPU 参照は f32 逐次加算なので長期では発散する前提

## 設計案（レビュー中）

### GPU カーネル（WGSL, wgpu-py）
- 遅延 18 step = 1 バッチ。t で出たスパイクは t+18 まで誰にも影響しないので、18 step 分の積分を 1 ディスパッチで行える
- バッチごとに 3 ディスパッチ:
  1. `integrate`: 1 thread / neuron。18 substep をループし、各 substep で `gin[substep][i]`（int32 固定小数点、1/65536 mV）を `atomicExchange(…, 0)` で読み取り・クリア、g に加算。発火したら `(neuron << 5) | substep` を spike log に atomic append、細胞ごとの発火カウンタを加算
  2. `prep`: 1 thread。log の head から indirect dispatch 引数（スパイク数）を作る
  3. `scatter`: 1 workgroup / spike (64 thread)。CSR 行を stride して `gin[substep][post]` に `atomicAdd(int)`。indirect dispatch
- 固定小数点の理由: (a) 整数加算は可換なので原子加算の順序不定でも結果が決定論的、(b) Polaris (RX 580) に buffer float atomic が無い
- 重み範囲: 0.275 mV × 最大数千シナプス → ±32,768 mV まで表現、量子化誤差 7.6 µV/辺
- spike log の容量: n（不応期 22 step > 遅延 18 step なので 1 バッチに 1 細胞 1 回まで）
- 外部入力 (drive) はホストから書き換え。観測（発火カウント / log）はホストが N バッチごとに読み戻し

### ソフトウェア構成
- Python 3.14、uv 管理、パッケージ名 `kobae`
- `graph.py`: feather → CSR (`npz`)。DoomFly のノード/エッジ方針・網膜投影 (R1-R6 → L1/L2/L3 の最頻 hex 列 → uv) を移植 (MIT)
- `cpu.py`: numba 参照カーネル
- `gpu.py` + `shaders/lif.wgsl`: 本体
- `server.py` + `viewer/`: aiohttp + WebSocket、three.js 点群（annotations の `somaLocation` 列）
- ターゲット: 開発機 RX 580 (Polaris, 256 GB/s, RADV) で開発、BC-250 (gfx1013, 382 GB/s, 16 GB 共有) で本計測

### ベンチマーク条件
- 疎: 糖受容ニューロン LB3c に 30 mV 定常電流（Eon Systems の表と同系統）
- 密: 全 R1-R6 に輝度 1.0（30·1/1.02 mV）+ ラミナ L1/L2/L3/L5 に 12 mV 恒常
- 無入力
- 実時間比 = シミュレーション秒 ÷ 壁時計秒（ロード除く）。Eon の表は FlyWire（~500 万結合）なので参考比較にとどめる
