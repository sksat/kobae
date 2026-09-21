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
