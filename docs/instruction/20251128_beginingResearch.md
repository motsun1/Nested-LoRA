# Nested LoRA for SEMA – Implementation Spec

## 0. Context & Goal

### Research context

* ベースリポジトリ：
  **SEMA (Self-Expansion of Pre-trained Models with Mixture of Adapters for Continual Learning)**

  * ViT ベースの画像分類のクラス増分 (class-incremental) Continual Learning を行うコードベース。
  * 既存実装は、タスクごとに Adapter/Expert を増やし、Router で組み合わせる MoA 系。

* 研究の目的：

  * Mixture-of-Adapters は「**どのアダプタを使うか（空間的・入力依存）**」にフォーカスした手法。
  * 本研究では、Nested Learning の発想を取り入れ、

    * **“速い LoRA（fast）” と “遅い LoRA（slow）”** を同じ層に持たせ、
    * **更新の時間スケール（タスク方向）で役割を分担**させる。
  * 目的は、**新タスクへの適応性（plasticity）と過去タスクの保持（stability）のトレードオフ**を、

    * 同じパラメータ規模の従来 Adapter/LoRA より改善できるか検証すること。

### このタスクでやること

* SEMA のコードベースに、**Nested LoRA** を新しい Adapter モードとして追加する。
* 最初のターゲットは **ViT ベースの画像分類 CIL 実験**。
* 実装バージョン：

  * **v0**: fast/slow 2 枚の LoRA を並列に足し合わせ、学習率や更新頻度で時間スケールを分ける。
  * **v1（あれば）**: タスク切り替え時に fast を slow に統合する「consolidation」ステップを追加。

---

## 1. Assumptions about the Repo

※ 仮の構造。実際のリポジトリ構造を確認して、適宜読み替えてください。

* モデル定義（ViT＋Adapter）があるファイル：

  * 例: `models/vit_adapter.py` / `models/adapter.py`
* LoRA または Adapter の実装：

  * 例: `modules/adapter.py` に `Adapter` クラスや LoRA 実装がある。
* Continual Learning のトレーニングループ：

  * 例: `train_cl.py` / `train_sema.py`
* クラス増分タスク設定：

  * 例: `datasets/cifar100_splits.py`

**前提**：
既存の Adapter/LoRA を完全に壊さず、「Nested LoRA」という新しいモードとして追加できるようにすること。

---

## 2. Feature Overview

### 2.1 Nested LoRA v0 – 概要

* 各対象層に対して、従来の LoRA を 2 枚に分ける：

  * `fast` LoRA：短期的に大きく更新される（高 lr、更新頻度高）
  * `slow` LoRA：小さく・ゆっくり更新される（低 lr、更新頻度低）
* Forward パスでは Router を使わず：

  * **常に両方の LoRA を足し合わせて使う**。

形式的には、もともと：

> ( y = W x + \Delta_{\text{lora}}(x) )

だったものを：

> ( y = W x + \Delta_{\text{slow}}(x) + \Delta_{\text{fast}}(x) )

に変更するイメージ。

### 2.2 Nested LoRA v1 – 概要（余裕があれば）

* タスク学習が終わるタイミングで、

  * `fast` の重みの一部を `slow` に統合する **consolidation** を行う。
  * その後 `fast` をリセットする。
* 擬似式：

> ( \theta_{\text{slow}} \leftarrow \theta_{\text{slow}} + \alpha , \theta_{\text{fast}} )
> ( \theta_{\text{fast}} \leftarrow 0 )

* α はハイパーパラメータ（例: 0.1）。

---

## 3. Config & API Design

### 3.1 新しい設定フラグ

実験を切り替えやすいように、設定ファイルや CLI に以下を追加してください：

* `--adapter_type`（既にあれば流用）

  * 候補: `none`, `sema_adapter`, `nested_lora`, ... など
* `--nested_lora` か `--use_nested_lora` (boolean)
* Nested LoRA 用のハイパーパラメータ：

  * `--nested_lora_rank`（LoRA の rank）
  * `--nested_lora_lr_fast`
  * `--nested_lora_lr_slow`
  * `--nested_lora_update_interval_slow`（slow を何ステップごとに更新するか。0 もしくは 1 なら毎ステップ）
  * （v1 用） `--nested_lora_consolidation_alpha`
  * （v1 用） `--nested_lora_use_consolidation` (bool)

---

## 4. Nested LoRA Module Design

### 4.1 既存 LoRA 実装の想定

既存の LoRA 実装が以下のような形だったと仮定します（実際のコードに合わせて調整してください）：

```python
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank, ...):
        super().__init__()
        # 低ランク A, B
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        # scaling 等

    def forward(self, x):
        return self.lora_B(self.lora_A(x)) * self.scaling
```

これを 2枚持つラッパを作るイメージです。

### 4.2 NestedLoRAAdapter クラス

新規クラス案：

```python
class NestedLoRAAdapter(nn.Module):
    def __init__(self, in_features, out_features, rank, **lora_kwargs):
        super().__init__()
        self.slow = LoRA(in_features, out_features, rank, **lora_kwargs)
        self.fast = LoRA(in_features, out_features, rank, **lora_kwargs)

    def forward(self, x):
        # 並列に足し合わせる
        return self.slow(x) + self.fast(x)
```

* 既存の Adapter クラスが「ベース線形層＋LoRA」を含んでいる場合：

  * ベース線形層はそのまま使い、
  * LoRA 部分だけ NestedLoRAAdapter に差し替える形で統合してください。

### 4.3 モデルへの組み込み

* ViT の各ブロックで Adapter/LoRA を挿入している箇所を特定し、

  * `LoRA` / `Adapter` を `NestedLoRAAdapter` に差し替えるコードパスを追加します。
* 例（疑似コード）：

```python
if config.adapter_type == "nested_lora":
    self.adapter = NestedLoRAAdapter(
        in_features=hidden_dim,
        out_features=hidden_dim,
        rank=config.nested_lora_rank,
        # 必要なら他の kwargs
    )
elif config.adapter_type == "sema_adapter":
    self.adapter = SEMAAdapter(...)
...
```

---

## 5. Optimizer & Training Loop

### 5.1 パラメタグループの分離

Optimizer 作成時に、Nested LoRA の fast/slow を別 param group に分けてください。

```python
slow_params = []
fast_params = []
base_params = []

for name, module in model.named_modules():
    if isinstance(module, NestedLoRAAdapter):
        slow_params += list(module.slow.parameters())
        fast_params += list(module.fast.parameters())
    else:
        # 既存 PTM やその他のパラメータ
        # もし他の Adapter を使わない場合は base 側だけになる
        pass

# 既存コードに base_params をどう扱っているかに合わせて調整
param_groups = []

if len(base_params) > 0:
    param_groups.append({"params": base_params, "lr": base_lr})

param_groups.append({
    "params": slow_params,
    "lr": cfg.nested_lora_lr_slow,
})

param_groups.append({
    "params": fast_params,
    "lr": cfg.nested_lora_lr_fast,
})

optimizer = torch.optim.Adam(param_groups, ...)
```

* `lr_fast >> lr_slow`（例: fast=1e-3, slow=1e-4 〜 1e-5）

### 5.2 更新頻度の違い（v0.5〜）

`nested_lora_update_interval_slow` を使って、slow の更新頻度を制御してください。

* `update_interval_slow = 0` or `1` のとき → slow も毎ステップ更新
* `update_interval_slow > 1` のとき → そのステップ数ごとに slow を更新

実装例（あくまでイメージ）：

```python
global_step = 0

for batch in dataloader:
    global_step += 1

    loss = compute_loss(model, batch)
    loss.backward()

    # fast は毎ステップ更新
    for group in optimizer.param_groups:
        # fast group を特定するために "name" 等をつけてもよい
        pass

    # シンプルに、以下のような2オプティマイザ構成でもOK
```

もし可能であれば：

* fast 用と slow 用で **別オプティマイザ** を使う方が実装が簡単です：

```python
optimizer_fast = Adam(fast_params, lr=cfg.nested_lora_lr_fast)
optimizer_slow = Adam(slow_params, lr=cfg.nested_lora_lr_slow)

for batch in dataloader:
    loss = ...
    loss.backward()

    optimizer_fast.step()
    optimizer_fast.zero_grad()

    if cfg.nested_lora_update_interval_slow <= 1 or global_step % cfg.nested_lora_update_interval_slow == 0:
        optimizer_slow.step()
        optimizer_slow.zero_grad()
    else:
        # slow の grad は破棄
        for p in slow_params:
            p.grad = None
```

どちらの構成でもよいので、**fast/slow で学習の時間スケールが変わるように**してください。

---

## 6. Task-Boundary Consolidation (Nested LoRA v1 – Optional)

※ 時間に余裕があれば実装してください。v0 が優先です。

### 6.1 目的

* タスク t の学習が終わったタイミングで、fast に蓄積された変化を slow に徐々に統合する。
* その後 fast をリセットすることで：

  * fast：短期記憶（タスク内での素早い適応）
  * slow：長期記憶（タスクを跨いで統合された知識）
    を表現する。

### 6.2 実装ポイント

* CL トレーニングループ内で、「タスク切り替え」を検知している箇所を探してください。

  * 例: 各タスクごとに `train_task(task_id)` を呼んでいる構造など。

* タスク終了直後に以下のような処理を追加：

```python
def consolidate_nested_lora(model, alpha: float):
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, NestedLoRAAdapter):
                slow_params = list(module.slow.parameters())
                fast_params = list(module.fast.parameters())
                for p_s, p_f in zip(slow_params, fast_params):
                    p_s.add_(alpha * p_f)  # slow ← slow + α * fast
                    p_f.zero_()            # fast ← 0
```

* `alpha` は `cfg.nested_lora_consolidation_alpha` から取得。
* `cfg.nested_lora_use_consolidation` が True のときのみ実行。

---

## 7. Logging & Debugging Aids

Nested LoRA の挙動を確認しやすいように、以下をログ化してください：

* fast / slow の学習率・更新間隔（設定値の確認）
* 各タスク終了時に：

  * 各タスクの accuracy
  * 全タスク平均 accuracy
  * forgetting 指標（既に実装があれば、それに準拠）
* v1 実装時：

  * consolidation 実行ログ（タスク ID と α）
  * 初期実験用：fast/slow のパラメータノルム（`||θ_fast||`, `||θ_slow||`）を簡単に print or log する。

---

## 8. Acceptance Criteria

このタスクの「完了」の定義は以下です：

1. **コンパイル & 実行**

   * `--adapter_type nested_lora`（または同等設定）で、CIFAR-100 split などの CIL 実験が実行できる。
   * 実行時にエラーなく学習ループが回る。

2. **機能動作**

   * fast/slow の LoRA パラメータがそれぞれ別 param group / optimizer で管理されている。
   * `nested_lora_update_interval_slow` の値を変えると、slow の更新頻度が変わることが確認できる（ログや breakpoint）。
   * （v1 実装時）タスク切り替え時に consolidation が呼ばれ、fast がリセットされる。

3. **再現性**

   * 既存の `sema_adapter` モードを壊さない。
   * 新しいモード追加により、既存の pipeline が極力そのまま使える。

---

## 9. Notes

* 可能な限り、既存の LoRA / Adapter 実装を再利用してください。

  * 例えば、`LoRA` クラスをそのまま 2 回インスタンス化して `NestedLoRAAdapter` を作る、という形。
* Router や Mixture-of-Experts 機構は、Nested LoRA では使用しません。

  * fast/slow は **常に両方の出力を足し合わせる**だけでOKです。
* 「直列構造（fast → slow）」は、今回のタスクでは必須ではありません。

  * 必要なら将来の拡張として検討します。
