# Nested LoRA for SEMA – Implementation Spec (Updated)

**Last Updated**: 2025-11-28  
**Status**: Ready for Implementation  
**Repository**: SEMA-CL (Nested-LoRA branch)

---

## Executive Summary

本ドキュメントは、SEMA（Self-Expansion of Pre-trained Models with Mixture of Adapters）のコードベースに **Nested LoRA** を実装するための詳細な仕様書です。

### Key Points

1. **目的**: Nested Learning の原理を LoRA に適用し、Continual Learning における plasticity-stability trade-off を改善
2. **実装方針**: 既存の SEMA の Adapter 機構を壊さず、新しい `ffn_adapter_type: "nested_lora"` として追加
3. **コア機能**: 
   - Fast LoRA（高学習率・素早い適応）と Slow LoRA（低学習率・安定した学習）を並列に配置
   - 異なる学習率と更新頻度で時間スケールを分離
   - オプションで consolidation（fast → slow への知識統合）
4. **実装難易度**: 中程度（既存の `Adapter` クラスを再利用可能）

---

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

## 1. Actual Repository Structure (Confirmed)

実際のリポジトリ構造を確認した結果、以下の構成となっています：

* **メインエントリーポイント**：
  * `main.py`：引数解析と `trainer.py` への委譲
  * `trainer.py`：トレーニングループの実装
  * `eval.py`：評価ロジック

* **モデル定義**：
  * `models/sema.py`：SEMA の Learner クラス（トレーニング・評価ロジック）
  * `backbone/vit_sema.py`：ViT + SEMA Adapter の実装
  * `backbone/sema_block.py`：`SEMAModules` クラス（複数の Adapter とRouter を管理）
  * `backbone/sema_components.py`：`Adapter` クラス、`AE`（オートエンコーダ）、`AdapterModule` クラス

* **ネットワーク構築**：
  * `utils/inc_net.py`：`SEMAVitNet` など、各種ネットワークラッパーの定義と `get_backbone()` 関数

* **設定ファイル**：
  * `exps/*.json`：各実験の設定（例：`sema_inr_10task.json`）

* **既存の Adapter 実装**：
  * `backbone/sema_components.py` の `Adapter` クラスが既にダウンプロジェクション+アッププロジェクションの構造を持つ
  * 初期化オプションとして `"lora"` が既に実装されている（`init_option == "lora"` で kaiming 初期化 + zero 初期化）

**重要な発見**：
1. SEMA では、各層に複数の `AdapterModule`（functional adapter + representation descriptor）を持ち、`Router` で重み付け混合を行う
2. `SEMAModules` クラスが adapter の追加・管理・outlier 検出を担当
3. optimizer は `models/sema.py` 内で、`functional` と `rd`（representation descriptor）で別々に管理されている
4. タスク終了時に `end_of_task_training()` が呼ばれ、freeze 処理が行われる

**前提**：
既存の SEMA の Adapter 機構を壊さず、「Nested LoRA」を新しいアダプタータイプとして追加できるようにすること。

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

実験を切り替えやすいように、JSON 設定ファイル（`exps/*.json`）に以下を追加します：

**既存の関連パラメータ**：
* `"ffn_adapter_type"`：既に存在（例：`"adaptmlp"`）
* `"ffn_num"`：adapter の bottleneck dimension（既存）

**Nested LoRA 用の新規パラメータ**：
* `"ffn_adapter_type": "nested_lora"`：Nested LoRA を使用することを指定
* `"nested_lora_rank"`：LoRA の rank（デフォルト: 16）
* `"nested_lora_lr_fast"`：fast LoRA の学習率（例：0.01）
* `"nested_lora_lr_slow"`：slow LoRA の学習率（例：0.001）
* `"nested_lora_update_interval_slow"`：slow を何ステップごとに更新するか（0 or 1 = 毎ステップ、>1 = 間引き）
* （v1 用） `"nested_lora_consolidation_alpha"`：consolidation の重み（例：0.1）
* （v1 用） `"nested_lora_use_consolidation"`：consolidation を使うか（boolean）

**設定例**（`exps/nested_lora_inr_10task.json`）：
```json
{
    "model_name": "sema",
    "ffn_adapter_type": "nested_lora",
    "nested_lora_rank": 16,
    "nested_lora_lr_fast": 0.01,
    "nested_lora_lr_slow": 0.001,
    "nested_lora_update_interval_slow": 5,
    "nested_lora_use_consolidation": false,
    "nested_lora_consolidation_alpha": 0.1
}
```

---

## 4. Nested LoRA Module Design

### 4.1 既存 Adapter 実装の確認

既存の `Adapter` クラス（`backbone/sema_components.py`）は以下のような構造です：

```python
class Adapter(nn.Module):
    def __init__(self, config=None, adapter_id=None, d_model=None, bottleneck=None,
                 dropout=0.0, init_option="bert", adapter_scalar="1.0",
                 adapter_layernorm_option="in"):
        super().__init__()
        self.n_embd = config.d_model  # 768
        self.down_size = bottleneck  # config.ffn_num
        
        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)
        
        # init_option == "lora" の場合：
        # down_proj: kaiming_uniform_
        # up_proj: zeros_

    def forward(self, x):
        down = self.down_proj(x)
        down = self.non_linear_func(down)
        output = self.up_proj(down)
        return output
```

これは実質的に LoRA 形式（低ランク分解）になっています。

### 4.2 NestedLoRAAdapter クラスの設計

`backbone/sema_components.py` に新しいクラスを追加します：

```python
class NestedLoRAAdapter(nn.Module):
    """
    Nested LoRA: fast と slow の 2 つの LoRA を並列に持つ。
    Forward 時は両方の出力を足し合わせる。
    """
    def __init__(self, config, adapter_id, dropout=0.0):
        super().__init__()
        self.config = config
        self.adapter_id = adapter_id
        rank = config.nested_lora_rank
        
        # slow LoRA: 長期記憶用
        self.slow = Adapter(
            config=config,
            adapter_id=f"{adapter_id}_slow",
            bottleneck=rank,
            dropout=dropout,
            init_option="lora",
            adapter_scalar="1.0",
            adapter_layernorm_option="none"
        )
        
        # fast LoRA: 短期記憶用
        self.fast = Adapter(
            config=config,
            adapter_id=f"{adapter_id}_fast",
            bottleneck=rank,
            dropout=dropout,
            init_option="lora",
            adapter_scalar="1.0",
            adapter_layernorm_option="none"
        )
    
    def forward(self, x):
        # 両方の LoRA 出力を足し合わせる
        slow_out = self.slow(x)
        fast_out = self.fast(x)
        return slow_out + fast_out
```

### 4.3 AdapterModule への統合

既存の `AdapterModule` クラスは以下の構造です：

```python
class AdapterModule(nn.Module):
    def __init__(self, config, adapter_id, writer):
        super().__init__()
        self.functional = Adapter(...)  # ← ここを差し替える
        self.rd = AE(self.config)  # representation descriptor
```

これを、`config.ffn_adapter_type` に応じて分岐させます：

```python
class AdapterModule(nn.Module):
    def __init__(self, config, adapter_id, writer):
        super().__init__()
        
        # Adapter type に応じて functional を選択
        if hasattr(config, 'ffn_adapter_type') and config.ffn_adapter_type == 'nested_lora':
            self.functional = NestedLoRAAdapter(
                config=config,
                adapter_id=adapter_id,
                dropout=0.1
            )
        else:
            # 既存の Adapter
            self.functional = Adapter(
                config=config,
                adapter_id=adapter_id,
                dropout=0.1,
                bottleneck=config.ffn_num,
                ...
            )
        
        # RD はそのまま
        if self.not_addition_layer:
            self.rd = None
        else:
            self.rd = AE(self.config)
```

---

## 5. Optimizer & Training Loop

### 5.1 パラメタグループの分離

現在の `models/sema.py` の `update_optimizer_and_scheduler()` は以下のように実装されています：

```python
def update_optimizer_and_scheduler(self, num_epoch=20, lr=None):
    lr = self.args["init_lr"] if lr is None else lr
    func_params = [p for n,p in self._network.named_parameters() 
                   if ('functional' in n or 'router' in n or 'fc' in n) and p.requires_grad]
    self.optimizer = optim.SGD/Adam(func_params, lr=lr, ...)
```

これを拡張して、**Nested LoRA の fast/slow を別 param group** に分けます。

**実装方針**：

```python
def update_optimizer_and_scheduler(self, num_epoch=20, lr=None):
    lr = self.args["init_lr"] if lr is None else lr
    
    # Nested LoRA を使っている場合
    if hasattr(self.args, 'ffn_adapter_type') and self.args['ffn_adapter_type'] == 'nested_lora':
        slow_params = []
        fast_params = []
        other_params = []
        
        for n, p in self._network.named_parameters():
            if not p.requires_grad:
                continue
            
            if 'functional.slow' in n:
                slow_params.append(p)
            elif 'functional.fast' in n:
                fast_params.append(p)
            elif 'functional' in n or 'router' in n or 'fc' in n:
                other_params.append(p)
        
        param_groups = [
            {'params': slow_params, 'lr': self.args['nested_lora_lr_slow']},
            {'params': fast_params, 'lr': self.args['nested_lora_lr_fast']},
            {'params': other_params, 'lr': lr}
        ]
    else:
        # 既存の処理
        func_params = [p for n,p in self._network.named_parameters() 
                       if ('functional' in n or 'router' in n or 'fc' in n) and p.requires_grad]
        param_groups = [{'params': func_params, 'lr': lr}]
    
    if self.args['optimizer'] == 'sgd':
        self.optimizer = optim.SGD(param_groups, momentum=0.9, weight_decay=self.args["weight_decay"])
    elif self.args['optimizer'] == 'adam':
        self.optimizer = optim.AdamW(param_groups, weight_decay=self.args["weight_decay"])
    
    self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_epoch, eta_min=self.min_lr)
```

### 5.2 更新頻度の違い（v0.5〜）

`nested_lora_update_interval_slow` を使って、slow の更新頻度を制御します。

**実装方針（`models/sema.py` の `_init_train()` 内）**：

```python
def _init_train(self, total_epoch, train_loader, test_loader, optimizer, scheduler, phase='func'):
    prog_bar = tqdm(range(total_epoch))
    global_step = 0
    
    for _, epoch in enumerate(prog_bar):
        self._network.train()
        for i, (_, inputs, targets) in enumerate(train_loader):
            global_step += 1
            
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            outcome = self._network(inputs)
            logits = outcome["logits"]
            
            # loss 計算
            if phase == "func":
                loss = F.cross_entropy(logits[:, :self._total_classes], targets)
            elif phase == "rd":
                loss = outcome["rd_loss"]
            
            optimizer.zero_grad()
            loss.backward()
            
            # Nested LoRA の場合：slow の更新頻度を制御
            if hasattr(self.args, 'ffn_adapter_type') and self.args['ffn_adapter_type'] == 'nested_lora':
                update_interval = self.args.get('nested_lora_update_interval_slow', 1)
                
                # slow の grad を条件に応じてクリア
                if update_interval > 1 and global_step % update_interval != 0:
                    for n, p in self._network.named_parameters():
                        if 'functional.slow' in n and p.grad is not None:
                            p.grad.zero_()
            
            optimizer.step()
            # ... (残りの処理)
```

**あるいは、より明示的に 2 つの optimizer を使う実装も可能**：

```python
# update_optimizer_and_scheduler を 2 回呼び出し、
# self.optimizer_fast と self.optimizer_slow を分ける
# _init_train 内で：
optimizer_fast.step()
if global_step % update_interval == 0:
    optimizer_slow.step()
```

どちらの方法でも実装可能ですが、**1 つの optimizer + grad zero** の方が既存コードへの影響が小さいです。

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

SEMA では、タスク終了時に `SEMAModules.end_of_task_training()` が呼ばれます（`models/sema.py` の `_train()` 末尾）。

ここで consolidation を追加します：

**`backbone/sema_block.py` の `SEMAModules` クラス**：

```python
def end_of_task_training(self):
    # Nested LoRA の consolidation
    if hasattr(self.config, 'ffn_adapter_type') and self.config.ffn_adapter_type == 'nested_lora':
        if self.config.get('nested_lora_use_consolidation', False):
            self.consolidate_nested_lora(self.config.nested_lora_consolidation_alpha)
    
    # 既存の処理
    self.freeze_functional()
    self.freeze_rd()
    self.reset_newly_added_status()
    self.added_for_task = False

def consolidate_nested_lora(self, alpha: float):
    """
    fast の重みを slow に統合し、fast をリセットする。
    """
    with torch.no_grad():
        for adapter in self.adapters:
            if hasattr(adapter.functional, 'slow') and hasattr(adapter.functional, 'fast'):
                # slow ← slow + α * fast
                for p_s, p_f in zip(adapter.functional.slow.parameters(), 
                                   adapter.functional.fast.parameters()):
                    p_s.add_(alpha * p_f)
                
                # fast ← 0
                for p_f in adapter.functional.fast.parameters():
                    p_f.zero_()
    
    logging.info(f"Consolidated Nested LoRA at layer {self.layer_id} with alpha={alpha}")
```

この処理により、fast が学習したタスク固有の知識が slow に蓄積され、fast は次のタスクで新たに適応できるようになります。

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

   * `exps/nested_lora_*.json` 設定ファイルで、CIFAR-100 または ImageNet-R の CIL 実験が実行できる。
   * 実行時にエラーなく学習ループが回る（`python main.py --config exps/nested_lora_*.json`）。

2. **機能動作**

   * fast/slow の LoRA パラメータがそれぞれ別 param group / 学習率で管理されている。
   * `nested_lora_update_interval_slow` の値を変えると、slow の更新頻度が変わることが確認できる（ログや breakpoint）。
   * （v1 実装時）タスク切り替え時に consolidation が呼ばれ、fast がリセットされる（ログで確認）。

3. **再現性**

   * 既存の `"ffn_adapter_type": "adaptmlp"` モードを壊さない。
   * 新しいモード追加により、既存の SEMA pipeline が極力そのまま使える。

4. **実装範囲**

   * **Phase 1（必須）**：
     * `NestedLoRAAdapter` クラスの実装（`backbone/sema_components.py`）
     * `AdapterModule` での分岐処理
     * optimizer の param group 分離（`models/sema.py`）
     * 設定ファイルの追加（`exps/nested_lora_*.json`）
   
   * **Phase 2（オプション）**：
     * slow 更新頻度の制御（`nested_lora_update_interval_slow`）
     * consolidation 処理（v1）

---

## 9. Implementation Roadmap

実装の優先順位と手順を明確化します：

### Phase 1: 基本的な Nested LoRA の実装（必須）

1. **`backbone/sema_components.py` の修正**
   * `NestedLoRAAdapter` クラスを追加
   * `AdapterModule.__init__()` で `config.ffn_adapter_type == 'nested_lora'` の分岐を追加

2. **`models/sema.py` の修正**
   * `update_optimizer_and_scheduler()` で fast/slow の param group 分離
   * デフォルト引数の追加（`args.get('nested_lora_lr_fast', 0.01)` など）

3. **設定ファイルの作成**
   * `exps/nested_lora_cifar.json` を作成（CIFAR-100 ベース）
   * `exps/nested_lora_inr_10task.json` を作成（ImageNet-R ベース）

4. **動作確認**
   * 小規模データセット（CIFAR-100、2タスク）で実行確認
   * パラメータ数、学習率が正しく設定されているか確認

### Phase 2: slow 更新頻度の制御（推奨）

5. **`models/sema.py` の `_init_train()` 修正**
   * `global_step` カウンタを追加
   * `nested_lora_update_interval_slow` に応じて slow の grad をクリア

6. **実験**
   * `update_interval_slow` を 1, 5, 10 で比較
   * 学習曲線・forgetting の差異を観察

### Phase 3: Consolidation（オプション・時間に余裕があれば）

7. **`backbone/sema_block.py` の修正**
   * `SEMAModules.consolidate_nested_lora()` メソッドを追加
   * `end_of_task_training()` 内で consolidation を呼び出し

8. **実験**
   * consolidation あり/なしで accuracy と forgetting を比較
   * `alpha` の値（0.05, 0.1, 0.2）を探索

### Phase 4: 本格的な実験と分析

9. **ベースライン比較**
   * 既存の SEMA（`adaptmlp`）との比較
   * 同じパラメータ数での公平な比較（rank を調整）

10. **論文用の分析**
    * パラメータノルム `||θ_fast||`, `||θ_slow||` の推移をログ
    * 各タスクでの fast/slow の寄与度を可視化（重みの L2 ノルムなど）

---

## 10. Notes and Considerations

### 10.1 既存実装の再利用

* 可能な限り、既存の `Adapter` クラスを再利用してください。
* `NestedLoRAAdapter` は `Adapter` を 2 回インスタンス化する形で実装します。
* Router や Mixture-of-Experts 機構は、Nested LoRA の内部では使用しません。
  * fast/slow は **常に両方の出力を足し合わせる**だけでOKです。
  * SEMA の Router は `SEMAModules` レベルで複数の `AdapterModule` を混合するために使われます。

### 10.2 直列構造について

* 「直列構造（fast → slow）」は、今回のタスクでは必須ではありません。
* 並列構造（fast + slow）の方が実装が単純で、かつ論文の Nested Learning の原理にも沿っています。
* 必要なら将来の拡張として検討します。

### 10.3 パラメータ数の公平性

* Nested LoRA（rank=16）は、通常の Adapter（bottleneck=16）の約 2 倍のパラメータを持ちます。
* 公平な比較のために：
  * **Option A**: Nested LoRA の rank を半分にする（rank=8）
  * **Option B**: 通常の Adapter の bottleneck を 2 倍にする（bottleneck=32）
* 実験では両方のオプションを試すことを推奨します。

### 10.4 SEMA の自動拡張との関係

* SEMA は outlier 検出により、必要に応じて新しい Adapter を追加します。
* Nested LoRA を使う場合でも、この自動拡張機構は維持されます。
* つまり、各 `AdapterModule` の `functional` が `NestedLoRAAdapter` になるだけで、
  SEMAModules レベルの複数 Adapter 管理は変わりません。

### 10.5 期待される効果

* **Fast LoRA**：タスク内での素早い適応により、新タスクの学習が効率化
* **Slow LoRA**：タスクを跨いだ一般的な特徴の抽出により、forgetting が軽減
* **Consolidation（v1）**：fast で学習した知識を slow に統合することで、長期記憶を強化

### 10.6 デバッグのヒント

* パラメータが正しく分離されているか確認：
  ```python
  for name, param in model.named_parameters():
      if param.requires_grad:
          print(name, param.shape, param.grad is not None)
  ```
* 学習率が正しく設定されているか確認：
  ```python
  for i, group in enumerate(optimizer.param_groups):
      print(f"Group {i}: lr={group['lr']}, #params={len(group['params'])}")
  ```
* slow の更新頻度を確認（ログに global_step と slow update のタイミングを出力）

---

## 11. References

* **SEMA 論文**: "Self-Expansion of Pre-trained Models with Mixture of Adapters for Continual Learning", CVPR 2025
* **Nested Learning 論文**: （NL.pdf を参照）
  * 複数の時間スケールでの学習の重要性
  * fast/slow network の相互作用
  * consolidation による知識の統合
* **LoRA**: "Low-Rank Adaptation of Large Language Models", Hu et al., ICLR 2022
* **Continual Learning**: Plasticity-Stability Dilemma の古典的な問題設定

---

## 12. Quick Start Guide (Implementation Checklist)

実装を開始する際のチェックリストです：

### Step 1: 環境確認
- [ ] リポジトリのクローン完了
- [ ] 既存の SEMA が動作することを確認（`python main.py --config exps/sema_inr_10task.json`）
- [ ] 依存パッケージのインストール完了

### Step 2: NestedLoRAAdapter の実装
- [ ] `backbone/sema_components.py` に `NestedLoRAAdapter` クラスを追加
- [ ] `AdapterModule.__init__()` で `ffn_adapter_type` による分岐を追加
- [ ] 簡単な単体テストで forward が動作することを確認

### Step 3: Optimizer の修正
- [ ] `models/sema.py` の `update_optimizer_and_scheduler()` を修正
- [ ] fast/slow の param group 分離を実装
- [ ] デバッグ用のログを追加（各 param group の学習率・パラメータ数）

### Step 4: 設定ファイルの作成
- [ ] `exps/nested_lora_cifar.json` を作成（CIFAR-100、2タスク）
- [ ] 必要なパラメータをすべて含める（nested_lora_rank, lr_fast, lr_slow など）

### Step 5: 動作確認
- [ ] 小規模実験で動作確認（CIFAR-100、init_cls=50, increment=10, 2タスクのみ）
- [ ] エラーなく学習が完了することを確認
- [ ] fast/slow のパラメータが正しく更新されていることを確認

### Step 6: slow 更新頻度の制御（Phase 2）
- [ ] `_init_train()` に `global_step` カウンタを追加
- [ ] `nested_lora_update_interval_slow` による grad クリアを実装
- [ ] update_interval を変えて動作確認

### Step 7: Consolidation の実装（Phase 3・オプション）
- [ ] `SEMAModules.consolidate_nested_lora()` を実装
- [ ] `end_of_task_training()` での呼び出しを追加
- [ ] consolidation の動作をログで確認

### Step 8: 本格実験
- [ ] ImageNet-R 10タスクでの実験
- [ ] ベースライン（通常の Adapter）との比較
- [ ] 結果の可視化と分析

---

## 13. Important Files Summary

実装で修正・追加するファイルの一覧：

| ファイルパス | 修正内容 | 優先度 |
|------------|---------|--------|
| `backbone/sema_components.py` | `NestedLoRAAdapter` クラス追加、`AdapterModule` 修正 | 必須 |
| `models/sema.py` | `update_optimizer_and_scheduler()` 修正、`_init_train()` 修正 | 必須 |
| `backbone/sema_block.py` | `consolidate_nested_lora()` 追加（v1） | オプション |
| `exps/nested_lora_cifar.json` | 新規作成 | 必須 |
| `exps/nested_lora_inr_10task.json` | 新規作成 | 必須 |

**修正不要なファイル**：
- `backbone/vit_sema.py`：変更なし
- `utils/inc_net.py`：変更なし（既存の `get_backbone()` がそのまま使える）
- `main.py`, `trainer.py`, `eval.py`：変更なし
