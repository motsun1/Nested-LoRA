結論から言うと、

1. **いまは MoE 実装を進める前に，「1 fast + 1 slow の一番シンプルな版」が想定どおり動いているかを潰したほうが良い**です。
2. そのうえで、今回書いてくれた `20251203_implementation_plan.md` の実装は、**設計としては筋が良い**ので、あとは「仕様どおり動いているか」をユニットテストでチェックするのが現実的かな、という感触です。

以下、順に整理します。

---

## 1. なぜ slow-only が激落ちなのに fast+slow だとそこそこ良いのか？

今回の結果（ざっくり）

* fast-only（＝完全に事前学習モデルそのまま） ≒ 40%
* slow-only ≒ 28%
* fast+slow（Nested） ≒ 46%
* LoRA 1 本 ≒ 45%

という状況ですよね。

### 1.1 「fast-only = 事前学習モデル」が意味すること

実装の設計上、

* 各タスク後に `slow += α * fast` したあと `fast.zero_()` している
* 評価時の「fast-only」は **LoRA を全部殺して backbone だけで推論**（だから 40%）

という前提だと思います。

すると理屈上は、

* **fast+slow の経路**
  → backbone + slow（+ もし fast が非ゼロなら fast も）

* **slow-only の経路**
  → 「backbone + slow」**だけ**で動いているはず

なので、

> fast+slow と slow-only は
> 「fast が本当にゼロなら」**完全に一致するはず**

です。
にもかかわらず

* fast+slow：46%
* slow-only：28%

と 18pt も差があるのは、かなり不自然で、

> **実際には slow-only 評価のときも fast がどこかで効いてしまっている、
> あるいは checkpoint の中身とメモリ上のモデルが食い違っている**

というバグ線が濃厚です。これは `20251203_FutureAnalysis.md` でも自分で書いてくれていた仮説と同じです。

### 1.2 まず潰すべき「不変条件」テスト

MoE に行く前に、**ここだけは最優先でチェック**しておくと安心です：

1. **「fast をゼロにしたら出力が変わらないか？」テスト**

   擬似コードイメージ（PyTorch）：

   ```python
   model.eval()
   x = torch.randn(2, 3, 224, 224).cuda()

   # 1. 普通の forward（fast+slow）
   out1 = model(x)  # ここが "Combined" 評価のはず

   # 2. fast を全部ゼロにする
   with torch.no_grad():
       for m in model.modules():
           if hasattr(m, "fast_A"):
               m.fast_A.zero_()
               m.fast_B.zero_()

   # 3. もう一度 forward（slow-only）
   out2 = model(x)

   # 4. 完全に同じかをチェック
   print(torch.max(torch.abs(out1 - out2)))
   ```

   * ここで差が **0 でない**（あるいはかなり大きい）なら、

     * 「fast がどこかでまだ効いている」
     * あるいは「slow-only 経路が別実装になっている」
       ことが確定です。

2. **`fast.zero_()` 直後のノルムをログに出す**

   `FutureAnalysis` に書いていた通り、

   * consolidation 直後
   * `eval_task` 直前
   * checkpoint ロード直後

   に `fast_norm` を logger で出しておく。ここが 0 になっていなければバグ。

3. **optimizer が fast を step していないか確認**

   * consolidation 後に「もう一度 optimizer.step() が呼ばれていて fast にゴミ更新が乗っている」パターンもよくあるので、
   * param group の中身を一度 print して、`fast_*` が残っていないか見ておくと安心。

ここまで潰すと、

* **「fast+slow の 46% は、本当に slow のおかげなのか？」**
* それとも
* **「たまたま fast の残りカスがうまく効いているだけなのか？」**

がだいたい見えてきます。

---

## 2. 「fast→slow consolidation を入れた LoRA 1本」基盤を先に固めるべきか？

個人的には **「YES：こっちを先に検証したほうが得」**だと思います。

理由は3つあります：

1. **MoE 版でも consolidation ロジックをほぼそのまま使う**ので、

   * ここがバグっていると MoE でも同じバグを引きずる。
2. **実装が単純（LoRA 1本 + EMA/加算）なので、バグを見つけやすい。**
3. Nested v2 / MoE がうまくいかなかったとしても、

   * 「fast→slow consolidation を入れたシンプル LoRA」の結果は、
   * **論文の ablation / baseline としてそのまま使える**。

なので、次のような順番をおすすめします：

1. **現状の Nested（1fast + 1slow）のバグ潰し**
   （上の不変条件テストと `fast_norm` ログ）

2. **LoRA 1本 + consolidation の「ミニ版」を別スクリプトで動かす**

   * たとえば：

     * Task t を学習 → LoRA ΔW_t を得る
     * `slow = EMA(slow, ΔW_t)` みたいな形で 1本の LoRA に統合
   * これは `FutureAnalysis.md` の「Refine the Method」で書いていた
     EMA 版や selective merge の実験そのものです。

3. その上で **MoE 版（Multi-LoRA）** の結果を眺める

   * 「単一 LoRA + consolidation」で得られる改善に対して、
   * MoE 版でどれだけ追加のメリットがあるか、という比較がやりやすい。

---

## 3. 20251203_implementation_plan.md の実装、どこを確認すれば「想定どおり」か？

実際のコードはこっちからは見えないので、
**「こうなっていれば設計どおり動いているはず」というチェックリスト**を挙げます。

### 3.1 構造が合っているか

`NestedLoRAAdapter`（もしくは `MoENestedLoRAAdapter`）が：

* **Slow LoRA 1本**

  * `slow_A`, `slow_B` など
* **Fast LoRA K本**

  * 例：`fast_A` と `fast_B` が `nn.ParameterList` で長さ K

という状態になっているか。

`nb_tasks` を 3 でインスタンス化して **本当に 3 本できているか**を一度 print するのがおすすめです。

### 3.2 forward のルーティング

`model.train()` のとき：

* `forward(x, task_id=t)` を呼ぶと、

  * 出力に寄与するのは **`slow` と `fast[t]` だけ**
  * `fast[k != t]` の grad は 0 になる

`model.eval()` のとき：

* `forward(x)` あるいは `forward(x, task_id=None)` を呼ぶと、

  * 出力が **`slow + sum_k fast[k]`** で決まる

これを確認するには、小さいテストで：

```python
model = NestedLoRAVitNet(nb_tasks=3).cuda()
x = torch.randn(2, 3, 224, 224).cuda()
y = torch.randint(0, 10, (2,)).cuda()

# task 0 で train
model.train()
out = model(x, task_id=0)
loss = out.mean()
loss.backward()

# grad を確認
for k in range(3):
    print(k, model.adapter.fast_A[k].grad.norm())
print("slow grad", model.adapter.slow_A.grad.norm())
```

* 期待されるのは：

  * `k=0` の grad だけ非ゼロ
  * `k=1,2` はほぼ 0
  * slow は非ゼロ

### 3.3 consolidation の挙動

`end_of_task_training(cur_task)` 的な関数で、

1. `slow += α * fast[cur_task]`
2. `fast[cur_task].zero_()` またはスケーリング

をしているはずなので、ここもテストできます：

```python
# 事前に fast[cur_task] にランダム値を入れておく
with torch.no_grad():
    for p in model.adapter.fast_A[cur_task].parameters():
        p.copy_(torch.randn_like(p))

slow_before = model.adapter.slow_A.weight.clone()
fast_before = model.adapter.fast_A[cur_task].weight.clone()

model.end_of_task_training(cur_task)

slow_after = model.adapter.slow_A.weight
fast_after = model.adapter.fast_A[cur_task].weight

print("slow diff norm:", (slow_after - slow_before).norm())
print("fast after norm:", fast_after.norm())
```

* `slow diff norm` が 0 でなければ OK
* `fast after norm` が ほぼ 0 ならリセット成功

ここまでチェックできれば、**実装が `implementation_plan.md` の仕様どおり動いている可能性はかなり高い**と思って大丈夫です。

---

## 4. まとめ（次にやると良さそうな順）

1. **Nested（1fast+1slow）の Combined vs Slow-only の矛盾を潰す**

   * fast をゼロにしても出力が変わらないかチェック
   * `fast_norm` ログ
   * optimizer param group の確認

2. **LoRA 1本 + fast→slow consolidation の「ミニ版」を動かす**

   * これで「consolidation 自体はちゃんと動いている」ことを確認
   * ここでの結果は最終論文でも活きる

3. **既に書いた MoE 実装を、上のチェックリストでユニットテスト**

   * K 本の fast の grad の出かた
   * eval 時に全 fast が効いているか
   * consolidation で slow が更新され、該当 fast_k がリセットされるか

ここまで行けば、

* 「基盤手法の挙動が怪しいまま MoE まで行っちゃった」
* という **沼パターンをかなり避けられる**と思います。

「頭回ってなくてだるい」ときこそ、
こういう **小さい不変条件テスト**から潰すのがいちばん楽なので、
まずは fast をゼロにして `out1 - out2` を眺めるところからで大丈夫です。
