# Transformer Debug Shape 学习笔记

这个文件配合 `transformer_debug.py` 阅读。debug 版本使用很小的参数，并在前向传播时打印关键 tensor 的 shape 和 causal mask，方便跟踪 Transformer 中每一步的数据流。

## 运行方式

```bash
cd docs/chapter2/code
python transformer_debug.py
```

脚本不依赖 `BertTokenizer.from_pretrained()`，不会联网下载模型。输入是手写的两个 token 序列。

## 符号约定

- `B = batch_size = 2`，一次输入 2 条样本。
- `T = sequence_length = 8`，每条样本有 8 个 token。
- `C = n_embd = dim = 16`，每个 token 的隐藏向量维度是 16。
- `H = n_heads = 4`，多头注意力有 4 个头。
- `D = head_dim = C / H = 4`，每个注意力头处理 4 维向量。
- `V = vocab_size = 32`，小词表大小是 32。

## Shape 逐步解释

`Embedding.idx` 的 shape 是 `(B, T)`，表示输入 token id。每个位置是一个整数，用来索引词表中的 token。

`Embedding.tok_emb` 的 shape 是 `(B, T, C)`。Embedding 层把每个 token id 查表成一个长度为 `C` 的向量，所以最后多出隐藏维度。

`PositionalEncoding.pe_slice` 的 shape 是 `(1, T, C)`。第一个维度为 1，可以广播到整个 batch；它给每个时间位置提供一个固定的位置向量。

`PositionalEncoding.output` 仍是 `(B, T, C)`。位置编码和 token embedding 相加，只改变向量内容，不改变 shape。

Attention 中的 `xq/xk/xv after wq/wk/wv` 是 `(B, T, C)`。线性层把输入 hidden state 投影成 Query、Key、Value。

`xq/xk/xv split heads` 是 `(B, T, H, D)`，表示把 16 维向量拆成 4 个头，每个头 4 维。

`xq/xk/xv transposed` 是 `(B, H, T, D)`。注意力计算希望每个头独立计算，所以把 head 维度提前。

`scores before mask` 是 `(B, H, T, T)`。每个 query 位置都要和所有 key 位置做一次相似度计算，所以最后两个维度是 `T x T`。

Decoder 的 `causal_mask` 是 `(1, 1, T, T)`，会广播到 `(B, H, T, T)`。二维视图中，对角线和左下角是 `0`，右上角是 `-inf`。第 `i` 个 token 只能看见自己和之前的 token，不能看见未来 token。

`attention weights` 仍是 `(B, H, T, T)`。softmax 后，每一行表示某个 query 位置对所有 key 位置的注意力分布。

`weighted value` 是 `(B, H, T, D)`。注意力权重乘以 Value 后，每个头得到自己的输出向量。

`merged heads` 是 `(B, T, C)`。多个头的结果被拼回隐藏维度。

Encoder 和 Decoder 的输入、每层输出、最终 norm 输出都是 `(B, T, C)`。它们在每个 token 位置上更新 hidden state，但保持 batch、序列长度和隐藏维度不变。

`lm_head.input` 是 `(B, T, C)`，表示 Decoder 输出的 hidden state。

`lm_head.logits` 是 `(B, T, V)`。`lm_head` 把每个 token 位置的 16 维 hidden state 投影到 32 个词表分数，用于预测下一个 token 或计算交叉熵 loss。
