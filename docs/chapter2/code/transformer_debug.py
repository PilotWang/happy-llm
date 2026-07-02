import torch
import math
from torch import nn
from dataclasses import dataclass
import torch.nn.functional as F


def print_shape(name, tensor):
    print(f"{name:<45} shape={tuple(tensor.shape)}")


@dataclass
class ModelArgs:
    n_embd: int  # 嵌入维度
    n_heads: int  # 头数
    dim: int  # 模型维度
    dropout: float
    max_seq_len: int
    vocab_size: int
    block_size: int
    n_layer: int


class MultiHeadAttention(nn.Module):

    def __init__(self, args: ModelArgs, is_causal=False, name="Attention"):
        # 构造函数
        # args: 配置对象
        super().__init__()
        # 隐藏层维度必须是头数的整数倍，因为后面我们会将输入拆成头数个矩阵
        assert args.dim % args.n_heads == 0
        # 每个头的维度，等于模型维度除以头的总数。
        self.head_dim = args.dim // args.n_heads
        self.n_heads = args.n_heads
        self.name = name

        # Wq, Wk, Wv 参数矩阵，每个参数矩阵为 n_embd x dim
        # 这里通过三个组合矩阵来代替了n个参数矩阵的组合，其逻辑在于矩阵内积再拼接其实等同于拼接矩阵再内积，
        # 不理解的读者可以自行模拟一下，每一个线性层其实相当于n个参数矩阵的拼接
        self.wq = nn.Linear(args.n_embd, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.n_embd, self.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.n_embd, self.n_heads * self.head_dim, bias=False)
        # 输出权重矩阵，维度为 dim x dim（head_dim = dim / n_heads）
        self.wo = nn.Linear(self.n_heads * self.head_dim, args.dim, bias=False)
        # 注意力的 dropout
        self.attn_dropout = nn.Dropout(args.dropout)
        # 残差连接的 dropout
        self.resid_dropout = nn.Dropout(args.dropout)
        self.is_causal = is_causal

        # 创建一个上三角矩阵，用于遮蔽未来信息
        # 注意，因为是多头注意力，Mask 矩阵比之前我们定义的多一个维度
        if is_causal:
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            # 注册为模型的缓冲区
            self.register_buffer("mask", mask)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):

        print(f"\n[{self.name}]")
        print_shape(f"{self.name}.q input", q)
        print_shape(f"{self.name}.k input", k)
        print_shape(f"{self.name}.v input", v)

        # 获取批次大小和序列长度，[batch_size, seq_len, dim]
        bsz, seqlen, _ = q.shape

        # 计算查询（Q）、键（K）、值（V）,输入通过参数矩阵层，维度为 (B, T, n_embed) x (n_embed, dim) -> (B, T, dim)
        xq, xk, xv = self.wq(q), self.wk(k), self.wv(v)
        print_shape(f"{self.name}.xq after wq", xq)
        print_shape(f"{self.name}.xk after wk", xk)
        print_shape(f"{self.name}.xv after wv", xv)

        # 将 Q、K、V 拆分成多头，维度为 (B, T, n_head, dim // n_head)，然后交换维度，变成 (B, n_head, T, dim // n_head)
        # 因为在注意力计算中我们是取了后两个维度参与计算
        # 为什么要先按B*T*n_head*C//n_head展开再互换1、2维度而不是直接按注意力输入展开，是因为view的展开方式是直接把输入全部排开，
        # 然后按要求构造，可以发现只有上述操作能够实现我们将每个头对应部分取出来的目标
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)
        print_shape(f"{self.name}.xq split heads", xq)
        print_shape(f"{self.name}.xk split heads", xk)
        print_shape(f"{self.name}.xv split heads", xv)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        print_shape(f"{self.name}.xq transposed", xq)
        print_shape(f"{self.name}.xk transposed", xk)
        print_shape(f"{self.name}.xv transposed", xv)

        # 注意力计算
        # 计算 QK^T / sqrt(d_k)，维度为 (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        print_shape(f"{self.name}.scores before mask", scores)

        # 掩码自注意力必须有注意力掩码
        if self.is_causal:
            assert hasattr(self, "mask")
            # 这里截取到序列长度，因为有些序列可能比 max_seq_len 短
            causal_mask = self.mask[:, :, :seqlen, :seqlen]
            print_shape(f"{self.name}.causal_mask", causal_mask)
            print(f"{self.name}.causal_mask[0, 0]:")
            print(causal_mask[0, 0])
            scores = scores + causal_mask
            print_shape(f"{self.name}.scores after mask", scores)

        # 计算 softmax，维度为 (B, nh, T, T)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        print_shape(f"{self.name}.attention weights", scores)

        # 做 Dropout
        scores = self.attn_dropout(scores)

        # V * Score，维度为(B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        output = torch.matmul(scores, xv)
        print_shape(f"{self.name}.weighted value", output)

        # 恢复时间维度并合并头。
        # 将多头的结果拼接起来, 先交换维度为 (B, T, n_head, dim // n_head)，再拼接成 (B, T, n_head * dim // n_head)
        # contiguous 函数用于重新开辟一块新内存存储，因为Pytorch设置先transpose再view会报错，
        # 因为view直接基于底层存储得到，然而transpose并不会改变底层存储，因此需要额外存储
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        print_shape(f"{self.name}.merged heads", output)

        # 最终投影回残差流。
        output = self.wo(output)
        print_shape(f"{self.name}.after wo", output)
        output = self.resid_dropout(output)
        print_shape(f"{self.name}.output", output)
        return output


class LayerNorm(nn.Module):
    """Layer Norm 层"""

    def __init__(self, features, eps=1e-6):
        super().__init__()
        # 线性矩阵做映射
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        # 在统计每个样本所有维度的值，求均值和方差
        mean = x.mean(-1, keepdim=True)  # mean: [bsz, max_len, 1]
        std = x.std(-1, keepdim=True)  # std: [bsz, max_len, 1]
        # 注意这里也在最后一个维度发生了广播
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class MLP(nn.Module):
    """前馈神经网络"""

    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        # 定义第一层线性变换，从输入维度到隐藏维度
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        # 定义第二层线性变换，从隐藏维度到输入维度
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        # 定义dropout层，用于防止过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 前向传播函数
        # 首先，输入x通过第一层线性变换和RELU激活函数
        # 最后，通过第二层线性变换和dropout层
        return self.dropout(self.w2(F.relu(self.w1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        # 一个 Layer 中有两个 LayerNorm，分别在 Attention 之前和 MLP 之前
        self.attention_norm = LayerNorm(args.n_embd)
        # Encoder 不需要掩码，传入 is_causal=False
        self.attention = MultiHeadAttention(args, is_causal=False, name="Encoder.self_attention")
        self.fnn_norm = LayerNorm(args.n_embd)
        self.feed_forward = MLP(args.dim, args.dim, args.dropout)

    def forward(self, x):
        # Layer Norm
        x = self.attention_norm(x)
        # 自注意力
        h = x + self.attention.forward(x, x, x)
        # 经过前馈神经网络
        out = h + self.feed_forward.forward(self.fnn_norm(h))
        return out


class Encoder(nn.Module):
    """Encoder 块"""

    def __init__(self, args):
        super(Encoder, self).__init__()
        # 一个 Encoder 由 N 个 Encoder Layer 组成
        self.layers = nn.ModuleList([EncoderLayer(args) for _ in range(args.n_layer)])
        self.norm = LayerNorm(args.n_embd)

    def forward(self, x):
        print("\n[Encoder]")
        print_shape("Encoder.input", x)
        "分别通过 N 层 Encoder Layer"
        for i, layer in enumerate(self.layers):
            x = layer(x)
            print_shape(f"Encoder.layer_{i}.output", x)
        x = self.norm(x)
        print_shape("Encoder.norm_output", x)
        return x


class DecoderLayer(nn.Module):
    """Decoder 层"""

    def __init__(self, args):
        super().__init__()
        # 一个 Layer 中有三个 LayerNorm，分别在 Mask Attention 之前、Self Attention 之前和 MLP 之前
        self.attention_norm_1 = LayerNorm(args.n_embd)
        # Decoder 的第一个部分是 Mask Attention，传入 is_causal=True
        self.mask_attention = MultiHeadAttention(args, is_causal=True, name="Decoder.mask_self_attention")
        self.attention_norm_2 = LayerNorm(args.n_embd)
        # Decoder 的第二个部分是 类似于 Encoder 的 Attention，传入 is_causal=False
        self.attention = MultiHeadAttention(args, is_causal=False, name="Decoder.cross_attention")
        self.ffn_norm = LayerNorm(args.n_embd)
        # 第三个部分是 MLP
        self.feed_forward = MLP(args.dim, args.dim, args.dropout)

    def forward(self, x, enc_out):
        # Layer Norm
        x = self.attention_norm_1(x)
        # 掩码自注意力
        x = x + self.mask_attention.forward(x, x, x)
        # 多头注意力
        x = self.attention_norm_2(x)
        h = x + self.attention.forward(x, enc_out, enc_out)
        # 经过前馈神经网络
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Decoder(nn.Module):
    """解码器"""

    def __init__(self, args):
        super(Decoder, self).__init__()
        # 一个 Decoder 由 N 个 Decoder Layer 组成
        self.layers = nn.ModuleList([DecoderLayer(args) for _ in range(args.n_layer)])
        self.norm = LayerNorm(args.n_embd)

    def forward(self, x, enc_out):
        print("\n[Decoder]")
        print_shape("Decoder.input", x)
        print_shape("Decoder.enc_out", enc_out)
        "Pass the input (and mask) through each layer in turn."
        for i, layer in enumerate(self.layers):
            x = layer(x, enc_out)
            print_shape(f"Decoder.layer_{i}.output", x)
        x = self.norm(x)
        print_shape("Decoder.norm_output", x)
        return x


class PositionalEncoding(nn.Module):
    """位置编码模块"""

    def __init__(self, args):
        super(PositionalEncoding, self).__init__()
        # Dropout 层
        # self.dropout = nn.Dropout(p=args.dropout)

        # block size 是序列的最大长度
        pe = torch.zeros(args.block_size, args.n_embd)
        position = torch.arange(0, args.block_size).unsqueeze(1)
        # 计算 theta
        div_term = torch.exp(
            torch.arange(0, args.n_embd, 2) * -(math.log(10000.0) / args.n_embd)
        )
        # 分别计算 sin、cos 结果
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        print("\n[PositionalEncoding]")
        print_shape("PositionalEncoding.input", x)
        # 将位置编码加到 Embedding 结果上
        pe_slice = self.pe[:, : x.size(1)].requires_grad_(False)
        print_shape("PositionalEncoding.pe_slice", pe_slice)
        x = x + pe_slice
        print_shape("PositionalEncoding.output", x)
        return x


class Transformer(nn.Module):
    """整体模型"""

    def __init__(self, args):
        super().__init__()
        # 必须输入词表大小和 block size
        assert args.vocab_size is not None
        assert args.block_size is not None
        self.args = args
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(args.vocab_size, args.n_embd),
            wpe=PositionalEncoding(args),
            drop=nn.Dropout(args.dropout),
            encoder=Encoder(args),
            decoder=Decoder(args),
        ))
        # 最后的线性层，输入是 n_embd，输出是词表大小
        self.lm_head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

        # 初始化所有的权重
        self.apply(self._init_weights)

        # 查看所有参数的数量
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    """统计所有参数的数量"""

    def get_num_params(self, non_embedding=False):
        # non_embedding: 是否统计 embedding 的参数
        n_params = sum(p.numel() for p in self.parameters())
        # 如果不统计 embedding 的参数，就减去
        if non_embedding:
            n_params -= self.transformer.wte.weight.numel()
        return n_params

    """初始化权重"""

    def _init_weights(self, module):
        # 线性层和 Embedding 层初始化为正则分布
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    """前向计算函数"""

    def forward(self, idx, targets=None):
        # 输入为 idx，维度为 (batch size, sequence length)；targets 为目标序列，用于计算 loss
        b, t = idx.size()
        assert t <= self.args.block_size, f"不能计算该序列，该序列长度为 {t}, 最大序列长度只有 {self.args.block_size}"

        # 通过 self.transformer
        # 首先将输入 idx 通过 Embedding 层，得到维度为 (batch size, sequence length, n_embd)
        print("\n[Embedding]")
        print_shape("Embedding.idx", idx)
        # 通过 Embedding 层
        tok_emb = self.transformer.wte(idx)
        print_shape("Embedding.tok_emb", tok_emb)
        # 然后通过位置编码
        pos_emb = self.transformer.wpe(tok_emb)
        # 再进行 Dropout
        x = self.transformer.drop(pos_emb)
        print_shape("Dropout.output", x)
        # 然后通过 Encoder
        enc_out = self.transformer.encoder(x)
        # 再通过 Decoder
        x = self.transformer.decoder(x, enc_out)

        if targets is not None:
            # 训练阶段，如果我们给了 targets，就计算 loss
            # 先通过最后的 Linear 层，得到维度为 (batch size, sequence length, vocab size)
            print("\n[lm_head]")
            print_shape("lm_head.input", x)
            logits = self.lm_head(x)
            print_shape("lm_head.logits", logits)
            print_shape("targets", targets)
            # 再跟 targets 计算交叉熵
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            print(f"loss: {loss.item():.6f}")
        else:
            # 推理阶段，我们只需要 logits，loss 为 None
            # 取 -1 是只取序列中的最后一个作为输出
            print("\n[lm_head]")
            lm_input = x[:, [-1], :]
            print_shape("lm_head.input last token", lm_input)
            logits = self.lm_head(lm_input)  # note: using list [-1] to preserve the time dim
            print_shape("lm_head.logits", logits)
            loss = None

        return logits, loss


def main():
    torch.manual_seed(42)
    torch.set_printoptions(linewidth=120)

    args = ModelArgs(
        n_embd=16,
        n_heads=4,
        dim=16,
        dropout=0.0,
        max_seq_len=8,
        vocab_size=32,
        block_size=8,
        n_layer=1,
    )
    print("开始 Transformer debug demo")
    print(args)

    inputs_id = torch.tensor(
        [
            [1, 5, 7, 9, 11, 13, 15, 2],
            [1, 4, 6, 8, 10, 12, 14, 2],
        ],
        dtype=torch.long,
    )
    targets = inputs_id.clone()
    print_shape("demo.inputs_id", inputs_id)
    print_shape("demo.targets", targets)

    transformer = Transformer(args)
    logits, loss = transformer.forward(inputs_id, targets=targets)

    print("\n[Debug result]")
    print_shape("final.logits", logits)
    if loss is not None:
        print(f"final.loss scalar: {loss.item():.6f}")


if __name__ == "__main__":
    main()
