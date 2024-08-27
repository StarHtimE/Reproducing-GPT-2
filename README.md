# Reproducing-GPT-2
Reporter: 金子恒
Github Author: Andrej Kapathy  
Code: [https://github.com/karpathy/build-nanogpt](https://github.com/karpathy/build-nanogpt)
## Introduction
**关于作者**
**选择 GPT-2 (124 M) 的原因** 
## GPT-2
#### GPT
我们首先关注 GPT-2 模型的基础架构：
![[Pasted image 20240805145749.png]]
###### init 函数部分
在这一部分，我们用到了 wte, wpe, h, ln，它们分别表示 token embedding, position embedding, block, LayerNorm。此外，我们还有一个 lm head，它是一个线性层，用于将词向量转为预测的 logits，并且它与 wte 共享权重。

![[Pasted image 20240818173919.png]]
###### init_weights 部分
它对线性层和嵌入层的权重作略微不同的初始化，主要体现在:

- 线性层的初始化 std 参数可能不为 0.02 。
- 带有偏置项的线性层会初始化为零。

![[Pasted image 20240818174001.png]]

###### forward 函数部分：
输入的 $idx$（词在词表中的索引序列）是一个形状为 $(B,T)$ 的 tensor，而 $pos$ 是一维的长为 $T$ 的 tensor。对 $idx$ 进行 token embedding（每个词的索引都变为一个 $n_{\text{embd}}$ 维向量）得到 $tok_{emb}$，对 $pos$ 进行 position embedding 得到 $pos_{emb}$：
$$
\begin{matrix}
idx \\
(B,T)
\end{matrix}\xrightarrow{\mathrm{wte}}\begin{matrix}
 tok_{emb}\\
(B,T,n_{\text{embd}})
\end{matrix}\quad\quad\begin{matrix}
pos \\
T
\end{matrix}\xrightarrow{\mathrm{wpe}}\begin{matrix}
 pos_{emb}\\
(T,n_{\text{embd}})
\end{matrix}
$$
然后将 $tok_{emb}$ 和 $pos_{emb}$ 相加（事实上，我们会将 $pos_{emb}$ 复制 $B$ 份得到一个形状为 $(B,T,n_{\text{embd}})$ 的 tensor 再进行相加操作）得到 $x$，$x$ 是一个形状为 $(B,T,n_{\text{embd}})$ 的 tensor。

在 embedding 之后，$x$ 将通过 $n_{\text{layer}}$ 个 transformer 块，并额外进行一次 LayerNorm。

最后，$x$ 将通过 lm head：为了更好的理解，我们先忽略 batchsize，将 $x$ 看成是一个 $(T,n_{\text{embd}})$ 的矩阵。$x$ 将通过 lm head 转为 logits：
$$\begin{matrix}
x \\
(T,n_{\text{embd}})
\end{matrix}\xrightarrow{\text{lm\_head}}\begin{matrix}
\text{logits} \\
(T,n_{\text{vocab}})
\end{matrix}
$$
即 $x$ 从 $T$ 个时间步的词向量变成了 $T$ 个时间步的词的下一个词在词表中出现的 logits 分布。

如果模型的输入有 targets 的话，我们还要计算 loss，这里的 loss 就是一维化的 logits 和 targets 的[[交叉熵]]。

![[Pasted image 20240818174239.png]]
#### Block
接着我们来关注 Block（transformer 块）部分:

![[Pasted image 20240805153857.png]]

Block 部分包含两个 [[LayerNorm]]，一个 Attention 和一个 MLP，最后再加以残差连接，即
$$
\begin{align}
&x_{\text{temp}}=x_{\text{in}}+\mathrm{CausalSelfAttention}(\mathrm{LayerNorm}(x_{\text{in}})) \\
&x_{\text{out}}=x_{\text{temp}}+\mathrm{MLP}(\mathrm{LayerNorm}(x_{\text{temp}}))
\end{align}
$$
#### MLP
在 Block 中，我们先来关注比较简单的 MLP 部分：

![[Pasted image 20240805155259.png]]

MLP 部分由两个线性层和一个 [[GELU]] 函数组成，事实上就是
$$
\mathrm{MLP}(x)=\mathrm{GELU}(xW_{1}+b_{1})\cdot W_{2}+b_{2}
$$
这里的两个线性层略有不同，从维度上来看：
$$
(B,T,n_{\text{embd}})\xrightarrow{\text{linear 1}}(B,T,4*n_{\text{embd}})\xrightarrow{\text{GELU}}(B,T,4*n_{\text{embd}})\xrightarrow{\text{linear 2}}(B,T,n_{\text{embd}})
$$
#### Attention
然后，我们来到 CausalSelfAttention 部分，这里的 Causal 指的是我们有一个对时间步的因果遮蔽处理。我们主要关注 forward 函数部分:
$$\begin{align}
&\begin{matrix}
x \\
(B,T,n_{\text{embd}})
\end{matrix}\xrightarrow{\text{c\_attn}}\begin{matrix}
 qkv \\
(B,T,3*n_{\text{embd}})
\end{matrix}\xrightarrow{\text{split}}\begin{matrix}
q,k,v \\
(B,T,n_{\text{embd}})
\end{matrix} \\ \\

&\text{c\_attn}(x)=xW=x\cdot(W_{q},W_{k},W_{v})=(xW_{q},xW_{k},xW_{v}):=(q,k,v)
\end{align}
$$
$q,k,v$ 都是形状为 $(B,T,n_{\text{embd}})$ 的 tensor，然后对 $q,k,v$ 作相同的处理：
$$
q,k,v:(B,T,n_{\text{embd}})\xrightarrow{\mathrm{view}}\left( B,T,n_{\text{head}},\frac{n_{\text{embd}}}{n_{\text{head}}} \right)\xrightarrow{\mathrm{transpose}}\left( B,n_{\text{head}},T,\frac{n_{\text{embd}}}{n_{\text{head}}} \right)
$$
上述操作在本质上就是引入**多头注意力**。然后我们先通过
$$
att=\text{softmax}\left( \text{masked}\left( \frac{qk^{T}}{\sqrt{ n_{\text{embd}}/n_{\text{head}} }} \right) \right)
$$
计算得到注意力权重 $att$，它是一个形状为 $(B,n_{head},T,T)$ 的 tensor，我们主要关注它的后两个维度，它相当于一个 token 与 token 之间的相似度矩阵。我们点乘 $att$ 和 $v$ 即得到 $y$：
$$
\begin{align}
att\cdot v&\rightarrow y \\
(B,n_{\text{head}},T,T)\cdot(B,n_{\text{head}},T,n_{\text{embd}}/n_{\text{head}})&\rightarrow(B,n_{\text{head}},T,n_{\text{embd}}/n_{\text{head}})
\end{align}
$$
然后对 $y$ 作如下处理（连结多个 head）：
$$
y:\left( B,n_{\text{head}},T,\frac{n_{\text{embd}}}{n_{\text{head}}} \right)\xrightarrow{\mathrm{transpose}}(B,T,n_{\text{head}},\frac{n_{\text{embd}}}{n_{\text{head}}} )\xrightarrow{\text{contiguous, view}}(B,T,n_{\text{embd}})
$$
最后，$y$ 将通过一个线性层得到最终输出。

![[Pasted image 20240805165030.png]]

## Data
#### FineWeb
原始的 GPT-2 使用的是 **WebText** 数据集，它包含 45 M 个链接、总计 45 GB 的文本，但它从未公开；GPT-3 使用的数据集是一个 mix，其中 60% 是 **Common Crawl** 数据集（互联网的一个完全随机的子集，包含很多噪声），22% 是 GPT-2 使用的 WebText 数据集，剩下的部分是一些书籍和 Wikipedia 内容，这一 mix 同样从未公开过。

**FineWeb** 数据集是从 Common Crawl 数据集中过滤出的高质量数据集，它包含多个版本和子集。
[FineWeb: decanting the web for the finest text data at scale - a Hugging Face Space by HuggingFaceFW](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1)

**FineWeb_Edu** 是其中的一个子集，也是我们此次所使用的数据集。它专注于教育领域的文本数据，包含 1.3 T 个 very high educational 内容和 5.4 T 个 high educational 内容。
[HuggingFaceFW/fineweb-edu · Datasets at Hugging Face](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)

![[Pasted image 20240810013432.png]]

我们首先下载 FineWeb_Edu 数据集，我们选择的是 10 B tokens 的 sample version（这个规模已经足够接近原始 GPT-2 的性能）。

然后我们 tokenize 数据集中所有的文档，保存在 `tokens_np_uint16` 中（$2^{16}-1=65536>50257$）

接着，我们将数据分片（分割成多个小文件，方便处理），分为 1 个验证分片和 99 个训练分片（每个分片都是一个 numpy 数组，包含 100 M tokens）
#### Data Loader
然后我们来看看数据加载器：我们首先定义了一个 **load_tokens** 函数来将 numpy 数组转为 torch tensor。然后，我们定义了 DataLoaderLite 类，我们首先用 load_tokens 函数将第一个分片的 tokens 存储在 `self.tokens` 中。然后我们主要来看看 **next_batch** 函数：

我们截取当前的 `self.tokens` 中的 $B*T$ 个 tokens 作为一个 batch，存储在 buf 中；然后，我们将 buf 的第一个到倒数第二个 token 作为输入 $x$，将 buf 的第二个到最后一个 token 作为 target $y$。然后再截取下一个 batch 直至这一分片剩余的 tokens 不够一个 batch 再转到下一个分片继续上述操作。

![[Pasted image 20240811130517.png]]

## Optimization
我们使用 `torch.optim` 中的 [[AdamW]] （**Adam with Weight Decay**）优化器来进行优化，它是在 [[Adam]] 优化器的基础上增加了**权重衰减**（**weight decay**）机制。AdamW 通过结合动量和自适应学习率调整来优化模型参数，并通过权重衰减来防止过拟合。

AdamW 的更新公式为：
$$\begin{cases}
&m_{t}=\beta_{1}\cdot m_{t-1}+(1-\beta_{1})\cdot g_{t} &\text{一阶动量估计}\\ \\

&v_{t}=\beta_{2}\cdot v_{t-1}+(1-\beta_{2})\cdot g_{t}^{2} &\text{二阶动量估计}\\ \\

&\displaystyle\hat{m}_{t}=\frac{m_{t}}{1-\beta_{1}^{t}}\quad \hat{v}_{t}=\frac{v_{t}}{1-\beta_{2}^{t}} &\text{偏差校正}\\   \\


&\displaystyleθ_{t+1}= \theta_t - \eta \cdot \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \cdot \theta_t \right)&\text{模型参数更新}
\end{cases}


$$
其中：
- $\theta_t$ 是模型参数。
- $\eta$ 是学习率。
- $m_t$ 是梯度的一阶动量估计（移动平均）。
- $v_t$ 是梯度的二阶动量估计（平方的移动平均）。
- $\beta_{1},\beta_{2}$ 分别是梯度的一阶、二阶动量估计的衰减率（通常取接近 $1$ 的常数）
- $\epsilon$ 是一个小常数，用于防止除零错误。（通常取很小的数，如 $10^{-8}$）
- $\lambda$ 是权重衰减系数。

在具体操作上，我们先定义优化器：获取模型的所有参数，分为 

- **decayed** parameters：包含所有维度 $\geq 2$ 的参数，这些通常是权重矩阵（例如线性层的权重和嵌入矩阵），会应用权重衰减。
- **non-decayed** parameters：包含所有维度 $<2$ 的参数，这些通常是偏置项（biases）和层归一化参数（LayerNorm parameters），不会应用权重衰减。

将 decayed parameters 和 non-decayed parameters 组装成 optim_group，前者应用权重衰减系数 $\lambda=0.1$，后者应用权重衰减系数 $\lambda=0$。

对参数 tensor 以及参数数量计数，下面是程序运行时获取的 decayed 和 non-decayed 参数：

- Num **decayed** parameter tensors: 50, with 124,354,560 parameters
- Num **non-decayed** parameter tensors: 98, with 121,344 parameters

然后我们将优化器保存为 optimizer，使用**动态学习率**和 fused，应用 $\beta_{1}=0.9,\beta_{2}=0.95,\epsilon=10^{-8}$。

![[Pasted image 20240812160715.png]]

然后，我们来看看优化的主进程：

这里我们使用了梯度累积步数，通过在多个小批次（micro-batches）上累积梯度，模拟一个大批次的效果，从而减少显存使用。

我们从数据加载器的 next_batch 函数获取 $x,y$（分别作为 input 和 target），然后 $x,y$ 经过模型得到 logits 和 loss，由于我们使用了梯度累计，所以我们要除以梯度累积步数得到平均损失。

然后，我们通过反向传播计算损失相对于模型参数的梯度，存储在每个参数的 `.grad` 中。

最后，我们获取当前的动态学习率并启动优化器进行一次迭代。

![[Pasted image 20240812161946.png]]

此外，我们也记录了一次优化所用的时间，以此计算 $\mathrm{toekns}/\sec$。
#### Batch Size
我们设置 $\text{total-batch-size}=2^{19}=524288$，$\text{micro-batch-size}=16$（视使用的 GPU 显存而定），那么它在单个 GPU 的梯度累计的步数为
$$
\text{grad-accum-steps}=\frac{\text{total-batch-size}}{\text{micro-batch-size}*\text{sequence-length}}=32
$$
#### Learning Rate
我们设置了**学习率**的 **warmup** 机制，我们设置 warmup 步数为 $N_{\mathrm{warmup}}=715$，设置最大步数为 $N=19073$，学习率可以由下式给出：
$$
\eta(n)=\begin{cases}
\displaystyle\frac{n+1}{N_{\text{warmup}}}\cdot \eta_{\text{max}},&\text{if }n<N_{\text{warmup}}\\
\displaystyle\eta_{\text{min}}+0.5*\left( 1.0+\cos\left( \pi * \frac{n-N_{\text{warmup}}}{N-N_{\text{warmup}}} \right) \right),&\text{if }N_{\text{warmup}}\leq k\leq N \\

\end{cases}
$$

![[Pasted image 20240811141515.png]]

这里，预热步数 ws 和最大步数 ms（一个 epoch 的步数）分别通过下面两式计算得到：
$$\begin{align}
\text{warmup-steps}&= \frac{\text{warmup-tokens}}{\text{total-batch-size}}=\frac{3.75\times 10^{8}}{2^{19}}\approx 715\\
\text{max-steps}&=\frac{\text{total-tokens}}{\text{total-batch-size}}=\frac{10^{9}}{2^{19}}\approx 19073
\end{align}
$$
其中 warmup-tokens 是由 GPT-3 的论文给出的。

![[Pasted image 20240811134240.png]]
## Evaluation
在第 $250k$ step （$k=1,2,\dots,76$）时，我们会进行一些评估操作：

- 计算模型在验证集上的损失
- 计算 [[HellaSwag]] 准确率
- 生成一些样本来观察文本生成效果

下面是模型在第 10000 步时输出的验证集损失和 HellaSwag Accuracy：
	**Validation loss**: 3.1925
	**HellaSwag accuracy**: 2890/10042=0.2878
#### HellaSwag
在此，简单介绍一下 [[HellaSwag]] 数据集以及如何使用 HellaSwag 数据集评估 GPT-2，它实际上是由选择题构成。每个选择题中只有一个选项是题干文本的自然延续，而其它的选项可能没有意义或无法自然衔接上文。下面是一个选择题示例：

![[Pasted image 20240811152218.png]]

HellaSwag 的一个优点在于：即使模型规模很小，也可以看到它相比随机选择的正确率（25%）有缓慢的改进（随着 steps 的增加）。

对于比较大的模型，HellaSwag 的评估方法是直接将题干和 4 个选项全部提供给语言模型，因此模型能够在给出最佳选项之前看到所有 4 个选项，这实际上是相对简单的。但对于 GPT-2 124 M 而言，我们无法完成上述操作，下面是我们的做法：

我们首先将选择题通过 render_example 函数转化为如下的形式：
$$
\left[\begin{align} 
\big|\overline{\underline{|\quad\quad\text{context tokens}\quad\quad|}}\big|&\quad \big|\overline{\underline{|\quad\quad\text{option 1}\quad\quad|}}\big| \\
\big|\overline{\underline{|\quad\quad\text{context tokens}\quad\quad|}}\big|&\quad \big|\overline{\underline{|\quad\text{option 2}\quad|}}\big|\\
\big|\overline{\underline{|\quad\quad\text{context tokens}\quad\quad|}}\big|&\quad \big|\overline{\underline{|\quad\quad\quad*\text{option 3}*\quad\quad\quad|}}\big| \\
\big|\overline{\underline{|\quad\quad\text{context tokens}\quad\quad|}}\big|&\quad \big|\overline{\underline{|\quad\space\space\text{option 4}\quad\space\space|}}\big|
\end{align}\right]
$$
上述函数返回 tokens, mask, label。我们将 tokens 输入到模型中得到 logits，然后使用 get_most_likely_row 函数得到预测标签 pred_norm，比较预测标签 pred_norm 和真实标签 label，如果两者相等则使变量 num_correct_norm 加一，最后的准确率就是
$$
\text{accuracy}=\frac{\text{num-correct-norm}}{\text{num-total}}
$$
这里，get_most_likely_row 函数通过计算[[交叉熵]]来得到平均损失，并返回平均损失最小的一行的标签。

![[Pasted image 20240811165522.png]]
#### Samples
在第 $250k$ step （$k=1,2,\dots,76$）时，我们还会进行一些采样来观察模型当前的文本生成效果。

下面是模型在第 250 步和第 10000 步时采样的内容：

- **250 steps**:
	**Rank 0 sample 0**: Hello, I'm a language model, with a number of a great life has more like.
	In our case in the time a good way for the way
	**Rank 0 sample 1**: Hello, I'm a language model, an increase the case.
	In turn, and make more than that he would have to it. That, which he
	**Rank 0 sample 2**: Hello, I'm a language model, the most importantly, for them to develop the case of the two types of the body of the most of the same data
	**Rank 0 sample 3**: Hello, I'm a language model, or just a person.
	The way to protect the best that have no more dangerous as possible, you can also provide
	
- **10000 steps**:
	**Rank 0 sample 0**: Hello, I'm a language model, and I need to be able to model with model-based language processing? So, what is my problem, I can
	**Rank 0 sample 1**: Hello, I'm a language model, so now I'm a language model on the left I wanna go with that... I'll give it, but I don
	**Rank 0 sample 2**: Hello, I'm a language model, so I get it... (maybe it's an absolute beginner to learn it)
	What is the difference between an XML
	**Rank 0 sample 3**: Hello, I'm a language model, you would think.
	In the early 1980 s, the technology had already made rapid changes to make it easier and simple

每次采样时，我们向模型输入文本："Hello, I'm a language model,"

我们对文本 encode 并转为 torch tensor 得到 $x_{gen}$，$x_{gen}$ 经过模型得到 logits，它是一个形状为 $(B,T,n_{vocab})$ 的 tensor，我们截取最后一个词的 logits，经过 softmax 函数得到它的下一个词在词表中的概率分布 probs。我们截取概率最高的 50 个单词，在这 50 个词中按照概率分布随机采样一个词 $x_{col}$，将这个词追加到 $x_{gen}$ 的末尾得到新的 $x_{gen}$ 并进行下一轮循环直至达到设定的 max-length。

$$
\begin{align}
&\quad|\overline{\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad}| \\
&\begin{matrix}
x_{gen} \\
(B,T)
\end{matrix}\xrightarrow{\text{model}}\text{logits}\xrightarrow{\text{softmax}}\text{probabilities}\xrightarrow{\text{sample in top-k}}\begin{matrix}
x_{col} \\
(B,1)
\end{matrix}\xrightarrow{\text{cat}}\begin{matrix}
\tilde{x}_{gen} \\
(B,T+1)
\end{matrix} 
\end{align}
$$
##### Result
```
using device: cuda
total desired batch size: 524288
=> calculated gradient accumulation steps: 32
found 99 shards for split train
found 1 shards for split val
num decayed parameter tensors: 50, with 124,354,560 parameters
num non-decayed parameter tensors: 98, with 121,344 parameters
using fused AdamW: True
validation loss: 10.9499
HellaSwag accuracy: 2504/10042=0.2494
step     0 | loss: 10.955019 | lr 8.3916e-07 | norm: 15.3426 | dt: 75229.21ms | tok/sec: 6969.21
step     1 | loss: 10.902682 | lr 1.6783e-06 | norm: 14.8635 | dt: 10453.95ms | tok/sec: 50152.16
step     2 | loss: 10.803941 | lr 2.5175e-06 | norm: 14.4644 | dt: 10473.97ms | tok/sec: 50056.27
step     3 | loss: 10.662144 | lr 3.3566e-06 | norm: 12.9321 | dt: 10493.70ms | tok/sec: 49962.19
step     4 | loss: 10.518556 | lr 4.1958e-06 | norm: 10.5766 | dt: 10493.59ms | tok/sec: 49962.70
step     5 | loss: 10.376809 | lr 5.0350e-06 | norm: 8.8723 | dt: 10504.89ms | tok/sec: 49908.96
step     6 | loss: 10.257343 | lr 5.8741e-06 | norm: 7.5764 | dt: 10506.50ms | tok/sec: 49901.29
step     7 | loss: 10.146833 | lr 6.7133e-06 | norm: 6.4492 | dt: 10520.90ms | tok/sec: 49832.98
step     8 | loss: 10.035719 | lr 7.5524e-06 | norm: 5.5063 | dt: 10533.38ms | tok/sec: 49773.96
step     9 | loss: 9.961247 | lr 8.3916e-06 | norm: 4.5939 | dt: 10537.17ms | tok/sec: 49756.06
step    10 | loss: 9.875684 | lr 9.2308e-06 | norm: 3.8963 | dt: 10538.38ms | tok/sec: 49750.32

......

Step 19060 | loss: 3.002416 | lr 6.0001 e-05 | norm: 0.3227 | dt: 10608.56 ms | tok/sec: 49421.21
Step 19061 | loss: 2.960302 | lr 6.0001 e-05 | norm: 0.2934 | dt: 10615.13 ms | tok/sec: 49390.63
Step 19062 | loss: 3.019575 | lr 6.0000 e-05 | norm: 0.3494 | dt: 10603.57 ms | tok/sec: 49444.47
Step 19063 | loss: 3.008058 | lr 6.0000 e-05 | norm: 0.3256 | dt: 10613.75 ms | tok/sec: 49397.05
Step 19064 | loss: 2.947979 | lr 6.0000 e-05 | norm: 0.3366 | dt: 10612.72 ms | tok/sec: 49401.85
Step 19065 | loss: 2.971504 | lr 6.0000 e-05 | norm: 0.3003 | dt: 10609.08 ms | tok/sec: 49418.81
Step 19066 | loss: 3.000040 | lr 6.0000 e-05 | norm: 0.3162 | dt: 10615.91 ms | tok/sec: 49387.02
Step 19067 | loss: 2.977490 | lr 6.0000 e-05 | norm: 0.3062 | dt: 10616.99 ms | tok/sec: 49381.97
Step 19068 | loss: 2.984937 | lr 6.0000 e-05 | norm: 0.2981 | dt: 10608.30 ms | tok/sec: 49422.44
Step 19069 | loss: 3.001659 | lr 6.0000 e-05 | norm: 0.3083 | dt: 10615.16 ms | tok/sec: 49390.49
Step 19070 | loss: 2.983717 | lr 6.0000 e-05 | norm: 0.3110 | dt: 10610.45 ms | tok/sec: 49412.43
Step 19071 | loss: 3.073220 | lr 6.0000 e-05 | norm: 0.3302 | dt: 10616.20 ms | tok/sec: 49385.63
Validation loss: 3.0331
HellaSwag accuracy: 3052/10042=0.3039
```

在 19073 steps，大约 10 B tokens 后，模型的验证集损失和 HellaSwag accuracy 分别定格在了 $3.0331$ 和 $0.3039$。

从**验证集损失**来看，复现的 GPT-2 124 M 超越了 OpenAI GPT-2 124 M。由于两者的数据集分布非常不同，因此这一对比并不公平，但仍然可以作为一个不错的交叉检查。

从 **HellaSwag accuracy** 来看，我们超越了随机选择的 $0.25$ 正确率和 OpenAI GPT-2 124 M 的 $0.2955$ 正确率，这是一个标准化的测试，所以具有不错的参考性。事实上，复现的 GPT-2 124 M 使用了 10 B tokens 用于训练，而 OpenAI GPT-2 124 M 使用了 100 B tokens 用于训练，即我们使用明显更少的 tokens 实现了超越 OpenAI GPT-2 124 M 的准确率。这可能有以下几点原因：

- OpenAI GPT-2 124 M 是在更广泛的数据分布上进行训练，广泛性主要体现在多语言以及它包含很多数学、代码内容，这些内容无法体现在 HellaSwag accuracy 上。而我们使用的 FineWebEdu 是一个纯英语的教育内容数据集，它更加匹配 HellaSwag 的测试。
- 另一方面，HellaSwag 是一个比较早的数据集，它可能以某种方式进入了 FineWebEdu，造成数据集污染。

HellaSwag 的对比并不完全严谨，但仍然具有很好的参考意义，它至少证明我们复现的效果相当不错。

- **Rank 0 sample 0**: Hello, I'm a language model, and I am a modeler and my design thinking is mostly based on human factors. And I want my business to look
- **Rank 0 sample 1**: Hello, I'm a language model, so a lot of my life, for example. One question that has only one answer: why do our students become fluent
- **Rank 0 sample 2**: Hello, I'm a language model, so I won't pretend to be someone who loves to talk a word of the day, but I'm a linguist
- **Rank 0 sample 3**: Hello, I'm a language model, but to be honest, I don't really support the idea that "human languages" are simply "man's best,"

从最后采样的一组文本来看，GPT-2 124 M 已经能够自然地衔接我们的输入（**“Hello, I'm a language model,”**）完成输出。相比刚开始近乎随机的输出，这一表现相当不错。

![[Pasted image 20240812010443.png]]

我们对 loss 和 HellaSwag accuracy 进行了可视化，可以看到随着模型的迭代，验证集损失不断下降，HellaSwag 波动上升。

Train loss 在 2500~5500 steps 出现了异常，这可能是因为 FineWebEdu 的样本存在某种顺序，没有被完全正确打乱。可能可以通过在训练前重新打乱样本来解决这个问题。
## Summary
在复现的过程中，我一步步地深入 GPT-2 的架构，对 GPT-2 的每个部分的输入输出都有了系统的认识。从维度上来说，这个过程就是
$$
(B,T)\xrightarrow{\text{embedding}}(B,T,n_{\text{embd}})\xrightarrow{\text{model}}(B,T,n_{\text{{vocab}}})
$$
具体来说，
- 我们首先通过 tiktoken 库将文本 token 化并转为词在词表中的索引序列并将所有数据写入 npy 文件；
- 我们通过 [[#Data Loader]] 得到一个 batch 的索引序列 idx，形状为 $(B,T)$；
- idx 通过 embedding 层，将每个索引转为一个 $n_{\text{embd}}$ 维向量，得到输入 $x$，形状为 $(B,T,n_{\text{embd}})$；
- $x$ 通过 $n_{\text{layer}}$ 个 transformer 块的处理，通过 attention 层建模单词之间的依赖关系，通过 MLP 层进一步变换和特征提取；
- 最后 $x$ 通过一个线性层转为 logits，形状为 $(B,T,n_{\text{vocab}})$；
	- 如果在训练阶段，则我们还要计算损失，模型将返回 logits 和 loss 给优化器，通过反向传播计算梯度，然后进行一次迭代。
	- 如果我们需要生成文本，则我们截取 logits 中最后一个时间步的数据，形状为 $(B,n_{\text{vocab}})$，然后对其进行 softmax 得到概率分布，在概率分布中随机采样，将该索引接在 $x$ 后面重新进入模型直至达到停止条件。最后通过 tiktoken 库 decode 得到生成的文本。

通过上述的复现过程，可以看出 GPT-2 具有不错的**文本生成能力**，在生成连贯、流畅且上下文相关的文本方面表现出色；

HellaSwag 测试则证明 GPT-2 具有一定的**零样本能力**（**zero-shot**），能够在没有特定任务训练的情况下执行一些 NLP 任务。

除此之外，我也简单尝试了通过 tinyshakespeare 微调 GPT-2 模型生成 shakespeare 风格文本的任务 （[[llmc]]），它证明 GPT-2 具有一定的**迁移学习能力**。
	Being barren savour, grant
	Everyone, every man and woman,
	Is heir to his life in unwritten words:
	The Salmon of the family
	How would gentlerish words indeed bring them
	Were men so much more the meanest of the heart,
	That a man feared all, but that

在复现的过程中，我对模型产生了许多的疑惑，包括：
- Attention 层和 MLP 层在 transformer 块中分别起到什么样的作用？
- 为什么要使用不同的权重矩阵得到 QKV？
- MLP 层使用了两个线性层，为什么第一个线性层提升了 $x$ 的维度，第二个线性层又把 $x$ 的维度降回原大小？
- 为什么 GPT-2 中 LayerNorm 的位置与原始 transformer （[1706.03762 (arxiv.org)](https://arxiv.org/pdf/1706.03762)）有所区别？
- 为什么要使用学习率预热机制？

正是这些疑问使我进一步探索。我通过查询各种资料，理解了这些问题的答案，从而获得了对 GPT-2 以及 transformer 的更进一步的理解。([[GPT-2 相关问题]])

## Possible improvements

- 尝试进行**多个 epoch** 的训练，观察模型的极限能力（HellaSwag accuracy）。
- 尝试新的动态学习率机制, 例如 **Linear Cooldown** 或更一般的版本： $$
\eta(n)=\begin{cases}
\frac{n+1}{N_{\text{warmup}}}\cdot \eta_{\text{max}},&\text{if }n<N_{\text{warmup}} \\
\eta_{\text{max}},&\text{if }N_{\text{warmup}}\leq n\leq N-N_{\text{decay}} \\
f(n,N,N_{\text{decay}}),&\text{if }n>N-N_{\text{decay}}
\end{cases}
$$ 这里 $f$ 关于 $n$ 单调递减。![[Pasted image 20240813163005.png]] 参考 [2405.18392 (arxiv.org)](https://arxiv.org/pdf/2405.18392)

- FineWebEdu 似乎造成了 2500~5500 steps 的训练集损失上的异常，可以考虑人工打乱数据集中的样本顺序或使用 FineWeb 的 classic 版本再次训练。

