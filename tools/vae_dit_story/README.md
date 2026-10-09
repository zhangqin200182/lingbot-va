# Wan2.2 VAE + DiT 可视化讲解(单页自包含 HTML)

`vae_dit_story.html` —— 一页讲清三件事,含 4 个可拖动的交互模块:

| 幕 | 内容 | 关键画面 |
|---|---|---|
| **第一幕** | VAE:像素 ↔ `8×8×48` latent | 逐层视野 σ 从 **1.71 px → 29.16 px**(17×);相邻 latent 视野余弦 **0.9959**;48 个通道尺度差 4.79 倍 |
| **第二幕** | DiT:先验 / 后验 / 引力场 | 计数贝叶斯(38,812 → 23,213 → **0.5981**);"为什么是平均"(10 个样本碰撞);引力矢量分解;**引力场等高线图(点源 + 等值线 + 速度箭头场)**;**从密度场到速度(样本 → 各 σ 下的高斯 → 似然 → × 先验 → 权重 → 速度)**;轨迹 vs 分布;起点轴与切口 |
| **第三幕** | 条件 = 一块"透镜" | 只换先验;同一 `x_σ` 不同 `c` → 输出 **−0.20 / −1.40 / +1.80**;训练注意力掩码(384×384) |
| **第四幕** | 视频链 vs 动作链 | **两条链的时序图**(共同 KV cache 的写入/读取时刻 + 掩码实测的跨模态可见性):过程解耦(独立 noise / σ 网格 / shift / 步数 / CFG / 损失),信息**单向**耦合(动作能看本 chunk 视频,视频看不到本 chunk 动作);**4.4 统一视角**:两条链 = 同一个联合分布按「视频先行」切出的两个条件切片,各自一个引力场 |
| **第五幕** | 横向对照:6 个世界-动作模型 | **「耦合方向 × 想象与执行的关系」谱系图**(lingbot-va / FastWAM / DreamZero / Cosmos 3 / Motus2 / GlanceWAM) + **五条判据及其实证数字**(隔离 mask 71.5→**47.0**、horizon 3 s 峰值 71.6、延迟 48 ms vs 1133–3812 ms) + **哪些属于推断、哪些有实证**的诚实边界表 |

## 内嵌的 VAE 探针

第一幕里**直接嵌入了** `tools/vae_rf_explorer/vae_rf_explorer.html`(通过 `srcdoc` iframe,原文件一行未改),
所以拖动 stage、勾"叠加相邻单元/差分"、切深色都能在**这一页里**直接用。
嵌入的好处是:探针本来就没有任何外链/fetch/`localStorage`,完全可以离线跑;页面也仍然是**单文件**。

## 4 个交互模块

1. **计数 → 后验**:拖 σ 与窗口宽度,看"落在窗口里的样本比例"如何逼近真正的后验(解析 CDF 计算,不是模拟)。
2. **引力合成**:拖 σ,看每个样本的"权重 × 位移"箭头如何合成,以及方向从"朝中心"翻成"朝 +1"。
3. **分布形状随 σ 变形**:拖 σ,看边缘分布从单峰变双峰,以及粒子束如何分开。
4. **起点轴上的切口**:拖权重 π,看切口 `Φ⁻¹(1−π)` 如何平移、60 个随机起点各自去哪。

## 使用

直接用浏览器打开 `vae_dit_story.html`(图片已 base64 内嵌,可离线、可单文件分享,约 3.9 MB)。

> 图里的中文需要字体支持;`story_two_chains.py` 会自动挂载 macOS 的 `Arial Unicode.ttf`,其他系统请改脚本顶部那段 `addfont` 的路径。

## 重新生成

```bash
# 1) 生成新图(需要 matplotlib / numpy)
MPLCONFIGDIR=/tmp/mplcache python story_figs.py                 # 视野 / 引力矢量 / 有效源 / 同输入不同c / 掩码
MPLCONFIGDIR=/tmp/mplcache python story_gravity_contour.py      # 引力场等高线图(点源 + 等值线 + 箭头场)
MPLCONFIGDIR=/tmp/mplcache python story_gravity_lens.py         # 加条件(透镜)后的等高线 / 箭头 / 轨迹对比
MPLCONFIGDIR=/tmp/mplcache python story_two_chains.py           # 第四幕:视频链/动作链时序 + KV cache
MPLCONFIGDIR=/tmp/mplcache python story_wam_landscape.py        # 第五幕:六工作谱系图 + 实证数字看板
MPLCONFIGDIR=/tmp/mplcache python story_block_flow.py           # (备用)block 内部流程:训练 vs 推理
MPLCONFIGDIR=/tmp/mplcache python story_blockmask.py            # (备用)BlockMask 矩阵 + 推理注意力
MPLCONFIGDIR=/tmp/mplcache python story_clean_segments.py       # (备用)clean 段的 4×4 可见性矩阵
MPLCONFIGDIR=/tmp/mplcache python story_train_sigma.py          # (备用)两条链训练时的 σ 分布
MPLCONFIGDIR=/tmp/mplcache python story_unified_view.py         # 4.4:两个引力场 + 透镜的随机性
MPLCONFIGDIR=/tmp/mplcache python story_density_to_velocity.py  # 从密度场到速度(1 维 (x,σ) 视角)
MPLCONFIGDIR=/tmp/mplcache python story_gravity_field.py        # (可选)样本 × σ 的权重矩阵

# 2) 把图片(含本项目里已有的图)内嵌进页面,并把 VAE 探针以 srcdoc iframe 嵌进来
python build_story.py          # 读 story_template.html + ../vae_rf_explorer/vae_rf_explorer.html
                               # → 写 vae_dit_story.html
```

`build_story.py` 里 `SLOTS` 指定了每张图用哪个 PNG;`CROPS_BY_NAME` 用于裁掉旧图里的空白。

**引力场等高线图**(`story_gravity_contour.py`,浅色背景,4 个 σ 并排):
180 个样本分三团 → **每个样本是一个点源**(质量);每个 σ 下各自发出高斯 `N(x; (1−σ)x₀, σ²I)`,
按先验叠加得到**密度场**,它的**等值线**就是"引力图的等高线";
叠在上面的**箭头场**是 `v(x) = (x − E[x₀|x])/σ`;黑星 = 读数,白点+轨迹 = 一个顺着场走的粒子。
σ 从 1 → 0.15:一个同心圆包 → 三个瓣 → 三个贴着样本的小团;`max|v|` 从 2.7 → 12.9。

**加条件(透镜)后的对比**(`story_gravity_lens.py`):同一批样本、**同一个起点**,只把先验从均等改成 `0.1/0.8/0.1`。
三样东西同时变:① **等值线**(右团核心变大、圈向它鼓过去);② **箭头场**(整体摆向被加权的团);
③ **轨迹** —— 无条件落到左团 `(−1.46,−0.85)`,加透镜落到右团 `(+1.31,−0.77)`,相距 **2.77**。
起点由脚本自动搜索得到(在所有起点里挑"两种先验下终点属于不同团且分离最大"的那个)。

**"从密度场到速度"图**(`story_density_to_velocity.py`)是 1 维 `(x, σ)` 视角,把整条链拆成 4 格:

| 面板 | 内容 |
|---|---|
| (1) | `(x, σ)` 平面上的**密度场**:每个样本在每个 σ 下是一个高斯 `N(x; (1−σ)x₀, σ²)`,在世界线视角下是一条**宽度 = σ 的直线**;σ=1 时全部挤在 x=0,往下呈扇形展开 |
| (2) | 在 σ=0.35 切一刀:每个样本一条钟形(统一放大 220 倍),**在读数处的高度 = 似然**;黑线是总密度 |
| (3) | `权重 ∝ 似然 × 先验`:实心圈(先验相等)落在同一条过原点直线上;空心方块(透镜 0.9/0.1)分裂成**两条斜率 9:1 的线** —— **先验就是斜率** |
| (4) | 权重 × 方向求和 → `E[x₀]=+0.730`、`v=(x_σ−E[x₀])/σ=−1.657` |

`story_block_flow.py` / `story_blockmask.py` / `story_clean_segments.py` / `story_train_sigma.py`
是第四幕的**备用素材**(已生成 PNG 放在 `docs/`,但页面里还没用):它们分别解释
「同一个 block 在训练/推理下算什么」「BlockMask 究竟长什么样」「clean 段的四条可见性」「两条链训练时的 σ 分布」。
需要的话可以把它们补成 4.5/4.6。

`story_gravity_field.py` 是**另一种视角**(横轴 = 样本索引,纵轴 = σ,每行 = 该 σ 下的权重向量),
能更直接看到"亮带从 600 个样本收缩到一两个";两张图互补,页面里用的是前者。

## 准确性边界

- **实跑确认**:DiT 形状与参数量(meta 设备前向)、贝叶斯计数的每个数字、切口公式 `Φ⁻¹(1−π)`、轨迹不交叉/可逆、比例恒定 0.4830;
- **VAE 视野**用的是权重无关的逐路径计数线性化(attention 视为"残差 + 均匀全局混合";空间只算单帧 `T=1`);autograd 对照列用的是**随机初始化**权重 —— 真实训练权重的 ERF 需用 `tools/vae_rf_explorer/probe_rf.py --vae-path` 重跑;
- 1 维两点数据(±1)玩具模型里的常数(0.618、`1/σ²`、`1/σ³`)是两点数据特有的;真实 24,576 维的常数不同,但两条**定性机制**(判别力随 σ 暴涨、中点稳定性翻转)普遍成立。

## 许可

本目录沿用仓库根目录的 Apache-2.0(与 `tools/vae_rf_explorer/` 的 MIT 不冲突)。
