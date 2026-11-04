# LLM 物体位姿推理与生成：Tokenization 方案与推理策略评估

训练数据格式:

"system": "You are a scene layout assistant. Your task is to predict new object pose based on the current scene layout, geometry of objects and instruction.\n\n## Coordinate System\nX: right, Y: forward, Z: up. Ground plane: XY.\n\n## Object Pose\n- center: (X, Y, Z) in meters\n- rotation: (roll, pitch, yaw) in degrees, range [0, 360). Yaw: rotation around Z, 0° points -Y, positive counterclockwise.\n- size: (width, depth, height) in meters\n\n## Physical Constraints\n- The new object must rest on a supporting surface. Its bottom must not float in air.\n- The new object must not overlap with any existing object. Ensure geometries do not intersect.\n\n## Output Format\nReturn one list: [x, y, z, roll, pitch, yaw]\n",
"instruction": "{\"boundary\": [[0.0, 0.0], [5.0, 0.0], [5.0, 8.0], [0.0, 8.0]], \"floor_pointcloud\": \"<floor_pointcloud>\", \"scene_height\": 5.0, \"existing_objects\": [{\"id\": 0, \"category\": \"sofa\", \"center\": [4.45, 5.65, 0.56], \"rotation\": [0, 0, 270], \"size\": [2.75, 1.12, 1.12], \"object_pointcloud\": \"<object_pointcloud>\"}, {\"id\": 1, \"category\": \"table\", \"center\": [2.7, 1.35, 0.38], \"rotation\": [0, 0, 180], \"size\": [1.81, 1.05, 0.76], \"object_pointcloud\": \"<object_pointcloud>\"}, {\"id\": 2, \"category\": \"sofa\", \"center\": [2.87, 7.5, 0.71], \"rotation\": [0, 0, 0], \"size\": [1.76, 0.95, 1.43], \"object_pointcloud\": \"<object_pointcloud>\"}], \"new_object\": {\"id\": 3, \"category\": \"chair\", \"size\": [1.31, 1.16, 1.28], \"new_object_pointcloud\": \"<new_object_pointcloud>\"}, \"instruction\": \"place the chair in center region, and next to the sofa (id 0), and behind the sofa (id 0).\"}",
"output": "[2.8, 3.99, 0.64, 0, 0, 180]",

## 1. 背景与核心痛点
在使用大语言模型 (LLM) 进行 3D 物体位姿 (Pose) 推理或场景布局生成时，最直观的做法是**直接输出保留小数的数值字符串**（如 `"x: 1.25, y: -0.5"`）。然而，这种方案在实际应用中存在致命缺陷：

* **Tokenizer 碎片化：** 主流 BPE 分词器对连续数字的切分不规律（如 `1.234` 可能被切分为 `1`、`.`、`234`）。这导致输出序列变长，且模型极易在多步预测中产生“语法崩溃”（如输出 `1..25` 等乱码）。
* **高位数值突变脆弱性：** 若引入哪怕极小的随机性（Temperature > 0），采样到次优 Token 会导致坐标发生巨大跳跃。例如目标是 `0.25`，小数点后第一位错采样为 `8`，则输出变为 `0.85`，导致物体在 3D 空间中瞬间穿模或飞出边界。

因此，**强烈不建议**在生产环境或复杂生成任务中直接使用数值字符串表示位姿。

---

## 2. 核心 Tokenization 改进方案 (推荐)

为了提升空间生成的稳定性和物理一致性，建议放弃字符串，采用以下三种业内主流的 Tokenization 范式之一：

### 方案 A：均匀离散化与自定义词表 (Uniform Binning) —— 【首选标配】
将连续的空间坐标（或旋转角度）划分为 $N$ 个均匀的网格（Bins），并在 LLM 词表中强行添加 $N$ 个专属 Token（如 `<loc_0>` 到 `<loc_1023>`）。
* **优势：** * 无论数值多复杂，一个坐标维度严格只占用 1 个 Token，大幅缩短上下文。
  * 彻底杜绝数值格式化错误和乱码。
  * 概率分布连续，即使采样发生微小偏移，物理空间上也只是轻微平移，不会出现坐标飞跃。
* **适用场景：** 绝大多数室内场景布局、常规物体位姿推理。

### 方案 B：多分辨率分层表示 (Hierarchical / Coarse-to-Fine)
将连续数值拆分为“粗略位置”和“局部偏移”两个 Token（如 `<coarse_15> <fine_8>`）。
* **优势：** 仅需向词表增加极少量 Token（几十个），即可表达极高精度的连续空间。
* **适用场景：** 需要超大场景且要求极高（如毫米级）精度的工业级应用。

### 方案 C：姿态向量量化 (Pose VQ-VAE) —— 【SOTA 方案】
预先训练一个 VQ-VAE，将包含 6DoF 的位姿向量压缩成 1-2 个离散的潜变量索引（Latent Codes）。LLM 直接学习预测这些 Codebook Token。
* **优势：** 实现了真正的“降维打击”，将 6 个独立维度的预测转换为对高级姿态语义的预测，极大程度保证了物体组合（如桌椅相对位置）的物理合法性。
* **适用场景：** 追求极致布局多样性和复杂物体强关联生成的自回归任务。

---

## 3. 针对“新增 Token 丢失预训练先验”的解答

团队可能会质疑：引入 `<loc_X>` 等全新 Token，会导致 LLM 丢失在预训练阶段积累的强大数学先验（如比大小）。

**结论：对于纯空间几何与位姿任务，新增 Token “利远大于弊”。**

* **规避语义污染：** 预训练语料中的数字带有强烈的文本语义（如年份、价格、算术）。专属的 `<loc_X>` 能为模型开辟一块纯粹的“空间视觉脑区”，避免被文本逻辑干扰。
* **回归转分类：** LLM 本质是离散分类器，极不擅长连续数值回归。新增 Token 将坐标预测完美转化为 Transformer 最擅长的“多分类选择题”。

**工程优化技巧（弥补先验丢失）：**
在微调阶段，可通过以下 Trick 帮助模型快速建立空间连续性概念：
1. **智能初始化 (Smart Initialization)：** 提取预训练模型中 `0-1023` 数字的 Embedding，拷贝给新增的 `<loc_0>`-`<loc_1023>` 作为初始权重。
2. **位置编码引入 (Positional Encoding)：** 对这批新 Token 的 Embedding 叠加一维正弦/余弦位置编码，从底层注入空间顺序特征。
3. **平滑损失函数 (Label Smoothing)：** 软化交叉熵目标（如给正确位置 0.8，左右相邻位置各 0.1），强迫模型学习坐标的邻近关系。

---

## 4. 推理阶段超参数 (Temperature) 设置规范

不同的 Tokenization 方案对应截然不同的生成策略：

| 业务目标 | Tokenization 方案 | Temperature 建议 | Top-P / Top-K | 附加策略 |
| :--- | :--- | :--- | :--- | :--- |
| **极致精准、确定性推理** | 任意方案 | **0.0 (贪婪解码)** | 无需设置 | 无 |
| **布局多样性 (妥协方案)** | 原始数值字符串 | `0.4` - `0.6` | 严格截断 (`P=0.5` 或 `K=2`) | **必须**配置受限解码 (如 Regex) 兜底格式 |
| **布局多样性 (推荐方案)** | `<loc_X>` 离散化 | `0.4` - `0.8` | 常规截断 (`P=0.85-0.9`) | 后处理可加入简单碰撞检测打分 |

**总结：** 若当前基建只能输出字符串，推理必须锁死 `Temperature = 0.0`；若业务需要生成多样的场景布局（必须 `Temperature > 0`），则**架构升级为 `<loc_X>` 离散化方案是绕不开的必经之路**。