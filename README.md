# Signal peptide prediction 

这个命令行工具基于 SignalP 6.0 论文独立实现。输入蛋白质 FASTA，输出 `OTHER`、`Sec/SPI`、`Sec/SPII`、`Tat/SPI`、`Tat/SPII`、`Sec/SPIII` 六类概率，以及所选类型的切割位点和信号肽分区。

## 方法

- 编码器使用冻结的 ESM-2 8M，输入最多 70 个 N 端残基。训练集中的逐位标签提供信号肽和成熟蛋白的边界。
- CRF 为每个类别限定可达的状态和状态顺序。训练对具有歧义的区域边界使用多标签掩码。模型从每类合法路径的总概率得到类别概率，从最可能的路径得到分区和切割位点。切割概率和可选的逐残基概率均以所选类别为条件。
- 保留 SignalP 数据的分区，剔除跨分区完全相同的模型输入。默认用第 1 折训练、第 2 折选择检查点、第 0 折测试。训练报告包含逐类 MCC1、MCC2、精确率、召回率、切割位点指标和 N 端跨膜负样本误报数。
- 该实现的编码器、区域弱标签和训练策略与论文发行版有差异。输出格式和概率的详细含义见下文。区域弱标签源自序列和标注规则，报告中的切割指标使用数据集给出的边界。

## 安装和获取数据

推荐 Python 3.12。在项目目录运行以下命令。

```sh
uv venv --python 3.12
uv pip install -e .
mkdir -p .local-data/esm2
curl -fL https://raw.githubusercontent.com/fteufel/signalp-6.0/main/data/train_set.fasta \
  -o .local-data/train_set.fasta
for file in config.json model.safetensors vocab.txt tokenizer_config.json special_tokens_map.json; do
  curl -fL "https://huggingface.co/biohub/esm2_t6_8M_UR50D/resolve/main/$file" \
    -o ".local-data/esm2/$file" || break
done
```

训练数据和编码器来自各自的发布方。运行时使用本地文件，训练和推理无需联网。检查点和报告记录训练数据及编码器文件的 SHA-256，推理时检查编码器文件与所用检查点一致。 [SignalP 数据与研究代码](https://github.com/fteufel/signalp-6.0/tree/main/data) · [ESM-2 模型卡](https://huggingface.co/biohub/esm2_t6_8M_UR50D)

## 训练与预测

```sh
.venv/bin/python -m sp_repro train \
  --data .local-data/train_set.fasta --out runs/esm2-crf \
  --cache .cache/esm2-features.pt --epochs 8

.venv/bin/python -m sp_repro predict \
  --fasta input.fasta --model runs/esm2-crf/model.pt \
  --organism other --out runs/my-predictions --per-residue
```

若序列来自真核生物，用 `--organism eukarya`，模型的候选类别为 `Sec/SPI` 和 `OTHER`。`other` 允许全部六类。训练和预测请各自指定新输出目录。项目已有 `runs/esm2-crf/model.pt` 时，可直接执行预测命令。`--per-residue` 可省略，以减少大批量任务的输出体积。

| 文件 | 内容 |
| --- | --- |
| `prediction_results.txt` | 每条序列的预测类型、六类概率、切割边界及其条件概率 |
| `output.json` | 同一批次的结构化结果、模型标识和可选逐残基概率 |
| `output.gff3` | 整段预测信号肽的坐标 |
| `region_output.gff3` | 对应类别可用的信号肽分区坐标 |
| `processed_entries.fasta` | 对预测有信号肽的条目，输出去除该片段后剩余的全长序列 |

坐标从 1 开始。切割 `23–24` 表示在第 23 和第 24 个残基之间切割。GFF3 的坐标两端均包含在内。`Sec/SPIII` 输出整段信号肽，SPII 类别还可报告切割后首位残基为 C 时的坐标。JSON 字段和置信度语义以本实现定义为准。

## 本次复现实验

在原始数据的三个分区上轮换训练、验证和测试。剔除 3 条跨分区的相同输入后，共获得 20,287 条留出预测。合并留出预测后，五类信号肽的宏平均 MCC2 为 **0.754**，六类准确率为 **95.41%**。

| 类型 | 测试记录 | MCC2 | 切割位点精确召回率 |
| --- | ---: | ---: | ---: |
| Sec/SPI | 2,582 | 0.851 | 71.22% |
| Sec/SPII | 1,613 | 0.887 | 91.51% |
| Tat/SPI | 365 | 0.811 | 54.52% |
| Tat/SPII | 32 | 0.273 | 53.13% |
| Sec/SPIII | 70 | 0.946 | 97.14% |

切割召回要求类别和坐标都正确。`Tat/SPII` 有 119 个预测，其中 17 个类别和切割坐标均正确。N 端前 40 个残基内包含跨膜标注的 1,840 条负样本中，有 218 条被判为信号肽。以上结果用于描述此基线在原论文数据内的表现，外部数据验证和与 DTU 发行版的逐条对照尚待开展。

本次汇总保存在 `runs/cross-validation.json`，分折报告位于 `runs/esm2-crf*/metrics.json`。其余两折可在训练命令中分别设置 `--test-fold 1 --validation-fold 0` 或 `--test-fold 2 --validation-fold 1`，并选择不同的输出目录。

运行算法测试的命令为 `.venv/bin/python -m unittest discover -s tests -v`。

参考文献为 Teufel et al., *Nature Biotechnology* 40, 1023–1025 (2022)。 [原始论文](https://www.nature.com/articles/s41587-021-01156-3) · [DTU SignalP 文档](https://services.healthtech.dtu.dk/services/SignalP-6.0/)

## 许可证

本仓库独立实现的代码及配套文档采用 Apache License 2.0。 [许可证全文](LICENSE)

SignalP 6.0 原始研究代码、第三方数据与 ESM‑2 权重分别适用其发布方的条款。训练生成的检查点保存在 `runs/`，与源码分别管理。 [SignalP 6.0 许可证](https://github.com/fteufel/signalp-6.0/blob/main/LICENSE) · [SignalP 6.0 数据页面](https://services.healthtech.dtu.dk/services/SignalP-6.0/) · [ESM‑2 模型卡](https://huggingface.co/biohub/esm2_t6_8M_UR50D)
