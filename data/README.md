# TRACE-SR 数据

TRACE-SR 做 query-based service recommendation：只给需求，模型推荐 API。这里有三个数据集，每个单独训练和评估。Q-ARE 已于 2026-11-14 移出（它原本的任务是从给定的候选列表里选 API，和我们的设定不一致），材料在 `../raw_material/qare_dropped/`。

这里只保留**有 API 描述的条目**：

- 样本只保留 gold API 全都有描述的。
- 每个数据集的 `apis.jsonl` 正好是这些样本用到的全部 gold API 及其描述，不含没被用到的候选 API。
- APIBench-Q 在划分前去掉了 gold 里有歧义标签的查询：标签没写类或模块（如 `split()`），而官方文档里有多个同名 API。删了哪些、为什么，见 `../raw_material/apibench_ambiguous_dropped/`。

原始数据、中间结果和预处理脚本都在 `../raw_material/`。

| 数据集 | train / dev / test | API 数 | 推荐对象 |
| --- | --- | --- | --- |
| `ProgrammableWeb/` | 4,610 / 576 / 577 | 1,212 | Web 服务，输入是 mashup 描述（DOEM 版本） |
| `APIBench_Q/java/` | 5,166 / 645 / 647 | 2,595 | JDK 等 Java 库方法 |
| `APIBench_Q/python/` | 1,911 / 238 / 240 | 660 | Python 标准库及 NumPy、pandas 等 |

划分都用 seed 42、80/10/10，并按归一化后的需求文本分组，所以三个集合的需求互不重叠。

## 文件格式

每个数据集文件夹里有 `train.jsonl`、`dev.jsonl`、`test.jsonl` 和 `apis.jsonl`。

样本（每行一条）：

```json
{"id": "apibench-java-4769", "request": "Test  input in while condition", "source": "Tutorial Websites",
 "gold_apis": [{"api_id": "java.lang.String::equalsIgnoreCase", "name": "java.lang.String.equalsIgnoreCase()"}]}
```

- `gold_apis` 只写编号和名称，描述去 `apis.jsonl` 里按 `api_id` 查。
- `source`（Stack Overflow 或 Tutorial Websites）只有 APIBench-Q 有。

API（每行一个）：

```json
{"api_id": "java.lang.String::equalsIgnoreCase", "name": "java.lang.String.equalsIgnoreCase()",
 "description": "Compares this String to another String, ignoring case considerations. …"}
```

- `name` 是 gold 标签原本的写法。
- 当原标签有拼写错误或没写类名时，会多一个 `documented_as` 字段，写明描述实际对应的 API，比如 `java.lang.String::solit` 对应 `java.lang.String.split`。

`manifest.json` 记录了各数据集的数量、过滤掉了多少，以及每个文件的 SHA-256。

## 描述来源

- **ProgrammableWeb：** 数据集自带的服务介绍；源数据为空的 10 个，从 Wayback Machine 存档的 ProgrammableWeb 页面逐字取回。
- **APIBench-Q Java：** BIKER 解析的 Java SE 8 Javadoc（已逐条对照 Oracle 官网核实），其余从官方文档补齐（Oracle、Android、OpenJFX、各项目 Javadoc）。
- **APIBench-Q Python：** 全部来自官方文档站（docs.python.org、NumPy、pandas、Django 等）。

## 使用时注意

- **ProgrammableWeb 有名称泄漏：** 约 42% 的需求文本里直接写出了 gold 服务名。逐条标记在 `raw_material/data/ProgrammableWeb/trace_sr_v1/splits/` 的 `gold_names_in_request` 字段。
- **APIBench-Q 要分开用：** Java 和 Python 是两个独立的数据集。

重新生成：在 `raw_material/` 下运行 `python -I data_prep/export_lean.py`。
