# 简化版动态信用风险项目（model / main / tools）

该目录是基于你的 `run_model.ipynb` 与 `动态模型实施方案.md` 抽出的简化工程版，目标是：

- 结构清晰：`model.py` / `main.py` / `tools.py`
- 直接可跑：`model` 直接输入 `train_df` 和 `test_df`
- 规则明确：无地区样本自动使用全国宏观数据

---

## 目录结构

```text
simple_project/
  ├─ main.py     # 主流程入口（读取数据、切分、训练、输出结果）
  ├─ model.py    # 核心模型（baseline + GAS + B-spline + ensemble）
  ├─ tools.py    # 数据读取、宏观合并、切分、评估工具
  ├─ README.md
  └─ outputs/    # 运行后自动生成
```

---

## 运行方式

在项目根目录 `C:\Users\DELL\Desktop\本科毕业论文` 下执行：

```powershell
python -m simple_project.main
```

输出文件在：

- `simple_project/outputs/train_predictions.csv`
- `simple_project/outputs/test_predictions.csv`
- `simple_project/outputs/metrics.json`
- `simple_project/outputs/model_meta.json`

---

## 输入数据格式要求（重点）

`model.py` 的主接口为：

```python
DynamicCreditModel.fit_predict(train_df, test_df, target_col="y", time_col="time")
```

### train_df 必需列

1. `y`：二分类目标（0/1）
2. `time`：时间列（可解析为日期）
3. 特征列：至少包含以下之一
   - 借款行为特征：`x_001, x_002, ...`
   - 宏观特征：`gov_macro_XX` 或 `nat_macro_XX`

### test_df 必需列

1. `time`：时间列
2. 与训练特征同名列（缺失会自动补 NaN 再做填补）
3. `y` 可选（有则会自动计算测试指标）

### 推荐保留列（用于宏观合并）

- `region`
- `province`

---

## 无地区样本的处理规则

在 `tools.merge_macro_with_fallback` 中已实现：

1. 先按 `time + province` 合并省级宏观（`govern_data.csv`）
2. 再按 `time` 合并全国宏观（`national_data.csv`）
3. 对于没有地区/省份，或者省级宏观缺失的样本，使用全国宏观回填

全国宏观路径固定为：

`C:\Users\DELL\Desktop\本科毕业论文\data\china_globaldata\national_data.csv`

---

## 与原 notebook 的关系

- Notebook 更适合实验和展示；
- 这个简化工程更适合“可复用调用”和“后续集成到脚本/服务”。

