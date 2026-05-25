# 09 — 数据管线

## 1. Manifest 格式（Parquet）

列（与 V2 兼容并扩展）：

| 列 | 类型 | 说明 |
|----|------|------|
| `Z` | int | 原子序数 |
| `ion_charge` | int | 电离度 |
| `nele` | int | 电子数 |
| `parent_config` | str | 如 `1s2 2p1` |
| `level_config` | str | CSF 组态 |
| `J` | float | 总角动量 |
| `parity` | int | 0/1 |
| `term` | str | 光谱项 |
| `level_eV` | float | **相对基态** 激发能 |
| `nist_uncertainty` | float | 可选 |
| `has_nist_level` | bool | 显式标记；否则由 `isfinite(level_eV)` 推导 `nist_mask` |

V3 新增：

| 列 | 说明 |
|----|------|
| `parent_config_id` | int32 hash |
| `csf_id` | 组内 CSF 索引 |
| `leading_percent` | FAC/NIST 参考（可选） |

## 2. 脚本

```text
scripts/v3_prepare_hydrogenic.py       # Z=1..10, n<=5, 与 V2 语义对齐
scripts/v3_prepare_nist_subset.py      # Z<=26, 闭壳层优先
scripts/v3_build_racah_cache.py        # 角向系数
scripts/v3_pack_batches.py             # 可选 TFRecord/npy
```

### 2.1 类氢 manifest

```text
每 (Z, ion) 一行 parent = "1s1"
多行 level：n=1..5 不同主量子数（与 V1.3 文档一致：Phase1 是不同 n，不是同组态精细结构）
```

### 2.2 NIST manifest

从 V2 `nist_loader` **复制逻辑**（PyTorch 脚本写 parquet），训练读 parquet（JAX）。

**行级样本**（继承 V1.3 方案 A）：`__len__ = nrow`，每行一个 `E_target` 标量。

## 3. DataLoader（JAX）

```python
class NumpyIterableDataset:
  def __iter__(self):
      for batch in batched_indices(...):
          yield collate_fn(rows)

# 或 grain / tensorflow_datasets，第一版用简单 numpy 迭代即可
```

`collate_fn` 输出 `01_architecture.md` 中的 batch 字典。

## 4. `config_parser.py`

从 `level_config` / `parent_config` 字符串解析：

```python
@dataclass
class Shell:
    n: int
    l: int
    twice_j: int
    occ: int

def parse_config_string(s: str) -> list[Shell]: ...
def shells_to_kappa_occ(shells) -> tuple[kappa, omega]: ...
```

**从 V2 复制** 后改为纯 Python（无 torch）。

## 5. 数据目录

```text
data_raw/nist/...
data_cache/manifest_hydrogenic_v3.parquet
data_cache/manifest_nist_v3.parquet
data_cache/racah_cache_v3.npz
```

## 6. 测试

- `test_config_parser_fe17`：解析复杂组态
- `test_collate_batch_shapes`
- `test_nist_loader_relative_energy`：最低能级为 0
- `test_nist_mask_false_for_missing_levels`：无 NIST 行 → `nist_mask=False`，`E_nist=NaN`
