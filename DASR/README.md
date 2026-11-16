# DASR

Direction-Aware Sparse Refinement，见 SDMamba-Net1114.docx 第 3.3 节、图 4。

## 原项目对应关系

- CCSA：`source/models/layers/freq_sparse.py` 中的 `CrackCandidateSparseAttention`，包含位置评分、top-k 候选选择、候选注意力、scatter 回填和残差门控。
- DEG：`source/models/layers/hog_edge.py` 中的 `HoGEdgeGateConv`、`EdgeConv`、`image2patches`、`patches2image`，包含 Sobel 方向统计和方向条件门控。
- 完整 DASR：CCSA → DEG，见 `reference/models/encoder/vss_block.py` 中的 `ccsa_deg` 分支。
- 原模型默认 candidate_ratio=0.125、min_tokens=64、max_tokens=1024、max_heads=4、nbins=180。

原始源码逐字节复制。由于 CCSA 和 FHE 原本共处 freq_sparse.py，该文件保留完整；DASR 使用其中的 CCSA 类。reference 用于核对集成位置，不是完整训练工程。

## 单独使用

安装 PyTorch 和 einops。在解压后的 DASR 文件夹运行以下代码。

```python
from pathlib import Path
import importlib.util

def load_source(relative_path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, Path(relative_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

import torch
from torch import nn
freq = load_source('source/models/layers/freq_sparse.py', 'dasr_sparse_source')
edge = load_source('source/models/layers/hog_edge.py', 'dasr_edge_source')
dasr = nn.Sequential(
    freq.CrackCandidateSparseAttention(dim=16, candidate_ratio=0.125,
        min_tokens=64, max_tokens=1024, max_heads=4),
    edge.HoGEdgeGateConv(in_dim=16, nbins=180),
).eval()
with torch.no_grad():
    output = dasr(torch.randn(1, 16, 32, 32))  # [1, 16, 32, 32]
```

原 DEG 实现按 2×2 划分子块；通道需适配原 GroupNorm，示例采用原模型的 16 通道。空间边长使用偶数且至少为 16，保证子块内有完整的 8×8 方向统计单元；原模型各阶段的 128、64、32、16 满足此条件。

原项目消融选择：构造 VSS 前设置 `MIXER_DASR_VARIANT=ccsa_deg`；只启用 DASR 时设置 `MIXER_SEMM_VARIANT=none`。不要直接导入 reference 中的 VSS：其完整工程依赖不属于这个独立结构包。

## 核验与来源

`SOURCE_MANIFEST.json` 提供路径与 SHA-256。已核对副本字节一致、Python 语法和 ZIP 完整性；打包环境未安装 PyTorch，未执行前向计算。保留原项目 LICENSE 和源码声明。
