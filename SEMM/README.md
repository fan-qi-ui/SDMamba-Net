# SEMM

Spectral-Enhanced Mamba Mixing，见 SDMamba-Net1114.docx 第 3.2 节、图 3。

## 原项目对应关系

- FHE：`source/models/layers/freq_sparse.py` 中的 `FrequencyHighEnhancement`。
- 四方向选择扫描和步长引导的全局/局部混合：`source/VMamba/models/vmamba.py` 中的 `SS2D`、`SS2Dv2.forward_corev2`。
- 输出门控、残差和 FFN：同文件中的 `TransMixer`。
- 完整 SEMM 组合：FHE → TransMixer，见 `reference/models/encoder/vss_block.py` 的 `fhe_mamba_gl` 分支。
- 四阶段通道为 [16, 32, 64, 128]，state_dim 为 [8, 8, 16, 16]，每阶段 depth=1、mlp_ratio=2，见 reference 中的模型构造函数。

原始源码逐字节复制。由于 FHE 和 CCSA 原本共处 freq_sparse.py，该文件保留完整；SEMM 使用其中的 FHE 类。附带 VMamba 的 Python 依赖源码和可选 CUDA 扩展构建源码，没有复制编译产物、缓存、权重或数据集。reference 仅用于核对集成位置，不是完整训练工程。

## 单独使用

安装兼容的 PyTorch 及 requirements.txt 中的依赖。在解压后的 SEMM 文件夹运行以下代码。

```python
from pathlib import Path
import importlib.util

def load_source(relative_path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, Path(relative_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

import sys
sys.path.insert(0, str(Path('source').resolve()))
import torch
from torch import nn
from VMamba.models.vmamba import TransMixer
freq = load_source('source/models/layers/freq_sparse.py', 'semm_frequency_source')
semm = nn.Sequential(
    freq.FrequencyHighEnhancement(dim=16, cutoff=0.18, temperature=18.0),
    TransMixer(hidden_dim=16, ssm_d_state=8, mlp_ratio=2.0,
               channel_first=True, use_global_local=True),
).eval()
# 先以较小特征图验证。全局注意力的内存开销随空间 token 数增加。
with torch.no_grad():
    output = semm(torch.randn(1, 16, 16, 16))  # [1, 16, 16, 16]
```

原项目消融选择：构造 VSS 前设置 `MIXER_SEMM_VARIANT=fhe_mamba_gl`；若只启用 SEMM，设置 `MIXER_DASR_VARIANT=none`。不要直接导入 reference 中的 VSS：其完整工程依赖不属于这个独立结构包。

Triton 和 selective_scan CUDA 扩展为原实现的可选加速依赖。可选扩展构建文件位于 `source/VMamba/models/kernels/selective_scan/`；未安装时原源码提供 PyTorch 回退。使用 CUDA 扩展前须匹配本机 PyTorch/CUDA 工具链。

## 核验与来源

`SOURCE_MANIFEST.json` 提供路径与 SHA-256。已核对副本字节一致、Python 语法和 ZIP 完整性；打包环境未安装 PyTorch，未执行前向计算。保留原项目 LICENSE 和源码版权声明；来源说明见原项目 MixerCSeg、VMamba 及相关依赖。
