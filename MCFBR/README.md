# MCFBR

Multi-scale Channel Fusion and Boundary Refinement Decoder，见 SDMamba-Net1114.docx 第 3.4 节、图 5。

## 原项目对应关系

- 主实现：`source/models/decoder/SRF.py` 中的 `SRFModule`。
- 浅层引导融合：`MLP`、`Conv2d_GN`、`SRModule`。
- 多尺度卷积：`MSDWConv`（膨胀率 1、2、3）。
- 边界细化：`BoundaryRefine`；预测头：`linear_pred`。
- 原模型调用见 `reference/models/segmentor/MixerCSeg.py`。reference 仅用于核对集成位置，不是完整训练工程。

`source` 和 `reference` 内的代码均按原文件逐字节复制，没有改名或改写类。完整 MCFBR 必须在构造解码器前设置 `MIXER_DECODER_VARIANT=full`，以免继承外部消融设置。

## 单独使用

安装 PyTorch。在解压后的 MCFBR 文件夹运行以下代码。使用文件加载避免触发原项目 models 包的其他依赖。

```python
from pathlib import Path
import importlib.util

def load_source(relative_path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, Path(relative_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

import os
import torch
os.environ['MIXER_DECODER_VARIANT'] = 'full'
srf = load_source('source/models/decoder/SRF.py', 'mcfbr_source')
decoder = srf.SRFModule([16, 32, 64, 128], mid_dim=8, size=(512, 512)).eval()
features = [torch.randn(1, c, s, s) for c, s in
            [(16, 128), (32, 64), (64, 32), (128, 16)]]
with torch.no_grad():
    prediction = decoder(features)  # [1, 1, 512, 512]，未经过 sigmoid 的输出
```

## 核验与来源

`SOURCE_MANIFEST.json` 提供原文件路径与 SHA-256。已核对副本字节一致、Python 语法和 ZIP 完整性；打包环境未安装 PyTorch，未执行前向计算。保留原项目 LICENSE；该许可文件不代表数据集的许可。
