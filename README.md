# SDMamba-Net

Code modules and the CrackMix dataset accompanying SDMamba-Net. The module descriptions follow Sections 3.2–3.4 and Figures 3–5 of the manuscript `SDMamba-Net1114.docx`.

## Code modules

| Module | Description | Source | Download | Chinese manual |
| --- | --- | --- | --- | --- |
| SEMM | Spectral-Enhanced Mamba Mixing | [SEMM](SEMM/) | [SEMM.zip](SEMM.zip) | [说明书](SEMM_说明书.txt) |
| DASR | Direction-Aware Sparse Refinement | [DASR](DASR/) | [DASR.zip](DASR.zip) | [说明书](DASR_说明书.txt) |
| MCFBR | Multi-scale Channel Fusion and Boundary Refinement Decoder | [MCFBR](MCFBR/) | [MCFBR.zip](MCFBR.zip) | [说明书](MCFBR_说明书.txt) |

Each module directory includes original implementation files under `source/`, integration references under `reference/`, a README with usage examples, requirements, and a SHA-256 source manifest. Original implementation names such as `TransMixer` and `SRFModule` are retained. These packages contain individual structures and integration references; a complete end-to-end training project and pretrained weights are not included.

## CrackMix dataset

Download [CrackMix.zip](CrackMix.zip) and extract it to obtain the `CrackMix/` directory. See [CrackMix 数据集说明书](CrackMix_数据集说明书.txt) for file formats, image sizes, pairing rules, preprocessing, and experimental use.

CrackMix combines 537 images from DeepCrack and 108 images from CrackMap. It corresponds to the manuscript's 645-image mixed dataset.

| Split | Images | Masks | DeepCrack images | CrackMap images |
| --- | ---: | ---: | ---: | ---: |
| Training | 384 | 384 | 300 | 84 |
| Test | 261 | 261 | 237 | 24 |
| Total | 645 | 645 | 537 | 108 |

```text
CrackMix/
  train_img/
  train_lab/
  test_img/
  test_lab/
  README.md
  FILE_MANIFEST.csv
```

Images are RGB; masks are grayscale PNG images with values 0 (background) and 255 (crack). Image–mask pairs share their filename stem. Original resolutions and train/test splits are preserved. The dataset is stored as a ZIP in this repository.

## Verification and attribution

[ARCHIVE_SHA256.json](ARCHIVE_SHA256.json) records archive sizes and SHA-256 checksums. Source manifests are provided within the module packages, and a per-file data manifest is included in CrackMix.zip. Copied files and ZIP integrity have been checked. Model forward passes and training were not run during packaging.

The code retains the original project's [LICENSE](LICENSE) and source copyright notices. Dataset usage remains subject to the original DeepCrack and CrackMap terms; the code license does not grant additional dataset rights.
