# Expose Your Disguise: Recovering Source Speaker Identity From Voice Conversion
This repository contains the official implementation for the ACM CCS 2026 paper "Expose Your Disguise: Recovering Source Speaker Identity From Voice Conversion".
## Dataset Acquisition
We utilize publicly available speech corpora, including LibriSpeech, VoxCeleb1, and VoxCeleb2, which are downloaded from their official sources.
- **LibriSpeech**: [https://www.openslr.org/12](https://www.openslr.org/12)
- **VoxCeleb1**: [https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html)
- **VoxCeleb2**: [https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html)
## Data Generation
We generate converted speech using several voice conversion methods.
- AGAIN-VC: A one-shot voice conversion using activation guidance and adaptive instance normalization.
- One-shot voice conversion by vector quantization.
- VQVC+: One-Shot Voice Conversion by Vector Quantization and U-Net Architecture.
- Any-to-many voice conversion with location-relative sequence-to-sequence modeling.
- FreeVC: Towards high-quality text-free one-shot voice conversion.
- Diff-HierVC: Diffusion-based Hierarchical Voice Conversion with Robust Pitch Generation and Masked Prior for Zero-shot Speaker Adaptation.
- DDDM-VC: Decoupled denoising diffusion models with disentangled representation and prior mixup for verified robust voice conversion.
### Directory Structure
Audio files should be stored under a common root, with each VC method maintaining a separate subdirectory. The structure is as follows.
```
data/
└── train(test)/
    └── {dataset}/ 
        └── {method_name}/               # e.g., Clean, FreeVC, Diff
            └── {source_speaker}-{target_speaker}/   # e.g., spk1-spk2
                └── {source_utt}-{target_utt}.wav    # e.g., uttA-uttB.wav
```
### CSV Format
`train.csv` and `test.csv` should be generated with the following fields.
| Column | Description |
|--------|-------------|
| `ID` | `{method_name}_{source_speaker}-{target_speaker}_{source_utt}-{target_utt}` |
| `wav` | Path to audio: `data/train(test)/{dataset}/{method_name}/{source_speaker}-{target_speaker}/{source_utt}-{target_utt}.wav` |
| `spk_id` | Source speaker ID |
| `lens` | Audio length |
| `VC` | VC method name (e.g., Clean, FreeVC, Diff) |
| `tar_wav` | Target ground-truth audio path: `data/train(test)/{dataset}/Clean/{target_speaker}/{target_utt}.wav`; **`none` for clean samples** |
## Model Training
`wget https://huggingface.co/datasets/Forliqr/TRIDENT_noise/resolve/main/RIRS_NOISES.zip`

`unzip RIRS_NOISES.zip -d noise/`

`python train.py train.yaml --data_folder=noise --data_parallel_backend`
## Model Evaluation
`python test.py --clean_dir=data/test/{dataset}/Clean --convert_dir=data/test/{dataset}/{method_name}`
