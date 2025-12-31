# SD-Fuse
(2026' InF) This is the offical implementation for the paper titled "SD-Fuse: An image structure-driven model for multi-focus image fusion".

### Model Weights

PyTorch model weights (.pt) are available on Google Drive. The current release is based on a traditional Sobel feature extractor; additional models will be released in future updates.

- Download: [https://drive.google.com/file/d/1NrZi1GSvMPKtOHion3RPSoxq1iv8DXhE/view?usp=sharing](https://drive.google.com/file/d/1NrZi1GSvMPKtOHion3RPSoxq1iv8DXhE/view?usp=sharing)

### Test

Run the following command to test the model:

```bash
python test.py --near_dir ./testdata/MFI-WHU/source_1 --far_dir ./testdata/MFI-WHU/source_2 --ckpt ./sobel_best.pt --out_fuse_dir ./outputs_MFIWHU_test/fuse
