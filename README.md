# SD-Fuse
(2025' Information Fusion) This is the offical implementation for the paper titled "SD-Fuse: An image structure-driven model for multi-focus image fusion". https://doi.org/10.1016/j.inffus.2025.104058

### Checkpoints

PyTorch model weights are available on Google Drive. The current release is based on a traditional Sobel extractor; additional models will be released in future updates.

- Download: [https://drive.google.com/file/d/1NrZi1GSvMPKtOHion3RPSoxq1iv8DXhE/view?usp=sharing](https://drive.google.com/file/d/1NrZi1GSvMPKtOHion3RPSoxq1iv8DXhE/view?usp=sharing)

### Test

Using the MFI-WHU dataset as an example. Run the following command to test the model:

--near_dir`: near-focus images (MFI-WHU source_1)  

--far_dir`: far-focus images (MFI-WHU source_2)  

--ckpt`: path to the '.pt' weights  

--out_fuse_dir`: output directory for fused results

```bash
python test.py --near_dir ./testdata/MFI-WHU/source_1 --far_dir ./testdata/MFI-WHU/source_2 --ckpt ./sobel_best.pt --out_fuse_dir ./outputs_MFIWHU_test/fuse
