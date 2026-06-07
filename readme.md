## Test-time Domain Generalization for Image Super-resolution (ICLR2026)

### [[Paper](https://openreview.net/pdf?id=jBuMH3DOPQ)] 



Zaizuo Tang<sup>1</sup>, Yubin Yang<sup>1</sup>, 

<sup>1</sup>State Key Laboratory for Novel Software Technology, Nanjing University, Nanjing, China<br>

## Abstract
Test-time domain generalization (TTDG) methods enhance the performance of neural networks on target domains by transferring the feature distribution of target samples to approximate that of the source domain, while avoiding the computational cost associated with fine-tuning on the target domain. However, existing TTDG methods primarily rely on style transfer strategies operating at a coarse granularity, which prove ineffective for pixel-level prediction tasks such as image super-resolution (SR). To address this limitation, we propose a multi-codebook based test-time domain generalization framework (MC-TTDG). Our method leverages both domain-specific and domain-invariant codebooks to achieve fine-grained representation learning on source domains, and performs pixel-level nearest-neighbor feature matching and transfer to accurately adjust target domain features. Furthermore, we introduce a voting-based strategy for optimal domain-specific codebook selection, which improves the precision of feature transfer through multi-party consensus. Extensive experiments across diverse data distributions, and network architectures demonstrate that the proposed method effectively transfers feature distributions for SR networks. Our code is available at https://github.com/ZaizuoTang/MC-TTDG.


### Train on SR

1. Download the DRealSR dataset.

    [[DRealSR Dataset](https://github.com/xiezw5/Component-Divide-and-Conquer-for-Real-World-Image-Super-Resolution)]

2. Start training:

    #### Train_stage1: 
    Train the encoder and decoder.

    python Stage1/basicsr/train.py -opt Stage1/options/train/train_MambaIR_SR_x4.yml

    #### Train_stage2: 
    Train multiple codebooks.

    python Stage2/basicsr/train.py -opt Stage2/options/train/train_MambaIR_SR_x4.yml

    #### Train_stage3: 
    Train the classification network.

    python Stage3/basicsr/train.py -opt Stage3/options/train/train_MambaIR_SR_x4.yml


### Test on SR

    python Stage3/basicsr/test.py -opt Stage3/options/test/test_MambaIR_SR_x4.yml


#### [[Weight](https://drive.google.com/drive/folders/1OQSKeI5SlOh9hSmu60d2CprlvMy5XdE8?usp=drive_link)]
    
    

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Acknowledgement

This code is based on [BasicSR](https://github.com/XPixelGroup/BasicSR) and [MambaIR](https://github.com/csguoh/MambaIR). Thanks for their awesome work.

## Contact

If you have any questions, feel free to approach me at tangzz@smail.nju.edu.cn
