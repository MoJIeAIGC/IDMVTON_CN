
<div align="center">
<h1>IDM-VTON: Improving Diffusion Models for Authentic Virtual Try-on in the Wild</h1>

<a href='https://idm-vton.github.io'><img src='https://img.shields.io/badge/Project-Page-green'></a>
<a href='https://arxiv.org/abs/2403.05139'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
<a href='https://huggingface.co/spaces/yisol/IDM-VTON'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-yellow'></a>
<a href='https://huggingface.co/yisol/IDM-VTON'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a>

</div>
## 上面是项目原信息地址

<h1 align="center">摩诘AI-VTON优化汉化版</h1>


Star ⭐ us if you like it!

---
### 本版本对原项目idm-VTON界面做了汉化，优化了原有的内置模特和服装，用SD重做了新模特，增加了一键安装功能和一键运行功能,category成衣类型选项，随机种子和生成数量可选。
### VTON可以对服装细节做很好的还原保留，对比目前其他同类模型OOTD等，对人物姿势动作识别以及服装还原效果更好。不过在使用上会有很多技巧，详情可以参考教程。

![teaser2](IDM-VTON/assets/teaser2.png)&nbsp;
![teaser](IDM-VTON/assets/teaser.png)&nbsp;
![teaser](IDM-VTON/assets/teaser3.png)&nbsp;


## install 如何安装
安装前需要你已在安装了python3.10.11或3.10.13 和C++生成工具

```
git clone https://github.com/MoJIeAIGC/IDMVTON_CN.git

运行 windows_install.bat 即可

```
windows_install.bat 会自动安装所有依赖和库，需自备梯子
不建议从清华源或阿里源安装，部分依赖版本对不上。 
 
## 模型下载
姿势动作人体模型请从此处下载：[ckpt](https://huggingface.co/yisol/IDM-VTON/tree/main).
* 所有文件都要下载
* 下载好后放入ckpt文件夹下。
```
ckpt
|-- densepose
    |-- model_final_162be9.pkl
|-- humanparsing
    |-- parsing_atr.onnx
    |-- parsing_lip.onnx

|-- openpose
    |-- ckpts
        |-- body_pose_model.pth
    
```

## 如何运行
安装好所有依赖后，运行
```
windows_run.bat
```
第一次运行请根据自己的显存情况进行填写相应的数字。
如果填写错误或想要重新选择，

请删除IDM-VTON目录下的batconfig.txt文件

![teaser](IDM-VTON/assets/t4.png)&nbsp;

或者也可以自行运行:

```
python IDM-VTON/app_VTON.py
```
首次运行会自动从huggingface下载[yisol/IDM-VTON](https://huggingface.co/yisol/IDM-VTON).
请自备梯子或镜像耐心等待安装。

## 教程
### 安装和使用教程：[B站摩诘AI](https://www.bilibili.com/video/BV1VF8Ge4EZh/?vd_source=25d3add966daa64cbb811354319ec18d)


## 演示地址

汉化版没有部署云端演示地址。
可查看原版huggingface上的演示地址 [demo](https://huggingface.co/spaces/yisol/IDM-VTON)

---

## Acknowledgements

Thanks [ZeroGPU](https://huggingface.co/zero-gpu-explorers) for providing free GPU.

Thanks [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter) for base codes.

Thanks [OOTDiffusion](https://github.com/levihsu/OOTDiffusion) and [DCI-VTON](https://github.com/bcmi/DCI-VTON-Virtual-Try-On) for masking generation.

Thanks [SCHP](https://github.com/GoGoDuck912/Self-Correction-Human-Parsing) for human segmentation.

Thanks [Densepose](https://github.com/facebookresearch/DensePose) for human densepose.

Thanks [FurkanGozukara](https://github.com/FurkanGozukara/IDM-VTON) for Install the package.



## Citation
```
@article{choi2024improving,
  title={Improving Diffusion Models for Virtual Try-on},
  author={Choi, Yisol and Kwak, Sangkyung and Lee, Kyungmin and Choi, Hyungwon and Shin, Jinwoo},
  journal={arXiv preprint arXiv:2403.05139},
  year={2024}
}
```


## License
The codes and checkpoints in this repository are under the [CC BY-NC-SA 4.0 license](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).


