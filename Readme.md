# 简介

本项目基于[EasyOCR](https://github.com/JaidedAI/EasyOCR) 和[ABINet](https://github.com/FangShancheng/ABINet) 实现，EasyOCR提供文本检测和定位框架，ABINet识别文本内容。

数据集基于米游社''采薇东篱夏''制作的[提瓦特字体](https://bbs.mihoyo.com/ys/article/9058992)
使用[TextRecognitionDataGenerator](https://github.com/Belval/TextRecognitionDataGenerator) 自动生成得到。
数据集生成过程中使用[albumentations](https://albumentations.readthedocs.io/) 库进行数据增强，提高模型泛化能力。

# 数据集制作
运行命令
```bash
cd data_gen
python text_gen.py
```
生成数据集，缺什么库安装什么就行。

# 使用
预训练模型：

链接：https://pan.baidu.com/s/1TrXBIybAO6-WmXHzzn0krw 
提取码：e3ah

放入ABINet文件夹中


识别图片
```bash
python demo_image.py -p <文件路径>
python demo_image.py -p <文件路径> --step #生成检测视频
```

识别视频
```bash
python demo_video.py -p <文件路径>
```

![识别效果 ](image/kl.png)
