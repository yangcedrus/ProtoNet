# Prototypical Networks for Few-shot Learning 论文复现
## ProtoNet Networks论文简介

<img src='./image/protonet.png' width='640' height='280'>


prototypical networks属于小样本学习中基于度量的算法。它将support
set的同一类样本的所有embedding的质心作为这一类的prototype。query直接用自己的embedding去找最近的prototype，从而完成分类。

详细分析算法流程：
输入训练集<img src="http://latex.codecogs.com/gif.latex?D=(x_1,y_1),...,(x_n,y_n)"/>，其中<img src="http://latex.codecogs.com/gif.latex?y_i"/>是标签，<img src="http://latex.codecogs.com/gif.latex?D_k"/>是标签<img src="http://latex.codecogs.com/gif.latex?y_i=k"/>的子训练集。

在每个episode中，随机选取<img src="http://latex.codecogs.com/gif.latex?N_s"/>个样本作为support set <img src="http://latex.codecogs.com/gif.latex?S_k"/>，同时选取同一类的不同的<img src="http://latex.codecogs.com/gif.latex?N_q"/>个样本作为query set <img src="http://latex.codecogs.com/gif.latex?S_q"/>,通过
<img src="http://latex.codecogs.com/gif.latex?c_k=\frac{1}{N_C}\sum_{(x_i,y_i)\in{S_K}}f_{\phi}(x_i)"/>计算每个类的原型向量<img src="http://latex.codecogs.com/gif.latex?c_k"/>。并依此对所有query set的样本进行分类，通过得到的损失更新网络。

## 原论文效果
<img src='./image/result.png' width='640' height='280'>

## 实验环境
- [Python-3.7]

- [PaddlePaddle-2.1.2]

- [Cuda-11.2 and Cudnn-8.1]

## 实验步骤
（一）数据处理

+ [数据集下载](https://drive.google.com/file/d/1Oq7JKbd8-6QgLXbZ1MW4Wkv39EgDBk5t/view?usp=sharing)
+ 数据集预处理:  需要将`miniImageNet`数据集处理为如下格式：所有图像放到`../miniImageNet/images/`目录下，然后将本库中`data/split/`下的csv文件放到`../miniImageNet/`目录下，最后修改`train.py`和`test.py`中的`data_root`参数为`miniImageNet`目录所在路径。

    
（二）修改参数配置

- `epoch` 训练的epoch个数
- `lr` 学习率
- `data_root` 数据所在路径
- `episode_size` 每次优化的任务数
- `train_episode` 在训练阶段每个epoch的任务数
- `test_episode` 在测试阶段每个epoch的任务数
- `way_num` 训练时类别数量
- `shot_num` 支持集中每个类所含样本数量
- `query_num` 查询集中每个类所含样本数量
- `test_way` 测试时类别数量
- `use_memory` 是否使用内存


（三）模型训练

    $ python /script/train_5w_1s.sh  用于训练5way-1shot
    $ python /script/train_5w_5s.sh  用于训练5way-5shot
    
    
[5way-1shot训练日志](https://drive.google.com/file/d/1Q7Wg6cUprnpfIiakEvKWv92Ij_L4RM1O/view?usp=sharing)
[5way-5shot训练日志](https://drive.google.com/file/d/1NaOANcTVkL_5ftpdszfo1w3ApBbwEraw/view?usp=sharing)


（四）模型测试

    $ python /script/test_5w_1s.sh  用于测试5way-1shot 
    $ python /script/test_5w_5s.sh  用于测试5way-5shot
    
    
[5way-1shot测试日志](https://drive.google.com/file/d/1NJjCmK8gwG2iq5WlEw_5StbxSY0HhWIz/view?usp=sharing)
[5way-5shot测试日志](https://drive.google.com/file/d/1jEFHm9JQ_iuUERGunUxGo4n3dbuVvDgH/view?usp=sharing)

## 模型性能
我们复现的方法在5way-1shot上超过论文的性能0.32%，在5way-5shot上超过论文的性能0.18%。并且我们的复现是使用6000个任务进行测试得到的结果，原文中使用的是600个任务，因此我们的结果具有更高的可信度。

**miniImageNet Dataset**

|  Setups  | 1-Shot 5-Way | 5-Shot 5-Way |   
|:--------:|:------------:|:------------:|
|   Paper  |  49.42±0.78  |  68.20±0.66  | 
|   Ours   |  49.74±0.25  |  68.38±0.21  | 


## Reference

```bibtex
@inproceedings{snell2017prototypical,
  title={Prototypical networks for few-shot learning},
  author={Snell, Jake and Swersky, Kevin and Zemel, Richard},
  booktitle={Proceedings of the 31st International Conference on Neural Information Processing Systems},
  pages={4080--4090},
  year={2017}
}
```
