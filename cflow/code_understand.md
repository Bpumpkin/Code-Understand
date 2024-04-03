## Cflow 缺陷检测代码理解
### 数据预处理
* 功能：  
> 输入图像的裁剪、归一化等操作，训练、测试、验证数据的划分比例、以及部分数据增强操作。  
> 重点会用到的操作是图像裁剪以及数据增强，可通过transform参数实现。倘若datamodule中transform设为None，engine.py中的_setup_transform()会选择anomalib库中定义好的transform。
* 实现:
> 数据预处理模块：
```python
datamodule = Folder(
    name="pencil",
    task=TaskType.CLASSIFICATION,
    root="pencilImage/",
    extensions=['.bmp','.BMP'],
    normal_dir="good",
    abnormal_dir="bad",
    normal_test_dir="test_good",
    normal_split_ratio=0.2,
    train_batch_size = 50,
    eval_batch_size = 50,
    transform = train_transform,
    num_workers = 8,
    test_split_ratio = 0.2,
    val_split_ratio = 0.2,
    seed = 666    
)
```
> _setup_transform模块：这里只贴出它默认选择的transform，这里的image_size可以在上面的datamodule里设置
```python
elif model.transform is None:
            image_size = datamodule.image_size if datamodule else None
            transform = model.configure_transforms(image_size)

# 对应的transformer：
 image_size = image_size or (256, 256)
        return Compose(
            [
                Resize(image_size, antialias=True),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ],
        )
```
### 模型结构
* 功能：  
> 结构的组成：基本操作、操作的连接、操作超参数等。  

* 实现:
> 模型载入模块，这里包括模型的训练和验证部分代码，但是需要重点看模型结构，在torch_model.py的CflowModel模块定义。
```python
model = cflow.lightning_model.Cflow(
        backbone='wide_resnet50_2',
        layers=('layer2', 'layer3', 'layer4'),
        pre_trained=False,
        fiber_batch_size=64,
        decoder='freia-cflow',
        condition_vector=128,
        coupling_blocks=8,
        clamp_alpha=1.9,
        permute_soft=False,
        lr=0.0002
    )
```
#### 1. <font color=red>encoder结构</font>，这里跟着论文结构图理解，如图中红框部分：  
![encoder](encoder.jpg "Encoder")
> 代码实现也比较简单，代码里直接根据名字调用了'wide_resnet50_2'，如上述Cflow中backbone的定义，目的是输出decoder的特征输入z，一层一层往下走(timm.py)可以看到encoder部分模型的实现代码如下：
```python
model = timm.create_model(
            self.backbone,
            pretrained=False,
            features_only=True,
            exportable=True,
        )
```
> 输出为按层索引的字典，目的是取出encoder不同尺度的特征输出，在图中可以直观地看到第一层输出的0-1-2-3，最后一层块输出0-1-2-3-4-5-6-7-8，中间的不是没有只是省略了，layers见上述Cflow中layers的定义，实现如下：
```python
features = dict(zip(self.layers, self.feature_extractor(inputs), strict=True))
```
> 这个地方如果自己要改的话，有两种情况
>> 1. timm中定义好的模型，可以通过改backbone参数实现，也就是模型名字  
>> 2. 自己写的模型，那就自己把模型(假设叫ownmodel,args为超参数)写好，在定义model后，就是前面的模型载入模块，补一句代码:
```python
model.model.encoder = ownmodel(args)
```
#### 2. <font color=red>Position Embedding</font>，对应decoder的条件输入c，如图中红框部分：  
![encoder](PE.jpg "Encoder")
> 代码实现调用了utils.py中的positional_encoding_2d()函数，目的是包含特征空间位置的正弦和余弦谐波，以个人觉得这个不需要详细理解代码，直接用就行，有兴趣可以再讨论。一层一层往下走(torch_model.py)可以看到PE部分的实现代码如下：
```python
pos_encoding = einops.repeat(
                positional_encoding_2d(self.condition_vector, im_height, im_width).unsqueeze(0),
                "b c h w-> (tile b) c h w",
                tile=batch_size,
            ).to(images.device)
```
#### 3. <font color=red>Decoder结构</font>，如图中红框部分：  
![encoder](decoder.jpg "Encoder")
> 这个地方是论文的创新点之一，就是条件归一化流模型，利用包含位置信息的Position Embedding作为条件，更好地进行缺陷定位。归一化流模型是一个概率生成模型，目的是估计特征向量z的对数似然，区分缺陷特征与正常特征。需要注意的是因为要处理不同尺度的特征z，所以decoder是有k个。代码实现调用了utils.py中的cflow_head()函数，可以看到decoder部分的实现代码如下：
```python
coder = SequenceINN(n_features)
    for _ in range(coupling_blocks):
        coder.append(
            AllInOneBlock,
            cond=0,
            cond_shape=(condition_vector,),
            subnet_constructor=subnet_fc,
            affine_clamping=clamp_alpha,
            global_affine_type="SOFTPLUS",
            permute_soft=permute_soft,
        )
    return coder
```
> 其中AllInOneBlock是具体每个decoder的实现，在all_in_one_block.py中实现，主要网络层是上述代码中“subnet_constructor=subnet_fc”定义的，也就是全连接层，其余是一些数学计算，这个地方不好在外部改，只能到anoamalib里的utils.py改下面这个函数：
```python
def subnet_fc(dims_in: int, dims_out: int) -> nn.Sequential:
    """Subnetwork which predicts the affine coefficients.

    Args:
        dims_in (int): input dimensions
        dims_out (int): output dimensions

    Returns:
        nn.Sequential: Feed-forward subnetwork
    """
    return nn.Sequential(nn.Linear(dims_in, 2 * dims_in), nn.ReLU(), nn.Linear(2 * dims_in, dims_out))
```
