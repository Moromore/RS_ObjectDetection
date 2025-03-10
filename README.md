## 目前遇到两个问题
- MAE模型 bbox loss不下降
- CLIP模型 训练后 mAP指标过低  <= 0.04

## 文件夹解释
- `configs` 配置文件
- `mmdet` 模型代码文件，包含backbones，necks，loss等函数
- `open_clip` open_cip库里面的文件，包含修改的model.py以及transformer.py
- `slurm-test` tranning的logs
- `tools` main函数代码，tools/train.py
- `sbatch-test.sh` 提交训练任务的脚本

## 训练脚本在 `sbatch-test.sh`
例如```python -u tools/train.py configs/obb/oriented_rcnn/vit_base_win/SatMAEpp_large.py```

## 代码流程
1. `tools/train.py`
2. 通过`configs/obb/oriented_rcnn/vit_base_win/*.py`配置模型
3. 通过`mmdet/` `models` `necks`等构造模型

## 修改部分
### 配置文件： `configs/obb/oriented_rcnn/vit_base_win/`
所有配置文件只修改了以下部分：
```angular2html
model = dict(
    type='OrientedRCNN',
    # pretrained='model_zoo/RS5M_ViT-B-32.pt',
    backbone=dict(
        type='CLIP',
        model_name='ViT-B-32',
        pretrained='model_zoo/RemoteCLIP-ViT-B-32.pt',
        ),
    neck=dict(
        type='FPN',
        in_channels=[768,768,768,768],
        out_channels=256,
        num_outs=5),
```
## 模型代码

只修改了backbones内文件`mmdet/models/backbones/`,

其中MAE模型基本没有修改，包含
- `SatMAE_vit_large.py`
- `SatMAEpp_vit_large.py`
- `scale_MAE_vit_large.py`
- `vitae_nc_win_rvsa_v3_wsz7.py`

CLIP base的模型都通过`CLIP.py`构建，修改部分包含
### 1.`CLIP.py`第251行
```angular2html
    def forward_features(self, x):
        x = self.model.forward(x)
        ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
        for i in range(len(ops)):
            x[i] = ops[i](x[i])
        return x
```
其中`ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]`
为按照其他模型添加的反卷积和池化等操作

### 2. open_clip库的部分，改动后的流程为

   - `CLIP.py`第198行 `clip_model = open_clip.create_model_and_transforms(model_name)`
   - `open_clip/model.py`第307行 `image_features = self.encode_image(...`
   - `open_clip/model.py`第266行 `features = self.visual(image)`
   - `open_clip/transformer.py`第588行，修改了VisionTransformer中的forward 函数，直接将x reshape成rpn_head的输入
   - `open_clip/transformer.py`第338行，修改了Transformer的forward函数，将原本只输出最后一层的feature改为了提取四层features

