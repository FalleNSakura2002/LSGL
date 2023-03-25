# 使用Huggingface模型

from transformers import MaskFormerFeatureExtractor, MaskFormerForInstanceSegmentation
from PIL import Image
import requests
import glob
import os
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
import shutil

# 检查文件夹是否存在
def checkdoc(filepath):
    if not os.path.exists(filepath):
        os.mkdir(filepath)
        
        
# 清理内存
def release_memory(segmodel):
    del class_queries_logits
    del feature_extractor
    del image
    del inputs
    del loader
    del masks_queries_logits
    del model
    del outputs
    if segmodel == 'maskformer-swin-large-ade' or segmodel == 'maskformer-swin-tiny-ade':
        del predicted_semantic_map
    if segmodel == 'maskformer-swin-large-coco':
        del predicted_panoptic_map
        del result
    print('影像处理与内存清理完成！')


# 分割影像
def imagesegment(inpimage,outimage,segmodel):
    # 清理文件夹
    shutil.rmtree(outimage)
    os.mkdir(outimage)
    
    # 打开图片并进行压缩
    image = Image.open(inpimage)
    image = image.resize((640,480))
    
    # 使用选定的模型处理数据
    if segmodel == 'maskformer-swin-large-ade':
        predicted_semantic_map = maskformer_sla(image)
    elif segmodel == 'maskformer-swin-large-coco':
        predicted_semantic_map = maskformer_slc(image)
    elif segmodel == 'maskformer-swin-tiny-ade':
        predicted_semantic_map = maskformer_sta(image)

    # loader使用torchvision中自带的transforms函数
    loader = transforms.Compose([
        transforms.ToTensor()])  

    unloader = transforms.ToPILImage()

    # 输出PIL格式图片
    def tensor_to_PIL(tensor):
        image = tensor.cpu().clone()
        image = image.squeeze(0)
        image = unloader(image)
        return image
    predicted_semantic_map = torch.tensor(predicted_semantic_map, dtype=torch.float32)
    try:
        imgname = inpimage.split('/')
        tensor_to_PIL(predicted_semantic_map).save(outimage+imgname[-1])
    except:
        print('请先建立输出路径文件夹！')

        
# maskformer-swin-large-ade模型       
def maskformer_sla(image):
    # 调用模型
    feature_extractor = MaskFormerFeatureExtractor.from_pretrained("facebook/maskformer-swin-large-ade")
    inputs = feature_extractor(images=image, return_tensors="pt")
    model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-large-ade")

    # 处理图片
    outputs = model(**inputs)
    class_queries_logits = outputs.class_queries_logits
    masks_queries_logits = outputs.masks_queries_logits
    predicted_semantic_map = feature_extractor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    
    # 返回结果
    return predicted_semantic_map


# maskformer-swin-large-coco模型
def maskformer_slc(image):
    # 调用模型
    feature_extractor = MaskFormerFeatureExtractor.from_pretrained("facebook/maskformer-swin-large-coco")
    model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-large-coco")
    inputs = feature_extractor(images=image, return_tensors="pt")

    # 处理图片
    outputs = model(**inputs)
    class_queries_logits = outputs.class_queries_logits
    masks_queries_logits = outputs.masks_queries_logits
    result = feature_extractor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    predicted_panoptic_map = result["segmentation"]
    
    # 返回结果
    return predicted_panoptic_map

# maskformer-swin-tiny-ade模型
def maskformer_sta(image):
    # 调用模型
    feature_extractor = MaskFormerFeatureExtractor.from_pretrained("facebook/maskformer-swin-tiny-ade")
    inputs = feature_extractor(images=image, return_tensors="pt")
    model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-tiny-ade")
    
    # 处理图片
    outputs = model(**inputs)
    class_queries_logits = outputs.class_queries_logits
    masks_queries_logits = outputs.masks_queries_logits
    predicted_semantic_map = feature_extractor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    
    # 返回结果
    return predicted_semantic_map

