# 提取建筑的分割结果
from PIL import Image
import glob
import numpy as np
import cv2
import os

# 获取遮罩图片
def getmask(maskpic):
  return Image.open(maskpic)

# 展示遮罩
def showmask(img, isgray=False):
  plt.axis("off")
  if isgray == True:
    plt.imshow(img, cmap='gray')
  else:
    plt.imshow(img)
  plt.show()

# 检查文件夹是否存在
def checkdoc(filepath):
    if not os.path.exists(filepath):
        os.mkdir(filepath)
        
# 检查图片场景类型
def imgscene(mask):
    # 读取路径
    mask_file_path = mask
    mask_images_path = glob.glob(os.path.join(mask_file_path + '*.png'))    
    # 遍历图片
    for maskpic in  mask_images_path:
        mask = getmask(maskpic)
        mask = np.array(mask.convert('L'))
        sky = np.sum(mask[...,:] == 254)
        size = mask.size
        sky_percent = (sky / size) * 100
        print('天空像元占比' + str(sky_percent))
    
    
# 处理文件
def proceed(mask, inp, inp_typ, segmodel):
    # 检查路径
    checkdoc(mask)
    checkdoc(inp)
    
    # 读取路径
    mask_file_path = mask
    inp_file_path = inp
    mask_images_path = glob.glob(os.path.join(mask_file_path + '*.png'))

    # 遍历图片
    for maskpic in  mask_images_path:
        
        # 输入图片进行处理
        mask = getmask(maskpic)
        mask = np.array(mask.convert('L'))
        if segmodel == 'maskformer-swin-large-ade':
            mask = np.where(mask[...,:] == 195, 255, mask)
            mask = np.where(mask[...,:] == 163, 255, mask)
            mask = np.where(mask[...,:] == 245, 255, mask)
            mask = np.where(mask[...,:] == 243, 255, mask)
            mask = np.where(mask[...,:] == 169, 255, mask)
            mask = np.where(mask[...,:] == 197, 255, mask)
            mask = np.where(mask[...,:] == 255, 255, 0)
        else:
            mask = np.where((mask[...,:] > 243)&(mask[...,:] < 255), 0, 255)

        # 转换掩膜矩阵的数据格式
        mask = mask.astype(np.uint8)
        Image.fromarray(mask)
        # showmask(Image.fromarray(mask), True)

        # 将矩阵转为图片输出
        img = Image.fromarray(mask)
        img.save(maskpic)

        #抠出黑底图像
        import cv2

        # 导入原图与背景图
        picpath = maskpic.split('/')
        picname = glob.glob(os.path.join(inp_file_path + picpath[-1]))
        picdir = picname[0]
        img = cv2.imread(picdir)
        img = cv2.resize(img, (640, 480))
        # except:
        #     img = cv2.imread(picname)
        #     img = cv2.resize(img, (640, 480))
        if (inp_typ=="inp_database"):
            back = cv2.imread("backg.png")
        else:
            back = cv2.imread("backg2.png")

        # 将mask图转化为灰度图
        mask = cv2.imread(maskpic,cv2.IMREAD_GRAYSCALE)

        # 将背景图resize到和原图一样的尺寸
        back = cv2.resize(back,(img.shape[1],img.shape[0]))

        # 这一步是将背景图中的遮罩部分抠出来，也就是人像部分的像素值为0
        scenic_mask =~mask
        scenic_mask = scenic_mask  / 255.0
        back[:,:,0] = back[:,:,0] * scenic_mask
        back[:,:,1] = back[:,:,1] * scenic_mask
        back[:,:,2] = back[:,:,2] * scenic_mask
        # 这部分是将遮罩外抠出来，也就是背景部分的像素值为0
        mask = mask / 255.0
        img[:,:,0] = img[:,:,0] * mask
        img[:,:,1] = img[:,:,1] * mask
        img[:,:,2] = img[:,:,2] * mask
        #这里做个相加就可以实现合并
        result = cv2.add(back,img)

        cv2.imwrite(maskpic, result)

