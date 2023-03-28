# 使用Loftr进行匹配点计算

# 引入所需模块
import torch
import cv2
import numpy as np
import matplotlib.cm as cm
import glob
import os
import shutil
import math

from PIL import Image
from src.utils.plotting import make_matching_figure
from src.loftr import LoFTR, default_cfg

# 进行算法配置
# 可以进行“户外”“室内”的数据集选择.
matcher = LoFTR(config=default_cfg)
matcher.load_state_dict(torch.load("./LSGL/LoFTR/weights/outdoor_ds.ckpt")['state_dict'])
matcher = matcher.eval().cuda()

# 统一路径将'\\'转为'/'
def Unipath(path):
    path = path.replace('\\','/')
    return path

# 清理文件夹
def clean(filepath):
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    else:
        shutil.rmtree(filepath)
        os.mkdir(filepath)

# 建立输出文件夹
def outputpath(respath, db_dir):
    os.mkdir(respath + str(db_dir)+ '/')
        
# 求取平均置信度
def avg_score(mconf):
    total_score = 0
    # 求取总置信度
    for score in mconf:
        total_score = total_score + score
    # 求取平均值
    avg = total_score / len(mconf)
    return avg

def Euc_distance(mkpts0, mkpts1):
    pre_act = 0
    pre_act_deg = 0
    all_dis = 0
    all_deg = 0
    # 计算平均欧氏距离与欧式角度
    for i in range(len(mkpts0)):
        poi = mkpts0[i]
        target = mkpts1[i]
        x = poi[0] - target[0]
        y = poi[1] - target[1]
        d = math.sqrt(x * x + y * y)
        degree = math.atan(y/x)
        # 取第一个点用于确认基准偏移
        all_dis += d
        all_deg += degree
    # 计算平均长度和平均角度
    base_distance = all_dis / len(mkpts0)
    base_degree = all_deg / len(mkpts0)
    # 计算平方和
    for i in range(len(mkpts0)):
        poi = mkpts0[i]
        target = mkpts1[i]
        x = poi[0] - target[0]
        y = poi[1] - target[1]
        d = math.sqrt(x * x + y * y)
        degree = math.atan(y/x)
        pre_act += pow((d-base_distance),2)
        pre_act_deg +=  pow((degree-base_degree),2)
    # 计算欧式距离的误差
    RMSE_dis = math.sqrt(pre_act/len(mkpts0))
    RMSE_deg = math.sqrt(pre_act_deg/len(mkpts0))
    #计算总RMSE
    RMSE = RMSE_dis / RMSE_deg
    print("RMSE为" + str(RMSE))
    return RMSE

# 计算最佳成绩
def getBestscore(before,after):
    if before < after:
        return after
    else:
        return before

# 读取遮罩影像
def readmask(mask_images_path, mkpts0, mkpts1, mconf):
    # 将遮罩转换为矩阵
    mask = mask_images_path[0]
    mask = Image.open(mask)
    mask_array = np.array(mask)
    # 限定位置
    poi = 0
    match_len = len(mkpts0)
    index = []
    while poi < match_len:
        # 检索匹配点坐标x,y位置
        x = mkpts0[poi][0]
        y = mkpts0[poi][1]
        # 判断点是否处于遮罩范围内
        x = int(x)
        y = int(y)
        poirgb = mask_array[y][x]
        if (poirgb == [255,255,255]).all():
            # 若处于遮罩范围内，则记录该点
            index.append(poi)
        poi = poi + 1
    # 返回处理结果
    mkpts0_tmp = np.delete(mkpts0,index,0)
    mkpts1_tmp = np.delete(mkpts1,index,0)
    mconf_tmp = np.delete(mconf,index)
    print(mconf_tmp)
    mkpts = [mkpts0_tmp,mkpts1_tmp,mconf_tmp]
    return mkpts

# 进行处理
def process(userpath, datapath, respath, mask_outpath):
    # 清理文件夹
    clean(respath)
    
    # 遍历预设文件夹内的图片
    db_file_path = datapath
    user_file_path = userpath
    for dirpath, db_dirnames, filenames in os.walk(db_file_path):
        print(db_dirnames)
        for db_dir in db_dirnames:
            db_images_path = glob.glob(os.path.join(db_file_path + db_dir + '/' + '*.png'))
            print(db_images_path)
            user_images_path = glob.glob(os.path.join(user_file_path + '*.png'))
            mask_images_path = glob.glob(os.path.join(mask_outpath + '*.png'))
            for j in user_images_path:
                j = Unipath(j)
                m = 0
                # 记录最佳评分
                bestscore = 0
                # 清理并建立输出文件夹
                outputpath(respath, db_dir)
                for i in db_images_path:
                    i = Unipath(i)

                    # 读取图片
                    img0_raw = cv2.imread(j, cv2.IMREAD_GRAYSCALE)
                    img1_raw = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
                    img0_raw = cv2.resize(img0_raw, (640, 480))
                    img1_raw = cv2.resize(img1_raw, (640, 480))

                    img0 = torch.from_numpy(img0_raw)[None][None].cuda() / 255.
                    img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.
                    batch = {'image0': img0, 'image1': img1}

                    # 使用LoFTR算法进行匹配
                    with torch.no_grad():
                        matcher(batch)
                        mkpts0 = batch['mkpts0_f'].cpu().numpy()
                        mkpts1 = batch['mkpts1_f'].cpu().numpy()
                        mconf = batch['mconf'].cpu().numpy()

                    # 依据遮罩处理匹配结果
                    mkpts = readmask(mask_images_path, mkpts0, mkpts1, mconf)
                    mkpts0 = mkpts[0]
                    mkpts1 = mkpts[1]
                    mconf = mkpts[2]
                    score = avg_score(mconf) * len(mkpts0)
                    RMSE = Euc_distance(mkpts0, mkpts1)

                    # 绘制匹配结果
                    color = cm.jet(mconf, alpha=0.7)
                    text = [
                        'LoFTR',
                        'Matches: {}'.format(len(mkpts0)),
                    ]
                    # 记录最高分
                    bestscore = getBestscore(bestscore,score)
                    # 输出结果
                    fig = make_matching_figure(img0_raw, img1_raw, mkpts0, mkpts1, color, mkpts0, mkpts1, path=respath + str(db_dir)+ '/' + str(m) + '_' + str(len(mkpts0)) + "_" + str(score) + "_" + str(RMSE)+".png")
                    m += 1
                
                # 修改文件夹名称
                os.rename(respath + str(db_dir)+ '/', respath + str(db_dir) + '_' + str(bestscore) + '/')
                # 完成后回报
                print('目标匹配完成')