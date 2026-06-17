"""
数据集下载脚本
自动下载并解压Pascal VOC 2012数据集
"""

import os
import urllib.request
import tarfile
from tqdm import tqdm


def download_file(url, output_path):
    """下载文件并显示进度条"""
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def download_voc2012(data_dir='data'):
    """下载VOC2012数据集"""
    os.makedirs(data_dir, exist_ok=True)
    
    # VOC2012数据集URL
    url = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'
    tar_path = os.path.join(data_dir, 'VOCtrainval_11-May-2012.tar')
    
    # 检查是否已下载
    if os.path.exists(tar_path):
        print(f"数据集已下载: {tar_path}")
    else:
        print("正在下载VOC2012数据集...")
        print(f"下载地址: {url}")
        print("这可能需要几分钟时间...")
        
        try:
            download_file(url, tar_path)
            print("下载完成!")
        except Exception as e:
            print(f"下载失败: {e}")
            print("请手动下载数据集并放置到data目录")
            print("下载地址: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar")
            return False
    
    # 解压数据集
    extract_dir = os.path.join(data_dir, 'VOCdevkit')
    if os.path.exists(extract_dir):
        print(f"数据集已解压: {extract_dir}")
    else:
        print("正在解压数据集...")
        try:
            with tarfile.open(tar_path, 'r') as tar:
                tar.extractall(data_dir)
            print("解压完成!")
        except Exception as e:
            print(f"解压失败: {e}")
            return False
    
    # 验证数据集
    voc_dir = os.path.join(extract_dir, 'VOC2012')
    if os.path.exists(voc_dir):
        print(f"\n数据集路径: {voc_dir}")
        
        # 统计图像数量
        image_dir = os.path.join(voc_dir, 'JPEGImages')
        if os.path.exists(image_dir):
            num_images = len([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
            print(f"图像数量: {num_images}")
        
        # 统计标注数量
        anno_dir = os.path.join(voc_dir, 'Annotations')
        if os.path.exists(anno_dir):
            num_annotations = len([f for f in os.listdir(anno_dir) if f.endswith('.xml')])
            print(f"标注数量: {num_annotations}")
        
        print("\n数据集准备完成!")
        return True
    else:
        print("数据集结构异常，请检查")
        return False


def create_train_val_split(data_dir='data'):
    """创建训练集和验证集划分"""
    voc_dir = os.path.join(data_dir, 'VOCdevkit', 'VOC2012')
    image_sets_dir = os.path.join(voc_dir, 'ImageSets', 'Main')
    
    os.makedirs(image_sets_dir, exist_ok=True)
    
    # 获取所有图像ID
    image_dir = os.path.join(voc_dir, 'JPEGImages')
    if not os.path.exists(image_dir):
        print("图像目录不存在")
        return
    
    image_ids = [os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith('.jpg')]
    image_ids.sort()
    
    # 划分训练集和验证集（80%训练，20%验证）
    split_idx = int(len(image_ids) * 0.8)
    train_ids = image_ids[:split_idx]
    val_ids = image_ids[split_idx:]
    
    # 写入文件
    with open(os.path.join(image_sets_dir, 'train.txt'), 'w') as f:
        f.write('\n'.join(train_ids))
    
    with open(os.path.join(image_sets_dir, 'val.txt'), 'w') as f:
        f.write('\n'.join(val_ids))
    
    print(f"训练集: {len(train_ids)}张")
    print(f"验证集: {len(val_ids)}张")
    print("数据集划分完成!")


def main():
    """主函数"""
    print("="*50)
    print("Pascal VOC 2012数据集下载工具")
    print("="*50)
    
    # 下载数据集
    if download_voc2012():
        # 创建训练验证划分
        create_train_val_split()
        
        print("\n数据集准备完成，可以开始训练了!")
        print("运行命令: python train.py")
    else:
        print("\n数据集准备失败，请检查网络连接或手动下载")


if __name__ == '__main__':
    main()
