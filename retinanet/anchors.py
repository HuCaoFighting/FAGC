import numpy as np
import torch
import torch.nn as nn


class Anchors(nn.Module): #手动设置每个特征图对应的anchor基础框大小、缩放比例和长宽比，如下定义：
    def __init__(self, pyramid_levels=None, strides=None, sizes=None, ratios=None, scales=None):
        super(Anchors, self).__init__()

        if pyramid_levels is None:
            self.pyramid_levels = [2, 3, 4, 5, 6] #特征金字塔的层数为3.4.5.6.7
        if strides is None:
            self.strides = [2 ** x for x in self.pyramid_levels] #特征图大小为 [原图大小/8, 原图大小/16, 原图大小/32, 原图大小/64, 原图大小/128]
        if sizes is None:
            self.sizes = [2 ** (x + 2) for x in self.pyramid_levels] #base_size设置为：[32,64,128,256,512]
            #即对于长宽为（原图大小/8，原图大小/8）的特征图，其特征图上的每个单元格cell对应原图区域上（32，32）大小的对应区域（这里对应的大小并不是实际感受野的大小，而是一种人为的近似设置）。

        #那么在大小为base_size的正方形框的基础上，对框进行长宽比例调整（3 种，分别为[0.5, 1, 2]）和缩放（3种，分别为[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]），便形成9种所谓的基础框/先验框anchor。（体现在不同的channel数）
        if ratios is None:
            self.ratios = np.array([0.5, 1, 2])
        if scales is None:
            self.scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    #实现所有anchor生成的程序为：
    def forward(self, image):
        # image=(B,C,H,W)
        image_shape = image.shape[2:] #取特原图的长宽HW
        image_shape = np.array(image_shape)
        # #计算每张图片的P3到p7对应的5个feature map的宽和高，返回list，存5个元素,[原图/8，原图/16，原图/32，原图/64，原图/128]
        image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]

        # compute anchors over all pyramid levels
        all_anchors = np.zeros((0, 4)).astype(np.float32) #新建一个空的数组
        #以每一个锚点为中心点（0,0），生成的9个anchor的坐标信息，格式为(x1, y1, x2, y2)，左上角点的坐标和右下角点的坐标
        for idx, p in enumerate(self.pyramid_levels):
            anchors         = generate_anchors(base_size=self.sizes[idx], ratios=self.ratios, scales=self.scales)
            shifted_anchors = shift(image_shapes[idx], self.strides[idx], anchors)
            all_anchors     = np.append(all_anchors, shifted_anchors, axis=0)

        all_anchors = np.expand_dims(all_anchors, axis=0)

        if torch.cuda.is_available():
            return torch.from_numpy(all_anchors.astype(np.float32)).cuda()
        else:
            return torch.from_numpy(all_anchors.astype(np.float32))

def generate_anchors(base_size=16, ratios=None, scales=None):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales w.r.t. a reference window.
    """

    if ratios is None:
        ratios = np.array([0.5, 1, 2])

    if scales is None:
        scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    num_anchors = len(ratios) * len(scales) # 计算Anchor的总数9

    # initialize output anchors 初始化输出的结果9×4的大小,最终为每行4个数是一个anchor的4个参数，一共9行9个anchor
    anchors = np.zeros((num_anchors, 4))

    # scale base_size
    #复制成2行，3列 ,即（2，3*3）
    #转置成（9，2），每行都是一组ratio和scale的组合，比例是base_size的
    #np.tile（a,(2，3)）函数的作用就是将函数对X轴复制两倍，Y周复制3倍（这里的复制指对a的整体进行复制）。
    anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T #2:指后两列

    # compute areas of anchors 其实2、3值是一样的（后两列）
    areas = anchors[:, 2] * anchors[:, 3]

    # correct for ratios实际2列上等于anchors[:, 2:]/sqrt（scales）而实际3列上等于anchors[:, 2:]×sqrt（scales）
    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales))) # repeats是各个ratios中的元素重复len(scales)次数
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales)) # 把H变成W的[0.5、0.5、0.5, 1、1、1 , 2，2，2]倍

    # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2) 转换anchors的形式
    #（指中心点（0，0）加宽高坐标）->沿着grid对角线向上平移到grid的中心点 (x1, y1, x2, y2) 
    # 动手画图就知道如何把Anchor第一列和第二列中的（0，0）坐标换位中心点坐标了

    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T #x1,x2
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

    return anchors

def compute_shape(image_shape, pyramid_levels):
    """Compute shapes based on pyramid levels.

    :param image_shape:
    :param pyramid_levels:
    :return:
    """
    image_shape = np.array(image_shape[:2])
    image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in pyramid_levels]
    return image_shapes


def anchors_for_shape(
    image_shape,
    pyramid_levels=None,
    ratios=None,
    scales=None,
    strides=None,
    sizes=None,
    shapes_callback=None,
):

    image_shapes = compute_shape(image_shape, pyramid_levels)

    # compute anchors over all pyramid levels
    all_anchors = np.zeros((0, 4))
    for idx, p in enumerate(pyramid_levels):
        anchors         = generate_anchors(base_size=sizes[idx], ratios=ratios, scales=scales)
        shifted_anchors = shift(image_shapes[idx], strides[idx], anchors) #每个网格对应的9个anchor，上一行生成的都是(0,0)的anchor
        all_anchors     = np.append(all_anchors, shifted_anchors, axis=0)

    return all_anchors


def shift(shape, stride, anchors):
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride

    # shift_x, shift_y：指生成网格grid，后面再加上anchors，就是每个网格对应的9个anchor-box

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel(), shift_y.ravel()
    )).transpose()
    #shifts也是(x,4)
    
    # a.ravel() #用ravel()方法将数组a拉成一维数组
    # np.vstack(): 行堆叠
    # transpose(): 转置
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = anchors.shape[0] #anchor是（9，4）
    K = shifts.shape[0]
    all_anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))

    return all_anchors

