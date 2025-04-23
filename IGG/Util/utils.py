import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
from PIL import Image,ImageDraw,ImageFont
import torch.nn.functional as F
import math
from torchvision import transforms as T, utils
import imageio

def get_text_list(textidx, text_data):
        textidx = textidx.reshape(-1)
        text_cond = [text_data[idx].strip() for idx in textidx]
        return text_cond

def get_text_list_from_time(src_time, tar_time):
    text_cond = []
    for i in range(len(src_time)):
        text_cond.append(f"given is a plant at {src_time[i]} hours. What would the plant look like at {tar_time[i]} hours?")
    return text_cond

def RGB2GRAY(img):#RGB: ([1, 3, 128, 128])   ([1, 3, 128, 128])
    # img = img.squeeze(0)
    # img = img.permute(1, 2, 0)
    # img = img.numpy()
    # img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
    # img = np.expand_dims(img, axis=0)
    B = img[:, 0:1]
    G = img[:, 1:2]
    R = img[:, 2:]
    img_grey = 0.299 * R + 0.587 * G + 0.114 * B
    
    return img_grey

# tensor of shape (channels, frames, height, width) -> gif
def video_tensor_to_gif(tensor, path, duration = 400, loop = 0, optimize = True, c=1):
    # print(tensor.shape)
    #tensor: [7, 20, 80, 80]  for way2: frames=20
    tensor = tensor[:c]
    channels, frames, height, width = tensor.shape
    
    # minimum = torch.min(tensor).item()
    # tensor = torch.cat((tensor, minimum*torch.ones([channels, 2, height, width], dtype=tensor.dtype, device=tensor.device)), dim=1)

    
    images = map(T.ToPILImage(), tensor.unbind(dim = 1))
    first_img, *rest_imgs = images
    first_img.save(path, save_all = True, append_images = rest_imgs, duration = duration, loop = loop, optimize = optimize)
    print(f"Saved gif to {path}\n")
    return images


from torchvision.transforms import ToPILImage
to_pil = ToPILImage()
def tensor_to_image(tensor, path):
    tensor = tensor.mul(255).byte()
    image = to_pil(tensor)
    image.save(path)
    print(f"Saved image to {path}\n")
    return image


def video_tensor_to_mp4(tensor, path, fps=10, c=1):
    tensor = tensor[:c]
    # tensor: (channels, frames, height, width)
    channels, frames, height, width = tensor.shape
    # 将 tensor 转为 numpy 数组
    video_array = tensor.permute(1, 2, 3, 0).detach().cpu().numpy()  # shape: (frames, height, width, channels)
    # 如果是单通道，扩展成 3 通道灰度
    if channels == 1:
        video_array = np.repeat(video_array, 3, axis=-1)
    # 使用 imageio 保存为 MP4
    writer = imageio.get_writer(path, fps=fps, codec='libx264', quality=10)
    for frame in video_array:
        print(np.max(frame), np.min(frame), np.mean(frame))
        frame = frame.astype(np.uint8)
        # print("frame:   ", frame.shape)
        writer.append_data(frame)
    writer.close()
    print(f"Saved mp4 to {path}\n")







class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list


class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):   #x: [20, 4096]    y: [20, 4096]
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)  #[20]
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)  #[20]

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)  #diff_norms:0.3271      y_norms:47.3393     /: 0.0069

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)
    


def to_numpy(arr):
    if isinstance(arr, np.ndarray):
        return arr
    try:
        from pycuda import gpuarray
        if isinstance(arr, gpuarray.GPUArray):
            return arr.get()
    except ImportError:
        pass
    try:
        import torch
        if isinstance(arr, torch.Tensor):
            return arr.cpu().numpy()
    except ImportError:
        pass

    raise Exception(f"Cannot convert type {type(arr)} to numpy.ndarray.")


from matplotlib import pyplot as plt
def Mgridplot(u, Hpath, Nx=64, Ny=64, displacement=True, color='red', dpi=128, scale=1, linewidth=0.2,**kwargs):
    """Given a displacement field, plot a displaced grid"""
    u = to_numpy(u)
    assert u.shape[0] == 1, "Only send one deformation at a time"
    # plt.figure(dpi= 128)
    plt.figure(figsize=(1,1))
    plt.xticks([])  # 去掉x轴 
    plt.yticks([])  # 去掉y轴
    plt.axis('off')  # 去掉坐标轴
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
    plt.margins(0,0)
    
    if Nx is None:
        Nx = u.shape[2]
    if Ny is None:
        Ny = u.shape[3]
    # downsample displacements
    h = np.copy(u[0,:,::u.shape[2]//Nx, ::u.shape[3]//Ny])

    # now reset to actual Nx Ny that we achieved
    Nx = h.shape[1]
    Ny = h.shape[2]
    # adjust displacements for downsampling
    h[0,...] /= float(u.shape[2])/Nx
    h[1,...] /= float(u.shape[3])/Ny

    if displacement: # add identity
        '''
            h[0]: 
        '''
        h[0,...] += np.arange(Nx).reshape((Nx,1))  #h[0]:  (118, 109)  add element: 118*1
        h[1,...] += np.arange(Ny).reshape((1,Ny))

    # put back into original index space
    h[0,...] *= float(u.shape[2])/Nx
    h[1,...] *= float(u.shape[3])/Ny
    # create a meshgrid of locations
    for i in range(h.shape[1]):
        plt.plot( h[0,i,:], h[1,i,:], color=color, linewidth=linewidth, **kwargs)
    for i in range(h.shape[2]):
        plt.plot(h[0,:,i], h[1,:,i],  color=color, linewidth=linewidth, **kwargs)
    plt.axis('equal')
    plt.gca().invert_yaxis()
    # plt.savefig(Hpath,dpi= dpi*20)
    plt.savefig(Hpath,dpi= dpi*scale,transparent=True)
    # plt.savefig(Hpath, dpi= dpi*20, transparent=True, bbox_inches='tight', pad_inches=0.0)
    plt.cla()
    plt.clf()
    plt.close()
    plt.close('all')



def MgridVelocity(u, v, vgrad, Hpath, Nx=64, Ny=64, displacement=True, color='red', dpi=128, scale=1, linewidth=0.2,**kwargs):
    """Given a displacement field, plot a displaced grid"""
    u = to_numpy(u)

    assert u.shape[0] == 1, "Only send one deformation at a time"
   
    # plt.figure(dpi= 128)
    plt.figure(figsize=(25,5))

    plt.subplot(1, 5, 1)
    # cancel all the axis
    

    if Nx is None:
        Nx = u.shape[2]
    if Ny is None:
        Ny = u.shape[3]
    # downsample displacements
    h = np.copy(u[0,:,::u.shape[2]//Nx, ::u.shape[3]//Ny])

    # now reset to actual Nx Ny that we achieved
    Nx = h.shape[1]
    Ny = h.shape[2]
    # adjust displacements for downsampling
    h[0,...] /= float(u.shape[2])/Nx
    h[1,...] /= float(u.shape[3])/Ny

    if displacement: # add identity
        '''
            h[0]: 
        '''
        h[0,...] += np.arange(Nx).reshape((Nx,1))  #h[0]:  (118, 109)  add element: 118*1
        h[1,...] += np.arange(Ny).reshape((1,Ny))

    # put back into original index space
    h[0,...] *= float(u.shape[2])/Nx
    h[1,...] *= float(u.shape[3])/Ny
    # create a meshgrid of locations
    for i in range(h.shape[1]):
        plt.plot( h[0,i,:], h[1,i,:], color=color, linewidth=linewidth, **kwargs)
    for i in range(h.shape[2]):
        plt.plot(h[0,:,i], h[1,:,i],  color=color, linewidth=linewidth, **kwargs)
    plt.axis('equal')
    plt.gca().invert_yaxis()
    # add title to the plot
    plt.title('phiinv')


    plt.subplot(1, 5, 2)
    plt.imshow(v[0], cmap='gray')
    plt.colorbar()
    plt.title('vx')


    plt.subplot(1, 5, 3)
    plt.imshow(v[1], cmap='gray')
    plt.colorbar()
    plt.title('vy')


    plt.subplot(1, 5, 4)
    plt.imshow(vgrad[0], cmap='gray')
    plt.colorbar()
    plt.title('v-grad-x')


    plt.subplot(1, 5, 5)
    plt.imshow(vgrad[1], cmap='gray')
    plt.colorbar()
    plt.title('v-grad-y')

    # plt.savefig(Hpath,dpi= dpi*20)
    plt.savefig(Hpath,dpi= dpi*scale,transparent=True)
    # plt.savefig(Hpath, dpi= dpi*20, transparent=True, bbox_inches='tight', pad_inches=0.0)
    plt.cla()
    plt.clf()
    plt.close()
    plt.close('all')


def identity(defshape, dtype=np.float32):
    """
    Given a deformation shape in NCWH(D) order, produce an identity matrix (numpy array)
    """
    dim = len(defshape)-2
    ix = np.empty(defshape, dtype=dtype)
    for d in range(dim):
        ld = defshape[d+2]
        shd = [1]*len(defshape)
        shd[d+2] = ld
        ix[:,d,...] = np.arange(ld, dtype=dtype).reshape(shd)
    return ix

def Mquiver(u, Nx=32, Ny=32, color='black', units='xy', angles='xy', scale=1.0, **kwargs):
    """Given a displacement field, plot a quiver of vectors"""
    u = to_numpy(u)
    assert u.shape[0] == 1, "Only send one deformation at a time"
    assert u.ndim == 4, "Only 2D deformations can use quiver()"
    from matplotlib import pyplot as plt
    if Nx is None:
        Nx = u.shape[2]
    if Ny is None:
        Ny = u.shape[3]
    # downsample displacements
    h = np.copy(u[:,:,::u.shape[2]//Nx, ::u.shape[3]//Ny])
    ix = identity(u.shape, u.dtype)[:,:,::u.shape[2]//Nx, ::u.shape[3]//Ny]
    # create a meshgrid of locations
    plt.quiver(ix[0,1,:,:], ix[0,0,:,:], h[0,1,:,:], h[0,0,:,:], color=color,
               angles=angles, units=units, scale=scale, **kwargs)
    plt.axis('equal')
    # plt.gca().invert_yaxis()
    plt.show()






def drawImage(GTphiinv):
    GTphiinv = GTphiinv.astype(np.float32)
    a1=GTphiinv[...,0]
    a2=GTphiinv[...,1]
    a3=GTphiinv[...,2]
    r = Image.fromarray(a1).convert('L')
    g = Image.fromarray(a2).convert('L')
    b = Image.fromarray(a3).convert('L')
    GTphiinv = Image.merge('RGB',(r,g,b))
    GTphiinv = GTphiinv.convert('RGB')
    return GTphiinv

def Torchinterp(src, phiinv):  
    phiinv = phiinv.clone()
    src = src.clone()
    #3D --  src:[1, 1, 64, 64, 64]     phiinv: [1, 64, 64, 64, 3]
    #2D --  src:[1, 1, 64, 64]     phiinv: [1, 64, 64, 2]

    # if(src.shape[-3]==1 and src.shape[-4]==1):
    #     src = src.squeeze(-3)
    #     phiinv = phiinv[...,0:2].squeeze(-4)
    mode='bilinear'
    shape = phiinv.shape[1:-1] 
    # normalize deformation grid values to [-1, 1] 
    for i in range(len(shape)):
        phiinv[...,i] = 2 * (phiinv[...,i] / (shape[i] - 1) - 0.5)
    return F.grid_sample(src, phiinv, align_corners=False,mode = mode, padding_mode= 'zeros')

def TorchinterpNearest(src, phiinv):  #3D: src:[1, 1, 64, 64, 64]     phiinv: [1, 64, 64, 64, 3]
    phiinv = phiinv.clone()
    src = src.clone()
    #3D --  src:[1, 1, 64, 64, 64]     phiinv: [1, 64, 64, 64, 3]
    #2D --  src:[1, 1, 64, 64]     phiinv: [1, 64, 64, 2]
    
    # if(src.shape[-3]==1 and src.shape[-4]==1):
    #     src = src.squeeze(-3)
        # phiinv = phiinv[...,0:2].squeeze(-4)
    mode='nearest'
    
    src = src.type(phiinv.dtype)
    shape = phiinv.shape[1:-1] 
    # normalize deformation grid values to [-1, 1] 
    for i in range(len(shape)):
        phiinv[...,i] = 2 * (phiinv[...,i] / (shape[i] - 1) - 0.5)
    return F.grid_sample(src, phiinv, align_corners=False,mode = mode, padding_mode= 'border')

def dice_coefficient_old(tensor_a, tensor_b):
    # print("dice_tensor_shape:  ",tensor_a.shape, tensor_b.shape)    #[10, 1, 128, 128] [10, 1, 128, 128]


    tensor_a = tensor_a.clone()
    tensor_b = tensor_b.clone()

    tensor_a[tensor_a<0.5] = 0
    tensor_a[tensor_a>=0.5] = 1

    intersection = torch.sum(tensor_a * tensor_b)  # Element-wise multiplication
    union = torch.sum(tensor_a) + torch.sum(tensor_b)
    
    print("union: ", union.shape)

    if union == 0:
        return 1.0  # Handle the case where both tensors are empty.
    res = (2.0 * intersection) / union
    print("res: ", res.shape)

    assert 4>8


    return res.item()

import cv2
class imsave_edge():
    def __init__(self):
        self.kernel = self.get_kernel()

    def get_kernel(self):
        kernel = np.array([[-1, -1, -1], [-1, 7.5, -1], [-1, -1, -1]], dtype=np.float32)
        return kernel

    def change_gray(self, inputs, min_value=-200, max_value=800):
        outputs = np.array(inputs, dtype=np.float32)
        outputs[outputs > max_value] = max_value
        outputs[outputs < min_value] = min_value
        outputs = (outputs - min_value) / (max_value - min_value)
        return outputs

    def get_edge(self, seg):
        outputs = cv2.filter2D(seg, -1, self.kernel)
        outputs = np.sign(outputs)
        return outputs

def Hausdorff_distance(tensor_a, tensor_b):
    getEdge = imsave_edge()
    tensor_a = tensor_a.clone().numpy().astype(np.float32)
    tensor_b = tensor_b.clone().numpy().astype(np.float32)
    # swich to numpy
    edge_a = getEdge.get_edge(tensor_a)
    edge_b = getEdge.get_edge(tensor_b)

    position_a = np.where(edge_a == 1)
    position_b = np.where(edge_b == 1)

    xyz_a = np.array([position_a[0], position_a[1], position_a[2]]).T
    xyz_b = np.array([position_b[0], position_b[1], position_b[2]]).T

    # print(xyz_a.shape, xyz_b.shape) (55777, 3) (57202, 3)

    # distances1to2 = torch.cdist(torch.tensor(xyz_a, dtype=torch.float32), torch.tensor(xyz_b, dtype=torch.float32)).min(dim=1).values
    # distances2to1 = torch.cdist(torch.tensor(xyz_b, dtype=torch.float32), torch.tensor(xyz_a, dtype=torch.float32)).min(dim=1).values

    distances1to2 = torch.cdist(torch.tensor(xyz_a, dtype=torch.float32).cuda(), torch.tensor(xyz_b, dtype=torch.float32).cuda())
    distances2to1 = torch.cdist(torch.tensor(xyz_b, dtype=torch.float32).cuda(), torch.tensor(xyz_a, dtype=torch.float32).cuda())

    # print(distances1to2.shape, distances2to1.shape)   torch.Size([55777, 57202]) torch.Size([57202, 55777])

    distances1to2 = torch.min(distances1to2, dim=1).values
    distances2to1 = torch.min(distances2to1, dim=1).values

    # print(distances1to2.shape, distances2to1.shape)   torch.Size([55777]) torch.Size([57202])


    hausdorff_distance = torch.max(torch.max(distances1to2), torch.max(distances2to1))

    # print(hausdorff_distance)   tensor(5.3852)
    #delete data
    del edge_a, edge_b, position_a, position_b, xyz_a, xyz_b, distances1to2, distances2to1, tensor_a, tensor_b
    return hausdorff_distance

def Hausdorff_distance(tensor_a, tensor_b):
    while tensor_a.shape[0] == 1:
        tensor_a = tensor_a.squeeze(0)
    while tensor_b.shape[0] == 1:
        tensor_b = tensor_b.squeeze(0)

    getEdge = imsave_edge()
    tensor_a = tensor_a.clone().numpy().astype(np.float32)
    tensor_b = tensor_b.clone().numpy().astype(np.float32)
    # swich to numpy
    edge_a = getEdge.get_edge(tensor_a)
    edge_b = getEdge.get_edge(tensor_b)

    position_a = np.where(edge_a == 1)
    position_b = np.where(edge_b == 1)

    xyz_a = np.array([position_a[0], position_a[1]]).T
    xyz_b = np.array([position_b[0], position_b[1]]).T

    # print(xyz_a.shape, xyz_b.shape) (55777, 3) (57202, 3)

    # distances1to2 = torch.cdist(torch.tensor(xyz_a, dtype=torch.float32), torch.tensor(xyz_b, dtype=torch.float32)).min(dim=1).values
    # distances2to1 = torch.cdist(torch.tensor(xyz_b, dtype=torch.float32), torch.tensor(xyz_a, dtype=torch.float32)).min(dim=1).values

    distances1to2 = torch.cdist(torch.tensor(xyz_a, dtype=torch.float32).cuda(), torch.tensor(xyz_b, dtype=torch.float32).cuda())
    distances2to1 = torch.cdist(torch.tensor(xyz_b, dtype=torch.float32).cuda(), torch.tensor(xyz_a, dtype=torch.float32).cuda())

    # print(distances1to2.shape, distances2to1.shape)   torch.Size([55777, 57202]) torch.Size([57202, 55777])

    distances1to2 = torch.min(distances1to2, dim=1).values
    distances2to1 = torch.min(distances2to1, dim=1).values

    # print(distances1to2.shape, distances2to1.shape)   torch.Size([55777]) torch.Size([57202])


    hausdorff_distance = torch.max(torch.max(distances1to2), torch.max(distances2to1))

    # print(hausdorff_distance)   tensor(5.3852)
    #delete data
    del edge_a, edge_b, position_a, position_b, xyz_a, xyz_b, distances1to2, distances2to1, tensor_a, tensor_b
    return hausdorff_distance.item()




def dice_coefficient(tensor_a, tensor_b, return_mean=True):
    # print("dice_tensor_shape:  ",tensor_a.shape, tensor_b.shape)    #[10, 1, 128, 128] [10, 1, 128, 128]
    tensor_a = tensor_a.clone()
    tensor_b = tensor_b.clone()

    tensor_a[tensor_a<0.5] = 0
    tensor_a[tensor_a>=0.5] = 1

    
    ndim = len(tensor_a.shape)
   
 
    if ndim == 3 or ndim == 2:
        intersection = torch.sum(tensor_a * tensor_b)
        union = torch.sum(tensor_a) + torch.sum(tensor_b)
    elif ndim == 4:
        intersection = torch.sum(tensor_a * tensor_b, dim=(1,2,3))  # Element-wise multiplication
        union = torch.sum(tensor_a, dim=(1,2,3)) + torch.sum(tensor_b,dim=(1,2,3))
    elif ndim == 5:
        intersection = torch.sum(tensor_a * tensor_b, dim=(1,2,3,4))  # Element-wise multiplication
        union = torch.sum(tensor_a, dim=(1,2,3,4)) + torch.sum(tensor_b,dim=(1,2,3,4))
    else:
        assert 4>9, "ndim is not 4 or 5 or 3 or 2"
    
    # print(intersection)
    # print(union)
    # if union == 0:
    #     return 1.0  # Handle the case where both tensors are empty.
    union[union==0] = 0.0000000001
    res = (2.0 * intersection) / union

    if return_mean:
        return torch.mean(res).item()
    res = [item.item() for item in res]
    return res

# def dice_coefficient_for_brain(transf_label, target_label, maskT=[2,41,4,43,8,47,10,49,11,50,12,51,17,53,16]):  ### transf_label, target_label:   cuda:tensor
def dice_coefficient_for_brain(transf_label, target_label, maskT=[2,41,4,43,10,49], return_mean=False): 
    transf_label = torch.round(transf_label)
    # print(rounded_tensor.dtype)
    # print(target_label.dtype)

    dice_one = []
    for cur_seg in maskT:
        cur_seg_value = torch.tensor([cur_seg])

        transf_label_ = transf_label.clone().cpu()
        mask = torch.isin(transf_label_, cur_seg_value)
        transf_label_[mask] = 1; transf_label_[~mask] = 0

        target_label_ = target_label.clone().cpu()
        mask = torch.isin(target_label_, cur_seg_value)
        target_label_[mask] = 1; target_label_[~mask] = 0
        
        dice_list_6 = dice_coefficient(transf_label_, target_label_,return_mean)
        dice_one.append(dice_list_6)                

    cur_seg_value = torch.tensor(maskT)

    transf_label_ = transf_label.clone().cpu()
    mask = torch.isin(transf_label_, cur_seg_value)
    transf_label_[mask] = 1; transf_label_[~mask] = 0
    
    target_label_ = target_label.clone().cpu()
    mask = torch.isin(target_label_, cur_seg_value)
    target_label_[mask] = 1; target_label_[~mask] = 0
    
    dice_list_6 = dice_coefficient(transf_label_, target_label_,return_mean)
    dice_one.append(dice_list_6)   

    if return_mean:
        return np.mean(dice_one, axis=1)
    # get round dice
    # dice_one = [round(i, 3) for i in dice_one]
    res1 = []
    res2 = []
    res3 = []
    [(res1.append(item[0]), res2.append(item[1]), res3.append(item[2])) for item in dice_one]
    # print("res1: ", res1)
    # print("res2: ", res2)
    # print("res3: ", res3)
    # return dice_one

    return [res1, res2, res3]

def plot_matrix_distribution(matrix, path, name):
    import matplotlib.pyplot as plt
    import numpy as np

    # Plot the distribution of the matrix
    plt.figure(figsize=(8, 6))
    plt.hist(matrix.flatten(), bins=30, color='blue', alpha=0.7)
    plt.title('name')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.grid(True)
    # plt.show()
    plt.savefig(f'{path}/{name}.png')


def compute_MSE_list(Sdef, tar): #torch.Size([20, 1, 64, 64]) torch.Size([20, 1, 64, 64])
    # torch.sum(tensor_a * tensor_b, dim=(1,2,3))
    return [torch.nn.MSELoss()(Sdef[i], tar[i]).item() for i in range(Sdef.shape[0])]


def optimize_2D(src, tar, batch_size, num_steps=10, imagesize=128, epoches=100):
    from EpdiffLib import Epdiff
    import lagomorph as lm
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # sigma = 0.02;alpha=1.0; gamma = 1.0;RegWeight=0.5 ## NeurNips2023
    sigma = 0.02;alpha=1.0; gamma = 1.0;RegWeight=0.5   ## CVPR2024
    fluid_params = [alpha, 0, gamma];   ## [alpha, ***, gamma]  # fluid_params = [0.1, 0, 0.01]; RegWeight = 1e4 ## 原论文
    metric = lm.FluidMetric(fluid_params)
    criterion = nn.MSELoss()
    MEpdiff = Epdiff(alpha=alpha, gamma=gamma)



    batch_size = batch_size
    lr = 1.5 * batch_size
    src = src.unsqueeze(1); tar = tar.unsqueeze(1)   #b,1,64,64   扩充channel
    src=src.permute(0,1,3,2).cuda(); tar=tar.permute(0,1,3,2).cuda()
    m = torch.zeros((batch_size,2,imagesize,imagesize), device=device, requires_grad=True)
    optimizer = torch.optim.SGD([{'params': [m], 'lr': lr}])

    for ep in range(epoches):
        # m = metric.flat(v)
        v = metric.sharp(m)

        phiinv, phi = MEpdiff.my_expmap_shooting(m, num_steps=num_steps)
        Sdef = lm.interp(src, phiinv)

        loss1 = criterion(tar,Sdef)
        loss2 =(v*m).sum() / (src.numel())
        loss = 0.5 * loss1/(sigma*sigma) + RegWeight * loss2

        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (ep+1)%25==0:
            print(ep, "  v-gradient:", round(torch.max(m.grad).item(),6), "   loss:  ", loss.item(), "  ", loss1.item(), "  ", loss2.item())


    u_seq, ui_seq_gt, v_seq = MEpdiff.my_expmap_u2phi(m, num_steps=num_steps)
    dfm_seq_gt = [lm.interp(src, u) for u in u_seq]
    dfm_seq_gt = [u.permute(0,1,3,2) for u in dfm_seq_gt]
    dfm_seq_gt = [src] + dfm_seq_gt
    
    u_seq = [torch.zeros_like(u_seq[0])] + u_seq
    ui_seq_gt = [torch.zeros_like(ui_seq_gt[0])] + ui_seq_gt


    dfm_seq_gt = torch.concat(dfm_seq_gt, dim=1)
    u_seq = torch.stack(u_seq, dim=1)
    ui_seq_gt = torch.stack(ui_seq_gt, dim=1)

    # print(dfm_seq_gt.shape, u_seq.shape, ui_seq_gt.shape)
    # return dfm_seq_gt, u_seq[-1], ui_seq_gt[-1]

    
    return dfm_seq_gt, u_seq, ui_seq_gt


def jacobian_determinant2(phiinv):  ##(20, 64, 64, 2)
    volshape = phiinv.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'


    J = np.gradient(phiinv)
    
    # 3D glow
    if nb_dims == 3:
        dx = J[0]
        dy = J[1]
        dz = J[2]

        # compute jacobian components
        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])
        res = Jdet0 - Jdet1 + Jdet2
         

    else:  # must be 2

        dfdx = J[0]
        dfdy = J[1]

        res = dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]
        res = dfdy[..., 0] * dfdx[..., 1] - dfdx[..., 0] * dfdy[..., 1]

    #make the numpy to 1 dimension array
    res = res.reshape(-1)
    # numpy to list
    res = res.tolist()
    
    return res




def jacobian_determinant_list(phiinv):  ##(20, 64, 64, 2)
    jab_list = []

    if phiinv.device.type == 'cuda':
        phiinv = phiinv.cpu().detach().numpy()
    else:
        phiinv = phiinv.numpy()
    
    for batch_i in range(phiinv.shape[0]):
        jab_list += jacobian_determinant2(phiinv[batch_i])

    return jab_list

def DX(u):              #DX   Y-dir grad
    [w,h] = u.shape
    dx = np.zeros((w,h))
    dx[:,1:] = u[:, 1:] - u[:, :-1]
    dx[:,0] = dx[:,1]
    return dx
def DY(u):              #DY   X-dir grad
    [w,h] = u.shape
    dy = np.zeros((w,h))
    dy[1:,:] = u[1:, :] - u[:-1, :]
    dy[0,:] = dy[1,:]
    return dy

def jacobian_determinant4(phiinv):
    # check inputs
    volshape = phiinv.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'
    

    """ # check inputs
    volshape = phiinv.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'
    # compute grid
    grid_lst = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, len(volshape))
    print(grid)
    assert 4>8 """

    
    
    # 3D glow
    if nb_dims == 3:
        dx = J[0]
        dy = J[1]
        dz = J[2]

        # compute jacobian components
        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

        return Jdet0 - Jdet1 + Jdet2

    else:  # must be 2
        Ju_x, Ju_y = DX(phiinv[...,0]),  DY(phiinv[...,0])
        Jv_x, Jv_y = DX(phiinv[...,1]),  DY(phiinv[...,1])


        return Ju_x * Jv_y - Jv_x * Ju_y
    

def CDX(u):              #DX   Y-dir grad
    # [c,w,h] = u.shape   #c,128,128
    # dx = torch.zeros((c,w,h))
    # dx = torch.zeros_like(u, requires_grad=True)
    # dx[:,:,1:] = u[:, :, 1:] - u[:, :, :-1]
    # dx[:,:,0] = dx[:, :,1]
    # return dx
    res = u[:, :, 1:] - u[:, :, :-1]
    res = torch.concat((res[:,:, 0:1], res), dim=-1)

    return res


def CDY(u):              #DY   X-dir grad
    # # [w,h] = u.shape
    # # dy = torch.zeros((c,w,h))
    # dy = torch.zeros_like(u, requires_grad=True)
    # dy[:, 1:,:] = u[:, 1:, :] - u[:, :-1, :]
    # dy[:, 0,:] = dy[:, 1,:]
    # return dy

    res = u[:, 1:, :] - u[:, :-1, :]
    res = torch.concat((res[:,0:1, :], res), dim=1)

    return res

def Cjacobian_determinant4(phiinv):
    # check inputs
    volshape = phiinv.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'
    

    Ju_x, Ju_y = CDX(phiinv[...,0]),  CDY(phiinv[...,0])
    Jv_x, Jv_y = CDX(phiinv[...,1]),  CDY(phiinv[...,1])
    
    # print(Ju_x.shape, Ju_y.shape, Jv_x.shape, Jv_y.shape) ###  ([352, 64, 64]) torch.Size([352, 64, 64]) torch.Size([352, 64, 64]) torch.Size([352, 64, 64])
    # assert 3>99
    return Ju_x * Jv_y - Jv_x * Ju_y
    






def jacobian_determinant_vm(phiinv): #(128, 128, 2)
    J = np.gradient(phiinv)
    dfdx = J[0]
    dfdy = J[1]

    return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]


def RegNorm_list(m,v):  ##([b, 64, 64, 3]) torch.Size([5, 64, 64, 3])
    print(m.shape, v.shape)
    # assert 4>8
    b,w,h,c = m.shape
    list = []
    for batch_i in range(m.shape[0]):
        # list += (m[batch_i]*v[batch_i]).sum() / (src.numel())
        temp = ((m[batch_i]*v[batch_i]).sum() / (w*h)) 
        list.append(temp.item())
    return list

  


class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)
    

def grad_x_2d(f): #input b,c,h,w
    # 计算列方向（x 方向）的梯度
    grad_x = (f[:, :, :, 2:] - f[:, :, :, :-2]) / 2  # 中心差分
    grad_x = torch.cat((f[:, :, :, 1:2] - f[:, :, :, 0:1], grad_x, f[:, :, :, -1:] - f[:, :, :, -2:-1]), dim=3)  # 边界用一侧差分
    return grad_x

def grad_y_2d(f):
    # 计算行方向（y 方向）的梯度
    grad_y = (f[:, :, 2:, :] - f[:, :, :-2, :]) / 2  # 中心差分
    grad_y = torch.cat((f[:, :, 1:2, :] - f[:, :, 0:1, :], grad_y, f[:, :, -1:, :] - f[:, :, -2:-1, :]), dim=2)  # 边界用一侧差分
    return grad_y



def get_source_gradient(f, scale_factor=1): #f: b,c,h,w
    # 计算输入张量的梯度
    grad_x = grad_x_2d(f)
    grad_y = grad_y_2d(f)

    if scale_factor == 1:
        return torch.cat((grad_x, grad_y), dim=1)
    

    #使用 interpolate 函数对梯度进行缩放
    zoomx = F.interpolate(grad_x, scale_factor=scale_factor, mode='bilinear', align_corners=False)
    zoomy = F.interpolate(grad_y, scale_factor=scale_factor, mode='bilinear', align_corners=False)

    return torch.cat((zoomx, zoomy), dim=0)


def getReductedData(trainX, k): #trainX: torch.Size([471, 784])
    # SVD
    U, S, VT = torch.linalg.svd(trainX, full_matrices=False)
    # Diemnsionality reduction on traning set and test set
    trainX = torch.matmul(trainX, VT[:k,:].T)
    return trainX, S

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
def FID(realImg, sampledImg, normalize=True):
    print(realImg.shape, sampledImg.shape)
    fid = FrechetInceptionDistance(feature=64, normalize=normalize)
    realImg = realImg.squeeze(1);sampledImg = sampledImg.squeeze(1)
    if realImg.shape[-1] == 3:
        realImg = realImg.permute(0, 3, 1, 2)
        sampledImg = sampledImg.permute(0, 3, 1, 2)
    fid.update(realImg, real=True)
    fid.update(sampledImg, real=False)
    return fid.compute()
def KID(realImg, sampledImg, normalize=True):
    print(realImg.shape, sampledImg.shape) #torch.Size([16, 1, 128, 128]) torch.Size([16, 1, 128, 128])
    kid = KernelInceptionDistance(feature=64, normalize=normalize, subset_size=min(realImg.shape[0], 50))
    realImg = realImg.squeeze(1);sampledImg = sampledImg.squeeze(1)
    if realImg.shape[-1] == 3:
        realImg = realImg.permute(0, 3, 1, 2)
        sampledImg = sampledImg.permute(0, 3, 1, 2)
    kid.update(realImg, real=True)
    kid.update(sampledImg, real=False)
    return kid.compute()


    
