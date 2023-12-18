from glob import glob
import torch, os
from torch.utils.data import Dataset
from PIL import Image as Image
import numpy as np
from torchvision.transforms import functional as F

class Img_DataSet(Dataset):
    def __init__(self, data_path, label_path):

        image = Image.open(data_path)
        label = Image.open(label_path)
        self.image = F.to_tensor(image)
        self.label = F.to_tensor(label)
        self.ori_shape = self.image.shape

        self.data_np = self.padding_img(self.image)
        self.new_shape = self.data_np.shape
        self.data_np = self.extract_ordered_overlap(self.data_np)

    def __getitem__(self, index):
        data = self.data_np[index]
        # target = self.label_np[index]
        data = torch.from_numpy(data).type(torch.FloatTensor)
        return data

    def __len__(self):
        return len(self.data_np)

    def padding_img(self, img):
        c,img_w,img_h = img.shape

        leftover_h = (img_h - 256) % 256
        leftover_w = (img_w - 256) % 256

        if (leftover_h != 0):
            h = img_h + (256 - leftover_h)
        else:
            h = img_h

        if (leftover_w != 0):
            w = img_w + (256 - leftover_w)
        else:
            w = img_w

        tmp_full_imgs = np.zeros((c,w,h))
        tmp_full_imgs[:,0:img_w, 0:img_h] = img

        return tmp_full_imgs

    # Divide all the full_imgs in pacthes
    def extract_ordered_overlap(self, img):
        img_c, img_w, img_h = img.shape

        N_patches_h = (img_h - 256) // 256 + 1
        N_patches_w = (img_w - 256) // 256 + 1
        N_patches_img =  N_patches_h * N_patches_w

        patches = np.empty((N_patches_img, 3, 256, 256))
        iter_tot = 0  # iter over the total number of patches (N_patches)
        for h in range(N_patches_h):
            for w in range(N_patches_w):
                patch = img[:,w * 256: w * 256+256,h * 256: h * 256+256]
                patches[iter_tot] = patch
                iter_tot += 1  # total
        assert (iter_tot == N_patches_img)
        return patches  # array with all the full_imgs divided in patches


class Recompone_tool():
    def __init__(self, img_ori_shape, img_new_shape):
        self.result = None
        self.ori_shape = img_ori_shape
        self.new_shape = img_new_shape


    def add_result(self, tensor):
        # tensor = tensor.detach().cpu() # shape: [N,class,s,h,w]
        # tensor_np = np.squeeze(tensor_np,axis=0)
        if self.result is not None:
            self.result = torch.cat((self.result, tensor), dim=0)
        else:
            self.result = tensor

    def recompone_overlap(self):
        """
        :param preds: output of model  shape：[N_patchs_img,3,patch_s,patch_h,patch_w]
        :return: result of recompone output shape: [3,img_s,img_h,img_w]
        """
        patch_w = self.result.shape[2]
        patch_h = self.result.shape[3]

        N_patches_h = (self.new_shape[2] - patch_h) // 256 + 1
        N_patches_w = (self.new_shape[1] - patch_w) // 256 + 1
        N_patches_img = N_patches_h * N_patches_w
        # print("N_patches_s/h/w:", N_patches_s, N_patches_h, N_patches_w)
        # print("N_patches_img: " + str(N_patches_img))
        assert (self.result.shape[0] == N_patches_img)

        full_prob = torch.zeros((3, self.new_shape[1], self.new_shape[2]))  # itialize to zero mega array with sum of Probabilities
        k = 0  # iterator over all the patches
        for h in range(N_patches_h):
            for w in range(N_patches_w):
                full_prob[:,w * 256:w * 256 + 256, h * 256:h  * 256 + 256] += self.result[k]
                k += 1
        assert (k == self.result.size(0))
        # print(final_avg.size())
        img = full_prob[:, :self.ori_shape[1], :self.ori_shape[2]]
        return img



def Test_Datasets(dataset_path):
    data_list = sorted(glob(os.path.join(dataset_path, 'blur/*')))
    label_list = sorted(glob(os.path.join(dataset_path, 'sharp/*')))
    #print(data_list)
    for datapath, labelpath in zip(data_list, label_list):
        yield Img_DataSet(datapath, labelpath),datapath.split('/')[-1]