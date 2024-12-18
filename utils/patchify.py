import torch
import torch.nn.functional as F
from einops import rearrange

def tensor_patchify(tensor, p, patch_size):
    # print(f"tensor input dtype {tensor.dtype}")
    if p == 1:
        return tensor
    b, n, c = tensor.shape
    # p = int(patch_size**0.5)
    # assert p in [2, 4]
    h = w = int(n**0.5)
    # print(type(p))
    # Reshape the input tensor to [b, c, h, w] before applying avg_pool2d
    tensor = tensor.reshape(b, h, w, c).permute(0, 3, 1, 2)  # [b, c, h, w]
    if p == 2:
        tensor_patches = F.unfold(tensor, kernel_size=(2,2), stride=(2,2)) # (b, c*p*p, num_patches)
    elif p == 4:
        tensor_patches = F.unfold(tensor, kernel_size=(4,4), stride=(4,4)) 
    elif p == 8:
        tensor_patches = F.unfold(tensor, kernel_size=(8,8), stride=(8,8)) 
    tensor_patches = tensor_patches.transpose(1, 2)  # (b, num_patches, c*p*p)
    tensor_patches = tensor_patches.view(b, -1, c, patch_size).permute(0, 1, 3, 2)  # (b,  num_patches, patch_size, c)
    tensor_patches = tensor_patches.mean(2)
    # Apply avg_pool2d with kernel size p and stride p
    # tensor = F.avg_pool2d(tensor, kernel_size=p, stride=p)
    # patchified_tensor = rearrange(tensor, 'b c h w -> b (h w) c')
    # print(f"tensor output dtype { patchified_tensor.dtype}")
    
    return tensor_patches# [b, n, c]

def target_patchify(tensor, p, patch_size):
    tensor = tensor.unsqueeze(-1)
    b, n, c = tensor.shape
    # p = int(patch_size**0.5)
    h = w = int(n**0.5)
    # print(tensor.dtype)
    # tensor = tensor.float()
    # print(tensor.dtype)

    tensor = tensor.reshape(b, h, w, c).permute(0, 3, 1, 2) 
    if p == 2:
        tensor_patches = F.unfold(tensor, kernel_size=(2,2), stride=(2,2)) # (b, c*p*p, num_patches)
    elif p == 4:
        tensor_patches = F.unfold(tensor, kernel_size=(4,4), stride=(4,4)) 
    elif p == 8:
        tensor_patches = F.unfold(tensor, kernel_size=(8,8), stride=(8,8)) 

    tensor_patches = tensor_patches.transpose(1, 2)  # (b, num_patches, c*p*p)
    tensor_patches = tensor_patches.view(b, -1, c, patch_size).permute(0, 1, 3, 2)  # (b,  num_patches, patch_size, c)
    return tensor_patches.squeeze(-1)
    # return tensor_patches.squeeze(-1).long()

# a = torch.tensor([[[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7],[8,8],[9,9],[10,10],[11,11],[12,12],[13,13],[14,14],[15,15],[16,16]]]).to(torch.float32)

# b = target_patchify(a, 4)
# print(b, b.shape, a.shape)