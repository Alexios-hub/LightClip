from torchvision.transforms import RandAugment,RandomResizedCrop, InterpolationMode
from torchvision.transforms import functional as F
from torchvision.transforms.autoaugment import _apply_op
from torch import Tensor

class myRandomResizedCrop(RandomResizedCrop):
    def __init__(self, size, scale=(0.08, 1.0)):
        super().__init__(size, scale)
    
    def forward(self, img, params):
        i, j, h, w = params
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation, antialias=self.antialias)
    
class myRandAugment(RandAugment):
    def __init__(self, num_ops=2, magnitude=9, num_magnitude_bins=31, interpolation=InterpolationMode.NEAREST, fill=None):
        super().__init__(num_ops, magnitude, num_magnitude_bins, interpolation, fill)
    
    def forward(self, img, params):
        fill = self.fill
        channels, height, width = F.get_dimensions(img)
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            elif fill is not None:
                fill = [float(f) for f in fill]
        op_meta = self._augmentation_space(self.num_magnitude_bins, (height, width))
        for param in params:
            op_index = param[0]
            op_name = list(op_meta.keys())[op_index]
            magnitude = param[1]
            img = _apply_op(img, op_name, magnitude, interpolation=self.interpolation, fill=fill)
        
        return img


random_resized_crop = myRandomResizedCrop(size=(256,256))
rand_augment = myRandAugment()