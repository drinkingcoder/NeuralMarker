from core.utils.transformation import SynthPairTnf, Theta_gen
import numpy as np
from core.utils.utils import refine_grid


class Augmentor:
    def __init__(self, args):

        H, W = args.image_size

        self.tnf_list = ['affine', 'hom', 'tps']
        self.tnf_generators = {tnf: SynthPairTnf(geometric_model=tnf, output_size=(H, W), use_cuda=False) for tnf in self.tnf_list}
        self.theta_generator = {tnf: Theta_gen(geometric_model=tnf, output_size=(H, W)) for tnf in self.tnf_list}

    def __call__(self, im, tnf_type, theta=None):
        '''
        Input: im: Torch.tensor[C, H, W]
               tnf_type: 'random', 'affine', 'hom', 'tps'
        Output: im: Torch.tensor[C, H, W]
                grid: Torch.tensor[H, W, 2]
        '''
        if tnf_type == 'random':
            tnf_type = self.tnf_list[np.random.randint(3)]
            theta = self.theta_generator[tnf_type]()
            batch = {'image': im[None], 'theta': theta[None]}
        else:
            batch = {'image': im[None], 'theta': theta[None]}
        tnf_res = self.tnf_generators[tnf_type](batch)
        im = tnf_res['target_image'][0]
        grid = refine_grid(tnf_res['warped_grid'][0])

        return im, grid, tnf_type, theta

