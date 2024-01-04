from asyncio import constants
from typing import Any
import torch
from unet import *
from dataset import *
from visualize import *
from anomaly_map import *
from metrics import *
from feature_extractor import *
from reconstruction import *
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2"

class Inference:
    def __init__(self, unet, config) -> None:
        self.test_dataset = Dataset_maker(
            root= config.data.data_dir,
            category=config.data.category,
            config = config,
            is_train=False,
        )
        self.testloader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size= config.data.test_batch_size,
            shuffle=False,
            num_workers= config.model.num_workers,
            drop_last=False,
        )
        self.unet = unet
        self.config = config
        self.reconstruction = Reconstruction(self.unet, self.config)
        
    def __call__(self) -> Any:
        os.makedirs("recons",exist_ok=True)
        i = 0
        with torch.no_grad():
            for input,label in self.testloader:
                input = input.to(self.config.model.device)
                x0 = self.reconstruction(input, input, self.config.model.w)[-1]
                for b in range(x0.shape[0]):
                    x = show_tensor_image(x0[b])
                    y = show_tensor_image(input[b])
                    image = np.vstack((x,y))
                    Image.fromarray(image).save(f"recons/{i}{b}.png")
                i+=1
                print(i,b)
