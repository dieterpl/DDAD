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
import matplotlib.pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2"
from skimage.metrics import structural_similarity as ssim

def compute_psnr(image1, image2):
    # Compute MSE
    mse = np.mean((image1 - image2) ** 2)

    # Calculate the maximum possible pixel value (assuming 8-bit images)
    max_pixel_value = 255.0

    # Compute PSNR
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))

    return psnr,image1

def compute_mse(image1, image2):
    # Compute MSE
    mse = np.mean((image1 - image2) ** 2)

    return mse,image1

def compute_ssim(image1, image2):
    # Open images using Pillow
    img1 = Image.fromarray(image1)
    img2 = Image.fromarray(image2)

    # Convert images to grayscale
    gray1 = img1.convert('L')
    gray2 = img2.convert('L')

    # Convert images to NumPy arrays
    array1 = np.array(gray1)
    array2 = np.array(gray2)

    # Compute SSIM score and difference image
    ssim_score, diff_image = ssim(array1, array2, full=True,window_size=111)

    # Convert the difference image back to PIL Image
    diff_image = (diff_image * 255).astype(np.uint8)
    diff_image = np.stack([diff_image] * 3, axis=-1)
    return ssim_score, diff_image

def compute_ssim_color(image1, image2):
    # Open images using Pillow
    img1 = Image.fromarray(image1)
    img2 = Image.fromarray(image2)

    # Convert images to NumPy arrays
    array1 = np.array(img1)
    array2 = np.array(img2)

    # Compute SSIM score for each color channel
    ssim_r, _ = ssim(array1[:,:,0], array2[:,:,0], full=True)
    ssim_g, _ = ssim(array1[:,:,1], array2[:,:,1], full=True)
    ssim_b, _ = ssim(array1[:,:,2], array2[:,:,2], full=True)

    # Average the SSIM scores
    ssim_score = (ssim_r + ssim_g + ssim_b) / 3.0

     # Convert the difference image back to PIL Image
    diff_image = (diff_image * 255).astype(np.uint8)
    diff_image = np.stack([diff_image] * 3, axis=-1)

    return ssim_score, diff_image

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
        scores = []
        with torch.no_grad():
            for input,label in self.testloader:
                input = input.to(self.config.model.device)
                x0 = self.reconstruction(input, input, self.config.model.w)[-1]
                for b in range(x0.shape[0]):
                    x = show_tensor_image(x0[b])
                    y = show_tensor_image(input[b])
                    image = np.hstack((x,y))
                    score,diff = compute_ssim(x,y)
                    image = np.hstack((image,diff))
                    print(i,np.round(score,2))
                    scores.append(score)
                    # Create plot in image
                    plt.plot(scores)
                    plt.ylim(0,1)
                    plt.savefig("plot.jpg")
                    plt.close()
                    plt.clf()
                    plot = np.array(Image.open("plot.jpg").resize((x.shape[0],x.shape[1])))
                    image = np.hstack((image,plot))
                    Image.fromarray(image).save(f"recons/{i}.png")
                    i+=1
        for i in np.argsort(scores):
            print(i,scores[i])
