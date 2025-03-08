import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image

os.chdir('gan-control')
sys.path.append('./src')

import torch
import torchvision.utils as vutils
from gan_control.inference.controller import Controller
from torchvision import transforms
from tqdm import tqdm


def initial_controller(controller_path):
    # load controller
    controller = Controller(controller_path)
    return controller

def initial_expression(expression_path):
    # Loading extracted attributes df
    attributes_df = pd.read_pickle(expression_path)
    expressions = attributes_df.expression3d.to_list()
    return expressions

def save_images(save_folder_path, image_tensors, num_gened_images, resize, folder_name):
    save_folder_path = os.path.join(save_folder_path, folder_name)
    os.makedirs(save_folder_path, exist_ok=True)
    for image in image_tensors:
        num_gened_images += 1
        image = transforms.Resize((resize, resize))(image)
        vutils.save_image(image, os.path.join(save_folder_path, f'{folder_name}{num_gened_images}.png'))

def genenerate_dataset(controller, save_folder_path, batch_size, num_gened_images, resize=512):
    # generate random face
    truncation = 0.7
    initial_image_tensors, initial_latent_z, initial_latent_w = controller.gen_batch(batch_size=batch_size, truncation=truncation)
        
    # save images
    save_images(save_folder_path, initial_image_tensors, num_gened_images, resize, folder_name='real_images')
    return initial_latent_z, initial_latent_w

def change_pose(controller, initial_latent_w, save_folder_path, num_gened_images, resize=512):
    dim = np.random.randint(low=1, high=3)
    degree = np.random.randint(low=-30, high=30)
    
    pose_control = torch.zeros((1, 3))
    pose_control[:, dim-1] = degree
    image_tensors, _, modified_latent_w = controller.gen_batch_by_controls(latent=initial_latent_w, input_is_latent=True, orientation=pose_control)
    
    # save images
    save_images(save_folder_path, image_tensors, num_gened_images, resize, folder_name='change_pose_images')
    
    return modified_latent_w

def smile(controller, expressions, initial_latent_w, save_folder_path, num_gened_images, resize=512):
    expression = torch.tensor([expressions[11]])
    image_tensors, _, modified_latent_w = controller.gen_batch_by_controls(latent=initial_latent_w, input_is_latent=True, expression=expression)

    save_images(save_folder_path, image_tensors, num_gened_images, resize, folder_name='smile_images')
    return modified_latent_w

def control_hair_color(controller, initial_latent_w, save_folder_path, resize, num_gened_images, color):
    bloand_color = torch.tensor([[0.73 , 0.62 , 0.36]])
    black_color = torch.tensor([[0.08 , 0.08 , 0.08]])
    
    if color == 'bloand':
        image_tensors, _, modified_latent_w = controller.gen_batch_by_controls(latent=initial_latent_w, input_is_latent=True, hair=bloand_color)
    elif color == 'black':
        image_tensors, _, modified_latent_w = controller.gen_batch_by_controls(latent=initial_latent_w, input_is_latent=True, hair=black_color)
    else:
        print('The color is invalid, please check again!')
    
    # save images
    save_images(save_folder_path, image_tensors, num_gened_images, resize, folder_name='control_hair_color')
    return modified_latent_w
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_folder', type=str, default=os.getcwd(), help='path to save images')
    parser.add_argument('--change_pose', type=bool, default=True, help='True|False')
    parser.add_argument('--smile', type=bool, default=True, help='True|False')
    parser.add_argument('--change_hair_color', type=bool, default=True, help='True|False')
    parser.add_argument('--color', type=str, default='bloand', help='bloand|black')
    parser.add_argument('--num_images', type=int, default=16, help='total generated images')
    parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
    parser.add_argument('--image_size', type=int, default=512, help='shape of generated images')
    
    args = parser.parse_args()
    num_batch = args.num_images // args.batch_size
    num_gened_images = 0
    
    # initial controller
    controller = initial_controller(controller_path='./resources/gan_models/controller_age015id025exp02hai04ori02gam15')
    
    # initial expression
    expression_path = './resources/ffhq_1K_attributes_samples_df.pkl'
    expressions = initial_expression(expression_path)
    
    for _ in tqdm(range(num_batch)):
        initial_latent_z, initial_latent_w = genenerate_dataset(
            controller=controller,
            save_folder_path=args.save_folder,
            batch_size=args.batch_size,
            resize=args.image_size,
            num_gened_images=num_gened_images
        )
        
        if args.change_pose:
            modified_latent_w = change_pose(
                controller=controller,
                initial_latent_w=initial_latent_w,
                save_folder_path=args.save_folder,
                resize=args.image_size,
                num_gened_images=num_gened_images
            )
            
        if args.smile:
            modified_latent_w = smile(
                controller=controller,
                expressions=expressions,
                initial_latent_w=initial_latent_w,
                save_folder_path=args.save_folder,
                resize=args.image_size,
                num_gened_images=num_gened_images
            )
        
        if args.change_hair_color:
            modified_latent_w = control_hair_color(
                controller=controller,
                initial_latent_w=initial_latent_w,
                save_folder_path=args.save_folder,
                resize=args.image_size,
                num_gened_images=num_gened_images,
                color=args.color
            )
        
        num_gened_images += args.batch_size
    
    if args.num_images % args.batch_size != 0:
        remain_image = args.num_images - num_batch * args.batch_size
        initial_latent_z, initial_latent_w = genenerate_dataset(
            controller=controller,
            save_folder_path=args.save_folder,
            batch_size=remain_image,
            resize=args.image_size,
            num_gened_images=num_gened_images
        )
        
        if args.change_pose:
            modified_latent_w = change_pose(
                controller=controller,
                initial_latent_w=initial_latent_w,
                save_folder_path=args.save_folder,
                resize=args.image_size,
                num_gened_images=num_gened_images
            )
            
        if args.smile:
            modified_latent_w = smile(
                controller=controller,
                expressions=expressions,
                initial_latent_w=initial_latent_w,
                save_folder_path=args.save_folder,
                resize=args.image_size,
                num_gened_images=num_gened_images
            )
        
        if args.change_hair_color:
            modified_latent_w = control_hair_color(
                controller=controller,
                initial_latent_w=initial_latent_w,
                save_folder_path=args.save_folder,
                resize=args.image_size,
                num_gened_images=num_gened_images,
                color=args.color
            )
        
if __name__ == '__main__':
    main()