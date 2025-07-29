from diffusers import EulerDiscreteScheduler
from migc.migc_utils import seed_everything
from migc.migc_pipeline import StableDiffusionMIGCPipeline, MIGCProcessor, AttentionStore
import os
import pandas as pd
import numpy as np
from utils import logger
import math
import torch
import time
import torchvision.utils
import torchvision.transforms.functional as tf


def normalize_data(data, size=512):
    return [[ [coord / size for coord in box] for box in data ]]

def read_prompts_csv(path):
    df = pd.read_csv(path, dtype={'id': str})
    conversion_dict = {}
    for i in range(len(df)):
        entry = {'prompt': df.at[i, 'prompt']}
        # Dynamically find all obj/bbox columns and keep original naming
        for col in df.columns:
            if col.startswith('obj'):
                idx = col[3:]
                bbox_col = f'bbox{idx}'
                obj_val = df.at[i, col]
                bbox_val = df.at[i, bbox_col] if bbox_col in df.columns else None
                # Always include obj and bbox, even if NaN for retro compatibility
                entry[col] = obj_val
                if bbox_col in df.columns:
                    entry[bbox_col] = bbox_val
        conversion_dict[df.at[i, 'id']] = entry
    return conversion_dict


def main():
    migc_ckpt_path = 'pretrained_weights/MIGC_SD14.ckpt'
    assert os.path.isfile(migc_ckpt_path), "Please download the ckpt of migc and put it in the pretrained_weighrs/ folder!"


    sd1x_path = '/sdb/zdw/weights/stable-diffusion-v1-4' if os.path.isdir('/sdb/zdw/weights/stable-diffusion-v1-4') else "CompVis/stable-diffusion-v1-4"
    # MIGC is a plug-and-play controller.
    # You can go to https://civitai.com/search/models?baseModel=SD%201.4&baseModel=SD%201.5&sortBy=models_v5 find a base model with better generation ability to achieve better creations.
    
    # Construct MIGC pipeline
    pipe = StableDiffusionMIGCPipeline.from_pretrained(
        sd1x_path)
    pipe.attention_store = AttentionStore()
    from migc.migc_utils import load_migc
    load_migc(pipe.unet , pipe.attention_store,
            migc_ckpt_path, attn_processor=MIGCProcessor)
    pipe = pipe.to("cuda")
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

    negative_prompt = 'worst quality, low quality, bad anatomy, watermark, text, blurry' # the one used by the authors
    # 'low quality, low res, distortion, watermark, monochrome, cropped, mutation, bad anatomy, collage, border, tiled' is the one we usually use

    height = 512
    width = 512
    seeds = range(1,9)
    bench = read_prompts_csv(os.path.join("prompts","openSet.csv"))

    model_name="openSet-M"
        
    if (not os.path.isdir("./results/"+model_name)):
        os.makedirs("./results/"+model_name)
    
    #intialize logger
    l=logger.Logger("./results/"+model_name+"/")

    # ids to iterate the dict
    ids = []
    for i in range(0,len(bench)):
        ids.append(str(i).zfill(4))

    print("Start of generation process")

    for id in ids:
        bboxes = []
        phrases = []
        # Dynamically find all obj/bbox columns for each id
        for col in bench[id]:
            if col.startswith('obj') and bench[id][col] is not None and not (isinstance(bench[id][col], (int, float)) and math.isnan(bench[id][col])):
                idx = col[3:]  # get the number after 'obj'
                bbox_col = f'bbox{idx}'
                if bbox_col in bench[id] and bench[id][bbox_col] is not None:
                    phrases.append(bench[id][col])
                    bboxes.append([int(x) for x in bench[id][bbox_col].split(',')])


        output_path = "./results/"+model_name+"/"+ id +'_'+bench[id]['prompt'] + "/"

        if (not os.path.isdir(output_path)):
            os.makedirs(output_path)

        print("Sample number ",id)
        torch.cuda.empty_cache()

        gen_images=[]
        gen_bboxes_images=[]
        #BB: [xmin, ymin, xmax, ymax] normalized between 0 and 1
        prompt_phrases = [[bench[id]['prompt']] + phrases]
        normalized_boxes = normalize_data(bboxes)

        print(f"Prompt phrases: {prompt_phrases}")
        print(f"# of bboxes: {len(normalized_boxes[0])}")

        for seed in seeds:
            print(f"Current seed is : {seed}")
            seed_everything(seed)

            # start stopwatch
            start = time.time()

            if torch.cuda.is_available():
                g = torch.Generator('cuda').manual_seed(seed)
            else:
                g = torch.Generator('cpu').manual_seed(seed)
            
            images = pipe(prompt=prompt_phrases, 
                        bboxes=normalized_boxes, 
                        num_inference_steps=50, 
                        guidance_scale=7.5, 
                        MIGCsteps=25, 
                        aug_phase_with_and=False,
                        generator=g,
                        height=height,
                        width=width,
                        negative_prompt=negative_prompt).images

            # end stopwatch
            end = time.time()
            # save to logger
            l.log_time_run(start, end)

            #save the newly generated image
            image=images[0]
            image.save(output_path +"/"+ str(seed) + ".jpg")
            gen_images.append(tf.pil_to_tensor(image))

            #draw the bounding boxes
            image=torchvision.utils.draw_bounding_boxes(tf.pil_to_tensor(image),
                                                        torch.Tensor(bboxes),
                                                        labels=phrases,
                                                        colors=['green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green'],
                                                        width=4,
                                                        font='font.ttf',
                                                        font_size=20)
            #list of tensors
            gen_bboxes_images.append(image)
            tf.to_pil_image(image).save(output_path+str(seed)+"_bboxes.png")

        # save a grid of results across all seeds without bboxes
        tf.to_pil_image(torchvision.utils.make_grid(tensor=gen_images,nrow=4,padding=0)).save(output_path +"/"+ bench[id]['prompt'] + ".png")

        # save a grid of results across all seeds with bboxes
        tf.to_pil_image(torchvision.utils.make_grid(tensor=gen_bboxes_images,nrow=4,padding=0)).save(output_path +"/"+ bench[id]['prompt'] + "_bboxes.png")

    # log gpu stats
    l.log_gpu_memory_instance()
    # save to csv_file
    l.save_log_to_csv(model_name)
    print("End of generation process for ", model_name)

if __name__ == '__main__':
    main()