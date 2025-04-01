# Testing MIGC for my master thesis

<div align="center">
<h1>[ CVPR2024 Highlight ] MIGC: Multi-Instance Generation Controller for Text-to-Image Synthesis</h1>
<h1>[ TPAMI2024 ] MIGC++: Advanced Multi-Instance Generation Controller for Image Synthesis</h1>

[[Dewei Zhou](https://scholar.google.com/citations?user=4C_OwWMAAAAJ&hl=en&oi=ao)], [[You Li](https://github.com/LeyRio)], [[Fan Ma](https://flowerfan.site/)], [[Xiaoting Zhang]()], [[Yi Yang](https://scholar.google.com/citations?user=RMSuNFwAAAAJ&hl=en)]

[[Paper](https://arxiv.org/pdf/2402.05408)] [[Project Page](https://migcproject.github.io/)]

<h3>Abstract</h3>

We present a Multi-Instance Generation (MIG) task, simultaneously generating multiple instances with diverse controls in one image. Given a set of predefined coordinates and their corresponding descriptions, the task is to ensure that generated instances are accurately at the designated locations and that all instances' attributes adhere to their corresponding description. This broadens the scope of current research on Single-instance generation, elevating it to a more versatile and practical dimension. Inspired by the idea of divide and conquer, we introduce an innovative approach named Multi-Instance Generation Controller (MIGC) to address the challenges of the MIG task. Initially, we break down the MIG task into several subtasks, each involving the shading of a single instance. To ensure precise shading for each instance, we introduce an instance enhancement attention mechanism. Lastly, we aggregate all the shaded instances to provide the necessary information for accurately generating multiple instances in stable diffusion (SD). To evaluate how well generation models perform on the MIG task, we provide a COCO-MIG benchmark along with an evaluation pipeline. Extensive experiments were conducted on the proposed COCO-MIG benchmark, as well as on various commonly used benchmarks. The evaluation results illustrate the exceptional control capabilities of our model in terms of quantity, position, attribute, and interaction.

</div>

## Quick start

```
conda create -n MIGC_diffusers python=3.9 -y
conda activate MIGC_diffusers
pip install -r requirement.txt
```

### Checkpoints

Download the [MIGC_SD14.ckpt (219M)](https://drive.google.com/file/d/1v5ik-94qlfKuCx-Cv1EfEkxNBygtsz0T/view?usp=sharing) and put it under the 'pretrained_weights' folder.

```
├── pretrained_weights
│   ├── MIGC_SD14.ckpt
├── migc
│   ├── ...
├── bench_file
│   ├── ...
```

If you want to use MIGC++, please download the [MIGC++\_SD14.ckpt (191M)](https://drive.google.com/file/d/1KI8Ih7SHISG9v9zRL1xhDIBsPjDHqPxI/view?usp=drive_link) and put it under the 'pretrained_weights' folder.

```
├── pretrained_weights
│   ├── MIGC++_SD14.ckpt
├── migc
│   ├── ...
├── bench_file
│   ├── ...
```

## Images generation

The .csv file containing the prompts should be inside a folder named `prompts` that is posiotioned in the root of the project.

Each row in the CSV defines a single generation sample and includes:

- a full prompt (used as global context),
- up to 4 objects with corresponding bounding boxes and local phrases.

The expected column format is:
`id,prompt,obj1,bbox1,obj2,bbox2,obj3,bbox3,obj4,bbox4`

To run the image generation from that prompt list, use the following command:

```bash
python run_migc.py
```

## Single Image Generation (from the original repository)

By using the following command, you can quickly generate an image with **MIGC**.

```
CUDA_VISIBLE_DEVICES=0 python inference_single_image.py
```

By using the following command, you can quickly generate an image with **MIGC++**, where both the box and mask are used to control the instance location.

```
CUDA_VISIBLE_DEVICES=0 python migc_plus_inference_single_image.py
```

## Citation

```
@inproceedings{zhou2024migc,
  title={Migc: Multi-instance generation controller for text-to-image synthesis},
  author={Zhou, Dewei and Li, You and Ma, Fan and Zhang, Xiaoting and Yang, Yi},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={6818--6828},
  year={2024}
}

@article{zhou2024migc++,
  title={Migc++: Advanced multi-instance generation controller for image synthesis},
  author={Zhou, Dewei and Li, You and Ma, Fan and Yang, Zongxin and Yang, Yi},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2024},
  publisher={IEEE}
}
```
