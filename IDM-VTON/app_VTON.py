import os
import gradio as gr
import argparse, torch, os
from PIL import Image
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
)
from diffusers import AutoencoderKL
from typing import List
from util.common import open_folder
from util.image import pil_to_binary_mask, save_output_image
from utils_mask import get_mask_location
from torchvision import transforms
import apply_net
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from detectron2.data.detection_utils import convert_PIL_to_numpy,_apply_exif_orientation
from torchvision.transforms.functional import to_pil_image
from util.pipeline import quantize_4bit, restart_cpu_offload, torch_gc

parser = argparse.ArgumentParser()
parser.add_argument("--share", type=str, default=False, help="Set to True to share the app publicly.")
parser.add_argument("--lowvram", action="store_true", help="Enable CPU offload for model operations.")
parser.add_argument("--load_mode", default=None, type=str, choices=["4bit", "8bit"], help="Quantization mode for optimization memory consumption")
parser.add_argument("--fixed_vae", action="store_true", default=True,  help="Use fixed vae for FP16.")
args = parser.parse_args()

load_mode = args.load_mode
fixed_vae = args.fixed_vae

dtype = torch.float16
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_id = 'yisol/IDM-VTON'
vae_model_id = 'madebyollin/sdxl-vae-fp16-fix'

dtypeQuantize = dtype

if(load_mode in ('4bit','8bit')):
    dtypeQuantize = torch.float8_e4m3fn

ENABLE_CPU_OFFLOAD = args.lowvram
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.allow_tf32 = False
need_restart_cpu_offloading = False

unet = None
pipe = None
UNet_Encoder = None
example_path = os.path.join(os.path.dirname(__file__), 'example')

def start_tryon(dict, garm_img, garment_des, category, is_checked, is_checked_crop, denoise_steps, is_randomize_seed, seed, number_of_images):
    global pipe, unet, UNet_Encoder, need_restart_cpu_offloading

    if pipe == None:
        unet = UNet2DConditionModel.from_pretrained(
            model_id,
            subfolder="unet",
            torch_dtype=dtypeQuantize,
        )
        if load_mode == '4bit':
            quantize_4bit(unet)
            
        unet.requires_grad_(False)
       
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            model_id,
            subfolder="image_encoder",
            torch_dtype=torch.float16,
            )
        if load_mode == '4bit':
            quantize_4bit(image_encoder)
        
        if fixed_vae:
            vae = AutoencoderKL.from_pretrained(vae_model_id, torch_dtype=dtype)
        else:            
            vae = AutoencoderKL.from_pretrained(model_id,
                                                subfolder="vae",
                                                torch_dtype=dtype,
            )

        # "stabilityai/stable-diffusion-xl-base-1.0",
        UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
            model_id,
            subfolder="unet_encoder",
            torch_dtype=dtypeQuantize,
        )
     
        if load_mode == '4bit':
            quantize_4bit(UNet_Encoder)

        UNet_Encoder.requires_grad_(False)
        image_encoder.requires_grad_(False)
        vae.requires_grad_(False)
        unet.requires_grad_(False)
              
        pipe_param = {
                'pretrained_model_name_or_path': model_id,
                'unet': unet,     
                'torch_dtype': dtype,   
                'vae': vae,
                'image_encoder': image_encoder,
                'feature_extractor': CLIPImageProcessor(),
            }
        
        pipe = TryonPipeline.from_pretrained(**pipe_param).to(device)
        pipe.unet_encoder = UNet_Encoder    
        pipe.unet_encoder.to(pipe.unet.device)

        if load_mode == '4bit':
            if pipe.text_encoder is not None:
                quantize_4bit(pipe.text_encoder)
            if pipe.text_encoder_2 is not None:
                quantize_4bit(pipe.text_encoder_2)
           
    else:
        if ENABLE_CPU_OFFLOAD:
            need_restart_cpu_offloading =True
    
    torch_gc() 
    parsing_model = Parsing(0)
    openpose_model = OpenPose(0)
    openpose_model.preprocessor.body_estimation.model.to(device)
    tensor_transfrom = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize([0.5], [0.5]),
                    ]
            )
    
    if need_restart_cpu_offloading:
        restart_cpu_offload(pipe, load_mode)
    elif ENABLE_CPU_OFFLOAD:
        pipe.enable_model_cpu_offload()

    #if load_mode != '4bit' :
    #    pipe.enable_xformers_memory_efficient_attention()    

    garm_img= garm_img.convert("RGB").resize((768,1024))
    human_img_orig = dict["background"].convert("RGB")    
    
    if is_checked_crop:
        width, height = human_img_orig.size
        target_width = int(min(width, height * (3 / 4)))
        target_height = int(min(height, width * (4 / 3)))
        left = (width - target_width) / 2
        top = (height - target_height) / 2
        right = (width + target_width) / 2
        bottom = (height + target_height) / 2
        cropped_img = human_img_orig.crop((left, top, right, bottom))
        crop_size = cropped_img.size
        human_img = cropped_img.resize((768,1024))
    else:
        human_img = human_img_orig.resize((768,1024))

    if is_checked:
        keypoints = openpose_model(human_img.resize((384,512)))
        model_parse, _ = parsing_model(human_img.resize((384,512)))
        mask, mask_gray = get_mask_location('hd', category, model_parse, keypoints)
        mask = mask.resize((768,1024))
    else:
        mask = pil_to_binary_mask(dict['layers'][0].convert("RGB").resize((768, 1024)))
        # mask = transforms.ToTensor()(mask)
        # mask = mask.unsqueeze(0)
    
    mask_gray = (1-transforms.ToTensor()(mask)) * tensor_transfrom(human_img)
    mask_gray = to_pil_image((mask_gray+1.0)/2.0)

    human_img_arg = _apply_exif_orientation(human_img.resize((384,512)))
    human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")

    args = apply_net.create_argument_parser().parse_args(('show', './configs/densepose_rcnn_R_50_FPN_s1x.yaml', './ckpt/densepose/model_final_162be9.pkl', 'dp_segm', '-v', '--opts', 'MODEL.DEVICE', 'cuda'))
    # verbosity = getattr(args, "verbosity", None)
    pose_img = args.func(args,human_img_arg)    
    pose_img = pose_img[:,:,::-1]    
    pose_img = Image.fromarray(pose_img).resize((768,1024))
    
    if pipe.text_encoder is not None:        
        pipe.text_encoder.to(device)

    if pipe.text_encoder_2 is not None:
        pipe.text_encoder_2.to(device)

    with torch.no_grad():
        # Extract the images
        with torch.cuda.amp.autocast(dtype=dtype):
            with torch.no_grad():
                prompt = "model is wearing " + garment_des
                negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
                with torch.inference_mode():
                    (
                        prompt_embeds,
                        negative_prompt_embeds,
                        pooled_prompt_embeds,
                        negative_pooled_prompt_embeds,
                    ) = pipe.encode_prompt(
                        prompt,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=True,
                        negative_prompt=negative_prompt,
                    )
                                    
                    prompt = "a photo of " + garment_des
                    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
                    if not isinstance(prompt, List):
                        prompt = [prompt] * 1
                    if not isinstance(negative_prompt, List):
                        negative_prompt = [negative_prompt] * 1
                    with torch.inference_mode():
                        (
                            prompt_embeds_c,
                            _,
                            _,
                            _,
                        ) = pipe.encode_prompt(
                            prompt,
                            num_images_per_prompt=1,
                            do_classifier_free_guidance=False,
                            negative_prompt=negative_prompt,
                        )

                    pose_img =  tensor_transfrom(pose_img).unsqueeze(0).to(device,dtype)
                    garm_tensor =  tensor_transfrom(garm_img).unsqueeze(0).to(device,dtype)
                    results = []
                    current_seed = seed
                    for i in range(number_of_images):  
                        if is_randomize_seed:
                            current_seed = torch.randint(0, 2**32, size=(1,)).item()                        
                        generator = torch.Generator(device).manual_seed(current_seed) if seed != -1 else None                     
                        current_seed = current_seed + i

                        images = pipe(
                            prompt_embeds=prompt_embeds.to(device,dtype),
                            negative_prompt_embeds=negative_prompt_embeds.to(device,dtype),
                            pooled_prompt_embeds=pooled_prompt_embeds.to(device,dtype),
                            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device,dtype),
                            num_inference_steps=denoise_steps,
                            generator=generator,
                            strength = 1.0,
                            pose_img = pose_img.to(device,dtype),
                            text_embeds_cloth=prompt_embeds_c.to(device,dtype),
                            cloth = garm_tensor.to(device,dtype),
                            mask_image=mask,
                            image=human_img, 
                            height=1024,
                            width=768,
                            ip_adapter_image = garm_img.resize((768,1024)),
                            guidance_scale=2.0,
                            dtype=dtype,
                            device=device,
                        )[0]
                        if is_checked_crop:
                            out_img = images[0].resize(crop_size)        
                            human_img_orig.paste(out_img, (int(left), int(top)))   
                            img_path = save_output_image(human_img_orig, base_path="outputs", base_filename='img', seed=current_seed)
                            results.append(img_path)
                        else:
                            img_path = save_output_image(images[0], base_path="outputs", base_filename='img')
                            results.append(img_path)
                    return results, mask_gray
    
garm_list = os.listdir(os.path.join(example_path,"cloth"))
garm_list_path = [os.path.join(example_path,"cloth",garm) for garm in garm_list]

human_list = os.listdir(os.path.join(example_path,"human"))
human_list_path = [os.path.join(example_path,"human",human) for human in human_list]

human_ex_list = []
for ex_human in human_list_path:
        ex_dict = {}
        ex_dict['background'] = ex_human
        ex_dict['layers'] = None
        ex_dict['composite'] = None
        human_ex_list.append(ex_dict)

image_blocks = gr.Blocks(theme=gr.themes.Soft()).queue()
with image_blocks as demo:
    demo.title="摩诘AI汉化优化版VTON_1.0"
    gr.Markdown("## 摩诘AI汉化优化版VTON，安装使用教程请查阅:" + "[使用教程](https://space.bilibili.com/483532108/)")
    gr.Markdown("模特定制优化，本地/云端部署协助，请加 WeChat:MojieAIGC"+"|| 项目git地址：https://github.com/MoJIeAIGC/IDMVTON_CN")
    with gr.Row():
        with gr.Column():
            imgs = gr.ImageEditor(sources='upload', type="pil", label='上装模特(768X1024)', interactive=True,crop_size="768:1024")
            with gr.Row():
                category = gr.Radio(choices=["upper_body", "lower_body", "dresses"], label="选择成衣类型", value="upper_body")
                is_checked = gr.Checkbox(label="Yes", info="使用自动上装蒙版",value=True)
            with gr.Row():
                is_checked_crop = gr.Checkbox(label="Yes", info="使用自动裁切和调整大小",value=True)

            example = gr.Examples(
                inputs=imgs,
                examples_per_page=5,
                examples=human_ex_list,
                label='基础模特'
            )

        with gr.Column():
            garm_img = gr.Image(label="替换服装白底图", sources='upload', type="pil")
            with gr.Row(elem_id="prompt-container"):
                with gr.Row():
                    prompt = gr.Textbox(placeholder="英文填写提示词服装描述", show_label=False, elem_id="prompt")
            example = gr.Examples(
                label='模板服装',
                inputs=garm_img,
                examples_per_page=5,
                examples=garm_list_path)
        with gr.Column():
            with gr.Row():
            # image_out = gr.Image(label="Output", elem_id="output-img", height=400)
                masked_img = gr.Image(label="服装蒙版", elem_id="masked-img",show_share_button=False,width=300)
            with gr.Row():
                btn_open_outputs = gr.Button("输出文件夹")
                btn_open_outputs.click(fn=open_folder)
        with gr.Column():
            with gr.Row():
            # image_out = gr.Image(label="Output", elem_id="output-img", height=400)
                image_gallery = gr.Gallery(label="生成图像", show_label=True)
            with gr.Row():
                try_button = gr.Button(value="启动")
                denoise_steps = gr.Number(label="生成步数，默认(30)", minimum=20, maximum=120, value=30, step=1)
                seed = gr.Number(label="Seed", minimum=-1, maximum=2147483647, step=1, value=1)
                is_randomize_seed = gr.Checkbox(label="随机种子", value=True)  
                number_of_images = gr.Number(label="生成数量", minimum=1, maximum=9999, value=1, step=1)


    try_button.click(fn=start_tryon, inputs=[imgs, garm_img, prompt, category, is_checked, is_checked_crop, denoise_steps, is_randomize_seed, seed, number_of_images], outputs=[image_gallery, masked_img],api_name='tryon')

    image_blocks.launch(show_api=False,server_port=7892,inbrowser=True,share=args.share,)

