import os, sys, shutil, torch, yaml
from torchvision import transforms
from PIL import Image
import torchvision.utils as vutils
from tqdm import tqdm

H, W = 256, 452 
repo_path = 'benchmarking-generative-models-for-domain-adaptation/models/unit'
checkpoint_path = "./w_backup/gen_00036000.pt"
out_d = "adapted_dataset/frames"

sys.path.append(repo_path)
from trainer import UNIT_Trainer

with open(os.path.join(repo_path, 'configs/unit_gta2city_folder.yaml'), 'r') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

cfg['vgg_w'] = 0
cfg['crop_image_height'], cfg['crop_image_width'] = H, W

trainer = UNIT_Trainer(cfg)
trainer.cuda().eval()


state = torch.load(checkpoint_path, map_location='cuda')
trainer.gen_a.load_state_dict(state['a'])
trainer.gen_b.load_state_dict(state['b'])
print(f"Modello 36k caricato.")

in_d = "./ego/EGO-CH-OBJ-SEG/real/train/frames"  
for r, _, f in os.walk(in_d):
    if any(x.lower().endswith(('.png', '.jpg', '.jpeg')) for x in f):
        in_d = r
        break

os.makedirs(out_d, exist_ok=True)
tfm = transforms.Compose([
    transforms.Resize((H, W)), 
    transforms.ToTensor(), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

valid_images = [f for f in os.listdir(in_d) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

print(f"Traduzione di {len(valid_images)} immagini in corso...")

with torch.no_grad():
    for n in tqdm(valid_images):
        img_path = os.path.join(in_d, n)
        input_img = tfm(Image.open(img_path).convert('RGB')).unsqueeze(0).cuda()
        
        content, _ = trainer.gen_a.encode(input_img)
        output_img = trainer.gen_b.decode(content)
        
        vutils.save_image(output_img, os.path.join(out_d, n), normalize=True, value_range=(-1, 1))

print("Creazione archivio ZIP...")
shutil.make_archive("risultati_36k", 'zip', out_d)
print("Tutto pronto per il download.")