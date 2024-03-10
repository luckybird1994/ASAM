from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse
import tensorboard

parser  = argparse.ArgumentParser()
parser.add_argument('--gpus',type=int)
parser.add_argument('--dataset',type=str)
parser.add_argument('--batchsize',type=int)
parser.add_argument('--epoch',type=int)
parser.add_argument('--json_path',type=str)
parser.add_argument('--resume',type=str, default='models/control_sd15_ini.ckpt')
args = parser.parse_args()


# Configs
resume_path = args.resume
batch_size = args.batchsize
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

# Misc
dataset = MyDataset(root=args.dataset,json_path=args.json_path)
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)

trainer = pl.Trainer(strategy="ddp", accelerator="gpu", devices=args.gpus, precision=32, callbacks=[logger], max_epochs=args.epoch) 


# Train!
trainer.fit(model, dataloader)
