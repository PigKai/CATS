import argparse
import re
from transformers import AutoModelForCausalLM
import numpy as np

parser = argparse.ArgumentParser(description="selection_openDomain_Tokens")
parser.add_argument("--basepath", type=str, default="/media/user/d/xyk/model/Qwen2.5-VL-7B-Instruct")
parser.add_argument("--configpath", type=str,
                    default="/media/user/e/xyk/eagle-eye2/eagle-eye2/train/qwenvl2_5_7B_config.json")
parser.add_argument("--lr", type=float, default=3e-5)
# init training batching / in selection part_1 process batch init ----> 1 / in selection part_2 training process batch init ----> 4
parser.add_argument("--bs", type=int, default=1)
parser.add_argument('--gradient-accumulation-steps', type=int, default=4)
parser.add_argument('--tmpdir', type=str, default='/media/user/d/xyk/data1/llava_instruct/get_data_qwen2.5vl/llava_instruct_150k_0_67999/0')
parser.add_argument("--cpdir_distill", type=str, default='/media/user/d/xyk/model/Eagle_Qwen2.5vl_draft/endospec_distill/stage1_llava20ep')
parser.add_argument("--cpdir_selection", type=str, default='/media/user/d/xyk/model/Eagle_Qwen2.5vl_draft/endospec_distill/stage2_scienceqa_distill_1ep')
parser.add_argument('--evapath', type=str, default='')
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--surpriseDir', type=str, default='/media/user/d/xyk/data1/ScienceQA/qwen2.5vl_distill/ScienceQA_0_12725/0')
# depend on the size of distill dataset
parser.add_argument("--distill_epoch", type=int, default=20)
# depend on the size of shareGPT
parser.add_argument("--selection_epoch", type=int, default=0)
parser.add_argument('--sr_option', type=bool, default=True)
parser.add_argument("--warmup_ratio", type=float, default=0.03)
parser.add_argument("--total_token_result_path",type=str,default='total_token_result.txt')
parser.add_argument("--selected_token_result_path",type=str,default='selected_token_result.txt')
parser.add_argument("--usage_time_result_path",type=str,default="usage_time_result.txt")
args = parser.parse_args()

training_config = {
    "lr": args.lr,
    "bs": args.bs,
    "gradient_accumulation_steps": args.gradient_accumulation_steps,
    # distill dataset path
    "datapath": f"{args.tmpdir}",
    "selection_training_path": f"{args.surpriseDir}",
    # distill dataset training epoch depend on the Dataset size
    "num_epochs": args.distill_epoch,
    "is_warmup": True,
    "p_w": 0.1,
    "v_w": 1.0,
    "topk_w": 0.1,
    "head_w": 0.1,
    "num_workers": 2,
    "embeding": True,
    "act": "No",
    "data_noise": True,
    "noise": "uniform",
    "mean": 0.0,
    "std": 0.2,
    "residual": "true,norm",
    "max_len": 2048,
    # During training, truncating the training sequences means that the larger the setting, the more training data is used, and the better the effect, but it also consumes more VRAM.
    "config_path": args.configpath,
    "b1": 0.9,
    "b2": 0.95,
    "grad_clip": 0.5,
    "save_freq": 5,
    "top_k": 5,
    "threshold": 1.0,
    "teacher_temp":0.5
}
import json
# 这个是防止浮点运算溢出
from safetensors import safe_open
import os
import torch
from transformers import AutoTokenizer

torch.backends.cuda.matmul.allow_tf32 = True
from accelerate import Accelerator
from accelerate.utils import set_seed

set_seed(0)

accelerator = Accelerator(mixed_precision='bf16',
                          gradient_accumulation_steps=training_config["gradient_accumulation_steps"])

try:
    from ..model.cnets1 import Model
    from ..model.configs import EConfig
except:
    from model.cnets1 import Model
    from model.configs import EConfig

from typing import Any, Dict, List

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import numpy as np
from transformers import get_linear_schedule_with_warmup, AutoConfig

if accelerator.is_main_process:
    import wandb
    os.environ["WANDB_MODE"] = "offline" 
    wandb.init(project='ess', entity="hulilihao", config=training_config)

baseconfig = AutoConfig.from_pretrained(args.basepath)
head = torch.nn.Linear(baseconfig.hidden_size, baseconfig.vocab_size, bias=False)

try:
    with open(os.path.join(args.basepath, "model.safetensors.index.json"), "r") as f:
        index_json = json.loads(f.read())
        head_path = index_json["weight_map"]["lm_head.weight"]
        with safe_open(os.path.join(args.basepath, head_path),
                       framework="pt",
                       device="cpu") as f:
            tensor_slice = f.get_slice("lm_head.weight")
            vocab_size, hidden_dim = tensor_slice.get_shape()
            tensor = tensor_slice[:, :hidden_dim].float()
except:
    with open(os.path.join(args.basepath, "pytorch_model.bin.index.json"), "r") as f:
        index_json = json.loads(f.read())
        head_path = index_json["weight_map"]["lm_head.weight"]
    weights = torch.load(os.path.join(args.basepath, head_path))
    tensor = weights["lm_head.weight"].float()

head.weight.data = tensor
head.eval()

for param in head.parameters():
    param.requires_grad = False


# list_ datapath
# 不用按序号匹配，支持随机游走
def normal_list_files(path):
    datapath = []
    for root, directories, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            datapath.append(file_path)
    return datapath


def list_files(path):
    pattern = re.compile(r'data_(\d+)\.ckpt$')  # 正则匹配数字部分
    files_with_numbers = []

    for root, _, files in os.walk(path):
        for file in files:
            match = pattern.match(file)
            if match:
                number = int(match.group(1))  # 提取并转换为整数
                full_path = os.path.join(root, file)
                files_with_numbers.append((number, full_path))

    files_with_numbers.sort(key=lambda x: x[0])

    datapath = [fp for _, fp in files_with_numbers]
    return datapath


class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.0):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        tensor = data["hidden_state_big"]
        noise = torch.randn(tensor.size()) * self.std + self.mean
        noisy_tensor = tensor + noise
        data["hidden_state_big"] = noisy_tensor
        return data


class AddUniformNoise:
    def __init__(self, std=0.0):
        self.std = std

    def __call__(self, data):
        tensor = data["hidden_state_big"]
        noise = (torch.rand_like(tensor) - 0.5) * self.std * 512 / tensor.shape[1]
        noisy_tensor = tensor + noise
        data["hidden_state_big"] = noisy_tensor
        return data


class CustomDataset(Dataset):
    def __init__(self, datapath, transform=None):
        # super().__init__()
        self.data = datapath
        self.transform = transform
        self.mask_cache = {}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = torch.load(self.data[index])
        new_data = {}
        hidden_state = data["hidden_state"][:training_config["max_len"]][None, :]
        input_ids = data["input_ids"][:training_config["max_len"]][None, :]
        if index in self.mask_cache:
            loss_mask = self.mask_cache[index]
        else:
            loss_mask = data["loss_mask"][:training_config["max_len"]][None, :]
            loss_mask = loss_mask[0].tolist()
            # <eos>
            loss_mask[-1] = 0
        # sequence length
        length = hidden_state.shape[1]
        attention_mask = [1] * length

        if index not in self.mask_cache:
            loss_mask[-1] = 0

        input_ids_target = input_ids[:, 1:]
        zeropadding = torch.tensor([[0]])
        input_ids_target = torch.cat((input_ids_target, zeropadding), dim=1)

        target = hidden_state[:, 1:, :]
        zeropadding = torch.zeros(1, 1, target.shape[2])
        target = torch.cat((target, zeropadding), dim=1)

        new_data["attention_mask"] = attention_mask
        new_data["loss_mask"] = loss_mask
        new_data["target"] = target
        new_data["hidden_state_big"] = hidden_state
        new_data["input_ids"] = input_ids_target
        new_data["index"] = index
        if self.transform:
            new_data = self.transform(new_data)

        return new_data


class DataCollatorWithPadding:

    def paddingtensor(self, intensors, N):
        B, n, S = intensors.shape
        padding_tensor = torch.zeros(B, N - n, S)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def paddingtensor2D(self, intensors, N):
        B, n = intensors.shape
        padding_tensor = torch.zeros(B, N - n, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max(item['hidden_state_big'].shape[1] for item in features)
        batch_input_ids = torch.cat([self.paddingtensor2D(item['input_ids'], max_length) for item in features])
        batch_hidden_states = torch.cat([self.paddingtensor(item['hidden_state_big'], max_length) for item in features])
        batch_target = torch.cat([self.paddingtensor(item['target'], max_length) for item in features])
        batch_loss_mask = torch.tensor(
            [item['loss_mask'] + [0] * (max_length - len(item['loss_mask'])) for item in features])
        batch_attention_mask = torch.tensor(
            [item['attention_mask'] + [0] * (max_length - len(item['attention_mask'])) for item in features])
        batch_indices = [item['index'] for item in features]
        # batch_loss_mask = torch.ones_like(batch_loss_mask)
        # batch_attention_mask=torch.ones_like(batch_attention_mask)
        batch = {
            "input_ids": batch_input_ids,
            "hidden_states": batch_hidden_states,
            "target": batch_target,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
            "index": batch_indices
        }
        return batch


def top_accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k)
        return res


def compute_loss(target, target_p, predict, loss_mask):
    # CE
    out_head = head(predict)
    out_logp = nn.LogSoftmax(dim=2)(out_head)
    plogp = target_p * out_logp
    ploss = -torch.sum(torch.sum(loss_mask * plogp, 2)) / (loss_mask.sum() + 1e-5)
    # smooth l1
    vloss = criterion(predict, target)
    vloss = torch.sum(torch.mean(loss_mask * vloss, 2)) / (loss_mask.sum() + 1e-5)
    return vloss, ploss, out_head


@torch.no_grad()
def getkacc(model, data, head, max_length=5):
    def generate(hidden_states, input_ids, head, max_length=4, use_cache=True):
        if use_cache:
            past_key_values = None
            for i in range(max_length):
                if past_key_values != None:
                    out_hidden, past_key_values = model(last_hidden, input_ids=token, past_key_values=past_key_values,
                                                        use_cache=True)
                else:
                    out_hidden, past_key_values = model(hidden_states, input_ids=input_ids, use_cache=True)
                last_hidden = out_hidden[:, -1:]
                last_headout = head(last_hidden)
                token = torch.argmax(last_headout, dim=-1)
                input_ids = torch.cat((input_ids, token), dim=1)

        else:
            raise NotImplementedError

        return input_ids

    hidden_states = data["hidden_states"]
    input_ids = data["input_ids"]
    loss_mask = data["loss_mask"]
    target = data["target"]
    total = [0 for _ in range(max_length)]
    correct = [0 for _ in range(max_length)]
    bs, seq_len = hidden_states.shape[0], hidden_states.shape[1]
    target_headout = head(target)
    target_ids = target_headout.argmax(dim=2)

    for pre_len in range(1, seq_len):
        if loss_mask[:, pre_len].sum() == 0:
            continue
        pre_hidden_states = hidden_states[:, :pre_len]
        pre_input_ids = input_ids[:, :pre_len]
        outs = generate(pre_hidden_states, pre_input_ids, head, max_length=max_length)
        generate_ids = outs[:, pre_len:]
        for bid in range(bs):
            for k in range(max_length):
                if loss_mask[bid, pre_len + k] == 0:
                    break
                if pre_len + k >= seq_len:
                    break
                total[k] += 1
                if generate_ids[bid, k] == target_ids[bid, pre_len + k - 1]:
                    correct[k] += 1
                else:
                    for kk in range(k + 1, max_length):
                        total[kk] += 1
                    break

    acc = [correct[i] / total[i] for i in range(len(correct))]
    return acc


if training_config["data_noise"]:
    if training_config["noise"] == "uniform":
        aug = AddUniformNoise(std=training_config["std"])
    else:
        aug = AddGaussianNoise(mean=training_config["mean"], std=training_config["std"])
else:
    aug = None

distill_trining_datapath = normal_list_files(training_config["datapath"])
selection_training_datapath = normal_list_files(training_config["selection_training_path"])
# init distill distill dataset
distill_trainpath = distill_trining_datapath[:int(len(distill_trining_datapath) * 0.95)]
distill_testpath = distill_trining_datapath[int(len(distill_trining_datapath) * 0.95):]
# init shareGPT selection dataset
selection_trainpath = selection_training_datapath[:int(len(selection_training_datapath) * 0.95)]
selection_testpath = selection_training_datapath[int(len(selection_training_datapath) * 0.95):]
# init distill_part dataset/dataloader
distill_traindataset = CustomDataset(distill_trainpath, transform=aug)
distill_testdataset = CustomDataset(distill_testpath)

distill_train_loader = DataLoader(distill_traindataset, batch_size=training_config["bs"], shuffle=True,
                                  collate_fn=DataCollatorWithPadding(), num_workers=training_config["num_workers"],
                                  pin_memory=True)
distill_test_loader = DataLoader(distill_testdataset, batch_size=training_config["bs"], shuffle=False,
                                 collate_fn=DataCollatorWithPadding(),
                                 num_workers=training_config["num_workers"], pin_memory=True)

if accelerator.is_main_process:
    if not os.path.exists(args.cpdir_distill):
        os.makedirs(args.cpdir_distill)
    if not os.path.exists(args.cpdir_selection):
        os.makedirs(args.cpdir_selection)

config = EConfig.from_pretrained(training_config["config_path"])
model = Model(config, load_emb=False, path=args.basepath)

criterion = nn.SmoothL1Loss(reduction="none")
optimizer = optim.AdamW(model.parameters(), lr=training_config["lr"],
                        betas=(training_config["b1"], training_config["b2"]))
# "num_epochs":args.distill_epoch, "num_selection_epoch"
distill_num_epoch = training_config["num_epochs"]
# selection_num_epoch = training_config["num_selection_epoch"]
is_warmup = training_config["is_warmup"]
# criterion

# num_warmup_steps = training_config["num_warmup_steps"]
# total_steps = training_config["total_steps"]
steps_per_epoch = int(np.ceil(len(distill_traindataset) / training_config["bs"]))
if len(distill_traindataset) % training_config["bs"] != 0:
    steps_per_epoch += 1
update_steps_per_epoch = int(np.ceil(steps_per_epoch / accelerator.gradient_accumulation_steps))
total_steps = training_config["num_epochs"] * update_steps_per_epoch
num_warmup_steps = int(args.warmup_ratio * total_steps)

if is_warmup:
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps
    )
    model, head, optimizer, distill_train_loader, distill_test_loader, scheduler = accelerator.prepare(
        model, head, optimizer, distill_train_loader, distill_test_loader, scheduler
    )
else:
    model, head, optimizer, distill_train_loader, distill_test_loader = accelerator.prepare(
        model, head, optimizer, distill_train_loader, distill_test_loader
    )
for epoch in range(distill_num_epoch + 1):
    top_3acc = [0 for _ in range(3)]
    correct = 0
    total = 0
    epoch_loss = 0
    num_batches = 0
    model.train()

    for batch_idx, data in enumerate(tqdm(distill_train_loader)):
        with accelerator.accumulate(model):
            optimizer.zero_grad()
            predict = model(data["hidden_states"], input_ids=data["input_ids"], attention_mask=data["attention_mask"])
            with torch.no_grad():
                target_head = head(data["target"])
                target_p = nn.Softmax(dim=2)(target_head)
                target_p = target_p.detach()
            loss_mask = data["loss_mask"][:, :, None]
            vloss, ploss, out_head = compute_loss(data["target"], target_p, predict, loss_mask)
            loss = training_config["v_w"] * vloss + training_config["p_w"] * ploss
            accelerator.backward(loss)
            accelerator.clip_grad_value_(model.parameters(), training_config["grad_clip"])
            optimizer.step()
            if is_warmup:
                scheduler.step()
        with torch.no_grad():
            _, predicted = torch.max(out_head, 2)
            _, target = torch.max(target_head, 2)
            ct = loss_mask.sum().item()
            cc = ((predicted == target) * loss_mask.squeeze()).sum().item()
            out_head = out_head.view(-1, target_head.shape[-1])[loss_mask.view(-1) == 1]
            target = target.view(-1)[loss_mask.view(-1) == 1]
            topkacc = top_accuracy(out_head, target, (1, 2, 3))
            for top_i in range(len(topkacc)):
                top_3acc[top_i] += topkacc[top_i]
            total += ct
            correct += cc
        if accelerator.is_main_process and ct != 0:
            logdict = {"train/lr": optimizer.optimizer.param_groups[0]["lr"], "train/vloss": vloss.item(),
                       "train/ploss": ploss.item(), "train/loss": loss.item(), "train/acc": cc / ct}
            for id, i in enumerate(top_3acc):
                logdict[f'train/top_{id + 1}_acc'] = topkacc[id].item() / ct
            wandb.log(logdict)
        del ploss, vloss
        epoch_loss += loss.item()
        num_batches += 1

    correct, total = torch.tensor(correct).cuda(), torch.tensor(total).cuda()
    correct, total = accelerator.gather_for_metrics((correct, total))
    correct, total = correct.sum().item(), total.sum().item()
    epoch_loss /= num_batches
    top_3acc = accelerator.gather_for_metrics(top_3acc)
    if accelerator.is_local_main_process:
        for id, i in enumerate(top_3acc):
            wandb.log({f'train/epochtop_{id + 1}_acc': i.sum().item() / total})
    if accelerator.is_local_main_process:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, distill_num_epoch, epoch_loss))
        print('Train Accuracy: {:.2f}%'.format(100 * correct / total))
        wandb.log({"train/epochacc": correct / total, "train/epochloss": epoch_loss})
    if (epoch + 1) % training_config["save_freq"]:
        top_3acc = [0 for _ in range(3)]
        correct = 0
        total = 0
        epoch_loss = 0
        num_batches = 0
        model.eval()

        k_acc = [[] for i in range(5)]
        for batch_idx, data in enumerate(tqdm(distill_test_loader)):
            with torch.no_grad():
                if batch_idx < 10:
                    acces = getkacc(model, data, head, max_length=5)
                    for i in range(len(acces)):
                        k_acc[i].append(acces[i])
                predict = model(data["hidden_states"], input_ids=data["input_ids"],
                                attention_mask=data["attention_mask"])
                target_head = head(data["target"])
                target_p = nn.Softmax(dim=2)(target_head)
                target_p = target_p.detach()
                loss_mask = data["loss_mask"][:, :, None]
                vloss, ploss, out_head = compute_loss(data["target"], target_p, predict, loss_mask)
                loss = training_config["v_w"] * vloss + training_config["p_w"] * ploss
                _, predicted = torch.max(out_head, 2)
                _, target = torch.max(target_head, 2)
                ct = loss_mask.sum().item()
                cc = ((predicted == target) * loss_mask.squeeze()).sum().item()
                out_head = out_head.view(-1, target_head.shape[-1])[loss_mask.view(-1) == 1]
                target = target.view(-1)[loss_mask.view(-1) == 1]
                topkacc = top_accuracy(out_head, target, (1, 2, 3))
                for top_i in range(len(topkacc)):
                    top_3acc[top_i] += topkacc[top_i]
                total += ct
                correct += cc
            epoch_loss += loss.item()
            num_batches += 1

        mean_acces = []
        for id, i in enumerate(k_acc):
            mean_acc = np.array(i).mean()
            mean_acc = torch.tensor(mean_acc).cuda()
            mean_acces.append(mean_acc)

        mean_acces = accelerator.gather_for_metrics(mean_acces)
        if accelerator.is_local_main_process:
            for id, i in enumerate(mean_acces):
                mean_acc = i.mean().item()
                wandb.log({f"test/{id}_acc": mean_acc})

        correct, total = torch.tensor(correct).cuda(), torch.tensor(total).cuda()
        correct, total = accelerator.gather_for_metrics((correct, total))
        correct, total = correct.sum().item(), total.sum().item()
        top_3acc = accelerator.gather_for_metrics(top_3acc)
        if accelerator.is_local_main_process:
            for id, i in enumerate(top_3acc):
                wandb.log({f'test/top_{id + 1}_acc': i.sum().item() / total})
        epoch_loss /= num_batches
        if accelerator.is_local_main_process:
            print('Test Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, distill_num_epoch, epoch_loss))
            print('Test Accuracy: {:.2f}%'.format(100 * correct / total))
            wandb.log({"test/epochacc": correct / total, "test/epochloss": epoch_loss})
            # accelerator.save_state(output_dir=f"{args.cpdir}/state_{epoch}")
            accelerator.save_model(model, save_directory=f"{args.cpdir_distill}/state_{epoch}")
print("finish distill training ---- next is the selection training part")
print("\n\n\n")
import time
combined_trainpath = distill_trainpath + selection_trainpath
distill_boundary_idx = len(distill_trainpath)
selection_traindataset = CustomDataset(combined_trainpath)
selection_testdataset = CustomDataset(selection_testpath)
stage2_train_bs = 1  # 或其他你显存允许的值
selection_train_loader = DataLoader(
    selection_traindataset,
    batch_size=stage2_train_bs,
    shuffle=True, # 混合后打乱，实现 Replay 效果
    collate_fn=DataCollatorWithPadding(),
    num_workers=training_config["num_workers"],
    pin_memory=True
)
selection_test_loader = DataLoader(selection_testdataset, batch_size=1, shuffle=False,
                                   collate_fn=DataCollatorWithPadding(),
                                   num_workers=training_config["num_workers"], pin_memory=True)

selection_train_loader, selection_test_loader = accelerator.prepare(
    selection_train_loader, selection_test_loader
)

steps_per_epoch_sel = len(selection_traindataset) // stage2_train_bs
if len(selection_traindataset) % stage2_train_bs != 0: steps_per_epoch_sel += 1
update_steps_per_epoch_sel = int(np.ceil(steps_per_epoch_sel / training_config["gradient_accumulation_steps"]))

# 使用 args 参数
actual_selection_epoch = args.selection_epoch
total_steps_selection = (actual_selection_epoch + 1) * update_steps_per_epoch_sel
num_warmup_steps_sel = 500

print(f"Stage 2 (Strategy C: Replay + OHEM) Total Steps: {total_steps_selection}")
print(f"Distill Data Count: {distill_boundary_idx}, Total Data Count: {len(combined_trainpath)}")

# 设置优化器和 Scheduler
optimizer.param_groups[0]['lr'] = 8e-6  # 保持低学习率
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps_sel,
    num_training_steps=total_steps_selection
)
scheduler = accelerator.prepare(scheduler)
global_sr_mean = 1.0
ema_momentum = 0.99
difficulty_scale = 0.8
total_token_list = []
selected_token_list = []
total_time_list = []
model.train()
for epoch in range(actual_selection_epoch + 1):
    top_3acc = [0 for _ in range(3)]
    correct = 0
    total = 0
    epoch_loss = 0
    num_batches = 0

    epoch_threshold_sum = 0
    start_time = time.time()
    for batch_idx, data in enumerate(tqdm(selection_train_loader)):
        with accelerator.accumulate(model):
            optimizer.zero_grad()
            predict = model(data["hidden_states"], input_ids=data["input_ids"], attention_mask=data["attention_mask"])
            with torch.no_grad():
                # 第一处更新
                target_head = head(data["target"])
                target_p = nn.Softmax(dim=2)(target_head / training_config["teacher_temp"])
                target_p = target_p.detach()
                # 在线更新
                out_head_for_sr = head(predict)
                probs_ssm = torch.softmax(out_head_for_sr, dim=-1)

                true_token_ids = torch.argmax(target_p, dim=-1, keepdim=True)
                true_token_probs = torch.gather(probs_ssm, 2, true_token_ids).squeeze(-1)

                # topk_probs, _ = probs_ssm.topk(training_config["top_k"], dim=-1)
                # avg_topk_probs = topk_probs.mean(dim=-1)

                # surprise_ratio = true_token_probs / (avg_topk_probs + 1e-8)
                # 第二除通用更新
                student_max_probs, _ = probs_ssm.max(dim=-1)
                surprise_ratio = true_token_probs / (student_max_probs + 1e-8)
                # 去除padding part 只对response部分进行更新
                raw_loss_mask = data["loss_mask"]
                for i in range(stage2_train_bs):
                   total_token_list.append(torch.sum(raw_loss_mask[i],dim=-1).item())
                valid_sr = surprise_ratio[raw_loss_mask.bool()]

                if valid_sr.numel() > 0:
                    batch_mean = valid_sr.mean().item()
                    global_sr_mean = ema_momentum * global_sr_mean + (1 - ema_momentum) * batch_mean

                current_threshold = global_sr_mean * difficulty_scale
                epoch_threshold_sum += current_threshold
                is_hard_token = surprise_ratio < current_threshold
                #soft_weight_map = torch.where(is_hard_token, torch.tensor(1.0).to(accelerator.device), torch.tensor(0.1).to(accelerator.device))
                batch_indices = data["index"] # List[int]
                is_replay_sample = torch.tensor(
                    [idx < distill_boundary_idx for idx in batch_indices],
                    device=accelerator.device
                ).unsqueeze(1)
                final_select_mask = is_hard_token | is_replay_sample
                if raw_loss_mask.dim() == 2:
                    dynamic_loss_mask = raw_loss_mask * final_select_mask.long()
                else:
                    dynamic_loss_mask = raw_loss_mask.squeeze(-1) * final_select_mask.long()
                for i in range(stage2_train_bs):
                   selected_token_list.append(torch.sum(dynamic_loss_mask[i], dim=-1).item())
                dynamic_loss_mask = dynamic_loss_mask[:, :, None]

            loss_mask = data["loss_mask"][:, :, None]
            loss_mask_for_stat = data["loss_mask"][:, :, None]
            # 生成损失部分的掩码,进行选择性训练。
            vloss, ploss, out_head = compute_loss(data["target"], target_p, predict, dynamic_loss_mask)
            loss = training_config["v_w"] * vloss + training_config["p_w"] * ploss

            accelerator.backward(loss)
            accelerator.clip_grad_value_(model.parameters(), training_config["grad_clip"])
            optimizer.step()
            if is_warmup:
                scheduler.step()
        with torch.no_grad():
            _, predicted = torch.max(out_head, 2)
            _, target = torch.max(target_head, 2)
            ct = loss_mask.sum().item()
            cc = ((predicted == target) * loss_mask_for_stat.squeeze()).sum().item()
            out_head = out_head.view(-1, target_head.shape[-1])[loss_mask.view(-1) == 1]
            target = target.view(-1)[loss_mask.view(-1) == 1]
            topkacc = top_accuracy(out_head, target, (1, 2, 3))
            for top_i in range(len(topkacc)):
                top_3acc[top_i] += topkacc[top_i]
            total += ct
            correct += cc
        if accelerator.is_main_process and ct != 0:
            logdict = {"train/lr": optimizer.optimizer.param_groups[0]["lr"], "train/vloss": vloss.item(),
                       "train/ploss": ploss.item(), "train/loss": loss.item(), "train/acc": cc / ct}
            for id, i in enumerate(top_3acc):
                logdict[f'train/top_{id + 1}_acc'] = topkacc[id].item() / ct
            wandb.log(logdict)
        del ploss, vloss
        epoch_loss += loss.item()
        num_batches += 1
    correct, total = torch.tensor(correct).cuda(), torch.tensor(total).cuda()
    correct, total = accelerator.gather_for_metrics((correct, total))
    correct, total = correct.sum().item(), total.sum().item()
    epoch_loss /= num_batches
    top_3acc = accelerator.gather_for_metrics(top_3acc)
    if accelerator.is_local_main_process:
        for id, i in enumerate(top_3acc):
            wandb.log({f'train/epochtop_{id + 1}_acc': i.sum().item() / total})
    if accelerator.is_local_main_process:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, actual_selection_epoch, epoch_loss))
        print('Train Accuracy: {:.2f}%'.format(100 * correct / total))
        avg_threshold = epoch_threshold_sum / (num_batches + 1e-5)
        print(f"Epoch {epoch + 1} Avg Threshold: {avg_threshold:.4f} (Global Mean SR: {global_sr_mean:.4f})")
        wandb.log({"train/epochacc": correct / total, "train/epochloss": epoch_loss})
    end_time = time.time()
    total_time_list.append(start_time - end_time)

    if (epoch + 1) % training_config["save_freq"]:
        top_3acc = [0 for _ in range(3)]
        correct = 0
        total = 0
        epoch_loss = 0
        num_batches = 0
        model.eval()

        k_acc = [[] for i in range(5)]
        for batch_idx, data in enumerate(tqdm(selection_test_loader)):
            with torch.no_grad():
                if batch_idx < 10:
                    acces = getkacc(model, data, head, max_length=5)
                    for i in range(len(acces)):
                        k_acc[i].append(acces[i])
                predict = model(data["hidden_states"], input_ids=data["input_ids"],
                                attention_mask=data["attention_mask"])
                target_head = head(data["target"])
                target_p = nn.Softmax(dim=2)(target_head)
                target_p = target_p.detach()
                loss_mask = data["loss_mask"][:, :, None]
                vloss, ploss, out_head = compute_loss(data["target"], target_p, predict, loss_mask)
                loss = training_config["v_w"] * vloss + training_config["p_w"] * ploss
                _, predicted = torch.max(out_head, 2)
                _, target = torch.max(target_head, 2)
                ct = loss_mask.sum().item()
                cc = ((predicted == target) * loss_mask.squeeze()).sum().item()
                out_head = out_head.view(-1, target_head.shape[-1])[loss_mask.view(-1) == 1]
                target = target.view(-1)[loss_mask.view(-1) == 1]
                topkacc = top_accuracy(out_head, target, (1, 2, 3))
                for top_i in range(len(topkacc)):
                    top_3acc[top_i] += topkacc[top_i]
                total += ct
                correct += cc
            epoch_loss += loss.item()
            num_batches += 1

        mean_acces = []
        for id, i in enumerate(k_acc):
            mean_acc = np.array(i).mean()
            mean_acc = torch.tensor(mean_acc).cuda()
            mean_acces.append(mean_acc)
        mean_acces = accelerator.gather_for_metrics(mean_acces)
        if accelerator.is_local_main_process:
            for id, i in enumerate(mean_acces):
                mean_acc = i.mean().item()
                wandb.log({f"test/{id}_acc": mean_acc})
        correct, total = torch.tensor(correct).cuda(), torch.tensor(total).cuda()
        correct, total = accelerator.gather_for_metrics((correct, total))
        correct, total = correct.sum().item(), total.sum().item()
        top_3acc = accelerator.gather_for_metrics(top_3acc)
        if accelerator.is_local_main_process:
            for id, i in enumerate(top_3acc):
                wandb.log({f'test/top_{id + 1}_acc': i.sum().item() / total})
        epoch_loss /= num_batches
        if accelerator.is_local_main_process:
            print('Test Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, actual_selection_epoch, epoch_loss))
            print('Test Accuracy: {:.2f}%'.format(100 * correct / total))
            wandb.log({"test/epochacc": correct / total, "test/epochloss": epoch_loss})
            # accelerator.save_state(output_dir=f"{args.cpdir}/state_{epoch}")
            accelerator.save_model(model, save_directory=f"{args.cpdir_selection}/state_{epoch}")
result_parent_path = '/home/lkw/result'
if not os.path.exists(result_parent_path):
    os.makedirs(result_parent_path)
total_token_result_path = os.path.join(result_parent_path,args.total_token_result_path)
selected_token_result_path = os.path.join(result_parent_path,args.selected_token_result_path)
usage_time_result_path = os.path.join(result_parent_path,args.usage_time_result_path)
with open(total_token_result_path,'w',encoding='utf-8') as f:
    f.write(str(total_token_list))
with open(selected_token_result_path,'w',encoding='utf-8') as f:
    f.write(str(selected_token_list))
# print(f"total_token:{total_token},selected_token:{selected_token}")
