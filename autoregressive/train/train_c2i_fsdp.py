# Modified from:
#   Large-DiT: https://github.com/Alpha-VLLM/LLaMA2-Accessory/blob/main/Large-DiT-ImageNet/train.py
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy, MixedPrecision, StateDictType, FullStateDictConfig
)
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy, size_based_auto_wrap_policy

import os
import time
import inspect
import functools
import argparse
import contextlib
from glob import glob
import wandb

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from utils.logger import create_logger
from dataset.build import build_dataset
from autoregressive.models.gpt import GPT_models, precompute_freqs_cis_2d



def setup_fsdp_sync(model: nn.Module, args: argparse.Namespace, device) -> FSDP:
    model = FSDP(
        model,
        auto_wrap_policy=functools.partial(
            lambda_auto_wrap_policy,
            lambda_fn=lambda m: m in model.get_fsdp_wrap_module_list(),
        ),
        # auto_wrap_policy=size_based_auto_wrap_policy,
        # process_group=fs_init.get_data_parallel_group(),
        device_id=device,
        sharding_strategy={
            "fsdp": ShardingStrategy.FULL_SHARD,
            "sdp": ShardingStrategy.SHARD_GRAD_OP,
            "hsdp": ShardingStrategy.HYBRID_SHARD,
        }[args.data_parallel],
        mixed_precision=MixedPrecision(
            param_dtype={
                "fp32": torch.float, "tf32": torch.float,
                "bf16": torch.bfloat16, "fp16": torch.float16,
            }[args.mixed_precision],
            reduce_dtype={
                "fp32": torch.float, "tf32": torch.float,
                "bf16": torch.bfloat16, "fp16": torch.float16,
            }[args.grad_precision or args.mixed_precision],
        ),
        sync_module_states=True,
        limit_all_gathers=True,
        use_orig_params=True,
    )

    torch.cuda.synchronize()

    return model



def creat_optimizer_by_name(model, weight_decay, learning_rate, betas, global_rank, logger):
    # start with all of the candidate parameters
    all_param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in all_param_dict.items() if p.requires_grad}
    
    # create optim groups. 
    # Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    
    # decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    # nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    
    # model params are flatten by fsdp, we need to set the params by its name
    decay_params = [p for n, p in param_dict.items() if 'norm' not in n]
    nodecay_params = [p for n, p in param_dict.items() if 'norm' in n]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    logger.info(f"(rank {global_rank}) num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    logger.info(f"(rank {global_rank}) num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    print(f"(rank {global_rank}) num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"(rank {global_rank}) num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    extra_args = dict(fused=True) if fused_available else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    logger.info(f"using fused AdamW: {fused_available}")
    return optimizer


def create_scheduler(optimizer, epochs, steps_per_epoch, epoch_range, warmup_epochs, decay_ratio, base_lr, end_lr):
    total_steps = epochs * steps_per_epoch
    segment_boundaries = [0] + epoch_range  # Add a starting point (0) for the first segment
    
    def lr_lambda(current_step):
        epoch_step = current_step // steps_per_epoch
        
        # Find which segment the current step belongs to
        for i in range(1, len(segment_boundaries)):
            if epoch_step < segment_boundaries[i]:
                segment_start_epoch = segment_boundaries[i-1]
                segment_end_epoch = segment_boundaries[i]
                break
        else:
            segment_start_epoch = segment_boundaries[-2]
            segment_end_epoch = epochs

        # Calculate start and end steps for this segment
        segment_start_step = segment_start_epoch * steps_per_epoch
        segment_end_step = segment_end_epoch * steps_per_epoch
        segment_steps = segment_end_step - segment_start_step
        num_decay_steps = segment_steps * decay_ratio
        num_warmup_steps = warmup_epochs * steps_per_epoch
        num_constant_steps = (segment_end_step - segment_start_step) - num_warmup_steps - num_decay_steps

        # 1. Warmup Phase (Linear increase from 0 to base_lr)
        if current_step < segment_start_step + num_warmup_steps:
            return  float(current_step - segment_start_step) / float(max(1, num_warmup_steps))

        # 2. Constant Phase (Constant learning rate at base_lr)
        if current_step < segment_start_step + num_warmup_steps + num_constant_steps:
            return 1.0  # Constant at base_lr during this phase

        # 3. Linear Decay Phase (Linear decay from base_lr to end_lr)
        if current_step < segment_start_step + num_warmup_steps + num_constant_steps + num_decay_steps:
            progress = float(current_step - (segment_start_step + num_warmup_steps + num_constant_steps)) / float(num_decay_steps)
            ratio = max(0.0, 1.0 - progress) 
            return (end_lr + (base_lr - end_lr) * ratio) / base_lr   # Linear decay to end_lr

        return base_lr  # Default return if no phase is found

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    assert args.gpt_type == 'c2i', "FSDP only supports c2i currently."
    # =======================================
    #    Initialize Distributed Training
    # =======================================
    dist.init_process_group("nccl")
    # init_distributed_mode(args)
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    global_rank = dist.get_rank()
    device = global_rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + global_rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={global_rank}, device={device}, seed={seed}, world_size={dist.get_world_size()}.")
    

    # =======================================
    #    Initialize logger and wandb
    # =======================================
    timestamp = None
    if global_rank == 0:
        timestamp = time.localtime()
        timestamp = int(time.strftime("%Y%m%d%H%M%S", timestamp))
    # Convert timestamp to a tensor for broadcasting
    timestamp_tensor = torch.tensor([timestamp] if timestamp is not None else [0.0], dtype=torch.double).to(device)
    # Broadcast the timestamp to all processes
    dist.broadcast(timestamp_tensor, src=0)
    # All processes receive the timestamp
    timestamp = int(timestamp_tensor.item())
    model_string_name = args.gpt_model.replace("/", "-")  # e.g., GPT/XL --> GPT-XL (for naming folders)
    experiment_dir = f"{args.results_dir}/{timestamp}-{model_string_name}"
    cloud_checkpoint_dir = f"{args.cloud_save_path}/{timestamp}-{model_string_name}"
    if global_rank == 0:
        os.makedirs(experiment_dir, exist_ok=True) # in each local machine
        os.makedirs(cloud_checkpoint_dir, exist_ok=True) # in one shared file storage
        logger = create_logger(experiment_dir)
    else:
        logger = create_logger(None)
    logger.info(f"Experiment directory created at {experiment_dir}")
    logger.info(f"Experiment directory created in cloud at {cloud_checkpoint_dir}")

    # training args
    logger.info(f"{args}")  

    # wandb
    if not args.no_wandb and global_rank == 0:
        os.environ["WANDB_DIR"] = experiment_dir   
        wandb.init(
            project=args.wandb_project, 
            name = f"{timestamp}-{model_string_name}",
            config=vars(args)
        )


    # ======================================================
    #     Initialize model and resume
    # ======================================================
    if args.drop_path_rate > 0.0:
        dropout_p = 0.0
    else:
        dropout_p = args.dropout_p
    latent_size = args.image_size // args.downsample_size
    model = GPT_models[args.gpt_model](
        vocab_size=args.vocab_size,
        block_size=latent_size ** 2,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
        resid_dropout_p=dropout_p,
        ffn_dropout_p=dropout_p,
        drop_path_rate=args.drop_path_rate,
        token_dropout_p=args.token_dropout_p,
    ).to(device)
    logger.info(f"GPT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    if args.gpt_resume:
        if global_rank == 0:  # other ranks receive weights in setup_fsdp_sync
            logger.info(f"Resuming model weights from: {args.gpt_resume}")
            model.load_state_dict(torch.load(os.path.join(
                args.gpt_resume, "consolidated.pth",
            ), map_location="cpu"), strict=True)

    model = setup_fsdp_sync(model, args, device)


    # ======================================================
    #     Initialize optimizer and resume
    # ======================================================
    optimizer = creat_optimizer_by_name(model, args.weight_decay, args.lr, (args.beta1, args.beta2), global_rank, logger)
    if args.gpt_resume:
        opt_state_world_size = len([
            x for x in os.listdir(args.gpt_resume)
            if x.startswith("optimizer.") and x.endswith(".pth")
        ])
        assert opt_state_world_size == dist.get_world_size(), (
            f"Resuming from a checkpoint with unmatched world size "
            f"({dist.get_world_size()} vs. {opt_state_world_size}) "
            f"is currently not supported."
        )
        logger.info(f"Resuming optimizer states from: {args.gpt_resume}")
        optimizer.load_state_dict(torch.load(os.path.join(
            args.gpt_resume,
            f"optimizer.{dist.get_rank():05d}-of-"
            f"{dist.get_world_size():05d}.pth",
        ), map_location="cpu"))



    # ======================================================
    #     Initialize Dataloader
    # ======================================================
    dataset = build_dataset(args)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=global_rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    flip_info = 'with' if dataset.flip else 'without'
    aug_info = 10 if 'ten_crop' in dataset.feature_dir else 1
    aug_info = 2 * aug_info if dataset.aug_feature_dir is not None else aug_info
    logger.info(f"Dataset contains {len(dataset):,} images ({args.code_path}) "
                f"{flip_info} flip augmentation and {aug_info} crop augmentation")
    
     #------------------set epoch ranges----
    assert args.patch_level <= 4 
    if args.patch_level == 3:
        patch_sizes = [16, 4, 1]
        epoch_range = [ int(args.epochs * args.epoch_rate ** i) for i in range(args.patch_level)][::-1]
        # epoch_range = [75, 150,300]
    elif args.patch_level == 2:
        patch_sizes = [4, 1]
        epoch_range = [ int(args.epochs * args.epoch_rate ** i) for i in range(args.patch_level)][::-1]
        # epoch_range = [150,300]
    elif args.patch_level == 4:
        patch_sizes = [64, 16, 4, 1]
        epoch_range = [ int(args.epochs * args.epoch_rate ** i) for i in range(args.patch_level)][::-1]
        # epoch_range = [75,150,300]
    else:
         patch_sizes = [1]
         epoch_range = [300]
    
    if args.scheduler:
        lr_scheduler = create_scheduler(optimizer=optimizer, epochs=args.epochs, warmup_epochs=args.warmup_epochs, epoch_range=epoch_range, decay_ratio= args.decay_ratio,
                                        steps_per_epoch=int(len(dataset) / args.global_batch_size), base_lr= args.lr, end_lr=args.end_lr)

    

    def get_patch_size(epoch):
        for idx, (patch_size, epoch_threshold) in enumerate(zip(patch_sizes, epoch_range)):
            if epoch < epoch_threshold:
                # assert patch_size in [1,4,16]
                return patch_size

    # ======================================================
    #   Start training !!!
    # ======================================================
    if args.gpt_resume:
        with open(os.path.join(args.gpt_resume, "resume_step.txt")) as f:
            train_steps = int(f.read().strip())
        start_epoch = int(train_steps / int(len(dataset) / args.global_batch_size))
        train_steps = int(start_epoch * int(len(dataset) / args.global_batch_size))
        logger.info(f"Initial state: steps={train_steps}, epochs={start_epoch}")
    else:
        train_steps = 0
        start_epoch = 0
    
    model.train()  # important! This enables embedding dropout for classifier-free guidance

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_time = time.time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        curr_patchsize = get_patch_size(epoch=epoch)
        p = int(curr_patchsize ** 0.5)
        logger.info(f"Beginning epoch {epoch}..., current patchsize={curr_patchsize} ")
        if not args.rescale_rope:
            if curr_patchsize == 1:
                freqs_cis = precompute_freqs_cis_2d(latent_size , model.module.config.dim // model.module.config.n_head, model.module.config.rope_base, model.module.config.cls_token_num)
            elif curr_patchsize == 4:
                freqs_cis = precompute_freqs_cis_2d(latent_size // 2 , model.module.config.dim // model.module.config.n_head, model.module.config.rope_base, model.module.config.cls_token_num)
            elif curr_patchsize == 16:
                freqs_cis = precompute_freqs_cis_2d(latent_size // 4, model.module.config.dim // model.module.config.n_head, model.module.config.rope_base, model.module.config.cls_token_num)
            elif curr_patchsize == 64:
                freqs_cis = precompute_freqs_cis_2d(latent_size // 8, model.module.config.dim // model.module.config.n_head, model.module.config.rope_base, model.module.config.cls_token_num)
        else:
            assert False
            freqs_cis = precompute_freqs_cis_2d_mean(latent_size, p, model.module.config.dim // model.module.config.n_head, model.module.config.rope_base, model.module.config.cls_token_num)
        
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            z_indices = x.reshape(x.shape[0], -1)
            c_indices = y.reshape(-1)
            assert z_indices.shape[0] == c_indices.shape[0]

            optimizer.zero_grad()
            with {
                "bf16": torch.cuda.amp.autocast(dtype=torch.bfloat16),
                "fp16": torch.cuda.amp.autocast(dtype=torch.float16),
                "fp32": contextlib.nullcontext(),
                "tf32": contextlib.nullcontext(),
            }[args.mixed_precision]: 
                _, loss = model(cond_idx=c_indices, idx=z_indices, targets=z_indices, p=p, patch_size=curr_patchsize, freqs_cis=freqs_cis)
            loss.backward()
            
            if args.max_grad_norm != 0.0:
            #   according to https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.clip_grad_norm_
            #   torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                model.clip_grad_norm_(args.max_grad_norm)
            optimizer.step()
            if args.scheduler:
                lr_scheduler.step()
            

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}, lr:{optimizer.state_dict()['param_groups'][0]['lr']}")
                if not args.no_wandb and global_rank == 0:
                    wandb.log({"train_loss": avg_loss}, step=train_steps)

                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time.time()


            # Save checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                cloud_checkpoint_path = f"{cloud_checkpoint_dir}/{train_steps:07d}"
                os.makedirs(cloud_checkpoint_path, exist_ok=True)

                ### saving model parameters
                with FSDP.state_dict_type(
                    model,
                    StateDictType.FULL_STATE_DICT,
                    FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
                ):
                    consolidated_model_state_dict = model.state_dict()
                    if global_rank == 0:
                        consolidated_fn = "consolidated.pth"
                        torch.save(consolidated_model_state_dict, 
                        os.path.join(cloud_checkpoint_path, consolidated_fn))
                dist.barrier()
                del consolidated_model_state_dict
                logger.info(f"Saved consolidated to {cloud_checkpoint_path}")

                ### saving optimizer
                opt_state_fn = (
                    f"optimizer.{dist.get_rank():05d}-of-"
                    f"{dist.get_world_size():05d}.pth"
                )
                torch.save(optimizer.state_dict(), os.path.join(cloud_checkpoint_path, opt_state_fn))
                dist.barrier()
                logger.info(f"Saved optimizer to {cloud_checkpoint_path}")

                ### saving training step
                if global_rank == 0:
                    with open(os.path.join(cloud_checkpoint_path, "resume_step.txt"), "w") as f:
                        print(train_steps, file=f)
                dist.barrier()
                logger.info(f"Saved training step to {cloud_checkpoint_path}")
                
    if True:
        cloud_checkpoint_path = f"{cloud_checkpoint_dir}/{train_steps:07d}"
        os.makedirs(cloud_checkpoint_path, exist_ok=True)

        ### saving model parameters
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
        ):
            consolidated_model_state_dict = model.state_dict()
            if global_rank == 0:
                consolidated_fn = "consolidated.pth"
                torch.save(consolidated_model_state_dict, 
                os.path.join(cloud_checkpoint_path, consolidated_fn))
        dist.barrier()
        del consolidated_model_state_dict
        logger.info(f"Saved consolidated to {cloud_checkpoint_path}")

        # ### saving optimizer
        # opt_state_fn = (
        #     f"optimizer.{dist.get_rank():05d}-of-"
        #     f"{dist.get_world_size():05d}.pth"
        # )
        # torch.save(optimizer.state_dict(), os.path.join(cloud_checkpoint_path, opt_state_fn))
        # dist.barrier()
        # logger.info(f"Saved optimizer to {cloud_checkpoint_path}")

        ### saving training step
        if global_rank == 0:
            with open(os.path.join(cloud_checkpoint_path, "resume_step.txt"), "w") as f:
                print(train_steps, file=f)
        dist.barrier()
        logger.info(f"Saved training step to {cloud_checkpoint_path}")

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-path", type=str, required=True)
    parser.add_argument("--cloud-save-path", type=str, required=True, help='please specify a cloud disk path, if not, local path')
    parser.add_argument("--no-local-save", action='store_true', help='no save checkpoints to local path for limited disk volume')
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-B")
    parser.add_argument("--gpt-resume", type=str, default=None, help="model, optimizer and argument path for resume training")
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="c2i", help="class-conditional or text-conditional")
    parser.add_argument("--vocab-size", type=int, default=16384, help="vocabulary size of visual tokenizer")
    parser.add_argument("--ema", action='store_true', help="whether using ema training")
    parser.add_argument("--cls-token-num", type=int, default=1, help="max token number of condition input")
    parser.add_argument("--dropout-p", type=float, default=0.1, help="dropout_p of resid_dropout_p and ffn_dropout_p")
    parser.add_argument("--token-dropout-p", type=float, default=0.1, help="dropout_p of token_dropout_p")
    parser.add_argument("--drop-path-rate", type=float, default=0.0, help="using stochastic depth decay")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--dataset", type=str, default='imagenet_code')
    parser.add_argument("--image-size", type=int, choices=[256, 384, 448, 512], default=256)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2, help="Weight decay to use")
    parser.add_argument("--beta1", type=float, default=0.9, help="beta1 parameter for the Adam optimizer")
    parser.add_argument("--beta2", type=float, default=0.95, help="beta2 parameter for the Adam optimizer")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=24)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--mixed-precision", type=str, choices=["fp32", "tf32", "fp16", "bf16"], default='bf16') 
    parser.add_argument("--data-parallel", type=str, choices=["sdp", "fsdp", "hsdp"], default="fsdp")
    parser.add_argument("--grad-precision", type=str, choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--wandb-project", type=str, default='c2i_fsdp')
    parser.add_argument("--no-wandb", action='store_true')

    parser.add_argument("--epoch-rate", type=float, default=0.67)
    parser.add_argument("--patch-level", type=int, default=4)   

    parser.add_argument("--rescale-rope", action='store_true', help='rescale rope') 
    parser.add_argument("--scheduler", action='store_true', help='use lr scheduler') 
    parser.add_argument("--decay-ratio", type=float, default=0.2)
    parser.add_argument("--end-lr", type=float, default=1e-5)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    args = parser.parse_args()
    main(args)
