import os
import numpy as np 
import torch
import argparse
import cv2

from gaussian_splatter import Gaussian_Splatter
from torchmetrics import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
# from tqdm import tqdm


#----------------------------------------------------------------------------------------------


class TrainModule:
    def __init__(self, gaussian_splatter: Gaussian_Splatter, options):
        self.lrs = [options.lr_opacity, options.lr_rgb_color, options.lr_position, options.lr_scale, options.lr_quaternion_rotation]   # gathered


        # Warmup phase : when i(*) <= options.n_its_warm_up
        # Post-warmup phase : when i(*) > options.n_its_warm_up
        dummy = (0.01) ** (1 / (options.n_its - options.n_its_warm_up))
        self.decay_factors = [
            lambda i1: i1 / options.n_its_warm_up if i1 <= options.n_its_warm_up else dummy ** (i1 - options.n_its_warm_up),
            lambda i2: i2 / options.n_its_warm_up if i2 <= options.n_its_warm_up else 1,
            lambda i3: i3 / options.n_its_warm_up if i3 <= options.n_its_warm_up else dummy ** (i3 - options.n_its_warm_up),
            lambda i4: i4 / options.n_its_warm_up if i4 <= options.n_its_warm_up else 1,        
            lambda i5: i5 / options.n_its_warm_up if i5 <= options.n_its_warm_up else 1,
        ]


        self.optimizer = torch.optim.Adam([    # learnable parameters from big learnable tensor container  (Learnables)
                {"params": gaussian_splatter.gaussian_3ds.opacity, "lr": options.lr_opacity * self.decay_factors[0](0)},     # initialize to iteration 0
                {"params": gaussian_splatter.gaussian_3ds.rgb_color, "lr": options.lr_rgb_color * self.decay_factors[1](0)},
                {"params": gaussian_splatter.gaussian_3ds.position, "lr": options.lr_position * self.decay_factors[2](0)},
                {"params": gaussian_splatter.gaussian_3ds.scale, "lr": options.lr_scale * self.decay_factors[3](0)},
                {"params": gaussian_splatter.gaussian_3ds.quaternion_rotation, "lr": options.lr_quaternion_rotation * self.decay_factors[4](0)},
            ],
            betas=(0.9, 0.99),
        )


        if not options.test:
            self.n_cameras = len(gaussian_splatter.imgs)
            self.test_split = np.arange(0, self.n_cameras, options.test_interval)
            self.train_split = np.array(list(set(range(self.n_cameras)) - set(self.test_split)))


        self.ssim_criterion = StructuralSimilarityIndexMeasure(data_range=1.0).to(gaussian_splatter.device)     # torchmetrics
        self.psnr_metrics = PeakSignalNoiseRatio().to(gaussian_splatter.device)     # torchmetrics


        self.l1_losses = np.zeros(options.num_records)
        self.psnr_records = np.zeros(options.num_records)
        self.ssim_losses = np.zeros(options.num_records)


        self.visible_grad_counter = 0    # count how many iterations each Gaussian was visible (based on culling_mask)
        self.curr_max_grad = torch.zeros_like(gaussian_splatter.gaussian_3ds.position)   
        # Each element of curr_max_grad will hold the max absolute gradient seen so far for that Gaussian’s position


    def train_step(self, i_iteration):
        _adaptive_control = (i_iteration > 600 and (i_iteration % options.adaptive_control_interval == 0))
        _adaptive_control_accum_start = i_iteration > 600 and (i_iteration + options.grad_accum_iters - 1) % options.adaptive_control_interval == 0


        self.optimizer.zero_grad()


        # forward
        camera_id = np.random.choice(self.train_split, 1)[0]   # random image for training from set
        rendered_image = gaussian_splatter(camera_id)


        # total loss
        l1_loss = ((rendered_image - gaussian_splatter.ground_truth).abs()).mean()
        ssim_loss = 1.0 - self.ssim_criterion(
            rendered_image.unsqueeze(0).permute(0, 3, 1, 2),    # put into (N, C, H, W) format
            gaussian_splatter.ground_truth.unsqueeze(0).permute(0, 3, 1, 2).to(rendered_image)    # put into (N, C, H, W) format
        )


        total_loss = (0.8 * l1_loss) + (0.2 * ssim_loss)


        # modified total loss
        if options.scale_regularization_factor > 0:      # If gaussians get too large, increase total_loss
            total_loss += options.scale_regularization_factor * gaussian_splatter.gaussian_3ds.scale.abs().mean()
        if options.opacity_regularization_factor > 0:
            opacity_sigmoid = gaussian_splatter.gaussian_3ds.opacity.sigmoid()
            total_loss += options.opacity_regularization_factor * (opacity_sigmoid * (1 - opacity_sigmoid)).mean()


        psnr = self.psnr_metrics(rendered_image, gaussian_splatter.ground_truth)


        total_loss.backward()
        self.optimizer.step()    # optimize step


        self.l1_losses = np.roll(self.l1_losses, 1)     # push oldest one to front to be replaced
        self.psnr_records = np.roll(self.psnr_records, 1)
        self.ssim_losses = np.roll(self.ssim_losses, 1)


        self.l1_losses[0] = l1_loss.item()
        self.psnr_records[0] = psnr.item()     # take scalar off GPU to host float, new one in 
        self.ssim_losses[0] = ssim_loss.item()


        avg_l1_loss = self.l1_losses[:min(i_iteration + 1, self.l1_losses.shape[0])].mean()     # thus far
        avg_ssim_loss = self.ssim_losses[:min(i_iteration + 1, self.ssim_losses.shape[0])].mean()     # thus far
        avg_psnr = self.psnr_records[:min(i_iteration + 1, self.psnr_records.shape[0])].mean()     # thus far


        # grad_info = [
            # gaussian_splatter.gaussian_3ds.opacity.grad.abs().mean(),
            # gaussian_splatter.gaussian_3ds.rgb_color.grad.abs().mean(),
            # gaussian_splatter.gaussian_3ds.position.grad.abs().mean(),
            # gaussian_splatter.gaussian_3ds.scale.grad.abs().mean(),
            # gaussian_splatter.gaussian_3ds.quaternion_rotation.grad.abs().mean(),
        # ]


        if _adaptive_control_accum_start:
            self.curr_max_grad = torch.zeros_like(gaussian_splatter.gaussian_3ds.position)
            self.visible_grad_counter = 0


        self.curr_max_grad = torch.max(gaussian_splatter.gaussian_3ds.position.grad.abs(), self.curr_max_grad)
        self.visible_grad_counter = 1


        if _adaptive_control:
            stats = (self.curr_max_grad / (self.visible_grad_counter + 1e-4)).unsqueeze(-1)    
            # small constant added to prevent division by 0


            gaussian_splatter.gaussian_3ds.adaptive_density_control(
                stats,
                split_or_clone_threshold=options.split_or_clone_threshold, 
                delete_threshold=options.delete_threshold, 
                gradient_threshold=options.gradient_threshold,
                use_clone=options.use_clone if (_adaptive_control) else False, 
                use_split=options.use_split if (_adaptive_control) else False,
                clone_dt=options.clone_dt,
            )


            self.optimizer = torch.optim.Adam([
                    {"params": gaussian_splatter.gaussian_3ds.opacity, "lr": options.lr_opacity * self.decay_factors[0](i_iteration)},
                    {"params": gaussian_splatter.gaussian_3ds.rgb_color, "lr": options.lr_rgb_color * self.decay_factors[1](i_iteration)},
                    {"params": gaussian_splatter.gaussian_3ds.position, "lr": options.lr_position * self.decay_factors[2](i_iteration)},
                    {"params": gaussian_splatter.gaussian_3ds.scale, "lr": options.lr_scale * self.decay_factors[3](i_iteration)},
                    {"params": gaussian_splatter.gaussian_3ds.quaternion_rotation, "lr": options.lr_quaternion_rotation * self.decay_factors[4](i_iteration)},
                ], betas=(0.9, 0.99),
            )


            self.curr_max_grad = torch.zeros_like(gaussian_splatter.gaussian_3ds.position)
            self.visible_grad_counter = 0     # start a fresh accumulation window


        for i_opt, (param_group, lr) in enumerate(zip(self.optimizer.param_groups, self.lrs)):
            param_group['lr'] = self.decay_factors[i_opt](i_iteration) * lr


        return {
            "image": rendered_image,
            "total_loss": (0.8 * avg_l1_loss) + (0.2 * avg_ssim_loss),
            "avg_l1_loss": avg_l1_loss,
            "avg_ssim_loss": avg_ssim_loss,
            "avg_psnr": avg_psnr,
            "n_tile_gaussians": gaussian_splatter.n_tile_gaussians,
            "n_gaussians": gaussian_splatter.n_gaussians,
            # "grad_info": grad_info,
        }


    def train(self):
        # bar = tqdm(range(0, options.n_its))

        for i_iteration in range(options.n_its):
            if i_iteration == 0:
                gaussian_splatter.render_downsample = 4
            elif i_iteration == 250:
                gaussian_splatter.render_downsample = 2
            elif i_iteration == 500:
                gaussian_splatter.render_downsample = 1
 
            
            output = self.train_step(i_iteration)  


            # avg_l1_loss = output["avg_l1_loss"]
            # avg_ssim_loss = output["avg_ssim_loss"]
            # avg_psnr = output["avg_psnr"]
            # n_tile_gaussians = output["n_tile_gaussians"]
            # n_gaussians = output["n_gaussians"]
            # grad_info = output["grad_info"]


            # grad_desc = "[{:.6f}|{:.6f}|{:.6f}|{:.6f}|{:.6f}]".format(*grad_info)    # opacity, rgb_color, position, scale, quaternion_rotation
            
            
            # bar.set_description(
                # desc=f"total_loss: {avg_l1_loss:.6f}/{avg_ssim_loss:.6f}/{avg_psnr:.4f}/[{n_tile_gaussians}/{n_gaussians}]:" +   # not really necessary
                                    # f"""lr: {self.optimizer.param_groups[0]['lr']:.4f}|   
                                            # {self.optimizer.param_groups[1]['lr']:.4f}|
                                            # {self.optimizer.param_groups[2]['lr']:.4f}|
                                            # {self.optimizer.param_groups[3]['lr']:.4f}|
                                            # {self.optimizer.param_groups[4]['lr']:.4f} """ + 
                                    # f"grad: {grad_desc}"
            # )


            rendered_image = output["image"]


            if i_iteration % options.save_image_interval == 0:
                img_npy = rendered_image.clip(0, 1).detach().cpu().numpy()
                os.makedirs("gsplat_output/train_imgs/", exist_ok=True)
                cv2.imwrite(f"gsplat_output/train_imgs/train_{i_iteration}.png", (img_npy * 255).astype(np.uint8)[...,::-1])


                checkpoint = {
                    "position": gaussian_splatter.gaussian_3ds.position,      # parameters required to 3D reconstruct
                    "opacity": gaussian_splatter.gaussian_3ds.opacity,
                    "rgb_color": gaussian_splatter.gaussian_3ds.rgb_color,
                    "quaternion_rotation": gaussian_splatter.gaussian_3ds.quaternion_rotation,
                    "scale": gaussian_splatter.gaussian_3ds.scale,
                }


                torch.save(checkpoint, "gsplat_output/checkpoint.pth")


            if i_iteration % (options.n_iters_test) == 0:
                test_psnrs = []
                test_ssims = []
                elapsed = 0


                for test_camera_id in self.test_split:    # the id number 
                    output = self.test(test_camera_id)
                    elapsed += output["render_time"]
                    test_psnrs.append(output["psnr"])
                    test_ssims.append(output["ssim"])


                    # save images
                    os.makedirs("gsplat_output/test_imgs/", exist_ok=True)
                    img_npy = output["image"].clip(0, 1).detach().cpu().numpy()
                    cv2.imwrite(f"gsplat_output/test_imgs/iter_{i_iteration}_cid_{test_camera_id}.png", (img_npy * 255).astype(np.uint8)[...,::-1])
                

                print("TEST PSNR INDIV: ", test_psnrs)
                
                print("TEST SSIM INDIV: ", test_ssims)
                print("TEST SPLIT PSNR: {:.4f}".format(np.mean(test_psnrs)))
                print("TEST SPLIT SSIM: {:.4f}".format(np.mean(test_ssims)))
                print("RENDERING SPEED: {:.4f}".format(len(self.test_split) / elapsed))


    @torch.no_grad()


    def test(self, camera_id, extrinsics=None, intrinsics=None):
        start_time = torch.cuda.Event(enable_timing=True)
        stop_time = torch.cuda.Event(enable_timing=True)


        start_time.record()


        gaussian_splatter.eval()     # put into evaluation mode
        rendered_image = gaussian_splatter(camera_id, extrinsics, intrinsics)     # index into the dataset


        stop_time.record()


        torch.cuda.synchronize()    # forces CPU to wait until all previously issued CUDA operations on a specific GPU device have completed.


        render_time = start_time.elapsed_time(stop_time) / 1000    # seconds per frame


        if camera_id is not None:
            psnr = self.psnr_metrics(rendered_image, gaussian_splatter.ground_truth).item()     # check what is ground truth
            ssim = self.ssim_criterion(rendered_image.unsqueeze(0).permute(0, 3, 1, 2),     # need to be in (N, C, H, W) format
                                       gaussian_splatter.ground_truth.unsqueeze(0).permute(0, 3, 1, 2).to(rendered_image)).item()   
                                       # ensure on same GPU, .item() pulls scalar off GPU in order to host float 


        gaussian_splatter.train()     # put back into training mode
        output = {"image": rendered_image}


        if camera_id is not None:
            output.update({
                "psnr": psnr, 
                "ssim": ssim,
                "render_time": render_time,
            })


        return output     # returns the image, psnr, ssim, render_time
    

#-------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_its", type=int, default=7001)
    parser.add_argument("--n_its_warm_up", type=int, default=300)
    parser.add_argument("--n_iters_test", type=int, default=200)
    parser.add_argument("--num_records", type=int, default=100)
    parser.add_argument("--save_image_interval", type=int, default=100)
    parser.add_argument("--adaptive_control_interval", type=int, default=100)      # densify every 100 iterations
    parser.add_argument("--data", type=str, default="reconstruction/dataset/raw/")   # CHECK!
    parser.add_argument("--scale_init_value", type=float, default=1)
    parser.add_argument("--opacity_init_value", type=float, default=0.3)


    parser.add_argument("--lr_opacity", type=float, default=0.03)
    parser.add_argument("--lr_rgb_color", type=float, default=0.003)
    parser.add_argument("--lr_position", type=float, default=0.03)
    parser.add_argument("--lr_scale", type=float, default=0.03)
    parser.add_argument("--lr_quaternion_rotation", type=float, default=0.003)


    parser.add_argument("--delete_threshold", type=float, default=1.5)
    parser.add_argument("--split_or_clone_threshold", type=float, default=0.05)
    parser.add_argument("--spherical_harmonics", type=int, default=0)
    parser.add_argument("--scale_regularization_factor", type=float, default=0)
    parser.add_argument("--opacity_regularization_factor", type=float, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--checkpoint", type=str, default="")


    parser.add_argument("--grad_accum_iters", type=int, default=50)
    parser.add_argument("--gradient_threshold", type=float, default=0.0002)
    parser.add_argument("--use_clone", type=int, default=0)
    parser.add_argument("--use_split", type=int, default=1)
    parser.add_argument("--clone_dt", type=float, default=0.01)


    parser.add_argument("--test", default=0, type=int)
    parser.add_argument("--test_interval", default=8, type=int)


    options = parser.parse_args()


    np.random.seed(options.seed)


    data_path = os.path.join(options.data, 'bins')    # CHECK!
    img_path = os.path.join(options.data, '35degree-4um-128p-0505')    # CHECK! 


    if options.checkpoint == "":
        options.checkpoint = None


    gaussian_splatter = Gaussian_Splatter(
        data_path,
        img_path, 
        spherical_harmonics=options.spherical_harmonics,
        scale_init_value=options.scale_init_value,
        opacity_init_value=options.opacity_init_value,
        load_checkpoint=options.checkpoint,
        test=options.test,
    )


    trainer = TrainModule(gaussian_splatter, options)
    trainer.train()
