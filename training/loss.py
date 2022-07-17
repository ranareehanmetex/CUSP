# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils import persistence
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix

from scipy.interpolate import interp1d
from scipy.stats import norm

import dnnlib

import torchvision.transforms as T
import pandas as pd

from training.tools import batch_blur_quick
from training.dataset import AgeDataset


# ----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain):  # to be overridden by subclass
        raise NotImplementedError()


# ----------------------------------------------------------------------------
#
# class StyleGAN2Loss(Loss):
#     def __init__(self, device, G_mapping, G_synthesis, D, augment_pipe=None, style_mixing_prob=0.9, r1_gamma=10,
#                  pl_batch_shrink=2, pl_decay=0.01, pl_weight=2):
#         super().__init__()
#         self.device = device
#         self.G_mapping = G_mapping
#         self.G_synthesis = G_synthesis
#         self.D = D
#         self.augment_pipe = augment_pipe
#         self.style_mixing_prob = style_mixing_prob
#         self.r1_gamma = r1_gamma
#         self.pl_batch_shrink = pl_batch_shrink
#         self.pl_decay = pl_decay
#         self.pl_weight = pl_weight
#         self.pl_mean = torch.zeros([], device=device)
#
#     def run_G(self, z, c, sync):
#         with misc.ddp_sync(self.G_mapping, sync):
#             ws = self.G_mapping(z, c)
#             if self.style_mixing_prob > 0:
#                 with torch.autograd.profiler.record_function('style_mixing'):
#                     cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
#                     cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff,
#                                          torch.full_like(cutoff, ws.shape[1]))
#                     ws[:, cutoff:] = self.G_mapping(torch.randn_like(z), c, skip_w_avg_update=True)[:, cutoff:]
#         with misc.ddp_sync(self.G_synthesis, sync):
#             img = self.G_synthesis(ws)
#         return img, ws
#
#     def run_D(self, img, c, sync):
#         if self.augment_pipe is not None:
#             img = self.augment_pipe(img)
#         with misc.ddp_sync(self.D, sync):
#             logits = self.D(img, c)
#         return logits
#
#     def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain):
#         assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
#         do_Gmain = (phase in ['Gmain', 'Gboth'])
#         do_Dmain = (phase in ['Dmain', 'Dboth'])
#         do_Gpl = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0)
#         do_Dr1 = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)
#
#         # Gmain: Maximize logits for generated images.
#         if do_Gmain:
#             with torch.autograd.profiler.record_function('Gmain_forward'):
#                 gen_img, _gen_ws = self.run_G(gen_z, gen_c, sync=(sync and not do_Gpl))  # May get synced by Gpl.
#                 gen_logits = self.run_D(gen_img, gen_c, sync=False)
#                 training_stats.report('Loss/scores/fake', gen_logits)
#                 training_stats.report('Loss/signs/fake', gen_logits.sign())
#                 loss_Gmain = torch.nn.functional.softplus(-gen_logits)  # -log(sigmoid(gen_logits))
#                 training_stats.report('Loss/G/loss', loss_Gmain)
#             with torch.autograd.profiler.record_function('Gmain_backward'):
#                 loss_Gmain.mean().mul(gain).backward()
#
#         # Gpl: Apply path length regularization.
#         if do_Gpl:
#             with torch.autograd.profiler.record_function('Gpl_forward'):
#                 batch_size = gen_z.shape[0] // self.pl_batch_shrink
#                 gen_img, gen_ws = self.run_G(gen_z[:batch_size], gen_c[:batch_size], sync=sync)
#                 pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
#                 with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients():
#                     pl_grads = \
#                     torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True,
#                                         only_inputs=True)[0]
#                 pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
#                 pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
#                 self.pl_mean.copy_(pl_mean.detach())
#                 pl_penalty = (pl_lengths - pl_mean).square()
#                 training_stats.report('Loss/pl_penalty', pl_penalty)
#                 loss_Gpl = pl_penalty * self.pl_weight
#                 training_stats.report('Loss/G/reg', loss_Gpl)
#             with torch.autograd.profiler.record_function('Gpl_backward'):
#                 (gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()
#
#         # Dmain: Minimize logits for generated images.
#         loss_Dgen = 0
#         if do_Dmain:
#             with torch.autograd.profiler.record_function('Dgen_forward'):
#                 gen_img, _gen_ws = self.run_G(gen_z, gen_c, sync=False)
#                 gen_logits = self.run_D(gen_img, gen_c, sync=False)  # Gets synced by loss_Dreal.
#                 training_stats.report('Loss/scores/fake', gen_logits)
#                 training_stats.report('Loss/signs/fake', gen_logits.sign())
#                 loss_Dgen = torch.nn.functional.softplus(gen_logits)  # -log(1 - sigmoid(gen_logits))
#             with torch.autograd.profiler.record_function('Dgen_backward'):
#                 loss_Dgen.mean().mul(gain).backward()
#
#         # Dmain: Maximize logits for real images.
#         # Dr1: Apply R1 regularization.
#         if do_Dmain or do_Dr1:
#             name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
#             with torch.autograd.profiler.record_function(name + '_forward'):
#                 real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
#                 real_logits = self.run_D(real_img_tmp, real_c, sync=sync)
#                 training_stats.report('Loss/scores/real', real_logits)
#                 training_stats.report('Loss/signs/real', real_logits.sign())
#
#                 loss_Dreal = 0
#                 if do_Dmain:
#                     loss_Dreal = torch.nn.functional.softplus(-real_logits)  # -log(sigmoid(real_logits))
#                     training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)
#
#                 loss_Dr1 = 0
#                 if do_Dr1:
#                     with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
#                         r1_grads = \
#                         torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True,
#                                             only_inputs=True)[0]
#                     r1_penalty = r1_grads.square().sum([1, 2, 3])
#                     loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
#                     training_stats.report('Loss/r1_penalty', r1_penalty)
#                     training_stats.report('Loss/D/reg', loss_Dr1)
#
#             with torch.autograd.profiler.record_function(name + '_backward'):
#                 (real_logits * 0 + loss_Dreal + loss_Dr1).mean().mul(gain).backward()


# ----------------------------------------------------------------------------

from training.networks import VGG
import torch.nn.functional as F
from torch import Tensor


# Mean-var loss Pan, H., Han, H., Shan, S., & Chen, X. (2018). Mean-variance loss for deep age estimation
# from a face. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition
class MeanVarLoss(torch.nn.Module):
    def __init__(self, m_lambda=0.2, v_lambda=0.05, ce_lambda=1., reduction='mean', **kwargs):
        super().__init__(**kwargs)
        self.m_lambda = m_lambda
        self.v_lambda = v_lambda
        self.ce_lambda = ce_lambda
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        target_int = torch.where(target)[1].to(input.device)
        age_enum = torch.arange(101)[None, :].type_as(input)
        # Probabilities
        psm = F.softmax(input, 1)
        # Calc age mean
        m = torch.sum(psm * age_enum, 1)

        # mean loss
        mloss = ((m - target_int) ** 2) / 2

        # variance loss
        vloss = (psm * (age_enum - m[:, None]) ** 2).sum(1)

        # cross-entropu loss
        celoss = F.cross_entropy(input, target_int, reduction='none')

        floss = mloss * self.m_lambda + vloss * self.v_lambda + celoss * self.ce_lambda
        if self.reduction != 'none':
            floss = getattr(floss, self.reduction)()
        return floss


# def get_activation(name, dic):
#     def hook(model, input, output):
#         dic[name] = output
#         return output
#
#     return hook
class ForwardHook:
    def __init__(self):
        self.do_add = False
        self.fw_list = []

    def __call__(self, model, input, output):
        if self.do_add:
            self.fw_list += [output]

    def start(self):
        self.do_add = True

    def clear(self):
        self.do_add = False
        self.fw_list = []


class RandomClassModifier:
    def __call__(self, labels):
        raise NotImplementedError





class RandomAge(RandomClassModifier):
    def __init__(self, age_min, age_max):
        self.min_n = age_min
        self.max_n = age_max

    def __call__(self, labels):
        tage = torch.zeros_like(labels)
        tage[
            torch.arange(labels.size(0)),
            torch.randint(low=self.min_n, high=self.max_n, size=(labels.size(0),))
        ] = 1.
        return tage


class SoftMarginRandomAge(RandomClassModifier):
    @staticmethod
    def build_sampler(a, age_min, age_max, bra):
        bins = 100
        awidth = age_max - age_min
        m = ((a - age_min) / awidth) * 2 * bra - bra
        z1 = norm.pdf(np.linspace(-bra, bra, bins), m, 1)
        z1 = z1.max() - z1

        z1s = np.cumsum(z1)
        z1s /= z1s[-1]

        f = interp1d(np.concatenate([[0], z1s]), np.linspace(0, 1, bins + 1) * awidth + age_min)
        return f

    def __init__(self, age_min, age_max, bra=3):
        self.adict = dict([
            (a, SoftMarginRandomAge.build_sampler(a, age_min, age_max, bra))
            for a in range(age_min, age_max)
        ])

    def __call__(self, labels):
        # matrix to int
        ages = torch.where(labels)[1]
        device = labels.device
        tage = np.array([
            self.adict[i.cpu().item()](r)
            for i, r in zip(ages, np.random.rand(len(ages)))
        ]).astype(int)
        tage_int = torch.tensor(tage)
        # targe age int to matrix
        tage = torch.zeros_like(labels)
        tage[torch.arange(len(tage_int)), tage_int] = 1

        return tage


class HardRandomAge(RandomClassModifier):
    def __init__(self, age_margin, age_min, age_max):
        self.age_margin = age_margin
        self.age_max = age_max
        self.age_min = age_min

    def __call__(self, labels):
        ages = torch.where(labels)[1]
        diff_val = self.age_margin
        device = labels.device
        age_max = torch.tensor(self.age_max).type_as(labels)
        age_min = torch.tensor(self.age_min).type_as(labels)

        if diff_val > (age_max - age_min) / 2:
            diff_val = (age_max - age_min) // 2

        low_n = torch.maximum(ages + 1 - diff_val, age_min) - age_min
        up_n = (age_max) - torch.minimum(ages + diff_val, age_max)

        off_a = (torch.rand(low_n.size()).to(device) * (up_n + low_n)).int()

        age_output = torch.where(
            off_a < low_n,
            ages - diff_val - off_a,
            ages + diff_val + off_a - low_n)

        return age_output


class ChangeOne(RandomClassModifier):
    def __call__(self, labels):
        labels = labels.detach().clone()
        larange = torch.arange(labels.size(0))
        rand_idx = torch.randint(high=labels.size(1), size=(labels.size(0),))
        labels[larange, rand_idx] = 1 - labels[larange, rand_idx]
        return labels

    def get_eval_labels(self, labels):
        all_labels = np.repeat(labels[None, :], labels.size + 1, axis=0)
        all_labels[1 + np.arange(labels.size), np.arange(labels.size)] = 1 - all_labels[
            np.arange(labels.size), np.arange(labels.size)]
        return all_labels


class ChangeAll(RandomClassModifier):
    def __call__(self, labels):
        labels = torch.zeros_like(labels)
        rand_idx = torch.randint(high=labels.size(1), size=(labels.size(0),))
        labels[torch.arange(labels.size(0)), rand_idx] = 1
        return labels

    def get_eval_labels(self, labels):
        return np.identity(labels.size, dtype=labels.dtype)


@persistence.persistent_class
class FrozenClassifier:
    def __init__(self, outclass):
        if outclass == 'one':
            self.topclass = self.top_quantile
        if outclass == 'max':
            self.topclass = self.top_one
        elif outclass == 'all':
            self.topclass = self.top_all
        else:
            raise NotImplementedError

    def top_quantile(self, x):
        return x[(x > torch.quantile(x, 0.9, dim=1, keepdim=True)).detach()]

    def top_one(self, x):
        return x.amax(1, keepdim=True)

    def top_all(self, x):
        return x

    """
    input:
        img: Tensor(float32) (B,C,W,H)
    """

    def __call__(self, img):
        raise NotImplementedError

    def get_classifier(self):
        raise NotImplementedError

    def get_hook(self):
        raise NotImplementedError

    def to(self, device):
        raise NotImplementedError

    def zero_grad(self, *args, **kwargs):
        raise NotImplementedError

    def __deepcopy__(self, memodict={}):
        raise NotImplementedError

    @property
    def device(self):
        raise NotImplementedError


@persistence.persistent_class
class DEXAgeClassifier(FrozenClassifier):

    def __init__(self, vgg_path, own_relu=False, outclass='all'):
        super().__init__(outclass=outclass)
        self.outclass = outclass
        self.vgg_path = vgg_path
        self.vgg_own_relu = own_relu
        self.classifier = VGG(own_relu=own_relu)
        vgg_state_dict = torch.load(vgg_path)
        vgg_state_dict = {k.replace('-', '_'): v for k, v in vgg_state_dict.items()}
        self.classifier.load_state_dict(vgg_state_dict)

    def __deepcopy__(self, memodict={}):
        return DEXAgeClassifier(self.vgg_path, own_relu=self.vgg_own_relu, outclass=self.outclass)

    def __call__(self, img, do_softmax=True):
        # Img (batch, c, w, h) -> RGB[-1,1] to BGR[0,255]
        img = (F.interpolate(img[:, [2, 1, 0]], size=(224, 224), mode='bilinear', align_corners=False) + 1) * 127.5
        # Substract ImageNet mean
        imnet_mean = torch.tensor([0.48501961, 0.45795686, 0.40760392]).type_as(img).view(1, -1, 1, 1) * 255.
        img = img - imnet_mean

        age_pb = self.classifier(img)['fc8']
        # age_enum = torch.arange(101)[None, :].type_as(age_pb)

        age_pred = F.softmax(age_pb, 1) if do_softmax else age_pb
        # return sm_age, torch.sum(sm_age * age_enum, 1)
        return age_pred

    def get_hook(self):
        return self.classifier.conv5_3

    def get_classifier(self):
        return self.classifier

    @property
    def device(self):
        return next(self.classifier.parameters()).device

    def to(self, device):
        self.classifier = self.classifier.to(device)
        return self

    def zero_grad(self, *args, **kwargs):
        return self.classifier.zero_grad(*args, **kwargs)


class MultiHead(torch.nn.Module):
    def __init__(self, nout, input_size=2048, hidden_size=128):
        super(MultiHead, self).__init__()
        self.nout = nout
        for n in range(nout):
            setattr(self, f'fc_{n}', torch.nn.Sequential(
                torch.nn.Linear(input_size, hidden_size),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(hidden_size, 1)
            ))

    def forward(self, x):
        x = torch.cat([
            getattr(self, f'fc_{n}')(x) for n in range(self.nout)
        ], axis=1)
        return x


@persistence.persistent_class
class ResNetClassifier(FrozenClassifier):
    def __init__(self, model_path, mask_path=None, outclass='one'):
        super().__init__(outclass=outclass)
        self.args = (model_path, mask_path, outclass)
        self.classifier = torch.load(model_path, map_location='cpu')
        self.normalize = T.Compose([
            T.Normalize(mean=[-1, -1, -1],
                        std=[2, 2, 2]),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])])
        self.classifier_mask = pd.read_csv(mask_path, header=None).to_numpy().ravel() if mask_path else None

    def __deepcopy__(self, memodict={}):
        return ResNetClassifier(*self.args)

    def to(self, device):
        self.classifier = self.classifier.to(device)
        return self

    def get_classifier(self):
        return self.classifier

    def get_hook(self):
        return self.classifier.layer4[2].conv3

    @property
    def device(self):
        return next(self.classifier.parameters()).device

    def __call__(self, img, *args, **kwargs):
        p = self.classifier(self.normalize(img))
        if self.classifier_mask is not None:
            p = p[:, self.classifier_mask]
        return p

    def zero_grad(self, *args, **kwargs):
        return self.classifier.zero_grad(*args, **kwargs)


@persistence.persistent_class
class VGGBNClassifier(FrozenClassifier):
    def __init__(self, model_path, outclass='one'):
        super().__init__(outclass=outclass)
        self.args = (model_path, outclass)
        self.classifier = torch.load(model_path, map_location='cpu')
        self.normalize = T.Compose([
            T.Normalize(mean=[-1, -1, -1],
                        std=[2, 2, 2]),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])])

    def __deepcopy__(self, memodict={}):
        return VGGBNClassifier(*self.args)

    def to(self, device):
        self.classifier = self.classifier.to(device)
        return self

    def get_classifier(self):
        return self.classifier

    def get_hook(self):
        return self.classifier.features[40]

    @property
    def device(self):
        return next(self.classifier.parameters()).device

    def __call__(self, img, *args, **kwargs):
        p = self.classifier(self.normalize(img))
        return p

    def zero_grad(self, *args, **kwargs):
        return self.classifier.zero_grad(*args, **kwargs)


class I2ILoss(Loss):
    def __init__(self, device, G, D,
                 classifier_kwargs, classification_loss_kwargs, random_class_kwargs,
                 loss_weights, disc_class, r1_gamma=10, rgb_reg=False, act_reg=None, mixing_probability=0.):
        super().__init__()
        self.device = device
        self.G = G
        self.D = D
        self.r1_gamma = r1_gamma

        self.loss_weights = loss_weights

        self.clearable = []

        self.classifier = dnnlib.util.construct_class_by_name(**classifier_kwargs).to(device)

        # Reconstruction loss
        self.recloss = torch.nn.L1Loss(reduction='mean')
        self.celoss = dnnlib.util.construct_class_by_name(**classification_loss_kwargs)

        self.random_class = dnnlib.util.construct_class_by_name(**random_class_kwargs)

        # RGB Regularization
        self.rgb_reg = rgb_reg
        self.fw_hook = ForwardHook()
        self.clearable.append(self.fw_hook)

        if self.rgb_reg:
            self.rgb_hooks = []
            self.activation = {}

            for _, m in [x for x in G.named_modules() if x[0][-5:] == 'torgb']:
                self.rgb_hooks.append(m.register_forward_hook(self.fw_hook))

        # Activation Regularization
        self.act_reg = act_reg
        if act_reg in ['l1', 'l2']:
            self.act_reg = ForwardHook()
            self.clearable.append(self.act_reg)

            self.act_hooks = []
            self.act_f = (lambda x: x.abs()) if act_reg == 'l1' else (lambda x: x ** 2)
            # [Content_encoder, Style_encoder & Image_decoder(conv)(NOT toRGB)]
            for _, m in [x for x in G.named_modules() if x[0][-5:-1] == 'conv']:
                self.act_hooks.append(m.register_forward_hook(self.act_reg))
        elif act_reg is not None:
            raise NotImplementedError

        self.mixing_probability = mixing_probability
        self.disc_class = disc_class

    def clear(self):
        for e in self.clearable:
            e.clear()

    def run_G(self, img, age, sync, return_msk=False, style_mixing=False):
        with misc.ddp_sync(self.G, sync):
            imgp = self.G(img, age, return_msk=return_msk,
                          style_mixing=style_mixing and (torch.rand([]) < self.mixing_probability))
        return imgp

    def run_D(self, img, c, sync):
        with misc.ddp_sync(self.D, sync):
            logits = self.D(img, c)
        return logits

    def _compute_loss_smooth(self, mat):
        return torch.sum(torch.abs(mat[:, :, :, :-1] - mat[:, :, :, 1:])) + \
               torch.sum(torch.abs(mat[:, :, :-1, :] - mat[:, :, 1:, :]))

    def accumulate_gradients(self, phase, real_img, real_class, sync, gain):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Dr1 = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)

        fake_class = self.random_class(real_class)

        self.fw_hook.start()

        if self.act_reg is not None:
            self.act_reg.start()

        # Gmain: Maximize logits for generated images.
        if do_Gmain:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                # Reconstruct
                rec_gen_img, rec_msk, rec_cam = self.run_G(real_img, real_class, sync=(sync), return_msk=True,
                                                           style_mixing=False)
                # Modify age
                mod_gen_img, mod_msk, mod_cam = self.run_G(real_img, fake_class, sync=(sync), return_msk=True,
                                                           style_mixing=False)

                # Cycle loss
                cyc_gen_img, cyc_msk, cyc_cam = self.run_G(mod_gen_img, real_class, sync=(sync), return_msk=True,
                                                           style_mixing=False)

                class_prediction = self.classifier(mod_gen_img)

                # Classification loss
                cla_loss = self.celoss(class_prediction, fake_class)
                # Reconstruction loss
                rec_loss = self.recloss(real_img, rec_gen_img)
                ksize = real_img.size(-1) // 6 * 2 + 1
                rec_fake_loss = self.recloss(batch_blur_quick(real_img, ksize), mod_gen_img)
                # Adversarial loss
                gen_logits = self.run_D(mod_gen_img, fake_class, sync=False)
                adv_loss = torch.nn.functional.softplus(-gen_logits).mean()  # -log(sigmoid(gen_logits))
                # Cycle consistency loss
                cycle_loss = self.recloss(real_img, cyc_gen_img)
                # RGB loss
                rgb_loss = 0.
                if self.rgb_reg is not None:
                    std_list = torch.stack([e.std() for e in self.fw_hook.fw_list])
                    if self.rgb_reg == 'margin':
                        std_min = std_list.min().item() * 3
                        std_list = torch.where(std_list > std_min, std_list, torch.zeros_like(std_list)) ** 2
                    elif self.rgb_reg == 'std2':
                        std_list = std_list ** 2
                    else:
                        raise NotImplementedError()

                    rgb_loss = std_list.mean()

                # self.fw_hook.clear()

                act_loss = 0.
                if self.act_reg is not None:
                    act_loss = torch.stack([self.act_f(e).mean() for e in self.act_reg.fw_list]).mean()
                    # self.act_reg.clear()

                self.clear()

                msk_loss = 0.
                # GANimation: Anatomically-aware Facial Animation from a Single Image CASE
                if (rec_msk is not None) and (rec_cam is None):
                    loss_mask_1 = torch.mean(rec_msk)
                    loss_mask_2 = torch.mean(mod_msk)
                    loss_mask_1_smooth = self._compute_loss_smooth(rec_msk)
                    loss_mask_2_smooth = self._compute_loss_smooth(mod_msk)

                    msk_loss = self.loss_weights.msk * (loss_mask_1 + loss_mask_2) + \
                               self.loss_weights.msk_smooth * (loss_mask_1_smooth + loss_mask_2_smooth)

                # Attribute Manipulation Generative Adversarial Networks for Fashion Images CASE
                if rec_cam is not None:  # Forget previous step &
                    loss_mask_1 = torch.nn.functional.l1_loss(rec_msk, F.interpolate(rec_cam, size=rec_msk.shape[-2:],
                                                                                     mode='bicubic',
                                                                                     align_corners=False))
                    loss_mask_2 = torch.nn.functional.l1_loss(mod_msk, F.interpolate(mod_cam, size=rec_msk.shape[-2:],
                                                                                     mode='bicubic',
                                                                                     align_corners=False))

                    msk_loss = self.loss_weights.cam * (loss_mask_1 + loss_mask_2) / 2.

                # Total loss
                total_loss = \
                    self.loss_weights.rec * rec_loss + \
                    self.loss_weights.cla * cla_loss + \
                    self.loss_weights.adv * adv_loss + \
                    self.loss_weights.rgb * rgb_loss + \
                    self.loss_weights.act * act_loss + \
                    self.loss_weights.fre * rec_fake_loss + \
                    self.loss_weights.cycle * cycle_loss + \
                    msk_loss

                # Log loss
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())

                training_stats.report('Loss/G/cla_loss', cla_loss)
                training_stats.report('Loss/G/rec_loss', rec_loss)
                training_stats.report('Loss/G/adv_loss', adv_loss)
                training_stats.report('Loss/G/rgb_loss', rgb_loss)
                training_stats.report('Loss/G/total_loss', total_loss)

            with torch.autograd.profiler.record_function('Gmain_backward'):
                total_loss.mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if do_Dmain:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                # Modify age
                mod_gen_img = self.run_G(real_img, fake_class, sync=(sync), return_msk=False, style_mixing=False)
                gen_logits = self.run_D(mod_gen_img, fake_class, sync=False)  # Gets synced by loss_Dreal.
                self.clear()

                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits)  # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain or do_Dr1:
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
                real_logits = self.run_D(real_img_tmp, real_class, sync=sync)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if do_Dmain:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits)  # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if do_Dr1:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = \
                        torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True,
                                            only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1, 2, 3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (real_logits * 0 + loss_Dreal + loss_Dr1).mean().mul(gain).backward()

# ----------------------------------------------------------------------------