from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as func


from accelerator.utilities.logging import get_logger
from accelerator.utilities.default_config import _DefaultConfig, dataclass

log = get_logger(__name__)


@dataclass
class GammaDefaultConfig(_DefaultConfig):
      use_gamma: bool = False
      gamma_type: int = 1430
      experimental_gamma: bool = False
      experimental_gamma_forward: bool = False
      experimental_gamma_resolution: int = 1000
      experimental_gamma_forward_num_frames_channels: int = 32
      experimental_gamma_forward_num_meta_channels: int = 10
      experimental_gamma_forward_bicubic: bool = False
      experimental_gamma_forward_skip_frames_channels: Optional[List[int]] = None
      experimental_gamma_forward_skip_meta_channels: Optional[List[int]] = None
      experimental_gamma_forward_apply_to_meta: bool = True
      experimental_gamma_forward_clamp_grid_sample: bool = False
      experimental_gamma_forward_monotonous: bool = True
      experimental_gamma_forward_trainable_endpoints: bool = False
      experimental_gamma_inverse: bool = False
      experimental_gamma_inverse_resolution: int = 1000
      experimental_gamma_inverse_num_channels: int = 3
      experimental_gamma_inverse_bicubic: bool = False
      experimental_gamma_inverse_skip_channels: Optional[List] = None
      experimental_gamma_inverse_clamp_grid_sample: bool = False
      experimental_gamma_inverse_monotonous: bool = True
      experimental_gamma_inverse_trainable_endpoints: bool = False


class Gamma(torch.nn.Module):

      def __init__(self, model_config):
            super(Gamma, self).__init__()
            self.model_config = model_config
            self._gamma_forward_bck = None
            self._gamma_inverse_bck = None

            self.frame_colors = 'RGB'
            self.inds_colors = {'R': 1, 'G': 2, 'B': 3}
            self.inds_frames = {
                  'R': range(0, 32, 4),
                  'B': range(3, 32, 4),
                  'G': list(set(range(0, 32)).difference(set(range(0, 32, 4))).difference(set(range(3, 32, 4))))
            }
            self.frame_exposures = 'NNNNNSSL'
            self.inds_exposure = {'N': 1, 'S': 2, 'L': 3}


            if model_config['use_gamma']:
                  if model_config['gamma_type'] == 843 or model_config['gamma_type'] == 743:
                        # ################## Polynomial transformation #######################################################

                        if model_config['gamma_type'] == 743:
                              log.info('WARNING: gamma type 743 is deprecated. Gamma 843 will be used instead')

                        init_g4 = [-0.2982, 0.9369, -1.8118, 2.1683]
                        init_d3 = [0.2430, 0.7012, 0.0171]
                        
                        if not model_config['experimental_gamma_forward'] and not model_config['equalization_forward']:
                              self.b1_arr = torch.nn.Parameter(torch.ones(4) * init_g4[0], requires_grad=True)
                              self.c1_arr = torch.nn.Parameter(torch.ones(4) * init_g4[1], requires_grad=True)
                              self.d1_arr = torch.nn.Parameter(torch.ones(4) * init_g4[2], requires_grad=True)
                              self.e1_arr = torch.nn.Parameter(torch.ones(4) * init_g4[3], requires_grad=True)
                              self._gamma_forward = self._gamma_forward_polynomial_splitted_upd
                        if not model_config['experimental_gamma_inverse']:
                              self.c2_arr = torch.nn.Parameter(torch.ones(3) * init_d3[0], requires_grad=True)
                              self.d2_arr = torch.nn.Parameter(torch.ones(3) * init_d3[1], requires_grad=True)
                              self.e2_arr = torch.nn.Parameter(torch.ones(3) * init_d3[2], requires_grad=True)                        
                              self._gamma_inverse = self._gamma_inverse_polynomial_splitted

                  if model_config['gamma_type'] == 943:
                        # ################## Polynomial transformation, exposure-dependent ###################################

                        init_g4 = [-0.2982, 0.9369, -1.8118, 2.1683]
                        init_d3 = [0.2430, 0.7012, 0.0171]

                        if not model_config['experimental_gamma_forward'] and not model_config['equalization_forward']:
                              self.b1_arr = torch.nn.Parameter(torch.ones(4) * init_g4[0], requires_grad=True)
                              self.c1_arr = torch.nn.Parameter(torch.ones(4) * init_g4[1], requires_grad=True)
                              self.d1_arr = torch.nn.Parameter(torch.ones(4) * init_g4[2], requires_grad=True)
                              self.e1_arr = torch.nn.Parameter(torch.ones(4) * init_g4[3], requires_grad=True)
                              self._gamma_forward = self._gamma_forward_polynomial_exposure_splitted
                        if not model_config['experimental_gamma_inverse']:
                              self.c2_arr = torch.nn.Parameter(torch.ones(3) * init_d3[0], requires_grad=True)
                              self.d2_arr = torch.nn.Parameter(torch.ones(3) * init_d3[1], requires_grad=True)
                              self.e2_arr = torch.nn.Parameter(torch.ones(3) * init_d3[2], requires_grad=True)
                              self._gamma_inverse = self._gamma_inverse_polynomial_splitted

                  if model_config['gamma_type'] == 945:
                        # ################## Polynomial transformation, exposure-dependent ###################################

                        init_g4 = [-0.2982, 0.9369, -1.8118, 2.1683]
                        init_d3 = [0.2430, 0.7012, 0.0171]

                        if not model_config['experimental_gamma_forward'] and not model_config['equalization_forward']:
                              self.b1_arr = torch.nn.Parameter(torch.ones(4) * init_g4[0], requires_grad=True)
                              self.c1_arr = torch.nn.Parameter(torch.ones(4) * init_g4[1], requires_grad=True)
                              self.d1_arr = torch.nn.Parameter(torch.ones(4) * init_g4[2], requires_grad=True)
                              self.e1_arr = torch.nn.Parameter(torch.ones(4) * init_g4[3], requires_grad=True)
                              self._gamma_forward = self._gamma_forward_polynomial_exposure_splitted_nometa
                        if not model_config['experimental_gamma_inverse']:
                              self.c2_arr = torch.nn.Parameter(torch.ones(3) * init_d3[0], requires_grad=True)
                              self.d2_arr = torch.nn.Parameter(torch.ones(3) * init_d3[1], requires_grad=True)
                              self.e2_arr = torch.nn.Parameter(torch.ones(3) * init_d3[2], requires_grad=True)
                              self._gamma_inverse = self._gamma_inverse_polynomial_splitted

                  if model_config['gamma_type'] == 947:
                        # ################## Polynomial transformation, exposure-dependent ###################################

                        init_g4 = [-0.2982, 0.9369, -1.8118, 2.1683]
                        init_d3 = [0.2430, 0.7012, 0.0171]

                        if not model_config['experimental_gamma_forward'] and not model_config['equalization_forward']:
                              self.b1_arr = torch.nn.Parameter(torch.ones(4) * init_g4[0], requires_grad=True)
                              self.c1_arr = torch.nn.Parameter(torch.ones(4) * init_g4[1], requires_grad=True)
                              self.d1_arr = torch.nn.Parameter(torch.ones(4) * init_g4[2], requires_grad=True)
                              self.e1_arr = torch.nn.Parameter(torch.ones(4) * init_g4[3], requires_grad=True)
                              self._gamma_forward = self._gamma_forward_polynomial_exposure_splitted_nometa_abs_nomask
                        if not model_config['experimental_gamma_inverse']:
                              self.c2_arr = torch.nn.Parameter(torch.ones(3) * init_d3[0], requires_grad=True)
                              self.d2_arr = torch.nn.Parameter(torch.ones(3) * init_d3[1], requires_grad=True)
                              self.e2_arr = torch.nn.Parameter(torch.ones(3) * init_d3[2], requires_grad=True)
                              self._gamma_inverse = self._gamma_inverse_polynomial_splitted

                  if model_config['gamma_type'] == 143:
                        ################## Polynomial transformation #######################################################

                        init_g4 = [-0.2982, 0.9369, -1.8118, 2.1683]
                        init_d3 = [0.2430, 0.7012, 0.0171]

                        if not model_config['experimental_gamma_forward'] and not model_config['equalization_forward']:
                              self.b1 = torch.nn.Parameter(
                                    torch.tensor(init_g4[0]), requires_grad=True)
                              self.c1 = torch.nn.Parameter(
                                    torch.tensor(init_g4[1]), requires_grad=True)
                              self.d1 = torch.nn.Parameter(
                                    torch.tensor(init_g4[2]), requires_grad=True)
                              self.e1 = torch.nn.Parameter(
                                    torch.tensor(init_g4[3]), requires_grad=True)
                              self._gamma_forward = self._gamma_forward_polynomial
                        if not model_config['experimental_gamma_inverse']:
                              self.c2 = torch.nn.Parameter(
                                    torch.tensor(init_d3[0]), requires_grad=True)
                              self.d2 = torch.nn.Parameter(
                                    torch.tensor(init_d3[1]), requires_grad=True)
                              self.e2 = torch.nn.Parameter(
                                    torch.tensor(init_d3[2]), requires_grad=True)                        
                              self._gamma_inverse = self._gamma_inverse_polynomial

                  if model_config['gamma_type'] == 1430:
                        ################## Polynomial transformation #######################################################

                        init_g4 = [-0.2982, 0.9369, -1.8118, 2.1683]
                        init_d3 = [0.2430, 0.7012, 0.0171]

                        if not model_config['experimental_gamma_forward'] and not model_config['equalization_forward']:
                              self.b1 = torch.tensor(init_g4[0])
                              self.c1 = torch.tensor(init_g4[1])
                              self.d1 = torch.tensor(init_g4[2])
                              self.e1 = torch.tensor(init_g4[3])
                              self._gamma_forward = self._gamma_forward_polynomial
                        if not model_config['experimental_gamma_inverse']:
                              self.c2 = torch.tensor(init_d3[0])
                              self.d2 = torch.tensor(init_d3[1])
                              self.e2 = torch.tensor(init_d3[2])                        
                              self._gamma_inverse = self._gamma_inverse_polynomial

                  if model_config['gamma_type'] == 1450: ## 1470 is the same as this one
                        ################## Polynomial transformation #######################################################

                        init_g4 = [-0.2982, 0.9369, -1.8118, 2.1683]
                        init_d3 = [0.2430, 0.7012, 0.0171]

                        if not model_config['experimental_gamma_forward'] and not model_config['equalization_forward']:
                              self.b1 = torch.tensor(init_g4[0])
                              self.c1 = torch.tensor(init_g4[1])
                              self.d1 = torch.tensor(init_g4[2])
                              self.e1 = torch.tensor(init_g4[3])
                              self._gamma_forward = self._gamma_forward_polynomial_nometa
                        if not model_config['experimental_gamma_inverse']:
                              self.c2 = torch.tensor(init_d3[0])
                              self.d2 = torch.tensor(init_d3[1])
                              self.e2 = torch.tensor(init_d3[2])
                              self._gamma_inverse = self._gamma_inverse_polynomial

            
            if model_config['experimental_gamma_forward'] and not model_config['equalization_forward']:
                  self._gamma_forward = ImprovedGammaJDD(resolution=model_config['experimental_gamma_forward_resolution'],
                                                      num_frames_channels=model_config['experimental_gamma_forward_num_frames_channels'],
                                                      num_meta_channels=model_config['experimental_gamma_forward_num_meta_channels'],
                                                      mode='bicubic' if model_config['experimental_gamma_forward_bicubic'] else 'bilinear',
                                                      skip_frames_channels=model_config['experimental_gamma_forward_skip_frames_channels'],
                                                      skip_meta_channels=model_config['experimental_gamma_forward_skip_meta_channels'],
                                                      apply_to_meta=model_config['experimental_gamma_forward_apply_to_meta'],
                                                      clamp_grid_sample=model_config['experimental_gamma_forward_clamp_grid_sample'],
                                                      monotonous=model_config['experimental_gamma_forward_monotonous'],
                                                      trainable_endpoints=model_config['experimental_gamma_forward_trainable_endpoints'])

            if model_config['experimental_gamma_inverse']:
                  self._gamma_inverse = ImprovedGammaJDD(resolution=model_config['experimental_gamma_inverse_resolution'],
                                                      num_frames_channels=model_config['experimental_gamma_inverse_num_channels'],
                                                      num_meta_channels=None,
                                                      mode='bicubic' if model_config['experimental_gamma_inverse_bicubic'] else 'bilinear',
                                                      skip_frames_channels=model_config['experimental_gamma_inverse_skip_channels'],
                                                      skip_meta_channels=[],
                                                      apply_to_meta=False,
                                                      clamp_grid_sample=model_config['experimental_gamma_inverse_clamp_grid_sample'],
                                                      monotonous=model_config['experimental_gamma_inverse_monotonous'],
                                                      trainable_endpoints=model_config['experimental_gamma_inverse_trainable_endpoints'])



      def switch_off(self):
            log.info('Gamma is switched off!')
            self._gamma_forward_bck = self._gamma_forward
            self._gamma_inverse_bck = self._gamma_inverse
            if isinstance(self._gamma_forward, ImprovedGammaJDD):
                  self._gamma_forward = ImprovedGammaIdentForward()
            else:
                  self._gamma_forward = self._gamma_identity
            if isinstance(self._gamma_inverse, ImprovedGammaJDD):
                  self._gamma_inverse = ImprovedGammaIdentInverse()
            else:
                  self._gamma_inverse = self._gamma_inverse_identity

      def switch_on(self):
            log.info('Gamma is switched on!')
            self._gamma_forward = self._gamma_forward_bck
            self._gamma_inverse = self._gamma_inverse_bck
            self._gamma_forward_bck = None
            self._gamma_inverse_bck = None


      def _calc_out(self, x, b1, c1, d1, e1):
            # TODO: decide what to do with this (some meta channels are larger than 1)
            sign = (1. - x).sign_()
            polynomial_mask = sign.clamp_(0.0, 1.0)
            #######################################

            # TODO: check device usage for the coefficients
            if b1.numel() == 1:
                  x2 = x * x
                  x3 = x2 * x
                  x4 = x3 * x
                  out = b1 * x4 + c1 * x3 + d1 * x2 + e1 * x
                  # out = b1 * x ** 4 + c1 * x ** 3 + d1 * x ** 2 + e1 * x
            else:
                  x = x.transpose(1, 3)
                  x2 = x * x
                  x3 = x2 * x
                  x4 = x3 * x
                  out = b1 * x4 + c1 * x3 + d1 * x2 + e1 * x
                  # out = b1 * x ** 4 + c1 * x ** 3 + d1 * x ** 2 + e1 * x
                  out = out.transpose(1, 3)
                  x = x.transpose(1, 3)

            # TODO: decide what to do with this (some meta channels are larger than 1)
            out = polynomial_mask * out + (1. - polynomial_mask) * x

            return out

      def _calc_out_abs_nomask(self, x, b1, c1, d1, e1):
            # TODO: decide what to do with this (some meta channels are larger than 1)
            # sign = (1. - x).sign_()
            # polynomial_mask = sign.clamp_(0.0, 1.0)
            #######################################

            # TODO: check device usage for the coefficients
            if b1.numel() == 1:
                  x2 = x * x
                  x3 = x2 * x
                  x4 = x3 * x
                  out = (b1 * x4 + c1 * x3 + d1 * x2 + e1 * x).abs()
                  # out = b1 * x ** 4 + c1 * x ** 3 + d1 * x ** 2 + e1 * x
            else:
                  x = x.transpose(1, 3)
                  x2 = x * x
                  x3 = x2 * x
                  x4 = x3 * x
                  out = (b1 * x4 + c1 * x3 + d1 * x2 + e1 * x).abs()
                  # out = b1 * x ** 4 + c1 * x ** 3 + d1 * x ** 2 + e1 * x
                  out = out.transpose(1, 3)
                  # x = x.transpose(1, 3)

            # TODO: decide what to do with this (some meta channels are larger than 1)
            # out = polynomial_mask * out + (1. - polynomial_mask) * x

            return out

      def _ref_frame_selector(self, frames, ref_frame_index):

            return frames[:, 4 * ref_frame_index: 4 * ref_frame_index + 4, ...]


      def print_gamma_coefs(self):
            if self.model_config['gamma_type'] == 843 or self.model_config['gamma_type'] == 943 or self.model_config['gamma_type'] == 743:
                  print(f'b1={self.b1_arr}; c1={self.c1_arr}, d1={self.d1_arr}, e1={self.e1_arr}; c2={self.c2_arr}, d2={self.d2_arr}, e2={self.e2_arr}')
            elif self.model_config['gamma_type'] == 1430 or self.model_config['gamma_type'] == 143:
                  print(f'b1={self.b1}; c1={self.c1}, d1={self.d1}, e1={self.e1}; c2={self.c2}, d2={self.d2}, e2={self.e2}')


      # ##################################### GAMMA-DEGAMMA FUNCTIONS ####################
      def _gamma_identity(self, frames, meta, ref_frame_index):

          ref_frame = self._ref_frame_selector(frames, ref_frame_index)

          return frames, meta, ref_frame

      def _gamma_inverse_identity(self, x):

            return x


      def _gamma_forward_polynomial(self, frames, meta, ref_frame_index):

            frames_gamma = self._calc_out(frames, self.b1, self.c1, self.d1, self.e1)
            meta_gamma = self._calc_out(meta, self.b1, self.c1, self.d1, self.e1)
            ref_frame = self._ref_frame_selector(frames_gamma, ref_frame_index)
            # self.print_gamma_coefs()

            return frames_gamma, meta_gamma, ref_frame


      def _gamma_forward_polynomial_nometa(self, frames, meta, ref_frame_index):

            frames_gamma = self._calc_out(frames, self.b1, self.c1, self.d1, self.e1)
            # meta_gamma = self._calc_out(meta, self.b1, self.c1, self.d1, self.e1)
            ref_frame = self._ref_frame_selector(frames_gamma, ref_frame_index)
            # self.print_gamma_coefs()

            return frames_gamma, meta, ref_frame


      def _gamma_inverse_polynomial(self, x):

            out = self.c2 * x ** 3 + self.d2 * x ** 2 + self.e2 * x

            return out

      def _gamma_inverse_splitted_linear_white(self, x):

            dark_part = self.c2 * x ** 3 + self.d2 * x ** 2 + self.e2 * x

            x0 = 0.5
            y0 = self.c2 * 0.125 + self.d2 * 0.25 + self.e2 * 0.5

            light_part = (y0 - 1) * (x - 1) / (x0 - 1) + 1

            sign = (0.5 - x).sign_()
            dark_mask = sign.clamp_(0.0, 1.0)

            out = dark_mask * dark_part + (1. - dark_mask) * light_part

            return out

      def _gamma_inverse_splitted_polynomial_white(self, x):

            dark_part = self.c2 * x ** 3 + self.d2 * x ** 2 + self.e2 * x

            x0 = 0.5
            y0 = self.c2 * 0.125 + self.d2 * 0.25 + self.e2 * 0.5

            light_part = (y0 - 1) * (x - 1) / (x0 - 1) + 1 + self.white_mult * (x - x0) * (x - 1)

            sign = (0.5 - x).sign_()
            dark_mask = sign.clamp_(0.0, 1.0)

            out = dark_mask * dark_part + (1. - dark_mask) * light_part

            return out


      def _gamma_forward_polynomial_splitted_upd(self, frames, meta, ref_frame_index):

            meta_gamma = self._calc_out(meta, self.b1_arr[0], self.c1_arr[0], self.d1_arr[0], self.e1_arr[0])

            b1_extracted = torch.ones(frames.shape[1], device=frames.device)
            c1_extracted = torch.ones(frames.shape[1], device=frames.device)
            d1_extracted = torch.ones(frames.shape[1], device=frames.device)
            e1_extracted = torch.ones(frames.shape[1], device=frames.device)
            i = 0
            for color in self.frame_colors:
                  ind_color = self.inds_colors[color]

                  b1_extracted[self.inds_frames[color]] = self.b1_arr[ind_color]
                  c1_extracted[self.inds_frames[color]] = self.c1_arr[ind_color]
                  d1_extracted[self.inds_frames[color]] = self.d1_arr[ind_color]
                  e1_extracted[self.inds_frames[color]] = self.e1_arr[ind_color]
                  i += 1

            frames_gamma = self._calc_out(frames, b1_extracted, c1_extracted, d1_extracted, e1_extracted)

            ref_frame = self._ref_frame_selector(frames_gamma, ref_frame_index)
            # self.print_gamma_coefs()

            return frames_gamma, meta_gamma, ref_frame


      def _gamma_forward_polynomial_exposure_splitted(self, frames, meta, ref_frame_index):

            meta_gamma = self._calc_out(meta, self.b1_arr[0], self.c1_arr[0], self.d1_arr[0], self.e1_arr[0])

            b1_extracted = torch.ones(frames.shape[1], device=frames.device)
            c1_extracted = torch.ones(frames.shape[1], device=frames.device)
            d1_extracted = torch.ones(frames.shape[1], device=frames.device)
            e1_extracted = torch.ones(frames.shape[1], device=frames.device)
            i = 0
            for exposure in self.frame_exposures:
                  ind_exp = self.inds_exposure[exposure]
                  inds_frames = slice(4 * i, 4 * i + 4)

                  b1_extracted[inds_frames] = self.b1_arr[ind_exp]
                  c1_extracted[inds_frames] = self.c1_arr[ind_exp]
                  d1_extracted[inds_frames] = self.d1_arr[ind_exp]
                  e1_extracted[inds_frames] = self.e1_arr[ind_exp]
                  i += 1

            frames_gamma = self._calc_out(frames, b1_extracted, c1_extracted, d1_extracted, e1_extracted)

            ref_frame = self._ref_frame_selector(frames_gamma, ref_frame_index)

            # self.print_gamma_coefs()
            return frames_gamma, meta_gamma, ref_frame

      def _gamma_forward_polynomial_exposure_splitted_nometa(self, frames, meta, ref_frame_index):

            # meta_gamma = self._calc_out(meta, self.b1_arr[0], self.c1_arr[0], self.d1_arr[0], self.e1_arr[0])

            b1_extracted = torch.ones(frames.shape[1], device=frames.device)
            c1_extracted = torch.ones(frames.shape[1], device=frames.device)
            d1_extracted = torch.ones(frames.shape[1], device=frames.device)
            e1_extracted = torch.ones(frames.shape[1], device=frames.device)
            i = 0
            for exposure in self.frame_exposures:
                  ind_exp = self.inds_exposure[exposure]
                  inds_frames = slice(4 * i, 4 * i + 4)

                  b1_extracted[inds_frames] = self.b1_arr[ind_exp]
                  c1_extracted[inds_frames] = self.c1_arr[ind_exp]
                  d1_extracted[inds_frames] = self.d1_arr[ind_exp]
                  e1_extracted[inds_frames] = self.e1_arr[ind_exp]
                  i += 1

            frames_gamma = self._calc_out(frames, b1_extracted, c1_extracted, d1_extracted, e1_extracted)

            ref_frame = self._ref_frame_selector(frames_gamma, ref_frame_index)

            # self.print_gamma_coefs()
            return frames_gamma, meta, ref_frame



      def _gamma_forward_polynomial_exposure_splitted_nometa_abs_nomask(self, frames, meta, ref_frame_index):

            # meta_gamma = self._calc_out(meta, self.b1_arr[0], self.c1_arr[0], self.d1_arr[0], self.e1_arr[0])

            b1_extracted = torch.ones(frames.shape[1]).to(frames.device)
            c1_extracted = torch.ones(frames.shape[1]).to(frames.device)
            d1_extracted = torch.ones(frames.shape[1]).to(frames.device)
            e1_extracted = torch.ones(frames.shape[1]).to(frames.device)
            i = 0
            for exposure in self.frame_exposures:
                  ind_exp = self.inds_exposure[exposure]
                  inds_frames = slice(4 * i, 4 * i + 4)

                  b1_extracted[inds_frames] = self.b1_arr[ind_exp]
                  c1_extracted[inds_frames] = self.c1_arr[ind_exp]
                  d1_extracted[inds_frames] = self.d1_arr[ind_exp]
                  e1_extracted[inds_frames] = self.e1_arr[ind_exp]
                  i += 1

            frames_gamma = self._calc_out_abs_nomask(frames, b1_extracted, c1_extracted, d1_extracted, e1_extracted)

            ref_frame = self._ref_frame_selector(frames_gamma, ref_frame_index)

            # self.print_gamma_coefs()
            return frames_gamma, meta, ref_frame


      def _gamma_inverse_polynomial_splitted(self, x):

            x3 = x ** 3
            x2 = x ** 2

            c_x3 = (x3.transpose(1, 3) * self.c2_arr.to(x.device)).transpose(1, 3)
            d_x2 = (x2.transpose(1, 3) * self.d2_arr.to(x.device)).transpose(1, 3)
            e_x = (x.transpose(1, 3) * self.e2_arr.to(x.device)).transpose(1, 3)

            out = c_x3 + d_x2 + e_x

            return out

class ImprovedGammaIdentForward(nn.Module):
      """
      Plug
      """
      def __init__(self):
            super(ImprovedGammaIdentForward, self). __init__()

      def _ref_frame_selector(self, frames, ref_frame_index):
            return frames[:, 4 * ref_frame_index: 4 * ref_frame_index + 4, ...]
      
      def forward(self, frames, meta, ref_frame_index):
          ref_frame = self._ref_frame_selector(frames, ref_frame_index)
          return frames, meta, ref_frame 
class ImprovedGammaIdentInverse(nn.Module):
      """
      Plug
      """
      def __init__(self):
            super(ImprovedGammaIdentInverse, self). __init__()

      
      def forward(self, x):
          return x

class ImprovedGamma(nn.Module):
      """
      Module that performs transform with functional dependency.
      Similar to polynomial gamma idea but more flexible.
      """

      def __init__(self,
                   resolution,
                   num_channels=None,
                   mode='bilinear',
                   skip_channels=[],
                   clamp_grid_sample=False,
                   monotonous=False,
                   trainable_endpoints=False):
            """
            Resolution is the number of learnable nodes uniformly distributed in [0, 1].
            Initializes with linspace, so initially output = input ;)

            Parameters:

            - resolution: int - number of points used for transformation

            - num_channels: int - number of channels of input data. If none, than
            the only one transformation will be applied for all channels

            - mode: 'bilinear', 'bicubic' - mode used in grid sample to interpolate

            - skip_channels: list - list of channels to which gamma is not applied

            - clamp_grid_sample: bool - if True than the output of grid sample is clamped
            to [0, 1] thus making transform from [0, 1] to [0, 1] implicitly

            - monotonous: bool - if True than abs is applied to deltas, making
            transform function monotone, if false than only 0 -> 0 and 1->1 restrictions
            are applied

            - trainable_endpoints: bool - if True than start points are trainable params,
            and cumsum is not normalized (thus there is no more 0->0 and 1->1 restriction).
            False will give old behaviour (0->0 and 1->1).
            EXTREMELY EXPERIMENTAL!!! CAUTION!!! GRAD EXPLOSION !!!

            """
            super(ImprovedGamma, self).__init__()
            self.eps = 1e-6
            self.resolution = resolution
            self.num_channels = num_channels
            self.skip_channels = skip_channels
            self.mode = mode
            self.clamp_grid_sample = clamp_grid_sample
            self.monotonous = monotonous
            self.trainable_endpoints = trainable_endpoints
            if num_channels is not None:  # per channel
                  values = torch.linspace(0, 1, resolution)
                  values = values.expand(num_channels, -1)
                  deltas = self.get_deltas(values)
                  self.deltas = torch.nn.Parameter(deltas, requires_grad=True)
                  if trainable_endpoints:
                        self.start_points = torch.nn.Parameter(torch.zeros(num_channels, 1), requires_grad=True)
            else:
                  values = torch.linspace(0, 1, resolution)
                  deltas = self.get_deltas(values)
                  self.deltas = torch.nn.Parameter(deltas, requires_grad=True)
                  if trainable_endpoints:
                        self.start_points = torch.nn.Parameter(torch.zeros(1), requires_grad=True)

      def forward(self, input):
            """
            input - tensor [N, C, W, H]
            """

            device = input.device
            output = torch.zeros(input.shape, device=device)
            values = self.get_values()
            for channel_idx in range(input.shape[1]):
                  if channel_idx not in (self.skip_channels):
                        x = input[:, channel_idx, :, :]
                        if self.num_channels is not None:  # per channel
                              x_transformed = self.apply_f(x, values[channel_idx, :], self.mode)
                        else:
                              x_transformed = self.apply_f(x, values, self.mode)
                        output[:, channel_idx, :, :] = x_transformed
                  else:
                        x = input[:, channel_idx, :, :]
                        output[:, channel_idx, :, :] = x

            return output

      def get_deltas(self, values):
            if len(values.shape) > 1:  # per channel
                  return (values[:, 1:] - values[:, :-1])
            else:
                  return (values[1:] - values[:-1])

      def get_values(self):
            if self.monotonous:
                  values = torch.abs(self.deltas) + self.eps
            else:
                  values = self.deltas + self.eps
            if self.num_channels is not None:  # per channel
                  if not self.trainable_endpoints:  # need to normalize thus getting 1 at the right end
                        values = values / values.sum(1)[:, None]
                  values = values.cumsum(1)
                  if self.trainable_endpoints:
                        values = torch.cat([self.start_points + self.eps, values], dim=1)
                  else:
                        values = torch.cat(
                              [torch.zeros(values.size(0), device=values.device)[:, None] + self.eps, values], dim=1)
                  values[:, -1] = values[:, -1] + self.eps
            else:
                  if not self.trainable_endpoints:  # need to normalize thus getting 1 at the right end
                        values = values / values.sum(0)
                  values = values.cumsum(0)
                  if self.trainable_endpoints:
                        values = torch.cat([self.start_points + self.eps, values], dim=0)
                  else:
                        values = torch.cat([torch.zeros(1, device=values.device) + self.eps, values], dim=0)
                  values[-1] = values[-1] + self.eps
            return values

      def apply_f(self, x, f, mode):
            f = f.reshape(1, 1, 1, -1).float()
            f = f.expand(x.shape[0], -1, -1, -1).to(x.device)
            grid = x.reshape(x.shape[0], 1, -1, 1) * 2 - 1
            grid = grid.expand(-1, -1, -1, 2)
            output = func.grid_sample(f, grid, mode=mode, align_corners=True).reshape(x.shape).to(x.device)
            if self.clamp_grid_sample:
                  output = torch.clamp(output, 0, 1)
            return output

class ImprovedGammaJDD(nn.Module):
      """
      Module that performs transform with functional dependency.
      Similar to polynomial gamma idea but more flexible.
      """

      def __init__(self,
                   resolution,
                   num_frames_channels=None,
                   num_meta_channels=None,
                   mode='bilinear',
                   skip_frames_channels=[],
                   skip_meta_channels=[],
                   apply_to_meta=True,
                   clamp_grid_sample=False,
                   monotonous=False,
                   trainable_endpoints=False):
            """
            Resolution is the number of learnable nodes uniformly distributed in [0, 1].
            Initializes with linspace, so initially output = input ;)

            Parameters:

            - resolution: int - number of points used for transformation

            - num_frames_channels: int - number of channels of input frames. If none, than
            the only one transformation will be applied for all frames channels

            - num_meta_channels: int - number of channels of input frames. If none, than
            the only one transformation will be applied for all meta channels

            - mode: 'bilinear', 'bicubic' - mode used in grid sample to interpolate

            - skip_frames_channels: list - list of frames channels to which gamma is not applied

            - skip_meta_channels: list - list of meta channels to which gamma is not applied

            - apply_to_meta: bool - apply gamma transformation to meta or no

            - clamp_grid_sample: bool - if True than the output of grid sample is clamped
            to [0, 1] thus making transform from [0, 1] to [0, 1] implicitly

            - monotonous: bool - if True than abs is applied to deltas, making
            transform function monotone, if false than only 0 -> 0 and 1->1 restrictions
            are applied

            - trainable_endpoints: bool - if True than start points are trainable params,
            and cumsum is not normalized (thus there is no more 0->0 and 1->1 restriction).
            False will give old behaviour (0->0 and 1->1).
            EXTREMELY EXPERIMENTAL!!! CAUTION!!! GRAD EXPLOSION !!!

            """
            super(ImprovedGammaJDD, self).__init__()

            self.apply_to_meta = apply_to_meta

            self.gamma_frames = ImprovedGamma(resolution=resolution,
                                              num_channels=num_frames_channels,
                                              mode=mode,
                                              skip_channels=skip_frames_channels,
                                              clamp_grid_sample=clamp_grid_sample,
                                              monotonous=monotonous,
                                              trainable_endpoints=trainable_endpoints
                                              )

            if apply_to_meta:
                  self.gamma_meta = ImprovedGamma(resolution=resolution,
                                                  num_channels=num_meta_channels,
                                                  mode=mode,
                                                  skip_channels=skip_meta_channels,
                                                  clamp_grid_sample=clamp_grid_sample,
                                                  monotonous=monotonous,
                                                  trainable_endpoints=trainable_endpoints
                                                  )

      def forward(self, *inputs):
            """
            input - tensor [N, C, W, H]
            """
            if len(inputs) == 1:  # degamma case: frame out
                  if self.apply_to_meta:
                        log.warning("!!! Apply to meta was selected, but input has only one argument !!!")
                  return self.gamma_frames(inputs[0])
            elif len(inputs) == 3:  # gamma case: frames, meta, ref_frame_index
                  frames, meta, ref_frame_index = inputs
                  frames = self.gamma_frames(frames)
                  if self.apply_to_meta:
                        meta = self.gamma_meta(meta.reshape(meta.shape[0], meta.shape[1], 1, 1)).squeeze()
                  ref_frame = self._ref_frame_selector(frames, ref_frame_index)
                  return frames, meta, ref_frame

      def _ref_frame_selector(self, frames, ref_frame_index):

            return frames[:, 4 * ref_frame_index: 4 * ref_frame_index + 4, ...]

# if __name__ == "__main__":
#       model_config = {
#                       'gamma': True,
#                       'gamma_type': 943,
#                       'experimental_gamma_forward': False,
#                       'experimental_gamma_inverse': False,
#                       'equalization_forward': False
#                       }
#       gamma = Gamma(model_config)
#
#       frames = torch.rand(10, 32, 32, 32)
#       meta = torch.rand(10, 10, 32, 32)
#       # meta = torch.rand(10, 10)
#       ref_frame_index = 0
#       frames_gamma, meta_gamma, ref_frame = gamma._gamma_forward(frames, meta, ref_frame_index)
#       print(frames_gamma.shape)
#       print(meta_gamma.shape)
#       print(ref_frame.shape)
#
#       test_tensor = torch.rand(10, 3, 32, 32)
#       out = gamma._gamma_inverse(test_tensor)
#       print(out.shape)