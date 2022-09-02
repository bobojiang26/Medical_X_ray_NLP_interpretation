import networks
import torch
import torch.nn as nn
import torch.nn.functional as F
import time


class FeatureExtractor(nn.Module):
  def __init__(self, opts):
    super(FeatureExtractor, self).__init__()

    # parameters
    lr = opts.lr
    self.codebook_size = opts.codebook_size
    self.embed_dim = opts.embed_dim
    self.beta = 0.25
    self.concat = opts.concat

    # codebook
    self.quantize = networks.VectorQuantizer(self.codebook_size, self.embed_dim, self.beta)
        
    # encoders
    self.enc = networks.E_content_codebook(opts.input_dim_a, opts.input_dim_b)

    if self.concat:
      self.gen = networks.G_concat(opts.input_dim_a, opts.input_dim_b, nz=self.nz)
    else:
      self.gen = networks.G(opts.input_dim_a, opts.input_dim_b, nz=self.nz)

    # optimizers
    self.quantize_opt = torch.optim.Adam(self.quantize.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)

    self.enc_opt = torch.optim.Adam(self.enc.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    
    self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)


  # load weights from scratch
  def initialize(self):
    self.quantize.apply(networks.gaussian_weights_init)

    self.enc.apply(networks.gaussian_weights_init)

    self.gen.apply(networks.gaussian_weights_init)


  # load weights when training from resume
  def set_scheduler(self, opts, last_ep=0):
    self.quantize_sch = networks.get_scheduler(self.quantize_opt, opts, last_ep)

    self.enc_sch = networks.get_scheduler(self.enc_opt, opts, last_ep)

    self.gen_sch = networks.get_scheduler(self.gen_opt, opts, last_ep)


  def setgpu(self, gpu):
    self.gpu = gpu
    self.quantize.cuda(self.gpu)
    self.enc.cuda(self.gpu)
    self.gen.cuda(self.gpu)


  def forward(self):
    # get z_content
    self.z_content = self.enc.forward(self.input)
    
    self.z_codebook_output, _, _ = self.quantize(self.z_content)

    # view vectors
    self.fe_output = self.z_codebook_output.view(-1,1)

    return self.fe_output


