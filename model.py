class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False, norm="batch", upsample="bilinear", use_tanh=True):
        super().__init__()
        self.outermost = outermost
        use_bias = norm == "instance"
        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv2d(input_nc, inner_nc, 4, 2, 1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        uprelu = nn.ReLU(True)

        if norm == "batch":
            downnorm = nn.BatchNorm2d(inner_nc)
            upnorm = nn.BatchNorm2d(outer_nc)
        elif norm == "instance":
            downnorm = nn.InstanceNorm2d(inner_nc)
            upnorm = nn.InstanceNorm2d(outer_nc)
        else:
            raise NotImplementedError()

        if outermost: 
            if upsample == "convtrans":
                upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, 4, 2, 1)
            else:
                upconv = nn.Sequential(
                    nn.UpsamplingBilinear2d(scale_factor=2),
                    nn.Conv2d(inner_nc * 2, outer_nc, 5, padding=2)
                )
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()] if use_tanh else [uprelu, upconv]
            model = down + [submodule] + up

        elif innermost: 
            if upsample == "convtrans":
                upconv = nn.ConvTranspose2d(inner_nc, outer_nc, 4, 2, 1, bias=use_bias)
            else:
                upconv = nn.Sequential(
                    nn.UpsamplingBilinear2d(scale_factor=2),
                    nn.Conv2d(inner_nc, outer_nc, 5, padding=2, bias=use_bias)
                )
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up

        else:  
            if upsample == "convtrans":
                upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, 4, 2, 1, bias=use_bias)
            else:
                upconv = nn.Sequential(
                    nn.UpsamplingBilinear2d(scale_factor=2),
                    nn.Conv2d(inner_nc * 2, outer_nc, 5, padding=2, bias=use_bias)
                )
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        return torch.cat([x, self.model(x)], 1)


class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm="batch", upsample="bilinear", use_tanh=True):
        super().__init__()
        # innermost block
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, submodule=None, norm=norm, innermost=True, upsample=upsample)
        # middle blocks
        for _ in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, submodule=unet_block, norm=norm, upsample=upsample)
        # gradually decreasing depth
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, submodule=unet_block, norm=norm, upsample=upsample)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, submodule=unet_block, norm=norm, upsample=upsample)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, submodule=unet_block, norm=norm, upsample=upsample)
        # outermost block
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm=norm, upsample=upsample, use_tanh=use_tanh)

    def forward(self, x):
        return self.model(x)
