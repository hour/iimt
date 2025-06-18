# author: Niwhskal
# github : https://github.com/Niwhskal/SRNet

import os
import torch
from torchvision.models import vgg19

import torchvision.transforms.functional as F
import torch

temp_shape = (0,0)

def calc_padding(h, w, k, s):
    
    h_pad = (((h-1)*s) + k - h)//2 
    w_pad = (((w-1)*s) + k - w)//2
    
    return (h_pad, w_pad)

def calc_inv_padding(h, w, k, s):
    h_pad = (k-h + ((h-1)*s))//2
    w_pad = (k-w + ((w-1)*s))//2
    
    return (h_pad, w_pad)


class Conv_bn_block(torch.nn.Module):
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        
        self._conv = torch.nn.Conv2d(*args, **kwargs)
        self._bn = torch.nn.BatchNorm2d(kwargs['out_channels'])
        
    def forward(self, input):
        return torch.nn.functional.leaky_relu(self._bn(self._conv(input)),negative_slope=0.2)

class Res_block(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self._conv1 = torch.nn.Conv2d(in_channels, in_channels//4, kernel_size = 1, stride =1)
        self._conv2 = torch.nn.Conv2d(in_channels//4, in_channels//4, kernel_size = 3, stride = 1, padding = 1)
        self._conv3 = torch.nn.Conv2d(in_channels//4, in_channels, kernel_size = 1, stride=1)
        self._bn = torch.nn.BatchNorm2d(in_channels)
       
    def forward(self, x):
        xin = x
        x = torch.nn.functional.leaky_relu(self._conv1(x),negative_slope=0.2)
        x = torch.nn.functional.leaky_relu(self._conv2(x),negative_slope=0.2)
        x = self._conv3(x)
        x = torch.add(xin, x)
        x = torch.nn.functional.leaky_relu(self._bn(x),negative_slope=0.2)
        return x

class encoder_net(torch.nn.Module):
    def __init__(self, in_channels, get_feature_map = False):
        super().__init__()
        self.cnum = 32
        self.get_feature_map = get_feature_map
        self._conv1_1 = Conv_bn_block(in_channels = in_channels, out_channels = self.cnum, kernel_size = 3, stride = 1, padding = 1)
        self._conv1_2 = Conv_bn_block(in_channels =self.cnum, out_channels = self.cnum, kernel_size = 3, stride = 1, padding = 1)
        #--------------------------
        self._pool1 = torch.nn.Conv2d(in_channels = self.cnum, out_channels = 2 * self.cnum, kernel_size = 3, stride = 2, padding = calc_padding(temp_shape[0], temp_shape[1], 3, 2))
        self._conv2_1 = Conv_bn_block(in_channels = 2* self.cnum, out_channels = 2*self.cnum, kernel_size = 3, stride = 1, padding = 1)
        self._conv2_2 = Conv_bn_block(in_channels = 2* self.cnum, out_channels = 2*self.cnum, kernel_size = 3, stride = 1, padding = 1)
        #---------------------------
        self._pool2 = torch.nn.Conv2d(in_channels = 2*self.cnum, out_channels = 4* self.cnum, kernel_size = 3, stride = 2, padding = calc_padding(temp_shape[0], temp_shape[1], 3, 2))
        self._conv3_1 = Conv_bn_block(in_channels = 4* self.cnum, out_channels = 4*self.cnum, kernel_size = 3, stride = 1, padding = 1)
        self._conv3_2 = Conv_bn_block(in_channels = 4* self.cnum, out_channels = 4*self.cnum, kernel_size = 3, stride = 1, padding = 1)
        #---------------------------
        self._pool3 = torch.nn.Conv2d(in_channels = 4*self.cnum, out_channels = 8* self.cnum, kernel_size = 3, stride = 2, padding = calc_padding(temp_shape[0], temp_shape[1], 3, 2))
        self._conv4_1 = Conv_bn_block(in_channels = 8* self.cnum, out_channels = 8*self.cnum, kernel_size = 3, stride = 1, padding = 1)
        self._conv4_2 = Conv_bn_block(in_channels = 8* self.cnum, out_channels = 8*self.cnum, kernel_size = 3, stride = 1, padding = 1)
        
    def forward(self, x):
        x = self._conv1_1(x)
        x = self._conv1_2(x)
        x = torch.nn.functional.leaky_relu(self._pool1(x),negative_slope=0.2)
        x = self._conv2_1(x)
        x = self._conv2_2(x)
        f1 = x
        x = torch.nn.functional.leaky_relu(self._pool2(x),negative_slope=0.2)
        x = self._conv3_1(x)
        x = self._conv3_2(x)
        f2 = x
        x = torch.nn.functional.leaky_relu(self._pool3(x),negative_slope=0.2)
        x = self._conv4_1(x)
        x = self._conv4_2(x)
        
        if self.get_feature_map:
            return x, [f2, f1]
        else:
            return x
        
        
class build_res_block(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self._block1 = Res_block(in_channels)
        self._block2 = Res_block(in_channels)
        self._block3 = Res_block(in_channels)
        self._block4 = Res_block(in_channels)
        
    def forward(self, x):
        x = self._block1(x)
        x = self._block2(x)
        x = self._block3(x)
        x = self._block4(x)
        return x
    
class decoder_net(torch.nn.Module):
    def __init__(self, in_channels, get_feature_map = False, mt =1, fn_mt=1):
        super().__init__()
        self.cnum = 32
        self.get_feature_map = get_feature_map
        self._conv1_1 = Conv_bn_block(in_channels = fn_mt*in_channels , out_channels = 8*self.cnum, kernel_size = 3, stride =1, padding = 1) 
        self._conv1_2 = Conv_bn_block(in_channels = 8*self.cnum, out_channels = 8*self.cnum, kernel_size = 3, stride =1, padding = 1)
        #-----------------
        self._deconv1 = torch.nn.ConvTranspose2d(8*self.cnum, 4*self.cnum, kernel_size = 3, stride = 2, padding = calc_inv_padding(temp_shape[0], temp_shape[1], 3, 2))
        self._conv2_1 = Conv_bn_block(in_channels = fn_mt*mt*4*self.cnum, out_channels = 4*self.cnum, kernel_size = 3, stride = 1, padding = 1)
        self._conv2_2 = Conv_bn_block(in_channels = 4*self.cnum, out_channels = 4*self.cnum, kernel_size = 3, stride = 1, padding = 1)
        #-----------------
        self._deconv2 = torch.nn.ConvTranspose2d(4*self.cnum, 2*self.cnum, kernel_size =3 , stride = 2, padding = calc_inv_padding(temp_shape[0], temp_shape[1], 3, 2))
        self._conv3_1 = Conv_bn_block(in_channels = fn_mt*mt*2*self.cnum, out_channels = 2*self.cnum, kernel_size = 3, stride = 1, padding = 1)
        self._conv3_2 = Conv_bn_block(in_channels = 2*self.cnum, out_channels = 2*self.cnum, kernel_size = 3, stride = 1, padding = 1)
        #----------------
        self._deconv3 = torch.nn.ConvTranspose2d(2*self.cnum, self.cnum, kernel_size =3 , stride = 2, padding = calc_inv_padding(temp_shape[0], temp_shape[1], 3, 2))
        self._conv4_1 = Conv_bn_block(in_channels = self.cnum, out_channels = self.cnum, kernel_size = 3, stride = 1, padding = 1)
        self._conv4_2 = Conv_bn_block(in_channels = self.cnum, out_channels = self.cnum, kernel_size = 3, stride = 1, padding = 1)
        
        
    def forward(self, x, fuse = None):
        if fuse and fuse[0] is not None:
            x = torch.cat((x, fuse[0]), dim = 1)
        x = self._conv1_1(x)
        x = self._conv1_2(x)
        f1 = x
        #----------
        x = torch.nn.functional.leaky_relu(self._deconv1(x), negative_slope=0.2)

        if fuse and fuse[1] is not None:
            x = torch.cat((x, fuse[1]), dim = 1)

        x = self._conv2_1(x)
        x = self._conv2_2(x)
        f2 = x
        #----------
        x = torch.nn.functional.leaky_relu(self._deconv2(x), negative_slope=0.2)

        if fuse and fuse[2] is not None:
            x = torch.cat((x, fuse[2]), dim = 1)
        
        x = self._conv3_1(x)
        x = self._conv3_2(x)
        f3 = x
        #----------
        x = torch.nn.functional.leaky_relu(self._deconv3(x), negative_slope=0.2)
        x = self._conv4_1(x)
        x = self._conv4_2(x)
        
        if self.get_feature_map:
            return x, [f1, f2, f3]        
        else:
            return x

class text_conversion_net(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.cnum = 32
        self._t_encoder = encoder_net(in_channels)
        self._t_res = build_res_block(8*self.cnum)
        
        self._s_encoder = encoder_net(in_channels)
        self._s_res = build_res_block(8*self.cnum)
        
        self._sk_decoder = decoder_net(16*self.cnum)
        self._sk_out = torch.nn.Conv2d(self.cnum, 1, kernel_size =3, stride = 1, padding = 1)
        
        self._t_decoder = decoder_net(16*self.cnum)
        self._t_cbr = Conv_bn_block(in_channels = 2*self.cnum, out_channels = 2*self.cnum, kernel_size = 3, stride = 1, padding = 1)
        
        self._t_out = torch.nn.Conv2d(2*self.cnum, 3, kernel_size = 3, stride = 1, padding = 1)
        
    def forward(self, x_t, x_s):
        x_t = self._t_encoder(x_t)
        x_t = self._t_res(x_t)
        
        x_s = self._s_encoder(x_s)
        x_s = self._s_res(x_s)
        
        x = torch.cat((x_t, x_s), dim = 1)

        y_sk = self._sk_decoder(x, fuse = None)
        y_sk_out = torch.sigmoid(self._sk_out(y_sk))        
        
        y_t = self._t_decoder(x, fuse = None)
        
        y_t = torch.cat((y_sk, y_t), dim = 1)
        y_t = self._t_cbr(y_t)
        y_t_out = torch.tanh(self._t_out(y_t))
        
        return y_sk_out, y_t_out
                                          
        
class inpainting_net(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.cnum = 32
        self._encoder = encoder_net(in_channels, get_feature_map = True)
        self._res = build_res_block(8*self.cnum)
        
        self._decoder = decoder_net(8*self.cnum,  get_feature_map = True, mt=2)
        self._out = torch.nn.Conv2d(self.cnum, 3, kernel_size = 3, stride = 1, padding = 1)
        
    def forward(self, x):
        x, f_encoder = self._encoder(x)
        x = self._res(x)
        x, fs = self._decoder(x, fuse = [None] + f_encoder)
        x = torch.tanh(self._out(x))
        return x, fs

class fusion_net(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.cnum = 32
        self._encoder = encoder_net(in_channels)
        self._res = build_res_block(8*self.cnum)
        self._decoder = decoder_net(8*self.cnum, fn_mt = 2)
        self._out = torch.nn.Conv2d(self.cnum, 3, kernel_size = 3, stride = 1, padding = 1)
         
    def forward(self, x, fuse):
        x = self._encoder(x)
        x = self._res(x)
        x = self._decoder(x, fuse = fuse)
        x = torch.tanh(self._out(x))
        return x
           
class Generator(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.cnum = 32
        self._tcn = text_conversion_net(in_channels)
        self._bin = inpainting_net(in_channels)
        self._fn = fusion_net(in_channels)
        
    def forward(self, i_t, i_s, gbl_shape):
        temp_shape = gbl_shape
        o_sk, o_t = self._tcn(i_t, i_s)
        o_b, fuse = self._bin(i_s)
        o_f = self._fn(o_t, fuse)
        return o_sk, o_t, o_b, o_f

class Discriminator(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.cnum = 32
        self._conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size = 3, stride = 2, padding = calc_padding(temp_shape[0], temp_shape[1], 3, 2))
        self._conv2 = torch.nn.Conv2d(64, 128, kernel_size = 3, stride = 2, padding = calc_padding(temp_shape[0], temp_shape[1], 3, 2))
        self._conv2_bn = torch.nn.BatchNorm2d(128)
        self._conv3 = torch.nn.Conv2d(128, 256, kernel_size = 3, stride = 2, padding = calc_padding(temp_shape[0], temp_shape[1], 3, 2))
        self._conv3_bn = torch.nn.BatchNorm2d(256)
        self._conv4 = torch.nn.Conv2d(256, 512, kernel_size = 3, stride = 2, padding = calc_padding(temp_shape[0], temp_shape[1], 3, 2))
        self._conv4_bn = torch.nn.BatchNorm2d(512)
        self._conv5 = torch.nn.Conv2d(512, 1,  kernel_size = 3, stride = 1, padding = 1)
        self._conv5_bn = torch.nn.BatchNorm2d(1)
     
    def forward(self, x):
        x = torch.nn.functional.leaky_relu(self._conv1(x), negative_slope=0.2)
        x = self._conv2(x)
        x = torch.nn.functional.leaky_relu(self._conv2_bn(x), negative_slope=0.2)
        x = self._conv3(x)
        x = torch.nn.functional.leaky_relu(self._conv3_bn(x), negative_slope=0.2)
        x = self._conv4(x)
        x = torch.nn.functional.leaky_relu(self._conv4_bn(x), negative_slope=0.2)
        x = self._conv5(x)
        x = self._conv5_bn(x)
        return x

class Vgg19(torch.nn.Module):
    def __init__(self):
        super(Vgg19, self).__init__()
        features = list(vgg19(pretrained = True).features)
        self.features = torch.nn.ModuleList(features).eval()

    def forward(self, x):
        results = []
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in {1, 6, 11, 20, 29}:
                results.append(x)
        return results
        
class SRNet(torch.nn.Module):
    def __init__(self):
        super(SRNet, self).__init__()
        self.G = Generator(in_channels = 3)
        self.D1 = Discriminator(in_channels = 6)
        self.D2 = Discriminator(in_channels = 6)
        self.vgg_features = Vgg19()
        self.K = torch.nn.ZeroPad2d((0, 1, 1, 0))

    # def from_pretrained(self, ckpt_path):
    #     checkpoint = torch.load(ckpt_path)
    #     self.G.load_state_dict(checkpoint['generator'])
    #     self.D1.load_state_dict(checkpoint['discriminator1'])
    #     self.D2.load_state_dict(checkpoint['discriminator2'])

    # def save(self, ckpt_path):
    #     torch.save({
    #         'generator': self.G.state_dict(),
    #         'discriminator1': self.D1.state_dict(),
    #         'discriminator2': self.D2.state_dict()
    #     }, ckpt_path)

    def forward(self, i_t, i_s, t_b=None, t_f=None):
        o_sk, o_t, o_b, o_f = self.G(i_t, i_s, (i_t.shape[2], i_t.shape[3])) #Adding dim info
        o_sk, o_t, o_b, o_f = self.K(o_sk), self.K(o_t), self.K(o_b), self.K(o_f)

        o_db_pred = self.D1(torch.cat((o_b, i_s), dim = 1))
        o_df_pred = self.D2(torch.cat((o_f, i_t), dim = 1))

        o_db_true, o_df_true, out_vgg = None, None, None
        if t_b is not None and t_f is not None:
            o_db_true = self.D1(torch.cat((t_b, i_s), dim = 1))
            o_df_true = self.D2(torch.cat((t_f, i_t), dim = 1))
            out_vgg = self.vgg_features(torch.cat((t_f, o_f), dim = 0))

        return (o_sk, o_t, o_b, o_f), o_db_pred, o_df_pred, o_db_true, o_df_true, out_vgg

    def predit(self):
        pass

    def save_example(self, example, savedir):
        with torch.no_grad():            
            i_t = example[0].cuda()
            i_s = example[1].cuda()
            name = str(example[2][0])
            
            o_sk, o_t, o_b, o_f = self.G(i_t, i_s, (i_t.shape[2], i_t.shape[3]))

            o_sk = o_sk.squeeze(0).to('cpu')
            o_t = o_t.squeeze(0).to('cpu')
            o_b = o_b.squeeze(0).to('cpu')
            o_f = o_f.squeeze(0).to('cpu')
            
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            
            o_sk = F.to_pil_image(o_sk)
            o_t = F.to_pil_image((o_t + 1)/2)
            o_b = F.to_pil_image((o_b + 1)/2)
            o_f = F.to_pil_image((o_f + 1)/2)
                            
            o_f.save(os.path.join(savedir, name + 'o_f.png'))
            o_sk.save(os.path.join(savedir, name + 'o_sk.png'))
            o_t.save(os.path.join(savedir, name + 'o_t.png'))
            o_b.save(os.path.join(savedir, name + 'o_b.png'))

# class SRNetOptimizer(torch.optim.Optimizer):
#     def __init__(self, model, learning_rate=1e-4, beta1=0.9, beta2=0.999):
#         self.G_solver = torch.optim.Adam(model.G.parameters(), lr=learning_rate, betas = (beta1, beta2))
#         self.D1_solver = torch.optim.Adam(model.D1.parameters(), lr=learning_rate, betas = (beta1, beta2))
#         self.D2_solver = torch.optim.Adam(model.D2.parameters(), lr=learning_rate, betas = (beta1, beta2))

#     def from_pretrained(self, ckpt_path):
#         checkpoint = torch.load(ckpt_path)
#         self.G_solver.load_state_dict(checkpoint['g_optimizer'])
#         self.D1_solver.load_state_dict(checkpoint['d1_optimizer'])
#         self.D2_solver.load_state_dict(checkpoint['d2_optimizer'])

#     def save(self, ckpt_path):
#         torch.save({
#             'g_optimizer': self.G_solver.state_dict(),
#             'd1_optimizer': self.D1_solver.state_dict(),
#             'd2_optimizer': self.D2_solver.state_dict(),
#         }, ckpt_path)

