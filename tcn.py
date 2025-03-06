import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm

def random_(weights, *args, **kwargs):
    return weights

class TCN(nn.Module):
    def __init__(self,
                 num_filters:int,
                 layers:dict,
                 dropout_prob:float,
                 channels_per_hidden_layer:int,
                 res_mapping=False,
                 weight_init_name='xavier_normal',
                 ):
        """
            num_filters(int): the size of the kernel in each residual block
            layers(dict): dictionary containing input and output channels in a list for each layer (ex. {"res1": (10, 40)}), each key is a layer type:
                res1 = residual layer -> (in, out)
                fcl1 = fully connected layer -> (in, out)
            dropout_prob(float): the probability of activating a dropout layer
            channels_per_hidden_layer: the number of channels "in-between" convs in res blocks
            res_mapping(bool): whether or not the input of each residual layer should be added to its output
            weight_init_name(str): the str corresponding to the weight initialization function (run TCN.list_init_funcs to see the options)

            example initilization:
                layers = {'res1': (14, 16),
                          'res2': (16, 32),
                          'res3': (32, 64),
                          'res4': (64, 32),
                          'res5': (32, 64),
                          'fcl1': (64, 1)
                          }

                model = TCN(num_filters=4,
                            layers=layers,
                            dropout_prob=0.3,
                            channels_per_hidden_layer=50,
                            res_mapping=True,
                            weight_init_name='random',
                            ).to(device=device)

        """

        super().__init__()
        self.dropout_prob = dropout_prob
        self.weight_init_name = weight_init_name
        self.weight_init_funcs = {'xavier_uniform': nn.init.xavier_uniform_,
                                  'xavier_normal': nn.init.xavier_normal_,
                                  'kaiming_uniform': nn.init.kaiming_uniform_,
                                  'kaiming_normal': nn.init.kaiming_normal_,
                                  'normal': nn.init.normal_,
                                  'uniform': nn.init.uniform_,
                                  'random': random_
                                  }
        self.available_layers = ['res', 'fcl']

        layer_list = []
        for i, key in enumerate(layers):
            for avail_key in self.available_layers:
                found = 0
                if avail_key in key:
                    found += 1
                    break

            if found == 0:
                raise RuntimeError(f"The key: '{key}' is not an available layer, please choose from: {self.available_layers}")

            in_out = layers[key]
            in_channels = in_out[0]
            out_channels = in_out[1]

            if 'res' in key:
                dilation = 2 ** i

                block = Residual_Block(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel=num_filters,
                                       dilation=dilation,
                                       dropout_prob=dropout_prob,
                                       channels_per_hidden_layer=channels_per_hidden_layer,
                                       res_mapping=res_mapping,
                                       weight_init_func=self.weight_init_funcs[weight_init_name]
                                       )
                layer_list.append(block)

            elif 'fcl' in key:
                self.linear = nn.Linear(in_channels, out_channels)
                self.weight_init_funcs[weight_init_name](self.linear.weight)

        self.layer_list = nn.ModuleList(layer_list)

    def forward(self, x):
        for layer in self.layer_list:
            x = layer(x)

        #combine the 0 and 2 dimensions and leave the channels untouched then transpose
        out = torch.cat([x[i, :, :] for i in range(x.shape[0])], dim = 1).transpose(0, 1).contiguous()
        out = self.linear(out).transpose(0, 1)
		#reshape back to original format
        out = torch.cat([out[:, i*x.shape[2]:(i+1)*x.shape[2]].unsqueeze(0) for i in range(x.shape[0])], dim = 0)
        return out

    def list_init_funcs(self):
        keys = list(self.weight_init_funcs.keys())
        for key in keys:
            print(key)

class Cut_End(nn.Module):
    def __init__(self, cut_size):
        super().__init__()
        self.cut_size = cut_size
    
    def forward(self, x):
        return x[:, :, :-self.cut_size].contiguous()


class Residual_Block(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel,
                 dilation,
                 dropout_prob,
                 channels_per_hidden_layer,
                 res_mapping,
                 weight_init_func
                 ):
        super().__init__()

        self.res_mapping = res_mapping
        self.init_func = weight_init_func
        padding = (kernel - 1) * dilation

        #self.pad = nn.ConstantPad1d((padding, 0), 0.0) #if we move where the input is padded and how, then the output can remain constant

        self.conv1 = weight_norm(nn.Conv1d(
            in_channels=in_channels,
            out_channels=channels_per_hidden_layer,
            kernel_size=kernel,
            padding=padding,
            dilation=dilation
            ), name='weight')
        self.chomp1 = Cut_End(padding)
        self.activation1 = nn.ReLU()
        #self.dropout1 = nn.Dropout1d(p=dropout_prob)
        self.dropout1 = nn.Dropout(dropout_prob)
        
        self.conv2 = weight_norm(nn.Conv1d(
            in_channels=channels_per_hidden_layer,
            out_channels=out_channels,
            kernel_size=kernel,
            padding=padding,
            dilation=dilation
            ), name='weight')
        self.chomp2 = Cut_End(padding)
        self.activation2 = nn.ReLU()
        #self.dropout2 = nn.Dropout1d(p=dropout_prob)
        self.dropout2 = nn.Dropout(dropout_prob)
        
        self.identity_map = nn.Conv1d(in_channels, out_channels, 1, padding=0) if in_channels != out_channels else None

        self.activation_final = nn.ReLU()

        self.init_weights()

    def forward(self, x):
        #out = self.pad(x)
        #out = self.conv1(out)
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.activation1(out)
        out = self.dropout1(out)
        #out = self.pad(out)
        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.activation2(out)
        out = self.dropout2(out)

        if self.res_mapping:
            res = x if self.identity_map is None else self.identity_map(x)
            return self.activation_final(out + res)

        else:
            return self.activation_final(out)

    def init_weights(self):
        #self.init_func(self.conv1.weight)
        #self.init_func(self.conv2.weight)
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)

        if self.identity_map is not None and self.res_mapping:
            #self.init_func(self.identity_map.weight)
            self.identity_map.weight.data.normal_(0, 0.01)
