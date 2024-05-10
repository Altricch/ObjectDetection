import torch
import torch.nn as nn

# SOURCE USED
# https://www.youtube.com/watch?v=n9_XyCGr-MI
# https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/object_detection/YOLO/model.py

# Each tuple is structure by 
# (kernel size, nr of filters as output, stride, padding)
# Padding is calculated by hand
# "M" = maxpool
architecture_config = [
    # Tuple: (kernel_size, num_filters, stride, padding)
    (7,64,2,3),
    "M",
    (3,192,1,1),
    "M",
    (1,128,1,0),
    (3,256,1,1),
    (1,256,1,0),
    (3,512,1,1),
    "M",
    # List: tuple((kernel_size, num_filter, stride, padding), (kernel_size, num_filter, stride, padding), num repeats)
    [(1,256,1,0), (3,512,1,1), 4],
    (1,512,1,0),
    (3,1024,1,1),
    "M",
    # List: tuple((kernel_size, num_filter, stride, padding), (kernel_size, num_filter, stride, padding), num repeats)
    [(1,512,1,0), (3,1024,1,1), 2],
    (3,1024,1,1),
    (3,1024,2,1),
    (3,1024,1,1),
    (3,1024,1,1),
]


# Reuse multiple times
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        # To make distribution somewhat more stable, not used in paper 
        # but nowadays state of the art
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))

# defines the entire Yolo architecture    
class Yolov1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(Yolov1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        # Framework by Joseph Redman, is just the conv_layers 
        # Built from the architecure
        self.darknet = self._create_conv_layers(self.architecture)
        # Fully connected layer at the end for classification
        self.fcs = self._create_fcs(**kwargs)
        
    def forward(self, x):
        x = self.darknet(x)
        # Flatten vector to prepare for linear layer
        flattened = torch.flatten(x, start_dim=1)
        return self.fcs(flattened)
    
    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels        
        
        # Iterate through the architecture defined beforehand
        for x in architecture:
            # check for tuple or not
            if type(x) == tuple:
                layers += [
                    CNNBlock(
                        # How tuple is structured of architecture
                        in_channels,
                        out_channels=x[1], 
                        kernel_size = x[0], 
                        stride=x[2], 
                        padding=x[3] 
                        )
                    ]
                
                # adjust in channels to x[1]= outchannels
                in_channels = x[1]
            
            # if we have "M":
            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                
            # if list, we have number of repreats
            elif type(x) == list:
                conv1 = x[0] #Tuple for first conv block
                conv2 = x[1] #Tuple for second conv block
                num_repeats = x[2] # Integer
                
                for _ in range(num_repeats):
                    # First tuple of the convolution iteration
                    layers += [
                        CNNBlock(
                            in_channels, 
                            conv1[1], 
                            kernel_size = conv1[0],
                            stride=conv1[2],
                            padding = conv1[3]
                            )
                        ]
                    
                    # Second tuple of the convolution iteration
                    layers += [
                        CNNBlock( 
                            #  Input now output of conv1 layer outchannels
                            conv1[1], 
                            conv2[1],
                            kernel_size = conv2[0],
                            stride=conv2[2],
                            padding = conv2[3])]
                    
                    # For next loop iteration update
                    in_channels = conv2[1]
                    
        return nn.Sequential(*layers)
    
    # Create Fully Connected layers
    def _create_fcs(self, split_size, num_boxes, num_classes):
        S,B,C = split_size, num_boxes, num_classes
        # print("S<C<B",S,B,C)
        # print("NN FLATTER", nn.Flatten())
        return nn.Sequential(
            nn.Flatten(),
            # TODO: Original paper 4096 instead of 496
            nn.Linear(1024 * S * S, 496),
            # TODO: Should be 0.5
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            # S x S x number of classes + bounding boxes * 5 values (probab. & 4 bbox values)
            nn.Linear(496, S*S*(C+B*5)), # will be reshaped into (S, S, 30) where C + B * 5 = 30
        )
        
def test(S = 7, B = 2, C = 20):
    model = Yolov1(split_size = S, num_boxes = B, num_classes= C)
    x = torch.randn((2, 3, 448, 448))
    print(model(x).shape)
    
if __name__ == "__main__":
    # 7x7x30 = 1470 which we can reshape, Thus, we have [2, 1470]
    test()
                
                
        
        









