import argparse

parser = argparse.ArgumentParser()
parser.add_argument('layers', help='the numbers of layers in the model', type=int)
parser.add_argument('kernel_size', help='the size of the kernel for the convultions', type=int)
args = parser.parse_args()

def compute_recp_field():
    sum = 0
    for i in range(args.layers):
        sum += 2*(args.kernel_size - 1) * (2**i)
        
    print(sum + 1)
    
compute_recp_field()