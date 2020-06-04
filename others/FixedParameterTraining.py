net = Network()

#查看模型的名称和其grad值
for name, value in net.named_parameters():
    print('name: {},\tgrad: {}'.format(name, value.requires_grad))

no_grad = [
    'conv1.weight',
    'conv1.bias',
    'conv2.weight',
    'conv2.bias'
]

net = Net.Network()
