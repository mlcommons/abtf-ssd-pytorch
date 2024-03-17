model = dict(
    image_size = (1000, 1000),
    feat_size = [(125, 125), (63, 63), (32, 32), (16, 16), (14, 14), (12, 12)],
    steps = [8, 16, 32, 64, 100, 300],
    scales = [21, 45, 99, 153, 207, 261, 315],
    aspect_ratios = [[2, 0.5], [2, 0.5, 3, 1/3], [2, 0.5, 3, 1/3], [2, 0.5, 3, 1/3], [2, 0.5], [2, 0.5]],
    pre_backbone = None,
    backbone = [{'block_no': 6, 'layer': 0, 'conv': [{1: {'kernel_size': (1, 1), 'padding': (0,0), 'stride': (1, 1)},
                                                      2: {'kernel_size': (3, 3), 'padding': (1,1), 'stride': (1, 1)}}],
                                'downsample': {'kernel_size': (1, 1), 'padding': 0, 'stride': (1, 1)}}],
    middle_blocks = [{'kernel_size': (1,3), 'padding': (0,1), 'stride': (1,2)}, 
                    {'kernel_size': (1,3), 'padding': (0,1), 'stride': (1,2)},
                    {'kernel_size': (1,3), 'padding': (0,1), 'stride': (1,2)},
                    {'kernel_size': (1,3), 'padding': (0,0), 'stride': (1,1)},
                    {'kernel_size': (1,3), 'padding': (0,0), 'stride': (1,1)}],
    head = {'num_defaults': [4, 6, 6, 6, 4, 4],
            'loc': [{'kernel_size': 3, 'padding': 1, 'stride': 1},
                    {'kernel_size': 3, 'padding': 1, 'stride': 1},
                    {'kernel_size': 3, 'padding': 1, 'stride': 1},
                    {'kernel_size': 3, 'padding': 1, 'stride': 1},
                    {'kernel_size': 3, 'padding': 1, 'stride': 1},
                    {'kernel_size': 3, 'padding': 1, 'stride': 1}],
            'conf': [{'kernel_size': 3, 'padding': 1, 'stride': 1},
                    {'kernel_size': 3, 'padding': 1, 'stride': 1},
                    {'kernel_size': 3, 'padding': 1, 'stride': 1},
                    {'kernel_size': 3, 'padding': 1, 'stride': 1},
                    {'kernel_size': 3, 'padding': 1, 'stride': 1},
                    {'kernel_size': 3, 'padding': 1, 'stride': 1}]
            }
)

dataset = dict(
    folders = ['10001_Urban_Clear_Noon'],
    cameras = ['Cognata_Camera_01_8M']
)
