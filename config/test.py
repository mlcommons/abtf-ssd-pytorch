model = dict(
    image_size = (480, 640),
    feat_size = [(60, 80), (30, 40), (15, 20), (8, 10), (6, 8), (4, 6)],
    steps = [8, 16, 32, 64, 100, 300],
    scales = [21, 45, 99, 153, 207, 261, 315],
    aspect_ratios = [[2, 0.5], [2, 0.5, 3, 1/3], [2, 0.5, 3, 1/3], [2, 0.5, 3, 1/3], [2, 0.5], [2, 0.5]],
    pre_backbone = None,
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
    folders = ['10001_Urban_Clear_Noon', '10003_Urban_Clear_Noon'],
    cameras = ['Cognata_Camera_01_8M', 'Cognata_Camera_02_8M']
)