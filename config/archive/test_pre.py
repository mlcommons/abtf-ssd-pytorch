model = dict(
    image_size = (2160, 3840),
    feat_size = [(34, 60), (17, 30), (9, 15), (5, 8), (3, 6), (1, 4)],
    steps = [8, 16, 32, 64, 100, 300],
    scales = [21, 45, 99, 153, 207, 261, 315],
    aspect_ratios = [[2, 0.5], [2, 0.5, 3, 1/3], [2, 0.5, 3, 1/3], [2, 0.5, 3, 1/3], [2, 0.5], [2, 0.5]],
    pre_backbone = [{'kernel_size': 3, 'padding': 1, 'stride': 2},
                    {'kernel_size': 3, 'padding': 1, 'stride': 2},
                    {'kernel_size': 3, 'padding': 1, 'stride': 2}],
    #pre_backbone = None,
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
    train_folders = ['10001_Urban_Clear_Noon'],
    val_folders = ['10001_Urban_Clear_Noon'],
    cameras = ['Cognata_Camera_01_8M']
)