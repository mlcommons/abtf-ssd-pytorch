model = dict(
    image_size = (2160, 3840),
    feat_size = [(270, 480), (135, 240), (68, 120), (34, 60), (17, 30), (9, 15)],
    steps = [8, 16, 32, 64, 128, 256],
    scales = [31, 67, 148, 229, 310, 392, 473],
    aspect_ratios = [[2, 0.5], [2, 0.5, 3, 1/3], [2, 0.5, 3, 1/3], [2, 0.5, 3, 1/3], [2, 0.5], [2, 0.5]],
    pre_backbone = None,
    backbone = [{'block_no': 6, 'layer': 0, 'conv': [{1: {'kernel_size': (1, 1), 'padding': (0,0), 'stride': (1, 1)},
                                                   2: {'kernel_size': (3, 3), 'padding': (1,1), 'stride': (1, 1)}}],
                        'downsample': {'kernel_size': (1, 1), 'padding': 0, 'stride': (1, 1)}}],
    middle_blocks = [{'kernel_size': (1,3), 'padding': (0,1), 'stride': (1,2)},
                    {'kernel_size': (1,3), 'padding': (0,1), 'stride': (1,2)},
                    {'kernel_size': (1,3), 'padding': (0,1), 'stride': (1,2)},
                    {'kernel_size': (1,3), 'padding': (0,1), 'stride': (1,2)},
                    {'kernel_size': (1,3), 'padding': (0,1), 'stride': (1,2)}],
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
    folders = ['10001_Urban_Clear_Noon', '10002_Urban_Clear_Morning', '10003_Urban_Clear_Noon', '10004_Urban_Clear_Noon', '10005_Urban_Clear_Noon'],
    cameras = ['Cognata_Camera_01_8M', 'Cognata_Camera_02_8M', 'Cognata_Camera_03_8M']
)
