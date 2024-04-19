model = dict(
    image_size = (1440, 2560),
    feat_size = [(180, 320), (90, 160), (45, 80), (23, 40), (12, 20), (6, 10)],
    steps = [8, 16, 32, 64, 128, 256],
    scales = [21, 45, 99, 153, 207, 261, 315],
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
    folders = ['10001_Urban_Clear_Noon', '10002_Urban_Clear_Morning', '10003_Urban_Clear_Noon', '10004_Urban_Clear_Noon', '10005_Urban_Clear_Noon',
               '10006_Urban_Clear_Noon', '10007_Highway_Clear_Morning', '10008_Highway_Clear_Noon', '10009_Urban_Rain_Morning', '10010_Urban_Rain_Morning', '10011_Urban_Rain_Evening',
               '10012_Urban_Rain_Evening', '10013_Highway_Rain_Morning', '10014_Highway_Rain_Noon', '10015_Highway_Clear_Evening', '10016_Highway_Clear_Evening',
               '10017_Urban_Rain_Evening', '10018_Urban_Rain_Evening', '10019_Urban_HeavyRain_Noon', '10020_Urban_HeavyRain_Afternoon', '10021_Urban_Clear_Morning', '10022_Urban_Clear_Morning',
               '10023_Urban_Rain_Morning', '10024_Urban_Rain_Evening', '10025_Urban_HeavyRain_Evening', '10026_Urban_HeavyRain_Afternoon'],
    cameras = ['Cognata_Camera_01_8M', 'Cognata_Camera_02_8M', 'Cognata_Camera_03_8M']
)
