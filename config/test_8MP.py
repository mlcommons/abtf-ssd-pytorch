model = dict(
    image_size = (2160, 3840),
    feat_size = [(270, 480), (135, 240), (68, 120), (34, 60), (32, 58), (30, 56)],
    steps = [8, 16, 32, 64, 100, 300],
    scales = [21, 45, 99, 153, 207, 261, 315],
    aspect_ratios = [[2, 0.5], [2, 0.5, 3, 1/3], [2, 0.5, 3, 1/3], [2, 0.5, 3, 1/3], [2, 0.5], [2, 0.5]]
)

dataset = dict(
    train_folders = ['10001_Urban_Clear_Noon'],
    val_folders = ['10001_Urban_Clear_Noon'],
    cameras = ['Cognata_Camera_01_8M']
)
