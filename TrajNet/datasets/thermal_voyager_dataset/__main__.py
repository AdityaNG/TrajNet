from . import *

if __name__ == "__main__":
    DEPTH_netS = [
        "DPTDepthModel",
        "DPT_BEiT_B_384",
        "DPT_BEiT_L_384",
        "DPT_BEiT_L_512",
        "DPT_Hybrid",
        "DPT_Large",
        "DPT_LeViT_224",
        "DPT_Next_ViT_L_384",
        "DPT_SwinV2_B_384",
        "DPT_SwinV2_L_384",
        "DPT_SwinV2_T_256",
        "DPT_Swin_L_384",
        "MiDaS",
        "MiDaS_small",
        "MidasNet",
        "MidasNet_small",
        "transforms",
    ]
    depth_net_type = "DPT_LeViT_224"
    # DEPTH_netS = ('MiDaS_small', 'DPT_Hybrid', 'DPT_Large')
    # depth_net_type = DEPTH_netS[0]
    dataset_base = "/home/shared/car_dataset/car_dataset/"

    # dir(midas_transforms): ['NormalizeImage', 'PrepareForNet', 'Resize', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__spec__', 'apply_min_size', 'beit512_transform', 'cv2', 'default_transform', 'dpt_transform', 'levit_transform', 'math', 'np', 'small_transform', 'swin256_transform', 'swin384_transform']
    # dir(midas_transforms): ['beit512_transform', 'default_transform', 'dpt_transform', 'levit_transform', 'small_transform', 'swin256_transform', 'swin384_transform']
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if depth_net_type == "DPT_Large" or depth_net_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    elif "LeViT" in depth_net_type:
        transform = midas_transforms.levit_transform
    elif "DPT_SwinV2_T_256" == depth_net_type:
        transform = midas_transforms.swin256_transform
    else:
        transform = midas_transforms.small_transform


    dataset = ThermalVoyager_ThermalDepthDataset(
        dataset_path="/home/shared/Thermal_Voyager/Processed/2023-04-18/1/",
        debug=False,
        transform=transform,
    )

    print(len(dataset))
    print(dataset[0]["timestamp"])
    print(dataset[len(dataset) // 4 - 1]["timestamp"])
    # Set video properties
    # fps = dataset.dataset_freq  # frames per second
    fps = 1  # frames per second
    # frame_size = (1234, 1280)  # frame size (width, height)
    # frame_size = (320, 720)  # frame size (width, height)
    frame_size = (720, 1280)  # frame size (width, height)
    frame_size = (1280, 720)  # frame size (width, height)

    # frame_size = (720, 320)  # frame size (width, height)

    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        "test_media/output_hist.mp4", fourcc, fps, frame_size
    )

    for frame_index in tqdm(range(len(dataset) // 4)):
        frame_data = dataset[frame_index]
        frame = frame_data["dataset_frame"]
        car_state_frame = frame_data["car_state_frame"]

        # continue
        zed2i_stereo_np = frame["zed2i_stereo_np"]
        zed2i_left_np = frame["zed2i_left_np"]
        zed2i_right_np = frame["zed2i_right_np"]

        # thermal_lwir_np = frame['thermal_lwir_np']
        # thermal_mwir_np = frame['thermal_mwir_np']

        thermal_lwir_np = frame["thermal_lwir_color_np"]
        thermal_mwir_np = frame["thermal_mwir_color_np"]

        zed_depth_lwir = (
            normalize(frame["zed_depth_lwir"]) * np.iinfo(np.uint8).max
        ).astype(np.uint8)
        zed_depth_mwir = (
            normalize(frame["zed_depth_mwir"]) * np.iinfo(np.uint8).max
        ).astype(np.uint8)
        lidar_depth_lwir = (
            normalize(frame["lidar_depth_lwir"]) * np.iinfo(np.uint8).max
        ).astype(np.uint8)
        lidar_depth_mwir = (
            normalize(frame["lidar_depth_mwir"]) * np.iinfo(np.uint8).max
        ).astype(np.uint8)

        # print('zed_depth_lwir.shape', zed_depth_lwir.shape)
        # print('zed_depth_mwir.shape', zed_depth_mwir.shape)
        # print('lidar_depth_lwir.shape', lidar_depth_lwir.shape)
        # print('lidar_depth_mwir.shape', lidar_depth_mwir.shape)
        # print('thermal_lwir_np.shape', thermal_lwir_np.shape, thermal_lwir_np.dtype)

        # lwir_depth = cv2.addWeighted(zed_depth_lwir, 0.5, lidar_depth_lwir, 0.5, 0.0)
        # lwir_depth = cv2.cvtColor(lwir_depth, cv2.COLOR_GRAY2RGB)
        # print('lwir_depth.shape', lwir_depth.shape, lwir_depth.dtype)
        # thermal_lwir_np = cv2.addWeighted(thermal_lwir_np, 0.5, lwir_depth, 0.5, 0.0)

        # mwir_depth = cv2.addWeighted(zed_depth_mwir, 0.5, lidar_depth_mwir, 0.5, 0.0)
        # mwir_depth = cv2.cvtColor(mwir_depth, cv2.COLOR_GRAY2RGB)
        # thermal_mwir_np = cv2.addWeighted(thermal_mwir_np, 0.5, mwir_depth, 0.5, 0.0)

        ##########################################
        # Visualization start
        final_vis = zed2i_left_np.copy()

        thermal_lwir_np = cv2.resize(thermal_lwir_np, (0, 0), fx=1.3, fy=1.3)
        final_vis[
            final_vis.shape[0] - thermal_lwir_np.shape[0] :,
            0 : thermal_lwir_np.shape[1],
            :,
        ] = thermal_lwir_np  # cv2.cvtColor(thermal_lwir_np, cv2.COLOR_GRAY2RGB)

        thermal_mwir_np = cv2.resize(thermal_mwir_np, (0, 0), fx=0.65, fy=0.65)
        final_vis[
            final_vis.shape[0] - thermal_mwir_np.shape[0] :,
            final_vis.shape[1] - thermal_mwir_np.shape[1] :,
            :,
        ] = thermal_mwir_np  # cv2.cvtColor(thermal_mwir_np, cv2.COLOR_GRAY2RGB)

        print("zed2i_left_np.shape", zed2i_left_np.shape)
        print("thermal_lwir_np.shape", thermal_lwir_np.shape)
        print("thermal_mwir_np.shape", thermal_mwir_np.shape)
        print("final_vis.shape", final_vis.shape)
        cv2.imwrite("test_media/zed2i_left_np.png", zed2i_left_np)
        cv2.imwrite("test_media/thermal_lwir_np.png", thermal_lwir_np)
        cv2.imwrite("test_media/thermal_mwir_np.png", thermal_mwir_np)
        cv2.imwrite("test_media/final_vis.png", final_vis)
        exit()

        out.write(final_vis)

        # print(final_frame.shape, depth_np.shape, thermal_lwir_np.shape, thermal_mwir_np.shape)

        # exit()
    out.release()
