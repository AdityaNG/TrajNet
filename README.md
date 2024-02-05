# Thermal Voyager: A Comparative Study of RGB and Thermal Cameras for Night-Time Autonomous Navigation

<img src="media/demo.gif">

<b>Abstract</b> Achieving reliable autonomous navigation during nighttime remains a substantial obstacle in the ﬁeld of robotics. Although systems utilizing Light Detection and Ranging (LiDAR) and Radio Detection and Ranging (RADAR) enables environmental perception regardless of lighting conditions, they face signiﬁcant challenges in environments with a high density of agents due to their dependence on active emissions. Cameras operating in the visible spectrum represent a quasi-passive alternative, yet they see a substantial drop in efﬁciency in low-light conditions, consequently hindering both scene perception and path planning. Here, we introduce a novel end-to-end navigation system, the ”Thermal Voyager”, which leverages infrared thermal vision to achieve true passive perception in autonomous entities. The system utilizes TrajNet to interpret thermal visual inputs to produce desired trajectories and employs a model predictive control strategy to determine the optimal steering angles needed to actualize those trajectories. We train the TrajNet utilizing a comprehensive video dataset incorporating visible and thermal footages alongside Controller Area Network (CAN) frames. We demonstrate that nighttime navigation facilitated by Long-Wave Infrared (LWIR) thermal cameras can rival the performance of daytime navigation systems using RGB cameras. Our work paves the way for scene perception and trajectory prediction empowered entirely by passive thermal sensing technology, heralding a new era where autonomous navigation is both feasible and reliable irrespective of the time of day. We make our code and thermal trajectory dataset public.


<b>Project Goal</b> Train TrajNet to produce a trajectory prediction for a given a BEV image.

# Getting Started

## Docker Environment

To build, use:
```bash
DOCKER_BUILDKIT=1 docker compose build
```

To run the interactive shell, use:
```bash
docker compose run dev
```

## Demo

## Train

Train v1 model on lwir_raw:
```bash
nohup docker compose run dev python3.9 -m TrajNet.scripts.train_TrajNet \
    --version 1 \
    --dataset lwir_raw \
    --model_type dpt_swin2_tiny_256 \
    --device cuda:1 \
    --sweep_json config/TrajDPT_V1_dpt_swin2_tiny_256_regress_Aug_12.json &

nohup docker compose run dev python3.9 -m TrajNet.scripts.train_TrajNet \
    --version 1 \
    --dataset lwir_raw \
    --model_type dpt_swin2_tiny_256 \
    --device cuda:1 \
    --sweep_json config/TrajDPT_V1_dpt_swin2_tiny_256_templates_Aug_12.json &
```

Train v1 model on rgb dataset:
```bash
nohup docker compose run dev python3.9 -m TrajNet.scripts.train_TrajNet \
    --version 1 \
    --dataset rgb \
    --model_type dpt_swin2_tiny_256 \
    --device cuda:0 \
    --sweep_json config/TrajDPT_V1_dpt_swin2_tiny_256_regress_Aug_12.json &

nohup docker compose run dev python3.9 -m TrajNet.scripts.train_TrajNet \
    --version 1 \
    --dataset rgb \
    --model_type dpt_swin2_tiny_256 \
    --device cuda:1 \
    --sweep_json config/TrajDPT_V1_dpt_swin2_tiny_256_templates_Aug_12.json &
```

Train v1 model on lwir_norm dataset:
```bash
nohup docker compose run dev python3.9 -m TrajNet.scripts.train_TrajNet \
    --version 1 \
    --dataset lwir_norm \
    --model_type dpt_swin2_tiny_256 \
    --device cuda:0 \
    --sweep_json config/TrajDPT_V1_dpt_swin2_tiny_256_regress_lwir_norm_Aug_12.json &

nohup docker compose run dev python3.9 -m TrajNet.scripts.train_TrajNet \
    --version 1 \
    --dataset lwir_norm \
    --model_type dpt_swin2_tiny_256 \
    --device cuda:1 \
    --sweep_json config/TrajDPT_V2_dpt_swin2_tiny_256_templates_lwir_norm_Aug_12.json &
```

Train v1 model on rgb dataset in simulator:
```bash
docker compose run dev python3.9 -m TrajNet.scripts.simulator_rl \
    --version 1 \
    --dataset rgb \
    --model_type dpt_swin2_tiny_256 \
    --device cuda:0 \
    --sweep_json config/TrajDPT_V1_dpt_swin2_tiny_256_regress_SIM.json

nohup docker compose run dev python3.9 -m TrajNet.scripts.train_TrajNet \
    --version 1 \
    --dataset rgb \
    --model_type dpt_swin2_tiny_256 \
    --device cuda:1 \
    --sweep_json config/TrajDPT_V1_dpt_swin2_tiny_256_templates_SIM.json &
```

Train v2 model on rgb dataset:
```bash
nohup docker compose run dev python3.9 -m TrajNet.scripts.train_TrajNet \
    --version 2 \
    --dataset rgb \
    --model_type dpt_swin2_tiny_256 \
    --device cuda:0 \
    --sweep_json config/TrajDPT_V2_dpt_swin2_tiny_256_regress_Aug_12.json &

nohup docker compose run dev python3.9 -m TrajNet.scripts.train_TrajNet \
    --version 2 \
    --dataset rgb \
    --model_type dpt_swin2_tiny_256 \
    --device cuda:0 \
    --sweep_json config/TrajDPT_V2_dpt_swin2_tiny_256_templates_Aug_12.json &
```

Train v2 model on lwir_raw dataset:
```bash
nohup docker compose run dev python3.9 -m TrajNet.scripts.train_TrajNet \
    --version 2 \
    --dataset lwir_raw \
    --model_type dpt_swin2_tiny_256 \
    --device cuda:1 \
    --sweep_json config/TrajDPT_V2_dpt_swin2_tiny_256_regress_lwir_raw_Aug_12.json &

nohup docker compose run dev python3.9 -m TrajNet.scripts.train_TrajNet \
    --version 2 \
    --dataset lwir_raw \
    --model_type dpt_swin2_tiny_256 \
    --device cuda:0 \
    --sweep_json config/TrajDPT_V2_dpt_swin2_tiny_256_templates_lwir_raw_Aug_12.json &
```

Train v2 model on lwir_norm dataset:
```bash
nohup docker compose run dev python3.9 -m TrajNet.scripts.train_TrajNet \
    --version 2 \
    --dataset lwir_norm \
    --model_type dpt_swin2_tiny_256 \
    --device cuda:1 \
    --sweep_json config/TrajDPT_V2_dpt_swin2_tiny_256_regress_lwir_norm_Aug_12.json &

nohup docker compose run dev python3.9 -m TrajNet.scripts.train_TrajNet \
    --version 2 \
    --dataset lwir_norm \
    --model_type dpt_swin2_tiny_256 \
    --device cuda:0 \
    --sweep_json config/TrajDPT_V2_dpt_swin2_tiny_256_templates_lwir_norm_Aug_12.json &
```

Train v3 model on rgb dataset:
```bash
nohup docker compose run dev python3.9 -m TrajNet.scripts.train_TrajNet \
    --version 3 \
    --dataset rgb \
    --model_type dpt_swin2_tiny_256 \
    --device cuda:1 \
    --sweep_json config/TrajDPT_V3_dpt_swin2_tiny_256_regress_Aug_12.json &

nohup docker compose run dev python3.9 -m TrajNet.scripts.train_TrajNet \
    --version 3 \
    --dataset rgb \
    --model_type dpt_swin2_tiny_256 \
    --device cuda:0 \
    --sweep_json config/TrajDPT_V3_dpt_swin2_tiny_256_templates_Aug_12.json &
```

Train v3 model on lwir_raw dataset:
```bash
nohup docker compose run dev python3.9 -m TrajNet.scripts.train_TrajNet \
    --version 3 \
    --dataset lwir_raw \
    --model_type dpt_swin2_tiny_256 \
    --device cuda:0 \
    --sweep_json config/TrajDPT_V3_dpt_swin2_tiny_256_regress_lwir_raw_Aug_12.json &

nohup docker compose run dev python3.9 -m TrajNet.scripts.train_TrajNet \
    --version 3 \
    --dataset lwir_raw \
    --model_type dpt_swin2_tiny_256 \
    --device cuda:0 \
    --sweep_json config/TrajDPT_V3_dpt_swin2_tiny_256_templates_lwir_raw_Aug_12.json &

Train v3 model on lwir_norm dataset:
```bash
nohup docker compose run dev python3.9 -m TrajNet.scripts.train_TrajNet \
    --version 3 \
    --dataset lwir_norm \
    --model_type dpt_swin2_tiny_256 \
    --device cuda:1 \
    --sweep_json config/TrajDPT_V3_dpt_swin2_tiny_256_regress_lwir_norm_Aug_12.json &

nohup docker compose run dev python3.9 -m TrajNet.scripts.train_TrajNet \
    --version 3 \
    --dataset lwir_norm \
    --model_type dpt_swin2_tiny_256 \
    --device cuda:0 \
    --sweep_json config/TrajDPT_V3_dpt_swin2_tiny_256_templates_lwir_norm_Aug_12.json &
```


## Generate template trajectories

Generate the trajectory templates
```bash
python3.9 -m TrajNet.scripts.generate_trajectory_templates \
    --dataset rgb \
    --val_percent 0.05 \
    --dataset_percentage 0.01
```

# Running a docker command 

```bash
docker compose run dev python3.9 -m TrajNet.scripts.train_TrajNet \
    --version 1 \
    --dataset rgb \
    --model_type dpt_swin2_tiny_256 \
    --device cuda:0 \
    --sweep_json config/TrajDPT_V1_dpt_swin2_tiny_256_templates_Aug_12.json &
```
