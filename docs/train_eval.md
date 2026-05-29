# Prerequisites

**Please ensure you have prepared the environment, the nuScenes dataset and the annotations.**

# Train and Test

## 2 Stage Training

CorDriver is trained with 2 stages. In the first stage, perception, planning and corridor prediction are trained for 48 epochs for a reasonable initialization. In the second stage, imitation loss is incorporated with differentiable optimization while the perception modules are frozen.

## Stage 1 Training
```shell
conda activate corridor
sh tools/scripts/train_stage_1.sh
```
It takes about 3 days to train on 4 A100s. You can download the stage 1 model via [google drive](https://drive.google.com/file/d/1KBYD4h4aKJHyOE9LWxw56k3Jky2-zP_E/view?usp=drive_link).

## Stage 2 Training with Trajectory Optimization
```shell
sh tools/scripts/train_stage_2.sh
```
It takes about 1 day to train the second stage. You can download the stage 2 model via [google drive](https://drive.google.com/file/d/1rVqycviJQOsJP3FYJLqNwsIEZ34lex5B/view?usp=drive_link).

## Evaluation 
Evaluation runs on one gpu.
```shell
sh tools/scripts/eval.sh
```
This evaluates the stage 2 model. Optionally, you can set the `clip` variable in `Corridor_stage.py` to enable corridor refinement, as described in Section III.D.