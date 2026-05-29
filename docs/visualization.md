# Visualization

We provide the script to visualize the Corridor prediction to a video.

## Visualize prediction

```shell
conda activate corridor
sh tools/scripts/visual.sh
```

It will generate images and merge them into a video. The parameter `--samples` controls the number of samples to render.
The image is composed of three part: projected corridor prediction on front view (upper), the reference trajectory on map and agent predictions (lower left) and optimized trajectory on g.t. map (lower right).
