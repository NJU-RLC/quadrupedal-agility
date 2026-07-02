## Raw Motion Capture Data
We use motion capture data provided by AI4Animation: https://github.com/sebastianstarke/AI4Animation/tree/master/AI4Animation/SIGGRAPH_2018. The original motion capture data is in BVH format. You may optionally install MotionBuilder on Windows and export it to FBX format.

## Run the following files one by one to export the JSON files required for training:
1. `get_dog_tpose.py`: obtain the quadruped T-pose from AI4Animation
2. `get_go2_tpose.py`: obtain the Go2 T-pose
3. `fbx2npy.py`: convert all FBX files to NPY files
4. `fbx_importer.py` (optional): visualize a specific motion capture clip
5. `clip_config.json`: edit the JSON configuration file to specify the file name and clip range
6. `key_points_retargetting.py`: perform motion retargeting and extract the desired clip
7. `json_exporter.py`: export the JSON-format files required for training

In `key_points_retargetting.py`, you can use `plot_skeleton_motion_interactive(target_motion)` to visualize the skeleton animation. The key bindings are:

- `w` - loop mode
- `x` - play/pause animation
- `z` - previous frame
- `c` - next frame
- `n` - quit window

