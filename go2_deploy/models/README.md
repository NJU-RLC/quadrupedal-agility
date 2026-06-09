# Model Weights

模型文件不建议提交到公开仓库。当前入口默认使用下面这些路径：

| 入口 | 模型路径 |
| --- | --- |
| `deploy_policy_with_controller_3` | `go2_deploy/models/ll_policy/3_model/model.pt` |
| `deploy_policy_with_controller_2` | `go2_deploy/models/ll_policy/2_model/model.pt` |
| `deploy_policy_with_controller_5` | `go2_deploy/models/ll_policy/5_model/model.pt` |
| `deploy_policy_with_depth_cruft` low | `go2_deploy/models/ll_policy/3_model/model.pt` |
| `deploy_policy_with_depth_cruft` high | `go2_deploy/models/high_policy/cruft/model.pt` |
| `deploy_policy_with_depth_hurdle` low | `go2_deploy/models/ll_policy/hurdle/model.pt` |
| `deploy_policy_with_depth_hurdle` high | `go2_deploy/models/high_policy/hurdle/model.pt` |
| recover | `go2_deploy/models/recover_policy/1_model/model.pt` |

暂时还缺 box 深度策略模型，已预留：

- `go2_deploy/models/ll_policy/box/model.pt`
- `go2_deploy/models/high_policy/box/model.pt`

低层 checkpoint 需要包含 `actor_critic` 和 `estimator`。

高层 checkpoint 需要包含 `estimator_state_dict`、`depth_encoder_state_dict` 和 `depth_actor_state_dict`。
