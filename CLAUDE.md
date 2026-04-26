# 仓库和论文基本信息

这个仓库fork了 "Adaptive Speculative Decoding with Reinforcement Learning" (https://arxiv.org/abs/2603.01639) 论文的代码仓库。请先阅读论文原文 @2603.01639v1.pdf 和 @README.md，了解其内容和方法，然后再查看代码实现。

# 任务目标

任务的目标是复现论文中的实验结果，主要是 Eagle3+RL 方法相对于 Eagle3 方法在不同模型上的性能提升。

# 修改和执行记录

- 新增目录：evaluate/ 用于存放评估相关的代码和结果，包括在不同模型、不同数据集的benchmark、结果记录、分析和可视化等等
- 新增目录：eagle/data/ 增加了额外的数据集，包括MATH、theoremqa等等
- 新增文件：checkpoints/ 用于存放RL模型的检查点文件，分别对应Qwen3、Llama3.1等模型，包括depth和size两种策略的模型

阅读完本文件，请输出 "已完成CLAUDE.md"。