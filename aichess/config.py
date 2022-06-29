CONFIG = {
    'dirichlet': 0.15,       # 国际象棋，0.3；日本将棋，0.15；围棋，0.03
    'play_out': 800,        # 每次移动的模拟次数
    'c_puct': 3,             # u的权重
    'buffer_size': 50000,   # 经验池大小
    'model_path': 'current_policy.model',      # 模型路劲
    'opt_path': 'current_policy.pdopt',   # 优化器路径
    'lr_multiple': 'lr_multiple.npy',   # lr缩放因子路径
    'train_data_buffer_path': 'train_data_buffer.pkl',   # 数据容器的路劲
    'batch_size': 4000,  # 每次更新的train_step数量
    'kl_targ': 0.02,  # kl散度控制
    'epochs' : 10,  # 每次更新的train_step数量
    'game_batch_num': 30000,  # 训练更新的次数
    'action_count': 100,  # 超过100步为平局，不加入训练
}