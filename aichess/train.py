"""使用收集到数据进行训练"""


import random
import numpy as np
import pickle
import time
from net import PolicyValueNet
from config import CONFIG


# 定义整个训练流程
class TrainPipeline:

    def __init__(self, init_model=None, init_opt=None):
        # 训练参数
        self.learn_rate = 1e-3
        self.temp = 1.0
        self.batch_size = CONFIG['batch_size']  # 训练的batch大小
        self.epochs = CONFIG['epochs']  # 每次更新的train_step数量
        self.kl_targ = CONFIG['kl_targ']  # kl散度控制
        self.check_freq = 100  # 保存模型的频率
        self.game_batch_num = CONFIG['game_batch_num']  # 训练更新的次数

        try:
            self.lr_multiplier = np.load(CONFIG['lr_multiple'])  # 基于KL自适应的调整学习率
        except:
            self.lr_multiplier = 1  # 基于KL自适应的调整学习率

        if init_model:
            try:
                self.policy_value_net = PolicyValueNet(model_file=init_model, opt_file=init_opt)
                print('已加载上次最终模型')
            except:
                # 从零开始训练
                print('模型路径不存在，从零开始训练')
                self.policy_value_net = PolicyValueNet()
        else:
            print('从零开始训练')
            self.policy_value_net = PolicyValueNet()

    def policy_updata(self):
        """更新策略价值网络"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)

        state_batch = [data[0] for data in mini_batch]
        state_batch = np.array(state_batch).astype('float32')

        mcts_probs_batch = [data[1] for data in mini_batch]
        mcts_probs_batch = np.array(mcts_probs_batch).astype('float32')

        winner_batch = [data[2] for data in mini_batch]
        winner_batch = np.array(winner_batch).astype('float32')

        # 旧的策略，旧的价值函数
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)

        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                state_batch,
                mcts_probs_batch,
                winner_batch,
                self.learn_rate * self.lr_multiplier
            )
            # 新的策略，新的价值函数
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)

            kl = np.mean(np.sum(old_probs * (
                np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                                axis=1))
            if kl > self.kl_targ * 4:  # 如果KL散度很差，则提前终止
                break

        # 自适应调整学习率
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = (1 -
                            np.var(np.array(winner_batch) - old_v.flatten()) /
                            np.var(np.array(winner_batch)))
        explained_var_new = (1 -
                            np.var(np.array(winner_batch) - new_v.flatten()) /
                            np.var(np.array(winner_batch)))

        print(("kl:{:.5f},"
               "lr_multiplier:{:.3f},"
               "loss:{},"
               "entropy:{},"
               "explained_var_old:{:.3f},"
               "explained_var_new:{:.3f}"
               ).format(kl,
                        self.lr_multiplier,
                        loss,
                        entropy,
                        explained_var_old,
                        explained_var_new))
        return loss, entropy

    def run(self):
        """开始训练"""
        try:
            for i in range(self.game_batch_num):
                if i >= 10:  # 连续更新10次，初始化模型
                    time.sleep(480)  # 每8分钟更新一次模型
                while True:
                    try:
                        with open(CONFIG['train_data_buffer_path'], 'rb') as data_dict:
                            data_file = pickle.load(data_dict)
                            self.data_buffer = data_file['data_buffer']
                            self.iters = data_file['iters']
                            del data_file
                        print('已载入数据')
                        break
                    except:
                        time.sleep(30)
                print('step i {}: '.format(self.iters), '数据总长：', len(self.data_buffer))
                if len(self.data_buffer) >= 4000:
                    self.batch_size = CONFIG['batch_size']
                elif len(self.data_buffer) >= 3000:
                    self.batch_size = 3000
                elif len(self.data_buffer) >= 2000:
                    self.batch_size = 2000
                elif len(self.data_buffer) >= 1000:
                    self.batch_size = 1000
                else:
                    self.batch_size = len(self.data_buffer)

                loss, entropy = self.policy_updata()

                # 保存模型
                self.policy_value_net.save_model(CONFIG['model_path'], CONFIG['opt_path'])
                np.save(CONFIG['lr_multiple'], self.lr_multiplier)
                if (i + 1) % self.check_freq == 0:
                    print('current selfplay batch: {}'.format(i + 1))
                    self.policy_value_net.save_model('models/current_policy_batch{}.model'.format(i + 1))
        except KeyboardInterrupt:
            print('\n\rquit')


training_pipeline = TrainPipeline(init_model='current_policy.model', init_opt='current_policy.pdopt')
training_pipeline.run()
