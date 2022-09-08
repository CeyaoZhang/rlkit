import abc

import gtimer as gt
from rlkit.core.rl_algorithm import BaseRLAlgorithm
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.samplers.data_collector import PathCollector


class BatchRLAlgorithm(BaseRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector: PathCollector,
            evaluation_data_collector: PathCollector,
            replay_buffer: ReplayBuffer,
            batch_size,
            max_path_length,
            num_epochs,
            num_eval_steps_per_epoch,
            num_expl_steps_per_train_loop,
            num_trains_per_train_loop,
            num_train_loops_per_epoch=1,
            min_num_steps_before_training=0,
            start_epoch=0, # negative epochs are offline, positive epochs are online
            save_replay_buffer=False
    ):
        super().__init__(
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            evaluation_data_collector,
            replay_buffer,
            save_replay_buffer,
        )
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training
        self._start_epoch = start_epoch

    def train(self):
        """Negative epochs are offline, positive epochs are online"""
        ## self.num_epochs=3000
        for self.epoch in gt.timed_for(
                range(self._start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            # print(f'-->this is epoch {self.epoch}')
            self.offline_rl = self.epoch < 0 ## always False
            self._begin_epoch(self.epoch) ## pass
            self._train() ## key to train
            self._end_epoch(self.epoch) ## save params, replay buffers and record stats
        print("\n---------------------Finish!!!----------------------------")

    def _train(self):

        ## at each iteration, we first collect data, perform meta-updates, then try to evaluate
        if self.epoch == 0 and self.min_num_steps_before_training > 0: 
            print('\ncollecting initial pool of data for train and eval')
            init_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length, ## self.max_path_length=200
                self.min_num_steps_before_training, ## min_num_steps_before_training=2000, paths=[10xpath]
                discard_incomplete_paths=False,
            )
            if not self.offline_rl: ## alway true
                self.replay_buffer.add_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)
            print('Done collecting initial pool of data for train and eval\n')

        ##
        self.eval_data_collector.collect_new_paths(
            self.max_path_length,
            self.num_eval_steps_per_epoch, ## num_eval_steps_per_epoch=600, paths=[3xpath]
            discard_incomplete_paths=True,
        )
        gt.stamp('evaluation sampling')

        ## get the expl data, save to replay buffer, and train the SAC with batch from RB
        for _ in range(self.num_train_loops_per_epoch): ## 1
            new_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_expl_steps_per_train_loop, ## num_expl_steps_per_train_loop=1000, paths=[5xpath]
                discard_incomplete_paths=False,
            )
            gt.stamp('exploration sampling', unique=False)

            if not self.offline_rl: ## alway true
                self.replay_buffer.add_paths(new_expl_paths)
            gt.stamp('data storing', unique=False)

            ## train the SAC
            self.training_mode(True)
            for _ in range(self.num_trains_per_train_loop): ## 1000
                train_data = self.replay_buffer.random_batch(self.batch_size) ## 256, train_data = dict{np.array}
                self.trainer.train(train_data) ## train the policy
            gt.stamp('training', unique=False)
            self.training_mode(False)
