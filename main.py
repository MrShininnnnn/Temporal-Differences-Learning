#! /usr/bin/env python
__author__ = 'Ning Shi'
__email__ = 'mrshininnnnn@gmail.com'


# import dependency
import argparse
import numpy as np
import matplotlib.pyplot as plt


class BoundedRandomWalks(object):
      """docstring for BoundedRandomWalks"""
      def __init__(self, random_seed=0, seq_len=10, batch_num=100, seq_num=10, unique=False):
            super(BoundedRandomWalks, self).__init__()
            # define the random seed to reproduce the same result
            self.random_seed = random_seed
            # a total of 100 training sets
            self.train_batch = batch_num
            # each training set contain 10 sequences
            self.batch_size = seq_num
            # limit the length of each sequence
            self.seq_len = seq_len
            # ensure no duplicates in a training set
            self.unique = unique
            # the true probabilities for each nonterminal states are 1/6, 1/3, 1/2, 2/3 and 5/6
            self.ideal_pred = np.array([1/6, 1/3, 1/2, 2/3, 5/6], dtype=np.float64).reshape(1, 5)
            # generate the training set
            self.train_set = self.generate_data()

      def cal_unique_rate(self, batch):
            # calculate the unique ratio of a training set
            unique_batch = []
            for seq in batch: 
                  act_seq = [np.argmax(a) for a in seq[:-1]] 
                  if act_seq not in unique_batch: 
                        unique_batch.append(act_seq)
            return len(unique_batch)/len(batch)

      def generate_data(self):
            # define the initial step at state D
            start_state = np.array([0, 0, 1, 0, 0], dtype=np.int).reshape(5, 1)
            # apply the random seed
            np.random.seed(self.random_seed)
            # initial the data list
            batch_list = []
            # for loop each batch
            while True:
                  # initial the batch list
                  sequence_list = []
                  # for loop each sequence
                  for _ in range(self.batch_size):
                        # every sequence begins in the center state D
                        step_list = [start_state]
                        step_position = 2
                        # while loop each step
                        while True:
                              # take a random walk
                              step_transition = np.random.choice([-1, 1])
                              step_position += step_transition
                              # terminate when A or G is reached
                              if step_position == -1:
                                    step_list.append(0)
                                    break
                              elif step_position == 5:
                                    step_list.append(1)
                                    break
                              else:
                                    # step representation
                                    state_vector = np.zeros((5, 1), dtype=np.int)
                                    state_vector[step_position] = 1
                                    # append step to the sequence
                                    step_list.append(state_vector)
                        # append sequence to the batch
                        sequence_list.append(step_list)
                  # append batch to the data list
                  if np.mean([len(s) for s in sequence_list]) <= self.seq_len:
                        if self.unique:
                              if self.cal_unique_rate(sequence_list) == 1:
                                    batch_list.append(sequence_list)
                        else: 
                              batch_list.append(sequence_list)
                  if len(batch_list) == self.train_batch:
                        break
            return batch_list

      def td_learning_op(self, seq_list, td_lambda, lr, w):
            # initialize the delta w
            delta_w = np.zeros((1, 5))
            # initialize the lambda
            lambda_matrix = np.ones((1, 1))
            # the sequence of steps
            step_list = seq_list[:-1]
            # the outcome of the sequence
            z = seq_list[-1]
            # for loop steps for repeated weight
            for t in range(len(step_list)):
                  # get steps so far with t
                  t_steps = np.array(step_list[0:t+1]).reshape(t+1, -1)
                  # terminal
                  if t == len(step_list) - 1:
                        cur_step = step_list[-1]
                        delta_p = z - np.dot(w, cur_step)
                  # non-terminal
                  else:
                        delta_p = np.dot(w, step_list[t+1]) - np.dot(w, step_list[t])
                  # calculate the delta w
                  delta_w += lr * delta_p * np.sum(np.dot(lambda_matrix.T, t_steps), axis=0)
                  # an exponential weighting with recency
                  lambda_matrix = np.concatenate(((lambda_matrix*td_lambda), np.ones((1, 1))))
            return delta_w

      def cal_rmse(self, pred, true):
            return np.sqrt(np.mean((pred-true)**2))

      def fig3(self):
            """
            Figure for the expeiment #1 Repeated Presentation
            + The weight vector was updated after the complete presentation of a training set
            + Each training set was presented repeatedly to each learning procedure until a convergence
            + The measure was averaged over 100 training sets
            """
            print('Reproduce Figure 3 in Sutton (1988)')
            # the path to save the results
            if self.unique:
                  FIGURE_3_PATH = 'img/exploration_figure_3.png'
            else: 
                  FIGURE_3_PATH = 'img/figure_3.png'
            # config for figure 3
            learning_rate = 0.01
            convergence_criteria = 0.001
            lambda_list = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
            # to save results of each lambda
            lambda_error_list = []
            # for loop each lambda value
            for lambda_val in lambda_list:
                  print('Train of Lambda {}'.format(lambda_val))
                  # to save results of each batch
                  batch_error_list = []
                  # for loop batches
                  for batch in self.train_set:
                        # initialize w randomly
                        w = np.random.rand(1, 5)
                        # start weight updating
                        while True:
                              # for loop sequences
                              delta_w = np.sum([self.td_learning_op(seq, lambda_val, learning_rate, w) for seq in batch], axis=0)
                              if np.linalg.norm(delta_w) > convergence_criteria:
                                    w += delta_w
                              else:
                                    break
                        # evaluation 
                        error = self.cal_rmse(w, self.ideal_pred)
                        batch_error_list.append(error)
                  # averaged measure over 100 training sets
                  lambda_error_list.append(np.mean(batch_error_list))
            # draw the figure 3
            plt.subplots(figsize = (8, 8), dpi=100)
            plt.plot(lambda_error_list, marker='o')
            plt.ylabel('ERROR USING α = {}'.format(learning_rate), fontsize=12)
            plt.xlabel('λ', fontsize=12)
            plt.xticks(range(len(lambda_error_list)), [str(l) for l in lambda_list])
            plt.title('Replication of Figure 3 in Sutton (1988)', fontsize=12)
            plt.savefig(FIGURE_3_PATH)
            print('Saving Figure 3 to {}\n'.format(FIGURE_3_PATH))

      def fig4(self):
            """
            The second experiment concerns the question of learning rate 
            when the training set is presented just once rather than repeatedly until convergence. 
            + Each training set was presented once to each procedure
            + Weight updates were performed after each sequence
            + Each learning procudure was applied with a range of values for the learning rate
            + All components of the weight vector were initially set to 0.5
            """
            print('Reproduce Figure 4 in Sutton (1988)')
            lambda_list = [0.0, 0.3, 0.8, 1.0]
            lr_list = np.arange(13)*0.05
            # the path to save the results
            if self.unique:
                  FIGURE_4_PATH = 'img/exploration_figure_4.png'
            else:
                  FIGURE_4_PATH = 'img/figure_4.png'
            # record results of various lambda
            evaluation_history_dict = dict()
            # for loop lambda from [0.0, 0.3, 0.8, 1.0]
            for lambda_val in lambda_list:
                  print('Train of Lambda {}'.format(np.round(lambda_val, decimals=2)))
                  # save results of each lambda
                  lambda_error_list = []
                  # for loop alpha as the learning rate
                  for learning_rate in lr_list:
                        # save results of each batch
                        batch_error_list = []
                        # for loop training set
                        for batch in self.train_set:
                              # initialize w
                              w = np.ones((1, 5))*0.5
                              # for loop sequences
                              for seq in batch:
                                    # update weight after each sequence
                                    w += self.td_learning_op(seq, lambda_val, learning_rate, w)
                              # evaluation after each batch
                              error = self.cal_rmse(w, self.ideal_pred)
                              batch_error_list.append(error)
                        lambda_error_list.append(np.mean(batch_error_list))
                  evaluation_history_dict[lambda_val] = lambda_error_list

            # cut off the y axis at error 0.7 for Lambda 1.0
            for i in range(len(evaluation_history_dict[1.0])):
                if evaluation_history_dict[1.0][i] > 0.7:
                    break
            evaluation_history_dict[1.0] = evaluation_history_dict[1.0][:i+1]

            # draw the figure 4
            plt.subplots(figsize = (8, 8), dpi=100)
            for lambda_val in lambda_list:
                  plt.plot(evaluation_history_dict[lambda_val], 
                        label='λ = {}'.format(np.round(lambda_val, decimals=2)), 
                        marker='o', 
                        ms=5)
            plt.xlabel('α', fontsize=12)
            plt.ylabel('ERROR', fontsize=12)
            plt.xticks([0, 2, 4, 6, 8, 10, 12], 
                  np.array([str(np.round(lr, decimals=2)) for lr in lr_list])[[0, 2, 4, 6, 8, 10, 12]])
            plt.legend(loc='best')
            plt.title('Replication of Figure 4 in Sutton (1988)', fontsize=12)
            plt.savefig(FIGURE_4_PATH)
            print('Saving Figure 4 to {}\n'.format(FIGURE_4_PATH))

      def fig5(self):
            """
            Figure 5 plots the best error level achieved for each λ value, 
            that is, using the α value that was best for that λ value.
            + Each data point represents the average over 100 training sets
            + The λ value is given by the horizontal coordinate
            + Each α value is picked from those shown in Figure 4 to yield the lowest error
            """
            print('Reproduce Figure 4 in Sutton (1988)')
            # the path to save the results
            if self.unique:
                  FIGURE_5_PATH = 'img/exploration_figure_5.png'
            else:
                  FIGURE_5_PATH = 'img/figure_5.png'
            print('Find Best Alpha for Each Lambda')
            # find best alpha for each lambda first
            lambda_list = np.arange(0, 1.1, 0.1)
            lr_list = np.arange(13)*0.05
            # record results of various lambda
            evaluation_history_dict = dict()
            # for loop each lambda value
            for lambda_val in lambda_list: 
                  print('Train of Lambda {}'.format(np.round(lambda_val, decimals=2)))
                  # to save results of each lambda
                  lambda_error_list = []
                  for learning_rate in lr_list:
                        # to save results of each batch
                        batch_error_list = []
                        # for loop batches
                        for batch in self.train_set:
                              # initialize w
                              w = np.ones((1, 5))*0.5
                              # start weight updating
                              for seq in batch:
                                  # update weight after each sequence
                                  w += self.td_learning_op(seq, lambda_val, learning_rate, w)
                              # evaluation 
                              error = self.cal_rmse(w, self.ideal_pred)
                              batch_error_list.append(error)
                        # averaged measure over 100 training sets
                        lambda_error_list.append(np.mean(batch_error_list))
                  evaluation_history_dict[lambda_val] = lambda_error_list
            # show best alpha for each lambda
            best_lr_list = []
            for lambda_val in evaluation_history_dict:
                  best_lr_index = np.argmin(evaluation_history_dict[lambda_val])
                  best_lr = lr_list[best_lr_index]
                  best_lr_list.append(np.round(best_lr, decimals=2))
                  print('Best Alpha {} for Lambda {}'.format(np.round(best_lr, decimals=2), 
                        np.round(lambda_val, decimals=2)))
            print('Re-Train Using Best Alpha for Each Lambda')
            # to save results of each lambda
            lambda_error_list = []
            # for loop each lambda value
            for lambda_val, learning_rate in zip(np.round(lambda_list, decimals=2), best_lr_list):
                  print('Train of Lambda {} Alpha {}'.format(lambda_val, learning_rate))
                  # to save results of each batch
                  batch_error_list = []
                  # for loop batches
                  for batch in self.train_set:
                        # initialize w
                        w = np.ones((1, 5))*0.5
                        # start weight updating
                        for seq in batch:
                              # update weight after each sequence
                              w += self.td_learning_op(seq, lambda_val, learning_rate, w)
                        # evaluation 
                        error = self.cal_rmse(w, self.ideal_pred)
                        batch_error_list.append(error)
                  # averaged measure over 100 training sets
                  lambda_error_list.append(np.mean(batch_error_list))
            # draw the figure 5
            plt.subplots(figsize = (8, 8), dpi=100)
            plt.plot(lambda_error_list, marker='o')
            plt.ylabel('ERROR USING BEST α', fontsize=12)
            plt.xlabel('λ', fontsize=12)
            plt.xticks([0, 2, 4, 6, 8, 10], np.round(np.array(lambda_list), decimals=2)[[0, 2, 4, 6, 8, 10]])
            plt.title('Replication of Figure 5 in Sutton (1988)', fontsize=12)
            plt.savefig(FIGURE_5_PATH)
            print('Saving Figure 5 to {}\n'.format(FIGURE_5_PATH))

def main():
      # default hyper-parameters
      # random_seed=0, seq_len=10, batch_num=100, seq_num=10, unique=False
      parser = argparse.ArgumentParser(description='Set Hyper-Parameters')
      parser.add_argument('--random_seed', type=int, default=0)
      parser.add_argument('--seq_len', type=int, default=10)
      parser.add_argument('--batch_num', type=int, default=100)
      parser.add_argument('--seq_num', type=int, default=10)
      parser.add_argument('--unique', type=bool, default=False)
      args = parser.parse_args()

      print('Start!')
      brw = BoundedRandomWalks(
            random_seed=args.random_seed, 
            seq_len = args.seq_len, 
            batch_num = args.batch_num, 
            seq_num = args.seq_num, 
            unique = args.unique)
      brw.fig3()
      brw.fig4()
      brw.fig5()
      print('Done!')

if __name__ == '__main__':
      main()