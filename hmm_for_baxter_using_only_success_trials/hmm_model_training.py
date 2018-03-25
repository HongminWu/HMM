#!/usr/bin/env python
import os
import numpy as np
from sklearn.externals import joblib
from sklearn.preprocessing import (
    scale,
    normalize
)
import util
import copy
import model_generation
import model_score
import training_config
import matplotlib.pylab as plt
import pandas as pd
from matplotlib import cm
from matplotlib import colors
import ipdb

def run(model_save_path,
    model_type,
    model_config,
    score_metric,
    trials_group_by_folder_name):

    trials_group_by_folder_name = util.make_trials_of_each_state_the_same_length(trials_group_by_folder_name)
    list_of_trials = trials_group_by_folder_name.values()

    trials_amount = len(trials_group_by_folder_name)

    if not os.path.isdir(model_save_path):
        os.makedirs(model_save_path)

    one_trial_data_group_by_state = list_of_trials[0]
    state_amount = len(one_trial_data_group_by_state)

    training_data_group_by_state = {}
    training_length_array_group_by_state = {}

    for state_no in range(1, state_amount+1):
        length_array = []
        for trial_no in range(len(list_of_trials)):
            length_array.append(list_of_trials[trial_no][state_no].shape[0])
            if trial_no == 0:
                data_tempt = list_of_trials[trial_no][state_no]
            else:
                data_tempt = np.concatenate((data_tempt,list_of_trials[trial_no][state_no]),axis = 0)
        training_data_group_by_state[state_no] = data_tempt
        training_length_array_group_by_state[state_no] = length_array

    if not os.path.isdir(model_save_path):
        os.makedirs(model_save_path)

    for state_no in range(1, state_amount+1):
        model_list = []
        model_generator = model_generation.get_model_generator(model_type, model_config)

        X = training_data_group_by_state[state_no]
        lengths = training_length_array_group_by_state[state_no]
        lengths[-1] -=1 # Adapting for bnpy's observation is firt-order autoregressive gaussian
        for model, now_model_config in model_generator:
            print
            print '-'*20
            print 'in state', state_no, ' working on config:', now_model_config
            model = model.fit(X, lengths=lengths)  #n_samples, n_features
            score = model_score.score(score_metric, model, X, lengths)

            if score == None:
                print "scorer says to skip this model, will do"
                continue

            model_list.append({
                "model": model,
                "now_model_config": now_model_config,
                "score": score
            })
            print 'score:', score
            print '='*20
            print

            model_generation.update_now_score(score)

        sorted_model_list = sorted(model_list, key=lambda x:x['score'])

        best = sorted_model_list[0]
        model_id = util.get_model_config_id(best['now_model_config'])

        joblib.dump(
            best['model'],
            os.path.join(model_save_path, "model_s%s.pkl"%(state_no,))
        )

        joblib.dump(
            best['now_model_config'],
            os.path.join(
                model_save_path,
                "model_s%s_config_%s.pkl"%(state_no, model_id)
            )
        )

        joblib.dump(
            None,
            os.path.join(
                model_save_path,
                "model_s%s_score_%s.pkl"%(state_no, best['score'])
            )
        )

        train_report = [{util.get_model_config_id(i['now_model_config']): i['score']} for i in sorted_model_list]
        import json
        json.dump(
            train_report,
            open(
                os.path.join(
                    model_save_path,
                    "model_s%s_training_report.json"%(state_no)
                ), 'w'
            ),
            separators = (',\n', ': ')
        )


'''
        # plot the hidden state sequence for each state
        print
        print
        print 'Finish fitting the posterior model -> Generating the hidden state sequence...'
        print
        print
        if model_type == 'hmmlearn\'s HMM':
            _, model.z = model.decode(X, algorithm="viterbi")

        elif model_type == 'BNPY\'s HMM':
            model.z = model.decode(X, lengths)

        elif model_type == 'PYHSMM\'s HMM':
            model.z = model.model.stateseqs[0]

        elif model_type == 'hmmlearn\'s GMMHMM':
            _, model.z = model.decode(X, algorithm="viterbi")

        else:
            print 'Sorry, this model cannot obtain the hidden state sequence'
            return

        im_data  = np.tile(model.z, 2)
        cmap =cm.get_cmap('jet',np.max(model.z))
        print np.unique(model.z)
        ax.imshow(im_data[None], aspect='auto', interpolation='nearest', vmin = 0, vmax = np.max(model.z), cmap = cmap, alpha = 0.5)
#        trial_len = len(model.z) / trials_amount
#        color=iter(cm.rainbow(np.linspace(0, 1, trials_amount)))
#        for iTrial in range(trials_amount):
#            ax.plot(model.z[iTrial*trial_len:(iTrial+1)*trial_len], color=next(color)) #, linewidth=2.0
#            plt.draw()

#        plt.gcf().suptitle('The hidden state_sequence of state_%d' % (state_no))
#        plt.gcf().savefig(model_save_path + '/hidden_state_seq.jpg', format="jpg")
        zdf = pd.DataFrame(model.z)
        zdf.to_csv(model_save_path + '/hidden_stateSeq.csv')
        joblib.dump(model.z, model_save_path + '/hidden_stateSeq.pkl')
'''
