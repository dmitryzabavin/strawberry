import numpy as np
import pandas as pd

import pulse_00_01_00

def get_gulp(some_gulps):
    if len(some_gulps) > 0:
        best_gulp = some_gulps[0]
        for every_gulp in some_gulps[1:]:
            if every_gulp['score_with_gulp'] > best_gulp['score_with_gulp']:
                best_gulp = every_gulp
        return best_gulp['chosen_with_gulp'], best_gulp['rest_after_gulp'], best_gulp['score_with_gulp']

from sklearn.model_selection import train_test_split

class strawberry:
    def __init__(self, pattern_dataset, source_dataset):
        self.pattern = pattern_dataset
        self.source = source_dataset
        self.columns = set(self.pattern.columns) and set(self.source.columns)

    def compare(self, columns=None):
        if columns is not None:
            columns = set(self.columns) and set(columns)
        else:
            columns = self.columns
        comparison_log = []
        for every_column in columns:
            comparison = pulse(self.pattern.loc[:, every_column], self.source.loc[:, every_column])
            is_different = comparison.is_different()
            is_similar = comparison.is_similar()
            comparison_log.append({
                'column': every_column,
                'is_different': is_different,
                'difference_score': 100 * (1 - is_different),
                'is_similar': is_similar,
                'similarity_score': 100 * (1 - is_similar)
            })
        return pd.DataFrame(comparison_log)

    def squeeze(self, n_samples=None, base_rate=0.1, gulp_rate=0.1, vain_steps=7, short_rate=0.9, filter_threshold = None, n_gulps = 2, random_state=0):
        if n_samples is None:
            n_samples = self.source.shape[0]
        rest_set, base_set = train_test_split(self.source, test_size=base_rate, random_state=random_state)
        chosen_set = base_set
        current_score = strawberry(self.pattern, base_set).compare().loc[:, 'is_similar'].median(axis=0, skipna=True)
#        current_score = strawberry(self.pattern, base_set).compare().loc[:, 'is_similar'].max(axis=0, skipna=True)
#        current_score = strawberry(self.pattern, base_set).compare().loc[:, 'is_similar'].sum(axis=0, skipna=True)
        current_gulp_rate = gulp_rate
        vain_steps_done = 0
        step_number = -1
        gulps = []
        while (round(current_gulp_rate * rest_set.shape[0]) > 1) and ((n_samples is None) or (n_samples > chosen_set.shape[0])):
            step_number += 1
            print('COLLECTED:', chosen_set.shape[0])
            if vain_steps_done == vain_steps:
                current_gulp_rate *= short_rate
                vain_steps_done = 0
                print('NEW GULP RATE: ', current_gulp_rate, '; NEW GULP SIZE: ', round(current_gulp_rate * rest_set.shape[0]))
            else:
#                rest_after_gulp, gulp = train_test_split(rest_set, test_size=gulp_rate, random_state=vain_steps_done)
                rest_after_gulp, gulp = train_test_split(rest_set, test_size=gulp_rate, random_state=(step_number + random_state))
                chosen_with_gulp = pd.concat([chosen_set, gulp])
                score_with_gulp = strawberry(self.pattern, chosen_with_gulp).compare().loc[:, 'is_similar'].median(axis=0, skipna=True)
#                score_with_gulp = strawberry(self.pattern, chosen_with_gulp).compare().loc[:, 'is_similar'].max(axis=0, skipna=True)
#                score_with_gulp = strawberry(self.pattern, chosen_with_gulp).compare().loc[:, 'is_similar'].sum(axis=0, skipna=True)
                print('SCORES - NEW: ', score_with_gulp, '; PREVIOUS: ', current_score)
                if score_with_gulp < current_score:
                    vain_steps_done = 0
                    if (filter_threshold is None) or ((float(current_score - score_with_gulp) / current_score) > filter_threshold):
                        current_score = score_with_gulp
                        chosen_set = chosen_with_gulp
                        rest_set = rest_after_gulp
                        print('RESULT SIZE: ', chosen_set.shape[0])
                    else:
                        if n_gulps is not None:
                            gulps.append({
                                'chosen_with_gulp': chosen_with_gulp,
                                'rest_after_gulp': rest_after_gulp,
                                'score_with_gulp': score_with_gulp
                            })
                            if len(gulps) >= n_gulps:
                                chosen_set, rest_set, current_score = get_gulp(gulps)
                                gulps = []
                                print('RESULT SIZE: ', chosen_set.shape[0])
                else:
                    vain_steps_done += 1
        return chosen_set


if __name__ == '__main__':
    pass
