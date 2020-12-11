import numpy as np
import pandas as pd
from statistics import median

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
    def get_discrete_columns(self, some_dataframe, some_threshold):
        return set(some_dataframe.columns[(some_dataframe.nunique() < some_threshold).values].values)

    def __init__(self, pattern_dataset, source_dataset, continues_columns=None, discrete_columns=None, distinct_values_threshold=100):
        self.pattern = pattern_dataset
        self.source = source_dataset
        self.columns = list(set(self.pattern.columns) and set(self.source.columns))
        if discrete_columns is not None:
            self.discrete_columns = list(set(self.columns) and set(discrete_columns))
        else:
            self.discrete_columns = list(get_discrete_columns(self.pattern, distinct_values_threshold) and get_discrete_columns(self.source, distinct_values_threshold))
        if continues_columns is not None:
            self.continues_columns = list(set(self.columns) and set(continues_columns))
        else:
            self.continues_columns = list(set(self.columns) - set(self.discrete_columns))

    def compare(self, columns=None):
        if columns is not None:
            columns = list(set(self.continues_columns) and set(columns))
        else:
            columns = self.continues_columns
        comparison_log = []
        for every_column in columns:
            comparison = pulse(self.pattern.loc[:, every_column], self.source.loc[:, every_column])
            is_different = comparison.is_different()
            is_similar = comparison.is_similar()
            comparison_log.append({
                'column': every_column,
                'is_different': is_different, # Square of probability-density-plots intersection. Approximately 1 - is_different
                'difference_score': 100 * (1 - is_different),
                'is_similar': is_similar, # 1 - square of probability-density-plots intersection. Approximately 1 - is_different
                'similarity_score': 100 * (1 - is_similar)
            })
        return pd.DataFrame(comparison_log)

    def discrete_compare(self, columns=None):
        if type(columns) != list:
            comparison_log = []
            pattern_counts = pd.DataFrame(self.pattern.groupby([columns]).size() / self.pattern.shape[0], columns=['pattern'])
            source_counts = pd.DataFrame(self.source.groupby([columns]).size() / self.source.shape[0], columns=['source'])
            pattern_nones = round(self.pattern.loc[:, [columns]].isna().values.sum() / self.pattern.shape[0], 3)
            source_nones = round(self.source.loc[:, [columns]].isna().values.sum() / self.source.shape[0], 3)

            comparison = pattern_counts.join(source_counts, how='outer').fillna(0.0)
            is_different = comparison.min(axis=1).sum().round(3) + min([pattern_nones, source_nones])
            is_similar = (((comparison['pattern'] - comparison['source']).abs().sum() + abs(pattern_nones - source_nones)) / 2.0).round(3)

            comparison_log.append({
                'column': columns,
                'is_different': is_different,
                # Square of probability-density-plots intersection. Approximately 1 - is_different
                'difference_score': 100 * (1 - is_different),
                'is_similar': is_similar,
                # 1 - square of probability-density-plots intersection. Approximately 1 - is_different
                'similarity_score': 100 * (1 - is_similar)
            })
            comparison_log = pd.DataFrame(comparison_log)
        else:
            if columns is not None:
                columns = list(set(self.discrete_columns) and set(columns))
            else:
                columns = self.discrete_columns
            comparison_log = pd.DataFrame()
#            if len(columns) > 0:
            for every_column in columns:
                comparison_log = pd.concat([comparison_log, self.discrete_compare(every_column)])
        return comparison_log

    def cross_effect(self, columns=None):
        if columns is not None:
            columns = list(set(self.columns) and set(columns))
        else:
            columns = self.columns
        pattern_min = self.pattern.loc[:, columns].min(axis=0)
        pattern_max = self.pattern.loc[:, columns].max(axis=0)
        source_min = self.source.loc[:, columns].min(axis=0)
        source_max = self.source.loc[:, columns].max(axis=0)
        total_min = pd.DataFrame(map(lambda some_x, some_y: some_y if (some_y < some_x) else some_x, pattern_min.values[0], source_min.values[0])).T
        total_min.columns = columns
        total_max = pd.DataFrame(map(lambda some_x, some_y: some_y if (some_y > some_x) else some_x, pattern_max.values[0], source_max.values[0])).T
        total_max.columns = columns
        total_scatter = total_max - total_min
        total_min = total_min.loc[:, total_scatter.loc[0, :] > 0]
#        total_max = total_max.loc[:, total_min.columns]
        total_scatter = total_scatter.loc[:, total_min.columns]
        pattern_cross_effect = ((self.pattern.loc[:, total_min.columns] - total_min) / total_scatter).max(axis=1)
        pattern_cross_effect.columns = ['cross_effect']
        source_cross_effect = ((self.pattern.loc[:, total_min.columns] - total_min) / total_scatter).max(axis=1)
        source_cross_effect.columns = ['cross_effect']
        return pattern_cross_effect, source_cross_effect

    def stratified_compare(self, columns=None):
        pattern_cross_effect, source_cross_effect = self.cross_effect(columns)
        cross_strawberry = strawberry(pattern_cross_effect, source_cross_effect)
        return cross_strawberry.compare(['cross_effect'])[0]#.is_different

    def full_compare(self, columnar_aggregation=max, output_aggregation=None, columns=None, stratified_columns=None, compare_metric='is_similar'): # 'is_similar' equals 'difference_score'
        if columns is not None:
            continues_columns = list(set(self.continues_columns) and set(columns))
            discrete_columns = list(set(self.discrete_columns) and set(columns))
            columns = list(set(self.columns) and set(columns))
        else:
            continues_columns = self.continues_columns
            discrete_columns = self.discrete_columns
            columns = self.columns
        columnar_result = pd.concat([self.compare(continues_columns), self.discrete_compare(discrete_columns)])
        columnar_result = columnar_aggregation(columnar_result.loc[:, [compare_metric]].values[0])
        if output_aggregation is None:
            return columnar_result
        else:
            if stratified_columns is None:
                stratified_columns = columns
            stratified_result = self.stratified_compare(stratified_columns).loc[:, [compare_metric]].values[0, 0]
            return output_aggregation([columnar_result, stratified_result])
#            return pd.concat([self.compare(columns).loc[:, ['difference_score']], self.stratified_compare(columns).loc[:, ['difference_score']]]).max(axis=0).values[0, 0]

    def squeeze(self, n_samples=None, base_rate=0.1, gulp_rate=0.1, vain_steps=7, short_rate=0.9, filter_threshold = None, n_gulps = 2, random_state=0):
        """
        :param n_samples: Required number of samples. If is None - algorithm works until encreasing of similarity between pattern and selected dataframes is possible.
        :param base_rate: Part of source dataframe (used to select samples from) which will be randomly selected as the initial solution.
        :param gulp_rate: Randomly selected part of source dataframe which (being appended to selection) will be used as attempt to encrease the similarity between pattern and selected dataframes.
        :param vain_steps: Limits the number of consequtive iterations without encreasing of similarity between pattern and selected dataframes.
        :param short_rate: Gives algorithm a chance to continue selection after vain_steps limit is exhausted. short_rate is a part - how much the gulp_rate will be decreased before continuetion.
        :param filter_threshold: Part - how much the similarity score must encrease to be considered as similarity encreasing.
        :param n_gulps: Global limit of iterations (attempts to encrease the similarity between pattern and selected dataframes).
        :param random_state: Random state will be used during samples selection.
        :return: Selected dataframe.
        """
        if n_samples is None:
            n_samples = self.source.shape[0]
        rest_set, base_set = train_test_split(self.source, test_size=base_rate, random_state=random_state)
        chosen_set = base_set

        current_score = strawberry(self.pattern, base_set).full_compare(median, max)
#        current_score = strawberry(self.pattern, base_set).compare().loc[:, 'is_similar'].median(axis=0, skipna=True)
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
                score_with_gulp = strawberry(self.pattern, chosen_with_gulp).full_compare(median, max)
#                score_with_gulp = strawberry(self.pattern, chosen_with_gulp).compare().loc[:, 'is_similar'].median(axis=0, skipna=True)
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
