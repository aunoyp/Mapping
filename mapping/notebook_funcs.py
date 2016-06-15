
import pandas as pd

def print_anova(res, title):
    p_rew = res.ix['C(rew)', 'PR(>F)']
    p_monkey = res.ix['C(monkey)', 'PR(>F)']
    p_int = res.ix['C(rew):C(monkey)', 'PR(>F)']

    print('*** 2-way anova on %s' %title)
    print('reward %1.4f, monkey %1.4f, interaction %1.4f' %(p_rew, p_monkey, p_int))

def print_wilcoxons(ba, p, p_monkey, p_dir, p_set, title):

    p_all = pd.concat([
        p.iloc[0],
        p_monkey.iloc[0],
        p_dir.iloc[0],
        p_set.iloc[0]
    ])

    print('*** Pair-wise Wilcoxons')
    for k,v in p_all.iteritems():
        print('\t%s:    %1.4f' %(k,v))