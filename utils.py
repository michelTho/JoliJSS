import numpy as np

def smooth_curve(tab, smooth_rate=0.9):
    return_tab = [tab[0]]
    for value in tab:
        return_tab.append(value * (1 - smooth_rate) + \
                            return_tab[-1] * smooth_rate)
    return return_tab
