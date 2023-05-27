import numpy as np

def generic_calc_q_error(card_preds, true_cards, if_print=False):
    true_cards += 1e-8
    card_preds = np.clip(card_preds, a_min=1e-8, a_max=None)
    q_error_1 = card_preds / true_cards
    q_error_2 = true_cards / card_preds
    q_error = np.array([q_error_1, q_error_2], dtype=np.float64)
    q_error = np.max(q_error, axis=0)
    q_error = np.reshape(q_error, [-1])
    if if_print == True:
        diff = card_preds - true_cards
        idxes = np.where(diff < 0)[0]
        print('idxes.shape =', idxes.shape, 'true_cards.shape =', true_cards.shape)
    return q_error

def generic_lower_q_error(card_preds, true_cards):
    true_cards += 1e-8
    card_preds = np.clip(card_preds, a_min=1e-8, a_max=None)
    q_error = true_cards / card_preds
    return q_error

def generic_upper_q_error(card_preds, true_cards):
    true_cards += 1e-8
    card_preds = np.clip(card_preds, a_min=1e-8, a_max=None)
    q_error = card_preds / true_cards
    return q_error

def query_driven_calc_card_preds(preds, var_preds, join_cards, y_type, card_log_scale):
    card_preds = preds

    if_card_log = False
    if y_type > 0:
        card_preds = preds * join_cards
    else:
        if_card_log = bool(card_log_scale)

    card_preds_2 = None
    if if_card_log == True:
        card_preds = np.exp(preds)
        card_preds_2 = np.exp(preds + var_preds / 2)
    return card_preds, card_preds_2

def query_driven_calc_q_error(preds, var_preds, true_cards, join_cards, y_type, card_log_scale):
    # min_val = np.min(preds)
    # assert min_val > 0
    card_preds, card_preds_2 = query_driven_calc_card_preds(preds, var_preds, join_cards, y_type, card_log_scale)

    q_error_2 = None
    if card_preds_2 is not None:
        q_error_2 = generic_calc_q_error(card_preds_2, true_cards)

    q_error_1 = generic_calc_q_error(card_preds, true_cards)
    return q_error_1, q_error_2
