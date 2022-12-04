import os
import config
import logging


def get_entities(seq):
    """
    Gets entities from sequence.

    Args:
        seq (list): sequence of labels.

    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).

    Example:
        seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        get_entities(seq)
        [('PER', 0, 1), ('LOC', 3, 3)]
    """
    # for nested list
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ['O']]
    prev_tag = 'O'
    prev_type = ''
    begin_offset = 0
    chunks = []
    for i, chunk in enumerate(seq + ['O']):
        tag = chunk[0]
        type_ = chunk.split('-')[-1]

        if end_of_chunk(prev_tag, tag, prev_type, type_):
            chunks.append((prev_type, begin_offset, i - 1))
        if start_of_chunk(prev_tag, tag, prev_type, type_):
            begin_offset = i
        prev_tag = tag
        prev_type = type_

    return chunks


def end_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk ended between the previous and current word.

    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.

    Returns:
        chunk_end: boolean.
    """
    chunk_end = False

    if prev_tag == 'S':
        chunk_end = True
    # pred_label中可能出现这种情形
    if prev_tag == 'B' and tag == 'B':
        chunk_end = True
    if prev_tag == 'B' and tag == 'S':
        chunk_end = True
    if prev_tag == 'B' and tag == 'O':
        chunk_end = True
    if prev_tag == 'I' and tag == 'B':
        chunk_end = True
    if prev_tag == 'I' and tag == 'S':
        chunk_end = True
    if prev_tag == 'I' and tag == 'O':
        chunk_end = True

    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        chunk_end = True

    return chunk_end


def start_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk started between the previous and current word.

    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.

    Returns:
        chunk_start: boolean.
    """
    chunk_start = False

    if tag == 'B':
        chunk_start = True
    if tag == 'S':
        chunk_start = True

    if prev_tag == 'S' and tag == 'I':
        chunk_start = True
    if prev_tag == 'O' and tag == 'I':
        chunk_start = True

    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True

    return chunk_start


def f1_score(y_true, y_pred, mode='dev'):
    """Compute the F1 score.

    The F1 score can be interpreted as a weighted average of the precision and
    recall, where an F1 score reaches its best value at 1 and worst score at 0.
    The relative contribution of precision and recall to the F1 score are
    equal. The formula for the F1 score is::

        F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.

    Returns:
        score : float.

    Example:
        y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        f1_score(y_true, y_pred)
        0.50
    """
    true_entities = set(get_entities(y_true))
    pred_entities = set(get_entities(y_pred))
    nb_correct = len(true_entities & pred_entities)
    nb_pred = len(pred_entities)
    nb_true = len(true_entities)

    p = nb_correct / nb_pred if nb_pred > 0 else 0
    r = nb_correct / nb_true if nb_true > 0 else 0
    score = 2 * p * r / (p + r) if p + r > 0 else 0
    if mode == 'dev':
        return score, p, r #f1
    else:
        f_score = {}
        for label in config.labels:
            true_entities_label = set()
            pred_entities_label = set()
            for t in true_entities:
                if t[0] == label:
                    true_entities_label.add(t)
            for pe in pred_entities:
                if pe[0] == label:
                    pred_entities_label.add(pe)
            nb_correct_label = len(true_entities_label & pred_entities_label)
            nb_pred_label = len(pred_entities_label)
            nb_true_label = len(true_entities_label)

            p_label = nb_correct_label / nb_pred_label if nb_pred_label > 0 else 0
            r_label = nb_correct_label / nb_true_label if nb_true_label > 0 else 0
            score_label = 2 * p_label * r_label / (p_label + r_label) if p_label + r_label > 0 else 0
            f_score[label] = (score_label, p_label, r_label)
        return f_score, score, p, r


def bad_case(y_true, y_pred, data):
    if not os.path.exists(config.case_dir):
        os.system(r"touch {}".format(config.case_dir))  # 调用系统命令行来创建文件
    output = open(config.case_dir, 'w')
    for idx, (t, p) in enumerate(zip(y_true, y_pred)):
        if t == p:
            continue
        else:
            output.write("bad case " + str(idx) + ": \n")
            output.write("sentence: " + str(data[idx]) + "\n")
            output.write("golden label: " + str(t) + "\n")
            output.write("model pred: " + str(p) + "\n")
    logging.info("--------Bad Cases reserved !--------")

def check_byRules(y_true, y_pred, data):
    if not os.path.exists(config.case_dir):
        os.system(r"touch {}".format(config.case_dir))  # 调用系统命令行来创建文件
    output = open(config.case_dir, 'w')
    # '劑'
    for idx, (t, p) in enumerate(zip(y_true, y_pred)):
        sent = data[idx]
        y_pred[idx] = fix_pred_tags(sent,p)
        # if t == p:
        #     continue
        # else:
        #     output.write("bad case " + str(idx) + ": \n")
        #     output.write("sentence: " + str(data[idx]) + "\n")
        #     output.write("golden label: " + str(t) + "\n")
        #     output.write("model pred: " + str(p) + "\n")
    logging.info("--------Check_byRules Done !--------")

def fix_pred_tags(sent, pred_tags):
    fix_list = []
    prev_end_idx = 0
    for idx, s in enumerate(sent):
        # print(s)
        if s == '劑' or s =='液':
            # print(sent[idx-1])
            # find_idx = idx
            end_idx = idx
            start_idx = 0
            find_p = False
            for i in range(idx - 1, prev_end_idx, -1):
                # print(i)
                # print(sent[i])
                if  (sent[i] == '%' or sent[i] == '％') and not find_p:
                    start_idx = i
                    find_p = True
                    continue

                if find_p and (sent[i].isnumeric() or sent[i] == '.'):
                    start_idx = i

                if find_p and sent[i].isalpha():
                    # start_idx = i + 1
                    break

                if find_p and sent[i] == '_':
                    # start_idx = i + 1
                    break

                if find_p and sent[i] == '(':
                    # start_idx = i + 1
                    break

                if find_p and sent[i] == ')':
                    # start_idx = i + 1
                    break

                if sent[i] == '。':
                    break
            if find_p:
                find_p = False
                prev_end_idx = end_idx
                fix_list.append((start_idx, end_idx))
                # print('start_idx = ', start_idx, 'end_idx = ', end_idx)

    # 修正B-Chemicals
    for idx, v in enumerate(fix_list):
        start_idx = v[0]
        end_idx = v[1]
        pred_tags[start_idx] = 'B-Chemicals'

        for p_idx in range(start_idx + 1, end_idx + 1):
            pred_tags[p_idx] = 'I-Chemicals'

    # 修正兩個Chemicals 當中錯誤的標注
    max_len = len(fix_list)
    for idx, v in enumerate(fix_list):
        if idx == max_len - 1:
            break
        start_idx = v[1]
        end_idx = fix_list[idx + 1][0]
        # print(start_idx, end_idx)

        for i in range(start_idx + 1, end_idx):
            if sent[i].isalpha():
                pred_tags[i] = 'O'

    # 修正B-Chemicals 和I-Chemicals，中間錯誤的標注
    find_b_Chemicals = False
    find_i_Chemicals = False
    b_Chemicals_list = []
    b_start = 0
    b_end = 0
    for idx, p in enumerate(pred_tags):

        if p == 'B-Chemicals' and not find_b_Chemicals:
            find_b_Chemicals = True
            b_start = idx
            continue

        if p == 'I-Chemicals' and find_b_Chemicals:
            find_i_Chemicals = True
            b_end = idx
            continue

        if p == 'I-Chemicals' and idx > 0 and pred_tags[idx - 1] == 'O' and find_b_Chemicals:
            b_end = idx
            continue

        if p == 'O' and idx < len(pred_tags) - 1 and pred_tags[idx + 1] == 'I-Chemicals' and find_b_Chemicals:
            b_end = idx
            continue

        if p == 'O' and find_b_Chemicals and find_i_Chemicals:
            find_b_Chemicals = False
            find_i_Chemicals = False
            if b_start != 0 and b_end != 0:
                b_Chemicals_list.append((b_start, b_end))
            continue

    # print(b_Chemicals_list)

    for idx, v in enumerate(b_Chemicals_list):
        start_idx = v[0]
        end_idx = v[1]
        # print(start_idx, end_idx)
        for i in range(start_idx + 1, end_idx + 1):
            pred_tags[i] = 'I-Chemicals'

    # 修正劑或液後面的異常標注(通常都是O，而非I-Chemicals)
    for idx, s in enumerate(sent):
        # print(s)
        if s == '劑' or s == '液':
            if idx < len(pred_tags) - 1 and pred_tags[idx + 1].split("-")[-1] == "Chemicals" and pred_tags[idx + 1] != "O":
                # print(pred_tags[idx + 1].split("-")[-1])
                pred_tags[idx + 1] = 'O'

    # 修正獨立出來的I-Chemicals，把它修正為'O'
    for idx, p in enumerate(pred_tags):
        # if p.split("-")[0] =='B' and not find_begin:
        #   find_begin = True
        #   continue

        if p.split("-")[0] =='I' and ((idx > 0 and pred_tags[idx-1] == 'O') or idx == 0):
            pred_tags[idx] = 'O'
            continue

    # 修正獨立出來的B-XXX，把它修正為'O'
    for idx, p in enumerate(pred_tags):

        if p.split("-")[0] == 'B' and ((idx < len(pred_tags) - 1 and (
                pred_tags[idx + 1] == 'O' or pred_tags[idx + 1].split("-")[0] == "B")) or idx == len(pred_tags) - 1):
            pred_tags[idx] = 'O'
            continue
    return pred_tags

if __name__ == "__main__":
    y_t = [['O', 'O', 'O', 'B-address', 'I-address', 'I-address', 'O'], ['B-name', 'I-name', 'O']]
    y_p = [['O', 'O', 'B-address', 'I-address', 'I-address', 'I-address', 'O'], ['B-name', 'I-name', 'O']]
    sent = [['十', '一', '月', '中', '山', '路', '电'], ['周', '静', '说']]
    bad_case(y_t, y_p, sent)
