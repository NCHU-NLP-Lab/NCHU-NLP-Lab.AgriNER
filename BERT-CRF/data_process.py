import os
import json
import logging
import numpy as np
import config
import random, string
import queue


class Processor:
    def __init__(self, config):
        self.data_dir = config.data_dir
        self.config = config

    def process(self):
        """
        process train and test data
        """
        for file_name in self.config.files:
            self.preprocess(file_name)

    def preprocess(self, mode):
        """
        params:
            words：将json文件每一行中的文本分离出来，存储为words列表
            labels：标记文本对应的标签，存储为labels
        examples:
            words示例：['生', '生', '不', '息', 'C', 'S', 'O', 'L']
            labels示例：['O', 'O', 'O', 'O', 'B-game', 'I-game', 'I-game', 'I-game']
        """
        input_dir = self.data_dir + str(mode) + '.json'
        output_dir = self.data_dir + str(mode) + '.npz'
        if os.path.exists(output_dir) is True:
            return
        word_list = []
        label_list = []
        with open(input_dir, 'r', encoding='utf-8') as f:
            # 先读取到内存中，然后逐行处理
            for line in f.readlines():
                # loads()：用于处理内存中的json对象，strip去除可能存在的空格
                json_line = json.loads(line.strip())

                text = json_line['text']
                words = list(text)
                # 如果没有label，则返回None
                label_entities = json_line.get('label', None)
                labels = ['O'] * len(words)

                if label_entities is not None:
                    for key, value in label_entities.items():
                        for sub_name, sub_index in value.items():
                            for start_index, end_index in sub_index:
                                assert ''.join(words[start_index:end_index + 1]) == sub_name
                                if start_index == end_index:
                                    labels[start_index] = 'S-' + key
                                else:
                                    labels[start_index] = 'B-' + key
                                    labels[start_index + 1:end_index + 1] = ['I-' + key] * (len(sub_name) - 1)
                word_list.append(words)
                label_list.append(labels)
                # 保存成二进制文件
            np.savez_compressed(output_dir, words=word_list, labels=label_list)
            logging.info("--------{} data process DONE!--------".format(mode))

    def preprocess2(self,mode):
            """
            將輸入的文本轉成 一行一個字元對一個label
            """
            input_dir = self.data_dir + str(mode) + '.json'
            output_dir = self.data_dir + str(mode) + '.txt'
            # if os.path.exists(output_dir) is True:
            #     return
            with open(input_dir, 'r', encoding='utf-8') as f:
                with open(output_dir, 'w') as t:
                    # 先读取到内存中，然后逐行处理
                    for line in f.readlines():
                        # loads()：用于处理内存中的json对象，strip去除可能存在的空格
                        json_line = json.loads(line.strip())

                        text = json_line['text']
                        words = list(text)
                        # 如果没有label，则返回None
                        label_entities = json_line.get('label', None)
                        labels = ['O'] * len(words)

                        if label_entities is not None:
                            for key, value in label_entities.items():
                                for sub_name, sub_index in value.items():
                                    for start_index, end_index in sub_index:
                                        assert ''.join(words[start_index:end_index + 1]) == sub_name
                                        if start_index == end_index:
                                            labels[start_index] = 'S-' + key
                                        else:
                                            labels[start_index] = 'B-' + key
                                            labels[start_index + 1:end_index + 1] = ['I-' + key] * (len(sub_name) - 1)

                        j_len = len(words)
                        s = ""
                        for j in range(j_len):
                            # print(data["words"][i][j])
                            # print(data["labels"][i][j])
                            s += words[j] + ' ' + labels[j] + '\n'
                        s += '\n'
                        # print(s)
                        t.write(s)
                logging.info("--------{} data process DONE!--------".format(mode))

    def preprocess3(self, mode, random_percentage):
        """
        將輸入的文本加入亂碼
        """
        input_dir = self.data_dir + str(mode) + '.json'
        output_dir = self.data_dir + str(mode)+ '_random_percentage_'+ str(random_percentage)  +'.npz'
        # if os.path.exists(output_dir) is True:
        #     return
        word_list = []
        label_list = []
        pass_count = 0
        with open(input_dir, 'r', encoding='utf-8') as f:
            # 先读取到内存中，然后逐行处理
            for line in f.readlines():
                # loads()：用于处理内存中的json对象，strip去除可能存在的空格
                json_line = json.loads(line.strip())

                text = json_line['text']
                words = list(text)
                # 如果没有label，则返回None
                label_entities = json_line.get('label', None)
                labels = ['O'] * len(words)

                # word總長:53896
                if label_entities is not None:
                    for key, value in label_entities.items():
                        for sub_name, sub_index in value.items():
                            for start_index, end_index in sub_index:
                                assert ''.join(words[start_index:end_index + 1]) == sub_name
                                if start_index == end_index:
                                    labels[start_index] = 'S-' + key
                                else:
                                    labels[start_index] = 'B-' + key
                                    labels[start_index + 1:end_index + 1] = ['I-' + key] * (len(sub_name) - 1)

                w_len = len(words)
                indexes_list = []
                for idx in range(w_len):
                    if labels[idx]=='O':
                        indexes_list.append(idx)

                random.shuffle(indexes_list)

                q = queue.Queue()
                for item in indexes_list:
                    q.put(item)

                # 根據random_percentage計算要擾亂的文本比率
                random_count =round(w_len * random_percentage/100)

                # print(w_len)

                for i in range(random_count):
                    if q.empty():
                        pass_count+=1
                        break
                    random_idx = int(q.get())
                    words[random_idx] = random.choice(string.ascii_letters)

                word_list.append(words)
                label_list.append(labels)
                # 保存成二进制文件
            np.savez_compressed(output_dir, words=word_list, labels=label_list)
            print(pass_count)
            logging.info("--------{} data process DONE!--------".format(mode))

    def preprocess4(self, mode):
        """
        params:
            words：将json文件每一行中的文本分离出来，存储为words列表
            labels：标记文本对应的标签，存储为labels
        examples:
            words示例：['生', '生', '不', '息', 'C', 'S', 'O', 'L']
            labels示例：['O', 'O', 'O', 'O', 'B-game', 'I-game', 'I-game', 'I-game']
        """
        input_dir = self.data_dir + str(mode) + '.json'
        # output_dir = self.data_dir + str(mode)+ '_add###_' +'.npz'
        output_dir = self.data_dir + str(mode)+ '_add_SEP_' +'.npz'
        # if os.path.exists(output_dir) is True:
        #     return
        word_list = []
        label_list = []
        pass_count = 0
        cou = 0
        with open(input_dir, 'r', encoding='utf-8') as f:
            # 先读取到内存中，然后逐行处理
            for line in f.readlines():
                # loads()：用于处理内存中的json对象，strip去除可能存在的空格
                json_line = json.loads(line.strip())

                text = json_line['text']
                words = list(text)
                # 如果没有label，则返回None
                label_entities = json_line.get('label', None)
                labels = ['O'] * len(words)


                # word總長:53896
                if label_entities is not None:
                    for key, value in label_entities.items():
                        for sub_name, sub_index in value.items():
                            # words.append('#')
                            for start_index, end_index in sub_index:
                                assert ''.join(words[start_index:end_index + 1]) == sub_name
                                if start_index == end_index:
                                    labels[start_index] = 'S-' + key
                                else:
                                    labels[start_index] = 'B-' + key
                                    labels[start_index + 1:end_index + 1] = ['I-' + key] * (len(sub_name) - 1)

                if label_entities is not None:
                    for key, value in label_entities.items():
                        for sub_name, sub_index in value.items():
                            # words.append('#')
                            words.append('[SEP]')
                            for s in sub_name:
                                words.append(s)

                            for start_index, end_index in sub_index:
                                assert ''.join(words[start_index:end_index + 1]) == sub_name
                                if start_index == end_index:
                                    # labels[start_index] = 'S-' + key
                                    x = []
                                    x.append('O')
                                    x.append('S-' + key)
                                    labels = labels + x
                                else:
                                    # labels[start_index] = 'B-' + key
                                    # labels[start_index + 1:end_index + 1] = ['I-' + key] * (len(sub_name) - 1)
                                    x = []
                                    x.append('O')
                                    x.append('B-'+key)
                                    # x.append(key)
                                    # x = 'O' + 'B-' + key
                                    # labels.append()
                                    x = list(x) + ['I-' + key] * (len(sub_name) - 1)
                                    # x.append(list(x) + ['I-' + key] * (len(sub_name) - 1))
                                    labels = labels + x
                                    # labels.append(x)
                                    # labels = labels + x
                                break

                if len(words) > 511:
                    cou +=1
                    words = words[:510]
                    labels = labels[:510]

                word_list.append(words)
                label_list.append(labels)
                # 保存成二进制文件
            np.savez_compressed(output_dir, words=word_list, labels=label_list)
            print(pass_count)
            print(cou)
            logging.info("--------{} data process DONE!--------".format(mode))


if __name__ == "__main__":
    p = Processor(config)
    # p.preprocess2("train")
    # p.preprocess2("test")
    # p.preprocess3("train_v8", 15)
    p.preprocess3("demo", 0)
    # p.preprocess4("train")