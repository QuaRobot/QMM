import argparse
import csv
from transformers import AutoTokenizer
import os
import configs.default as cd


class Data_Processor_Wordnet:

    def __init__(self, args):

        self.realtokenizer = AutoTokenizer.from_pretrained(args.DATA.plm)
        # self.imagtokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        self.data_dir = args.DATA.data_dir
        self.sep_puncs = args.DATA.sep_puncs
        self.max_left_len = args.DATA.max_left_len  # the previously set max length of the left input
        self.max_right_len = 10
        self.use_pos = args.DATA.use_pos
        self.use_eg_sent = args.DATA.use_eg_sent

    def __str__(self):
        pattern = '''Data Configs: 
        data_dir: {} 
        sep_puncs: {} 
        max_left_len: {} 
        max_right_len: {} 
        use_pos: {}
        use_eg_sent: {}'''
        return pattern.format(self.data_dir, self.sep_puncs, self.max_left_len,
                              self.max_right_len, self.use_pos, self.use_eg_sent)

    def _get_examples(self, file_dir):
        """

        :param file_dir: a vua csv file
        :return:
        """
        max = 0
        examples = []
        pos = {'VERB':0,
               'ADV':1,
               'PART':2,
               'ADJ':3,
               'PRON':4,
               'ADP':5,
               'DET':6,
               'CCONJ':7,
               'NOUN':8,
               'PROPN':9,
               'NUM':10,
               'INTJ':11,
               'SYM':12,
               'PUNCT':13,
               'X':14}


        with open(file_dir, 'r', encoding='gbk') as f:
            lines = csv.reader(f)
            next(lines)  # skip the headline
            num1 = 0
            for i, line in enumerate(lines):
                sentence, label, target_position, target_word, pos_tag, gloss = line
                # print(sentence, label, target_position, target_word, pos_tag, gloss)
                # input()
                label = int(label)
                target_position = int(target_position)
                if target_position >= max:
                    max = target_position

                #pos_tag = self.realtokenizer(pos_tag.lower(), padding='max_length', max_length=5, add_special_tokens=True, truncation=True)

                pos_tag = pos['VERB']
                """
                for the real part
                """
                # convert sentence to ids: <s> sentence </s>
                tmp = self.realtokenizer(sentence, padding='max_length', max_length=self.max_left_len, add_special_tokens=True, truncation=True)
                ids_r = tmp['input_ids']
                att_r = tmp['attention_mask']
                # target word may be cut into word pieces by tokenizer, find the range of pieces
                tar_start, tar_end = target_align(target_position, sentence, tokenizer=self.realtokenizer)

                if tar_start+(tar_end-tar_start)>145:
                    num1+=1
                    print(num1,tar_start,tar_end)
                    continue
                tar_r = [0] * self.max_left_len
                tar_r[tar_start: tar_end] = [1] * (tar_end - tar_start)

                position = 2
                for i in range(tar_start-1,-1,-1):
                    tar_r[i] = position
                    position+=1
                position = 2
                for i in range(tar_end, self.max_left_len):
                    tar_r[i] = position
                    position+=1

                gloss_list = []
                for j in gloss.split("<FF>")[0:-1]:
                    if 'metaphor' not in j:
                        gloss_list.append(j)
                if len(gloss_list)==0:
                    gloss_list.append(target_word)
                gloss_len = len(gloss_list)
                gloss_att = [1]*gloss_len
                while gloss_len < 10:
                    gloss_list.append('')
                    gloss_len += 1
                    gloss_att.append(0)
                gloss_att = gloss_att[0:10]
                tmp = self.realtokenizer(gloss_list[0:10], padding='max_length', max_length=self.max_right_len, add_special_tokens=True, truncation=True)
                ids_rg = tmp['input_ids']
                att_rg = tmp['attention_mask']


                example = [ids_r, att_r, tar_r, ids_rg, att_rg, pos_tag, gloss_att, label]
                examples.append(example)

                if (i + 1) % 10000 == 0:
                    print(f'{i + 1} sentences have been processed.')


            print(f'{file_dir} finished, the max target is '+str(max))


        return examples


class VUA_All_Processor(Data_Processor_Wordnet):
    def __init__(self, args):
        super(VUA_All_Processor, self).__init__(args)

    def get_train_data(self):
        train_data_path = os.path.join(self.data_dir, 'VUA_All/g_train.csv')
        train_data = self._get_examples(train_data_path)
        return train_data

    def get_val_data(self):
        val_data_path = os.path.join(self.data_dir, 'VUA_All/g_val.csv')
        val_data = self._get_examples(val_data_path)
        return val_data

    def get_test_data(self):
        test_data_path = os.path.join(self.data_dir, 'VUA_All/g_test.csv')
        test_data = self._get_examples(test_data_path)
        return test_data

    def get_acad(self):
        data_path = os.path.join(self.data_dir, 'VUA_All/genre/g_acad.csv')
        data = self._get_examples(data_path)
        return data

    def get_conv(self):
        data_path = os.path.join(self.data_dir, 'VUA_All/genre/g_conv.csv')
        data = self._get_examples(data_path)
        return data

    def get_fict(self):
        data_path = os.path.join(self.data_dir, 'VUA_All/genre/g_fict.csv')
        data = self._get_examples(data_path)
        return data

    def get_news(self):
        data_path = os.path.join(self.data_dir, 'VUA_All/genre/g_news.csv')
        data = self._get_examples(data_path)
        return data

    def get_adj(self):
        data_path = os.path.join(self.data_dir, 'VUA_All/pos/g_adj.csv')
        data = self._get_examples(data_path)
        return data

    def get_adv(self):
        data_path = os.path.join(self.data_dir, 'VUA_All/pos/g_adv.csv')
        data = self._get_examples(data_path)
        return data

    def get_noun(self):
        data_path = os.path.join(self.data_dir, 'VUA_All/pos/g_noun.csv')
        data = self._get_examples(data_path)
        return data

    def get_verb(self):
        data_path = os.path.join(self.data_dir, 'VUA_All/pos/g_verb.csv')
        data = self._get_examples(data_path)
        return data


class Verb_Processor(Data_Processor_Wordnet):
    def __init__(self, args):
        super(Verb_Processor, self).__init__(args)

    def get_train_data(self):
        data_path = os.path.join(self.data_dir, 'VUA_Verb/g_train.csv')
        data = self._get_examples(data_path)
        return data

    def get_val_data(self):
        data_path = os.path.join(self.data_dir, 'VUA_Verb/g_val.csv')
        data = self._get_examples(data_path)
        return data

    def get_test_data(self):
        data_path = os.path.join(self.data_dir, 'VUA_Verb/g_test.csv')
        data = self._get_examples(data_path)
        return data

    def get_trofi(self):
        data_path = os.path.join(self.data_dir, 'TroFi/TroFi.csv')
        data = self._get_examples(data_path)
        return data

    def get_mohx(self):
        data_path = os.path.join(self.data_dir, 'MOH-X/g_MOH-X.csv')
        data = self._get_examples(data_path)
        return data




def target_align(target_position, sentence, tokenizer):
    """
    A target may be cut into word pieces by Tokenizer, this func tries to find the start and end idx of the target word
    after tokenization.
    NOTICE: we return a half-closed range. eg. [0, 6) for start_idx=0 and end_idx=6

    :param target_position: (int) the position of the target word in the original sentence
    :param sentence: (string) original sentence
    :param tokenizer: an instance of Transformers Tokenizer
    :return: (tuple of int) the start and end idx of the target word in the tokenized form
    """
    start_idx = 1  # if take the [CLS] into consideration
    end_idx = 0
    for j, word in enumerate(sentence.split()):
        if not j == 0:
            word = ' ' + word
        word_tokens = tokenizer.tokenize(word)
        if not j == target_position:  # if current word is not target word, just count its length
            start_idx += len(word_tokens)
        else:  # else, calculate the end position
            end_idx = start_idx + len(word_tokens)
            break  # once find, stop looping.
    return start_idx, end_idx


def parse_option():
    parser = argparse.ArgumentParser(description='Train on MOH-X dataset, do cross validation')
    parser.add_argument('--cfg', type=str, default='../configs/mohx.yaml', metavar="FILE",
                        help='path to config file')
    parser.add_argument('--gpu', default='0', type=str, help='gpu device ids')
    parser.add_argument('--eval', action='store_true', help="evaluation only")
    parser.add_argument('--log', default='mohx', type=str)
    args, unparsed = parser.parse_known_args()
    config = cd.get_config(args)

    return config

if __name__ == '__main__':
    args = parse_option()
    Verb_Processor(args).get_mohx()
