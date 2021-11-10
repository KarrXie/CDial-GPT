import os
import random
import pickle
import json
import copy
import argparse
from entity_recognize import kd_ana

parser = argparse.ArgumentParser()
parser.add_argument("--origin_data_path", default='./data', type=str)
parser.add_argument("--output_data_path", default='./data/CDial_data', type=str)
parser.add_argument("--max_text_lenght", default=1020, type=int)
parser.add_argument("--epoch", default=10, type=int)
args = parser.parse_args()

feature2chinese = {"Disease": "疾病", "Symptom": "症状", "Attribute": "属性", "Test": "检查", "Medicine": "药物"}


# 用于将同一句话中的不同特征进行整合
def cat_feature_info(dt):
    together_list = []
    for feature in feature2chinese.keys():
        together_list.extend(dt[feature])
    return together_list

def dev_type2train_type(input_list):
    output_list = []
    for text in input_list:
        role, sent = text[:2], text[3:]
        entity = list(kd_ana.convert_sen_to_entity_set(sent))
        role = "Patient" if role == "患者" else "Doctor"
        output_list.append({"Attribute": entity, 'Disease': [], 'Medicine': [], 'Sentence': sent, 'Symptom': [], 'Test': [], 'id': role})
    return output_list


# 将文本长度大于1020的对话，进行截断
def cut_dialog_without_entity(inp_list, inp_entity, max_lenght=args.max_text_lenght - 3):
    lens, cnt = 0, 0
    need_sub_char_num = 7  # speaker1 和 speaker2 都是8个字符
    for inp in inp_list:
        lens += len(inp) - need_sub_char_num
    if lens <= max_lenght:
        return inp_list, inp_entity
    while cnt < (lens - max_lenght):
        sub_cnt = len(inp_list[0]) - need_sub_char_num - (lens - max_lenght - cnt)
        if sub_cnt < 20:
            sen = inp_list.pop(0)
            _ = inp_entity.pop(0)
            cnt += len(sen) - need_sub_char_num
        else:
            inp_list[0] = inp_list[0][:8] + inp_list[0][-sub_cnt:]
            break
    return inp_list, inp_entity


with open(os.path.join(args.origin_data_path, "./train/train.pk"), "rb") as f:
    train_data = pickle.load(f)
with open(os.path.join(args.origin_data_path, "./evalution/dev.pk"), "rb") as f:
    dev_data = pickle.load(f)
with open(os.path.join(args.origin_data_path, "./test/test.pk"), "rb") as f:
    test_data = pickle.load(f)
with open(os.path.join(args.origin_data_path, "./test/test_gt.txt"), "r", encoding='utf8') as f:
    test_data_reference = [line.strip() for line in f.readlines()]
with open(os.path.join(args.origin_data_path, "./test/test_sample.pk"), "rb") as f:
    test_sample = pickle.load(f)
with open(os.path.join(args.origin_data_path, "./test/test_sample_reference.pk"), "rb") as f:
    test_sample_reference = pickle.load(f)
print("train:{},dev:{} test_sample:{}, test_reference:{}".format(len(train_data), len(dev_data), len(test_sample), len(test_sample_reference)))
dev_test_data = dev_data + test_data + test_sample

dev_test_data_end_with_doctor = copy.deepcopy(dev_test_data)
dev_test_data_for_generation = []
for dtd in dev_test_data_end_with_doctor:
    try:
        while dtd['history'][-1][:2] == "患者":
            dtd["history"].pop()
        dev_test_data_for_generation.append(dtd)
    except:
        continue
del dev_test_data_end_with_doctor

# 将用于文本生成的测试集数据和验证集数据转换成训练集的格式，从而统一处理
dev_test_data_for_generation2train_type = []
for dt in dev_test_data_for_generation:
    dev_test_data_for_generation2train_type.append(dev_type2train_type(dt["history"]))

# 将验证集和测试集中的数据合并到训练集中，共同用于训练文本生成任务
train_data += dev_test_data_for_generation2train_type

# 生成 文本生成任务训练数据
train_num = len(train_data)
_train_data = []
train_data_dict = {"input": [], "entity": []}
random.shuffle(train_data)
for t in range(args.epoch):
    random.shuffle(train_data)
    for r, dt in enumerate(train_data):
        _train_data.append(dt)
        if len(_train_data) == 1000 or r == train_num - 1:
            for i in range(1, 30):  # 最多支持30轮对话
                break_forword = False
                for _dt in _train_data:
                    temp_dialog, temp_entity = [], []
                    begin_role = 'doctor' if _dt[0]["id"] == "Doctor" else 'patient'
                    patient_in_flag = False
                    last_id = ""
                    cnt = 0
                    for seg in _dt:
                        feature_info = cat_feature_info(seg)
                        if seg["id"] == "Doctor":
                            cnt += 1
                            prefix_id = 'speaker2'
                        else:
                            patient_in_flag = True
                            prefix_id = 'speaker1'
                        dialo_sentence = prefix_id + seg['Sentence']
                        temp_dialog.append(dialo_sentence)
                        temp_entity.append(feature_info)
                        if (cnt == i and begin_role == 'patient') or (
                                cnt > i and begin_role == 'doctor' and patient_in_flag):
                            break_forword = True
                            temp_dialog, temp_entity = cut_dialog_without_entity(temp_dialog, temp_entity)
                            if temp_dialog and temp_entity:
                                train_data_dict['input'].append(temp_dialog)
                                train_data_dict['entity'].append(temp_entity)
                            break

                if not break_forword:
                    break
            _train_data = []

# 生成验证数据集
dev_data_dict = {"input": [], "entity": []}
for history, response in zip(test_sample, test_sample_reference):
    temp_dialog, temp_entity = [], []
    for text in history['history']:
        role, sent = text[:2], text[3:]
        entity = list(kd_ana.convert_sen_to_entity_set(sent))
        prefix_id = "speaker2" if role == "医生" else "speaker1"
        temp_dialog.append(prefix_id + sent)
        temp_entity.append(entity)
    entity = list(kd_ana.convert_sen_to_entity_set(response))
    temp_dialog.append('speaker2' + response)
    temp_entity.append(entity)
    temp_dialog, temp_entity = cut_dialog_without_entity(temp_dialog, temp_entity)
    dev_data_dict['input'].append(temp_dialog)
    dev_data_dict['entity'].append(temp_entity)

# 生成测试数据集
test_data_dict = {"input": [], "entity": []}
for history, response in zip(test_data, test_data_reference):
    temp_dialog, temp_entity = [], []
    for text in history['history']:
        role, sent = text[:2], text[3:]
        entity = list(kd_ana.convert_sen_to_entity_set(sent))
        prefix_id = "speaker2" if role == "医生" else "speaker1"
        temp_dialog.append(prefix_id + sent)
        temp_entity.append(entity)
    entity = list(kd_ana.convert_sen_to_entity_set(response))
    temp_dialog.append('speaker2' + response)
    temp_entity.append(entity)
    temp_dialog, temp_entity = cut_dialog_without_entity(temp_dialog, temp_entity)
    test_data_dict['input'].append(temp_dialog)
    test_data_dict['entity'].append(temp_entity)

dialog_data = {"train": train_data_dict["input"], "valid": dev_data_dict["input"], "test": test_data_dict["input"]}
entity_data = {"train": train_data_dict["entity"], "valid": dev_data_dict["entity"], "test": test_data_dict["entity"]}
with open(os.path.join(args.output_data_path, "dialog_data_{}.json".format(args.max_text_length)), "w", encoding="utf8") as f:
    json.dump(dialog_data, f, ensure_ascii=False)
with open(os.path.join(args.output_data_path, "entity_data_{}.json".format(args.max_text_length)), "w", encoding="utf8") as f:
    json.dump(entity_data, f, ensure_ascii=False)








