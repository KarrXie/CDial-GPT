from evaluate import entity_dict, get_location, get_youyin, get_tong, get_time


class KD_Metric:
    def __init__(self):
        self._pred_true = 0
        self._total_pred = 0
        self._total_true = 0
        self.norm_dict = entity_dict

    def convert_sen_to_entity_set(self, sen):
        entity_set = set()
        for entity in self.norm_dict.keys():
            if entity in sen:
                entity_set.add(self.norm_dict[entity])
        if get_location(sen):
            entity_set.add('位置')
        if get_youyin(sen):
            entity_set.add('诱因')
        if get_tong(sen):
            entity_set.add("性质")
        if get_time(sen):
            entity_set.add('时长')
        return list(entity_set)


kd_ana = KD_Metric()
if __name__ == '__main__':
    sentence = "下午吃完饭就肚子疼，吃了点整肠生，感觉还是疼，也没有吃什么辛辣生冷的食物，这种不舒服大概多久了"
    print(kd_ana.convert_sen_to_entity_set(sentence))