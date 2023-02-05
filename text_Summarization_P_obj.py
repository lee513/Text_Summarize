import sys
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from konlpy.tag import Mecab



class Text_Summarizer():
    def __init__(self):
        
        self.mecab = Mecab(dicpath=r"C:\mecab\mecab-ko-dic")
        #rank높은 문장 갯수
        self.top_n = 3
        
        ##불용어 사전
        stop_words_f ="./stopwords.txt"
        with open(stop_words_f, "r", encoding='utf-8') as f:
             stop_words = f.readlines()
        self.stop_words = [stop_word.strip() for stop_word in stop_words]
        
        
    #문장 cleaning
    def cleaning(self, sentences):
        for i in range(len(sentences)):
            sentences[i] = re.sub('\《.*\》|\s-\s.*', '', sentences[i])
            sentences[i] = re.sub('\(.*\)|\s-\s.*', '', sentences[i])
            #필드의 태그를 모두 제거
            sentences[i] = re.sub('(<([^>]+)>)', '', sentences[i])
            # 개행문자 제거
            sentences[i] = re.sub('\n', ' ', sentences[i])
            #특수문자 제거
            sentences[i] = re.sub(r'[^\w\s]', '', sentences[i])
        return sentences    
    
    #mecab으로 조사, 전치사 분리
    def mecab_morphs(self, sentences):
        for i, sentence in enumerate(sentences):
            sentences[i] = self.mecab.morphs(sentence)
        return sentences
    
    def sentence_similarity(self, sent1, sent2):
        
        #토큰이 비어있으면 제거
        for i in range(len(sent1)):
            if range(len(sent1[i])) == 0:
                sent1.pop(i)
        for i in range(len(sent2)):
            if range(len(sent2[i])) == 0:
                sent2.pop(i)
        #비교할 문장들의 토큰합
        all_words = list(set(sent1 + sent2))
        
        #cosine 유사도를 구할 vector 생성
        vector1 = [0] * len(all_words)
        vector2 = [0] * len(all_words)
        
        # 첫 문장 생성
        for w in sent1:
            if w in self.stop_words:
                continue
            vector1[all_words.index(w)] += 1
        # 두번째 문장 생성
        for w in sent2:
            if w in self.stop_words:
                continue
            vector2[all_words.index(w)] += 1
        
        #유사도 계산
        return 1 - cosine_distance(vector1, vector2)
    

    def graph_draw(self):
        #각 노드는 각 문장을 의미 엣지의 두께는 연결된 노드들의 코사인 유사도
        size = 10
        size_similarity_matrix = self.similarity_matrix.flatten()*size
        nx.draw(self.G, with_labels=True, node_color='white', width=size_similarity_matrix.tolist())


    def __call__(self, document):
        
        #self.document = document.split(". ")
        
        #최종 output으로 나올 문서 백업
        sentences = document.copy()
        
        c_sentences = self.cleaning(document)
        morph_c_sentences = self.mecab_morphs(c_sentences)
        
        #유사도행렬 구성
        self.similarity_matrix = np.zeros((len(sentences), len(sentences)))
        for idx1 in range(len(sentences)):
            for idx2 in range(len(sentences)):
                if idx1 == idx2: #같은 문장은 계산하지않습니다.
                    continue 
                self.similarity_matrix[idx1][idx2] = self.sentence_similarity(morph_c_sentences[idx1], morph_c_sentences[idx2])
    
        #유사도행렬을 인접행렬로 취급 무방향 그래프를 그린다.
        self.G = nx.from_numpy_array(self.similarity_matrix)
        
        
        #유사도 행렬로 pagerank 알고리즘 적용
        scores = nx.pagerank(self.G)
        
        #랭크 높은순으로 최종 요약 제공
        ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
        
        summarize_text = []
        for i in range(self.top_n):
          summarize_text.append("".join(ranked_sentence[i][1]))
    
        return " ".join(summarize_text)




## 데이터 불러오기

val_df = pd.read_pickle("val_df.pkl")


#사용예시
text_sumarize = Text_Summarizer()
model_system = text_sumarize("")
text_sumarize.graph_draw()


#모델 추론
inferences = []
for i in range(len(val_df.text)):
    result = text_sumarize(val_df.text[i])
    inferences.append(result)
    print("{a}경과중".format(a=i))

import pickle
#mecab으로 나눈 추론 모델 저장
# with open('mecab_inferences_valdata.pkl','wb') as f:
#     pickle.dump(inferences, f)

with open('mecab_inferences_valdata.pkl','rb') as f:
    inferences = pickle.load(f)

references = []
for i in range(len(val_df.extractive)):
    val = " ".join(val_df.extractive[i])
    references.append(val)

print(len(inferences), len(references))

from rouge import Rouge

rouge = Rouge()
scores = rouge.get_scores(inferences, references, avg=True)

#with open('model_1_valdata_score.pkl','wb') as f:
#    pickle.dump(scores, f)
    
with open('model_1_valdata_score.pkl','rb') as f:
    scores = pickle.load(f)

print(" Rouge-1: recall: {:.3f} precision: {:.3f} f-1: {:.3f}".format(scores['rouge-1']['r'], scores['rouge-1']['p'], scores['rouge-1']['f']),'\n',
      "Rouge-2: recall: {:.3f} precision: {:.3f} f-1: {:.3f}".format(scores['rouge-2']['r'], scores['rouge-2']['p'], scores['rouge-2']['f']),'\n',
      "Rouge-L: recall: {:.3f} precision: {:.3f} f-1: {:.3f}".format(scores['rouge-l']['r'], scores['rouge-l']['p'], scores['rouge-l']['f']))



#################
##mecab으로 #일반명사 #고유명사 #대명사 #동사 #형용사
class Text_Summarizer_2():
    def __init__(self):
        
        self.mecab = Mecab(dicpath=r"C:\mecab\mecab-ko-dic")
        #rank높은 문장 갯수
        self.top_n = 3
        
        ##불용어 사전
        stop_words_f ="./stopwords.txt"
        with open(stop_words_f, "r", encoding='utf-8') as f:
             stop_words = f.readlines()
        self.stop_words = [stop_word.strip() for stop_word in stop_words]
        
        
    #문장 cleaning
    def cleaning(self, sentences):
        for i in range(len(sentences)):
            sentences[i] = re.sub('\《.*\》|\s-\s.*', '', sentences[i])
            sentences[i] = re.sub('\(.*\)|\s-\s.*', '', sentences[i])
            #필드의 태그를 모두 제거
            sentences[i] = re.sub('(<([^>]+)>)', '', sentences[i])
            # 개행문자 제거
            sentences[i] = re.sub('\n', ' ', sentences[i])
            #특수문자 제거
            sentences[i] = re.sub(r'[^\w\s]', '', sentences[i])
        return sentences    
    
    #mecab으로 #일반명사 #고유명사 #대명사 #동사 #형용사
    def mecab_pos(self, sentences):
        
        for i, sentence in enumerate(sentences):
            sent_pos = []
            for word in self.mecab.pos(sentence):
                if word[1] in ['NNG', 'NNP','NP','VV','VA' ]:
                    sent_pos.append(word[0])
            sentences[i] = sent_pos
        return sentences
    
    def sentence_similarity(self, sent1, sent2):
        
        #토큰이 비어있으면 제거
        for i in range(len(sent1)):
            if range(len(sent1[i])) == 0:
                sent1.pop(i)
        for i in range(len(sent2)):
            if range(len(sent2[i])) == 0:
                sent2.pop(i)
                
        #비교할 문장들의 토큰합
        all_words = list(set(sent1 + sent2))
        
        #cosine 유사도를 구할 vector 생성
        vector1 = [0] * len(all_words)
        vector2 = [0] * len(all_words)
        
        # 첫 문장 생성
        for w in sent1:
            if w in self.stop_words:
                continue
            vector1[all_words.index(w)] += 1
        # 두번째 문장 생성
        for w in sent2:
            if w in self.stop_words:
                continue
            vector2[all_words.index(w)] += 1
        
        #유사도 계산
        return 1 - cosine_distance(vector1, vector2)
    

    def graph_draw(self):
        #각 노드는 각 문장을 의미 엣지의 두께는 연결된 노드들의 코사인 유사도
        size = 10
        size_similarity_matrix = self.similarity_matrix.flatten()*size
        nx.draw(self.G, with_labels=True, node_color='#DEF0ED',
                node_size=300,
                width=size_similarity_matrix.tolist())
        

    def __call__(self, document):
        
        #self.document = document.split(". ")
        
        #최종 output으로 나올 문서 백업
        sentences = document.copy()
        
        c_sentences = self.cleaning(document)
        morph_c_sentences = self.mecab_pos(c_sentences)
        
        #유사도행렬 구성
        self.similarity_matrix = np.zeros((len(sentences), len(sentences)))
        for idx1 in range(len(sentences)):
            for idx2 in range(len(sentences)):
                if idx1 == idx2: #같은 문장은 계산하지않습니다.
                    continue 
                self.similarity_matrix[idx1][idx2] = self.sentence_similarity(morph_c_sentences[idx1], morph_c_sentences[idx2])
    
        #유사도행렬을 인접행렬로 취급 무방향 그래프를 그린다.
        self.similarity_matrix = np.nan_to_num(self.similarity_matrix)
        self.G = nx.from_numpy_array(self.similarity_matrix)
        
        
        #유사도 행렬로 pagerank 알고리즘 적용
        scores = nx.pagerank(self.G)
        
        #랭크 높은순으로 최종 요약 제공
        ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
        
        summarize_text = []
        for i in range(self.top_n):
          summarize_text.append("".join(ranked_sentence[i][1]))
        
        summarize_text = " ".join(summarize_text)
        
        return summarize_text.replace('\n', ' ')


#사설 문서 요약 검증
import pandas as pd
val_df = pd.read_pickle("val_df.pkl")

references = []
for i in range(len(val_df.extractive)):
    val = " ".join(val_df.extractive[i])
    references.append(val)

#검증 데이터 모델 추론
text_sumarize = Text_Summarizer_2()
inferences = []
for i in range(len(val_df.text)):
    result = text_sumarize(val_df.text[i])
    inferences.append(result)
    print("{a}경과중".format(a=i))

print(len(inferences), len(references))

from rouge import Rouge

rouge = Rouge()
scores = rouge.get_scores(inferences, references, avg=True)
scores

#mecab으로 나눈 추론 모델 저장
# with open('model_2_valdata_score.pkl','wb') as f:
#     pickle.dump(scores, f)

import pickle
with open('model_2_valdata_score.pkl','rb') as f:
    scores = pickle.load(f)

print(" Rouge-1: recall: {:.3f} precision: {:.3f} f-1: {:.3f}".format(scores['rouge-1']['r'], scores['rouge-1']['p'], scores['rouge-1']['f']),'\n',
      "Rouge-2: recall: {:.3f} precision: {:.3f} f-1: {:.3f}".format(scores['rouge-2']['r'], scores['rouge-2']['p'], scores['rouge-2']['f']),'\n',
      "Rouge-L: recall: {:.3f} precision: {:.3f} f-1: {:.3f}".format(scores['rouge-l']['r'], scores['rouge-l']['p'], scores['rouge-l']['f']))



#inference
file_name = "test1.txt"
file = open(file_name, "r", encoding="UTF-8")
document = file.read()
article = document.split(". ")

text_summarize = Text_Summarizer_2()
summarize = text_summarize(article)
print(summarize)
text_summarize.graph_draw()








