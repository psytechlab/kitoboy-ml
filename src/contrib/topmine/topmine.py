"""This is an algorithm for a phrase mining from a text corpus.

Based on original code: https://github.com/anirudyd/topmine/tree/master
Paper: [Scalable Topical Phrase Mining from Text Corpora](https://vldb.org/pvldb/vol8/p305-ElKishky.pdf)"""

import re
from collections import Counter
import math
import heapq
import sys

class PhraseMining(object):
    """
    PhraseMining performs frequent pattern mining followed by agglomerative clustering
    on the input corpus and then stores the results in intermediate files.
    
    :param min_support:
        minimum support threshold which must be satisfied by each phrase during frequent
        pattern mining.
    :param max_phrase_size:
        maximum allowed phrase size.
    :param alpha:
        threshold for the significance score.
    """

    def __init__(self, min_support=10, max_phrase_size=40, alpha=4):
        self.min_support = min_support
        self.max_phrase_size = max_phrase_size
        self.alpha = alpha

    def _frequent_pattern_mining(self, documents, word_freq, active_indices):
        """
        Performs frequent pattern mining to collect aggregate counts for all contiguous phrases in the 
        input document that satisfy a certain minimum support threshold.

        Parameters:
        @documents: the input corpus
        @word_freq: raw frequency of each word in the input corpus
        @active_indices: set of active indices
        """ 
        hash_counter = word_freq
        n = 2
        
        #iterate until documents is empty
        while(len(documents) > 0):
            temp_documents = []
            new_active_indices = []
            #go over each document
            for d_i,doc in enumerate(documents):
                #get set of indices of phrases of length n-1 with min support
                new_word_indices = []
                word_indices = active_indices[d_i]
                for index in word_indices:
                    words = doc.split()
                    if index+n-2 < len(words):
                        key = ""
                        for i in range(index, index+n-2+1):
                            if i == index+n-2:
                                key = key + words[i]
                            else:
                                key = key + words[i] + " "

                        #check if the phrase 'key' meets min support
                        if hash_counter[key] >= self.min_support:
                            new_word_indices.append(index)

                #remove the current document if there is no more phrases of length
                #n which satisfy the minimum support threshold
                if len(new_word_indices) != 0:
                    new_active_indices.append(new_word_indices)
                    temp_documents.append(doc)
                    words = doc.split()
                    for idx, i in enumerate(new_word_indices[:-1]):
                        phrase = ""
                        if (new_word_indices[idx+1] == i + 1):
                            for idx in range(i, i+n):
                                if idx == i+n-1:
                                    phrase += words[idx]
                                else:
                                    phrase += words[idx] + " "
                        hash_counter[phrase] += 1

            documents = temp_documents
            active_indices = new_active_indices
            n += 1
            if n == self.max_phrase_size:
                break
        
        hash_counter = Counter(x for x in hash_counter.elements() if hash_counter[x] >= self.min_support)
        
        return hash_counter 

    def _agglomerative_clustering(self, doc, hash_counter, total_words):
        """
        Performs agglomerative clustering to get meaningful phrases from the input document.

        Parameters:
        @doc: input corpus
        @hash_counter: map from phrases to their respective raw frequency
        @total_words: total count of the words in input corpus.
        """
        sig_map = {}
        phrases = doc.split()
        while(True):
            max_sig = float("-inf")
            max_pair = -1
            for index, word in enumerate(phrases[:-1]):
                phrase = phrases[index]+" "+phrases[index+1]
                if phrase not in sig_map:
                    sig_score = self._significance_score(phrases[index], phrases[index+1], hash_counter, total_words)
                    sig_map[phrase] = sig_score
                
                if(max_sig < sig_map[phrase]):
                    max_sig = sig_map[phrase]
                    max_pair = index

            if(max_sig < self.alpha):
                break
                
            #merge max pair
            merged_phrase = phrases[max_pair] + " "+ phrases[max_pair+1]
            
            #fix phrases
            phrases[max_pair] = merged_phrase
            phrases.pop(max_pair+1)
        
        return phrases
            
    def _significance_score(self, phrase1, phrase2, hash_counter, total_words):
        """
        Calculates the signifance score of the phrase obtained by joining phrase1 
        and phrase2. The significance score basically measures how unlikely is the
        new phrase. The more unlikely it is, the more informative it will be.

        Parameters:
        @phrase1: first phrase
        @phrase2: second phrase
        @hash_counter: map from phrases to their respective raw frequency
        @total_words: total count of the words in input corpus.
        """
        combined_phrase = phrase1+" "+phrase2
        combined_size = len(combined_phrase.split())
        actual_occurence = hash_counter[combined_phrase]
        numerator = hash_counter[phrase1]*hash_counter[phrase2]
        
        if actual_occurence == 0:
            return float("-inf")
        
        denominator = total_words * total_words
        independent_prob = numerator/denominator
        independent_prob *= 2
        
        expected_occurence = independent_prob*total_words
        
        return (actual_occurence-expected_occurence)/math.sqrt(max(actual_occurence, expected_occurence))

    def _get_true_frequency(self, hash_counter):
        """
        Updates the raw frequency of the phrases to get their true frequencies.
        """
        self.true_counter = Counter(hash_counter)
        for key in hash_counter:
            val = key.split()
            if len(val) <= 1:
                continue
            substr1 = " ".join(val[0:-1])
            substr2 = " ".join(val[1:])
            self.true_counter[substr1] -= hash_counter[key]
            self.true_counter[substr2] -= hash_counter[key]

    def _get_word_freq(self, documents):
        """
        Calculates the frequency of each word in the input document.
        """
        total_words = 0
        word_freq = Counter()
        active_indices = []
        for doc_index, doc in enumerate(documents):
            words = doc.split()
            word_indices = []
            for word_index, word in enumerate(words):
                word_freq[word] += 1
                word_indices.append(word_index)
                total_words += 1
            active_indices.append(word_indices)

        return total_words, word_freq, active_indices

    def _get_partitioned_docs(self, document_range, doc_phrases):
        """
        Partitions the input document based on the punctuations.
        """
        partitioned_docs = []
        start = 0
        end = 0
        for idx in document_range:
            end = idx
            final_doc = []
            for i in range(start, end):
                final_doc.extend(doc_phrases[i])
            partitioned_docs.append(final_doc)
            start = end

        return partitioned_docs

    def _process_partitioned_docs(self, partitioned_docs_):
        vocab = {}
        index_vocab = []
        partitioned_docs = []
        
        word_counter = 0
        for document_index, document in enumerate(partitioned_docs_):
            document_of_phrases = []
            for phrase in document:
                phrases_of_words = []
                for word in phrase.split():
                    if word not in vocab:
                        vocab[word] = word_counter
                        index_vocab.append(word)
                        word_counter += 1
                    phrases_of_words.append(vocab[word])
                document_of_phrases.append(phrases_of_words)
            partitioned_docs.append(document_of_phrases)
        return partitioned_docs, index_vocab

    def fit(self, document_list: list[list[str]]):
        """
        Runs the phrase mining algorithm.

        Parameters:
        @document_list: list of documents splitted into phrase-invariant segments (sentences as a common way).
        @min_support: minimum support threshold which must be satisfied by each phrase.
        @max_phrase_size: maximum allowed phrase size
        @alpha: threshold for the significance score
        """


        num_docs = len(document_list)
        
        i = 0
        document_range = []
        for d in document_list:
            i += len(d)
            document_range.append(i)
        
        documents = [y for x in document_list for y in x]

        #calculate frequency of all words
        total_words, word_freq, active_indices = self._get_word_freq(documents)

        vocab_size = len(word_freq)

        #run frequent pattern mining 
        hash_counter = self._frequent_pattern_mining(documents, word_freq, active_indices)
        
        #update true count of each phrase
        self._get_true_frequency(hash_counter)

        #run agglomerative clustering
        doc_phrases = []
        for doc in documents:
            doc_phrases.append(self._agglomerative_clustering(doc, hash_counter, total_words))

        partitioned_docs = self._get_partitioned_docs(document_range, doc_phrases)
        partitioned_docs, index_vocab = self._process_partitioned_docs(partitioned_docs)

        return partitioned_docs, index_vocab

    def get_frequent_phrases(self, min_support):
        """
        Returns the most frequent phrases in the corpus that occur more than 
        the minimum support in descending order of frequency
        """
        frequent_phrases = []
        for key,value in self.true_counter.most_common():
            if value >= min_support and len(key.split(" "))>1:
                frequent_phrases.append((key, value))
            elif value < min_support:
                break
        return frequent_phrases