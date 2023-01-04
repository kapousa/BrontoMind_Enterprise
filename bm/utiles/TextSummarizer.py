import ssl

import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer


class TextSummarizer:

    def __init__(self):
        ''' Constructor for this class. '''
        # Create some member animals
        self.members = ['Tiger', 'Elephant', 'Wild Cat']

    def generate_summary_from_file(self, file_name, top_n=100):
        """Generate summary of the contnents in the given file"""
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context

        try:
            stop_words = stopwords.words('english')
        except:
            nltk.download("stopwords")
            stop_words = stopwords.words('english')

        summarize_text = []

        # Step 1 - Read text anc split it
        sentences = self.__read_article(file_name)

        # Step 2 - Generate Similary Martix across sentences
        sentence_similarity_martix = self.__build_similarity_matrix(sentences, stop_words)

        # Step 3 - Rank sentences in similarity martix
        sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
        scores = nx.pagerank(sentence_similarity_graph)

        # Step 4 - Sort the rank and pick top sentences
        ranked_sentence = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

    def generate_summary_from_text(self, text):
        """Generate summary of the given text"""
        # Set up the summarizer
        summarizer = LsaSummarizer()

        # Set the number of sentences you want in the summary
        summarizer.stop_words = ["."]
        summarizer.bonus_words = [""]
        summarizer.stigma_words = [""]
        summarizer.null_words = [""]

        # Set the input text you want to summarize
        parser = PlaintextParser.from_string(text, Tokenizer("english"))

        # Generate the summary
        summary = summarizer(parser.document, 2)

        # Print the summary
        final_summary = ""
        for sentence in summary:
            final_summary = final_summary + ' '.join(sentence.words) + '\n'
        return final_summary

    def __read_article(self, file_name: str):
        """ Read the contents from the given file """
        file = open(file_name, "r")
        filedata = file.readlines()
        article = filedata[0].split(". ")
        sentences = []

        for sentence in article:
            print(sentence)
            sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
        sentences.pop()

        return sentences

    def __sentence_similarity(self, sent1, sent2, stopwords=None):
        if stopwords is None:
            stopwords = []

        sent1 = [w.lower() for w in sent1]
        sent2 = [w.lower() for w in sent2]

        all_words = list(set(sent1 + sent2))

        vector1 = [0] * len(all_words)
        vector2 = [0] * len(all_words)

        # build the vector for the first sentence
        for w in sent1:
            if w in stopwords:
                continue
            vector1[all_words.index(w)] += 1

        # build the vector for the second sentence
        for w in sent2:
            if w in stopwords:
                continue
            vector2[all_words.index(w)] += 1

        return 1 - cosine_distance(vector1, vector2)

    def __build_similarity_matrix(self, sentences, stop_words):
        # Create an empty similarity matrix
        similarity_matrix = np.zeros((len(sentences), len(sentences)))

        for idx1 in range(len(sentences)):
            for idx2 in range(len(sentences)):
                if idx1 == idx2:  # ignore if both are same sentences
                    continue
                similarity_matrix[idx1][idx2] = self.__sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

        return similarity_matrix

# text_summarizer = TextSummarizer()
# print("----------------generate_summary_from_file--------------------")
# text_summarizer.generate_summary_from_file("test.txt", top_n=5)
# print("----------------generate_summary_from_text--------------------")
# text = "Artificial stupidity is a term used to describe the limitations or flaws of artificial intelligence (AI) systems. It refers to the fact that despite the advanced capabilities of AI, it can still make mistakes or exhibit behavior that is not intelligent. One example of artificial stupidity is when an AI system is unable to understand or process a request or command because it lacks the necessary context or knowledge. For example, a voice recognition system may not be able to accurately transcribe a conversation because it doesn't understand the accents or dialects being used. Another example is when an AI system makes decisions or takes actions that are not logical or beneficial. This can occur when an AI system is not properly programmed or trained, or when it is given incomplete or inaccurate data. For example, a self-driving car may make a wrong turn because it was not programmed to recognize a detour or road closure. There are several ways to address artificial stupidity in real life. One approach is to ensure that AI systems are properly programmed and trained. This can involve providing the AI with a large and diverse dataset to learn from, as well as testing and debugging the AI to ensure it is functioning correctly. Another approach is to use human oversight and intervention when necessary. This can involve having humans monitor the actions of AI systems and intervene when necessary to prevent mistakes or errors. Finally, it is important to recognize the limitations of AI and not rely on it solely for decision-making. While AI can be a valuable tool, it is important to use it in conjunction with human judgment and expertise. In conclusion, artificial stupidity is a term used to describe the limitations or flaws of AI systems. It is important to recognize these limitations and take steps to address them, such as properly programming and training AI systems and using human oversight and intervention when necessary. By doing so, we can help ensure that AI is used effectively and intelligently in real life."
# print(text_summarizer.generate_summary_from_text(text))
