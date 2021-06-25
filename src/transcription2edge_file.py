""" Module containing the functions used to get the edge files from the transcription

The functions in this module compile the several steps from the transcription to the edge .txt
for one file, also contains the methods to get the word frecuency. Requires nltk to be installed.
Contains the following functions:
    * get_word_frecuency - calculates the frecuency of each keyword
    * export_word_frecuency - calculates and exports the frecuency of each keyword
    * get_match_list - gets the keywords that appear in each line
    * get_cooccurences - returns a list containing the cooccurences
    * get_edges - returns a dictionary with the edges and their weights
    * export_edges - export the weighted edges to a txt file
"""

from nltk import SnowballStemmer


def get_word_frequency(transcription_file, keywords):
    """Returns the frecuency of every word in keywords over the transcription file

    :type transcription_file: str
    :param transcription_file: source or path to the transcription,
        must be a .txt file
    :type keywords: list
    :param keywords: keywords to be searched
    :rtype: dict
    :return: dictionary with words as keys and their frequencies as values
    """
    # getting the transcription lines
    lines = []
    transcription = open(transcription_file, 'r', encoding='utf-8')
    for l in transcription:
        lines.append(l)
    transcription.close()

    # stemming the words
    s = SnowballStemmer('spanish')
    stemmed_keywords = [(s.stem(w), w) for w in keywords]

    words_freq = {}
    # getting the frequencies
    for l in lines:
        list_of_words = l.strip('\n').strip(' ').split(' ')
        for word in list_of_words:
            for tup in stemmed_keywords:
                if s.stem(word) == tup[0]:
                    words_freq[tup[0]] = words_freq.get(tup[0], 0) + 1

    # normalizing
    total = 0
    for key in words_freq.keys():
        total = total + words_freq[key]

    for key in words_freq.keys():
        words_freq[key] = words_freq[key] / total

    return words_freq


def export_word_frequency(transcription_file, keywords, frequency_file):
    """Exports the word frecuency to a txt file

    :type transcription_file: str
    :param transcription_file: path of the transcription, must be a .txt file
    :type keywords: list
    :param keywords: keywords to be searched and exported
    :type frequency_file: str
    :param frequency_file: path to the output
    """
    # getting word frequency
    word_freq = get_word_frequency(transcription_file, keywords)

    # exporting to file
    stemmer = SnowballStemmer("spanish")
    file = open(frequency_file, 'w+', encoding='utf-8')
    for word in keywords:
        key = stemmer.stem(word)
        file.write(word + " " + str(word_freq.get(key, 0)) + "\n")

    file.close()


def get_match_list(transcription_file, keywords, stemmed=False):
    """Returns a matchlist containing the keywords for each line
    in the transcription

    :type transcription_file: str
    :param transcription_file: path of the transcription
    :type keywords: list
    :param keywords: keywords to be searched
    :rtype: list
    :return: the keywords that appeared in each line,
        minding the order of appereance
    """
    # getting the transcription lines
    lines = []
    transcription = open(transcription_file, 'r', encoding='utf-8')
    for l in transcription:
        lines.append(l)
    transcription.close()

    # stemming the keywords if not stemmed
    stemmer = SnowballStemmer("spanish")
    stemmed_keywords = keywords if stemmed else [stemmer.stem(w) for w in keywords] 

    # calculating the matches
    match_list = []
    i = 0

    while i < len(lines) - 1:

        sentence_1 = lines[i].strip('\n').strip(' ')
        sentence_2 = lines[i + 1].strip('\n').strip(' ')

        sentence_10_secs = sentence_1 + ' ' + sentence_2

        sentence_words = sentence_10_secs.split(' ')

        match = ''
        for word in sentence_words:
            for k in stemmed_keywords:
                if k == stemmer.stem(word):
                    match += k + '-'
        match_list.append(match)
        i += 1

    return match_list


def get_cooccurrences(transcription_file, keywords, stemmed=False):
    """Returns a co-occurence list between the keywords,
     elements may be repeated

    :type transcription_file: str
    :param transcription_file: path of the transcription
    :type keywords: list
    :param keywords: keywords to be searched
    :rtype: list
    :return: pairs of strings containing the co occurrences
    """
    # getting the match list
    match_list = get_match_list(transcription_file, keywords, stemmed)
    cooccurrence_list = []

    # obtaining the co occurrences
    for line in match_list:
        line = line.strip('-')
        line_list = line.split('-')

        i = 0

        while i < len(line_list) - 1:
            word_1 = line_list[i]
            word_2 = line_list[i + 1]
            pair = (word_1, word_2)
            cooccurrence_list.append(pair)
            i += 1

    return cooccurrence_list


def get_edges(transcription_file, keywords, stemmed=False):
    """Returns the edges of the graph and their weights from
    the transcription

    :type transcription_file: str
    :param transcription_file: path for the transcription
    :type keywords: list
    :param keywords: keywords to be searched, also the nodes of the graph
    :rtype: dict
    :return: edges as keys and weights as values
    """
    # getting the co occurrenes
    cooccurrence_list = get_cooccurrences(transcription_file, keywords, stemmed)

    edges_list = []

    # getting the edges without repetition
    for e in cooccurrence_list:
        if e not in edges_list and e[0] != '' and e[1] != '' and e[0] != e[1]:
            edges_list.append(e)

    edges = {}
    # getting the weights
    for edge in edges_list:
        weight = cooccurrence_list.count(edge)
        edges[edge] = weight

    return edges


def export_edges(transcription_file, edges_file, keywords, stemmed=False):
    """exports the weighted edges to a txt file

    :type transcription_file: str
    :param transcription_file: path for the transcription
    :type keywords: list
    :param keywords: keywords to be searched
    :type edges_file: str
    :param edges_file: path for the output
    """
    # getting the edges
    edges = get_edges(transcription_file, keywords, stemmed)
    # exporting to the file
    file = open(edges_file, 'w', encoding='utf-8')

    for key, val in edges.items():
        file.write(key[0] + "," + key[1] + "," + str(val) + "\n")

    file.close()
