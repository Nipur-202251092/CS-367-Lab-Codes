
import nltk
import string
import heapq

def preprocess_text(text):
    sentences = nltk.sent_tokenize(text)
    sentences = [s.lower().translate(str.maketrans('', '', string.punctuation)) for s in sentences]
    return sentences


def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    previous_row = range(len(s2) + 1)
    
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def heuristic(remaining_doc1, remaining_doc2):
    if not remaining_doc2:
        return float('inf')
    return sum(min(levenshtein_distance(s1, s2) for s2 in remaining_doc2) for s1 in remaining_doc1)


def a_star_search(doc1_sentences, doc2_sentences):
    def levenshtein_distance(s1, s2):
        if len(s1) < len(s2):
            return levenshtein_distance(s2, s1)
        
        previous_row = range(len(s2) + 1)
        
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def heuristic(remaining_doc1, remaining_doc2):
        if not remaining_doc2:
            return float('inf')
        return sum(min(levenshtein_distance(s1, s2) for s2 in remaining_doc2) for s1 in remaining_doc1)

    open_list = []
    heapq.heappush(open_list, (0, 0, 0, []))
    closed_list = set()

    while open_list:
        current_cost, skip_cost, pos, alignment = heapq.heappop(open_list)

        if pos >= len(doc2_sentences):
            return alignment

        if pos + 1 < len(doc2_sentences):
            heapq.heappush(open_list, (skip_cost + heuristic(doc1_sentences, doc2_sentences[pos+1:]), skip_cost, pos + 1, alignment + [("", doc2_sentences[pos])]))

        for i, sentence in enumerate(doc1_sentences):
            cost = levenshtein_distance(sentence, doc2_sentences[pos])
            heapq.heappush(open_list, (current_cost + cost, skip_cost, pos + 1, alignment + [(sentence, doc2_sentences[pos])]))

        if not doc1_sentences:
            return alignment

    return []


def detect_plagiarism(doc1, doc2):
    doc1_sentences = preprocess_text(doc1)
    doc2_sentences = preprocess_text(doc2)

    alignment = a_star_search(doc1_sentences, doc2_sentences)

    if alignment is None:
        alignment = []

    
    detected = False
    for sentence_pair in alignment:
        if levenshtein_distance(sentence_pair[0], sentence_pair[1]) < 20:  
            print(f"Potential Plagiarism Detected:\nDoc1: {sentence_pair[0]}\nDoc2: {sentence_pair[1]}\n")
            detected = True


    if not detected:
        print("No potential plagiarism detected.")



# Test Case 1: Identical Documents
doc1 = "Artificial intelligence is transforming the world. It is changing how we work and live."
doc2 = "Artificial intelligence is transforming the world. It is changing how we work and live."
print("Test Case 1: Identical Documents")
detect_plagiarism(doc1, doc2)

# Test Case 2: Slightly Modified Document
doc3 = "Artificial intelligence is revolutionizing our lives. It is altering the way we work and live."
print("\nTest Case 2: Slightly Modified Document")
detect_plagiarism(doc1, doc3)

# Test Case 3: Completely Different Documents
doc4 = "Cloud computing provides scalable resources for businesses. It allows for better data management."
print("\nTest Case 3: Completely Different Documents")
detect_plagiarism(doc1, doc4)

# Test Case 4: Partial Overlap
doc5 = "Artificial intelligence is transforming various industries. Some techniques are used to enhance productivity."
print("\nTest Case 4: Partial Overlap")
detect_plagiarism(doc1, doc5)

