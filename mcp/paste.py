import yake

def extract_keywords_yake_filtered(text, max_keywords=5):
    kw_extractor = yake.KeywordExtractor(top=max_keywords*5, stopwords=None)
    keywords = kw_extractor.extract_keywords(text)
    keywords_sorted = sorted(keywords, key=lambda x: x[1])

    filtered = []
    seen_word_sets = []

    for phrase, score in keywords_sorted:
        phrase_set = set(phrase.lower().split())
        if any(len(phrase_set & s) / len(phrase_set | s) > 0.6 for s in seen_word_sets):
            continue
        filtered.append((phrase, score))
        seen_word_sets.append(phrase_set)
        if len(filtered) >= max_keywords:
            break
    return filtered

# Example usage:
text = "Natural language processing enables computers to understand human language."

keywords_filtered = extract_keywords_yake_filtered(text)

for kw, score in keywords_filtered:
    print(f"{kw}: {score:.4f}")
