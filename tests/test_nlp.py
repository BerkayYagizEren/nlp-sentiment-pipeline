from nlp.pipeline import TextPreprocessor, TFIDFVectorizer, SentimentPipeline

def test_preprocessor():
    p = TextPreprocessor()
    tokens = p.tokenize("This is a Fantastic NLP project, works 100%!")
    assert "fantastic" in tokens
    assert "nlp" in tokens
    assert "is" not in tokens  # Stopword removed

def test_tfidf_vectorizer():
    v = TFIDFVectorizer(ngram_range=(1, 2))
    docs = [["machine", "learning"], ["deep", "learning"]]
    v.fit(docs)
    vec = v.transform(["machine", "learning"])
    assert "machine" in vec
    assert "machine_learning" in vec

def test_sentiment_pipeline():
    pipeline = SentimentPipeline()
    train_data = [
        ("The quality is superb, absolutely love it!", "positive"),
        ("Terrific execution, well done and highly recommended.", "positive"),
        ("Horrible and broken, total waste of money.", "negative"),
        ("Worst support ever, completely disappointed.", "negative")
    ]
    pipeline.train(train_data)

    pred_pos, conf_pos = pipeline.predict("I really love the superb quality")
    assert pred_pos == "positive"
    assert conf_pos > 0.5

    pred_neg, conf_neg = pipeline.predict("Disappointed and worst product")
    assert pred_neg == "negative"
    assert conf_neg > 0.5
