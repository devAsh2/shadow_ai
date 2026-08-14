#Zero-download custom NLP stub

from presidio_analyzer.nlp_engine import NlpEngine, NlpArtifacts

# Minimal dummy NLP engine that satisfies Presidio's abstract methods
class DummyNlpEngine(NlpEngine):
    def __init__(self):
        pass

    def load(self):
        # No model loading needed
        return self

    def is_loaded(self):
        return True

    def process_text(self, text, language):
        tokens = text.split()
            return NlpArtifacts(
                tokens=tokens,
                lemmas=tokens,
                tokens_indices=[(text.find(t), text.find(t) + len(t)) for t in tokens],
                entities=[],
                language=language
            )

    def process_batch(self, texts, language):
        # Return empty artifacts for each text
        return [self.process_text(t, language) for t in texts]

    def is_stopword(self, word, language):
        return False

    def is_punct(self, word, language):
        return False

    def get_supported_entities(self):
        # No built-in entities
        return []

    def get_supported_languages(self):
        # Only English in this dummy example
        return ["en"]