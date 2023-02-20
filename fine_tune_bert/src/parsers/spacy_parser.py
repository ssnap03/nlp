from src.parsers.parser import Parser
from src.dependency_parse import DependencyParse

import spacy


class SpacyParser(Parser):

    def __init__(self, model_name: str):
        self.model = spacy.load(model_name)

    def parse(self, sentence: str, tokens: list) -> DependencyParse:
        spacy_doc = spacy.tokens.doc.Doc(self.model.vocab, words=tokens)
        spacy_doc = self.model(spacy_doc)
        text = sentence
        token_id = {}
        tokens = []
        heads = []
        deprel = []
        for token in spacy_doc:
            if token.head == token:
                head_idx = 0
            else:
                head_idx = token.head.i - spacy_doc[0].i + 1
            head_idx = str(head_idx)
            heads.append(head_idx)
            token_id[token.text] = token.i
            tokens.append(token.text)
            deprel.append(token.dep_)
        data_dict = {
            "text": text,
            "tokens": tokens,
            "head": heads,
            "deprel": deprel,
        }
        return DependencyParse.from_huggingface_dict(data_dict)
