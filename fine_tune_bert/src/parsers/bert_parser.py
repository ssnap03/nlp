from src.parsers.parser import Parser
from src.dependency_parse import DependencyParse
from finetune_bert import pos_id_to_label_vocab, dep_id_to_label_vocab
from transformers import DistilBertTokenizer

import torch


class BertParser(Parser):
    """Represents a full parser that decodes parse trees from a finetuned BERT model."""

    def __init__(self, model_path: str, mst: bool = False):
        """
        Arguments:
            model_path: Path from which to load saved model.
            mst: Whether to use MST decoding or argmax decoding.
        """
        self.mst = mst
        self.model = torch.load(model_path)
        self.model.eval()
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    def get_filter_indices(self, bert_tokens):
        indices_to_remove = []
        for batch in bert_tokens:
            remove_next = False
            for i, word in enumerate(batch):
                remove = False
                if remove_next:
                    remove = True
                    remove_next = False
                token = self.tokenizer.convert_ids_to_tokens(word.item())
                if token == '-':
                    remove = True
                    remove_next = True
                elif token.startswith('##') or token == '[CLS]' or token == '[SEP]':
                    remove = True
                elif token == '[PAD]':
                    break
                if remove:
                    indices_to_remove.append(i)
        return indices_to_remove

    def parse(self, sentence: str, tokens: list) -> DependencyParse:
        bert_tokens = self.tokenizer.encode(sentence, add_special_tokens=True, max_length=126, padding=True,
                                            return_tensors='pt')

        filter_indices = self.get_filter_indices(bert_tokens)

        attention_mask = torch.ones(bert_tokens.shape)

        pos_out, dep_out = self.model(bert_tokens, attention_mask)
        if not self.mst:
            pos_out = torch.argmax(pos_out.squeeze(), dim=1)
            dep_out = torch.argmax(dep_out.squeeze(), dim=1)

        for ind in filter_indices:
            pos_out[ind] = 0
            dep_out[ind] = 0

        pos_out = (pos_out[pos_out.nonzero()]).squeeze()
        dep_out = (dep_out[dep_out.nonzero()]).squeeze()

        pos_out = [pos_id_to_label_vocab[p] for p in pos_out]
        dep_out = [dep_id_to_label_vocab[d] for d in dep_out]

        pos_out = [str((int(hd)+i+1)) if hd!='0' else '0' for i, hd in enumerate(pos_out)]

        data_dict = {
            "text": sentence,
            "tokens": tokens,
            "head": pos_out,
            "deprel": dep_out,
        }
        return DependencyParse.from_huggingface_dict(data_dict)