from src.dependency_parse import DependencyParse


def get_metrics(predicted: DependencyParse, labeled: DependencyParse) -> dict:
    n_tokens = len(labeled.tokens)
    n_uas = 0
    n_las = 0
    for i in range(n_tokens):
        if predicted.heads[i] == labeled.heads[i]:
            if predicted.deprel[i].lower() == labeled.deprel[i].lower():
                n_las += 1
            n_uas += 1
    uas = n_uas/n_tokens
    las = float(n_las/n_tokens)
    return {
        "uas": uas,
        "las": las,
    }
