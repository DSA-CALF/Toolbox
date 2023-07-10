from sklearn.metrics import roc_auc_score, make_scorer

# Scoring functions
def Gini(
    y: list[float], y_pred: list[float], sample_weight: list[float] = None
) -> float:
    """Gini calculation for a binary classification.

    Args:
        y (list[float]): List of real target.
        y_pred (list[float]): List of probability prediction using a Machine Learning model.
        sample_weight (list[float], optional): List of weights used for prediction. Defaults to None.

    Returns:
        float: Gini calculated between real target and estimation.
    """
    ### Calculate score ###
    score = 2 * roc_auc_score(y, y_pred, sample_weight=sample_weight) - 1

    return score


def AUC(
    y: list[float], y_pred: list[float], sample_weight: list[float] = None
) -> float:
    """AUC calculation for a binary classification.

    Args:
        y (list[float]): List of real target.
        y_pred (list[float]): List of probability prediction using a Machine Learning model.
        sample_weight (list[float], optional): List of weights used for prediction. Defaults to None.

    Returns:
        float: AUC calculated between real target and estimation.
    """
    ### Calculate score ###
    score = roc_auc_score(y, y_pred, sample_weight=sample_weight)

    return score


### Transform functions into scorer using sklearn.make_scorer ###
Gini = make_scorer(Gini, greater_is_better=True, needs_proba=True)
AUC = make_scorer(AUC, greater_is_better=True, needs_proba=True)
