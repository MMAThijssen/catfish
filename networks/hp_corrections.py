import numpy as np

def clean_classifications(predictions, threshold=15):
    """
    Corrects class prediction to negative label if positive stretch is shorter than threshold.
    
    Args:
        predictions -- list of int, predicted class labels
        threshold -- int, threshold to correct stretch [default: 15]
    
    Returns: corrected predictions
    """
    compressed_predictions = [[predictions[0],0]]

    for p in predictions:
        if p == compressed_predictions[-1][0]:
            compressed_predictions[-1][1] += 1
        else:
            compressed_predictions.append([p, 1])

    for pred_ci, pred_c, in enumerate(compressed_predictions):
        if pred_c[0] != 0 and pred_c[1] < threshold:

            compressed_predictions[pred_ci][0] = 0
            
    return np.concatenate([np.repeat(pred_c[0], pred_c[1]) for pred_c in compressed_predictions])
    
    
prediction = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

print(clean_classifications(prediction))
