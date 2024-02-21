import numpy
import sklearn.metrics

def precision_recall_curve(y_true, pred_scores, thresholds):
    """
    This function is used to generate the precision recall curve.
    This returns the two lists of precisions, recalls.
    """
    precisions = []
    recalls = []
    
    for threshold in thresholds:
        y_pred = ["positive" if score >= threshold else "negative" for score in pred_scores]

        precision = sklearn.metrics.precision_score(y_true=y_true, y_pred=y_pred, pos_label="positive")
        recall = sklearn.metrics.recall_score(y_true=y_true, y_pred=y_pred, pos_label="positive")
        
        precisions.append(precision)
        recalls.append(recall)

    return precisions, recalls

def average_precision(precisions, recalls):
    """
    Using the precisions, recalls values calculated using the precision_recall_curve() 
    this function returns the average precision
    """
    
    precisions.append(1)
    recalls.append(0)

    precisions = numpy.array(precisions)
    recalls = numpy.array(recalls)

    ap = numpy.sum((recalls[:-1] - recalls[1:]) * precisions[:-1])
    
    return ap

def IoU(boxA, boxB):
    """
    Calculates the intersect over union of two boxes
    """
    
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

def find_AP(real_caption, BLIP_caption):
    """
    This function loops through each box and phrase found in a frame using the real caption and the BLIP generated caption. 
    Then returns the average precision for that frame.
    """
    
    y_true = [] #If two phrases are similiar they are assumed to be the same 
    pred_scores = [] # IoU value for the two bounding boxes.
    # The thresholds used in the precision recall curve
    thresholds=numpy.arange(start=0.2, stop=0.7, step=0.05)
    for real_boxes, real_phrases in zip(real_caption["boxes"], real_caption["phrases"]):
        for BLIP_boxes, BLIP_phrases in zip(BLIP_caption["boxes"], BLIP_caption["phrases"]):
            
            iou_value = IoU(real_boxes, BLIP_boxes)
            pred_scores.append(iou_value)
            
            sim_score = compute_bert_similarity(real_phrases, BLIP_phrases)
            
            if sim_score > 0.5:
                y_true.append("positive")
            else:
                y_true.append("negative")
    
    precisions, recalls = precision_recall_curve(y_true, pred_scores, thresholds)
    ap =  average_precision(precisions, recalls)
    
    return ap

from sklearn.metrics.pairwise import cosine_similarity
def compute_bert_similarity(real_phrases, BLIP_phrases):
    """
    This function is used to caluclate the cosine similarity of two phrases using BERT.
    """
    encode_phrases = apply_word_embedding([real_phrases, BLIP_phrases])
    sim_score = cosine_similarity(encode_phrases[0].reshape(1, -1), encode_phrases[1].reshape(1, -1))

    return sim_score

from sentence_transformers import SentenceTransformer
bert_model = SentenceTransformer('distilbert-base-nli-mean-tokens')
def apply_word_embedding(sentences:list):
  """
  Takes in a list of sentences and returns a vector of embeddings for each sentence
  """
  return bert_model.encode(sentences)