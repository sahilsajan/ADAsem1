import math
import sys
import copy

#Dataset is list of lists[[[122,3,4,5,6], CLASS],[[122,3,4,5,6], CLASS] ]
def read_data(file_path):
    ret_data = []

    curr_lines = open(file_path, 'r').readlines()
    curr_arr = curr_lines[0].split(',')[0:-1] #Gets the name of attributes considered. Does not include Shelf number
    for i in range(1,len(curr_lines)):
        #skipping first line with labels
        curr_lin = curr_lines[i].strip()
        curr_Arr = curr_lin.split(',')
        curr_label = curr_Arr[-1]
        curr_vals = curr_Arr[0:2] #name, mfr
        curr_vals += [float(n) for  n in curr_Arr[2:-1]]
        ret_data.append([curr_vals, curr_label])

    return ret_data


class KNN:
    def __init__(self):
        self.train_samples = []
        self.k = 1

    def preprocess_single_sample(self, curr):
        mfr_conversion = {'G': [1.0, 0.0, 0.0, 0.0, 0.0],
                          'K': [0.0, 1.0, 0.0, 0.0, 0.0],
                          'N': [0.0, 0.0, 1.0, 0.0, 0.0],
                          'P': [0.0, 0.0, 0.0, 1.0, 0.0],
                          'Q': [0.0, 0.0, 0.0, 0.0, 1.0]}
        name, mfr = curr[:2]
        n = [float(x) for x in curr[2:]]
        return mfr_conversion.get(mfr,[0.0, 0.0, 0.0, 0.0, 0.0])+n

    def preprocess_training_set(self):
        self.train_samples = [[self.preprocess_single_sample(sample[0]), sample[1]] for sample in self.train_samples]

    def euclidean(self, sample_0, sample_1):
        return math.sqrt(sum((x - y)**2 for x,y in zip(sample_0,sample_1)))

    def get_k_nearest(self, k, test_sample):
        d = [(train_sample[0],train_sample[1],self.euclidean(train_sample[0],test_sample)) for train_sample in self.train_samples]
        d.sort(key=lambda x: x[2])
        return d[:k]

    def predict_majority_vote(self, k, test_sample):
        k_nearest=self.get_k_nearest(k, test_sample)
        counts ={}
        for _, label, _ in k_nearest:
            counts[label] =counts.get(label, 0) + 1
        predicted_label=max(counts,key=counts.get)
        return predicted_label, counts[predicted_label]

    def predict_weighted_vote(self, k, test_sample):
        k_nearest = self.get_k_nearest(k, test_sample)
        weights = {}
        for _, label, dist in k_nearest:
            weight = 1 /(dist**2)
            weights[label] = weights.get(label, 0) + weight
        predicted_label = max(weights, key=weights.get)
        return predicted_label, weights[predicted_label]

    def train(self, dataset):
        self.train_samples = dataset

    def predict(self, test_instance=[], k=1, version=None):
        if version == 'majority_vote':
            return self.predict_majority_vote(k, test_instance)
        if version == 'weighted':
            return self.predict_weighted_vote(k, test_instance)
        return None


class NB:
    def __init__(self):
        self.training_set = []  
        self.bins = {}
        self.priors = {}

    def preprocess_dataset(self, dataset):
        min_vals = [float('inf')]*len(dataset[0][0][2:])
        max_vals = [float('-inf')]*len(dataset[0][0][2:])

        for sample in dataset:
            values = sample[0][2:]
            for i in range(len(values)):
                min_vals[i] =min(min_vals[i], values[i])
                max_vals[i] =max(max_vals[i],values[i])

        self.bins = {i: (min_vals[i], max_vals[i], (max_vals[i] - min_vals[i]) / 4) for i in range(len(min_vals))}

        def discretize(values):
            return [sample[0][1]] + [min(int((values[i] - self.bins[i][0]) / self.bins[i][2]), 3) for i in range(len(values))]

        return [(discretize(sample[0][2:]), sample[1]) for sample in dataset]

    def calculate_prior(self, class_input):
        total_samples = len(self.training_set)
        class_count = sum(1 for sample in self.training_set if sample[1] == class_input)
        return class_count / total_samples

    def calculate_cond_prob(self, X_attr, X_value, Y):
        count_y=sum(1 for sample in self.training_set if sample[1] ==Y)
        count_x_and_y=sum(1 for sample in self.training_set if sample[1] ==Y and sample[0][X_attr]==X_value)
        return count_x_and_y/count_y

    def calculate_cond_prob_m_estimate(self,X_attr,X_value,Y):
        p=1/4
        m=3
        count_y = sum(1 for sample in self.training_set if sample[1] == Y)
        count_x_and_y = sum(1 for sample in self.training_set if sample[1] == Y and sample[0][X_attr] == X_value)
        return (count_x_and_y + m * p) / (count_y + m)

    def train(self):
        for sample in self.training_set:
            label = sample[1]
            if label not in self.priors:
                self.priors[label] = self.calculate_prior(label)

    def preprocess_test_instance(self, test):
        values = test[2:]
        discretized_values = [test[1]] + [min(int((values[i] - self.bins[i][0]) / self.bins[i][2]), 3) for i in range(len(values))]
        return discretized_values

    def predict(self, test_instance, method):
        max_prob = float('-inf')
        best_label = None
        for label in self.priors:
            prob = self.priors[label]
            for i in range(1, len(test_instance)):
                if method == 'basic':
                    prob *= self.calculate_cond_prob(i, test_instance[i], label)
                elif method == 'm-estimate':
                    prob *= self.calculate_cond_prob_m_estimate(i, test_instance[i], label)
            if prob > max_prob:
                max_prob = prob
                best_label = label
        return best_label, max_prob


def micro_prec_recall_fscore(golds, preds):
    tp = sum(1 for g, p in zip(golds, preds) if g == p)
    total_pred = len(preds)
    total_actual = len(golds)
    prec = tp / total_pred
    recall = tp / total_actual
    f1 = 2 * prec * recall / (prec + recall)
    return prec, recall, f1


def macro_prec_recall_fscore(golds, preds):
    classes = set(golds)
    precs, recalls = [], []
    for cls in classes:
        tp = sum(1 for g, p in zip(golds, preds) if g == p == cls)
        fp = sum(1 for g, p in zip(golds, preds) if p == cls and g != cls)
        fn = sum(1 for g, p in zip(golds, preds) if g == cls and p != cls)
        prec = tp / (tp + fp)
        recall = tp / (tp + fn)
        precs.append(prec)
        recalls.append(recall)
    macro_prec = sum(precs) / len(precs)
    macro_recall = sum(recalls) / len(recalls)
    macro_f1 = 2 * macro_prec * macro_recall / (macro_prec + macro_recall)
    return macro_prec, macro_recall, macro_f1




###TESTS
###YOU MAY NOT REMOVE OR EDIT THE CODE BELOW THIS POINT!
num_incorrect= 0
train_dataset = read_data(sys.argv[1])

#Test sample 1 is modified cheerios
test_sample_1 = ['New Cereal', 'G', 111.0, 6.1, 2.0, 291.0, 2.1, 17.0, 1.1, 104.1, 25.0]

my_knn = KNN()
my_knn.train(copy.deepcopy(train_dataset))

my_knn.preprocess_training_set()

found = False
for curr_tr in my_knn.train_samples:
    if curr_tr == [[1.0, 0.0, 0.0, 0.0, 0.0, 110.0, 2.0, 2.0, 180.0, 1.5, 10.5, 10.0, 70.0, 25.0], '1']:
        found = True


if(not found or len(my_knn.train_samples) !=66 or len(my_knn.train_samples[0]) != 2 or len(my_knn.train_samples[0][0]) != 14):
    num_incorrect += 1
    print("Error with KNN preprocessing training set")

test_sample_formatted_1 = my_knn.preprocess_single_sample(test_sample_1)

if(test_sample_formatted_1 != [1.0, 0.0, 0.0, 0.0, 0.0, 111.0, 6.1, 2.0, 291.0, 2.1, 17.0, 1.1, 104.1, 25.0]):
    num_incorrect += 1
    print("Error with KNN preprocessing test sample")

my_prediction = my_knn.predict(test_instance=test_sample_formatted_1, k=3, version='majority_vote')
if(my_prediction[0] != '1'):
    num_incorrect += 1
    print("Error with majority vote KNN label")

if(my_prediction[1] != 2):
    num_incorrect += 1
    print("Error with majority vote KNN vote count")

my_prediction_weight = my_knn.predict(test_instance=test_sample_formatted_1, k=3, version='weighted')
if(my_prediction_weight[0] != '1'):
    num_incorrect += 1
    print("Error with weighted vote KNN label")

if(round(my_prediction_weight[1],3) != 0.353):
    num_incorrect += 1
    print("Error with weighted vote KNN weight sum")



my_naive_Bayes = NB()
formatted_dataset = my_naive_Bayes.preprocess_dataset(copy.deepcopy(train_dataset))
my_naive_Bayes.training_set = formatted_dataset

found = False
for tr in formatted_dataset:
    if( tr == (['G', 2, 0, 1, 2, 0, 1, 2, 0, 1], '1')):
        found = True


if(not found or len(formatted_dataset) != 66 or len(formatted_dataset[0]) != 2 or len(formatted_dataset[0][0]) != 10):
    num_incorrect += 1
    print("Error with naive Bayes preprocessing of dataset")

my_prior_prob = my_naive_Bayes.calculate_prior('2')
my_naive_Bayes.train()
my_formatted_test_sample_1= my_naive_Bayes.preprocess_test_instance(test_sample_1)

if(my_formatted_test_sample_1 != ['G', 2, 3, 1, 3, 0, 2, 0, 1, 1]):
    num_incorrect += 1
    print("Error with naive Bayes preprocessing of test sample")

x_atr = 1
x_val = 2
Y_class = "1"
cond_prob = my_naive_Bayes.calculate_cond_prob(x_atr,x_val,Y_class) #Good

if(round(cond_prob,2) != 0.53):
    num_incorrect += 1
    print("Error with naive Bayes calculate_cond_prob")

cond_prob_m = my_naive_Bayes.calculate_cond_prob_m_estimate(x_atr,x_val,Y_class) #Good

if(round(cond_prob_m,2) != 0.49):
    num_incorrect += 1
    print("Error with naive Bayes calculate_cond_prob_m_estimate")

my_pred =  my_naive_Bayes.predict(my_formatted_test_sample_1, 'basic')#Good 
if(my_pred[0] != '1'):
    num_incorrect += 1
    print("Error with naive Bayes predict label with basic")

if(round(my_pred[1], 7) != 0.0000163):
    num_incorrect += 1
    print("Error with naive Bayes predict score with basic")

my_pred_m =  my_naive_Bayes.predict(my_formatted_test_sample_1, 'm-estimate')#Good 

if(my_pred_m[0] != '1'):
    num_incorrect += 1
    print("Error with naive Bayes predict label with m-estimate")

if(round(my_pred_m[1],8) != 0.00001115):
    num_incorrect += 1
    print("Error with naive Bayes predict score with m-estimate")



true_labels = [1,1,1,2,2,2,3,3,4,4]
predicted_labels = [1,1,2,2,2,3,3,4,4,4]

micro_prec, micro_recall, micro_fscore = micro_prec_recall_fscore(true_labels, predicted_labels)
if(round(micro_prec,2) != 0.70 or round(micro_recall,2) != 0.70):
    num_incorrect += 1
    print("Error with micro precision or recall")

if(round(micro_fscore,2 )!= 0.70):
    num_incorrect += 1
    print("Error with micro F1score")

macro_prec, macro_recall, macro_fscore = macro_prec_recall_fscore(true_labels, predicted_labels)
if(round(macro_prec,2) != 0.71 or round(macro_recall,2) != 0.71):
    num_incorrect += 1
    print("Error with macro precision or recall")

if(round(macro_fscore,2) != 0.69):
    num_incorrect += 1
    print("Error with macro F1score")

Total_test = 18
num_correct = Total_test-num_incorrect
print(num_correct,'/',Total_test,"correctly passed")
