#Assignment 1 solution for CSCI 528
import sys, os
import math
#YOU MAY NOT IMPORT ANY OTHER LIBRARIES OR PACKAGES
#If you import additional libraries, your code will automatically be marked out of 50% and all code that uses the library will be marked as incorrect

#Using the bike-sharing-dataset
#run this code from command line as pyton

#Dataset that is returned is a dicationary
#Keys should be instance id (as an int), value should be a list of data values in the same order as the data file 
#For example, the first row should be added to the dictionary as {1: ["2011-01-01", 1, 0, 1, 0, 6, 0, 2, 0.344167, 0.363625, 0.805833, 0.160446, 331, 654, 985]} 
#All values should be floats except for date, which should be a string
#column labels start indexing at 0 with date
def read_data(file_path):
    ret_data = {}
    column_labels = {}
    curr_lines = open(file_path, 'r').readlines()
    curr_arr = curr_lines[0].split(',')[1:]
    for n in range(len(curr_arr)):
        column_labels[curr_arr[n].strip()] = n
    for i in range(1,len(curr_lines)):
        #skipping first line with labels
        curr_lin = curr_lines[i].strip()
        curr_Arr = curr_lin.split(',')
        curr_vals = [curr_Arr[1]] #date
        curr_vals += [float(n) for  n in curr_Arr[2:]]
        ret_data[int(curr_Arr[0])] = curr_vals
    return ret_data, column_labels

#Get number of samples
def count_samples(my_data):
    number = len(my_data)
    return number

#Get the min and max values of a column
def get_min_max(my_data, column_idx):
    temp = [row[column_idx] for row in my_data.values()]
    my_min=min(temp)
    my_max = max(temp)

    return (my_min, my_max)

#Use sample standard deviation
def get_mean_median_standard_deviation(my_data, column_idx):
    temp =[row[column_idx] for row in my_data.values()]
    my_mean = sum(temp) / len(temp)
    sortedNum = sorted(temp)
    mid = len(sortedNum) //2
    if len(temp) % 2 == 0:
        my_median == (sortedNum[mid-1] + sortedNum[mid])/2
    else:
        my_median = sortedNum[mid]
    variance = sum((x-my_mean) ** 2 for x in temp) / (len(temp) -1)
    my_standard_deviation = math.sqrt(variance)

    return (my_mean, my_median, my_standard_deviation)


#Using Cosine similarity, find the most similar sample from the dataset
#Ignore the attribute for the date

def cosine_similarity(vec1, vec2): 
    dot_product = sum(a * b for a, b in zip(vec1, vec2)) 
    magnitude1 = math.sqrt(sum(a ** 2 for a in vec1)) 
    magnitude2 = math.sqrt(sum(b ** 2 for b in vec2)) 
    if magnitude1 == 0 or magnitude2 == 0: 
        return 0 
    return dot_product / (magnitude1 * magnitude2)

#Return the sample ID and the Euclidean distance
#Param: sample is a list of values in the same order as the dataset, which does not include an ID.

def get_most_similar_sample(my_data, sample):
    max_sim = 0
    sim_ID = None
    numeric_sample = [float(val) for val in sample[1:]]
    for row_id, row in my_data.items():
        numeric_row = [float(val) for val in row[1:]]
        similarity = cosine_similarity(numeric_row, numeric_sample)
        if similarity > max_sim:
            max_sim = similarity
            sim_ID = row_id

    return (sim_ID, max_sim)


#calculate the Pearson correlation between the two attributes, given by their index
def get_pearson_correlation(my_data, attr_a_idx, attr_b_idx):
    column_a = [row[attr_a_idx] for row in my_data.values()]
    column_b = [row[attr_b_idx] for row in my_data.values()]
    mean1 = sum(column_a)/len(column_a)
    mean2 = sum(column_b)/len(column_b)
    covariance = sum((a-mean1) * (b-mean2) for a,b in zip(column_a, column_b))
    standard_deviation1 = math.sqrt(sum((a-mean1)**2 for a in column_a))
    standard_deviation2 = math.sqrt(sum((b-mean2)**2 for b in column_b))
    return covariance/(standard_deviation1 * standard_deviation2)

#find the attribute from a list of candidate attributes that has the largest absolute pearson correlation with the target attribute
def find_attribute_with_largest_correlation(my_data, my_data_labels, candidate_attrs, target_attr):
    targetIndex = my_data_labels[target_attr]
    max_correlation = 0
    best_cand = 'None'
    for attribute in candidate_attrs:
        attr_idx = my_data_labels[attribute]
        corr = abs(get_pearson_correlation(my_data, attr_idx, targetIndex))
        if corr > max_correlation:
          max_correlation = corr
          best_cand = attribute
    return (best_cand, max_correlation)

###TESTS
###YOU MAY NOT REMOVE OR EDIT THE CODE BELOW THIS POINT!
bike_data, data_labels = read_data(sys.argv[1])
#print(bike_data)
num_checks_failed = 0
total_num_checks = 8

#1
if(count_samples(bike_data) != 731):
    print("FAILED CHECK 1")
    num_checks_failed += 1

#2
curr_min, curr_max = get_min_max(bike_data, data_labels['temp'])
if(round(curr_min, 3) != 0.059 or round(curr_max,3) != 0.862):
    print("FAILED CHECK 2")
    num_checks_failed += 1

#3
curr_mean, curr_median, curr_standard_deviation = get_mean_median_standard_deviation(bike_data, data_labels['windspeed'])
if(round(curr_mean,3) != 0.190 or round(curr_median, 3) != 0.181):
    print("FAILED CHECK 3")
    num_checks_failed += 1

#4
if(round(curr_standard_deviation, 3) != 0.077):
    print("FAILED CHECK 4")
    num_checks_failed += 1

#5
test_sample = ["2012-01-02",1,0,1,0,0,0,2,0.343478,0.253739,0.126087,0.208539,4,8,700]
most_sim_Id, cosine_sim = get_most_similar_sample(bike_data, test_sample)
if(most_sim_Id != 513):
    print("FAILED CHECK 5")
    num_checks_failed += 1

#6
if( round(cosine_sim,3) != 0.823):
    print("FAILED CHECK 6")
    num_checks_failed += 1

#7
pearson_corr = get_pearson_correlation(bike_data,data_labels['month'],data_labels['count'])
if(round(pearson_corr, 3) != 0.280):
    print("FAILED CHECK 7")
    num_checks_failed += 1


#8
attrib_to_compare = ['year', 'month', 'temp', 'humidity', 'windspeed']
target_attr = 'count'
attr_corr, corr = find_attribute_with_largest_correlation(bike_data, data_labels, attrib_to_compare, target_attr)
if(attr_corr != 'temp' or round(corr, 3) !=0.627):
    print("FAILED CHECK 8")
    num_checks_failed += 1

print("Completed:",total_num_checks-  num_checks_failed,"out of",total_num_checks,"checks")
