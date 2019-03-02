import math

'''
Prints the results of the three classifications methods to the 'output.txt' file in the requested format.
'''
def print_results_to_file(results_dict ,accuracy_dict):
    dtl_classified_testing_data_list = results_dict['dtl classified testing data list']
    knn_classified_testing_data_list = results_dict['knn classified testing data list']
    nb_classified_testing_data_list = results_dict['nb classified testing data list']
    dtl_accuracy = accuracy_dict['dtl accuracy']
    knn_accuracy = accuracy_dict['knn accuracy']
    nb_accuracy = accuracy_dict['nb accuracy']
    tests_num = len(dtl_classified_testing_data_list)
    with open('output.txt', 'w') as f:
        f.write("Num\tDT\tKNN\tnaiveBase\n")
        for i in range(tests_num):
            f.write(str(i+1) + "\t" + dtl_classified_testing_data_list[i][-1][1] + "\t" +
                    knn_classified_testing_data_list[i][-1][1] + "\t" + nb_classified_testing_data_list[i][-1][1] + "\n")
        f.write("\t" + str(math.ceil(dtl_accuracy*100)/100) + "\t" + str(math.ceil(knn_accuracy*100)/100) + "\t" +
                str(math.ceil(nb_accuracy*100)/100))

'''
Classification using Naive-Bayes.  
'''
def naive_bayes_classifier(data_to_be_tested, arguments_dict):
    training_data_list = arguments_dict['training data list']

    '''
    Finds the index of the attribute 'attribute' in the training_example.
    '''
    def find_attribute_index(attribute, training_data_list):
        for index in range(len(training_data_list[0])):
            if training_data_list[0][index][0] == attribute:
                return index

    '''
    Returns the number of optional values for the given attribute in the given training examples.
    '''
    def att_optional_values_num(attribute, training_data_list):
        att_optional_values_list = []
        attribute_index = find_attribute_index(attribute, training_data_list)
        for example in training_data_list:
            if example[attribute_index][1] not in att_optional_values_list:
                att_optional_values_list.append(example[attribute_index][1])
        return len(att_optional_values_list)

    '''
    Returns the number of training examples where the classification is the given one, and the value of
    attribute attribute_value_tuple[0] is attribute_value_tuple[1].
    '''
    def find_rel_occurrences_num(training_data_list, attribute_value_tuple, classification):
        rel_occurrences_num = 0
        attribute_index = find_attribute_index(attribute_value_tuple[0], training_data_list)
        for training_data in training_data_list:
            if training_data[-1][1] == classification:
                if training_data[attribute_index][1] == attribute_value_tuple[1]:
                    rel_occurrences_num += 1
        return rel_occurrences_num

    '''
    Computes conditional probability.
    '''
    def cond_prob(att_val_rel_occurrences_num, class_occurrences_num, optional_values_num):
        return (att_val_rel_occurrences_num + 1)/(class_occurrences_num + optional_values_num)

    '''
    Computes the number of each optional classification.
    '''
    def class_occurrences(training_data_list):
        # Maps between each classification and the number of its occurrences
        classifications_num = {}
        for training_data in training_data_list:
            if training_data[-1][1] in classifications_num.keys():
                classifications_num[training_data[-1][1]] += 1
            else:
                classifications_num[training_data[-1][1]] = 1
        return classifications_num

    examples_num = len(training_data_list)
    class_occurrences_num = class_occurrences(training_data_list)
    final_classifications_prob = {}
    for classification in class_occurrences_num.keys():
        final_classifications_prob[classification] = 1
        for i in range(len(data_to_be_tested)-1):
            att_val_rel_occurrences_num = find_rel_occurrences_num(training_data_list, data_to_be_tested[i], classification)
            optional_values_num = att_optional_values_num(data_to_be_tested[i][0], training_data_list)
            final_classifications_prob[classification] *= cond_prob(att_val_rel_occurrences_num,
                                                                    class_occurrences_num[classification],
                                                                    optional_values_num)
        final_classifications_prob[classification] *= (class_occurrences_num[classification]/examples_num)
    return max(final_classifications_prob, key=final_classifications_prob.get)

'''
Print the given decision tree to the 'output_tree.txt' file.
'''
def print_decision_tree_to_file(decision_tree):
    def traverse_and_print_tree(f, current_node, depth):
        for att_val in sorted(current_node.successors.keys()):
            for i in range(depth):
                f.write("\t")
            if depth:
                f.write("|")
            if type(current_node.successors[att_val]) is Leaf:
                f.write(current_node.attribute + "=" + att_val + ":" + current_node.successors[att_val].classification + "\n")
            else:
                f.write(current_node.attribute + "=" + att_val + "\n")
                traverse_and_print_tree(f, current_node.successors[att_val], depth + 1)

    current_node = decision_tree

    with open('output_tree.txt', 'w') as f:
        traverse_and_print_tree(f, current_node, 0)

'''
Represents the 'data_to_be_tested' as a dictionary.
'''
def data_as_dict(data_to_be_tested):
    data_dict = {}
    for i in range(len(data_to_be_tested)-1):
        data_dict[data_to_be_tested[i][0]] = data_to_be_tested[i][1]
    return data_dict

'''
Finds the classification for 'data_to_be_tested', according to the given decision tree.
'''
def decision_tree_classifier(data_to_be_tested, arguments_dict):
    decision_tree = arguments_dict['decision tree']
    tested_data_dict = data_as_dict(data_to_be_tested)
    current_node = decision_tree
    while type(current_node) is not Leaf:
        current_node = current_node.successors[tested_data_dict[current_node.attribute]]
    return current_node.classification

'''
Leaf represents a classification.
'''
class Leaf:
    # Constructor
    def __init__(self, classification):
        self.classification = classification

'''
Node represents an attribute, and its 'successors' dictionary maps between each of its values and
the next node.
'''
class Node:
    # Constructor
    def __init__(self, attribute):
        self.attribute = attribute
        # Maps between each value of 'attribute' and its subtree.
        self.successors = {}

'''
The function 'decision_tree_learning' is recursive, so the job of 'dtl_top_level' is to make the first call to
'decision_tree_learning', with the right arguments.
'''
def dtl_top_level(training_examples):

    '''
    Finds the index of the attribute 'attribute' in the training_example.
    '''
    def find_attribute_index(attribute, training_examples):
        for index in range(len(training_examples[0])):
            if training_examples[0][index][0] == attribute:
                return index

    '''
    Filters the training examples where the value of 'best_attribute' is 'attribute_value'.
    '''
    def filter_relevant_training_examples(best_attribute, attribute_value, training_examples):
        relevant_training_examples_list = []
        # Find the index of the attribute 'best_attribute' in a training_example
        attribute_index = find_attribute_index(best_attribute, training_examples)
        # For each example check if its 'best_attribute' vlue is 'attribute_value'
        for example in training_examples:
            if example[attribute_index][1] == attribute_value:
                relevant_training_examples_list.append(example)
        return relevant_training_examples_list

    '''
    Returns a list of the possible values for the given 'best_attribute'.
    '''
    def possible_attribute_values(best_attribute, training_examples):
        possible_attribute_values_list = []
        # Find the index of the attribute 'best_attribute' in a training_example
        attribute_index = find_attribute_index(best_attribute, training_examples)
        # Find all the possible values for the attribute
        for example in training_examples:
            if example[attribute_index][1] in possible_attribute_values_list:
                continue
            else:
                possible_attribute_values_list.append(example[attribute_index][1])
        return possible_attribute_values_list

    '''
    Returns a dictionary that maps between each of 'attribute' values and its classifications data.
    The classifications data will be organized in a dictionary that will map each classification value
    to the number of its appearances. {'attribute value': {'classification value': num of appearances}...}.
    '''
    def classifications_by_attribute_values(attribute, training_examples):
        att_val_to_class_data_dict = {}
        attribute_index = find_attribute_index(attribute, training_examples)
        for example in training_examples:
            if example[attribute_index][1] in att_val_to_class_data_dict.keys():
                if example[-1][1] in att_val_to_class_data_dict[example[attribute_index][1]].keys():
                    att_val_to_class_data_dict[example[attribute_index][1]][example[-1][1]] += 1
                else:
                    att_val_to_class_data_dict[example[attribute_index][1]][example[-1][1]] = 1
            else:
                att_val_to_class_data_dict[example[attribute_index][1]] = {}
                att_val_to_class_data_dict[example[attribute_index][1]][example[-1][1]] = 1
        return att_val_to_class_data_dict

    '''
    Returns a dictionary that maps between a classification value and the number of its appearances.
    '''
    def get_classifications_data(training_examples):
        classifications_data = {}
        for example in training_examples:
            if example[-1][1] in classifications_data.keys():
                classifications_data[example[-1][1]] += 1
            else:
                classifications_data[example[-1][1]] = 1
        return classifications_data

    '''
    Entropy measures the impurity of 'training_examples'.
    '''
    def entropy(classifications_data):
        examples_num = sum(classifications_data.values())
        entropy_val = 0
        for class_appearances in classifications_data.values():
            entropy_val += (-(class_appearances/examples_num) * math.log((class_appearances/examples_num), 2))
        return entropy_val

    '''
    Returns the expected reduction in entropy due to sorting 'training_examples' on attribute 'attribute'. 
    '''
    def gain(attribute, training_examples):
        classifications_data = get_classifications_data(training_examples)
        entropy_val = entropy(classifications_data)
        att_val_to_class_data_dict = classifications_by_attribute_values(attribute, training_examples)
        examples_num = len(training_examples)
        reduction_val = 0
        for class_data_dict in att_val_to_class_data_dict.values():
            reduction_val += (sum(class_data_dict.values())/examples_num) * entropy(class_data_dict)
        return entropy_val - reduction_val

    '''
    Chooses the attribute we should check at the current node.
    '''
    def choose_attribute(attributes_list, training_examples):
        max_gain = 0
        chosen_attribute = None
        for attribute in attributes_list:
            current_attribute_gain = gain(attribute, training_examples)
            if current_attribute_gain > max_gain:
                max_gain = current_attribute_gain
                chosen_attribute = attribute
            elif current_attribute_gain == max_gain:
                if len(attributes_list) == 1:
                    max_gain = current_attribute_gain
                    chosen_attribute = attribute
        return chosen_attribute

    '''
    Returns True if all training examples have the same classification, else returns False.
    '''
    def same_classification_for_all(training_examples):
        classification = training_examples[0][-1][1]
        for example in training_examples:
            if example[-1][1] != classification:
                return False
        return True

    '''
    Creates a dictionary that maps each attribute to list of all of its possible values.
    '''
    def create_attribute_values_dict(training_examples):
        attribute_values_dict = {}
        for example in training_examples:
            for i in range(len(example)-1):
                if example[i][0] in attribute_values_dict.keys():
                    if example[i][1] not in attribute_values_dict[example[i][0]]:
                        attribute_values_dict[example[i][0]].append(example[i][1])
                else:
                    attribute_values_dict[example[i][0]] = []
                    attribute_values_dict[example[i][0]].append(example[i][1])
        return attribute_values_dict

    '''
    In order to handle cases of equality as defined in the instructions.
    '''
    def my_max(classification_dict):
        positive_expressions = ['t', 'T', 'true', 'True', 'yes', 'Yes', '1']
        negative_expressions = ['f', 'F', 'false', 'False', 'no', 'No', '0']
        max_appearances = 0
        final_classifications_list = []
        for appearances_num in classification_dict.values():
            if appearances_num > max_appearances:
                max_appearances = appearances_num
        for classification, appearances_num in classification_dict.items():
            if appearances_num == max_appearances:
                final_classifications_list.append(classification)
        if len(final_classifications_list) == 1:
            return final_classifications_list[0]
        else:
            for classification in final_classifications_list:
                if classification in positive_expressions:
                    return classification


    '''
    Returns the most common class among the examples.
    '''
    def mode(training_examples):
        classification_dict = {}
        for example in training_examples:
            if example[-1][1] in classification_dict.keys():
                classification_dict[example[-1][1]] += 1
            else:
                classification_dict[example[-1][1]] = 1
        # return max(classification_dict, key=classification_dict.get)
        return my_max(classification_dict)

    '''
    Returns all of the example's attributes.
    '''
    def get_attributes(training_examples):
        attributes_list = []
        for i in range(len(training_examples[0])-1):
            attributes_list.append(training_examples[0][i][0])
        return attributes_list

    '''
    Decision tree learning (id3) algorithm implementation.
    '''
    def decision_tree_learning(training_examples, attributes_list, default, attribute_values_dict):
        if len(training_examples) == 0:
            return Leaf(default)
        elif same_classification_for_all(training_examples):
            return Leaf(training_examples[0][-1][1])
        elif len(attributes_list) == 0:
            return Leaf(mode(training_examples))
        else:
            best_attribute = choose_attribute(attributes_list, training_examples)
            tree = Node(best_attribute)
            '''
            There are two options here:
            The first one is that in each recursive call to 'decision_tree_learning', we will iterate over the current
            optional values of 'best_attribute'. That means, its optional values from the current 'training_examples'.
            In this case we won't see classifications that does not appear in the examples of 'training_examples'.
            The second option is that in each recursive call to 'decision_tree_learning', we will iterate over all of
            the possible values of 'best_attribute'. That means, its optional values from the original 'training_examples'. 
            In this case we might see some classifications that does not appear in the examples of 'training_examples'.
            I wrote code for both of the options: the one in comment stands for the first options, and the currently
            running code stands for the second.   
            '''
            # for attribute_value in possible_attribute_values(best_attribute, training_examples):
            for attribute_value in attribute_values_dict[best_attribute]:
                '''
                if best_attribute == 'pclass' and attribute_value == 'crew':
                    print(best_attribute + "=" + attribute_value)
                '''
                relevant_training_examples_list = filter_relevant_training_examples(best_attribute, attribute_value,
                                                                                    training_examples)
                sub_attributes_list = attributes_list[:]
                sub_attributes_list.remove(best_attribute)
                subtree = decision_tree_learning(relevant_training_examples_list, sub_attributes_list,
                                                 mode(training_examples), attribute_values_dict)
                # Add a branch to 'tree' with label 'attribute_value' and subtree 'subtree'
                tree.successors[attribute_value] = subtree
            return tree

    # List of all attributes in the examples
    attributes_list = get_attributes(training_examples)
    default = mode(training_examples)

    # Create a dictionary that will map each attribute to all of its possible values
    attribute_values_dict = create_attribute_values_dict(training_examples)

    return decision_tree_learning(training_examples, attributes_list, default, attribute_values_dict)



'''
K-Nearest Neighbors algorithm implementation.
'''
def k_nearest_neighbors(data_to_be_tested, arguments_dict):

    training_data_list = arguments_dict['training data list']
    k = arguments_dict['k']

    '''
    Finds and returns the most frequent classification of the 'top_k_list' training data elements.
    This will be the classification of the tested case.
    '''
    def most_frequent_class(top_k_list):
        classification_dict = {}
        for element in top_k_list:
            if element[1] in classification_dict.keys():
                classification_dict[element[1]] += 1
            else:
                classification_dict[element[1]] = 1
        return max(classification_dict, key=classification_dict.get)

    '''
    In case that new training data for the 'top_k_list' is found, we need to go over the 'top_k_list',
    in order to reorganize it if needed.
    '''
    def bubble_down(top_k_list, from_index):
        if from_index == len(top_k_list) - 1:
            return
        for i in range(from_index, len(top_k_list) - 1):
            if top_k_list[from_index][0] < top_k_list[from_index + 1][0]:
                bubble_down(top_k_list, from_index + 1)
                top_k_list[from_index + 1][0] = top_k_list[from_index][0]
                top_k_list[from_index + 1][1] = top_k_list[from_index][1]

    '''
    Maintains 'top_k_list' to contain the training data most relevant to the current case (minimal Hamming distance).
    '''
    def maintain_top_k(top_k_list, distance_val, classification):
        for i in range(len(top_k_list)):
            if distance_val < top_k_list[i][0]:
                bubble_down(top_k_list, i)
                top_k_list[i][0] = distance_val
                top_k_list[i][1] = classification
                break

    '''
    Returns the Hamming distance (according to our relevant definition) between two attributes values strings.
    '''
    def hamming_distance(str1, str2):
        if str1 != str2:
            return 1
        return 0

    '''
    Calculates the distance between 'data_to_be_tested' and 'training_data'.
    '''
    def distance(data_to_be_tested, training_data):
        counter = 0
        for i in range(len(data_to_be_tested) - 1):
            counter += hamming_distance(data_to_be_tested[i][1], training_data[i][1])
        return counter

    # K-NN algorithm
    top_k_list = [[1000, None] for i in range(k)]
    for i in range(len(training_data_list)):
        distance_val = distance(data_to_be_tested, training_data_list[i])
        # 'training_data_list[i][-1][1]' contains the classification of the data
        maintain_top_k(top_k_list, distance_val, training_data_list[i][-1][1])
    return most_frequent_class(top_k_list)


'''
Gets a list of cases to be tested, and for each case computes its classification using the given algorithm.
Returns a list of the given cases with their classifications.
'''
def classification_by_algorithm(data_to_be_tested_list, algorithm, arguments_dict):
    classified_testing_data_list = []
    for i in range(len(data_to_be_tested_list)):
        classified_testing_data = []
        for j in range(len(data_to_be_tested_list[i]) - 1):
            classified_testing_data.append(data_to_be_tested_list[i][j])
        classification = algorithm(data_to_be_tested_list[i], arguments_dict)
        classified_testing_data.append((data_to_be_tested_list[i][-1][0], classification))
        classified_testing_data_list.append(classified_testing_data)
    return classified_testing_data_list


'''
Checks the algorithm's accuracy by comparing its classifications results to the given testing_data_list classifications.
'''
def check_algorithm_accuracy(classified_testing_data_list, given_testing_data_list):
    matching_counter = 0
    for i in range(len(classified_testing_data_list)):
        if classified_testing_data_list[i][-1][1] == given_testing_data_list[i][-1][1]:
            matching_counter += 1
    return matching_counter / len(classified_testing_data_list)


'''
Reads the data from the given file, organizes the data in a list and returns that list.
'''
def file_data_processor(file_name):
    with open(file_name) as f:
        line = f.readline().rstrip('\n')
        # The first line of the file contains the attributes
        attributes_list = line.split('\t')
        # Will hold all of the examples in the file, each organized as list of tuples [(attribute, value)...]
        file_data_list = []
        # Read the rest of the lines in the file, each line represents an example
        line = f.readline().rstrip('\n')
        while line and line is not "":
            current_example_values_list = line.split('\t')
            # Organize the example's data as list of tuples [(attribute, value)...]
            current_example_organized_data = []
            for i in range(len(attributes_list)):
                current_example_organized_data.append((attributes_list[i], current_example_values_list[i]))
            file_data_list.append(current_example_organized_data)
            line = f.readline().rstrip('\n')
    return file_data_list


# Process the training data
training_data_list = file_data_processor('train.txt')
# Process the data to be tested
testing_data_list = file_data_processor('test.txt')

# Classify the 'testing_data_list' using the dtl algorithm
decision_tree = dtl_top_level(training_data_list)
# Print the decision tree to the file 'output_tree.txt'
print_decision_tree_to_file(decision_tree)
dtl_classified_testing_data_list = classification_by_algorithm(testing_data_list, decision_tree_classifier,
                                                           {'decision tree': decision_tree})
dtl_accuracy = check_algorithm_accuracy(dtl_classified_testing_data_list, testing_data_list)

# Classify the 'testing_data_list' using the 'knn' algorithm
knn_classified_testing_data_list = classification_by_algorithm(testing_data_list, k_nearest_neighbors,
                                                           {'training data list': training_data_list, 'k': 5})
knn_accuracy = check_algorithm_accuracy(knn_classified_testing_data_list, testing_data_list)

# Classify the 'testing_data_list' using the naive bayes formula
nb_classified_testing_data_list = classification_by_algorithm(testing_data_list, naive_bayes_classifier,
                                                           {'training data list': training_data_list})
nb_accuracy = check_algorithm_accuracy(nb_classified_testing_data_list, testing_data_list)
# Print the results to the file 'output.txt'
print_results_to_file({'dtl classified testing data list': dtl_classified_testing_data_list,
                       'knn classified testing data list': knn_classified_testing_data_list,
                       'nb classified testing data list': nb_classified_testing_data_list},
                      {'dtl accuracy': dtl_accuracy, 'knn accuracy': knn_accuracy, 'nb accuracy': nb_accuracy})