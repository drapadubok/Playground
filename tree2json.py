### Code to create hierarchy for decision tree visualization

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import json

def rules(clf, features, labels, node_index=0):
    """Structure of rules in a fit decision tree classifier
    
    Parameters
    ----------
    clf : DecisionTreeClassifier
        A tree that has already been fit.

    features, labels : lists of str
        The names of the features and labels, respectively.

    """
    node = {}
    if clf.tree_.children_left[node_index] == -1:  # indicates leaf
        count_labels = zip(clf.tree_.value[node_index, 0], labels)
        node['name'] = ', '.join(('{} of {}'.format(int(count), label)
                                  for count, label in count_labels))
    else:
        feature = features[clf.tree_.feature[node_index]]
        threshold = clf.tree_.threshold[node_index]
        node['name'] = '{} > {}'.format(feature, threshold)
        left_index = clf.tree_.children_left[node_index]
        right_index = clf.tree_.children_right[node_index]
        node['children'] = [rules(clf, features, labels, right_index),
                            rules(clf, features, labels, left_index)]
    return node

####
data = load_iris()

clf = DecisionTreeClassifier(max_depth=3)
clf.fit(data.data, data.target)

rules(clf, data.feature_names, data.target_names)

r = rules(clf, data.feature_names, data.target_names)
with open(rules.json', 'w') as f:
    f.write(json.dumps(r))
