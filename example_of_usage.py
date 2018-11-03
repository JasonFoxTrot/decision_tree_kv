import decision_tree_kv
from typing import List, Dict, Tuple, Union
my_data: List[List[Union[str, int, float]]]

r: Dict[str, int]
s: str = ''
tree: decision_tree_kv.decisionnode

# we have data
# all columns are features, but last column is a target
# column:
# Site from, Country, Have read FAQ, Page count before register, Subscription
my_data = [
    ['slashdot', 'USA', 'yes', 18, 'None'],
    ['google', 'France', 'yes', 23, 'Premium'],
    ['digg', 'USA', 'yes', 24, 'Basic'],
    ['kiwitobes', 'France', 'yes', 23, 'Basic'],
    ['google', 'UK', 'no', 21, 'Premium'],
    ['(direct)', 'New Zealand', 'no', 12, 'None'],
    ['(direct)', 'UK', 'no', 21, 'Basic'],
    ['google', 'USA', 'no', 24, 'Premium'],
    ['slashdot', 'France', 'yes', 19, 'None'],
    ['digg', 'USA', 'no', 18, 'None'],
    ['google', 'UK', 'no', 18, 'None'],
    ['kiwitobes', 'UK', 'no', 19, 'None'],
    ['digg', 'New Zealand', 'yes', 12, 'Basic'],
    ['slashdot', 'UK', 'no', 21, 'None'],
    ['google', 'UK', 'yes', 18, 'Basic'],
    ['kiwitobes', 'France', 'yes', 19, 'Basic']
]


#
# building tree for categories
# last column is label (string)
# target column must be last
#

# build tree with Gini impurity as a score-function
tree = decision_tree_kv.buildtree(
    my_data,
    scoref=decision_tree_kv.giniimpurity
)

#
# print tree to console
#


decision_tree_kv.printtree(tree)
s = decision_tree_kv.tree_to_str(tree)
print(s)
# or
decision_tree_kv.printtree(tree)


#
# print tree as picture
#

# save picture to hard drive
decision_tree_kv.drawtree(tree, jpeg='treeview.jpg')
# open picture
# decision_tree_kv.showtree(tree)


#
# classification other data
#

# classification standart
# use it if you haven't None in data-set
r = decision_tree_kv.classify(['(direct)', 'USA', 'yes', 5], tree)
print(r)
r = decision_tree_kv.classify(['google', 'USA', 'yes', 8], tree)
print(r)
r = decision_tree_kv.classify(['google', 'France', 'no', 15], tree)
print(r)

# classification modify
# use it if you have None in data-set
r = decision_tree_kv.mdclassify(['(direct)', 'USA', 'yes', 5], tree)
print(r)
r = decision_tree_kv.mdclassify(['google', None, 'yes', None], tree)
print(r)
r = decision_tree_kv.mdclassify(['google', 'France', None, None], tree)
print(r)


#
# building tree for numbers
# last column is a price (integer)
#

# columns:
# "carat","cut","color","clarity","depth","table","x","y","z","price2"
diamonds_data: List[List[Union[float, int, str]]]
diamonds_data = [
    [0.4, "Premium", "E", "SI2", 62.9, 59, 4.69, 4.66, 2.94, 855],
    [0.31, "Ideal", "H", "VVS2", 61.3, 56, 4.37, 4.4, 2.69, 778],
    [0.31, "Very Good", "I", "SI1", 61.4, 61, 4.34, 4.39, 2.68, 408],
    [0.51, "Ideal", "E", "SI1", 62.7, 55, 5.1, 5.07, 3.19, 1546],
    [0.7, "Good", "D", "VS1", 60.4, 63, 5.68, 5.74, 3.45, 3139],
    [0.41, "Ideal", "D", "VVS1", 60.6, 57, 4.82, 4.79, 2.91, 1582],
    [0.4, "Very Good", "D", "VS2", 60.8, 59, 4.71, 4.77, 2.88, 868],
    [0.36, "Premium", "E", "VS2", 62.5, 58, 4.57, 4.52, 2.84, 878],
    [0.3, "Very Good", "G", "VVS2", 63.1, 56, 4.23, 4.2, 2.66, 878],
    [1.22, "Very Good", "H", "SI1", 62.4, 57, 6.78, 6.82, 4.24, 6323],
    [1.52, "Ideal", "I", "VVS1", 61.9, 56, 7.34, 7.37, 4.55, 10968],
    [1.7, "Premium", "I", "VS2", 61.7, 59, 7.63, 7.68, 4.72, 11257],
    [0.3, "Premium", "G", "VVS1", 60.7, 58, 4.36, 4.34, 2.64, 1013],
    [0.41, "Good", "G", "SI2", 63.7, 55, 4.7, 4.75, 3.01, 570],
    [0.56, "Very Good", "F", "SI1", 63.9, 59, 5.16, 5.22, 3.32, 1425],
    [0.33, "Ideal", "F", "VVS1", 61.1, 56, 4.45, 4.48, 2.73, 893],
    [0.59, "Ideal", "H", "VS2", 62.2, 59, 5.38, 5.34, 3.33, 1648],
    [0.51, "Very Good", "I", "SI2", 63.3, 58, 5.09, 5.02, 3.2, 995],
    [0.73, "Premium", "E", "VS2", 61.6, 59, 5.77, 5.73, 3.54, 2821],
    [0.29, "Ideal", "G", "VVS1", 62.4, 56, 4.22, 4.24, 2.64, 673],
    [1, "Fair", "G", "SI1", 68.2, 60, 6.02, 5.94, 4.08, 3833]
]

# building tree
diamonds_tree: decision_tree_kv.decisionnode = decision_tree_kv.buildtree(
    diamonds_data,
    scoref=decision_tree_kv.variance
)
# decision_tree_kv.drawtree(diamonds_tree,'diamonds_tree.jpg')
# decision_tree_kv.printtree(diamonds_tree)
# decision_tree_kv.showtree(tree)
