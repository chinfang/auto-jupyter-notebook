import nbformat as nbf
import warnings
import os
import sys

warnings.simplefilter(action="ignore", category=RuntimeWarning)

nb = nbf.v4.new_notebook()

text_title = """\
# Automatic Jupyter Notebook for OpenML dataset"""

text_model = """\
Build Random Forest model from the dataset and compute important features. """

text_plot = """\
Plot Top-20 important features for the dataset. """

text_run = """\
Choose desired dataset and generate the most important plot. """

text_landmarkers = """\
The following Landmarking meta-features were calculated and stored in MongoDB: (Matthias Reif et al. 2012, Abdelmessih et al. 2010)

The accuracy values of the following simple learners are used: Naive Bayes, Linear Discriminant Analysis, One-Nearest Neighbor, Decision Node, Random Node.

- **Naive Bayes Learner** is a probabilistic classifier, based on Bayes’ Theorem:
$$ p(X|Y) = \\frac{p(Y|X) \cdot p(X)}{p(Y)} $$

    where p(X) is the prior probability and p(X|Y) is the posterior probability. It is called naive, because it
    assumes independence of all attributes to each other.
- **Linear Discriminant Learner** is a type of discriminant analysis, which is understood as the grouping and separation of categories according to specific features. Linear discriminant is basically finding a linear combination of features that separates the classes best. The resulting separation model is a line, a plane, or a hyperplane, depending on the number of features combined. 

- **One Nearest Neighbor Learner** is a classifier based on instance-based learning. A test point is assigned to the class of the nearest point within the training set. 

- **Decision Node Learner** is a classifier based on the information gain of attributes. The information gain indicates how informative an attribute is with respect to the classification task using its entropy. The higher the variability of the attribute values, the higher its information gain. This learner selects the attribute with the highest information gain. Then, it creates a single node decision tree consisting of the chosen attribute as a split node. 

- **Randomly Chosen Node Learner** is a classifier that results also in a single decision node, based on a randomly chosen attribute. """

text_distances = """\
The similarity between datasets were computed based on the distance function and stored in MongoDB: (Todorovski et al. 2000)
    $$ dist(d_i, d_j) = \sum_x{\\frac{|v_{x, d_i}-v_{x, d_j}|}{max_{k \\neq i}(v_x, d_k)-min_{k \\neq i}(v_x, d_k)}}$$

where $d_i$ and $d_j$ are datasets, and $v_{x, d_i}$ is the value of meta-attribute $x$ for dataset $d_i$. The distance is divided by the range to normalize the values, so that all meta-attributes have the same range of values. """

code_library = """\
%matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter

plt.rcParams['figure.dpi']= 120
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8 

from preamble import *
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from pymongo import MongoClient"""

code_forest = """\
def build_forest(dataset):    
    data = oml.datasets.get_dataset(dataset) 
    X, y, features = data.get_data(target=data.default_target_attribute, return_attribute_names=True); 
    forest = Pipeline([('Imputer', preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)),
                       ('classifiers', RandomForestClassifier(n_estimators=100, random_state=0))])
    forest.fit(X,y)
    
    importances = forest.steps[1][1].feature_importances_
    indices = np.argsort(importances)[::-1]
    return data.name, features, importances, indices """

code_feature_plot = """\
def plot_feature_importances(features, importances, indices):
    a = 0.8
    f_sub = []
    max_features = 20

    for f in range(min(len(features), max_features)): 
            f_sub.append(f)

    # Create a figure of given size
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    # Set title
    ttl = dataset_name

    df = pd.DataFrame(importances[indices[f_sub]][::-1])
    df.plot(kind='barh', ax=ax, alpha=a, legend=False, edgecolor='w', 
            title=ttl, color = [plt.cm.viridis(np.arange(len(df))*10)])

    # Remove grid lines and plot frame
    ax.grid(False)
    ax.set_frame_on(False)

    # Customize title
    ax.set_title(ax.get_title(), fontsize=14, alpha=a, ha='left', x=0, y=1.0)
    plt.subplots_adjust(top=0.9)

    # Customize x tick lables
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.locator_params(axis='x', tight=True, nbins=5)

    # Customize y tick labels
    yticks = np.array(features)[indices[f_sub]][::-1]
    ax.set_yticklabels(yticks, fontsize=8, alpha=a)
    ax.yaxis.set_tick_params(pad=2)
    ax.yaxis.set_ticks_position('none')  
    ax.set_ylim(ax.get_ylim()[0]-0.5, ax.get_ylim()[1]+0.5) 

    # Set x axis text
    xlab = 'Feature importance'
    ax.set_xlabel(xlab, fontsize=10, alpha=a)
    ax.xaxis.set_label_coords(0.5, -0.1)

    # Set y axis text
    ylab = 'Feature'
    ax.set_ylabel(ylab, fontsize=10, alpha=a)
    plt.show() """


code_get_landmarkers = """\
def connet_mongoclient(host):
    client = MongoClient('localhost', 27017)
    db = client.landmarkers
    return db
    
def get_landmarkers_from_db():
    db = connet_mongoclient('109.238.10.185')
    collection = db.landmarkers2
    df = pd.DataFrame(list(collection.find()))
    
    landmarkers = pd.DataFrame(df['score'].values.tolist())
    datasetID = df['dataset'].astype(int)
    datasets = oml.datasets.get_datasets(datasetID)
    return df, landmarkers, datasetID, datasets """

code_get_distances = """\
def get_distance_from_db():
    db = connet_mongoclient('109.238.10.185')
    collection = db.distance
    df = pd.DataFrame(list(collection.find()))
    distance = list(df['distance'].values.flatten())
    return distance """


code_compute_similar_datasets = """\
def compute_similar_datasets(dataset):
    n = 30
    dataset_index = df.index[datasetID == dataset][0]
    similar_dist = distance[:][dataset_index]
    similar_index = np.argsort(similar_dist)[:n]
    return similar_index """

code_get_datasets_name = """\
def get_datasets_name(datasets, similar_index):
    n = 30
    datasets_name = []

    for i in similar_index:
        datasets_name.append(datasets[i].name)    
    return datasets_name """

code_run = """\
dataset_name, features, importances, indices = build_forest(dataset)
plot_feature_importances(features, importances, indices)"""

code_landmarkers_plot = """\
sns.set(style="whitegrid", font_scale=0.75)
f, ax = plt.subplots(figsize=(8, 4))

df, landmarkers, datasetID, datasets = get_landmarkers_from_db()
landmarkers.columns = ['One-Nearest Neighbor', 'Linear Discriminant Analysis', 'Gaussian Naive Bayes', 
                       'Decision Node', 'Random Node']

distance = np.squeeze(get_distance_from_db())
similar_index = compute_similar_datasets(dataset)
sns.violinplot(data=landmarkers.iloc[similar_index], palette="Set3", bw=.2, cut=1, linewidth=1)
sns.despine(left=True, bottom=True) """

code_similarity_plot = """\
datasets_name = get_datasets_name(datasets, similar_index)
sns.set(style="white")
corr = pd.DataFrame(distance[similar_index[:, None], similar_index], 
                    index = datasets_name, columns = datasets_name)

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(corr, mask=mask, cmap = "BuPu_r", vmax= 1,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})"""



def main():
    # print command line arguments
    for dataset in sys.argv[1:]:
        print("Generating jupyter notebook for dataset "+str(dataset)+"...")
        generate_jnb(dataset)

def generate_jnb(dataset):
	nb['cells'] = [nbf.v4.new_markdown_cell(text_title),
	               nbf.v4.new_code_cell(code_library),
	               nbf.v4.new_markdown_cell(text_model),
	               nbf.v4.new_code_cell(code_forest),
	               nbf.v4.new_markdown_cell(text_plot),
	               nbf.v4.new_code_cell(code_feature_plot),
	               nbf.v4.new_markdown_cell(text_run),
	               nbf.v4.new_code_cell("dataset ="+ str(dataset)),
	               nbf.v4.new_code_cell(code_run),
	               nbf.v4.new_markdown_cell(text_landmarkers),
	               nbf.v4.new_code_cell(code_get_landmarkers),
	               nbf.v4.new_markdown_cell(text_distances),
	               nbf.v4.new_code_cell(code_get_distances),
	               nbf.v4.new_code_cell(code_compute_similar_datasets),
	               nbf.v4.new_code_cell(code_get_datasets_name),
	               nbf.v4.new_code_cell(code_landmarkers_plot),
	               nbf.v4.new_code_cell(code_similarity_plot)]

	fname = str(dataset)+'.ipynb'

	with open(fname, 'w') as f:
	    nbf.write(nb, f)


	os.system("jupyter nbconvert --execute --inplace %s"%(fname))

if __name__ == "__main__":
    main()

