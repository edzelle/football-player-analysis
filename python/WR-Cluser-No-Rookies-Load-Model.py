import tensorflow as tf
import psycopg2
import pandas as pd
import json
import os
import itertools
import numpy as np
import matplotlib.pyplot as plt  # for plotting
from matplotlib import animation  # animate 3D plots
from mpl_toolkits.mplot3d import Axes3D  # 3D plots
from keras import regularizers

# Scikit learn
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn import manifold
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.cluster import AffinityPropagation, MeanShift, HDBSCAN
import scipy.cluster.hierarchy as hcluster


# TensorFlow and Keras
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import backend as K
import random
import math
from numpy import unique



def getdbconnectionfromconfiguration(config):
    """ Connect to the PostgreSQL database server """
    if os.path.exists(config):
        with open(config, 'r', encoding="utf-8-sig") as f:
            d = json.load(f)

            dbconfig = d['Database']
            host = dbconfig['Host']
            database = dbconfig['Database']
            user = dbconfig['User']
            pw = dbconfig['Password']
            port = dbconfig['Port']

            connectionString = "Server={0}; User Id={1}; Database={2}; Port={3}; Password={4};Include Error Detail=true;".format(
                host, database, user, port, pw)
            with psycopg2.connect(
                    host=host,
                    database=database,
                    user=user,
                    password=pw,
                    port=port
            ) as cnn:
                print('Connected to the PostgreSQL server.')
                return cnn

def zeropadrecordsalginbyage(minage, maxage, records):
    data = []

    currentage = minage
    rowindex = 0
    while (currentage <= maxage):
        if rowindex < len(records):
            currentyear = records.iloc[rowindex,4]
            if (currentage != currentyear):
                data.extend(np.zeros((1, records.shape[1])))
            else:
                data.append(records.iloc[rowindex,].values)
                rowindex += 1
        else:
            data.extend(np.zeros((1, records.shape[1])))
        currentage += 1
    return data

def train_test_split_as_tensor(data, labels, training_size):
    random_split_index = random.randint(0, data.shape[0]-1)
    num_records_in_training = math.floor(data.shape[0] * training_size)
    if (random_split_index + num_records_in_training) > (data.shape[0] - 1):
        train_data = data[random_split_index:]
        train_labels = labels[random_split_index:]
        remainder = num_records_in_training - train_data.shape[0]
        if remainder > 0:
            train_data = tf.concat([train_data, data[:remainder]], 0) ## train data should have shape (num_records_in_training, None, feature_len)
            train_labels = tf.concat([train_labels, labels[:remainder]], 0)
            test_data = data[remainder: random_split_index]
            test_labels = tf.convert_to_tensor(labels[remainder: random_split_index])
    else:
        train_data = data[random_split_index: random_split_index + num_records_in_training]
        train_labels = labels[random_split_index: random_split_index + num_records_in_training]
        test_data = tf.concat([data[random_split_index + num_records_in_training:],data[:random_split_index-1]], 0)
        test_labels = tf.concat([labels[random_split_index + num_records_in_training:], labels[:random_split_index-1]], 0)

    return train_data, test_data, train_labels, test_labels

def repeat_vector(args):
        layer_to_repeat = args[0]
        sequence_layer = args[1]
        return layers.RepeatVector(K.shape(sequence_layer)[1])(layer_to_repeat)

def repeat_vector2(args):
    sequential_input = args[1]
    to_be_repeated = K.expand_dims(args[0],axis=1)

    # set the one matrix to shape [ batch_size , sequence_length_based on input, 1]
    one_matrix = K.ones_like(sequential_input[:,:,:1])
    
    # do a mat mul
    return K.batch_dot(one_matrix,to_be_repeated)

def write_clusters_and_players_to_file(player_names, player_ids, clusters, clusterType):
    players_and_clusters = np.transpose([player_names, player_ids, clusters])

    with open('WR-'+clusterType+'-cluster-output-no-rookies-encoded.csv', 'w') as myfile:
        for name, player_id, cluster in players_and_clusters:
            myfile.write(player_id +"," + name +"," + cluster + ",\n")

def compute_affinity_clusters(encoded_items):
    Sum_of_squared_distances = []
    x = np.linspace(.5,1,50, endpoint=False)
    for i in x:
        model = AffinityPropagation(damping=i)
        model.fit(encoded_items)
        result = model.predict(encoded_items)
        if len(set(result)) > 1:
            Sum_of_squared_distances.append(calinski_harabasz_score(encoded_items, result))
        else: Sum_of_squared_distances.append(100000000)

    min_threshold = x[np.argmin(Sum_of_squared_distances)]
    model = AffinityPropagation(damping=min_threshold)
    clusters = model.fit_predict(encoded_items)
    return (Sum_of_squared_distances[np.argmin(Sum_of_squared_distances)], clusters)

def compute_meanshift_clusters(encoded_items):
    Sum_of_squared_distances = []
    x = np.linspace(.01,5,50, endpoint=False)
    for bandwidth in x:
        mean_model = MeanShift(bandwidth=bandwidth)
        # assign each data point to a cluster
        clusters = mean_model.fit_predict(encoded_items)
        if len(set(clusters)) > 1:
            Sum_of_squared_distances.append(silhouette_score(encoded_items, clusters))
    
    min_bandwidth = x[np.argmin(Sum_of_squared_distances)]
    model = MeanShift(bandwidth=min_bandwidth)
    clusters = model.fit_predict(encoded_items)
    return (Sum_of_squared_distances[np.argmin(Sum_of_squared_distances)], clusters)

def compute_fcluster_clusters(encoded_items):
    Sum_of_squared_distances = []
    x = np.linspace(.05,1,50)
    for k in x:
        clusters = hcluster.fclusterdata(encoded_items, k, criterion="distance")
        if len(set(clusters)) > 1:
            Sum_of_squared_distances.append(calinski_harabasz_score(encoded_items, clusters))
        else:
            break

    ## Min Value
    min_threshold = x[np.argmin(Sum_of_squared_distances)]
    clusters = hcluster.fclusterdata(encoded_items, min_threshold, criterion="distance")
    return (Sum_of_squared_distances[np.argmin(Sum_of_squared_distances)], clusters)

def compute_kmeans_clusters(encoded_items):
    Sum_of_squared_distances_s = []
    Sum_of_squared_distances_c = []

    a = range(2,50)
    for k in a:
        km = KMeans(init='k-means++', n_clusters=k, n_init=10, random_state=16)
        clusters = km.fit_predict(encoded_items)
        if len(set(clusters > 1)):
            Sum_of_squared_distances_s.append(silhouette_score(encoded_items, clusters))
            Sum_of_squared_distances_c.append(calinski_harabasz_score(encoded_items, clusters))

    min_threshold = a[np.argmin(Sum_of_squared_distances_s)]
    km = KMeans(init='k-means++', n_clusters=min_threshold, n_init=10)
    clusters = km.fit_predict(encoded_items)
    return (Sum_of_squared_distances_s[np.argmin(Sum_of_squared_distances_s)], clusters)

def compute_hdbscan_clusters(encoded_items):
    Sum_of_squared_distances = []
    a = range(2,30)
    for k in a:
        hdb = HDBSCAN(min_cluster_size=k)
        clusters = hdb.fit_predict(encoded_items)
        Sum_of_squared_distances.append(silhouette_score(encoded_items, clusters))

    min_threshold = a[np.argmax(Sum_of_squared_distances)]
    hdb = HDBSCAN(min_cluster_size=4)
    clusters = hdb.fit_predict(encoded_items)
    return (Sum_of_squared_distances[np.argmin(Sum_of_squared_distances)], clusters)


def performClusteringForModel(encoding_dim, normalized_input):
    encoder_model = tf.keras.models.load_model('python/models/{0}-encoder-WR-cluster-output-no-rookies-1990-td.keras'.format(encoding_dim), safe_mode=False)
    encoded_items = encoder_model(normalized_input)
    ##min_affinity = compute_affinity_clusters(encoded_items)
    ##min_hdbscan = compute_hdbscan_clusters(encoded_items)
    min_kmeans = compute_kmeans_clusters(encoded_items)

    if encoding_dim > 2:
        min_meanshift = compute_meanshift_clusters(encoded_items)
       ## min_fcluster = compute_fcluster_clusters(encoded_items)
        min_fcluster = []
        min_affinity =[]
        min_hdbscan = []
        return (min_affinity, min_meanshift, min_fcluster, min_kmeans, min_hdbscan)
    else:
        return (min_kmeans)

cur_path = os.path.abspath(os.path.dirname(__file__))
new_path = os.path.join(cur_path, "..\\FootballDataReader.Host\\appsettings.json")

conn = getdbconnectionfromconfiguration(new_path)

queryString = """select
p.name,
p.height,
p.weight,
ps.player_id,
ps.player_age,
ps.games_played,
ps.games_started,
prs.targets,
prs.receptions,
prs.catch_per,
prs.yards,
prs.yards_per_rec,
prs.tds,
prs.first_downs_receiving,
prs.receiving_success_rate,
prs.longest_reception,
prs.yards_per_target,
prs.receptions_per_game,
prs.yards_per_game,
prs.fumbles,
prs.overall_usage,
prs.pass_usage,
prs.rush_usage,
prs.first_down_usage,
prs.second_down_usage,
prs.third_down_usage,
prs.standard_downs_usage,
prs.passing_downs_usage,
prs.average_ppa_all,
prs.average_ppa_pass,
prs.average_ppa_rush,
prs.average_ppa_first_down,
prs.average_ppa_second_down,
prs.average_ppa_third_down,
prs.average_ppa_standard_downs,
prs.average_ppa_passing_down,
prs.total_ppa_all,
prs.total_ppa_pass,
prs.total_ppa_rush,
prs.total_ppa_first_down,
prs.total_ppa_second_down,
prs.total_ppa_third_down,
prs.total_ppa_standard_downs,
prs.total_ppa_passing_down
from football.player_season ps,
football.players p,
football.player_receiving_stats prs
where p.id = ps.player_id
and p.position = 'WR'
and prs.player_id = ps.player_id
and prs.year = ps.year
and ps.is_college_season = false
and p.id <> 1402
order by ps.player_id, ps.year;"""

df = pd.read_sql_query(queryString, con=conn)
df = df.fillna(value=0)

minage = df.iloc[:,5].min()
maxage = df.iloc[:,5].max()

groupeddf = df.groupby(df.columns[0])

full_data_as_list = df.values.tolist()

zero_padded_input_data = []
player_names = []
player_ids = []
plain_input_data = []
ragged_tensor_splits = []
for k, g in groupeddf:
    g.pop('name')
    ragged_tensor_splits.append(g.shape[0])
    plain_input_data.extend(g.values.tolist())
    zero_padded_input_data.append(zeropadrecordsalginbyage(minage, maxage, g))
    player_id = g['player_id'].tolist()[0]
    player_names.append(k)
    player_ids.append(player_id)


ragged_tensor = tf.RaggedTensor.from_row_lengths(values=plain_input_data, row_lengths=ragged_tensor_splits)

ragged_tensor = ragged_tensor.to_tensor()

ragged_tensor = tf.math.l2_normalize(ragged_tensor, axis = -1)

## normalized_input = tf.RaggedTensor.from_row_lengths(values=normalized_input, row_lengths=ragged_tensor_splits)

nonZeroRows = tf.reduce_sum(tf.abs(ragged_tensor), 2) > 0 
n_timesteps = ragged_tensor.shape.as_list()[1] # should be 11

ragged_tensor = tf.ragged.boolean_mask(ragged_tensor, nonZeroRows)

full_dim = (ragged_tensor.shape.as_list()[1], ragged_tensor.shape.as_list()[2],)

n_features = ragged_tensor.shape.as_list()[2] # 44

normalized_input = ragged_tensor

affinity_results = {}
meanshift_results = {}
fcluster_results = {}
kmeans_results = {}
hdbscan_results = {}
for i in range (8,9):
    cluster_results = performClusteringForModel(i, normalized_input)  
    ##affinity_results[i] = cluster_results[0]
    if i > 1:  
        meanshift_results[i] = cluster_results[1]  
    ##    fcluster_results[i] = cluster_results[2]  
        kmeans_results [i] = cluster_results[3]
     ##   hdbscan_results[i] = cluster_results[1]

##min_affinity = min(affinity_results.items(), key=lambda x: x[1][0]) 
min_meanshift = min(meanshift_results.items(), key=lambda x: x[1][0]) 
##min_fcluster = min(fcluster_results.items(), key=lambda x: x[1][0]) 
min_kmeans = min(kmeans_results.items(), key=lambda x: x[1][0]) 
##min_hdbscan = max(hdbscan_results.items(), key=lambda x: x[1][0]) 



##write_clusters_and_players_to_file(player_names, player_ids, min_affinity[1][1], '1990-affinity-{0}-dims'.format(min_affinity[0]))
##write_clusters_and_players_to_file(player_names, player_ids, min_hdbscan[1][1], '1900-hdbscan-{0}-dims'.format(min_hdbscan[0]))
write_clusters_and_players_to_file(player_names, player_ids, min_meanshift[1][1], '1900-meanshift-{0}-dims'.format(min_meanshift[0]))
##write_clusters_and_players_to_file(player_names, player_ids, min_fcluster[1][1], '1900-fcluster-{0}-dims'.format(min_fcluster[0]))
write_clusters_and_players_to_file(player_names, player_ids, min_kmeans[1][1], '1990-kmeans-{0}-dims'.format(min_kmeans[0]))