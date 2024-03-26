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

def create_cluster_prob_plot(name, labels, y_pred, labels_with_players, cluster_type):
    fig, (ax, ax_table) = plt.subplots(nrows=2, gridspec_kw=dict(height_ratios=[5,1]), figsize=(20, 8))

    ax_table.axis('off')
    ax.bar(labels, y_pred[i], color ='maroon', 
            width = 0.4)
    ax.set_xlabel("Cluster Projection")
    ax.set_xticks(labels)
    ax.margins(x=0.007)
    ax.set_ylabel("Projected Cluster Probability")
    ax.set_ylabel("Probability")

    ax_table = plt.table(cellText=labels_with_players,
                    loc='bottom',
                    cellLoc = 'center',
                    rowLoc = 'center')
    plt.subplots_adjust(bottom=0.45)
    for k, cell in ax_table._cells.items():
        cell.get_text().set_rotation(90)
        cell.set_height(3)
    plt.text(-0.02, -3.9,'WRs in Cluster', rotation=90)

    ax.set_title("{0} {1} Cluster Probabily Projection".format(name, cluster_type))
    plt.savefig('Predictions/{0}_{1}_Projection_v3.png'.format(name, cluster_type), bbox_inches='tight')
    plt.close()

cur_path = os.path.abspath(os.path.dirname(__file__))
new_path = os.path.join(cur_path, "..\\FootballDataReader.Host\\appsettings.json")
cluster_type = 'kmeans'
conn = getdbconnectionfromconfiguration(new_path)

query = """
    select
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
	football.player_receiving_stats prs,
	(select p.id from football.players p,
	football.player_season ps,
	football.player_receiving_stats prs
	where p.year_turned_pro is null 
	and p.position = 'WR'
	and p.id = ps.player_id
	and ps.year = to_date('01012023','mmddyyyy')
	and ps.year = prs.year
	and ps.player_id = prs.player_id
	and ps.is_college_season = true) rookies
	where p.id = ps.player_id
	and p.position = 'WR'
	and prs.player_id = ps.player_id
	and prs.year = ps.year
	and ps.is_college_season = true
	and rookies.id = p.id
	order by ps.player_id, ps.year;
"""

jjquery = """
	select
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
	and ps.is_college_season = true
	and (p.id = 1282 or p.id = 1460 or p.id = 1182)
"""

kmeans_player_comps_query = """
WITH  TopComp  AS (
  SELECT name, wr_cluster_label_kmeans, ROW_NUMBER()  OVER (PARTITION BY  wr_cluster_label_kmeans  ORDER BY  yards DESC)  AS  row_num
  FROM
    football.players p,
	(select ps.player_id, sum(yards) yards from football.player_receiving_stats prs,
	 football.player_season ps where prs.year = ps.year
	 and prs.player_id = ps.player_id
	 and ps.is_college_season is not true
	group by ps.player_id order by yards desc) ps
	Where p.id = ps.player_id
	and p.position = 'WR'
	and p.wr_cluster_label_kmeans is not null
),
BottomComp  AS (
  SELECT name, wr_cluster_label_kmeans, ROW_NUMBER()  OVER (PARTITION BY  wr_cluster_label_kmeans  ORDER BY  yards asc)  AS  row_num
  FROM
    football.players p,
	(select ps.player_id, sum(yards) yards from football.player_receiving_stats prs,
	 football.player_season ps where prs.year = ps.year
	 and prs.player_id = ps.player_id
	 and ps.is_college_season is not true
	group by ps.player_id order by yards asc) ps
	Where p.id = ps.player_id
	and p.position = 'WR'
	and p.wr_cluster_label_kmeans is not null
)
SELECT *, 1 t FROM TopComp  WHERE row_num <= 1
Union
Select * , 2 t FROM BottomComp WHERE row_num <=1
order by wr_cluster_label_kmeans, t asc;
"""

player_comps_query = ''
encoding_dims = 16
if cluster_type == 'kmeans':
	player_comps_query = kmeans_player_comps_query
    
player_comps = pd.read_sql_query(player_comps_query, con=conn)
df = pd.read_sql_query(query, con=conn)
df = df.fillna(value=0)

groupeddf = df.groupby(df.columns[0])

player_names = []
plain_input_data = []
ragged_tensor_splits = []
for k, g in groupeddf:
    g.pop('name')
    ragged_tensor_splits.append(g.shape[0])
    plain_input_data.extend(g.values.tolist())
    player_names.append(k)

grouped_players_by_label = player_comps.groupby(player_comps.columns[1])
labels_with_players = []
for k, g in grouped_players_by_label:
    labels_with_players.append(g.name.tolist())

labels_with_players = np.transpose(np.array(labels_with_players))

ragged_tensor = tf.RaggedTensor.from_row_lengths(values=plain_input_data, row_lengths=ragged_tensor_splits)

ragged_tensor = ragged_tensor.to_tensor()

classifier = tf.keras.models.load_model('python/models/classifiers/WR-rookie-{0}-classifier-{1}-dims.keras'.format(cluster_type, encoding_dims), safe_mode=False)

y_pred = classifier.predict(ragged_tensor)
y_pred_classes = tf.argmax(y_pred, axis=1)

labels = np.linspace(0, len(y_pred[0])-1, len(y_pred[0]))

for i in range(0, y_pred.shape[0]):    
    create_cluster_prob_plot(player_names[i], labels, y_pred, labels_with_players, cluster_type)