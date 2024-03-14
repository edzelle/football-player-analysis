import tensorflow as tf
import psycopg2
import pandas as pd
import json
import os
import numpy as np
import matplotlib.pyplot as plt  # for plotting
from matplotlib import animation  # animate 3D plots
from mpl_toolkits.mplot3d import Axes3D  # 3D plots
from keras import regularizers
import seaborn as sns

# Scikit learn
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn import manifold

# TensorFlow and Keras
import tensorflow as tf
from tensorflow import keras
from keras import layers, Sequential
from keras import backend as K
import random

from imblearn.over_sampling import SMOTE

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


            with psycopg2.connect(
                    host=host,
                    database=database,
                    user=user,
                    password=pw,
                    port=port
            ) as cnn:
                print('Connected to the PostgreSQL server.')
                return cnn
            
def train_test_split_as_tensor_balance_category(data, labels, training_size, useage_count):
	train_data = []
	test_data = []
	train_labels = []
	test_labels = []
	train_splits = []
	test_splits = []

	for label_usage in useage_count:
		label = label_usage[0]
		indicies_with_label = [i for i, e in enumerate(labels) if e == label]
        
		train_num = round(len(indicies_with_label) * training_size)
		if (len(indicies_with_label) < 2):
			continue
		train_index = random.sample(indicies_with_label, 1)
		indicies_with_label.pop(indicies_with_label.index(train_index[0]))

		train_data.extend(data[train_index])  
		train_splits.append(data[train_index].shape[0])
		train_labels.append(labels[train_index[0]])
          
		test_index = random.sample(indicies_with_label, 1)
		indicies_with_label.pop(indicies_with_label.index(test_index[0]))
          
		test_data.extend(data[test_index])
		test_splits.append(data[test_index].shape[0])  
		test_labels.append(labels[test_index[0]])

		training_indicies = random.sample(indicies_with_label, min(train_num, len(indicies_with_label)))
		testing_indicies = [x for x in indicies_with_label if x not in training_indicies]
            
		for index in training_indicies:
			train_data.extend(data[index])  
			train_splits.append(data[index].shape[0])  
			train_labels.append(labels[index])
                  
		for index in testing_indicies:
			test_data.extend(data[index])  
			test_splits.append(data[index].shape[0])  
			test_labels.append([index])
	
	train_data = tf.RaggedTensor.from_row_lengths(values=train_data, row_lengths=train_splits)
	test_data = tf.RaggedTensor.from_row_lengths(values=test_data, row_lengths=test_splits)

	return train_data, test_data, train_labels, test_labels

def balance_data_with_oversampling(label_count, data, labels):
	max_label_count = label_count[:,1].max()
	##for i in 
	x = 0
	return data, labels

def train_test_split_as_tensor_balance_category_v2(data, labels, training_size, useage_count, player_names, label_count):
	
	##data, labels = balance_data_with_oversampling(label_count, data, labels)
	
	train_data = []
	test_data = []
	train_labels = []
	test_labels = []
	train_splits = []
	test_splits = []
	player_names_train = []
	player_names_test = []

	max_label_count = label_count[:,1].max()

	for label_usage in useage_count:
		label = label_usage[0]
		indicies_with_label = [i for i, e in enumerate(labels) if e == label]
        

		train_num = round((max_label_count * 0.2) * training_size)
		if (len(indicies_with_label) < 2):
			continue
		
		train_index = random.sample(indicies_with_label, 1)
		indicies_with_label.pop(indicies_with_label.index(train_index[0]))

		train_data.extend(data[train_index])  
		train_splits.append(data[train_index].shape[0])
		train_labels.append(labels[train_index[0]])
          
		test_index = random.sample(indicies_with_label, 1)
		indicies_with_label.pop(indicies_with_label.index(test_index[0]))
          
		test_data.extend(data[test_index])
		test_splits.append(data[test_index].shape[0])  
		test_labels.append(labels[test_index[0]])
		player_names_test.append(player_names[test_index[0]])


		training_indicies = random.sample(indicies_with_label, min(train_num, len(indicies_with_label)))
		testing_indicies = [x for x in indicies_with_label if x not in training_indicies]
            
		for index in training_indicies:
			train_data.extend(data[index])  
			train_splits.append(data[index].shape[0])  
			train_labels.append(labels[index])
			player_names_test.append(player_names[index])
                  
		for index in testing_indicies:
			test_data.extend(data[index])  
			test_splits.append(data[index].shape[0])  
			test_labels.append(labels[index])
			player_names_test.append(player_names[index])
	
	train_data = tf.RaggedTensor.from_row_lengths(values=train_data, row_lengths=train_splits)
	test_data = tf.RaggedTensor.from_row_lengths(values=test_data, row_lengths=test_splits)
	## one hot encodes labels
	train_labels = pd.get_dummies(train_labels).astype('float32').values 
	test_labels = pd.get_dummies(test_labels).astype('float32').values 

	return train_data, test_data, train_labels, test_labels

cur_path = os.path.abspath(os.path.dirname(__file__))
new_path = os.path.join(cur_path, "..\\FootballDataReader.Host\\appsettings.json")

conn = getdbconnectionfromconfiguration(new_path)

meanshift_query_string ="""
select
p.name,
clust.wr_cluster_label_meanshift wr_cluster_label,
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
(select p.id, p.wr_cluster_label_meanshift from (select count(*), wr_cluster_label_meanshift
		from football.players where position = 'WR'
		group by wr_cluster_label_meanshift
		order by count desc) c,
	football.players p 
	where c.count > 1
	and c.wr_cluster_label_meanshift = p.wr_cluster_label_meanshift
	union
	select p.id, 500 wr_cluster_label from (select count(*), wr_cluster_label_meanshift
	from football.players where position = 'WR'
	group by wr_cluster_label_meanshift
	order by count desc) c,
	football.players p
	where c.count = 1
	and p.wr_cluster_label_meanshift = c.wr_cluster_label_meanshift) clust
where p.id = ps.player_id
and p.position = 'WR'
and prs.player_id = ps.player_id
and prs.year = ps.year
and ps.is_college_season = true
and p.id <> 1402
and p.id = clust.id
order by ps.player_id, ps.year;
"""

kmeans_query_string ="""
select
p.name,
clust.wr_cluster_label_kmeans wr_cluster_label,
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
(select p.id, p.wr_cluster_label_kmeans from (select count(*), wr_cluster_label_kmeans
		from football.players where position = 'WR'
		group by wr_cluster_label_kmeans
		order by count desc) c,
	football.players p 
	where c.count > 1
	and c.wr_cluster_label_kmeans = p.wr_cluster_label_kmeans
	union
	select p.id, 500 wr_cluster_label from (select count(*), wr_cluster_label_kmeans
	from football.players where position = 'WR'
	group by wr_cluster_label_kmeans
	order by count desc) c,
	football.players p
	where c.count = 1
	and p.wr_cluster_label_kmeans = c.wr_cluster_label_kmeans) clust
where p.id = ps.player_id
and p.position = 'WR'
and prs.player_id = ps.player_id
and prs.year = ps.year
and ps.is_college_season = true
and p.id <> 1402
and p.id = clust.id
order by ps.player_id, ps.year;
"""


affinity_query_string = """select
p.name,
clust.wr_cluster_label,
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
(select p.id, p.wr_cluster_label from (select count(*), wr_cluster_label
		from football.players where position = 'WR'
		group by wr_cluster_label
		order by count desc) c,
	football.players p 
	where c.count > 1
	and c.wr_cluster_label = p.wr_cluster_label
	union
	select p.id, 500 wr_cluster_label from (select count(*), wr_cluster_label
	from football.players where position = 'WR'
	group by wr_cluster_label
	order by count desc) c,
	football.players p
	where c.count = 1
	and p.wr_cluster_label = c.wr_cluster_label) clust
where p.id = ps.player_id
and p.position = 'WR'
and prs.player_id = ps.player_id
and prs.year = ps.year
and ps.is_college_season = true
and p.id <> 1402
and p.id = clust.id
order by ps.player_id, ps.year;"""

df = pd.read_sql_query(meanshift_query_string, con=conn)
df = df.fillna(value=0)

groupeddfbylabel = df.groupby(df.columns[1])

labels = []
count = []

## Remove single label rows
for label, group in groupeddfbylabel:
	labels.append(label)
	groupdfbyname = group.groupby(df.columns[0])
	count.append(groupdfbyname.ngroups)
label_count = np.transpose([labels, count])


condition = list(filter(lambda x: x[1] > 1, label_count))
clusters_subject_to_condition = [i[1] for i in condition]
##df = df[df['wr_cluster_label'].isin(clusters_subject_to_condition)]

## Normalize Data

groupeddf = df.groupby(df.columns[1])

filtered_count = []
player_names = []	
plain_input_data = []
ragged_tensor_splits = []
player_labels = []
for label, group in groupeddf:
	groupdfbyname = group.groupby(df.columns[0])
	filtered_count.append([label, groupdfbyname.ngroups])
	for name, innerGroup in groupdfbyname:
		innerGroup.pop('name')
		innerGroup.pop('wr_cluster_label')
		ragged_tensor_splits.append(innerGroup.shape[0])
		plain_input_data.extend(innerGroup.values.tolist())
		player_names.append(name)
		player_labels.append(label)
          


ragged_tensor = tf.RaggedTensor.from_row_lengths(values=plain_input_data, row_lengths=ragged_tensor_splits)

ragged_tensor = ragged_tensor.to_tensor()

ragged_tensor = tf.math.l2_normalize(ragged_tensor, axis = -1)

nonZeroRows = tf.reduce_sum(tf.abs(ragged_tensor), 2) > 0 

ragged_tensor = tf.ragged.boolean_mask(ragged_tensor, nonZeroRows)

full_dim = (ragged_tensor.shape.as_list()[1], ragged_tensor.shape.as_list()[2],)

n_features = ragged_tensor.shape.as_list()[2]

train_data, test_data, train_labels, test_labels = train_test_split_as_tensor_balance_category_v2(ragged_tensor, player_labels, 0.8, filtered_count, player_names, label_count)

num_classes = train_labels.shape[1]
input_layer = keras.Input(shape=full_dim, ragged=True)

time_distributed_layer1 = layers.TimeDistributed(layers.Dense(n_features))(input_layer)

lstm_layer1 = layers.Bidirectional(layers.LSTM(64, dropout=0.4, return_sequences=True))(time_distributed_layer1)
lstm_layer2 = layers.Bidirectional(layers.LSTM(48, dropout=0.4, return_sequences=True))(lstm_layer1)

lstm_layer3 = layers.Bidirectional(layers.LSTM(32, dropout=0.4, return_sequences=False))(lstm_layer2)

dense_layer1 = layers.Dense(64, activation='sigmoid')(lstm_layer3)
dense_layer2 = layers.Dense(48, activation='sigmoid')(dense_layer1)
dense_layer3 = layers.Dense(32, activation='sigmoid')(dense_layer2)

output_layer = layers.Dense(num_classes, activation='softmax')(dense_layer3)

model = keras.models.Model(inputs=input_layer, outputs=output_layer)
model.summary()

model.compile(optimizer="adam", loss=tf.keras.losses.categorical_crossentropy , metrics=[tf.keras.metrics.CategoricalAccuracy()])

epoch_size = 300

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)

history = model.fit(train_data, train_labels, epochs=epoch_size, batch_size=10, callbacks=[callback], shuffle=True, validation_data=(test_data, test_labels))

num_epochs = len(history.epoch)

history_fig, (ax1, ax2) = plt.subplots(2, sharex=True)
history_fig.suptitle('Classifier Training Performance')
ax1.plot(range(0,num_epochs), history.history['categorical_accuracy'], color='blue')
ax1.plot(range(0,num_epochs), history.history['val_categorical_accuracy'], color='red', alpha=0.9)
ax1.set(ylabel='Reconstruction Accuracy')
ax2.plot(range(0,num_epochs), np.log10(history.history['loss']), color='blue')
ax2.plot(range(0,num_epochs), np.log10(history.history['val_loss']), color='red', alpha=0.9)
ax2.set(ylabel='log_10(loss)', xlabel='Training Epoch')

history_fig.show()

y_pred = model.predict(test_data)
y_pred = tf.argmax(y_pred, axis=1)

y_true = tf.argmax(test_labels, axis=1)
confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)


plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx,
            xticklabels=set(y_true.numpy()),
            yticklabels=set(y_true.numpy()),
            annot=True, fmt='g')
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.show()