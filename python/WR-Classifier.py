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

def load_encoder_model():
	return tf.keras.models.load_model('python/models/8-encoder-WR-cluster-output-rookies-1990-td.keras', safe_mode=False)


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
		player_names_train.append(player_names[train_index[0]])
 
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
			player_names_train.append(player_names[index])
                  
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

def train_test_split_as_tensor_balance_category_v3(data, labels, training_size, useage_count, player_names, label_count):
	
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
        
		if max_label_count > label_usage[1]:
			generated_samples_for_label = generateSamples(label, max_label_count - label_usage[1])

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

def create_classifier_model(ragged_tensor, player_labels, encoder_model, cluster_type):

	encoded_data = encoder_model.predict(ragged_tensor)


	oversample = SMOTE(k_neighbors=1)
	X, y = oversample.fit_resample(encoded_data, player_labels)
	y = pd.get_dummies(y).astype('float32').values 
	##train_data, test_data, train_labels, test_labels = train_test_split_as_tensor_balance_category_v2(ragged_tensor, player_labels, 0.8, filtered_count, player_names, label_count)

	train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)



	input_layer = keras.Input(shape=encoded_data.shape[1])


	dense_layer1 = layers.Dense(64, activation='relu')(input_layer)
	dense_layer2 = layers.Dense(48, activation='relu')(dense_layer1)
	dense_layer3 = layers.Dense(32, activation='relu')(dense_layer2)

	output_layer = layers.Dense(len(set(player_labels)), activation='softmax')(dense_layer3)

	model = keras.models.Model(inputs=input_layer, outputs=output_layer)
	model.summary()

	model.compile(optimizer="adam", loss=tf.keras.losses.categorical_crossentropy , metrics=[tf.keras.metrics.CategoricalAccuracy()])

	epoch_size = 1000

	history = model.fit(train_X, train_y, epochs=epoch_size, batch_size=10, shuffle=True, validation_data=(test_X, test_y))

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

	y_pred = model.predict(test_X)
	y_pred = tf.argmax(y_pred, axis=1)

	y_true = tf.argmax(test_y, axis=1)
	confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)


	plt.figure(figsize=(10, 8))
	sns.heatmap(confusion_mtx,
				xticklabels=set(y_true.numpy()),
				yticklabels=set(y_true.numpy()),
				annot=True, fmt='g')
	plt.xlabel('Prediction')
	plt.ylabel('Label')
	plt.show()

	model.save('python/models/WR-rookie-{0}-cluster-classifier-from-8-dims.keras'.format(cluster_type))

def load_classifier_model(cluster_type):
	try:
		return tf.keras.models.load_model('python/models/WR-rookie-{0}-cluster-classifier-from-8-dims.keras'.format(cluster_type), safe_mode=False)
	except:
		return None

cur_path = os.path.abspath(os.path.dirname(__file__))
new_path = os.path.join(cur_path, "..\\FootballDataReader.Host\\appsettings.json")

conn = getdbconnectionfromconfiguration(new_path)
cluster_type = "kmeans"
query_string = ''
if cluster_type == "kmeans":
	query_string ="""
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
elif cluster_type == "kmeans":
	query_string ="""
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
elif cluster_type == "affinity":

	query_string = """select
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

df = pd.read_sql_query(query_string, con=conn)
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


condition = list(filter(lambda x: x[1] == 1, label_count))
clusters_subject_to_condition = [i[0] for i in condition]
df = df[~df['wr_cluster_label'].isin(clusters_subject_to_condition)]

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

classifier_model = load_classifier_model(cluster_type)

encoder_model = load_encoder_model()

if classifier_model == None:
	classifier_model = create_classifier_model(ragged_tensor, player_labels, encoder_model, cluster_type)

full_model = keras.Model(encoder_model.input, classifier_model(encoder_model.output), name="Full_Classifier")

y_pred = full_model.predict(ragged_tensor)

y_pred = tf.argmax(y_pred, axis=1)

y = pd.get_dummies(player_labels).astype('float32').values 
y_true = tf.argmax(y, axis=1)

confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx,
			xticklabels=set(y_true.numpy()),
			yticklabels=set(y_true.numpy()),
			annot=True, fmt='g')
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.show()

full_model.save('python/models/classifiers/WR-rookie-{0}-classifier.keras'.format(cluster_type))
