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

# Scikit learn
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn import manifold

# TensorFlow and Keras
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import backend as K
import random
import math


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

def create_auto_encoder(encoding_dims, ragged_tensor):
    full_dim = (ragged_tensor.shape.as_list()[1], ragged_tensor.shape.as_list()[2],)
    n_timesteps = ragged_tensor.shape.as_list()[1] # should be 11

    n_features = ragged_tensor.shape.as_list()[2] # 44 

    encoding_dim1 = 256
    encoding_dim2 = 64
    encoding_dim3 = 32
    encoding_dim4 = encoding_dims # we will use these 100 dimensions for clustering

    # This is our encoder input
    encoder_input_data = keras.Input(shape=full_dim, ragged=True)

    # the encoded representation of the input
    time_distributed_layer1 = layers.TimeDistributed(layers.Dense(n_features))(encoder_input_data)

    encoded_layer1 = layers.LSTM(encoding_dim1, return_sequences=False)(time_distributed_layer1)
    encoded_layer2 = keras.layers.Dense(encoding_dim2, activation='relu')(encoded_layer1)
    encoded_layer3 = keras.layers.Dense(encoding_dim3, activation='relu')(encoded_layer2)

    # Note that encoded_layer3 is our 3 dimensional "clustered" layer, which we will later use for clustering
    encoded_layer4 = keras.layers.Dense(encoding_dim4, activation='relu', name="ClusteringLayer")(encoded_layer3)

    encoder_model = keras.Model(encoder_input_data, encoded_layer4)

    # the reconstruction of the input
    decoded_layer4 = keras.layers.Dense(encoding_dim3, activation='relu')(encoded_layer4)

    decoded_layer3 = keras.layers.Dense(encoding_dim2, activation='relu')(decoded_layer4)
    decoded_layer2 = keras.layers.Dense(encoding_dim1, activation='relu')(decoded_layer3)
    repeat_vector_layer = keras.layers.Lambda(repeat_vector2, output_shape=(None, full_dim[1])) ([decoded_layer2, encoder_input_data])

    ##repeat_vector_layer = layers.RepeatVector(11)(decoded_layer2)
    decoded_layer1 = layers.LSTM(n_features, return_sequences=True)(repeat_vector_layer)
    time_distributed_layer = layers.TimeDistributed(layers.Dense(n_features))(decoded_layer1)


    # This model maps an input to its autoencoder reconstruction
    autoencoder_model = keras.Model(encoder_input_data, outputs=time_distributed_layer, name="Encoder")
    autoencoder_model.summary()


    # compile the model
    autoencoder_model.compile(optimizer="RMSprop", loss=tf.keras.losses.mean_squared_error, metrics=[keras.metrics.Accuracy()])

    normalized_input = ragged_tensor

    epoch_size = 1000


    train_data, test_data, train_labels, test_labels = train_test_split_as_tensor(normalized_input, player_names, 0.8)

    history = autoencoder_model.fit(train_data, train_data, epochs=epoch_size, batch_size=22, shuffle=True, validation_data=(test_data, test_data))


    history_fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    history_fig.suptitle('Autoencoder Training Performance')
    ax1.plot(range(0,epoch_size), history.history['val_accuracy'], color='blue')
    ax1.set(ylabel='Reconstruction Accuracy')
    ax2.plot(range(0,epoch_size), np.log10(history.history['loss']), color='blue')
    ax2.plot(range(0,epoch_size), np.log10(history.history['val_loss']), color='red', alpha=0.9)
    ax2.set(ylabel='log_10(loss)', xlabel='Training Epoch')

    history_fig.show()

    autoencoder_model.save('python/models/{0}-autoencoder-WR-cluster-output-no-rookies-1990-td.keras'.format(encoding_dims))
    encoder_model.save('python/models/{0}-encoder-WR-cluster-output-no-rookies-1990-td.keras'.format(encoding_dims))

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
plain_input_data = []
ragged_tensor_splits = []
for k, g in groupeddf:
    g.pop('name')
    ragged_tensor_splits.append(g.shape[0])
    plain_input_data.extend(g.values.tolist())
    zero_padded_input_data.append(zeropadrecordsalginbyage(minage, maxage, g))

    player_names.append(k)


ragged_tensor = tf.RaggedTensor.from_row_lengths(values=plain_input_data, row_lengths=ragged_tensor_splits)

ragged_tensor = ragged_tensor.to_tensor()

ragged_tensor = tf.math.l2_normalize(ragged_tensor, axis = -1)

## normalized_input = tf.RaggedTensor.from_row_lengths(values=normalized_input, row_lengths=ragged_tensor_splits)

nonZeroRows = tf.reduce_sum(tf.abs(ragged_tensor), 2) > 0 

ragged_tensor = tf.ragged.boolean_mask(ragged_tensor, nonZeroRows)


print("Training model with {0} encoded dims".format(2))
create_auto_encoder(2, ragged_tensor)