import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses


class NetworkPacketAutoencoder(Model):
    """Autoencoder for network packets."""
    def __init__(self, sample_size, encoding_size=2, number_of_hidden_layers=4,
                 nodes_of_hidden_layers=(256, 64, 32, 8)):
        # TODO: Auch die anderen Layer abhängig von der Sample size machen
        super(NetworkPacketAutoencoder, self).__init__()
        if not len(nodes_of_hidden_layers) == number_of_hidden_layers:
            raise ValueError(f"The requested number of hidden layers ({number_of_hidden_layers}) does not correspond "
                             f"to the number of nodes for the hidden layers ({len(nodes_of_hidden_layers)})")
        layers_list_encoder = [layers.Input(shape=(sample_size))]
        for i in range(number_of_hidden_layers):
            layers_list_encoder.append(layers.Dense(nodes_of_hidden_layers[i], activation="relu"))
        layers_list_encoder.append(layers.Dense(encoding_size, activation="relu"))
        self.encoder = tf.keras.Sequential(layers_list_encoder)

        layers_list_decoder = []
        for i in range(number_of_hidden_layers):
            layers_list_decoder.append(
                layers.Dense(nodes_of_hidden_layers[len(nodes_of_hidden_layers) - 1 - i], activation="relu"))
        layers_list_decoder.append(layers.Dense(sample_size, activation="relu"))
        self.decoder = tf.keras.Sequential(layers_list_decoder)

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
