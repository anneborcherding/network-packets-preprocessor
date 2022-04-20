from keras.callbacks import History


class PreprocessorInterface():
    """This is an informal interface for the preprocessor (duck typing)."""

    def set_pcap(self, path: str) -> str:
        """
        Sets the PCAP to be used in the following.
        There can be more than one active PCAP.

        :param path: Path to the PCAP.
        :return: ID for the PCAP. This ID has the format "pcap_INT".
        """
        pass

    def get_macs(self, pcap_id: str) -> list:
        """
        Returns a list of MAC addresses that are included in the PCAP with the given ID.

        :param pcap_id: ID of the PCAP.
        :return: List of MACs.
        """
        pass

    def get_ips(self, pcap_id: str, mac_addr: str) -> list:
        """
        Returns the IP addresses for a given MAC address.

        :param pcap_id: ID of the PCAP.
        :param mac_addr: MAC address as string
        :return: List of IPs associated with the MAC address in the PCAP.
        """

    def get_connections(self, pcap_id: str) -> dict:
        """
        Returns a dict indicating between which MACs an connection exists.

        :param pcap_id: ID of the PCAP.
        :return: Dict of dicts of MACs and connections. Example: {
            '00:11:22:33:44:55':{'TCP':['00:11:22:33:44:66','00:11:22:33:44:77'], 'UDP':['00:11:22:33:44:66',
            '00:11:22:33:44:88']}, '00:11:22:33:44:66':{'TCP':['00:11:22:33:44:77']}}
        """
        pass

    def get_packets(self, pcap_id: str) -> list:
        """
        Returns the packets of the PCAP with the given ID as a list.

        :param pcap_id: ID of the PCAP.
        :return: List of packets.
        """
        pass

    def get_packets_protocols(self, pcap_id: str) -> (list, list):
        """
        Returns the packets of the PCAP with the given ID as a list as well as the protocols of the packet.

        :param pcap_id: ID of the PCAP.
        :return: List of packets and list of protocols. To access the highest protocol use something like
            packets, protocols = get_packet_protocols
            highest_protocols = [proto[-1] for proto in protocols]
        """

    def set_parameters_autoencoder(self, sample_size=150, encoding_size=2, number_of_hidden_layers=4,
                                   nodes_of_hidden_layers=(256, 64, 32, 8), loss="MSE", epochs=100, optimizer="adam"):
        """
        Sets the parameters of the autoencoder.

        :param sample_size: Size of the samples.
        :param encoding_size: Size of the encoding.
        :param number_of_hidden_layers: Number of hidden layers.
        :param nodes_of_hidden_layers: Number of nodes in the hidden layers. len(nodes_of_hidden_layers) needs to be equal to number_of_hidden_layers.
        :param loss: The loss function to use. Can be "MSE" or "MAE".
        :param epochs: Number of epochs for the training.
        :param optimizer: Optimizer to be used. Can be one of tf.keras.optimizers.
        """
        pass

    def set_parameters_pca(self, encoding_size=2):
        """
        Sets the parameters of the PCA.

        :param encoding_size: Size of the encoding.
        """
        pass

    def set_preprocessing(self, scaling_method: str, normalization_method: str, sample_size: int):
        """
        Sets the current preprocessing method.
        :param scaling_method: The method to be used for scaling. Possible values: ["Length", "ValueLength"]
        :param normalization_method: The method to be used for normalization. Possible values: ["None", "L1", "L2"]
        :param sample_size: The sample size.
        """
        pass

    def train_autoencoder(self, pcap_id: str) -> History:
        """
        Trains an autoencoder on the PCAP with the given ID and the given arguments.

        :param pcap_id: ID of the PCAP.
        :return: history of the training process.
        """
        pass

    def train_pca(self, pcap_id: str) -> (float, float):
        """
        Trains an PCA on the PCAP with the given ID and the given arguments.

        :param pcap_id: ID of the PCAP.
        :return: Loss on the train data and on the test data.
        """
        pass

    def encode_pca(self, pcap_id: str) -> list:
        """
         Uses the preprocessing method with the given ID to encode the packets in the PCAP with the given ID.

        :param pcap_id: ID of the PCAP.
        :return: List of encoded packets.
        """
        pass

    def encode_autoencoder(self, pcap_id: str) -> list:
        """
         Uses the preprocessing method with the given ID to encode the packets in the PCAP with the given ID.

        :param pcap_id: ID of the PCAP.
        :return: List of encoded packets.
        """
        pass
