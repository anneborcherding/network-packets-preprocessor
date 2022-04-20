from scapy.layers.inet import IP
from scapy.layers.l2 import Ether
from scapy.utils import rdpcap
from scapy.layers import all
import os.path
from dataclasses import dataclass
import numpy as np
import tensorflow as tf
from scapy.all import raw
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from keras.callbacks import History

from preprocessor.preprocessor_interface import PreprocessorInterface
from preprocessor.autencoder import NetworkPacketAutoencoder


def get_packet_layers(pkt):
    """Returns the layers of the given packet."""
    counter = 0
    layers = []
    while True:
        layer = pkt.getlayer(counter)
        if layer is None or "Padding" in layer.name or "Raw" in layer.name:
            break
        layers.append(layer.name)
        counter += 1
    return layers


def bytes_as_bits_list(input_bytes) -> list[float]:
    """returns a list of floats representing the bits of the given bytearray"""
    bytes_as_bits = ''.join(format(byte, '08b') for byte in input_bytes)
    # make sure there are no decimal places (kinda ugly maybe)
    res = list([float(int(float(a))) for a in bytes_as_bits])
    return res


def scale_packet_values(input_bytes) -> list[float]:
    """returns a list of floats representing the bytes of the given bytearray scaled to [0,1].
        Idea from Chiu et al. (2020)"""
    res = [byte / 255 for byte in input_bytes]
    return res


def scale_packet_length(input_bytes, length=1500) -> list[float]:
    """returns the bytes trimmed or padded to the given length (in bytes).
        Idea from Chiu et al. (2020)"""
    res = input_bytes.ljust(length, b'\0') if len(input_bytes) <= length else input_bytes[0:length]
    res_list = [byte / 1 for byte in res]
    return res_list


def normalize_lone(input_array: np.array) -> np.array:
    """normalizes the given numpy array with the L1 norm"""
    return input_array / np.linalg.norm(input_array, ord=1)


def normalize_ltwo(input_array: np.array) -> np.array:
    """normalizes the given numpy array with the L2 norm"""
    return input_array / np.linalg.norm(input_array)


class Preprocessor(PreprocessorInterface):
    """Class representing the preprocessor. Implements the PreprocessorInterface"""
    pcaps = {}
    """ids as key and paths as value"""
    devices = {}
    """macs as key and ips as value"""
    devices_filled = False
    pcap_next_id = 0
    """counter to keep track of pcap ids"""
    scaling_methods = ["Length", "ValueLength"]
    """possible values for scaling"""
    normalization_methods = ["None", "L1", "L2"]
    """possible values for normalization"""

    @dataclass
    class PreprocessingConfig:
        """Dataclass to contain the configuration for the preprocessing."""
        scaling_method: str = "None"
        normalization_method: str = "None"
        sample_size: int = 1500
        # pcap_filter = ""

    @dataclass
    class AutencoderConfig:
        """Dataclass to contain the configuration for the autoencoder"""
        autoencoder = None
        encoding_size = 2
        optimizer: str = "adam"
        loss: str = "MSE"
        epochs: int = 100

    @dataclass
    class PCAConfig:
        """Dataclass to contain the configuration for the PCA"""
        pca = None
        encoding_size = 2

    def set_pcap(self, path: str) -> str:
        if not os.path.isfile(path):
            raise FileNotFoundError()
        try:
            rdpcap(path)
        except Exception as e:
            raise e
        new_id = f"pcap_{self.pcap_next_id}"
        self.pcaps[new_id] = rdpcap(path)
        self.pcap_next_id = self.pcap_next_id + 1
        self.devices_filled = False
        return new_id

    def process_macs_and_ips(self, pcap_id: str):
        self.devices = {}
        packets = self.pcaps[pcap_id]
        for pkt in packets:
            if not Ether in pkt:
                continue
            macs = [pkt[Ether].src, pkt[Ether].dst]
            ips = [pkt[IP].src, pkt[IP].dst] if IP in pkt and pkt[IP].src and pkt[IP].dst else None
            for i, mac in enumerate(macs):
                if mac in self.devices.keys():
                    if ips and ips[i] not in self.devices[mac]:
                        self.devices[mac].append(ips[i])
                else:
                    if ips:
                        self.devices[mac] = [ips[i]]
                    else:
                        self.devices[mac] = []
        self.devices_filled = True

    def get_macs(self, pcap_id: str) -> list:
        if pcap_id not in self.pcaps.keys():
            raise ValueError("This is an invalid pcap ID.")
        if not self.devices_filled:
            self.process_macs_and_ips(pcap_id)
        return list(self.devices.keys())

    def get_ips(self, pcap_id:str, mac_addr) -> list:
        if pcap_id not in self.pcaps.keys():
            raise ValueError("This is an invalid pcap ID.")
        if not self.devices_filled:
            self.process_macs_and_ips(pcap_id)
        if mac_addr not in self.devices.keys():
            raise ValueError("This is not a known MAC address.")
        else:
            return self.devices[mac_addr]

    def get_connections(self, pcap_id: str) -> dict:
        packets = self.pcaps[pcap_id]
        res = {}
        for pkt in packets:
            if Ether not in pkt:
                continue
            src = pkt[Ether].src
            if src not in res.keys():
                res[src] = {}
            dst = pkt[Ether].dst
            if dst not in res.keys():
                res[dst] = {}
            for layer in get_packet_layers(pkt):
                if layer in res[src].keys():
                    if dst not in res[src][layer]:
                        res[src][layer].append(dst)
                else:
                    res[src][layer] = [dst]
                if layer in res[dst].keys():
                    if src not in res[dst][layer]:
                        res[dst][layer].append(src)
                else:
                    res[dst][layer] = [src]
        return res

    def get_packets(self, pcap_id: str) -> list:
        packets = self.pcaps[pcap_id]
        return packets

    def get_packets_protocols(self, pcap_id: str) -> (list, list):
        packets = self.pcaps[pcap_id]
        protocols = []
        for pkt in packets:
            protocols.append(get_packet_layers(pkt))
        return packets, protocols

    def set_preprocessing(self, scaling_method: str = "Length", normalization_method: str = "None", sample_size=1500):
        if scaling_method not in self.scaling_methods:
            raise ValueError(
                f"Possible values for scaling_method are f{self.scaling_methods} but you set f{scaling_method}.")
        if normalization_method not in self.normalization_methods:
            raise ValueError(
                f"Possible values for scaling_method are f{self.normalization_methods} but you set f{normalization_method}.")
        self.PreprocessingConfig.scaling_method = scaling_method
        self.PreprocessingConfig.normalization_method = normalization_method
        self.PreprocessingConfig.sample_size = sample_size

    def preprocess_packets(self, pcap_id: str) -> np.array:
        """Preprocesses the packets in the given pcap with the configuration defined in the preprocessing dataclass"""
        packets = self.pcaps[pcap_id]
        packets_binary = [raw(pkt) for pkt in packets]
        # Note: Ugly coding, do not copy this approach :D
        preprocessed_packets = []
        if self.PreprocessingConfig.scaling_method == "Length":
            preprocessed_packets = np.matrix(
                [scale_packet_length(pkt, self.PreprocessingConfig.sample_size) for pkt in packets_binary])
        elif self.PreprocessingConfig.scaling_method == "ValueLength":
            preprocessed_packets = np.matrix(
                [scale_packet_values(scale_packet_length(pkt, self.PreprocessingConfig.sample_size)) for pkt in
                 packets_binary])

        if self.PreprocessingConfig.normalization_method == "L1":
            preprocessed_packets = normalize_lone(preprocessed_packets)
        elif self.PreprocessingConfig.normalization_method == "L2":
            preprocessed_packets = normalize_ltwo(preprocessed_packets)
        return preprocessed_packets

    def set_parameters_autoencoder(self, sample_size=150, encoding_size=2, number_of_hidden_layers=4,
                                   nodes_of_hidden_layers=(256, 64, 32, 8), loss="MSE", epochs=100, optimizer="adam"):
        self.AutencoderConfig.autoencoder = NetworkPacketAutoencoder(sample_size, encoding_size,
                                                                     number_of_hidden_layers, nodes_of_hidden_layers)
        if loss == "MSE":
            self.AutencoderConfig.loss = tf.keras.losses.mse
        if loss == "MAE":
            self.AutencoderConfig.loss = tf.keras.losses.mae
        self.AutencoderConfig.epochs = epochs
        self.AutencoderConfig.encoding_size = encoding_size
        # TODO check if the value makes sense
        self.AutencoderConfig.optimizer = optimizer

    def set_parameters_pca(self, encoding_size=2):
        self.PCAConfig.pca = PCA(n_components=encoding_size)
        self.PCAConfig.encoding_size = encoding_size

    def train_autoencoder(self, pcap_id: str) -> History:
        preprocessed_packets = self.preprocess_packets(pcap_id)
        train_data, test_data = train_test_split(preprocessed_packets, test_size=0.2)
        self.AutencoderConfig.autoencoder.compile(optimizer=self.AutencoderConfig.optimizer,
                                                  loss=self.AutencoderConfig.loss)
        history = self.AutencoderConfig.autoencoder.fit(train_data, train_data,
                                                        epochs=self.AutencoderConfig.epochs,
                                                        # batch_size=512,
                                                        validation_data=(test_data, test_data),
                                                        shuffle=True)
        return history

    def train_pca(self, pcap_id: str) -> (float, float):
        preprocessed_packets = self.preprocess_packets(pcap_id)
        train_data, test_data = train_test_split(preprocessed_packets, test_size=0.2)
        res_train = self.PCAConfig.pca.fit_transform(train_data)
        inv_pca_train = self.PCAConfig.pca.inverse_transform(res_train)
        res_test = self.PCAConfig.pca.transform(test_data)
        inv_pca_test = self.PCAConfig.pca.inverse_transform(res_test)
        mae = tf.keras.losses.MeanAbsoluteError()
        return mae(train_data, inv_pca_train), mae(test_data, inv_pca_test)

    def encode_pca(self, pcap_id: str) -> list:
        preprocessed_packets = self.preprocess_packets(pcap_id)
        result = []
        for pkt in preprocessed_packets:
            result.append(self.PCAConfig.pca.transform(np.array(pkt))[0])
        return result

    def encode_autoencoder(self, pcap_id: str) -> list:
        preprocessed_packets = self.preprocess_packets(pcap_id)
        result = []
        for pkt in preprocessed_packets:
            result.append(self.AutencoderConfig.autoencoder.encoder.predict(np.array(pkt))[0])
        return result
