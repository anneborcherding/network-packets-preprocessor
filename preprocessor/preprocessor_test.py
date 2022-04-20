"""Tests for the preprocessor. Use pytest to run it."""
from unittest import mock

from preprocessor.preprocessor import Preprocessor
from unittest.mock import MagicMock

preprocessor = Preprocessor()
preprocessor_bug = Preprocessor()

small_mac_one = "00:50:56:ef:02:f5"
small_mac_two = "00:0c:29:0f:94:2e"
small_pcap_id = "pcap_0"


########################################################################################################################
# Tests for API functions
########################################################################################################################


def test_set_pcap():
    assert len(preprocessor.pcaps) == 0
    preprocessor.set_pcap("small_example.pcapng")
    assert len(preprocessor.pcaps) == 1
    assert "pcap_0" in preprocessor.pcaps
    try:
        preprocessor.set_pcap("not_existing_pcap.pcapng")
    except FileNotFoundError:
        assert True
    else:
        assert False


def test_get_macs():
    pcap_id = preprocessor.set_pcap("small_example.pcapng")
    macs = preprocessor.get_macs(pcap_id)
    assert len(macs) == 2
    assert small_mac_one in macs
    assert small_mac_two in macs
    try:
        preprocessor.get_macs("this is an invalid id")
    except ValueError:
        assert True
    else:
        assert False


def test_get_ips():
    pcap_id = preprocessor.set_pcap("small_example.pcapng")
    macs = preprocessor.get_macs(pcap_id)
    ips_small_mac_one = preprocessor.get_ips(pcap_id, small_mac_one)
    ips_small_mac_two = preprocessor.get_ips(pcap_id, small_mac_two)
    assert len(ips_small_mac_one) == 2
    assert "192.168.5.2" in ips_small_mac_one
    assert "93.184.216.34" in ips_small_mac_one
    assert len(ips_small_mac_two) == 1
    assert ips_small_mac_two[0] == "192.168.5.128"


def test_get_connections():
    conns = preprocessor.get_connections(small_pcap_id)
    assert len(conns) == 2
    assert small_mac_one in conns
    assert small_mac_two in conns
    assert len(conns[small_mac_one]) == len(conns[small_mac_two])
    assert "Ethernet" in conns[small_mac_one]
    assert "Ethernet" in conns[small_mac_two]
    assert "IP" in conns[small_mac_one]
    assert "IP" in conns[small_mac_two]
    assert "UDP" in conns[small_mac_one]
    assert "UDP" in conns[small_mac_two]
    assert "TCP" in conns[small_mac_one]
    assert "TCP" in conns[small_mac_two]
    assert "DNS" in conns[small_mac_one]
    assert "DNS" in conns[small_mac_two]


def test_get_packets():
    packets = preprocessor.get_packets(small_pcap_id)
    assert len(packets) == 15
    assert packets[0].src == small_mac_two
    assert packets[0].dst == small_mac_one


def test_get_packets_protocols():
    packets, protocols = preprocessor.get_packets_protocols(small_pcap_id)
    assert len(packets) == 15
    assert packets[0].src == small_mac_two
    assert packets[0].dst == small_mac_one
    highest_protocols = [proto[-1] for proto in protocols]
    assert highest_protocols[10] == "TCP"


def test_set_preprocessing():
    preprocessor.set_preprocessing("ValueLength", "None", 150)
    assert preprocessor.PreprocessingConfig.sample_size == 150
    assert preprocessor.PreprocessingConfig.scaling_method == "ValueLength"
    assert preprocessor.PreprocessingConfig.normalization_method == "None"
    try:
        preprocessor.set_preprocessing("NOT EXISTING VALUE", "None")
    except ValueError:
        assert True
    else:
        assert False
    try:
        preprocessor.set_preprocessing("Length", "NOT EXISTING VALUE")
    except ValueError:
        assert True
    else:
        assert False


def test_set_parameters_autoencoder():
    assert preprocessor.AutencoderConfig.autoencoder is None
    preprocessor.set_parameters_autoencoder(150, 2)
    assert preprocessor.AutencoderConfig.autoencoder
    assert preprocessor.AutencoderConfig.encoding_size == 2


def test_set_parameters_pca():
    assert preprocessor.PCAConfig.pca is None
    preprocessor.set_parameters_pca(2)
    assert preprocessor.PCAConfig.pca
    assert preprocessor.PCAConfig.encoding_size == 2


def test_train_autoencoder():
    preprocessor.set_parameters_autoencoder(150, 2)
    history = preprocessor.train_autoencoder("pcap_0")
    assert len(history.epoch) == 100
    assert len(history.history["loss"]) == 100
    assert len(history.history["val_loss"]) == 100


def test_train_pca():
    preprocessor.set_parameters_pca(2)
    assert preprocessor.PCAConfig.pca
    assert preprocessor.PCAConfig.encoding_size == 2
    mae_train, mae_test = preprocessor.train_pca("pcap_0")
    assert float(mae_train) < 0.4
    assert float(mae_train) > 0
    assert float(mae_test) < 0.4
    assert float(mae_test) > 0


def test_encode_pca():
    encoded_values = preprocessor.encode_pca("pcap_0")
    assert len(encoded_values) == 15
    for value in encoded_values:
        assert len(value) == 2


def test_encode_autoencoder():
    encoded_values = preprocessor.encode_autoencoder("pcap_0")
    assert len(encoded_values) == 15
    for value in encoded_values:
        assert len(value) == 2


def test_bug():
    pcap_id = preprocessor_bug.set_pcap("example.pcapng")
    preprocessor_bug.set_preprocessing("Length", "L1", 150)

    preprocessor_bug.set_parameters_autoencoder(number_of_hidden_layers=4, nodes_of_hidden_layers=(256, 64, 32, 8), loss="MAE", epochs=100, optimizer="adam")
    preprocessor_bug.set_parameters_pca(2)
    history = preprocessor_bug.train_pca(pcap_id)
    encoded_values = preprocessor_bug.encode_pca(pcap_id)
    history_ae = preprocessor_bug.train_autoencoder(pcap_id)
    encoded_values_ae = preprocessor_bug.encode_autoencoder(pcap_id)
    assert history is not None
    assert history_ae is not None
    assert len(encoded_values) == 543
    assert len(encoded_values_ae) == 543
    assert True

########################################################################################################################
# Tests for non-API functions
########################################################################################################################

def test_preprocess_packets():
    mock_function = MagicMock(return_value=[0.153, 0.173, 0.328])
    mock_function_one = MagicMock(return_value=[0.153, 0.173, 0.328])

    preprocessor.set_preprocessing("Length", "None", 150)
    with mock.patch("preprocessor.preprocessor.scale_packet_length", mock_function):
        preprocessed_packets = preprocessor.preprocess_packets("pcap_0")
    mock_function.assert_called()
    assert len(preprocessed_packets) == 543

    preprocessor.set_preprocessing("ValueLength", "None", 150)
    with mock.patch("preprocessor.preprocessor.scale_packet_length", mock_function):
        with mock.patch("preprocessor.preprocessor.scale_packet_values", mock_function_one):
            preprocessed_packets = preprocessor.preprocess_packets("pcap_0")
    mock_function.assert_called()
    mock_function_one.assert_called()
    assert len(preprocessed_packets) == 543
