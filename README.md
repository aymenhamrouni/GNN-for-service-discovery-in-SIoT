# GNN Clustering for Social IoT

This project implements Graph Neural Network (GNN) based clustering for Social Internet of Things (SIoT) networks. It includes implementations of node2vec for node embeddings and community detection using the Louvain method.

## Citation

If you use this code in your research, please cite our work:

```bibtex
@inproceedings{hamrouni2022service,
  title={Service Discovery in Social Internet of Things using Graph Neural Networks},
  author={Hamrouni, A. and Ghazzai, H. and Massoud, Y.},
  booktitle={2022 IEEE 65th International Midwest Symposium on Circuits and Systems (MWSCAS)},
  year={2022},
  pages={1-4},
  doi={10.1109/MWSCAS54063.2022.9859333}
}
```

### Keywords
Performance evaluation, Circuits and systems, Simulation, Urban areas, Standardization, Dynamic scheduling, Graph neural networks, Service discovery, Resource allocation, Graph neural network, Social internet of things, Smart city

## Features

- Node2Vec implementation for node embeddings
- Community detection using Louvain method
- Dynamic relationship creation for SIoT devices
- Clustering analysis tools

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the setup script:
```bash
python setup.py install
```

## Usage

The main functionality can be accessed through `main.py`. Additional scripts are available for specific tasks:

- `communityDetectionLouvain.py`: Community detection implementation
- `createSocialFriendshipOnwershipRelations.py`: Creates social relationships between devices
- `createColocationDevicesRelationsDynamic.py`: Creates colocation relationships
- `Clustering.py`: Clustering analysis tools

## Data

The project uses edge lists and device position data stored in the `data/` directory.

## License

MIT License 