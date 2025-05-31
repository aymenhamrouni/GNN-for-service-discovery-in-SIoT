# GNN Clustering for Social IoT

This project implements Graph Neural Network (GNN) based clustering for Social Internet of Things (SIoT) networks. It includes implementations of node2vec for node embeddings and community detection using the Louvain method.

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