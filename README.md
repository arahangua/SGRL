# Semantic Graph Reinforcement Learning Project

This project implements a semantic graph reinforcement learning system using heterogeneous graph neural networks and the Mamba architecture.

## Features

- Heterogeneous Graph Neural Networks
- Mamba-style transformations for node embeddings
- Reinforcement Learning with Graph-based environments
- Graph expressivity evaluation
- LLM feedback integration

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/semantic-graph-rl.git
   cd semantic-graph-rl
   ```

2. Install dependencies using Poetry:
   ```
   poetry install
   ```

## Project Structure

- `semantic_graph_rl/`
  - `data/`: Data loading and processing
  - `models/`: Neural network models
  - `utils/`: Utility functions
- `tests/`: Unit tests
- `main.py`: Main entry point

## Usage

To run the main script:

```
python main.py
```

## Development

To run tests:

```
pytest
```

To format code:

```
black .
```

To check code style:

```
flake8
```

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the GPL-3.0 License - see the [LICENSE](LICENSE) file for details.
