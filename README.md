# 🌐 Semantic (Heterogeneous) Graph Reinforcement Learning

[![Pre-commit](https://img.shields.io/badge/Pre--commit-Enabled-blue?logo=pre-commit)](https://pre-commit.com/)
[![Black](https://img.shields.io/badge/Code%20Style-Black-000000?logo=black)](https://github.com/psf/black)
[![Isort](https://img.shields.io/badge/Imports-Isort-ef8336?logo=isort)](https://pycqa.github.io/isort/)
[![Ruff](https://img.shields.io/badge/Linter-Ruff-000000?logo=ruff)](https://github.com/charliermarsh/ruff)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/yourusername/semantic-graph-rl)
[![License](https://img.shields.io/badge/License-GPL--3.0-green)](LICENSE)

<img src="static/EAAC_knowledge_graph_lc_example.png" alt="Semantic Graph Reinforcement Learning" width="600"/>

*EAAC example graph (from [here](https://github.com/arahangua/EAAC))*

This project implements a semantic graph reinforcement learning system using heterogeneous graph neural networks and the Mamba architecture.

## ✨ Features

- 🌟 Heterogeneous Graph Neural Networks
- 🐍 Mamba-style transformations for node embeddings
- 🤖 Reinforcement Learning with Graph-based environments
- 📊 Graph expressivity evaluation
- 🧠 LLM feedback integration

## 🚀 Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/semantic-graph-rl.git
   cd semantic-graph-rl
   ```

2. Install dependencies using Poetry:
   ```sh
   poetry install
   ```

## 📂 Project Structure

- `semantic_graph_rl/`
  - `data/`: 📥 Data loading and processing
  - `models/`: 🧠 Neural network models
  - `utils/`: 🛠️ Utility functions
- `tests/`: 🧪 Unit tests
- `main.py`: 🚪 Main entry point

## 🛠️ Usage

To run the main script:

```sh
python main.py
```

## 🧑‍💻 Development

### Pre-commit Hooks

This project uses `pre-commit` to manage pre-commit hooks for code formatting and linting. The following tools are integrated:
- `black` for code formatting
- `isort` for import sorting
- `ruff` for linting

To set up pre-commit hooks, run:

```sh
pre-commit install
```

To run the hooks on all files, use:

```sh
pre-commit run --all-files
```

[![Pre-commit](https://img.shields.io/badge/Pre--commit-Enabled-blue?logo=pre-commit)](https://pre-commit.com/)
[![Black](https://img.shields.io/badge/Code%20Style-Black-000000?logo=black)](https://github.com/psf/black)
[![Isort](https://img.shields.io/badge/Imports-Isort-ef8336?logo=isort)](https://pycqa.github.io/isort/)
[![Ruff](https://img.shields.io/badge/Linter-Ruff-000000?logo=ruff)](https://github.com/charliermarsh/ruff)

To run tests:

```sh
pytest
```

To format code:

```sh
black .
```

To check code style:

```sh
flake8
```

## 🤝 Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## 📜 License

This project is licensed under the GPL-3.0 License - see the [LICENSE](LICENSE) file for details.
