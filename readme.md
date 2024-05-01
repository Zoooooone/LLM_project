# Personal Knowledge Database Assistant

![GitHub commit activity](https://img.shields.io/github/commit-activity/m/zoooooone/LLM_project?label=activity) &ensp; ![GitHub last commit (branch)](https://img.shields.io/github/last-commit/Zoooooone/LLM_project/master) &ensp; ![GitHub repo size](https://img.shields.io/github/repo-size/Zoooooone/LLM_project) &ensp; ![Code Climate maintainability](https://img.shields.io/codeclimate/maintainability-percentage/Zoooooone/LLM_project) &ensp; ![GitHub issues](https://img.shields.io/github/issues/Zoooooone/LLM_project)

This is a personal knowledge base assistant created using LLMs. The project has been designed and developed meticulously to ensure efficient management and retrieval of vast and intricate information, making it a powerful tool for information acquisition.

This project refers to this **[tutorial](https://github.com/logan-zou/Chat_with_Datawhale_langchain/tree/main)**.

# Start-up

## Hardware Requirements

- **CPU**: Intel Core i5 or higher
- **RAM**: 4GB or higher
- **OS**: Windows, macOS, Linux

## Settings

- **Clone the repo**:
    ```bash
    git clone https://github.com/Zoooooone/LLM_project.git
    cd LLM_project
    ```

- **Create a Conda environment and install dependencies**:
  - Python version 3.9 or higher
    ```bash
    conda create -n llm-project python=3.10.11
    conda activate llm-project
    pip install -r requirements.text
    ```

- **Run the project**:
    ```bash
    ./run.sh
    ```
    or
    ```bash
    python -m project.serve.main
    ```