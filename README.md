# IMDb Movie Data Project

This project provides tools for working with IMDb movie metadata, including fine-tuning language models and building a Gradio-based chatbot interface for movie data exploration.

## Project Structure

```
IMDb/
├── finetune/
│   ├── notebook.ipynb
│   └── data/
│       └── data.json
├── gradio/
│   ├── embedding_pipeline.py
│   ├── main.py
│   ├── model_chat.py
│   ├── data/
│   │   └── movies_metadata.csv
│   └── models/
├── raw_data/
│   ├── credits.csv
│   ├── keywords.csv
│   ├── movies_metadata.csv
│   ├── ratings.csv
│   └── notebook.ipynb
├── requirements.txt
├── run_gradio.bat
```

## Setup

1. **Install dependencies**
   
   Open a terminal in the project directory and run:
   
   ```powershell
   pip install -r requirements.txt
   ```

2. **(Optional) Hugging Face Login**
   
   Some scripts require access to Hugging Face models. Log in with:
   
   ```powershell
   huggingface-cli login
   ```

## Fine-tuning

- Use `finetune/notebook.ipynb` for model fine-tuning.
- Training data is stored in `finetune/data/data.json`.
- Example code to load the dataset:

  ```python
  from datasets import load_dataset
  dataset = load_dataset("json", data_files="data/data.json")
  ```

## Gradio Chatbot

- The chatbot interface is implemented in `gradio/main.py`.
- Embedding and retrieval pipeline is in `gradio/embedding_pipeline.py`.
- Movie metadata for retrieval is in `gradio/data/movies_metadata.csv`.

To run the Gradio app:

```powershell
./run_gradio.bat
```

or

```powershell
python gradio/main.py
```

## Data

- **Raw data**: `raw_data/` contains original CSVs and exploratory notebooks.
- **Processed data**: Used for fine-tuning and retrieval, located in `finetune/data/` and `gradio/data/`.

## Requirements

See `requirements.txt` for all dependencies.

## Notes

- Make sure the data paths in your scripts match the actual folder structure.
- If you encounter Hugging Face authentication errors, ensure you are logged in and have access to the required models.

## License

This project is for educational and research purposes. See individual files for additional licensing information if applicable.

## Contact

For questions or issues, please open an issue in this repository.
