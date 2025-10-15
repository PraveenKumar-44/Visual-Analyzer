text
# 🔍 Visual Product Matcher

**Visual Product Matcher** is a Python-based web application that allows users to find visually similar products from an existing dataset.  
By uploading an image or providing an image URL, the app returns the closest matching items using precomputed image embeddings for fast and accurate retrieval.

---

## 🚀 Features

- Upload product images in **JPG, JPEG, PNG, or WEBP** formats.
- Alternatively, provide a **direct image URL**.
- Search for visually similar products using **precomputed embeddings**.
- Display matching results in a **clean, responsive Streamlit interface**.
- Gracefully handles **missing or invalid dataset images**.
- Can be run locally or **deployed easily on Streamlit Cloud**.

---

## 📂 Folder Structure

visual_product_matcher/
│
├─ app.py # Main Streamlit application
├─ embeddings.npy # Precomputed image embeddings for the dataset
├─ extract_embeddings.py # Script to extract embeddings from images
├─ model.py # Feature extraction / similarity model
├─ metadata.json # Metadata about the products
├─ requirements.txt # Python dependencies
├─ run_preprocess.sh # Shell script to preprocess dataset
├─ data/ # Folder containing product images
│ └─ 123/
│ ├─ 1.jpg
│ ├─ 2.jpg
│ ├─ 10.jpg
│ └─ ... other images
└─ README.md # Project documentation

text

**Note:**  
A `venv/` folder is not included and should be created locally when setting up the environment.

---

## ⚙️ Installation

### 1. Clone the repository
git clone https://github.com/your-username/visual_product_matcher.git
cd visual_product_matcher

text

### 2. Create and activate a virtual environment
python -m venv venv

text

**On Windows:**
venv\Scripts\activate

text

**On macOS/Linux:**
source venv/bin/activate

text

### 3. Install dependencies
pip install -r requirements.txt

text

### 4. Preprocess the dataset and extract embeddings
bash run_preprocess.sh

text

### 5. Run the Streamlit app
streamlit run app.py

text

---

## 🧠 Usage

1. Upload a product image or paste a valid image URL.  
2. Click **"Query"** to find visually similar products.  
3. View the top matching results below the input area, complete with thumbnail previews and metadata.

---

## ☁️ Deployment on Streamlit Cloud

You can deploy this app directly on **Streamlit Cloud**:

- Include the `data/123/` folder, `embeddings.npy`, and `metadata.json` files in your repository.
- Missing images or embeddings will not crash the app, but may lead to incomplete results.
- Customize the `requirements.txt` file to ensure all dependencies are installed in the deployment environment.

---

## 🧩 Project Components

### `app.py`
The main Streamlit interface responsible for image uploads, URL input, similarity queries, and result visualization.

### `extract_embeddings.py`
Script to compute embeddings for all dataset images. This creates the `embeddings.npy` file used for real-time similarity matching.

### `model.py`
Defines the image feature extraction model (e.g., ResNet, ViT, or custom CNN) used to convert images into vector embeddings.

### `metadata.json`
Stores product-related metadata — such as product names, categories, prices, or descriptions — displayed alongside query results.

### `embeddings.npy`
Binary NumPy array containing precomputed embeddings for fast similarity search (avoids recomputation during runtime).

### `run_preprocess.sh`
Shell script automating the preprocessing pipeline, including image validation, embedding extraction, and dataset setup.

---

## 🧰 Requirements

- Python **3.10+**
- **Streamlit** for the web interface
- **Pillow (PIL)** for image processing
- **NumPy** for numerical computations
- **scikit-learn** for nearest neighbors similarity search
- Additional dependencies listed in `requirements.txt`

Install all via:
pip install -r requirements.txt

text

---

## 🤝 Contributing

Contributions are welcome!  
Follow these steps to get started:

1. Fork the repository
2. Create a new branch
git checkout -b feature/my-feature

3. Make your changes and commit
git commit -m "Add new feature"

4. Push to your branch
git push origin feature/my-feature

5. Open a Pull Request on GitHub
text

---

## 🪪 License

This project is released under the **MIT License**.  
You’re free to use, modify, and distribute it for personal or commercial purposes.

---

## 📬 Contact

If you have questions, feedback, or feature requests, feel free to open an issue or reach out via GitHub.
