# CLIP + VQGAN Text-to-Image & Video Generation

A deep learning project that combines **OpenAI CLIP** and **VQGAN/Taming Transformers** to generate images from natural-language text prompts.

The project optimizes a VQGAN latent representation using CLIP's text-image similarity. The generated latent representations can then be interpolated to create smooth visual transitions and exported as an MP4 video.

## 🚀 Features

* Text-to-image generation using CLIP + VQGAN
* CLIP-based semantic image optimization
* Positive and negative text prompts
* Random image cropping and augmentation
* GPU acceleration with PyTorch
* Latent-space optimization
* Latent interpolation between generated images
* MP4 video generation from interpolated frames
* Google Colab compatible workflow

## 🧠 Architecture

```text
                 Text Prompt
                     │
                     ▼
              ┌─────────────┐
              │    CLIP     │
              │ Text Encoder│
              └──────┬──────┘
                     │
                     │ Text Embedding
                     ▼
              ┌─────────────┐
              │  Similarity │
              │    Loss     │
              └──────┬──────┘
                     │
                     ▼
              ┌─────────────┐
              │   VQGAN     │
              │  Decoder    │
              └──────┬──────┘
                     │
                     ▼
              Generated Image
                     │
                     ▼
             Latent Interpolation
                     │
                     ▼
                Video (.mp4)
```

## 🛠️ Technologies

* Python
* PyTorch
* Torchvision
* OpenAI CLIP
* VQGAN
* Taming Transformers
* NumPy
* Pandas
* Matplotlib
* Pillow
* OmegaConf
* PyYAML
* ImageIO
* Google Colab
* NVIDIA CUDA

## 📂 Project Structure

```text
clip-vqgan-text-to-image-video/
│
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── notebooks/
│   └── clip_vqgan_text_to_image.ipynb
│
├── src/
│   ├── clip_utils.py
│   ├── vqgan_utils.py
│   ├── image_utils.py
│   ├── optimization.py
│   └── video_utils.py
│
├── outputs/
│   ├── images/
│   └── videos/
│
├── assets/
│   └── screenshots/
│
└── models/
    └── README.md
```

## ⚙️ How It Works

### 1. Load CLIP

The project loads the `ViT-B/32` CLIP model and uses its text and image encoders to measure semantic similarity.

### 2. Load VQGAN

A pretrained VQGAN model from the Taming Transformers architecture is loaded.

### 3. Initialize Latent Representation

A trainable latent tensor is initialized and optimized using AdamW.

### 4. Encode the Prompt

The input text prompt is converted into a CLIP text embedding.

Example:

```python
include = ["A BLUE TREE IN THE FOREST"]
exclude = "watermark"
```

### 5. Generate Image

The VQGAN decoder converts the optimized latent representation into an image.

### 6. CLIP Optimization

The generated image is compared with the text embedding using cosine similarity.

The optimization encourages the generated image to become semantically similar to the input prompt while penalizing unwanted concepts.

### 7. Generate Multiple Concepts

The notebook experiments with prompts such as:

```text
A BLUE TREE IN THE FOREST
KIDS PLAYING IN MOON
FLOWERS DANCING
```

### 8. Latent Interpolation

The generated latent vectors are interpolated to create smooth transitions between generated images.

```text
Image A
   │
   ▼
Latent Space
   │
   ├── Interpolation
   │
   ▼
Image B
```

### 9. Generate Video

The interpolated frames are converted into an MP4 video at 25 FPS.

## 💻 Installation

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

For GPU execution, install a PyTorch version compatible with your CUDA environment.

## ▶️ Running the Project

The easiest way to run the project is through Google Colab.

Open:

```text
notebooks/clip_vqgan_text_to_image.ipynb
```

Run the notebook cells sequentially.

A CUDA-enabled GPU is recommended because CLIP + VQGAN optimization is computationally intensive.

## 🎨 Example Prompts

```text
A BLUE TREE IN THE FOREST
```

```text
KIDS PLAYING IN MOON
```

```text
FLOWERS DANCING
```

Negative prompt:

```text
watermark
```

## 🎥 Video Generation

The project stores generated latent representations and interpolates between them.

The interpolation frames are generated at:

```text
25 FPS
```

and exported as:

```text
video.mp4
```

## 📊 Key Concepts Demonstrated

* Generative AI
* Multimodal deep learning
* Text-image embeddings
* CLIP
* VQGAN
* Transformer-based architectures
* Latent-space optimization
* Cosine similarity
* Image augmentation
* GPU acceleration
* Latent interpolation
* AI-generated video

## 📚 Learning Outcomes

This project demonstrates how a text representation can be connected to visual generation using a multimodal embedding model.

It also provides practical experience with:

* PyTorch model loading
* GPU computation
* pretrained generative models
* optimization of latent representations
* image transformations
* semantic similarity
* video frame generation

## ⚠️ Notes

The original notebook is designed primarily for a GPU-enabled Google Colab environment.

Model checkpoints can be very large, so they should **not be committed directly to GitHub**. Instead, document the download/setup process in `models/README.md`.

Similarly, generated videos and large output files should generally be excluded from Git using `.gitignore`.

## 📌 Future Improvements

* [ ] Build a Streamlit web interface
* [ ] Add configurable prompts
* [ ] Add configurable negative prompts
* [ ] Add image download functionality
* [ ] Add batch generation
* [ ] Add more VQGAN checkpoints
* [ ] Improve latent interpolation
* [ ] Add automatic video generation
* [ ] Add experiment tracking
* [ ] Package the project as a reusable Python application

## 👨‍💻 Author

**Manas Ranjan Meher**

MCA Graduate | Aspiring Software Engineer
Python | Data Science | Generative AI | Agentic AI

---

⭐ If you find this project useful, consider giving the repository a star.
