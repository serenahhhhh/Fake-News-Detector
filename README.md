# 📰 Fake News Detector

A sleek web application that classifies news as **Real** or **Fake** using a trained LSTM (Long Short-Term Memory) neural network. The app features a modern glassmorphic UI and leverages deep learning through a Flask backend.

---

## 🚀 Features

- 🧠 LSTM-based model for binary news classification
- 💾 Pre-trained model and tokenizer included (no training required)
- 🎨 Aesthetic glassmorphism UI with CSS gradients and smooth animations
- ⚙️ Flask-powered backend
- 💡 Input any news content and receive instant prediction with confidence


---

## 🧠 Model Details

| Component      | Description |
|----------------|-------------|
| **Model**      | LSTM (Keras Sequential) |
| **Framework**  | TensorFlow / Keras |
| **Tokenizer**  | Keras Tokenizer (Pickled) |
| **Files Used** | `lstm_model.h5`, `tokenizer.pkl` |
| **Dataset**    | Real and fake news dataset (e.g., Kaggle) |

---

## 🛠️ Tech Stack

| Layer      | Technology                 |
|------------|----------------------------|
| Frontend   | HTML, CSS (glassmorphism)  |
| Fonts      | Google Fonts – Open Sans   |
| Backend    | Python Flask               |
| ML Model   | Keras (TensorFlow backend) |
| Serialization | Pickle (`tokenizer.pkl`) |


