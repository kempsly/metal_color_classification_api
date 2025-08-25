# Metal Color Classification API

This API classifies images of metals into one of eight categories: silver, white\_gold, platinum, and others. Built using TensorFlow and Flask, it leverages a MobileNetV3-based deep learning model trained on a custom dataset.

---

## 🚀 Features

* **Image Classification**: Classify metal images into 8 predefined categories.
* **REST API**: Accessible via HTTP POST requests.
* **Model Formats**: Supports both `.keras` and `.h5` formats.
* **Deployment**: Optimized for deployment on Render.

---

## 📦 Prerequisites

* Python 3.10+
* TensorFlow 2.10+
* Flask
* Gunicorn (for production)
* Render account for deployment([syveco.com][1], [1st Coast Metal Roofing Supply][2])

---

## 🛠 Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/kempsly/metal_color_classification_api
   cd metal_color_classification_api
   ```



2. **Set Up Virtual Environment**:

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```



3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```



4. **Download Model Weights**:

   Ensure you have the trained model files (`metal_model.keras` or `metal_model.h5`) in the project directory.

---

## 🧪 Testing Locally

Run the Flask application:

```bash
python app.py
```



The API will be accessible at `http://127.0.0.1:5001`.

To test the API, use the following `curl` command:

```bash
curl -X POST -H "Content-Type: application/json" -d '{"url": "https://example.com/metal_image.jpg"}' http://127.0.0.1:5001/predict
```



Replace the URL with the path to your test image.

---

## 📁 Project Structure

```
metal_color_classification_api/
├── app.py                # Flask application
├── model.py              # Model loading and prediction logic
├── requirements.txt      # Python dependencies
├── Procfile              # Deployment configuration for Render
└── README.md             # Project documentation
```



---

## 🚀 Deployment on Render

1. **Create a New Web Service**:

   * Go to [Render](https://render.com/).

   * Click on "New" and select "Web Service".

   * Connect your GitHub repository.

   * Choose the branch to deploy.

   * Set the environment to `Python 3.10`.

   * Set the build command to:

     ```bash
     pip install -r requirements.txt
     ```

   * Set the start command to:

     ```bash
     gunicorn app:app
     ```

2. **Deploy**:

   * Click "Create Web Service".
   * Render will build and deploy your application.([GitHub][3], [GitHub][4])

---

## 📡 API Usage

### Endpoint: `/predict`

* **Method**: `POST`

* **Content-Type**: `application/json`

* **Request Body**:

```json
  {
    "url": "https://example.com/metal_image.jpg"
  }
```



* **Response**:

```json
  {
    "predictions": [
      {
        "class": "silver",
        "probability": 0.9545
      },
      {
        "class": "white_gold",
        "probability": 0.0115
      },
      {
        "class": "platinum",
        "probability": 0.0104
      }
    ],
    "success": true
  }
```



### Example `curl` Command:([Stack Overflow][5])

```bash
curl -X POST -H "Content-Type: application/json" -d '{"url": "https://example.com/metal_image.jpg"}' http://your-render-app-url/predict
```



Replace `your-render-app-url` with your deployed application's URL.

---

```



Ensure the `test_dir` points to your test dataset.

---

## 📝 Notes

* Ensure that the model file (`metal_model.keras` or `metal_model.h5`) is included in your deployment.
* The model expects images of size 224x224 pixels.
* The API returns the top 3 predicted classes with their probabilities.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---


