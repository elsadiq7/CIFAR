import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model as lm
from PIL import Image
import plotly.graph_objects as go

# Load your trained CIFAR model
model = lm('models/build_model_1_v2.keras')

# Define the CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


# Function to preprocess the image and predict the class
def classify_image(image):
    # Ensure the image is in the right format
    image = Image.fromarray(image).convert('RGB')

    # Resize the image to (32, 32) as CIFAR-10 uses 32x32 images
    image = image.resize((32, 32))

    # Convert the image to an array and preprocess it
    image_array = np.array(image).astype(np.float32) / 255.0  # Normalize to [0, 1]

    # Expand dimensions to match model input shape (1, 32, 32, 3)
    image_array = np.expand_dims(image_array, axis=0)

    # Get predictions
    predictions = model.predict(image_array)[0]  # Get the prediction array for the first image

    # Get the predicted class and its confidence
    predicted_class_idx = np.argmax(predictions)
    predicted_class = class_names[predicted_class_idx]
    predicted_confidence = predictions[predicted_class_idx] * 100  # Convert to percentage

    # Print predicted class and confidence
    predicted_info = f"Predicted Class: {predicted_class} with {predicted_confidence:.2f}% confidence."

    # Create a Plotly bar chart for class confidence levels
    fig = go.Figure(go.Bar(
        x=predictions * 100,  # Convert probabilities to percentages
        y=class_names,
        orientation='h',
        marker=dict(color='skyblue'),
        text=[f"{conf:.1f}%" for conf in predictions * 100],  # Show percentage labels
        hoverinfo="text"
    ))

    # Update layout for better presentation
    fig.update_layout(
        title="Class Confidence Levels",
        xaxis_title="Confidence (%)",
        yaxis_title="Classes",
        xaxis=dict(range=[0, 100]),  # Set x-axis to 0-100%
        yaxis=dict(categoryorder='total ascending'),  # Sort bars by confidence
        bargap=0.2
    )

    return predicted_info, fig


# Define the Gradio interface
interface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="numpy"),
    outputs=["text", gr.Plot()],
    title="CIFAR Image Classification",
    description="Upload an image, and the model will classify it as one of the CIFAR-10 classes."
)

# Launch the interface
interface.launch()





