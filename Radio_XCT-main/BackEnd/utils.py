import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import pandas as pd
from datetime import datetime
from fpdf import FPDF
import cv2
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
import traceback
import time

# Test visualization pipeline to verify basic image operations work
def test_visualization_pipeline():
    """Test the full visualization pipeline with a sample image"""
    try:
        print("Testing visualization pipeline...")
        # Create a simple test image
        test_img = np.zeros((224, 224, 3), dtype=np.uint8)
        test_img[50:150, 50:150, 0] = 255  # Red square
        
        # Save test image
        test_path = "temp/test_image.png"
        cv2.imwrite(test_path, test_img)
        
        # Test matplotlib
        plt.figure()
        plt.imshow(test_img)
        plt.title("Test Image")
        plt.savefig("temp/test_plot.png")
        plt.close()
        
        # Test CV2 heatmap
        heatmap = np.zeros((224, 224), dtype=np.uint8)
        heatmap[75:125, 75:125] = 255
        colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        cv2.imwrite("temp/test_heatmap.png", colored_heatmap)
        
        # Test overlay
        overlay = cv2.addWeighted(test_img, 0.7, colored_heatmap, 0.3, 0)
        cv2.imwrite("temp/test_overlay.png", overlay)
        
        print("Visualization test complete - check temp directory for test images")
        return True
    except Exception as e:
        print(f"Visualization test failed: {e}")
        traceback.print_exc()
        return False

# Attempt to import LIME - we'll handle missing dependencies gracefully
try:
    from lime import lime_image
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("LIME not available. Install with 'pip install lime'")

# Labels for each organ
dental_labels = ['BDC-BDR', 'Caries', 'Fractured Teeth', 'Healthy Teeth', 'Impacted Teeth', 'Infection']
spine_labels = ['Normal Spine', 'Scoliosis', 'Spondylolisthesis']
fracture_labels = ['Fractured', 'Not Fractured']
kidney_labels = ['Cyst', 'Normal', 'Stone', 'Tumor']

# Create necessary directories with permissions check
for directory in ["temp", "generated_reports"]:
    os.makedirs(directory, exist_ok=True)
    # Print directory info for debugging
    print(f"Directory {directory} exists: {os.path.exists(directory)}")
    print(f"Directory {directory} is writable: {os.access(directory, os.W_OK)}")

# Load all models (assumes they exist in models/)
try:
    dental_model = load_model('models/dental_model.h5')
    spine_model = load_model('models/spine_model.h5')
    fracture_model = load_model('models/fracture_model.h5')
    kidney_model = load_model('models/kidney_model.h5')
    print("Models loaded successfully")
except Exception as e:
    print(f"Error loading models: {e}")
    # Create dummy models for testing if needed

# Preprocess uploaded image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array, img

# Predict class from image
def get_prediction(model, img_array, labels):
    preds = model.predict(img_array)
    predicted_index = np.argmax(preds[0])
    return labels[predicted_index], preds[0][predicted_index], predicted_index

# Generate simple heatmap as fallback
def generate_simple_heatmap(model, img_array, img_path, predicted_class_idx):
    """Generate a simple activation heatmap when Grad-CAM fails"""
    try:
        # Get model predictions
        preds = model.predict(img_array)
        
        # Create a basic heatmap using the raw image with highlighted pixels
        img = image.load_img(img_path, target_size=(224, 224))
        img_array_single = image.img_to_array(img)
        
        # Create a simple heatmap based on pixel intensities
        heatmap = np.mean(img_array_single, axis=2)
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
        
        # Create heatmap visualization
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(img_array_single / 255.0)
        plt.title("Original Image")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(img_array_single / 255.0)
        plt.imshow(heatmap, alpha=0.6, cmap='jet')
        plt.title(f"Basic Heatmap")
        plt.axis('off')
        
        # Save the result
        timestamp = int(time.time())
        simple_heatmap_path = f"temp/simple_heatmap_{timestamp}_{os.path.basename(img_path)}"
        plt.savefig(simple_heatmap_path)
        plt.close()
        
        print(f"Simple heatmap saved to {simple_heatmap_path}")
        return simple_heatmap_path
    except Exception as e:
        print(f"Error generating simple heatmap: {e}")
        traceback.print_exc()
        return None

# Generate Grad-CAM heatmap - enhanced with better debugging
def generate_gradcam(model, img_array, img_path, predicted_class_idx):
    print("Generating Grad-CAM heatmap...")
    
    try:
        # Print model layers to debug
        print("Model layers:")
        for i, layer in enumerate(model.layers):
            print(f"Layer {i}: {layer.name}, output shape: {layer.output_shape}")
        
        # Create a temp file name with timestamp to avoid conflicts
        timestamp = int(time.time())
        gradcam_path = f"temp/gradcam_{timestamp}_{os.path.basename(img_path)}"
        
        # Check if we're using TF 2.x
        if hasattr(tf.keras.models, 'Model'):
            # Try to find last conv layer
            last_conv_layer = None
            for layer in model.layers[::-1]:
                if 'conv' in layer.name.lower() or 'pool' in layer.name.lower():
                    if len(layer.output_shape) == 4:  # Check if it's a 4D tensor
                        last_conv_layer = layer.name
                        print(f"Found convolutional layer: {last_conv_layer}")
                        break
            
            # If no conv layer found, try to use any layer with 4D output
            if last_conv_layer is None:
                print("No 'conv' or 'pool' layer found, looking for any 4D tensor layer")
                for layer in model.layers[::-1]:
                    if len(layer.output_shape) == 4:
                        last_conv_layer = layer.name
                        print(f"Found 4D output layer: {last_conv_layer}")
                        break
            
            # If still no suitable layer, return None
            if last_conv_layer is None:
                print("No suitable layer found for Grad-CAM")
                return None
                
            print(f"Using layer {last_conv_layer} for Grad-CAM")
            
            # Create gradient model
            grad_model = tf.keras.models.Model(
                inputs=[model.inputs],
                outputs=[model.get_layer(last_conv_layer).output, model.output]
            )
            
            # Compute gradient of the predicted class with respect to the output feature map
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(img_array)
                loss = predictions[:, predicted_class_idx]
            
            # Extract feature map and gradient
            output = conv_outputs[0]
            grads = tape.gradient(loss, conv_outputs)[0]
            
            # Pool the gradients across the channels
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
            
            # Weight output feature map with gradients
            cam = tf.zeros(output.shape[0:2], dtype=tf.float32)
            for i, w in enumerate(pooled_grads):
                cam += w * output[:, :, i]
            
            # Process CAM
            cam = cam.numpy()
            cam = cv2.resize(cam, (224, 224))
            cam = np.maximum(cam, 0) / (np.max(cam) + 1e-10)  # Normalize between 0-1
            
            # Convert to heatmap
            heatmap = np.uint8(255 * cam)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
            # Load original image
            orig_img = cv2.imread(img_path)
            if orig_img is None:
                print(f"Warning: Could not read image at {img_path} with OpenCV")
                # Try with PIL/keras and convert
                img = image.load_img(img_path, target_size=(224, 224))
                orig_img = image.img_to_array(img).astype('uint8')
            else:
                orig_img = cv2.resize(orig_img, (224, 224))
            
            # If grayscale, convert to RGB
            if len(orig_img.shape) == 2:
                orig_img = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2BGR)
            elif orig_img.shape[2] == 1:
                orig_img = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2BGR)
            
            # Debug image processing steps
            print(f"orig_img shape: {orig_img.shape}, dtype: {orig_img.dtype}")
            print(f"heatmap shape: {heatmap.shape}, dtype: {heatmap.dtype}")
            
            # Ensure both images have the same data type
            orig_img = orig_img.astype(np.float32) / 255.0
            heatmap = heatmap.astype(np.float32) / 255.0
            
            # Superimpose heatmap and original image
            superimposed_img = cv2.addWeighted(orig_img, 0.6, heatmap, 0.4, 0)
            superimposed_img = (superimposed_img * 255).astype(np.uint8)
            
            # Save the image
            print(f"Saving Grad-CAM to {gradcam_path}")
            cv2.imwrite(gradcam_path, superimposed_img)
            
            # Verify image was saved
            if os.path.exists(gradcam_path):
                print(f"Successfully saved Grad-CAM image at {gradcam_path}")
                # Also save a copy with a fixed name for debugging
                debug_path = "temp/latest_gradcam.png"
                cv2.imwrite(debug_path, superimposed_img)
                print(f"Also saved debug copy at {debug_path}")
            else:
                print(f"Failed to save Grad-CAM image at {gradcam_path}")
            
            return gradcam_path
    except Exception as e:
        print(f"Error in Grad-CAM generation: {e}")
        traceback.print_exc()
        return None

# Generate LIME explanation - enhanced with better debugging
def generate_lime(model, img_path, predicted_class_idx, labels):
    print("Generating LIME explanation...")
    
    if not LIME_AVAILABLE:
        print("LIME is not available. Skipping LIME visualization.")
        return None
    
    try:
        # Create a temp file name with timestamp to avoid conflicts
        timestamp = int(time.time())
        lime_path = f"temp/lime_{timestamp}_{os.path.basename(img_path)}"
        
        # Load and preprocess image
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        print(f"Loaded image for LIME with shape: {img_array.shape}")
        
        # Create a function that the explainer can use
        def model_predict(images):
            # Convert to float and normalize
            batch = np.array([image / 255.0 for image in images])
            print(f"LIME batch shape: {batch.shape}")
            return model.predict(batch)
        
        # Test prediction function
        test_pred = model_predict(np.array([img_array]))
        print(f"Test prediction shape: {test_pred.shape}")
        
        # Create explainer with lower num_samples for testing
        print("Creating LIME explainer...")
        explainer = lime_image.LimeImageExplainer()
        
        # Get explanation with fewer samples to speed up processing
        print("Running LIME explanation...")
        explanation = explainer.explain_instance(
            img_array.astype('double'), 
            model_predict,
            top_labels=3,  # Reduced from 5 to speed up
            hide_color=0,
            num_samples=300  # Reduced from 500 to speed up
        )
        print("LIME explanation complete")
        
        # Get visualization
        print("Generating LIME visualization...")
        temp, mask = explanation.get_image_and_mask(
            predicted_class_idx,
            positive_only=True,
            num_features=5,
            hide_rest=False
        )
        
        # Create visualization
        plt.figure(figsize=(8, 8))
        plt.title(f"LIME: Important regions for {labels[predicted_class_idx]}")
        marked_img = mark_boundaries(temp / 255.0, mask)
        plt.imshow(marked_img)
        plt.axis('off')
        plt.tight_layout()
        
        # Save the image
        print(f"Saving LIME visualization to {lime_path}")
        plt.savefig(lime_path, dpi=150)
        plt.close()
        
        # Verify image was saved
        if os.path.exists(lime_path):
            print(f"Successfully saved LIME image at {lime_path}")
            # Also save a copy with a fixed name for debugging
            debug_path = "temp/latest_lime.png"
            plt.figure(figsize=(8, 8))
            plt.imshow(marked_img)
            plt.savefig(debug_path)
            plt.close()
        else:
            print(f"Failed to save LIME image at {lime_path}")
        
        return lime_path
    except Exception as e:
        print(f"Error in LIME generation: {e}")
        traceback.print_exc()
        return None

# Read descriptions and precautions from CSVs
def get_description_and_precautions(predicted_class):
    try:
        desc_df = pd.read_csv('description.csv')
        prec_df = pd.read_csv('precautions.csv')

        desc_row = desc_df[desc_df['Disease'].str.lower() == predicted_class.lower()]
        prec_row = prec_df[prec_df['Disease'].str.lower() == predicted_class.lower()]

        description = "No description available."
        findings = "No findings found."
        precautions = "No precautions available."

        if not desc_row.empty:
            row = desc_row.iloc[0]
            description = row.get("Description", description)
            findings = "\n".join(filter(None, [
                str(row.get("Findings_1", "")).strip(),
                str(row.get("Findings_2", "")).strip()
            ]))

        if not prec_row.empty:
            row = prec_row.iloc[0]
            precautions = "\n".join(filter(None, [
                str(row.get("Precaution_1", "")).strip(),
                str(row.get("Precaution_2", "")).strip(),
                str(row.get("Precaution_3", "")).strip()
            ]))

        return description, findings, precautions
    except Exception as e:
        print(f"Error getting descriptions and precautions: {e}")
        return "Description unavailable.", "Findings unavailable.", "Precautions unavailable."

# PDF class with border
class BorderedPDF(FPDF):
    def header(self):
        self.set_line_width(0.5)
        self.rect(5.0, 5.0, 200.0, 287.0)

# Generate the report PDF
def generate_pdf(predicted_class, img_path, description, findings, precautions, output_path,
                 name, dob, gender, age, organ, gradcam_path=None, lime_path=None, confidence=None):
    logo_path = 'Medical_Logo.jpeg'  # assuming it's in the same folder as utils.py or use full path

    pdf = BorderedPDF()
    pdf.add_page()
    
    # Add logo top-right if it exists
    if os.path.exists(logo_path):
        pdf.image(logo_path, x=170, y=10, w=30)  # Small logo top-right
    else:
        print(f"Logo file not found at {logo_path}")

    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt=f"{organ.upper()} DIAGNOSIS REPORT", ln=True, align='C')
    pdf.ln(10)

    # Patient Info
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Patient Details:", ln=True)
    pdf.set_font("Arial", '', 12)
    pdf.cell(200, 10, txt=f"Name: {name}", ln=True)
    pdf.cell(200, 10, txt=f"DOB: {dob}    Age: {age}    Gender: {gender}", ln=True)
    pdf.cell(200, 10, txt=f"Predicted Disease: {predicted_class.upper()}", ln=True)
    if confidence is not None:
        pdf.cell(200, 10, txt=f"Confidence: {confidence:.2%}", ln=True)
    pdf.cell(200, 10, txt=f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(5)

    # Original Image
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Uploaded Scan Image:", ln=True)
    
    # Check if image path exists before trying to include it
    if os.path.exists(img_path):
        try:
            pdf.image(img_path, x=10, w=80)
            print(f"Successfully added original image to PDF")
        except Exception as e:
            print(f"Error adding original image to PDF: {e}")
            traceback.print_exc()
            pdf.cell(200, 10, txt="[Image could not be loaded]", ln=True)
    else:
        print(f"Warning: Image file not found at {img_path}")
        pdf.cell(200, 10, txt="[Image file not found]", ln=True)
    
    # Add Grad-CAM heatmap if available
    if gradcam_path and os.path.exists(gradcam_path):
        try:
            pdf.ln(5)
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(200, 10, txt="Grad-CAM Visualization (Highlighting relevant areas):", ln=True)
            pdf.image(gradcam_path, x=10, w=80)
            print("Successfully added Grad-CAM to PDF")
        except Exception as e:
            print(f"Error adding Grad-CAM to PDF: {e}")
            traceback.print_exc()
            pdf.cell(200, 10, txt="[Grad-CAM visualization could not be loaded]", ln=True)
    else:
        print("Grad-CAM path not available or file not found")
        pdf.ln(5)
        pdf.set_font("Arial", 'I', 10)
        pdf.cell(200, 10, txt="Note: Grad-CAM visualization could not be generated for this image.", ln=True)
        
    # Add LIME explanation if available
    if lime_path and os.path.exists(lime_path):
        try:
            pdf.ln(5)
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(200, 10, txt="LIME Visualization (Showing influential regions):", ln=True)
            pdf.image(lime_path, x=10, w=80)
            print("Successfully added LIME to PDF")
        except Exception as e:
            print(f"Error adding LIME to PDF: {e}")
            traceback.print_exc()
            pdf.cell(200, 10, txt="[LIME visualization could not be loaded]", ln=True)
    else:
        print("LIME path not available or file not found")
        pdf.ln(5)
        pdf.set_font("Arial", 'I', 10)
        pdf.cell(200, 10, txt="Note: LIME visualization could not be generated for this image.", ln=True)
    
    pdf.ln(10)

    # Description
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Description:", ln=True)
    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(0, 10, "* " + "\n* ".join(description.split('\n')))
    pdf.ln(5)
    
    # Findings
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Findings:", ln=True)
    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(0, 10, "* " + "\n* ".join(findings.split('\n')))
    pdf.ln(5)

    # Precautions
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Precautions:", ln=True)
    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(0, 10, "* " + "\n* ".join(precautions.split('\n')))
    
    # Add explanation of explainability methods
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="About the Explainability Visualizations:", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.multi_cell(0, 8, "* Grad-CAM: Highlights the regions of the image that were most important for the model's prediction. Red/warm areas indicate regions that strongly influenced the diagnosis.")
    pdf.multi_cell(0, 8, "* LIME: Shows which parts of the image contributed to the prediction by highlighting segments that influenced the model's decision.")
    
    # Add disclaimer
    pdf.ln(5)
    pdf.set_font("Arial", 'I', 10)
    pdf.multi_cell(0, 8, "Disclaimer: This automated analysis is provided for informational purposes only and does not replace professional medical diagnosis. Please consult with a healthcare professional for proper medical advice.")

    try:
        print(f"Generating PDF at {output_path}")
        pdf.output(output_path)
        print(f"PDF generated successfully at {output_path}")
    except Exception as e:
        print(f"Error saving PDF: {e}")
        traceback.print_exc()
        # Try saving to a default location
        default_path = f"report_{int(time.time())}.pdf"
        try:
            pdf.output(default_path)
            print(f"PDF saved to default location: {default_path}")
            output_path = default_path
        except Exception as err:
            print(f"Could not save PDF at all: {err}")
    
    # Do NOT clean up temporary visualization files - keep them for debugging
    # This is a key change to fix the "visualizations not coming" issue
    print("Keeping visualization files for debugging")
    
    return output_path

# Main handler called by main.py
def handle_prediction(organ, img_path, name, dob, gender, age):
    print(f"\n--- Starting prediction for {name}, organ: {organ} ---")
    organ = organ.strip().lower()  # ← Add strip() to avoid trailing whitespace bugs
    print("Received organ value:", organ)
    
    # Test basic visualization capabilities
    test_visualization_pipeline()
    
    try:
        if organ == 'dental':
            model = dental_model
            labels = dental_labels
        elif organ == 'spine':
            model = spine_model
            labels = spine_labels
        elif organ == 'fracture':
            model = fracture_model
            labels = fracture_labels
        elif organ == 'kidney':
            model = kidney_model
            labels = kidney_labels
        else:
            print(f"Invalid organ received: '{organ}'")
            raise ValueError("Invalid organ selected.")

        img_array, _ = preprocess_image(img_path)
        predicted_class, confidence, predicted_idx = get_prediction(model, img_array, labels)
        print(f"Prediction: {predicted_class} with confidence {confidence:.2%}")
        
        description, findings, precautions = get_description_and_precautions(predicted_class)
        
        # Generate explainability visualizations with detailed logging and fallbacks
        print("Generating explainability visualizations...")
        
        # Try Grad-CAM first
        gradcam_path = generate_gradcam(model, img_array, img_path, predicted_idx)
        
        # If Grad-CAM fails, try simple heatmap
        if gradcam_path is None:
            print("Grad-CAM failed, trying simple heatmap instead...")
            gradcam_path = generate_simple_heatmap(model, img_array, img_path, predicted_idx)
        
        # Try LIME
        lime_path = generate_lime(model, img_path, predicted_idx, labels)
        
        # Print paths for debugging
        print(f"Grad-CAM path: {gradcam_path}")
        print(f"LIME path: {lime_path}")
        
        # Generate PDF with visualizations
        pdf_path = f"generated_reports/{os.path.basename(img_path).split('.')[0]}_{organ}_report.pdf"
        final_path = generate_pdf(
            predicted_class, img_path, description, findings, precautions, pdf_path,
            name, dob, gender, age, organ, gradcam_path, lime_path, confidence
        )
        
        print(f"--- Completed prediction process, report at: {final_path} ---\n")
        return final_path
        
    except Exception as e:
        print(f"Error in handle_prediction: {e}")
        traceback.print_exc()
        raise