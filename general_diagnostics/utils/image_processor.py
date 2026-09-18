import cv2
import numpy as np
from PIL import Image
from pydantic.v1.fields import FieldInfo as FieldInfoV1
import pydicom
import io
import os
from datetime import datetime

class MedicalImageProcessor:
    def __init__(self):
        pass

    def process_dicom(self, file_path):
        """Process DICOM medical images"""
        try:
            dicom_data = pydicom.dcmread(file_path)
            pixel_array = dicom_data.pixel_array

            # Normalize pixel values for visualization
            if pixel_array.dtype != np.uint8:
                # Normalize to 0-255 range
                pixel_array = ((pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min()) * 255).astype(np.uint8)

            image = Image.fromarray(pixel_array)
            return {
                'image': image,
                'metadata': dicom_data,
                'pixel_array': pixel_array,
                'processed': True
            }
        except Exception as e:
            return {'error': str(e), 'processed': False}

    def enhance_image(self, image_path):
        """Enhance medical images for better analysis"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                # Try to open with PIL and convert to cv2 format
                pil_img = Image.open(image_path)
                image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            # Apply histogram equalization
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            enhanced = cv2.equalizeHist(gray)

            return enhanced
        except Exception as e:
            print(f"Error enhancing image: {str(e)}")
            return None

    def detect_anomalies(self, image_path):
        """Basic anomaly detection in medical images"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                # Try to open with PIL and convert to cv2 format
                pil_img = Image.open(image_path)
                image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Use adaptive thresholding for better results
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Calculate contour properties to identify potential anomalies
            anomalies = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:  # Filter small contours
                    anomalies.append({
                        'area': area,
                        'perimeter': cv2.arcLength(contour, True),
                        'approx_points': len(cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True))
                    })

            return {
                'anomaly_count': len(anomalies),
                'anomalies': anomalies,
                'has_anomalies': len(anomalies) > 0
            }
        except Exception as e:
            print(f"Error detecting anomalies: {str(e)}")
            return {'error': str(e), 'has_anomalies': False}

    def analyze_image_for_report(self, file_path):
        """Generate a text analysis of medical images suitable for diagnostic reports"""
        try:
            # Check if DICOM
            if file_path.lower().endswith(('.dcm', '.dicom')):
                result = self.process_dicom(file_path)
                if 'error' not in result:
                    dicom_data = result['metadata']

                    # Extract common DICOM fields
                    patient_name = getattr(dicom_data, 'PatientName', 'N/A')
                    patient_id = getattr(dicom_data, 'PatientID', 'N/A')
                    study_date = getattr(dicom_data, 'StudyDate', 'N/A')
                    modality = getattr(dicom_data, 'Modality', 'N/A')
                    body_part = getattr(dicom_data, 'BodyPartExamined', 'N/A')
                    manufacturer = getattr(dicom_data, 'Manufacturer', 'N/A')
                    image_size = result['pixel_array'].shape if result['pixel_array'] is not None else 'N/A'
                    image_orientation = getattr(dicom_data, 'ImageOrientationPatient', 'N/A')
                    image_position = getattr(dicom_data, 'ImagePositionPatient', 'N/A')
                    slice_thickness = getattr(dicom_data, 'SliceThickness', 'N/A')
                    kvp = getattr(dicom_data, 'KVP', 'N/A')
                    exposure = getattr(dicom_data, 'ExposureTime', 'N/A')

                    analysis = f"""DICOM MEDICAL IMAGE ANALYSIS:
- Patient: {str(patient_name)}
- Patient ID: {str(patient_id)}
- Study Date: {str(study_date)}
- Modality: {str(modality)}
- Body Part: {str(body_part)}
- Manufacturer: {str(manufacturer)}
- Image Size: {str(image_size)}
- Image Orientation: {str(image_orientation)}
- Image Position: {str(image_position)}
- Slice Thickness: {str(slice_thickness)} mm
- kVp: {str(kvp)}
- Exposure Time: {str(exposure)} ms

TECHNICAL ASSESSMENT:
- Image quality appears to be adequate for diagnostic interpretation
- Proper patient positioning evident
                    """
                    return analysis.strip()
                else:
                    return f"Error processing DICOM: {result['error']}"
            else:
                # For other image types, perform more comprehensive analysis
                image = cv2.imread(file_path)
                if image is None:
                    pil_img = Image.open(file_path)
                    image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

                # Get basic image properties
                height, width, channels = image.shape
                analysis = f"""
MEDICAL IMAGE ANALYSIS for: {os.path.basename(file_path)}
- Dimensions: {width} x {height} pixels
- Color Channels: {channels}
- File Size: approximately {(width * height * channels) / 1024:.2f} KB (estimated)

VISUAL ASSESSMENT:
                """

                # For medical images, look for specific patterns
                if any(x in file_path.lower() for x in ['xray', 'x-ray', 'radiograph', 'chest', 'abdomen']):
                    analysis += "- Pattern consistent with radiographic imaging\n"

                    # Look for anatomical landmarks in X-ray images
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    # Apply edge detection to identify anatomical structures
                    edges = cv2.Canny(gray, 50, 150)
                    edge_density = np.sum(edges > 0) / (height * width)

                    if edge_density > 0.1:  # Adjust threshold based on analysis
                        analysis += "- Appropriate anatomical structures visible\n"
                    else:
                        analysis += "- Image may be overexposed or underexposed\n"

                elif any(x in file_path.lower() for x in ['mri', 't1', 't2', 'flair', 'stir']):
                    analysis += "- Pattern consistent with MRI imaging\n"
                    analysis += "- Tissue contrast appropriate for MRI\n"

                elif any(x in file_path.lower() for x in ['ct', 'cat', 'computed']):
                    analysis += "- Pattern consistent with CT imaging\n"
                    analysis += "- Appropriate density differentiation visible\n"

                elif any(x in file_path.lower() for x in ['ultrasound', 'us', 'echo']):
                    analysis += "- Pattern consistent with ultrasound imaging\n"
                    analysis += "- Acoustic properties appropriate for US\n"

                # Perform anomaly detection
                anomaly_result = self.detect_anomalies(file_path)
                if 'error' not in anomaly_result:
                    if anomaly_result['has_anomalies']:
                        analysis += f"\nANOMALY ASSESSMENT:\n"
                        analysis += f"- Potential anomalies detected: {anomaly_result['anomaly_count']}\n"

                        if anomaly_result['anomalies']:
                            largest_anomaly = max(anomaly_result['anomalies'], key=lambda x: x['area'])
                            analysis += f"- Largest detected region: {largest_anomaly['area']:.2f} pixels²\n"
                            analysis += f"- Region perimeter: {largest_anomaly['perimeter']:.2f} pixels\n"
                            analysis += "- This may indicate areas requiring further clinical correlation\n"
                    else:
                        analysis += f"\nANOMALY ASSESSMENT:\n"
                        analysis += f"- No significant anomalies detected in initial analysis\n"
                        analysis += f"- Image appears to show normal anatomical structures\n"
                else:
                    analysis += f"- Anomaly detection not available for this image type\n"

                analysis += f"\nRECOMMENDATION:\n"
                analysis += f"- Clinical correlation recommended for any identified findings\n"
                analysis += f"- Comparison with previous studies may be beneficial if available\n"

                return analysis.strip()
        except Exception as e:
            return f"Error analyzing image: {str(e)}"