import streamlit as st
import cv2
import numpy as np
from PIL import Image
import img2pdf
import io
from io import BytesIO
import os
import tempfile

# Set app title and description
st.set_page_config(page_title="Advanced Document Scanner", layout="wide")
st.title("📄 Advanced Document Scanner")
st.markdown("""
Upload an image of a document and get a perfectly aligned, enhanced digital copy. 
This tool automatically detects document edges, corrects perspective, and improves image quality.
""")

# Initialize session state
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None

# Sidebar with settings
st.sidebar.title("Processing Settings")
st.sidebar.header("Edge Detection")
detection_threshold = st.sidebar.slider("Edge Detection Threshold", 10, 200, 100, 10,
                                        help="Adjust for better edge detection in low contrast images")

st.sidebar.header("Image Enhancement")
brightness = st.sidebar.slider("Brightness", 0.1, 3.0, 1.0, 0.1)
contrast = st.sidebar.slider("Contrast", 0.1, 3.0, 1.5, 0.1)
sharpness = st.sidebar.slider("Sharpness", 0.0, 3.0, 1.5, 0.1)
noise_reduction = st.sidebar.slider("Noise Reduction", 0, 100, 30, 5)

st.sidebar.header("Output Options")
download_format = st.sidebar.selectbox("Download Format", ["PNG", "JPEG", "PDF"])
show_guides = st.sidebar.checkbox("Show Processing Guides", True)

# Function to apply image enhancements
def enhance_image(image, brightness=1.0, contrast=1.0, sharpness=1.0):
    """Apply brightness, contrast, and sharpness adjustments"""
    # Apply brightness and contrast
    matrix = np.ones(image.shape) * 127.5 * (1 - contrast)
    adjusted = np.clip(contrast * image.astype(np.float32) + matrix + (brightness-1)*128, 0, 255).astype(np.uint8)
    
    # Apply sharpness
    if sharpness != 1.0:
        kernel = np.array([[-1, -1, -1],
                           [-1, 9*sharpness, -1],
                           [-1, -1, -1]])
        adjusted = cv2.filter2D(adjusted, -1, kernel)
    
    return adjusted

# Function to find document contour
def find_document_contours(image, threshold=100):
    """Process image to find document contours"""
    # Convert to grayscale and blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection
    edged = cv2.Canny(blurred, threshold//2, threshold)
    
    # Find contours
    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    
    # Find rectangular contours
    document_contour = None
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        
        if len(approx) == 4:
            document_contour = approx
            break
    
    return document_contour, edged

# Function for perspective correction
def four_point_transform(image, pts):
    """Perspective transform to get top-down view of document"""
    # Order points consistently: top-left, top-right, bottom-right, bottom-left
    rect = pts.reshape(4, 2)
    ordered = np.zeros((4, 2), dtype="float32")
    
    # Top-left: smallest sum, bottom-right: largest sum
    s = rect.sum(axis=1)
    ordered[0] = rect[np.argmin(s)]
    ordered[2] = rect[np.argmax(s)]
    
    # Top-right: smallest difference, bottom-left: largest difference
    diff = np.diff(rect, axis=1)
    ordered[1] = rect[np.argmin(diff)]
    ordered[3] = rect[np.argmax(diff)]
    
    # Calculate dimensions
    top_width = np.linalg.norm(ordered[0] - ordered[1])
    bottom_width = np.linalg.norm(ordered[3] - ordered[2])
    max_width = max(int(top_width), int(bottom_width))
    
    top_height = np.linalg.norm(ordered[0] - ordered[3])
    bottom_height = np.linalg.norm(ordered[1] - ordered[2])
    max_height = max(int(top_height), int(bottom_height))
    
    # Destination points for top-down view
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")
    
    # Apply perspective transformation
    matrix = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
    
    return warped

# Image upload and processing
uploaded_file = st.file_uploader("Upload document image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    cv_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.session_state.uploaded_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    
    # Display uploaded image and find document
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Document")
        st.image(st.session_state.uploaded_image, use_column_width=True)
        
        # Show guides
        if show_guides:
            st.info("""
            **Tips for best results:**
            1. Place document on a contrasting background
            2. Ensure all corners are visible
            3. Avoid shadows and glare
            4. Capture from directly above
            """)

    with col2:
        # Find document contour
        contour, edge_map = find_document_contours(cv_img, detection_threshold)
        contour_found = contour is not None
        
        if contour_found:
            # Draw contour on image
            contour_img = cv_img.copy()
            cv2.drawContours(contour_img, [contour], -1, (0, 0, 255), 3)
            
            # Apply perspective correction
            warped = four_point_transform(cv_img, contour)
            
            # Apply enhancements
            enhanced = enhance_image(warped, brightness, contrast, sharpness)
            
            # Noise reduction
            enhanced = cv2.fastNlMeansDenoisingColored(
                enhanced, 
                None, 
                noise_reduction, 
                noise_reduction, 
                7, 
                21
            )
            
            # Convert to RGB for display
            enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
            st.session_state.processed_image = enhanced_rgb
            
            # Process steps visualization
            st.subheader("Processing Steps")
            
            tab1, tab2, tab3 = st.tabs(["Edge Detection", "Document Outline", "Enhanced Document"])
            
            with tab1:
                st.image(edge_map, caption="Edge Detection Map", use_column_width=True, clamp=True)
            
            with tab2:
                contour_img_rgb = cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB)
                st.image(contour_img_rgb, caption="Detected Document Outline", use_column_width=True)
                
                # Show contour info
                st.success(f"Document contour successfully detected with 4 corners")
            
            with tab3:
                st.image(enhanced_rgb, caption="Enhanced Document", use_column_width=True)
                
                # Create download buttons
                st.subheader("Download Enhanced Document")
                
                if download_format == "PNG":
                    buf = io.BytesIO()
                    enhanced_pil = Image.fromarray(enhanced_rgb)
                    enhanced_pil.save(buf, format="PNG")
                    st.download_button(
                        label="Download PNG",
                        data=buf.getvalue(),
                        file_name="document.png",
                        mime="image/png",
                        use_container_width=True
                    )
                elif download_format == "JPEG":
                    buf = io.BytesIO()
                    enhanced_pil = Image.fromarray(enhanced_rgb)
                    enhanced_pil.save(buf, format="JPEG", quality=95)
                    st.download_button(
                        label="Download JPEG",
                        data=buf.getvalue(),
                        file_name="document.jpg",
                        mime="image/jpeg",
                        use_container_width=True
                    )
                elif download_format == "PDF":
                    # Convert OpenCV image (BGR) to grayscale
                    enhanced_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

                    # Convert grayscale OpenCV image to PIL Image
                    pil_image = Image.fromarray(enhanced_gray)

                    # Convert PIL image to bytes (in RGB format, since img2pdf expects RGB)
                    # img2pdf works best with RGB images
                    img_buffer = BytesIO()
                    pil_image.save(img_buffer, format="JPEG")
                    img_buffer.seek(0)

                    # Convert image to PDF using img2pdf
                    pdf_bytes = img2pdf.convert([img_buffer.getvalue()])  # list of image bytes


                    # Create download button
                    st.download_button(
                        label="Download PDF",
                        data=pdf_bytes,
                        file_name="document.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                
                st.success("Document processed successfully!")
        else:
            st.warning("Could not detect document edges. Try adjusting the threshold.")
            st.image(edge_map, caption="Edge Detection Map", use_column_width=True, clamp=True)
            st.info("""
            **Detection Tips:**
            1. Increase detection threshold for faint edges
            2. Ensure the document has clear borders
            3. Try different image contrast
            """)

# Sample images and tutorials
if uploaded_file is None:
    st.markdown("## How It Works")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image("https://images.unsplash.com/photo-1507673192339-e405ca69a349?auto=format&fit=crop&q=80&w=1000", 
                 caption="1. Upload Document Photo")
        st.markdown("Take a photo of your document with your smartphone or camera")

    with col2:
        st.image("https://cdn.pixabay.com/photo/2017/05/20/20/33/contour-2329969_1280.jpg", 
                 caption="2. Automatic Edge Detection")
        st.markdown("Our algorithm finds the document borders and corrects perspective")

    with col3:
        st.image("https://media.istockphoto.com/id/962130962/vector/clean-document-scan.jpg?s=612x612&w=0&k=20&c=Rb0_VEYVw-LSlNEzWCw9Qykm7ZWU_W2mkKb3yl8wQvY=", 
                 caption="3. Enhanced Digital Result")
        st.markdown("Download a clean, enhanced digital document")

    st.markdown("""
    ## Perfect for Scanning:
    - Receipts and invoices
    - Contracts and legal documents
    - Notes and handwritten pages
    - Printed photos and artwork
    - Book pages and magazine articles
    """)
