"""
Advanced color detection utility using multiple algorithms for accurate color identification.
Combines K-means clustering, histogram analysis, and adaptive thresholding.
"""
import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
import webcolors
from scipy import stats


def preprocess_image_for_color_detection(image):
    """
    Preprocess image to improve color detection accuracy.
    
    Args:
        image: BGR image (numpy array)
        
    Returns:
        numpy array: Preprocessed image
    """
    # Apply bilateral filter to reduce noise while preserving edges
    filtered = cv2.bilateralFilter(image, 9, 75, 75)
    
    # Enhance contrast using CLAHE in LAB color space
    lab = cv2.cvtColor(filtered, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return enhanced


def remove_background_colors(image, mask_threshold=0.1):
    """
    Remove likely background colors (edges) to focus on object colors.
    
    Args:
        image: BGR image (numpy array)
        mask_threshold: Threshold for edge detection
        
    Returns:
        numpy array: Image with background pixels masked
    """
    # Create a mask to exclude edge pixels (likely background)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # Dilate edges to create a larger exclusion zone
    kernel = np.ones((3, 3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    
    # Create mask for center region (more likely to be object)
    h, w = image.shape[:2]
    center_mask = np.zeros((h, w), dtype=np.uint8)
    center_h, center_w = int(h * 0.2), int(w * 0.2)
    center_mask[center_h:h-center_h, center_w:w-center_w] = 255
    
    # Combine masks
    final_mask = cv2.bitwise_and(center_mask, cv2.bitwise_not(edges_dilated))
    
    # Apply mask to image
    masked_image = cv2.bitwise_and(image, image, mask=final_mask)
    
    return masked_image, final_mask


def get_dominant_colors_advanced_kmeans(image, k_range=(3, 8)):
    """
    Advanced K-means clustering with optimal k selection.
    
    Args:
        image: BGR image (numpy array)
        k_range: Range of k values to test
        
    Returns:
        list: List of dominant colors as RGB tuples with frequencies
    """
    # Preprocess image
    processed_image = preprocess_image_for_color_detection(image)
    
    # Remove background colors
    masked_image, mask = remove_background_colors(processed_image)
    
    # Get valid pixels (non-zero after masking)
    valid_pixels = masked_image[mask > 0]
    
    if len(valid_pixels) < 50:  # Fallback to original image if too few pixels
        valid_pixels = processed_image.reshape(-1, 3)
    
    # Convert to RGB for sklearn
    rgb_pixels = valid_pixels[:, [2, 1, 0]]  # BGR to RGB
    
    # Find optimal k using elbow method
    best_k = find_optimal_k(rgb_pixels, k_range)
    
    # Apply K-means with optimal k
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(rgb_pixels)
    centers = kmeans.cluster_centers_
    
    # Calculate frequencies
    label_counts = Counter(labels)
    total_pixels = len(labels)
    
    dominant_colors = []
    for i, center in enumerate(centers):
        if i in label_counts:
            frequency = label_counts[i] / total_pixels
            rgb_color = tuple(map(int, center))
            dominant_colors.append((rgb_color, frequency))
    
    # Sort by frequency
    dominant_colors.sort(key=lambda x: x[1], reverse=True)
    
    return dominant_colors


def find_optimal_k(data, k_range):
    """
    Find optimal number of clusters using elbow method.
    
    Args:
        data: Pixel data
        k_range: Range of k values to test
        
    Returns:
        int: Optimal k value
    """
    if len(data) < 100:
        return min(3, len(data) // 10 + 1)
    
    inertias = []
    k_values = range(k_range[0], min(k_range[1] + 1, len(data) // 10))
    
    for k in k_values:
        if k > len(data):
            break
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=5)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
    
    # Simple elbow detection
    if len(inertias) < 2:
        return k_values[0]
    
    # Find the point with maximum curvature
    diffs = np.diff(inertias)
    diffs2 = np.diff(diffs)
    
    if len(diffs2) > 0:
        elbow_idx = np.argmax(diffs2) + 2  # +2 because of double diff
        return k_values[min(elbow_idx, len(k_values) - 1)]
    
    return k_values[len(k_values) // 2]  # Default to middle value


def classify_color_robust(rgb_color, brightness_context=None):
    """
    Robust color classification with adaptive thresholds.
    
    Args:
        rgb_color: RGB color tuple (r, g, b)
        brightness_context: Average brightness of the image for context
        
    Returns:
        str: Color category name
    """
    r, g, b = rgb_color
    
    # Convert to multiple color spaces for better analysis
    hsv = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
    lab = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2LAB)[0][0]
    
    h, s, v = hsv
    l_val, a_val, b_val = lab
    
    # Adaptive thresholds based on context
    dark_threshold = 35 if brightness_context and brightness_context < 100 else 45
    bright_threshold = 180 if brightness_context and brightness_context > 150 else 200
    saturation_threshold = 20 if brightness_context and brightness_context < 100 else 35
    
    # Enhanced color classification
    if v < dark_threshold:
        return "black"
    elif v > bright_threshold and s < saturation_threshold:
        return "white"
    elif s < saturation_threshold:  # Grayscale
        if v > 170:
            return "light_gray"
        elif v > 100:
            return "gray"
        elif v > 60:
            return "dark_gray"
        else:
            return "black"
    else:  # Colored objects with better hue ranges
        # Use both HSV and LAB for better color distinction
        if (h <= 15 or h >= 165) and s > 40:  # Red range expanded
            return "red"
        elif 15 < h <= 30 and s > 40:  # Orange
            return "orange"
        elif 30 < h <= 45 and s > 40:  # Yellow
            return "yellow"
        elif 45 < h <= 80 and s > 40:  # Green
            return "green"
        elif 80 < h <= 100 and s > 40:  # Cyan
            return "cyan"
        elif 100 < h <= 130 and s > 40:  # Blue
            return "blue"
        elif 130 < h <= 150 and s > 40:  # Purple
            return "purple"
        elif 150 < h < 165 and s > 40:  # Pink/Magenta
            return "pink"
        else:
            # Fallback using LAB color space
            if a_val > 10 and abs(b_val) < 20:  # Red-ish
                return "red"
            elif b_val > 15 and a_val < 10:  # Yellow-ish
                return "yellow"
            elif a_val < -10 and abs(b_val) < 20:  # Green-ish
                return "green"
            elif b_val < -15 and abs(a_val) < 20:  # Blue-ish
                return "blue"
            else:
                return "gray"  # Uncertain colors default to gray


def analyze_color_distribution(dominant_colors):
    """
    Analyze color distribution to make better decisions.
    
    Args:
        dominant_colors: List of (rgb_color, frequency) tuples
        
    Returns:
        dict: Analysis results
    """
    if not dominant_colors:
        return {"primary_color": "unknown", "confidence": 0.0}
    
    # Calculate color diversity
    total_colors = len(dominant_colors)
    primary_frequency = dominant_colors[0][1]
    
    # Check if one color is clearly dominant
    if primary_frequency > 0.6:  # Single dominant color
        confidence = "high"
    elif primary_frequency > 0.4:  # Moderately dominant
        confidence = "medium"
    else:  # Multiple colors compete
        confidence = "low"
    
    return {
        "primary_color": dominant_colors[0][0],
        "primary_frequency": primary_frequency,
        "total_colors": total_colors,
        "confidence": confidence
    }


def detect_dominant_color(image_crop):
    """
    Advanced dominant color detection using optimized K-means clustering and multiple color spaces.
    
    Args:
        image_crop: BGR image crop (numpy array)
        
    Returns:
        str: Dominant color name
    """
    if image_crop is None or image_crop.size == 0:
        return "unknown"
    
    try:
        # Resize for optimal processing (larger size for better accuracy)
        target_size = 150 if min(image_crop.shape[:2]) > 150 else min(image_crop.shape[:2])
        if image_crop.shape[0] != target_size or image_crop.shape[1] != target_size:
            image_crop = cv2.resize(image_crop, (target_size, target_size), interpolation=cv2.INTER_AREA)
        
        # Calculate brightness context for adaptive thresholding
        gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        
        # Get dominant colors using advanced K-means
        dominant_colors = get_dominant_colors_advanced_kmeans(image_crop, k_range=(3, 7))
        
        if not dominant_colors:
            return "unknown"
        
        # Analyze color distribution
        analysis = analyze_color_distribution(dominant_colors)
        
        # Enhanced color voting with confidence weighting
        color_votes = {}
        confidence_weights = {"high": 1.5, "medium": 1.2, "low": 1.0}
        base_weight = confidence_weights.get(analysis["confidence"], 1.0)
        
        for rgb_color, frequency in dominant_colors[:4]:  # Consider top 4 colors
            # Skip very small color patches (adaptive threshold)
            min_threshold = 0.03 if analysis["confidence"] == "high" else 0.08
            if frequency < min_threshold:
                continue
                
            color_name = classify_color_robust(rgb_color, avg_brightness)
            weight = frequency * base_weight
            
            # Boost weight for colors that are more likely to be object colors
            if color_name in ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'cyan']:
                weight *= 1.3  # Boost colored objects
            elif color_name in ['white', 'light_gray'] and avg_brightness > 150:
                weight *= 1.2  # Boost light colors in bright images
            elif color_name == 'black' and avg_brightness < 80:
                weight *= 0.8  # Reduce black weight in dark images (might be shadows)
            
            if color_name in color_votes:
                color_votes[color_name] += weight
            else:
                color_votes[color_name] = weight
        
        if not color_votes:
            # Fallback to the most dominant color
            most_dominant_rgb = dominant_colors[0][0]
            return classify_color_robust(most_dominant_rgb, avg_brightness)
        
        # Find the color with highest weighted vote
        dominant_color = max(color_votes.items(), key=lambda x: x[1])[0]
        
        # Post-processing: Handle edge cases
        if dominant_color == 'gray' and len([c for c in color_votes.keys() if c not in ['gray', 'light_gray', 'dark_gray', 'black', 'white']]) > 0:
            # If gray is detected but there are colored alternatives, choose the strongest colored one
            colored_votes = {k: v for k, v in color_votes.items() if k not in ['gray', 'light_gray', 'dark_gray']}
            if colored_votes and max(colored_votes.values()) > color_votes[dominant_color] * 0.7:
                dominant_color = max(colored_votes.items(), key=lambda x: x[1])[0]
        
        # Enhanced debug information
        print(f"[ColorDebug] Image brightness: {avg_brightness:.1f}")
        print(f"[ColorDebug] K-means colors: {[(classify_color_robust(rgb, avg_brightness), f'{freq:.3f}') for rgb, freq in dominant_colors[:4]]}")
        print(f"[ColorDebug] Color votes: {[(k, f'{v:.3f}') for k, v in sorted(color_votes.items(), key=lambda x: x[1], reverse=True)]}")
        print(f"[ColorDebug] Final color: {dominant_color} (confidence: {analysis['confidence']})")
        
        return dominant_color
        
    except Exception as e:
        print(f"[ColorDetection] Error: {e}")
        return "unknown"


def detect_color_from_bbox(frame, bbox):
    """
    Detect color from a bounding box region in a frame.
    
    Args:
        frame: BGR image frame (numpy array)
        bbox: Bounding box coordinates [x1, y1, x2, y2]
        
    Returns:
        str: Detected color name
    """
    if frame is None or frame.size == 0:
        return "unknown"
    
    try:
        x1, y1, x2, y2 = map(int, bbox)
        
        # Ensure bbox is within frame bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            return "unknown"
        
        # Extract ROI from frame
        roi = frame[y1:y2, x1:x2]
        
        return detect_dominant_color(roi)
        
    except Exception as e:
        print(f"[ColorDetection] Error in bbox detection: {e}")
        return "unknown"
