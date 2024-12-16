# preprocessing.py

import logging
import numpy as np
import pandas as pd
import requests
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer

# Initialize the YOLO model globally to avoid reloading it repeatedly
yolo_model = YOLO('yolo11n.pt')
yolo_model.overrides['verbose'] = False

def detect_objects(image):
    """
    Detect objects in an image using YOLO.
    Returns the top 3 detected objects.
    """
    try:
        results = yolo_model(image)
        detections_df = results[0].to_df()

        if detections_df.empty:
            return {'contain_1': None, 'contain_2': None, 'contain_3': None}

        object_counts = detections_df['name'].value_counts()
        top_objects = object_counts.index.tolist()[:3]

        return {
            'contain_1': top_objects[0] if len(top_objects) > 0 else None,
            'contain_2': top_objects[1] if len(top_objects) > 1 else None,
            'contain_3': top_objects[2] if len(top_objects) > 2 else None,
        }
    except Exception as e:
        logging.error(f"Error in object detection: {e}")
        return {'contain_1': None, 'contain_2': None, 'contain_3': None}

def analyze_colors(image):
    """
    Analyze the colors in an image and compute dominant color, brightness, and color diversity.
    """
    try:
        pixels = np.array(image).reshape(-1, 3)
        dominant_color = np.mean(pixels, axis=0)
        brightness = np.mean(np.sqrt(np.sum(pixels**2, axis=1)))
        color_diversity = len(np.unique(pixels, axis=0))

        return {
            'dominant_color_r': dominant_color[0],
            'dominant_color_g': dominant_color[1],
            'dominant_color_b': dominant_color[2],
            'brightness': brightness,
            'color_diversity': color_diversity
        }
    except Exception as e:
        logging.error(f"Error in color analysis: {e}")
        return {'dominant_color_r': None, 'dominant_color_g': None, 'dominant_color_b': None, 'brightness': None, 'color_diversity': None}

def process_thumbnail(index, url):
    """
    Process a single thumbnail: download the image, detect objects, and analyze colors.
    """
    try:
        response = requests.get(url, timeout=5)
        image = Image.open(BytesIO(response.content)).convert("RGB")

        detection_results = detect_objects(image)
        color_analysis = analyze_colors(image)

        return index, {**detection_results, **color_analysis, 'thumbnail_url': url}
    except Exception as e:
        logging.error(f"Error processing thumbnail {url}: {e}")
        return index, {'thumbnail_url': url, 'error': str(e)}

def process_thumbnails(thumbnail_urls):
    """
    Process a list of thumbnail URLs using multithreading.
    """
    results_list = [None] * len(thumbnail_urls)
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_thumbnail, i, url): i for i, url in enumerate(thumbnail_urls)}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Thumbnails", unit="image"):
            index, result = future.result()
            results_list[index] = result
    return pd.DataFrame(results_list)

def clean_text_column(column):
    """
    Clean and preprocess text columns by removing special characters and converting to lowercase.
    """
    return (
        column.fillna("")
        .str.replace(r"[^\w\s]", " ", regex=True)
        .str.lower()
        .str.strip()
    )

def encode_text_features(data, text_features, model_name='Snowflake/snowflake-arctic-embed-l-v2.0'):
    """
    Generate SentenceTransformer embeddings for specified text features.
    """
    model = SentenceTransformer(model_name, trust_remote_code=True)
    sentence_embeddings = {}
    for feature in text_features:
        text_data = data[feature].tolist()
        embeddings = model.encode(text_data, show_progress_bar=True)
        sentence_embeddings[feature] = pd.DataFrame(
            embeddings,
            columns=[f"{feature}_dim{i}" for i in range(embeddings.shape[1])]
        )
    return pd.concat(sentence_embeddings.values(), axis=1)

def reduce_embeddings(embeddings, embedding_dim=4):
    """
    Reduce embeddings dimensionality using PCA.
    """
    pca = PCA(n_components=embedding_dim)
    reduced_embeddings = pca.fit_transform(np.array(embeddings))
    return pd.DataFrame(reduced_embeddings, columns=[f"text_embeddings_{i}" for i in range(embedding_dim)])

def preprocess_data(data):
    """
    Main preprocessing pipeline:
    - Process thumbnails (YOLO + color analysis)
    - Clean text features
    - Generate and reduce text embeddings
    """
    print("Processing thumbnails...")
    thumbnail_urls = data['video_default_thumbnail']
    results_df = process_thumbnails(thumbnail_urls)

    print("Merging thumbnail results with original data...")
    final_data = pd.merge(data, results_df, how="left", left_on='video_default_thumbnail', right_on='thumbnail_url')

    print("Cleaning text features...")
    text_features = ['video_title']
    for feature in text_features:
        final_data[feature] = clean_text_column(final_data[feature])

    print("Generating text embeddings...")
    embedded_text = encode_text_features(final_data, text_features)

    print("Reducing text embeddings...")
    reduced_embedding_df = reduce_embeddings(embedded_text)

    print("Merging embeddings into final data...")
    final_data = pd.concat([final_data, reduced_embedding_df], axis=1)

    print("Adding tag count...")
    final_data['tag_count'] = final_data['video_tags'].apply(lambda x: 0 if x == 'No tags' else len(x.split(',')))

    print("Preprocessing complete.")
    return final_data
