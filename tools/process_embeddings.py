import json
import torch
import clip
from sklearn.decomposition import PCA
import argparse

def process_annotations(file_path, output_path):
    # Load the JSON data
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    model, _ = clip.load("ViT-B/32", device=device)
    
    # Collect all "text" from "humanml3d_XXX" annotations
    texts = []
    annotations_list = []
    
    for key, value in data.items():
        annotations = value.get('annotations', [])
        for annotation in annotations:
            seg_id = annotation.get('seg_id', '')
            if seg_id.startswith("humanml3d_"):
                # Skip if 'clip_embedding' and 'clip_embedding_2d' are already present
                # if 'clip_embedding' in annotation and 'clip_embedding_2d' in annotation:
                #     continue
                text = annotation.get('text', '')
                texts.append(text)
                annotations_list.append((annotation, key))  # Store annotation and parent key
    
    # Compute embeddings
    with torch.no_grad():
        # CLIP has a maximum context length of 77 tokens.
        text_tokens = clip.tokenize(texts, truncate=True).to(device)
        text_embeddings = model.encode_text(text_tokens)
        text_embeddings = text_embeddings.cpu().numpy()
    
    # Perform PCA to reduce embeddings to 2D
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(text_embeddings)
    
    # Update the annotations with embeddings
    for i, (annotation, parent_key) in enumerate(annotations_list):
        # Original embedding
        embedding = text_embeddings[i].tolist()
        # 2D embedding
        embedding_2d = embeddings_2d[i].tolist()
        # Append to annotation
        annotation['clip_embedding'] = embedding
        annotation['clip_embedding_2d'] = embedding_2d
        
        # Update the annotation in the original data
        for idx, ann in enumerate(data[parent_key]['annotations']):
            if ann['seg_id'] == annotation['seg_id']:
                data[parent_key]['annotations'][idx] = annotation
                break
    
    print(text_embeddings.shape)
    print(embeddings_2d.shape)
    # Write the updated data back to JSON
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get CLIP embeddings for annotations')
    parser.add_argument('--file_path', type=str, default='../datasets/merge/babel_humanml3d_kitml.json',
                        help='Path to the JSON annotation file')
    parser.add_argument('--output_path', type=str, default='/scratch/izar/jiaxu/babel_humanml3d_kitml_embedding.json',
                        help='Path to output file')


    args = parser.parse_args()

    # Call the function with the file path
    process_annotations(args.file_path, args.output_path)
