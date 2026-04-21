import base64
import json
import os
import io
from flask import Flask, request
from google.cloud import storage, firestore
import numpy as np
from numba import njit
from PIL import Image

app = Flask(__name__)
storage_client = storage.Client()
firestore_client = firestore.Client(database="wfc-db")

def extract_patterns_and_rules(image_array, N=3):
    height, width, _ = image_array.shape
    patterns = []
    
    for y in range(height):
        for x in range(width):
            patch = np.zeros((N, N, 3), dtype=np.uint8)
            for dy in range(N):
                for dx in range(N):
                    patch[dy, dx] = image_array[(y + dy) % height, (x + dx) % width]
            patterns.append(patch)
            
    unique_patterns, counts = np.unique(np.array(patterns), axis=0, return_counts=True)
    num_patterns = len(unique_patterns)
    weights = counts / counts.sum()
    
    rules = np.zeros((num_patterns, 4, num_patterns), dtype=np.bool_)
    for i, p1 in enumerate(unique_patterns):
        for j, p2 in enumerate(unique_patterns):
            rules[i, 0, j] = np.array_equal(p1[1:, :], p2[:-1, :])
            rules[i, 1, j] = np.array_equal(p1[:, 1:], p2[:, :-1])
            rules[i, 2, j] = np.array_equal(p1[:-1, :], p2[1:, :])
            rules[i, 3, j] = np.array_equal(p1[:, :-1], p2[:, 1:])
            
    return unique_patterns, weights, rules

@njit
def execute_wfc(grid_size, num_patterns, rules, weights):
    wave = np.ones((grid_size, grid_size, num_patterns), dtype=np.bool_)
    max_stack = grid_size * grid_size * num_patterns
    stack_y = np.zeros(max_stack, dtype=np.int32)
    stack_x = np.zeros(max_stack, dtype=np.int32)
    dirs = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])
    
    while True:
        min_entropy = 99999.0
        min_y, min_x = -1, -1
        
        for y in range(grid_size):
            for x in range(grid_size):
                valid_states = 0
                sum_weights = 0.0
                for t in range(num_patterns):
                    if wave[y, x, t]:
                        valid_states += 1
                        sum_weights += weights[t]
                        
                if valid_states == 0:
                    return np.zeros((1,1), dtype=np.int32)
                elif valid_states > 1:
                    entropy = sum_weights - (np.random.rand() * 0.001)
                    if entropy < min_entropy:
                        min_entropy = entropy
                        min_y, min_x = y, x
                        
        if min_y == -1:
            break 
            
        possible_tiles = []
        for t in range(num_patterns):
            if wave[min_y, min_x, t]:
                possible_tiles.append(t)
                
        chosen = possible_tiles[np.random.randint(len(possible_tiles))]
        for t in range(num_patterns):
            wave[min_y, min_x, t] = (t == chosen)
            
        stack_y[0], stack_x[0] = min_y, min_x
        stack_ptr = 1
        
        while stack_ptr > 0:
            stack_ptr -= 1
            cy, cx = stack_y[stack_ptr], stack_x[stack_ptr]
            
            for d in range(4):
                ny, nx = cy + dirs[d, 0], cx + dirs[d, 1]
                if 0 <= ny < grid_size and 0 <= nx < grid_size:
                    allowed = np.zeros(num_patterns, dtype=np.bool_)
                    for ct in range(num_patterns):
                        if wave[cy, cx, ct]:
                            for nt in range(num_patterns):
                                if rules[ct, d, nt]:
                                    allowed[nt] = True
                    
                    changed = False
                    for nt in range(num_patterns):
                        if wave[ny, nx, nt] and not allowed[nt]:
                            wave[ny, nx, nt] = False
                            changed = True
                            
                    if changed:
                        stack_y[stack_ptr], stack_x[stack_ptr] = ny, nx
                        stack_ptr += 1
                        
    result = np.zeros((grid_size, grid_size), dtype=np.int32)
    for y in range(grid_size):
        for x in range(grid_size):
            for t in range(num_patterns):
                if wave[y, x, t]:
                    result[y, x] = t
                    break
    return result

@app.route('/', methods=['POST'])
def pubsub_push():
    envelope = request.get_json()
    if not envelope or 'message' not in envelope:
        return 'Bad Request', 400

    msg_data = base64.b64decode(envelope['message']['data']).decode('utf-8')
    work_order = json.loads(msg_data)

    input_bucket = work_order.get('input_bucket')
    input_file = work_order.get('input_filename')
    output_bucket = work_order.get('output_bucket')
    job_id = work_order.get('job_id')
    patch_size = int(work_order.get('patch_size', 3))

    if not all([input_bucket, input_file, output_bucket, job_id]):
        return 'Missing required info in payload', 400
    
    try:
        in_blob = storage_client.bucket(input_bucket).blob(input_file)
        img_bytes = in_blob.download_as_bytes()
        seed_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        seed_array = np.array(seed_img)
        
        patterns, weights, rules = extract_patterns_and_rules(seed_array, N=patch_size)
        num_patterns = len(patterns)
        
        grid_size = 50 
        while True:
            result_grid = execute_wfc(grid_size, num_patterns, rules, weights)
            if result_grid.shape != (1, 1):
                break
                
        final_array = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)
        for y in range(grid_size):
            for x in range(grid_size):
                pattern_idx = result_grid[y, x]
                final_array[y, x] = patterns[pattern_idx][0, 0]
                
        final_img = Image.fromarray(final_array, 'RGB')
        out_io = io.BytesIO()
        final_img.resize((500, 500), Image.NEAREST).save(out_io, format='PNG')
        
        out_name = f"generated-{job_id}.png"
        out_blob = storage_client.bucket(output_bucket).blob(out_name)
        out_blob.upload_from_string(out_io.getvalue(), content_type='image/png')
        
        public_url = f"https://storage.googleapis.com/{output_bucket}/{out_name}"

        doc_ref = firestore_client.collection("wfc_jobs").document(job_id)
        doc_ref.update({
            "status": "COMPLETE",
            "output_url": public_url
        })
        
        return 'Success', 200

    except Exception as e:
        print(f"Pipeline Error: {e}")
        firestore_client.collection("wfc_jobs").document(job_id).update({"status": "ERROR"})
        return 'Internal Server Error', 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
