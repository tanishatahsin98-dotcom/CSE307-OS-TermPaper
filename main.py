import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.tree import DecisionTreeClassifier

# ==========================================
# 1. Generate Workload with a Shift
# ==========================================
np.random.seed(42)
# Phase 1: Locality-heavy (Sequential/Looping with small noise)
# Accessing a small set of pages (e.g., 0 to 9)
phase1 = np.random.randint(0, 10, size=500).tolist()

# Phase 2: Random / Bursty
# Accessing a much larger set of pages completely randomly (e.g., 0 to 99)
phase2 = np.random.randint(0, 100, size=500).tolist()

# Combine both phases to create the full trace
trace = phase1 + phase2
shift_point = 500
FRAME_SIZE = 4

print(f"Total workload size: {len(trace)} requests")
print(f"Shift occurs at request #{shift_point}")
print("-" * 40)

# ==========================================
# 2. Classical Algorithms Implementation
# ==========================================
def run_fifo(trace, frame_size, shift_point):
    frames = []
    faults_before, faults_after = 0, 0
    
    for i, page in enumerate(trace):
        if page not in frames:
            if i < shift_point:
                faults_before += 1
            else:
                faults_after += 1
                
            if len(frames) >= frame_size:
                frames.pop(0) # Evict oldest (First In)
            frames.append(page)
            
    return faults_before, faults_after

def run_lru(trace, frame_size, shift_point):
    frames = []
    faults_before, faults_after = 0, 0
    
    for i, page in enumerate(trace):
        if page not in frames:
            if i < shift_point:
                faults_before += 1
            else:
                faults_after += 1
                
            if len(frames) >= frame_size:
                frames.pop(0) # Evict least recently used
            frames.append(page)
        else:
            frames.remove(page)
            frames.append(page) # Update recency
            
    return faults_before, faults_after

def run_optimal(trace, frame_size, shift_point):
    frames = []
    faults_before, faults_after = 0, 0
    
    for i, page in enumerate(trace):
        if page not in frames:
            if i < shift_point:
                faults_before += 1
            else:
                faults_after += 1
                
            if len(frames) >= frame_size:
                # Find the page to evict (the one used furthest in future)
                farthest = -1
                evict_page = frames[0]
                for f in frames:
                    try:
                        next_use = trace[i+1:].index(f)
                    except ValueError:
                        next_use = float('inf')
                    
                    if next_use > farthest:
                        farthest = next_use
                        evict_page = f
                frames.remove(evict_page)
            frames.append(page)
            
    return faults_before, faults_after

# ==========================================
# 3. Learned / Adaptive Component (Decision Tree)
# ==========================================
def run_learned_model(trace, frame_size, shift_point):
    """
    A lightweight learned model. It trains a Decision Tree during a warmup 
    period to mimic Optimal behavior using 'recency' and 'frequency' as features.
    """
    frames = []
    faults_before, faults_after = 0, 0
    
    # Track features
    freq = defaultdict(int)
    recency = defaultdict(int)
    
    # For training the model
    X_train = []
    y_train = []
    model = DecisionTreeClassifier(max_depth=3)
    is_trained = False
    WARMUP = 100 # Use first 100 requests to gather training data
    
    for i, page in enumerate(trace):
        freq[page] += 1
        recency[page] = i
        
        if page not in frames:
            if i < shift_point:
                faults_before += 1
            else:
                faults_after += 1
                
            if len(frames) >= frame_size:
                # Eviction logic
                if i < WARMUP or not is_trained:
                    # Collect data acting like Optimal for warmup
                    farthest = -1
                    evict_idx = 0
                    for idx, f in enumerate(frames):
                        try:
                            next_use = trace[i+1:].index(f)
                        except ValueError:
                            next_use = float('inf')
                        if next_use > farthest:
                            farthest = next_use
                            evict_idx = idx
                            
                    # Record features (recency distance, frequency) for all pages in frame
                    for idx, f in enumerate(frames):
                        recency_dist = i - recency[f]
                        X_train.append([recency_dist, freq[f]])
                        y_train.append(1 if idx == evict_idx else 0)
                        
                    frames.pop(evict_idx)
                else:
                    # Model makes the decision
                    predictions = []
                    for f in frames:
                        recency_dist = i - recency[f]
                        pred = model.predict_proba([[recency_dist, freq[f]]])[0]
                        # Probability of being the optimal eviction candidate (class 1)
                        prob_evict = pred[1] if len(pred) > 1 else 0
                        predictions.append(prob_evict)
                    
                    evict_idx = np.argmax(predictions)
                    frames.pop(evict_idx)
                    
            frames.append(page)
            
        # Train model after warmup
        if i == WARMUP and not is_trained and len(np.unique(y_train)) > 1:
            model.fit(X_train, y_train)
            is_trained = True
            
    return faults_before, faults_after

# ==========================================
# 4. Execution & Results
# ==========================================
fifo_before, fifo_after = run_fifo(trace, FRAME_SIZE, shift_point)
lru_before, lru_after = run_lru(trace, FRAME_SIZE, shift_point)
opt_before, opt_after = run_optimal(trace, FRAME_SIZE, shift_point)
ml_before, ml_after = run_learned_model(trace, FRAME_SIZE, shift_point)

print("Results (Page Faults):")
print(f"FIFO    -> Before Shift: {fifo_before:3d} | After Shift: {fifo_after:3d} | Total: {fifo_before + fifo_after}")
print(f"LRU     -> Before Shift: {lru_before:3d} | After Shift: {lru_after:3d} | Total: {lru_before + lru_after}")
print(f"Optimal -> Before Shift: {opt_before:3d} | After Shift: {opt_after:3d} | Total: {opt_before + opt_after}")
print(f"Learned -> Before Shift: {ml_before:3d} | After Shift: {ml_after:3d} | Total: {ml_before + ml_after}")

# Plotting the results
labels = ['FIFO', 'LRU', 'Optimal', 'Learned (DT)']
before_faults = [fifo_before, lru_before, opt_before, ml_before]
after_faults = [fifo_after, lru_after, opt_after, ml_after]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, before_faults, width, label='Phase 1: Locality-Heavy', color='skyblue')
rects2 = ax.bar(x + width/2, after_faults, width, label='Phase 2: Random (Shift)', color='salmon')

ax.set_ylabel('Number of Page Faults')
ax.set_title(f'Page Faults Before and After Workload Shift (Frame Size = {FRAME_SIZE})')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('result_graph.png') # Saves the graph as an image file
print("-" * 40)
print("Graph saved as 'result_graph.png'.")
plt.show()