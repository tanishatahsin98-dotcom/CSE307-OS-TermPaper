# CSE-307 Term Paper: Learned Page Replacement

## Overview
This repository contains the implementation for the CSE-307 Operating Systems Term Paper (Track 1). The project simulates classical page replacement algorithms (FIFO, LRU, Optimal) and compares them against a lightweight Machine Learning model (Decision Tree) under shifting workload conditions. 

## Experimental Setup
* **Algorithms Implemented:** FIFO, LRU, Optimal (Belady's), and a Learned Decision Tree Classifier.
* **Workload:** 1000 total page requests. 
  * **Phase 1 (Requests 0-499):** Locality-heavy (simulating sequential/looping behavior with a small page pool).
  * **Phase 2 (Requests 500-999):** Random/Bursty shift (simulating sudden random access across a larger page pool).
* **Frame Size:** 4

## How to Run
1. Ensure Python is installed on your system.
2. Install the required dependencies:
   ```bash
   pip install numpy pandas scikit-learn matplotlib
