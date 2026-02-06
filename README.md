# Sentinel: Automated Model Monitoring & Self-Healing Pipeline

## Table of Contents

- [The Architecture](#the-architecture)
- [Project Status: The Simulation](#project-status-the-simulation)
- [The Self-Healing Loop (How it Works)](#the-self-healing-loop-how-it-works)
- [Tech Stack & Design Patterns](#tech-stack--design-patterns)
- [The Workflow](#the-workflow)
  - [1. The "Saboteur" (Data Simulation)](#1-the-saboteur-data-simulation)
  - [2. The "Endpoint" (Modular Monitoring)](#2-the-endpoint-modular-monitoring)
  - [3. The "Red Phone" (Automated Retraining)](#3-the-red-phone-automated-retraining)
- [How to Run This Project](#how-to-run-this-project)
  - [1. Build the Monitor](#1-build-the-monitor)
  - [2. Run the Saboteur (Create Bad Data)](#2-run-the-saboteur-create-bad-data)
  - [3. Run the QC Check Manually](#3-run-the-qc-check-manually)
- [Future Roadmap (Scaling to Production)](#future-roadmap-scaling-to-production)
  - [Immediate Refinements (Next 2 Weeks - Local Development Focus)](#immediate-refinements-next-2-weeks---local-development-focus)
  - [Deep Learning & MLOps Expansions](#deep-learning--mlops-expansions)
  - [Infrastructure & Cloud Migration](#infrastructure--cloud-migration-delayed)

## The Architecture
*"Models degrade. Good systems heal themselves."*

This project implements an **end-to-end MLOps Drift Detection System** for a Video Frame Interpolation (VFI) model. Instead of manually checking for performance decay (**fondly named "wobbly chairs"**) or data quality issues (**"softwood"**), this system automates the entire **Quality Control loop** using **Containerized Microservices** and **Automated Orchestration**.

---

## Project Status: The Simulation
To test this system properly, I am currently running it in a controlled, **simulated environment**. This allows me to prove that the "safety net" works before trusting it with a live model.

| Component | Status | Notes |
| :--- | :--- | :--- |
| **The Model** | 🔹 **Simulated** | My actual VFI model is currently still in training. For now, I am using a "mock" (a stand-in) to ensure the monitoring system triggers correctly, regardless of the specific model inside. |
| **Data Source** | 🔹 **Synthetic** | I created specific test data to "force" the system to react. By feeding it intentionally perfect or intentionally bad data, I can guarantee that the system correctly spots the difference between a **PASS** and a **FAIL**. |
| **Scalability** | 🔹 **Modular Design** | Currently, this detects drift in **Video** inputs. However, the system is built like building blocks. In the future, I can easily "snap in" new blocks for Audio or Text models without breaking the existing structure. |

---

## The Self-Healing Loop (How it Works)

```mermaid
flowchart TD
    subgraph INITIALIZATION["🚀 Initialization"]
        A[sentinel_watch.py] --> B[SentinelWatch.watch]
        B --> C[Simulate Cloud Connection]
    end

    subgraph DRIFT["📊 Step 1: Data Drift Detection"]
        C --> D[Run Drift Check]
        D --> E[detector_data_drift/pipeline.py]
        E --> F[Detect Domain]
        F --> G[Extract Embeddings (MobileNetV2/ResNet)]
        G --> H[Run Analyzer (Ensemble Metrics)]
        H --> I{Drift Detected?}
    end

    subgraph ARCHIVE["📁 Data Archival"]
        I -->|Yes/No| J[Archive Incoming Data]
        J --> K[Record Drift Result to SQLite]
        K --> L{Check Drift History<br/>Threshold Met?}
    end

    subgraph PASS_PATH["✅ No Action Required"]
        I -->|PASS| M[Log OK Status]
        L -->|No| N[Log - Waiting for Threshold]
    end

    subgraph RETRAIN["⚙️ Step 2: Trigger Retraining"]
        L -->|Yes| O[Send Retraining Alert]
        O --> P[ExecutionEngine]
        P --> Q[LocalDriver.start]
        Q --> R[Run Training Script]
        R --> S{Training<br/>Successful?}
        S -->|No| T[Send Critical Alert]
        S -->|Yes| U[Log Metrics]
    end

    subgraph DECAY["🔍 Step 3: Decay Check / Gatekeeper"]
        U --> V[Run Decay Pipeline]
        V --> W[detector_model_decay/pipeline.py]
        W --> X[Load Challenger Model]
        X --> Y[Run VFI Inference on Golden Set]
        Y --> Z[Extract Embeddings]
        Z --> AA[Compare with Production Model]
        AA --> AB{Decay Check<br/>Passed?}
    end

    subgraph DEPLOY["🚀 Deployment"]
        AB -->|Yes| AC[Simulate Deployment]
        AC --> AD[Send Deployment Alert]
        AD --> AE[Update Baselines]
        AE --> AF[GoldenSetCurator]
        AF --> AG[DataRotator]
        AG --> AH[Purge Drift History]
        AH --> AI[✅ Self-Healing Complete]
    end

    subgraph ABORT["❌ Deployment Aborted"]
        AB -->|No| AJ[Send Decay Fail Alert]
        AJ --> AK[❌ Deployment Blocked]
    end

    subgraph SERVICES["🔧 Background Services"]
        direction LR
        SVC1[Distiller<br/>distiller.py] --> SVC2[Creates Latent-Space<br/>Models for Drift Detection]
        SVC3[SentinelAlert<br/>alert_utils.py] --> SVC4[Email + System<br/>Notifications]
    end

    style INITIALIZATION fill:#1a1a2e,stroke:#00E5FF,color:#fff
    style DRIFT fill:#16213e,stroke:#0F3460,color:#fff
    style ARCHIVE fill:#1a1a2e,stroke:#E94560,color:#fff
    style PASS_PATH fill:#0d3320,stroke:#00ff88,color:#fff
    style RETRAIN fill:#3d1a1a,stroke:#ff6b6b,color:#fff
    style DECAY fill:#1a2a3d,stroke:#4ecdc4,color:#fff
    style DEPLOY fill:#1a3d1a,stroke:#95e88a,color:#fff
    style ABORT fill:#3d1a1a,stroke:#ff4757,color:#fff
    style SERVICES fill:#2d2d44,stroke:#9b59b6,color:#fff
```

## Tech Stack & Design Patterns

| Component | Tech | Role (The Factory Analogy) |
| :--- | :--- | :--- |
| **Drift Detection** | **Keras / PyTorch / ONNX** | The QC Sensor: Uses model-agnostic transfer learning to extract feature **embeddings** from video frames (The "Touch Test"). |
| **Statistical Analysis** | **Ensemble Metrics** | The Judge: Uses a weighted vote of **Cosine Similarity (90%)**, Wasserstein Distance, KL-Divergence, and MMD to robustly detect drift. |
| **Infrastructure** | **Docker** | The QC Booth: A portable, isolated environment that ensures the monitor runs identically on any machine. |
| **Orchestration** | **Execution Engine** | The Manager: Automates the schedule, triggers `LocalDriver` or distributed jobs, and handles the "**Red Phone**" logic. |

---

## The Workflow

### 1. The "Saboteur" (Data Simulation)
To prove the system works, I built a `data_saboteur.py` script that synthetically generates **"drifted" data** (noise, blur, low-light) to simulate real-world camera failures.

### 2. The "Endpoint" (Modular Monitoring)
The monitoring logic is decoupled from the orchestration.

* It runs as a **stateless Docker Container**.
* It accepts a volume of data and a baseline reference.
* It outputs a strictly typed status (`PASS`/`FAIL`) and a **Drift Score**.

> **Why this matters:** This architecture is **model-agnostic**. I can swap the internal logic for an NLP monitor, and the infrastructure remains unchanged.

### 3. The "Red Phone" (Automated Retraining)
When drift is detected (threshold exceeded), the system doesn't just alert—it acts. The primary workflow automatically triggers a secondary **Retraining Pipeline** via the `ExecutionEngine`. This simulates a **continuous training (CT) loop**, dispatching training jobs (locally or to specific hardware) to fix the issue.

---

## How to Run This Project

**Prerequisites:**

* Docker installed
* Python 3.9+

### 1. Build the Monitor:
```bash
docker build -t vfi-monitor .
```

### 2. Run the Saboteur (Create Bad Data):
```bash
python data_tools/data_saboteur.py
```

### 3. Run the QC Check Manually:
```bash
docker run \
  -v $(pwd)/data/drifted_frames:/app/incoming_data \
  -v $(pwd)/temp_status:/app/status_output \
  vfi-monitor
```

### Future Roadmap (Scaling to Production)

#### Immediate Refinements (Next 2 Weeks - Local Development Focus)

These steps are designed to be completed fully offline to optimize resources:

* **✅ Visualization Dashboard:** `dashboard.py` is now live! It connects the **Drift Score** output to a **Streamlit** dashboard, providing a real-time "Pulse Check" of the system.
* ~~**Decoupling:** Complete the implementation of external configuration management (e.g., Environment Variables, K8s ConfigMaps) to remove all hardcoded parameters.~~

#### Deep Learning & MLOps Expansions

* **✅ Model Drift (Concept Drift):** The system now includes `detector_model_decay` to detect **Model Decay**. It creates a feedback loop to compare predictions against ground truth labels (Golden Set).
* **Horizontal Scaling (Model Agnosticism):** Once the VFI pipeline is perfected, I will generalize the architecture. The aim is for a modular monitoring approach that can easily monitor models pulled directly from **Hugging Face** or **Kaggle** using the same Sentinel container.
* **A/B Testing:** Implement a **Canary Deployment** strategy, allowing the "Champion" model and the "Challenger" model to run side-by-side on live data to compare performance safely.

#### Infrastructure & Cloud Migration (Delayed)

* **Cloud Deployment:** Deploy the VFI Model and the Sentinel Monitor to **Google Cloud Platform (GCP)**.
* **Orchestrator Upgrade:** Migrate the orchestration layer from GitHub Actions to **Kubernetes (K8s)** to demonstrate production scaling, service management, and high availability.
