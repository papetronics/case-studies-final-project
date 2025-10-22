# ✅ Fleet Pape: Local-to-Fleet MLOps Roadmap

A realistic, prioritized list of deliverables to turn my experiments from “chaotic local loops” into a reproducible mini lab.  
Each block is **15–120 minutes**, so I can slot them into hobby time.

---

## 🎯 Core Goal

- 🔹 One repo = one W&B project
- 🔹 One Job snapshot = one frozen Docker image per commit
- 🔹 Local agent does dev work for free
- 🔹 Amazon SageMaker fleet does sweeps when it’s worth paying
- 🔹 Everything is versioned, reproducible, and easy to rerun

---

## ✅ PHASE 1 — Local Baseline (zero cost)

---

### **1.** Local `wandb launch` flow (30–60 min)
- ✅ `train.py` with `wandb.init()` and `wandb.finish()`
- ✅ Use `wandb launch` CLI → confirm runs show in Dashboard
- ✅ Local `launch-agent` polls queue and runs jobs

**Outcome:** Queue → agent loop works.

---

### **2.** Pin Python env (15–30 min)
- ✅ Create `requirements.txt` with pinned versions (`torch==...`, `pytorch-lightning==...`)
- ✅ Confirm fresh venv works identically

**Outcome:** No surprises in Docker or on other machines.

---

### **3.** Git commit discipline (15 min)
- ✅ `main` is my “known good” (LKG) branch
- ✅ Push every run worth training — no uncommitted “mystery heads”
- ✅ Use SSH and ensure the Git access token is not exposed anywhere.

**Outcome:** Every run can be traced to code.

---

## ✅ PHASE 2 — Local “multi-idea” fleet

---

### **4.** Dev branch run pattern (30–60 min)
- ✅ Test feature branch → push → `wandb launch` with `--git-version branchname`
- ✅ Confirm each run pins to that branch SHA
- ✅ Agent can queue and run multiple jobs back to back

**Outcome:** I can push 5 ideas → GPU stays busy while I sleep.

---

### **5.** Basic sweep YAML (30–45 min)
- ✅ Write `sweep.yaml` for simple params (`lr`, `epochs`, etc.)
- ✅ Test creating a Sweep in Dashboard → confirm agent runs multiple jobs

**Outcome:** Launch queue + Dashboard sweeps work locally.

---

## ✅ PHASE 3 — Automated CI/CD Job Creation

---

### **6.** Add `wandb job create` in CI (30–60 min)
- ✅ After any branch is updated, `wandb job create` with the repo+branch as the job name
- ✅ Confirm Job appears in W&B workspace with `:latest`
- ✅ Confirm we can run a job

**Outcome:** No babysitting job definitions in W&B.

---

## ✅ PHASE 4 — Real “Fleet Pape” sweep test

---

### **7.** Amazon SageMaker auto-launch + autokill test (60–90 min)
- ✅ Create terraformer infra (in `pape-lab` repo)
- ✅ Request capacity
- ✅ Create and push base Docker image (in `pape-lab` repo)
- ✅ Tested creating a run from W&B and seeing it succeed in AWS!

**Outcome:** Prove I can scale out with zero babysitting.

---

## ✅ PHASE 5 — Template & docs

---

### **8.** Extract `pape-lab` repo (60–90 min)
- ✅ Terraformer config for AWS infra
- ✅ Scripts for base Docker image creation and management
- ✅ `README.md` with instructions

**Outcome:** Infra repo (`pape-lab`) automates AWS setup and base image updates.

---

### **9.** Extract `pape-lab-project` repo (60–90 min)
- ✅ `Dockerfile.wandb`
- ✅ Scripts to manually trigger a job
- ✅ Starter code
- ✅ `README.md` with instructions

**Outcome:** Template repo (`pape-lab-project`) for future ideas.

---

## ⬜ PHASE 6 — Migrate VQ-VAE + AR

---

### **10.** Add `wandb.init()` to VQ-VAE script (30–60 min)
- ⬜ Log config: `epochs`, `start_lr`, `commit penalty`
- ⬜ Log checkpoints as W&B Artifacts

**Outcome:** VQ-VAE is Launch-ready.

---

### **11.** Add AR + quantizer pattern (30–60 min)
- ⬜ Feed output of VQ-VAE → AR → log AR results to W&B
- ⬜ Runs through same queue, proven chained pipeline

**Outcome:** Full multi-step experiment is reproducible.

---

## ⬜ PHASE 7 — Productionize repos

---

### **12.** Automate `pape-lab-template` workflows (60–90 min)
- ⬜ Automatically build Docker image on changes to `main`
- ⬜ Push image to GHCR (or DockerHub for testing)
- ⬜ Create a W&B job for `main` using the built image
- ⬜ Automatically create a git-based W&B job whenever a branch is pushed

**Outcome:** Template repo (`pape-lab-template`) handles CI/CD for experiments.

---

### **13.** Automate `pape-lab` workflows (60–90 min)
- ⬜ Automatically run terraformer on changes to `main` terraformer config
- ⬜ Automatically rebuild base Docker image on changes to its `Dockerfile` or `requirements.txt`
- ⬜ Push rebuilt image to GHCR (or DockerHub for testing)

**Outcome:** Infra repo (`pape-lab`) handles AWS infra updates and base image management.

---

### **14.** Switch W&B to using pre-built images directly (30–60 min)
- ⬜ `wandb launch job-foo:latest` → local agent pulls frozen image, not raw Git
- ⬜ Runs reproducibly, no surprises

**Outcome:** Ready for SageMaker fleet later.

---

## ⚡️ Final Takeaway

- 🏃 *80% of the real benefit comes from Phases 1–3.*
- 🗃️ *Prebuilt images + CI give me reproducibility.*
- 🚀 *Amazon SageMaker fleet makes sweeping cost-effective when it’s worth paying.*
- 🧩 *Two repos (`pape-lab` and `pape-lab-project`) mean I never reinvent the wheel.*

---

> **Commit every run. Snapshot every env. Never lose your best idea to a half-finished folder again.**