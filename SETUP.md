# Bank MLOps Pipeline — Ubuntu Setup Guide
Target machine: Ubuntu, user `dhiraj-bohara`, project root `/home/dhiraj-bohara/projects/MLops`

All hardcoded paths in the code have been changed from the old laptop's
`/home/bishnu-upadhyay/projects/MLops` to `/home/dhiraj-bohara/projects/MLops`.
Conda has also been replaced with a plain Python venv at
`/home/dhiraj-bohara/projects/MLops/venv`.

Follow the steps **in order**. Do each one in its own terminal tab and leave
long-running ones (webserver, scheduler, mlflow, api, streamlit) running.

---

## 0. Put the files in place

```bash
mkdir -p /home/dhiraj-bohara/projects/MLops
# copy everything from this package into that folder, preserving structure:
# dags/bank.py, api/app.py, steamlit/main.py, the 8 *.py helper scripts,
# airflow.cfg, webserver_config.py, requirements.txt
```

Also copy your dataset and trained model in:
```bash
cp bank-additional-full.csv /home/dhiraj-bohara/projects/MLops/data/
cp bank_model.pkl /home/dhiraj-bohara/projects/MLops/model/   # optional, training will recreate it
```

## 1. System packages

```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3-pip docker.io curl
sudo systemctl enable --now docker
sudo usermod -aG docker $USER   # then log out/in once so docker works without sudo
```

## 2. Create the virtual environment

```bash
cd /home/dhiraj-bohara/projects/MLops
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

## 3. Install Apache Airflow (must go first, via the official constraints file)

```bash
AIRFLOW_VERSION=2.9.3
PYTHON_VERSION=3.11
CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"

pip install "apache-airflow==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}"
pip install apache-airflow-providers-mysql
```

## 4. Install the rest of the project dependencies

```bash
pip install -r requirements.txt
```
If pip reports a version conflict between something in `requirements.txt` and
Airflow's constraints, relax that one pin (e.g. drop the `==x.y.z` to a `>=`)
and reinstall — the pins are a known-good combination, not hard requirements.

## 5. Start MySQL (MariaDB) and Redis in Docker

One-time container creation:
```bash
docker run -d --name mariadb_container \
  -e MYSQL_ROOT_PASSWORD=rootpass \
  -e MYSQL_DATABASE=bank \
  -e MYSQL_USER=bishnu \
  -e MYSQL_PASSWORD='bishnu@pass123' \
  -p 3306:3306 mariadb:11

docker run -d --name redis_container -p 6379:6379 redis:7
```

Give MariaDB ~15 seconds to initialize, then check it:
```bash
docker exec -i mariadb_container mariadb -u bishnu -p'bishnu@pass123' bank -e "SELECT 1;"
```
You should see `1` printed back. (After this first creation, `docker start
mariadb_container redis_container` — which is what the DAG itself runs — is
enough to bring them back up.)

> Note: the DB user/password (`bishnu` / `bishnu@pass123`) match what's
> already hardcoded in the Python scripts. You can change them, but if you
> do, update the `db_url` strings in `preprocessing.py`, `datadrift.py`,
> `conceptdrift.py`, `schema_table.py`, `data_injection.py`, `app.py`, and
> `bank.py` to match.

## 6. Initialize Airflow

```bash
export AIRFLOW_HOME=/home/dhiraj-bohara/projects/MLops
source venv/bin/activate

airflow db init
airflow users create \
  --username admin --password admin \
  --firstname Dhiraj --lastname Bohara \
  --role Admin --email you@example.com
```

`AIRFLOW_HOME` must be exported in **every new terminal** you use for Airflow
commands (webserver, scheduler, trigger, etc.), otherwise Airflow falls back
to `~/airflow` and won't see your `airflow.cfg` or DAGs.

A convenience: add this to `~/.bashrc` so it's automatic:
```bash
echo 'export AIRFLOW_HOME=/home/dhiraj-bohara/projects/MLops' >> ~/.bashrc
```

## 7. Start Airflow (two terminals, both with venv activated + AIRFLOW_HOME exported)

Terminal A:
```bash
airflow webserver --port 8080
```

Terminal B:
```bash
airflow scheduler
```

Open http://localhost:8080 and log in with `admin` / `admin`. You should see
`bank_ml_full_pipeline_dag` in the DAG list (it's paused by default — toggle
it on, or trigger it manually with the ▶ button).

## 8. Run the pipeline

From the UI, click the DAG → trigger (▶). Or from a terminal:
```bash
airflow dags trigger bank_ml_full_pipeline_dag
```

Watch task progress in the Graph view. Order is:
`activate_env → start_docker → docker_exec → mlflow_task → create_bank_bigtable
→ ge_validate_bank_data → data_injection → data_preprocessing → model_training
→ monitor_data_drift_task → monitor_concept_drift_task`

The first full run will take a while — `model_training` runs 50 Optuna trials.

If a task fails, click it → **Logs** to see the actual Python traceback; that's
the fastest way to diagnose anything environment-specific I couldn't predict
from here (e.g. a missing system library for one of the ML packages).

## 9. MLflow UI

The DAG starts it automatically on port 5000, but you can also run it by hand
any time:
```bash
source venv/bin/activate
mlflow ui --host 0.0.0.0 --port 5000 \
  --backend-store-uri file:///home/dhiraj-bohara/projects/MLops/mlflow/mlruns \
  --default-artifact-root /home/dhiraj-bohara/projects/MLops/mlflow/artifacts
```
Open http://localhost:5000 to see experiment runs, params, and metrics.

## 10. FastAPI prediction service

Run after `model_training` has completed at least once and MLflow has a
model version aliased `Production` for `Bank_XGB_Model` (the DAG does this
for you):
```bash
cd /home/dhiraj-bohara/projects/MLops/api
source ../venv/bin/activate
uvicorn app:app --reload --port 8000
```
Check: http://localhost:8000/health → `{"status": "ok"}`

## 11. Streamlit dashboard

```bash
cd /home/dhiraj-bohara/projects/MLops/steamlit
source ../venv/bin/activate
streamlit run main.py
```
Opens at http://localhost:8501 — fill the form, click **Predict**, it calls
the FastAPI service on port 8000.

---

## Run order summary (every time you restart your machine)

1. `docker start mariadb_container redis_container`
2. `source venv/bin/activate && export AIRFLOW_HOME=/home/dhiraj-bohara/projects/MLops`
3. `airflow webserver --port 8080` (terminal A)
4. `airflow scheduler` (terminal B)
5. Trigger/let the DAG run (includes starting MLflow)
6. `uvicorn app:app --reload --port 8000` (from `api/`)
7. `streamlit run main.py` (from `steamlit/`)

## Known things to double check on your machine

- **Python version**: this guide assumes Python 3.11 is available (`python3.11 --version`). If not, `sudo apt install python3.11` may need the deadsnakes PPA on older Ubuntu releases.
- **great_expectations 0.15.x / evidently 0.4.x**: these are pinned to match the older API style your scripts use (`context.add_datasource(...)`, `from evidently.report import Report`, etc.). Newer major versions of either library changed their APIs and would break the scripts as written.
- **Email**: `email_on_failure` is `False` in `bank.py`, so the leftover email addresses in the scripts and `airflow.cfg` are inert. Update them only if you want real failure emails (and configure SMTP in `airflow.cfg`).
