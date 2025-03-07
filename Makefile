# Déclaration des variables
PYTHON=python3
ENV_NAME=venv
REQUIREMENTS=requirements.txt
IMAGE_NAME=ml-docker-app
CONTAINER_NAME=ml-app

# 1. Configuration de l'environnement
setup:
	@echo "Création de l'environnement virtuel et installation des dépendances..."
	@$(PYTHON) -m venv $(ENV_NAME)
	@$(ENV_NAME)/bin/pip install -r $(REQUIREMENTS)

# 2. Qualité du code
lint:
	@echo "Vérification du code..."
	@$(ENV_NAME)/bin/flake8 --max-line-length=120 --exclude=$(ENV_NAME) . || true
	@$(ENV_NAME)/bin/black --exclude "$(ENV_NAME)/.*" . || true
	@$(ENV_NAME)/bin/bandit -r . -x $(ENV_NAME)/ || true

# 3. Préparation des données
data:
	@echo "Préparation des données..."
	@. $(ENV_NAME)/bin/activate && $(PYTHON) -c "from model_pipeline import test_prepare_data; test_prepare_data()"

# 4. Entraînement du modèle
train:
	@echo "Entraînement du modèle..."
	@$(ENV_NAME)/bin/python model_pipeline.py

# 5. Exécution des tests
test:
	@echo "Exécution des tests..."
	@export PYTHONPATH=$(PWD) && $(ENV_NAME)/bin/pytest tests/

# 6. Évaluation du modèle
evaluate:
	@echo "Évaluation du modèle..."
	@$(ENV_NAME)/bin/python -c "from main import test_load_model, test_evaluate_model, test_prepare_data; \
		X_train, X_test, y_train, y_test = test_prepare_data(); \
		model = test_load_model('xgboost_model.pkl'); \
		test_evaluate_model(model, X_test, y_test)"

# 7. Déploiement du modèle
deploy:
	@echo "Déploiement du modèle..."
	@. $(ENV_NAME)/bin/activate && $(PYTHON) --deploy

# 8. Démarrage du serveur Jupyter Notebook
.PHONY: notebook
notebook:
	@echo "Démarrage de Jupyter Notebook..."
	@$(ENV_NAME)/bin/jupyter notebook

# 9. Nettoyage de l'environnement
clean:
	@echo "Suppression des fichiers temporaires..."
	@find . -type f -name "*.install" -exec rm -f {} +
	@find . -type f -name "*.pyc" -exec rm -f {} +
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@echo "Nettoyage terminé."

# 10. Exécution complète du pipeline
all: setup lint data train evaluate test deploy
	@$(ENV_NAME)/bin/flake8 --max-line-length=120 --exclude=$(ENV_NAME) . || true
	@$(ENV_NAME)/bin/black --exclude "$(ENV_NAME)/.*" . || true
	@$(ENV_NAME)/bin/bandit -r . -x $(ENV_NAME)/ || true

# 11. Exécution de l'API FastAPI
.PHONY: run_api
run_api:
	@echo "Démarrage de l'API FastAPI..."
	@$(ENV_NAME)/bin/uvicorn app:app --reload --host 127.0.0.1 --port 8000 &
	@sleep 2
	@if command -v xdg-open > /dev/null; then xdg-open http://127.0.0.1:8000/docs; \
	elif command -v open > /dev/null; then open http://127.0.0.1:8000/docs; \
	elif command -v start > /dev/null; then start http://127.0.0.1:8000/docs; \
	else echo "Impossible d'ouvrir Swagger automatiquement. Ouvre http://127.0.0.1:8000/docs manuellement."; fi

# 12. Construction de l'image Docker
.PHONY: build
build:
	@echo "Construction de l'image Docker..."
	@docker build -t $(IMAGE_NAME) .

# 13. Exécution du conteneur Docker
.PHONY: run
run:
	@echo "Lancement du conteneur Docker..."
	@docker run -d -p 5001:5001 --name $(CONTAINER_NAME) $(IMAGE_NAME)

# 14. Nettoyage des conteneurs Docker
.PHONY: docker-clean
docker-clean:
	@echo "Arrêt et suppression du conteneur Docker..."
	@docker stop $(CONTAINER_NAME) || true
	@docker rm $(CONTAINER_NAME) || true
