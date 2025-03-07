pipeline {
    agent any

    environment {
        PYTHON = 'python3'
        ENV_NAME = 'venv'
        REQUIREMENTS = 'requirements.txt'
    }

    stages {
        stage('Setup Environment') {
            steps {
                script {
                    echo "Création de l'environnement virtuel et installation des dépendances..."
                    sh "${PYTHON} -m venv ${ENV_NAME}"
                    sh "${ENV_NAME}/bin/pip install -r ${REQUIREMENTS}"
                }
            }
        }

        stage('Lint Code') {
            steps {
                script {
                    echo "Vérification du code..."
                    sh "${ENV_NAME}/bin/flake8 --max-line-length=120 --exclude=${ENV_NAME} . || true"
                    sh "${ENV_NAME}/bin/black --exclude \"${ENV_NAME}/.*\" . || true"
                    sh "${ENV_NAME}/bin/bandit -r . -x ${ENV_NAME}/ || true"
                }
            }
        }

        stage('Prepare Data') {
            steps {
                script {
                    echo "Préparation des données..."
                    sh ". ${ENV_NAME}/bin/activate && ${PYTHON} -c 'from model_pipeline import test_prepare_data; test_prepare_data()'"
                }
            }
        }

        stage('Train Model') {
            steps {
                script {
                    echo "Entraînement du modèle..."
                    sh "${ENV_NAME}/bin/python model_pipeline.py"
                }
            }
        }

        stage('Run Tests') {
            steps {
                script {
                    echo "Exécution des tests..."
                    sh "export PYTHONPATH=${PWD} && ${ENV_NAME}/bin/pytest tests/"
                }
            }
        }

        stage('Evaluate Model') {
            steps {
                script {
                    echo "Évaluation du modèle..."
                    sh "${ENV_NAME}/bin/python -c 'from main import test_load_model, test_evaluate_model, test_prepare_data; X_train, X_test, y_train, y_test = test_prepare_data(); model = test_load_model(\"xgboost_model.pkl\"); test_evaluate_model(model, X_test, y_test)'"
                }
            }
        }

        stage('Deploy Model') {
            steps {
                script {
                    echo "Déploiement du modèle..."
                    sh ". ${ENV_NAME}/bin/activate && ${PYTHON} --deploy"
                }
            }
        }

        stage('Run API') {
            steps {
                script {
                    echo "Démarrage de l'API FastAPI..."
                    sh "${ENV_NAME}/bin/uvicorn app:app --reload --host 127.0.0.1 --port 8000 &"
                    sleep 2
                    sh """
                        if command -v xdg-open > /dev/null; then xdg-open http://127.0.0.1:8000/docs;
                        elif command -v open > /dev/null; then open http://127.0.0.1:8000/docs;
                        elif command -v start > /dev/null; then start http://127.0.0.1:8000/docs;
                        else echo "Impossible d'ouvrir Swagger automatiquement. Ouvre http://127.0.0.1:8000/docs manuellement."; fi
                    """
                }
            }
        }
    }

    post {
        always {
            echo "Nettoyage de l'environnement..."
            sh "find . -type f -name '*.install' -exec rm -f {} +"
            sh "find . -type f -name '*.pyc' -exec rm -f {} +"
            sh "find . -type d -name '__pycache__' -exec rm -rf {} +"
            echo "Nettoyage terminé."
        }
    }
}