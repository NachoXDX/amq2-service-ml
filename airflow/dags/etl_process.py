import datetime

from airflow.decorators import dag, task

markdown_text = """
### ETL Student Perfomance Dataset.
In this DAG, data is used from https://www.kaggle.com/datasets/harshadapatil31/student-performance-and-study-habits-dataset/data
The flow is:
    . Download raw data
    . Store it in a s3 bucket.
    . Preprocessed using sklearn column trasnformer
    . Stores the processed data splitted and the column trasnformer's .joblib
All data stored is going to be located at s3://data/student_performance/YYMMDDHHMMSS/
"""

default_args = {
    'owner': "Grupo 4",
    'depends_on_past': False,
    'schedule_interval': None,
    'retries': 1,
    'retry_delay': datetime.timedelta(minutes=5),
    'dagrun_timeout': datetime.timedelta(minutes=15)
}

# Define what the ETL  expects to receive and treatment to be done
expected_features = {
    'student_id':{
        'type':'int64',
        'transform': 'drop', 
    },
    'gender':{
        'type':'category',
        'valid': ['Female', 'Male'],
        'transform': 'ohe'
    },
    'study_time_hours':{
        'type':'float64',
        'transform': 'num',
    },
    'attendance_percent':{
        'type':'float64',
        'transform': 'num',
    },
    'sleep_hours':{
        'type':'float64',
        'transform': 'num',
    },
    'parental_education':{
        'type':'category',
        'valid': ['High School', 'Bachelors', 'Masters', 'PhD'],
        'transform': 'ohe'
    },
    'internet_access':{
        'type':'category',
        'valid': ['Yes', 'No'],
        'transform': 'ohe'
    },
    'extracurricular_activities':{
        'type':'category',
        'valid': ['Yes', 'No'],
        'transform': 'ohe'
    },
    'part_time_job':{
        'type':'category',
        'valid': ['Yes', 'No'],
        'transform': 'ohe'
    },
    'previous_grade':{
        'type':'float64',
        'transform': 'num', 
    },
    'final_exam_score':{
        'type':'float64',
        'transform': 'num', 
    },
    'final_grade':{
        'type':'category',
        'valid': ['A', 'B', 'C', 'D', 'E', 'F'],
        'transform': 'label'
    }
}


@dag(
    dag_id="process_etl_student_performance",
    description="ETL process for student performance.",
    doc_md=markdown_text,
    tags=["ETL", "Student Performance"],
    default_args=default_args,
    catchup=False,
)
def process_etl_student_performance():

    @task.virtualenv(
        task_id="obtain_original_data",
        requirements=[
            "awswrangler==3.6.0",
            "gdown==5.1.0"],
        system_site_packages=True
    )
    def get_data() -> str:
        """
        Load the raw data from google drive, returns the moment when it happend
        """
        import awswrangler as wr
        import datetime
        import gdown
        import os
        import tempfile

        def drive_to_s3(date_str: str):
            file_id = "1r1vNWVotPfX1tpA26jKE7CBpDzeUNGMr"            
            s3_destination_path = f"s3://data/student_performance/{date_str}/raw.csv" 

            # Usar un dir temporal que se borra solo al terminar
            with tempfile.TemporaryDirectory() as temp_dir:
                local_file_path = os.path.join(temp_dir, "raw.csv")
                print("Iniciando descarga desde Google Drive...")
                gdown.download(
                    id=file_id, 
                    output=local_file_path, 
                    quiet=False
                )
                print(f"Descarga completa. Subiendo archivo a {s3_destination_path}...")
                wr.s3.upload(
                    local_file=local_file_path, 
                    path=s3_destination_path
                )
            print("Proceso completado. El archivo temporal fue eliminado del servidor.")

        date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        drive_to_s3(date_str)
        return date_str

    @task.virtualenv(
        task_id="check_original_data",
        requirements=[
            "awswrangler==3.6.0"],
        system_site_packages=True
    )
    def check_data(date_str: str, expected_features: dict) -> None:
        """
        Check the raw data
        """
        import awswrangler as wr
        from airflow.exceptions import AirflowFailException
        import pandas as pd

        # Read dataset
        df = wr.s3.read_csv(f"s3://data/student_performance/{date_str}/raw.csv")

        # Check features
        expected_set = set(expected_features.keys())
        actual_set = set(df.columns)
        missing_features = expected_set - actual_set
        added_features = actual_set - expected_set
        is_exact_set_match = (missing_features == set() and added_features == set())

        # Log results
        if is_exact_set_match:
            print(f"Feature names and quantities came as expected")
        elif len(missing_features) > 0:
            print(f"Missing features ({len(missing_features)}): {missing_features}")
            raise AirflowFailException("CRITICAL: Missing features. Halting DAG execution.")
        else:
            print(f"Added/Extra features ({len(added_features)}): {added_features}")
            raise AirflowFailException("CRITICAL: Added/Extra features. Halting DAG execution.")
    
        # Check dtypes
        dtype_mapping = {
            col: pd.CategoricalDtype(categories=spec['valid']) 
                if spec['type'] == 'category' and 'valid' in spec 
                else spec['type']
            for col, spec in expected_features.items()
        }
        try:
            df = df.astype(dtype_mapping)
            print("Data types converted")
        except Exception as e:
            print(f"Cant convert datatypes: {e}")
            raise AirflowFailException("CRITICAL: Cant convert datatypes. Halting DAG execution.")


    @task.virtualenv(
        task_id="check_missing",
        requirements=[
            "awswrangler==3.6.0"],
        system_site_packages=True
    )
    def check_missing(date_str: str) -> None:
        """
        Check the missing values
        """
        import awswrangler as wr
        from airflow.exceptions import AirflowFailException

        # Read dataset
        df = wr.s3.read_csv(f"s3://data/student_performance/{date_str}/raw.csv")

        #Check
        THRESHOLD = 0.2
        feats_affected = []
        miss = df.isnull().mean()
        for col, value in miss.items():
            if value > 0 and value < THRESHOLD:
                print(f"Some missing values in {col}: {value}")
            elif value >= THRESHOLD:
                print(f"Critical missing values in {col}: {value}")
                feats_affected.append(col)
        if feats_affected:
            raise AirflowFailException("CRITICAL: Too much missing values. Halting DAG execution.")


    @task.virtualenv(
        task_id="transform_data",
        requirements=[
            "awswrangler==3.6.0",
            "scikit-learn==1.3.2",
            "cloudpickle==3.0.0"],
        system_site_packages=True
    )
    def transform_data(date_str: str, expected_features: dict) -> None:
        """
        Trasnforms raw data with Column Transformer using expected_features as reference.
        Handles splitting, missing values, outliers, encoding, scaling and droping
        The result is saved in a bucket in:
            X_train.csv,
            X_test.csv,
            y_train.csv,
            y_test.csv,
            etl_preprocessor.pkl
        """
        from sklearn.model_selection import train_test_split
        from sklearn.base import BaseEstimator, TransformerMixin
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        import cloudpickle
        from airflow.exceptions import AirflowFailException
        import awswrangler as wr
        import os
        import tempfile

        TEST_SIZE = 0.2
        RANDOM_STATE = 42
        IQR_MULTIPLIER = 1.5
        
        # Transformers
        class IQROutlierHandler(BaseEstimator, TransformerMixin):
            """Detects and clips outliers using IQR."""
            def __init__(self, multiplier=1.5):
                self.multiplier = multiplier
                self.lower_bounds_ = {}
                self.upper_bounds_ = {}

            def fit(self, X, y=None):
                # Learns the limits with the training data
                for col in X.columns:
                    q1 = X[col].quantile(0.25)
                    q3 = X[col].quantile(0.75)
                    iqr = q3 - q1
                    self.lower_bounds_[col] = q1 - (self.multiplier * iqr)
                    self.upper_bounds_[col] = q3 + (self.multiplier * iqr)
                return self

            def transform(self, X):
                X_trans = X.copy()
                for col in X.columns:
                    X_trans[col] = X_trans[col].clip(lower=self.lower_bounds_[col], upper=self.upper_bounds_[col])
                return X_trans
            
            def get_feature_names_out(self, input_features=None):
                return input_features

        class FrequencyEncoder(BaseEstimator, TransformerMixin):
            """Encodes categories based on their frequency of appearance."""
            def __init__(self):
                self.mapping_ = {}

            def fit(self, X, y=None):
                for col in X.columns:
                    # Dict with relative freqs
                    self.mapping_[col] = X[col].value_counts(normalize=True).to_dict()
                return self

            def transform(self, X):
                X_trans = X.copy()
                for col in X.columns:
                    # Maps using 0 if unseen
                    X_trans[col] = X_trans[col].map(self.mapping_[col]).fillna(0)
                return X_trans
            
            def get_feature_names_out(self, input_features=None):
                return input_features

        # Load data
        df = wr.s3.read_csv(f"s3://data/student_performance/{date_str}/raw.csv")

        # Define variables for pipeline
        target_col = "final_grade"
        num_cols = [k for k,v in expected_features.items() if v['transform'] == 'num']
        cat_ohe_cols = [k for k,v in expected_features.items() if v['transform'] == 'ohe']
        cat_freq_cols = [k for k,v in expected_features.items() if v['transform'] == 'freq']
        cols_to_drop = [k for k,v in expected_features.items() if v['transform'] == 'drop']

        # Check if any was left behind
        total_feats = len(num_cols) + len(cat_freq_cols) + len(cat_ohe_cols) + len(cols_to_drop)
        if total_feats != len(df.keys())-1:
            raise AirflowFailException("CRITICAL: Some features are not being processed. Halting DAG execution.")
        else:
            print("All features to be processed")

        #Beggin Processing

        # Split
        X = df.drop(columns=[target_col])
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y
        )

        #Join categoricals for intialization
        cat_cols = cat_ohe_cols + cat_freq_cols

        #Send each categorical to its encoder
        categorical_encoders = ColumnTransformer(
            transformers=[
                ('ohe', OneHotEncoder(sparse_output=False,handle_unknown='ignore'), cat_ohe_cols),
                ('freq', FrequencyEncoder(), cat_freq_cols)
            ],
            remainder='drop',
            n_jobs=1
        )

        #Pipeline for categoricals
        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoders', categorical_encoders)
        ])

        #Pipeline for numerics
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('outliers', IQROutlierHandler(multiplier=IQR_MULTIPLIER)),
            ('scaler', StandardScaler())
        ])

        #Preprocesador master
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_pipeline, num_cols),
                ('cat', cat_pipeline, cat_cols),
                ('dropper', 'drop', cols_to_drop),
            ],
            remainder='drop',
            n_jobs=1
        )

        preprocessor.set_output(transform="pandas")

        #Fit in train and transform both
        X_train_processed = preprocessor.fit_transform(X_train)            
        X_test_processed = preprocessor.transform(X_test)

        # Save .pkl of the processor
        with tempfile.TemporaryDirectory() as temp_dir:
            local_export_path = os.path.join(temp_dir, "etl_preprocessor.pkl")
            with open(local_export_path, "wb") as f:
                cloudpickle.dump(preprocessor, f)
            s3_destination = f"s3://data/student_performance/{date_str}/etl_preprocessor.pkl"
            wr.s3.upload(
                local_file=local_export_path,
                path=s3_destination
            )

        def save_to_csv(df, path):
            wr.s3.to_csv(df=df,
                         path=path,
                         index=False)

        save_to_csv(X_train_processed, f"s3://data/student_performance/{date_str}/X_train.csv")
        save_to_csv(X_test_processed, f"s3://data/student_performance/{date_str}/X_test.csv")
        save_to_csv(y_train, f"s3://data/student_performance/{date_str}/y_train.csv")
        save_to_csv(y_test, f"s3://data/student_performance/{date_str}/y_test.csv")

    @task.virtualenv(
    task_id="log_to_mlFlow",
    requirements=[
        "awswrangler==3.6.0",
        "mlflow==2.10.1"],
    system_site_packages=True
    )
    def log_to_mlFlow(date_str: str, expected_features: dict) -> None:
        """
        Logs a run on MLflow to keep track of the ETL execution:
        dataset, schema, preprocessor artifact, and descriptive metrics.
        """
        import mlflow
        import awswrangler as wr
        import pandas as pd
        import json
        import tempfile
        import os

        data_path = f"s3://data/student_performance/{date_str}/raw.csv"
        df = wr.s3.read_csv(data_path)

        mlflow.set_tracking_uri('http://mlflow:5000')
        experiment = mlflow.set_experiment("Student Performance")

        with mlflow.start_run(
            run_name='ETL_run_' + date_str,
            experiment_id=experiment.experiment_id,
            tags={"experiment": "etl", "dataset": "Student Performance"},
            log_system_metrics=True
        ):
            #Log Dataset
            mlflow_dataset = mlflow.data.from_pandas(
                df,
                source=data_path,
                targets="final_grade",
                name="student_performance_data_complete"
            )
            mlflow.log_input(
                mlflow_dataset,
                context="Dataset",
                tags={"provenance": "https://www.kaggle.com/datasets/harshadapatil31/student-performance-and-study-habits-dataset/data"}
            )

            #Log Expected Schema
            with tempfile.TemporaryDirectory() as temp_dir:
                schema_path = os.path.join(temp_dir, "expected_features.json")
                with open(schema_path, "w") as f:
                    json.dump(expected_features, f, indent=2)
                mlflow.log_artifact(schema_path, artifact_path="schema")

            #Log Preprocessor
            preprocessor_s3_path = f"s3://data/student_performance/{date_str}/etl_preprocessor.pkl"
            with tempfile.TemporaryDirectory() as temp_dir:
                local_preprocessor_path = os.path.join(temp_dir, "etl_preprocessor.pkl")
                wr.s3.download(path=preprocessor_s3_path, local_file=local_preprocessor_path)
                mlflow.log_artifact(local_preprocessor_path, artifact_path="preprocessor")

            mlflow.log_param("preprocessor_s3_path", preprocessor_s3_path)
            mlflow.log_param("sklearn_version", "1.3.2")
            mlflow.log_param("cloudpickle_version", "3.0.0")

    
    date_str = get_data()
    validate_step = check_data(date_str, expected_features)
    missing_step = check_missing(date_str)
    transform_step = transform_data(date_str, expected_features)
    mlflow_step = log_to_mlFlow(date_str, expected_features)

    validate_step >> missing_step >> transform_step >> mlflow_step

dag = process_etl_student_performance()