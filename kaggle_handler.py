import os
from typing import Any, Optional
from config import Config
from constant import IS_KAGGLE

from kaggle.api.kaggle_api_extended import KaggleApi # type: ignore

class KaggleHandler:
    def __init__(self, config: Config):
        self.config = config
        self.api = KaggleApi()
        self.api.authenticate()
    
    def upload_pipeline(self, pipeline: Any, dataset_title: Optional[str] = None):
        """Upload pipeline to Kaggle dataset"""
        if IS_KAGGLE:
            raise ValueError("This function is for local environment only")
        
        tmp_dir = "./kaggle_upload"
        os.makedirs(tmp_dir, exist_ok=True)
        
        filename = f"{self.config.dataset_name.split('/')[-1]}.pkl"
        pipeline_path = os.path.join(tmp_dir, filename)
        
        original_path = pipeline.config.model_path
        pipeline.config.model_path = os.path.join(os.path.dirname(original_path), filename)
        
        pipeline.save()
        
        import shutil
        shutil.copy2(pipeline.config.model_path, pipeline_path)
        
        metadata = {
            "title": dataset_title or self.config.dataset_name.split('/')[-1],
            "id": f"{self.config.dataset_name}",
            "licenses": [{"name": "CC0-1.0"}]
        }
        
        import json
        with open(os.path.join(tmp_dir, "dataset-metadata.json"), "w") as f:
            json.dump(metadata, f)
        
        try:
            print(f"Creating new dataset: {self.config.dataset_name}")
            self.api.dataset_create_new(
                folder=tmp_dir,
                public=False, # For private datasets
                quiet=False
            )
            print("Dataset created successfully")
            
        except Exception as e:
            print(f"Error creating Kaggle dataset: {e}")
            raise
        finally:
            shutil.rmtree(tmp_dir)