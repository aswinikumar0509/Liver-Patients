from Patinet_Model import utils
from Patinet_Model.logger import logging
from Patinet_Model.exception import PatientsException
from Patinet_Model.entity import config_entity
from Patinet_Model.entity import artifact_entity
from typing import Optional
from sklearn.pipeline import Pipeline
import pandas as pd
from Patinet_Model import utils
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.combine import SMOTETomek
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from Patinet_Model.config import TARGET_COLUMN
from sklearn.preprocessing import LabelEncoder
import os,sys

class DataTransformation:

    def __init__(self, data_transformation_config:config_entity.DataTransformationConfig,data_ingestion_artifact:artifact_entity.DataIngestionArtifact):

        try:
            logging.info(f"{'>>'*20} Data Transformation {'<<'*20}")
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
        except Exception as e:
            raise PatientsException(e,sys)

    @classmethod
    def get_data_transformer_object(cls)->Pipeline:

        try:
            Simple_Imputer = SimpleImputer(strategy = 'constant',fill_value = 0)
            Robust_Scaler = RobustScaler()
            pipeline = Pipeline(steps = [('Imputer',Simple_Imputer),('RobustScaler',Robust_Scaler)])
            return pipeline

        except Exception as e:
            raise PatientsException(e,sys)

    def initiate_data_transformation(self,)->artifact_entity.DataIngestionArtifact:

        try:
            #reading training and testing file
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            train_df['Gender'] = train_df['Gender'].replace({'Male': 0, 'Female': 1})
            test_df['Gender'] = test_df['Gender'].replace({'Male': 0, 'Female': 1})


            # Selecting input feature from the test and train df

            input_feature_train_df = train_df.drop(TARGET_COLUMN,axis=1)
            input_feature_test_df = test_df.drop(TARGET_COLUMN,axis=1)

            # Selecting the target feature from the train and test df

            target_column_train_df = train_df[TARGET_COLUMN]
            target_column_test_df = test_df[TARGET_COLUMN]

            label_encoder = LabelEncoder()
            label_encoder.fit(target_column_train_df)

            #Transform the target columns

            target_feature_train_arr = label_encoder.transform(target_column_train_df)
            target_feature_test_arr = label_encoder.transform(target_column_test_df)

            transformation_pipeline = DataTransformation.get_data_transformer_object()
            transformation_pipeline.fit(input_feature_train_df)

            # Transformation
            input_feature_train_arr = transformation_pipeline.transform(input_feature_train_df)
            input_feature_test_arr = transformation_pipeline.transform(input_feature_test_df)

            smt = SMOTETomek(random_state=42)
            logging.info(f"Before resampling in training set Input: {input_feature_train_arr.shape} Target:{target_feature_train_arr.shape}")
            input_feature_train_arr, target_feature_train_arr = smt.fit_resample(input_feature_train_arr, target_feature_train_arr)
            logging.info(f"After resampling in training set Input: {input_feature_train_arr.shape} Target:{target_feature_train_arr.shape}")
            
            logging.info(f"Before resampling in testing set Input: {input_feature_test_arr.shape} Target:{target_feature_test_arr.shape}")
            input_feature_test_arr, target_feature_test_arr = smt.fit_resample(input_feature_test_arr, target_feature_test_arr)
            logging.info(f"After resampling in testing set Input: {input_feature_test_arr.shape} Target:{target_feature_test_arr.shape}")

            #target encoder
            train_arr = np.c_[input_feature_train_arr, target_feature_train_arr ]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_arr]


            #save numpy array
            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_train_path,
                                        array=train_arr)

            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_test_path,
                                        array=test_arr)


            utils.save_object(file_path=self.data_transformation_config.tranformation_object_path,
             obj=transformation_pipeline)

            utils.save_object(file_path=self.data_transformation_config.target_encoder_path,
            obj=label_encoder)



            data_transformation_artifact = artifact_entity.DataTransformationArtifact(
                transform_object_path=self.data_transformation_config.tranformation_object_path,
                transformed_train_path = self.data_transformation_config.transformed_train_path,
                transformed_test_path = self.data_transformation_config.transformed_test_path,
                target_encoder_path = self.data_transformation_config.target_encoder_path

            )
            logging.info(f"Data transformation object {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise PatientsException(e, sys)