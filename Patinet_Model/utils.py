import pandas as pd 
from Patinet_Model.logger import logging
from Patinet_Model.exception import PatientsException
import os
import sys
import  numpy as np 
import yaml
import dill

def get_collection_as_dataframe(database_name:str,collection_name:str)->pd.DataFrame:

    try:
        logging.info(f"Reading the data from database : {database_name} and collection {collection_name}")
        #df = pd.DataFrame(list(mongo_client[database_name][colllection_name].find()))
        df = pd.read_csv('indian_liver_patient.csv')
        logging.info(f"Founds Columns : {df.columns}")
        if "_id" in df.columns:
            logging.info(f"Dropping columns : _id")
            df = df.drop("_id",axis=1)

        logging.info(f"Row and columns in df : {df.shape}")
        return df
    except Exception as e:
        raise PatientsException(error_message=e, error_detail=sys)
    
def write_ymal_file(file_path , data : dict):
    try:
        file_dir = os.path.dirname(file_path)
        os.makedirs(file_dir,exist_ok=True)
        with open(file_path,"w") as file_writer:
            yaml.dump(data,file_writer)
    except Exception as e :
        raise PatientsException(e, sys)
    
def convert_columns_float(df:pd.DataFrame,exclude_columns:list)->pd.DataFrame:
    try:
        for column in df.columns:
            if column not in exclude_columns:
                df[column]=df[column].astype('float')
        return df
    except Exception as e:
        raise e

def save_object(file_path: str, obj: object) -> None:
    try:
        logging.info("Entered the save_object method of utils")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        logging.info("Exited the save_object method of utils")
    except Exception as e:
        raise PatientsException(e, sys) from e


def load_object(file_path: str, ) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} is not exists")
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise PatientsException(e, sys) from e

def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise PatientsException(e, sys) from e

def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise PatientsException(e, sys) from e