import sys , os

def error_messafe_details(error,error_details:sys):
    _, _, exc_tb = error_details.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occurred python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )
    return error_message



class PatientsException(Exception):

    def __init__(self,error_message, error_detail:sys):
        self.error_message = error_message(
            error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message