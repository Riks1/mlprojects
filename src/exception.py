import sys
def error_message_details(error,error_message:sys):
    _,_,er = error_message.exc_info()
    filename = er.tb_frame.f_code.co_filename
    line_num = er.tb_lineno
    return "There is an error in file [{0}] at line [{1}] : [{2}]".format(filename,line_num,str(error))

class CustomException(Exception):
    def __init__(self,error_message,error_detail):  
        super.__init__(error_message)
        self.error_message = error_message_details(error_message,error_detail)
    def __str__(self):
        return self.error_message

