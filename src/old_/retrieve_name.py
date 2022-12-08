import inspect

def retrieve_name(var):
    # code from: https://stackoverflow.com/questions/18425225/getting-the-name-of-a-variable-as-a-string
    
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var][0]

# def get_df_name(df):
# # code from: https://stackoverflow.com/questions/31727333/get-the-name-of-a-pandas-dataframe

#     name =[x for x in globals() if globals()[x] is df][0]
#     return name