def cleaning_header(df, lower_case = True):
    '''Function that changes all names to lower case if the parameter lower_case is True or to upper case if
    the parameter lower_case is False. Also deletes the white spaces at the end and at first of the columns name'''
    if(lower_case):
        df.columns = df.columns.str.lower()
    else:
        df.columns = df.columns.str.upper()
        
    df.columns = df.columns.str.strip()
    return(df)





