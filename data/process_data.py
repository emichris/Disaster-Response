import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
     Function reads in both datasets
     It returns a dataframe with messages and categories merged on 'id'
     
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    
    return df
    

def clean_data(df):
    '''
    This is the transformation step of the ETL process:
    Function splits the categories column into separate, clearly named columns,
    converts values to binary, and drops duplicates.
    '''
    categories = df.categories.str.split(';', expand=True)
    first_row = categories.iloc[0] # select the first row of the categories dataframe
    category_colnames = first_row.apply(lambda x: x[:-2]) # Extract colum names
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    df.drop(columns=['categories'], inplace=True)
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(inplace=True)
        
    return df


def save_data(df, database_filename):
    '''
    This is the load step of the ETL process. Function writes the dataframe into an SQLite database in the specified database file path.
    '''
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql(database_filename, engine, index=False) 


def main():
    '''
    Combines all three functions above to perform the ETL process taking user input: messages_filepath, categories_filepath, database_filepath
    '''
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()