import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Loads and merges messages and categories datasets
    Args:
        messages_filepath (string): filepath to messages csv file
        categories_filepath (string): filepath to categories csv file
    Returns:
        combined input datasets as pandas dataframe
    '''
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # return merged datasets
    return pd.merge(messages, categories, on='id')


def clean_data(df):
    '''
    Cleans merged messages and categories datasets
    Args:
        df (pandas dataframe): merged messages and categories datasets
    Returns:
        cleaned dataset with category per column
    '''
    
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';', expand=True)
    
    # select the first row of the categories dataframe
    row = categories[:1]
    
    # use this row to extract a list of new column names for categories.
    category_colnames = row.apply(lambda c: c.str.split('-')[0][0], axis=0)
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    # convert category values to just numbers 0 or 1
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(int)

    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    # drop duplicates
    df.drop_duplicates(subset='message', inplace=True)
    
    # return clean dataset
    return df


def save_data(df, database_filename):
    '''
    Saves dataframe to sqlite database
    Args:
        df (pandas dataframe): dataset to be saved in sqlite database
        database_filename (string): filepath to sqlite database file 
    Returns:
        database_filename (string): filepath to newly created sqlite database file
    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('messages', engine, index=False, if_exists='replace')
    
    return database_filename


def main():
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