import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load messages and categories csv files into dataframes
    and merge them into a single dataframe called df
    Inputs:
    ------
    messages_filepath: file path for messages.csv
    categories_filepath: file path for categories.csv

    Return:
    ------
    df combining messages and categories
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')

    return df

def clean_data(df):
    """
    Perform data wrangling on df

    Return:
    ------
    a clean dataset ready for further analysis
    """

    #Split categories into separate category columns
    categories = pd.DataFrame(data=df.categories.str.split(";", expand=True))

    # Use the first row to extract a list of new column names for categories
    row = categories.iloc[0, :]
    category_colnames = row.apply(lambda x: x[:-2])

    # rename the columns of `categories`
    categories.columns = category_colnames

    #Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    #Replace categories column in df with new category columns
    df.drop(columns='categories', inplace=True)

    # Concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # Remove duplicates
    df.drop_duplicates(inplace=True)
    return df

def save_data(df, database_filename):
    """
    Save the clean dataset into an sqlite database

    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('messages_categories', engine, index=False)


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
