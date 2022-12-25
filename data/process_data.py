'''top level functions to load/clean/save data for disaster response project.'''
import sys
import re
import pandas as pd
import sqlalchemy as db

CAT_PAT = re.compile(r'([a-z_]+)')
BIN_PAT = re.compile(r'([01])')


def load_data(messages_filepath: str, categories_filepath: str):
    '''load message and respective categories data files and merge them.'''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    d_f = pd.merge(messages, categories, on='id')
    return d_f


def clean_data(d_f: pd.DataFrame):
    '''clean given dataframe based on prior analysis.'''
    cat_bin = d_f.categories.str.split(';', expand=True)
    row = d_f.categories[0]
    category_colnames = CAT_PAT.findall(row)
    cat_bin.columns = category_colnames
    for column in cat_bin:
    # set each value to be the last character of the string
        cat_bin[column] = cat_bin[column].str[-1]
    # convert column from string to numeric
        cat_bin[column] = cat_bin[column].astype(int)

    d_f = d_f.drop(['categories'], axis=1)
    df_interim = pd.concat([d_f, cat_bin], axis=1)

    # drop duplicates by row
    df_interim.drop_duplicates(inplace=True)

    # find duplicates by each column
    recur_id= df_interim.id.value_counts().loc[lambda x: x>1].index
    recur_msg = df_interim.message.value_counts().loc[lambda x: x>1].index
    recur_org = df_interim.original.value_counts().loc[lambda x: x>1].index

    df_interim = df_interim[~df_interim.id.isin(recur_id)]
    df_interim = df_interim[~df_interim.message.isin(recur_msg)]
    df_final = df_interim[~df_interim.original.isin(recur_org)]

    return df_final



def save_data(d_f, database_filename):
    '''save cleaned dataframe to a database.'''
    engine = db.create_engine(f'sqlite:///{database_filename}')
    d_f.to_sql('supervised_data', engine, index=False)


def main():
    '''run all.'''
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        d_f = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        d_f = clean_data(d_f)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(d_f, database_filepath)

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
