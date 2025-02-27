import pandas as pd
import numpy as np
import tqdm
import os
import argparse
import psycopg2

from ai_clinician.data_extraction.sql.queries_31 import SQL_QUERY_FUNCTIONS
from ai_clinician.preprocessing.columns import RAW_DATA_COLUMNS, STAY_ID_OPTIONAL_DTYPE_SPEC

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import os
import pandas as pd
import tqdm
import psycopg2

def load_data(pg_conn, file_name, query_fn, output_dir, skip_if_present=False):
    """
    Loads data from PostgreSQL using the SQL query provided by the file name.
    For 'ce' (chartevents), loads data in chunks to handle large datasets.
    """
    
    if file_name == 'ce':
        # Handle chartevents separately by processing in batches
        id_step = int(1e6)
        id_max = int(1e7)
        id_conversion = int(3e7)

        # Loop over the ID range in steps to process data in chunks
        for i in range(0, id_max, id_step):
            out_path = os.path.join(output_dir, file_name + str(i) + str(i + id_step) + '.csv')
            # out_path = os.path.join(output_dir, f"{file_name}_{i}_{i + id_step}.csv")

            # Skip if the file already exists and skip_if_present is True
            if skip_if_present and os.path.exists(out_path):
                print(f"File {out_path} exists, skipping")
                continue

            # Generate query for the current batch
            query_ = query_fn(id_conversion + i, id_conversion + id_step + i)
            print(f"Executing query for batch {i} to {i + id_step}:\n{query_}")

            # Execute the query using the PostgreSQL connection
            with pg_conn.cursor() as cursor:
                cursor.execute(query_)
                results = cursor.fetchall()

            # Convert the query results to a pandas DataFrame
            result = pd.DataFrame([dict(zip(range(len(res)), res)) for res in tqdm.tqdm(results, desc=f"{file_name}_{i}_{i + id_step}")])

            # Rename columns based on the predefined schema for chartevents
            result.columns = RAW_DATA_COLUMNS['ce']

            # Adjust column data types if needed
            for col in result.columns:
                if col in STAY_ID_OPTIONAL_DTYPE_SPEC:
                    result[col] = result[col].astype(STAY_ID_OPTIONAL_DTYPE_SPEC[col])

            # Save the DataFrame to a CSV file
            result.to_csv(out_path, index=False)

        return  # End early since chartevents are processed in batches

    # General case for all other file types
    out_path = os.path.join(output_dir, file_name + '.csv')
    if skip_if_present and os.path.exists(out_path):
        print('File exists, skipping')
        return

    # Generate the query using the provided query function
    query = query_fn()
    if not query:
        return

    print(query)

    # Execute the query and load the data into a DataFrame
    with pg_conn.cursor() as cursor:
        cursor.execute(query)
        results = cursor.fetchall()

    # Create a DataFrame from the results
    result = pd.DataFrame(results, columns=RAW_DATA_COLUMNS[file_name])

    # Adjust column data types if needed
    for col in result.columns:
        if col in STAY_ID_OPTIONAL_DTYPE_SPEC:
            result[col] = result[col].astype(STAY_ID_OPTIONAL_DTYPE_SPEC[col])

    # Save the DataFrame to a CSV file
    result.to_csv(out_path, index=False)

def main():
    parser = argparse.ArgumentParser(description=('Loads data from PostgreSQL and '
        'saves them as local CSV files.'))
    parser.add_argument('db_url', type=str, help='PostgreSQL database URL (e.g., postgres://user:password@host:port/dbname)')
    parser.add_argument('--out', dest='output_dir', type=str, default=None,
                        help='Directory in which to output (default is ../data directory)')
    parser.add_argument('--skip-existing', dest='skip_existing', action='store_true', default=False,
                        help='If passed, skip existing CSV files')

    args = parser.parse_args()

    # Connect to PostgreSQL database
    pg_conn = psycopg2.connect(args.db_url)

    out_dir = args.output_dir or os.path.join(PARENT_DIR, 'data', 'raw_data')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        
    for file_name, fn in SQL_QUERY_FUNCTIONS.items():
        print(file_name)
        load_data(pg_conn, file_name, fn, out_dir, skip_if_present=args.skip_existing)

    # Close the database connection
    pg_conn.close()

if __name__ == '__main__':
    main()
