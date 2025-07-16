import gzip
import csv
import pandas as pd

# download here: https://archive.ics.uci.edu/dataset/280/higgs

# as the actual higgs.csv file is too big, it's  been gz'd
# and we cannot unpack it, the computer crashes
# so, the below code splits it into 11 csv files, and then combines 5 of them later

INPUT_FILE = 'HIGGS.csv.gz'
ROWS_PER_CHUNK = 1000000
MAX_CHUNKS = 11  

def split(input_gz, rows_per_chunk, max_chunks):

    # splits higgs.csv.gz it into 11 files

    with gzip.open(input_gz, 'rt') as f:
        reader = csv.reader(f)
        chunk = 1
        row_count = 0
        out_file = open(f'higgs_part_{chunk}.csv', 'w', newline='')
        writer = csv.writer(out_file)

        for row in reader:
            if row_count == rows_per_chunk:
                out_file.close()
                chunk += 1
                if chunk > max_chunks:
                    break
                out_file = open(f'higgs_part_{chunk}.csv', 'w', newline='')
                writer = csv.writer(out_file)
                row_count = 0
            writer.writerow(row)
            row_count += 1

        out_file.close()

split(INPUT_FILE, ROWS_PER_CHUNK, MAX_CHUNKS)

# now, combine 5 of them into one, named "higgs_prime.csv"

df_prime = pd.read_csv("higgs_part_1.csv")
df_prime = df_prime.drop(columns = df_prime.columns[0])

for i in range(2, 6):
    df = pd.read_csv(f"higgs_part_{i}.csv")
    df = df.drop(columns = df.columns[0])

    df_prime = pd.concat([df_prime, df])
    print(f"Part {i} done.")

df_prime.to_csv("higgs_prime.csv", header = False, index = False)