import csv

with open('jeopardy_subset2.csv', 'r', newline='') as infile, open('jeopardy_questions2.csv', 'w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    for row in reader:
        if row:  # skip empty rows
            writer.writerow([row[0]])

# import csv

# input_file = 'JEOPARDY_CSV.csv'
# output_file = 'filtered.csv'

# with open(input_file, 'r', newline='', encoding='utf-8') as infile, \
#      open(output_file, 'w', newline='', encoding='utf-8') as outfile:

#     reader = csv.reader(infile)
#     writer = csv.writer(outfile)

#     for row in reader:
#         # Check if any cell in the row contains '<a href='
#         if not any('<a href=' in cell for cell in row):
#             writer.writerow(row)