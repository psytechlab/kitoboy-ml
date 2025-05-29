"""Compute emotional score from the questionary"""
import csv

if __name__ == "__main__":
    score_mapping = {
        "вовсе нет": 0,
        "редко": 1,
        "иногда": 2,
        "часто": 3,
        "постоянно": 4
    }

    input_file = 'input.csv'
    output_file = 'output.csv'
    with open(input_file, mode='r', encoding='utf-8') as infile, \
        open(output_file, mode='w', newline='', encoding='utf-8') as outfile:

        reader = csv.reader(infile)
        writer = csv.writer(outfile, quoting=csv.QUOTE_ALL)

        headers = next(reader)
        headers.append("Общий результат")
        writer.writerow(headers)

        for row in reader:
            # counting sum of score starting from 7 column (previous contains meta)
            score_sum = sum(score_mapping.get(row[i], 0) for i in range(7, len(row)))
            row.append(str(score_sum))
            writer.writerow(row)

    print(f"Обработка завершена. Результат сохранен в файле {output_file}.")

