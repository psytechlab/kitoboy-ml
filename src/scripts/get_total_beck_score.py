"""Compute the Beck score based on csv-table with answers on Beck Depression Inventory"""
import csv
import json



def calculate_score(row: list[str], mapping: dict[str, int]) -> int:
    """Calculate Beck score for one row,
    which is corresponded to one person.

    Args:
        row (list[str]): The row with selected answers.
        mapping (dict[str, int]): Mapping from answers to value.

    Returns:
        int: The total Beck score.
    """

    score = 0
    # 6 and 27 are actual answers.
    for i in range(6, 27):
        answer = row[i]
        if answer in mapping:
            score += mapping[answer]
    return score


if __name__ == "__main__":

    with open('references/beck_test_mapping.json', 'r', encoding='utf-8') as json_file:
        mapping = json.load(json_file)
        
    with open('Текст_шкалы_депрессии_Бека.csv', 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, quotechar='"')
        header = next(csv_reader)

        with open('output.csv', 'w', encoding='utf-8', newline='') as out_csv:
            csv_writer = csv.writer(out_csv, quotechar='"', quoting=csv.QUOTE_ALL)

            csv_writer.writerow(header + ["Общий результат"])

            for row in csv_reader:
                score = calculate_score(row, mapping)
                csv_writer.writerow(row + [score])

