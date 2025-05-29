"""Script for parsing pobedish.ru"""
#from selenium import webdriver
#from selenium.webdriver.common.by import By
import time
import requests
from bs4 import BeautifulSoup
import csv
from lxml import etree
import pandas as pd
from uuid import uuid4


def split_info_column(value: str) -> pd.DataFrame:
    """Split column with info.

    Args:
        value (str): The value of the field.

    Returns:
        pd.DataFrame: The structured info.
    """
    parts = value.split(",")

    # Извлечение имени
    name = parts[0].strip()

    # Проверка на наличие возраста и даты
    if len(parts) > 1:
        age_date_part = parts[1].split("/")

        # Извлечение возраста и даты
        if len(age_date_part) > 1:
            age = age_date_part[0].replace("возраст:", "").strip()
            datetime = age_date_part[1].strip().replace(".", "-")
        else:
            age = age_date_part[0].replace("возраст:", "").strip()
            datetime = ""
    else:
        age = ""
        datetime = ""

    return pd.Series([name, age, datetime])


if __name__ == "__main__":
    # 1. Получение ссылок со страниц
    driver = webdriver.Chrome()

    with open("links.txt", "w") as file:
        for i in range(1, 1559):
            url = f"https://pobedish.ru/main/help?action=&keyword=&where=&page={i}"
            driver.get(url)

            time.sleep(2)

            for j in range(2, 22):
                try:
                    link_element = driver.find_element(
                        By.XPATH, f'//*[@id="content"]/a[{j}]'
                    )
                    link = link_element.get_attribute("href")
                    if link:
                        file.write(link + "\n")
                except:
                    continue

    driver.quit()

    # 2. Парсинг данных с полученных ссылок
    with open("links.txt", "r") as links_file:
        urls = links_file.readlines()

        with open("forum_pobedish.csv", mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["thread_id", "Информация", "text"])

            topic_id = 1  # id темы

            for url in urls:
                url = url.strip()

                response = requests.get(url)
                # Проверяем работу страницы
                if response.status_code != 200:
                    print(f"Ошибка при загрузке страницы: {response.status_code}")
                    break

                # Парсим HTML-контент страницы
                soup = BeautifulSoup(response.content, "html.parser")

                # Извлекаем имена и комментарии
                names = soup.find_all(class_="ist")
                root = etree.HTML(str(soup))  # Так как BeautifulSoup не работает с xpath
                raw_comments = root.xpath('//*[@id="content"]/text()')

                # Функция для обработки списка комментариев
                # По-другому правильно комментарии не извлечь(
                def process_comments(comments):
                    processed_comments = []
                    current_comment = []

                    # Маркеры начала нового комментария
                    comment_start_markers = ["\r\n\t", "\r\n\t\t", "\t"]

                    for line in comments:
                        # Проверяем, начинается ли строка с маркера нового комментария
                        if any(line.startswith(marker) for marker in comment_start_markers):
                            # Если уже есть накопленный комментарий, сохраняем его
                            if current_comment:
                                processed_comment = "".join(current_comment).strip()
                                if processed_comment:  # Иногда попадаются пустые строки
                                    processed_comments.append(processed_comment)
                            current_comment = []

                        # Добавляем строку в текущий комментарий
                        current_comment.append(line.strip())

                    # Добавляем последний комментарий, если он есть
                    if current_comment:
                        processed_comment = "".join(current_comment).strip()
                        if processed_comment:
                            processed_comments.append(processed_comment)

                    return processed_comments

                # Обработка списка комментариев
                cleaned_comments = process_comments(raw_comments)
                topic_id += 1

                for name, comment in zip(names, cleaned_comments):
                    name_text = name.text.strip()
                    writer.writerow([topic_id, name_text, comment])

    # 3. Обработка CSV-файла
    df = pd.read_csv("forum_pobedish.csv")

    df[["name", "age", "datetime"]] = df["Информация"].apply(split_info_column)

    df.drop(columns=["Информация"], inplace=True)

    # Добавление домена
    df["src"] = "pobedish.ru"

    # Добавление столбца 'id' для каждого комментария
    df["id"] = df.apply(lambda _: uuid4().hex, axis=1)

    df.to_csv("itog.csv", index=False)
