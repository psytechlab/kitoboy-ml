"""Script for parsing of palata6.net forum"""
# -*- coding: utf-8 -*-
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec
from selenium.common.exceptions import NoSuchElementException
import time
import pandas as pd
from bs4 import BeautifulSoup
from uuid import uuid4

def get_thread_id(topic: str) -> int:
    """Create thread_id for each topic.

    Args:
        topic (str): The topic link.

    Returns:
        int: The corresponding thread_id.
    """
    global current_id
    if topic not in topic_to_thread_id:
        topic_to_thread_id[topic] = current_id
        current_id += 1
    return topic_to_thread_id[topic]

# Функция для парсинга текущей страницы
def parse_current_page(driver: webdriver):
    """Parse current page in driver.

    Args:
        driver (webdriver): The selenium driver.
    """
    # Нахождение и извлечение нужной информации
    topic_element = driver.find_element(By.XPATH, "//*[@class='main_topic_title']")
    topic = topic_element.text

    post_wraps = driver.find_elements(By.XPATH, "//*[@class='post_wrap']")

    for post_wrap in post_wraps:
        # Извлекаем имя автора
        try:
            author_element = post_wrap.find_element(
                By.XPATH, ".//*[@class='author vcard']/a | .//*[@class='guest']"
            )
            nick_name = author_element.text
        except NoSuchElementException:
            nick_name = ""
        names.append(nick_name)

        # Извлекаем город
        try:
            city_element = post_wrap.find_element(
                By.XPATH,
                ".//ul[@class='user_fields']//span[@class='ft' and contains(text(), 'Город:')]/following-sibling::span[@class='fc']",
            )
            city = city_element.text
        except NoSuchElementException:
            city = ""
        cities.append(city)

        # Извлекаем комментарий
        try:
            comment_element = post_wrap.find_element(
                By.XPATH, ".//*[@class='post entry-content ']"
            )
            inner_html = comment_element.get_attribute("innerHTML")
            soup = BeautifulSoup(inner_html, "html.parser")
            for unwanted in soup.select(".quote, .citation, .edit"):
                unwanted.decompose()
            comment = soup.get_text(strip=True)
        except NoSuchElementException:
            comment = ""
        text.append(comment)

        # Извлекаем время публикации
        try:
            time_element = post_wrap.find_element(By.XPATH, ".//*[@class='published']")
            time = time_element.text
        except NoSuchElementException:
            time = ""
        date.append(time)

        # Добавляем тему для каждого комментария
        thread.append(topic)


# Листаем страницы комментариев и парсим каждую из них
def iterate_over_comment_pages():
    """Iterate over comment pages and parse them."""
    while True:
        try:
            next_button = WebDriverWait(driver, 2).until(
                ec.element_to_be_clickable(
                    (
                        By.CSS_SELECTOR,
                        "#content > div.topic.hfeed > div.topic_controls.clear > ul.pagination.left > li.next > a",
                    )
                )
            )
            next_button.click()
            time.sleep(2)
            parse_current_page(driver)
        except Exception as e:
            print("Не удалось найти кнопку или произошла ошибка:", e)
            break


def convert_date(date_str: str) -> str:
    """Convert date into standard form '%d-%m-%Y - %H:%M'.

    Args:
        date_str (str): The source date string

    Returns:
        str: Standardized date.
    """
    day, month_russian, year_time = date_str.split(" ", 2)
    month_numeric = months_translation.get(month_russian)
    new_date_str = f"{day}-{month_numeric}-{year_time}"
    date_obj = datetime.strptime(new_date_str, "%d-%m-%Y - %H:%M")
    return date_obj.strftime("%d-%m-%Y - %H:%M")

if __name__ == "__main__":

    # Открываем сайт
    driver = webdriver.Chrome()
    driver.get("http://www.palata6.net/forum/index.php?showtopic=217")
    names = []
    text = []
    thread = []
    date = []
    cities = []

    # Парсим первую страницу
    parse_current_page(driver)
    iterate_over_comment_pages()

    # Для нахождения и остановки на последнем комментарии
    while True:
        try:
            next_button = WebDriverWait(driver, 2).until(
                ec.element_to_be_clickable(
                    (
                        By.CSS_SELECTOR,
                        "#content > div.topic.hfeed > ul.topic_jump.right.clear > li.next > a",
                    )
                )
            )
            next_button.click()
            time.sleep(2)
            parse_current_page(driver)
            iterate_over_comment_pages()
        except Exception as e:
            print("Не удалось найти кнопку или произошла ошибка:", e)
            break

    # Создание DataFrame и сохранение данных в CSV
    df = pd.DataFrame(
        {"name": names, "city": cities, "text": text, "datetime": date, "thread": thread}
    )
    df.to_csv("forum_comments.csv", index=False, encoding="utf-8")
    print("Данные сохранены в файл forum_comments.csv")

    # Начало обработки CSV файла
    filename = "forum_comments.csv"  # Замените на путь к вашему файлу
    df = pd.read_csv(filename)

    # Добавляем столбец src с доменом
    df["src"] = "palata6.net"

    # Добавляем столбец id с уникальными идентификаторами
    df["id"] = [uuid4().hex for _ in range(len(df))]

    # Создаем словарь для хранения уникальных идентификаторов для каждого thread
    topic_to_thread_id = {}
    current_id = 1


    # Добавляем столбец thread_id
    df["thread_id"] = df["topic"].apply(get_thread_id)


    # Удаляем столбец 'thread'
    df = df.drop(columns=["thread"])

    # Словарь для перевода русских месяцев в числовой формат
    months_translation = {
        "Январь": "01",
        "Февраль": "02",
        "Март": "03",
        "Апрель": "04",
        "Май": "05",
        "Июнь": "06",
        "Июль": "07",
        "Август": "08",
        "Сентябрь": "09",
        "Октябрь": "10",
        "Ноябрь": "11",
        "Декабрь": "12",
    }


    # Применяем функцию к столбцу 'datetime'
    df["datetime"] = df["datetime"].apply(convert_date)

    # Сохраните измененный CSV файл
    output_filename = "forum_comments.csv"
    df.to_csv(output_filename, index=False)
