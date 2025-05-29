"""Скрипт парсинаг форумов 2ch.hk и archivach.org"""
from datetime import datetime
from uuid import uuid4
import requests
from bs4 import BeautifulSoup
from pathlib import Path
import csv
import re
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("2ch_arhivach_parser.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

CSV_FIELD_NAMES = ["id", "text_id", "thread_id", "src", "datetime", "name", "replies_to", "content"]


def simplify_and_convert_datetime(raw_datetime):
    """
    Упрощает строку с датой и временем, удаляя день недели, и преобразует её в нужный формат.
    """
    date_part = " ".join(raw_datetime.split()[:1] + raw_datetime.split()[2:])
    dt = datetime.strptime(date_part, "%d/%m/%y %H:%M:%S")
    return dt.strftime('%d-%m-%Y %H-%M-%S')


def extract_thread_info(thread_url, source):
    """
    Извлекает информацию о разделе и thread_id из URL.

    :param thread_url: URL треда.
    :param source: Источник данных ("2ch" или "arhivach").
    :return: tuple(section или "src", thread_id) - строка раздела/источника и ID треда.
    """
    try:
        if source == "2ch":
            # Извлечение раздела
            parts = thread_url.split("/")
            src = "/".join([parts[2], parts[3]])

            # Извлечение thread_id
            thread_id_match = re.search(r"/res/(\d+)\.html", thread_url)
            thread_id = thread_id_match.group(1) if thread_id_match else ""
            return src, thread_id

        elif source == "arhivach":
            # Извлечение thread_id
            thread_id_match = re.search(r"/thread/(\d+)/$", thread_url)
            thread_id = thread_id_match.group(1) if thread_id_match else ""
            return "arhivach", thread_id
    except Exception as e:
        logging.error(f"Ошибка извлечения информации из URL: {e}")
        return "", ""


def extract_post_data(post, source, thread_id, src):
    """
    Извлекает данные из одного поста.

    :param post: HTML блок поста.
    :param source: Источник данных ("2ch" или "arhivach").
    :param thread_id: Идентификатор треда.
    :param src: Источник данных.
    :return: Словарь с данными поста.
    """
    id = uuid4().hex  # Уникальный идентификатор

    # Извлечение text_id
    text_id = post.get("data-num" if source == "2ch" else "postid", "")

    # Извлечение времени и даты
    time_tag = post.find("span", class_="post__time" if source == "2ch" else "post_time")
    post_time = time_tag.text if time_tag else ""
    if post_time != "":
        post_time = simplify_and_convert_datetime(post_time)

    # Извлечение ссылок на ответы
    reply_links = post.find_all("a", class_="post-reply-link")
    replies_to = [link.get("data-num") for link in reply_links if link.get("data-num")]

    # Извлечение текста сообщения
    article = post.find("article", class_="post__message") if source == "2ch" else post.find("div", class_="post_comment_body")
    post_message = ""
    if article:
        for unwanted_tag in article.find_all(class_=["post-reply-link", "unkfunc"]):
            unwanted_tag.decompose()

        for br_tag in article.find_all("br"):
            br_tag.replace_with(" ")

        post_message = article.text.strip()

    return {
        "id": id,
        "text_id": text_id,
        "thread_id": thread_id,
        "src": src,
        "datetime": post_time,
        "name": "anonimous",
        "replies_to": ",".join(replies_to),
        "content": post_message
    }


def parse_thread(thread_url, source, output_dir="data"):
    """
    Парсит указанный тред и извлекает данные из каждого поста.

    :param thread_url: URL треда.
    :param source: Источник данных ("2ch" или "arhivach").
    :param output_dir: Директория для сохранения результата.
    """
    try:
        # Отправляем GET-запрос на страницу
        response = requests.get(thread_url)
        response.raise_for_status()
    except requests.RequestException as e:
        logging.error(f"Ошибка при загрузке страницы {thread_url}: {e}")
        return None

    try:
        # Используем BeautifulSoup для разбора HTML
        soup = BeautifulSoup(response.text, 'html.parser')

        # Определяем блоки постов
        post_classes = ["post post_type_reply", "post post_type_oppost post_withimg"] if source == "2ch" else ["post"]
        posts = soup.find_all("div", class_=post_classes)

        # Извлекаем данные thread_id и источник
        src, thread_id = extract_thread_info(thread_url, source)

        parsed_posts = [extract_post_data(post, source, thread_id, src) for post in posts]

    except Exception as e:
        logging.error(f"Ошибка при обработке страницы {thread_url}: {e}")

        return None

    # Сохраняем результат в CSV
    output_path = save_to_csv(parsed_posts, output_dir, thread_id)
    logging.info(f"Данные успешно сохранены в {output_path}")

    return output_path


def save_to_csv(parsed_posts, output_dir, thread_id):
    """
    Сохраняет данные постов в CSV.

    :param parsed_posts: Список словарей с данными постов.
    :param output_dir: Директория для сохранения.
    :param thread_id: Идентификатор треда.
    :return: Путь до сохраненного файла.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / f"{thread_id}_posts.csv"

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=CSV_FIELD_NAMES
        )
        writer.writeheader()
        writer.writerows(parsed_posts)

    return output_path


# Пример вызова для 2ch
# thread_url_2ch = "https://2ch.hk/psy/res/1629837.html"
# parse_thread(thread_url_2ch, source="2ch", output_dir="data_2ch")

# Пример вызова для arhivach
# thread_url_arhivach = "https://arhivach.xyz/thread/1105569/"
# parse_thread(thread_url_arhivach, source="arhivach", output_dir="data_arhivach")
