"""Script for parsing forum pobedish.ru"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import uuid
from datetime import datetime
import re

# Базовый URL форума
base_url = "https://pobedish.ru/forum/viewforum.php?f=12&start="


def clean_text(text: str):
    """Clear text from irrelevant symbols.

    Args:
        text (str): The source text.

    Returns:
        str: Cleaned text.
    """
    text = re.sub(r"\r", "", text)
    text = re.sub(r"[\n\t]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def collect_all_links(start_page: int = 0, end_page: int = 1650):
    """Collect of links available in the forum.

    Args:
        start_page (int, optional): The start page to collect. Defaults to 0.
        end_page (int, optional): End page.. Defaults to 1650.

    Returns:
        list[str]: All collected links with threads.
    """
    all_links = []
    for page_number in range(start_page, end_page + 1, 25):
        links = get_links_from_page(page_number)
        all_links.extend(links)
        print(f"Собрано {len(links)} ссылок с страницы {page_number}")

    return all_links


# Функция для получения ссылок с одной страницы
def get_links_from_page(page_number: int):
    """Get links from one page.

    Args:
        page_number (int): Page number.

    Returns:
        list[str]: Found links.
    """
    url = f"{base_url}{page_number}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    links = []
    items = soup.find_all("li", class_="row bg2")

    for item in items:

        title = item.find("div", class_="list-inner")
        a_tag = title.find("a")
        if a_tag and "href" in a_tag.attrs:
            link = "https://pobedish.ru/forum/" + a_tag["href"].split("/")[1]
            links.append(link)

    return links


def get_all_comment_pages(topic_url: str):
    """Collect comment pages.

    Args:
        topic_url (str): Topic url with comment pages.

    Returns:
        list[str]: List with all comments pages in topic.
    """
    response = requests.get(topic_url)
    soup = BeautifulSoup(response.text, "html.parser")
    page_numbers = set()

    # Найти все ссылки на дополнительные страницы комментариев
    pagination_links = soup.find_all("div", class_="pagination")
    for pages in pagination_links:
        pages_links = pages.find_all("a", class_="button")
        for page_link in pages_links:
            number = page_link.text
            if number.isdigit():
                new_number = (int(number) - 1) * 10
                page_numbers.add(str(new_number))

    if len(page_numbers) > 0:
        page_numbers.add("0")

    return sorted(page_numbers)


def get_data(link: str):
    """Get data from the comment.

    Args:
        link (str): The link to be parsed.

    Returns:
        list[dict[str:str]]: Collected data
    """
    response = requests.get(link)
    soup = BeautifulSoup(response.text, "html.parser")

    posts = []

    post_blocks = soup.find_all("div", class_="postbody")
    for post in post_blocks:
        content = post.find("div", class_="content")
        for blockquote in content.find_all(
            "blockquote"
        ):  # удаление цитаты, на которую отвечает автор
            blockquote.decompose()
        post_context = clean_text(content.text)

        meta = post.find("p", class_="author")

        author = meta.find("a", class_="username-coloured").text

        time_tag = meta.find("time")
        datetime_value = time_tag["datetime"]
        dt = datetime.fromisoformat(datetime_value)
        formatted_date = dt.strftime("%d-%m-%Y %H-%M-%S")

        if post_context and author:
            posts.append(
                {"comment": post_context, "author": author, "datetime": formatted_date}
            )

    return posts

if __name__ == "__main__":

    collected_links = collect_all_links()
    print("Все ссылки собраны:", collected_links)

    data = []
    for i, link in enumerate(collected_links):
        comments_and_authors = []
        print(f"{i+1}/{len(collected_links)}")

        # Получаем все страницы с комментариями
        page_numbers = get_all_comment_pages(link)
        if not page_numbers:
            comments_and_authors.extend(get_data(link))
        else:
            # Иначе обрабатываем все страницы с комментариями
            for page_number in page_numbers:
                page_url = f"{link}&start={page_number}"
                comments_and_authors.extend(get_data(page_url))

        for post in comments_and_authors:
            data.append(
                {
                    "id": str(uuid.uuid4()),
                    "thread_id": i + 1,
                    "text": post["comment"],
                    "src": "pobedish.ru_forum_viewforum.php?f=12",
                    "datetime": post["datetime"],
                    "name": post["author"],
                    "age": None,
                    "city": None,
                }
            )

    df = pd.DataFrame(data)
    df.to_csv("pobedish_ru_forum_12.csv", index=False)
