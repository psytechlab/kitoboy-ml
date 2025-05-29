"""Script for parsing forum psyche.guru."""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import uuid
from datetime import datetime

# Базовый URL форума
BASE_URL = "https://psyche.guru/forum/forum/110-camoubijstvo-suicid/page/"


def get_links_from_page(page_number: str):
    """Collect links from one page.

    Args:
        page_number (str): Page number.

    Returns:
        list[str]: Collected links.
    """
    url = f"{BASE_URL}{page_number}/"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    links = []
    # Найти все элементы с классом "ipsDataItem ipsDataItem_responsivePhoto"
    items = soup.find_all("li", class_="ipsDataItem ipsDataItem_responsivePhoto")

    for item in items:
        # Найти элемент h4 с классом "ipsDataItem_title"
        title = item.find("h4", class_="ipsDataItem_title ipsContained_container")
        if title:
            # Найти ссылку внутри элемента span
            link = title.find("span", class_="ipsType_break ipsContained")
            if link:
                a_tag = link.find("a")
                if a_tag and "href" in a_tag.attrs:
                    links.append(a_tag["href"])

    return links


# Главная функция для сбора ссылок с нескольких страниц
def collect_all_links(start_page: int, end_page: int):
    """Collect links from pages.

    Args:
        start_page (int): Start page to scan.
        end_page (int): End page.

    Returns:
        list[str]: Collected links
    """
    all_links = []
    for page_number in range(start_page, end_page + 1):
        links = get_links_from_page(page_number)
        all_links.extend(links)
        print(f"Собрано {len(links)} ссылок с страницы {page_number}")

    return all_links


# Функция для получения названия с указанной ссылки
def get_title_from_link(link: str):
    """Get title from provided link.

    Args:
        link (str): Link to get title.

    Returns:
        str | None: Title or None if not found.
    """
    response = requests.get(link)
    soup = BeautifulSoup(response.text, "html.parser")

    # Найти элемент span с классом "ipsType_break ipsContained"
    title_span = soup.find("span", class_="ipsType_break ipsContained")
    if title_span:
        return title_span.get_text(strip=True)
    return None


# Функция для получения всех страниц с комментариями для данного топика
def get_all_comment_pages(topic_url: str):
    """Collect pages with comments from topic.

    Args:
        topic_url (str): Topic to collect comments.

    Returns:
        list[str]: Collected pages.
    """
    response = requests.get(topic_url)
    soup = BeautifulSoup(response.text, "html.parser")

    # Найти все ссылки на дополнительные страницы комментариев
    pagination_links = soup.find_all("li", class_="ipsPagination_page")

    # Извлечь уникальные номера страниц
    page_numbers = set()
    for link in pagination_links:
        text_num = link.text
        page_numbers.add(text_num)

    return sorted(page_numbers)


def get_data(link: str):
    """Get data from the page.

    Args:
        link (str): Link to page to data collect.

    Returns:
        list[dict[str:str]]: List of collected data in dictionary.
    """
    response = requests.get(link)
    soup = BeautifulSoup(response.text, "html.parser")

    posts = []

    # Найти все блоки постов
    post_blocks = soup.find_all(
        "article",
        class_="cPost ipsBox ipsResponsive_pull ipsComment ipsComment_parent ipsClearfix ipsClear ipsColumns ipsColumns_noSpacing ipsColumns_collapsePhone",
    )

    # print(post_blocks)
    for post in post_blocks:
        # Найти комментарий в блоке
        comment_text = post.find("div", class_="cPost_contentWrap").find(
            "div",
            class_="ipsType_normal ipsType_richText ipsPadding_bottom ipsContained",
        )
        for blockquote in comment_text.find_all(
            "blockquote"
        ):  # удаление цитаты, на которую отвечает автор
            blockquote.decompose()

        paragraphs = comment_text.find_all("p")
        comment = " ".join(paragraph.get_text(strip=True) for paragraph in paragraphs)

        comment_meta = (
            post.find(
                "div",
                class_="ipsComment_meta ipsType_light ipsFlex ipsFlex-ai:center ipsFlex-jc:between ipsFlex-fd:row-reverse",
            )
            .find("div", class_="ipsType_reset ipsResponsive_hidePhone")
            .find("time")
        )

        date_time = comment_meta["title"] if comment_meta else None
        date_obj = datetime.strptime(date_time, "%d.%m.%Y %H:%M")
        formatted_date_str = date_obj.strftime("%d-%m-%Y %H:%M:%S")

        # Найти имя автора в блоке
        author_block = post.find(
            "aside",
            class_="ipsComment_author cAuthorPane ipsColumn ipsColumn_medium ipsResponsive_hidePhone",
        )
        author_name = author_block.find("strong") if author_block else None
        author = author_name.get_text(strip=True) if author_name else None

        if comment and author:
            posts.append(
                {"comment": comment, "author": author, "datetime": formatted_date_str}
            )

    return posts

if __name__ == "__main__":
    # Собираем ссылки с 1 по 6 страницы
    collected_links = collect_all_links(1, 6)
    print("Все ссылки собраны:", collected_links)

    data = []
    for i, link in enumerate(collected_links):
        title = get_title_from_link(link)
        comments_and_authors = []

        # Получаем все страницы с комментариями
        page_numbers = get_all_comment_pages(link)

        # Если нет дополнительных страниц, обрабатываем первую страницу
        if not page_numbers:
            comments_and_authors.extend(get_data(link))
        else:
            # Иначе обрабатываем все страницы с комментариями
            for page_number in page_numbers:
                page_url = f"{link}/page/{page_number}/"
                comments_and_authors.extend(get_data(page_url))

        if title:
            for post in comments_and_authors:
                data.append(
                    {
                        "id": str(uuid.uuid4()),
                        "thread_id": i + 1,
                        "text": post["comment"],
                        "src": "forum_psyche.guru",
                        "datetime": post["datetime"],
                        "name": post["author"],
                        "age": None,
                        "city": None
                        # 'title': title
                    }
                )

    df = pd.DataFrame(data)
    df.to_csv("psych_forum_comments.csv", index=False)
