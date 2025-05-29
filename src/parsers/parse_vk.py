"""Script for parsing VK users posts"""

import requests
import pandas as pd

ACCESS_TOKEN = "token"


class VKPostFetcher:
    """Main parser class."""

    def __init__(self, access_token: str = ACCESS_TOKEN):
        self.access_token = access_token
        self.api_version = "5.199"

    def get_vk_wall_posts(
        self, user_id: str = None, domain: str = None, count: int = 100, offset: int = 0
    ):
        """Get user posts by calling the API.

        Args:
            user_id (str, optional): The list with user page ids. Defaults to None.
            domain (str, optional): User address of the pages. Defaults to None.
            count (int, optional): Number of posts to be fetched. Defaults to 100.
            offset (int, optional): The offset for the API download. Defaults to 0.

        Returns:
            tuple(list, int): The list with user wall data and total amount of posts.
        """
        url = "https://api.vk.com/method/wall.get"
        params = {
            "owner_id": user_id,  # если задан id
            "count": count,
            "offset": offset,
            "domain": domain,  # если задан никнейм
            "access_token": self.access_token,
            "filter": "owner",  # только посты владельца
            "v": self.api_version,
        }

        response = requests.get(url, params=params)

        if response.status_code == 200:
            data = response.json()
            if "error" in data:
                print(f"Error: {data['error']['error_msg']}")
                return [], 0
            else:
                total_posts = data["response"]["count"]  # всего постов
                posts = data["response"]["items"]  # полученные посты
                return posts, total_posts
        else:
            print(f"Failed to fetch posts: {response.status_code}")
            return [], 0

    def create_dataframe(
        self, user_ids: list[str] = None, domains: list[str] = None, count: int = 100
    ):
        """Create dataframe from the user data.

        Args:
            user_ids (list[str], optional): The list with user page ids. Defaults to None.
            domains (list[str], optional): User address of the pages. Defaults to None.
            count (int, optional): Number of posts to be fetched. Defaults to 100.

        Returns:
            pd.DataFrame: The complete dataframe with user data.
        """
        posts_list = []
        post_id = 0

        # Для каждого user_id
        if len(user_ids) > 0:
            for user_id in user_ids:
                offset = 0
                while True:
                    posts, total_posts = self.get_vk_wall_posts(
                        user_id=user_id, count=count, offset=offset
                    )
                    if not posts:  # если постов больше нет, выходим из цикла
                        break

                    for post in posts:
                        posts_list.append(
                            {
                                "id": post_id,
                                "text": post.get("text", ""),
                                "source": "vk",
                                "user": user_id,
                            }
                        )
                        post_id += 1

                    offset += count  # увеличиваем смещение
                    if offset >= total_posts:  # если загрузили все посты, выходим
                        break

        # Для каждого domain
        if len(domains) > 0:
            for domain in domains:
                offset = 0
                while True:
                    posts, total_posts = self.get_vk_wall_posts(
                        domain=domain, count=count, offset=offset
                    )
                    if not posts:
                        break

                    for post in posts:
                        posts_list.append(
                            {
                                "id": post_id,
                                "text": post.get("text", ""),
                                "source": "vk",
                                "user": domain,
                            }
                        )
                        post_id += 1

                    offset += count
                    if offset >= total_posts:
                        break

        df = pd.DataFrame(posts_list)
        return df

if __name__ == "__main__":
    user_ids = []
    domains = ["kharlamovgarik"]

    fetcher = VKPostFetcher()
    df = fetcher.create_dataframe(user_ids, domains)

    df.to_csv("vk_parsed.csv", sep="|")
