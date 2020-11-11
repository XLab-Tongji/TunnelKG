from requests import get
from regex import compile, match
from bs4 import BeautifulSoup as BS
from bs4.element import ResultSet
from concurrent.futures import Future, ThreadPoolExecutor as TPE
from queue import Queue
from sqlite3.dbapi2 import Connection, Cursor, Error, IntegrityError, connect
from threading import Lock
from urllib.parse import quote, unquote

EntityTuple = tuple[str, str, str, str, float]

STOP_BY_DEPTH = True
MAX_DEPTH = 3
STOP_BY_CONTENT = False
THREAD_POOL_SIZE = 18
HEADER: dict[str, str] = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/67.0.3396.99 Safari/537.36"
}
INIT_WORDS: list[str] = ["盾构"]
DOMAIN_NAME = "https://baike.baidu.com"
DB_PATH = "entities.db"

alias_db_queue: Queue[tuple[str, str]] = Queue()
crwal_task_queue: Queue[tuple[str, int, str]] = Queue()
db_write_lock = Lock()
entity_db_queue: Queue[EntityTuple] = Queue()
key_hash_set: set[int] = set()
key_hash_set_lock = Lock()
stdout_lock = Lock()
thrd_pool: TPE = None


def ContentStopper() -> bool:
    pass

def WriteEntityDB() -> None:
    db_conn: Connection = connect(DB_PATH)
    db_cursor: Cursor = db_conn.cursor()
    (entity, href, html, parent, tf_idf) = entity_db_queue.get()

    try:
        # with db_write_lock:
        db_cursor.execute("INSERT INTO Entities "
                            "(entity, url, html, parent, tf_idf) "
                            "VALUES(?, ?, ?, ?, ?)",
                            (entity, DOMAIN_NAME +
                            href, html, parent, tf_idf)
                            )

        with stdout_lock:
            print("[WRITE]" + entity)
    except IntegrityError:
        with stdout_lock:
            print("[ADDED]" + entity)
    except Error as e:
        with stdout_lock:
            print("[ERROR]" + entity)
            print(e)
    finally:
        db_conn.commit()
        db_conn.close()
        entity_db_queue.task_done()


def WriteAliasDB() -> None:
    db_conn: Connection = connect(DB_PATH)
    db_cursor: Cursor = db_conn.cursor()
    (entity, alias) = alias_db_queue.get()

    try:
        with db_write_lock:
            db_cursor.execute("INSERT INTO Aliases "
                              "(entity, alias) "
                              "VALUES(?, ?)",
                              (entity, alias)
                              )
        with stdout_lock:
            print("[WRITE]" + alias)
    except Error as e:
        with stdout_lock:
            print("[ERROR]" + alias)
            print(e)
    finally:
        db_conn.commit()
        db_conn.close()
        alias_db_queue.task_done()



def CrawlPage() -> None:
    global thrd_pool

    (href, depth, parent) = crwal_task_queue.get()

    word = unquote(match(r"/item/([^/]*).*", href).group(1))
    visited: bool = False
    url: str = DOMAIN_NAME + href
    resp = get(url=url, headers=HEADER)
    resp.encoding = "utf-8"
    html = resp.text
    soup = BS(html, "html.parser")
    entity: str = soup.find(
        "dd", class_="lemmaWgt-lemmaTitle-title").find("h1").string
    paras: ResultSet = soup.find_all("div", class_=compile("para"))

    if (not STOP_BY_CONTENT) or (not ContentStopper(soup)):
        if entity == word:
            with stdout_lock:
                print("[FIND]" + entity)
            entity_db_queue.put((entity, href, html, parent, 0.0))
            thrd_pool.submit(WriteEntityDB)
        else:
            with stdout_lock:
                print("[ALIAS]" + word)
            alias_db_queue.put((entity, word))
            thrd_pool.submit(WriteAliasDB)
            hash_entity = hash(entity)
            with key_hash_set_lock:
                visited = (hash_entity in key_hash_set)
                if not visited:
                    key_hash_set.add(hash_entity)
            if not visited:
                entity_db_queue.put((entity, href, html, parent, 0.0))
                thrd_pool.submit(WriteEntityDB)

        if (not STOP_BY_DEPTH) or (not depth == MAX_DEPTH):
            for para in paras:
                anchors: ResultSet = para.find_all(
                    "a", href=compile(r"/item/.*"))
                for anchor in anchors:
                    href: str = anchor.get("href")
                    key = match(r"/item/([^/]*).*", href).group(1)
                    hash_key = hash(key)
                    with key_hash_set_lock:
                        visited = (hash_key in key_hash_set)
                        if not visited:
                            key_hash_set.add(hash_key)
                    if not visited:
                        crwal_task_queue.put((href, depth + 1, entity))
                        thrd_pool.submit(CrawlPage)
    crwal_task_queue.task_done()


if __name__ == "__main__":
    thrd_pool = TPE(max_workers=THREAD_POOL_SIZE)

    db_write_lock.acquire()
    for words in INIT_WORDS:
        key_hash_set.add(hash(quote(words)))
    for words in INIT_WORDS:
        href: str = "/item/" + quote(words)
        crwal_task_queue.put((href, 0, ""))
        thrd_pool.submit(CrawlPage)
    crwal_task_queue.join()
    thrd_pool.shutdown()
    entity_db_queue.join()
    alias_db_queue.join()
    print("[FINISH]")
