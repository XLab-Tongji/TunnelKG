# 颜泽皓-第四周-crawler

这一周的工作内容主要是实现一个百度百科的DFS爬虫。

## 文档树

```
README.md
crawler.py
```

## crawler.py

本文件使用了`requests `库和`BeautifulSoup4`库。前者提供了HTTP的友好操作接口。后者提供了HTML的解析器。

采用了线程池并发处理。结果写入sqlite3数据库。

文档对于每个文档处理如下`CrawlPage() `：

```python
# 网址中的词语
word = unquote(match(r"/item/([^/]*).*", href).group(1))
# 通过链接构造url
url: str = DOMAIN_NAME + href
# 获取html
resp = get(url=url, headers=HEADER)
resp.encoding = "utf-8"
html = resp.text
soup = BS(html, "html.parser")
# 词条标题
entity: str = soup.find(
	"dd", class_="lemmaWgt-lemmaTitle-title").find("h1").string
# 词条正文
paras: ResultSet = soup.find_all("div", class_=compile("para"))


if entity == word: # 如果网址中的词语与词条标题一致
	'''
	写入实体数据库
	'''
else:
	'''
	1. 写入别名数据库
	2. 查看词条集合是否已经被访问
	3. 若没有被访问，则将词条标题写入实体数据库
	4. 将别名写入词条集合
	'''
if (not STOP_BY_DEPTH) or (not depth == MAX_DEPTH): # 未到最大深度
	'''
	1. 提取正文中的每一个超链接，判断是否指向词条
	2. 对于词条超链接，查看词条集合是否一访问过
	3. 未访问的词条链接提交入线程池
	'''
```

现在面临的问题是：在写入时会发生数据库写访问冲突，加上锁之后会导致性能下降。收到的实体词语质量不佳。

