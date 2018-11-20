from pyvirtualdisplay import Display
from selenium import webdriver
import time
import os
import json
from tqdm import tqdm

display = Display(visible=0, size=(1024, 1024))
display.start()
main_url = "https://openreview.net/group?id=ICLR.cc/2019/Conference#all-submissions"
# some options to run on **server**
# chrome_options = webdriver.ChromeOptions()
# chrome_options.add_argument('--headless')
# chrome_options.add_argument('--no-sandbox')
# chrome_options.add_argument('--disable-dev-shm-usage')
# browser = webdriver.Chrome(chrome_options=chrome_options)

browser = webdriver.Chrome()
browser.get(main_url)
SCROLL_PAUSE_TIME = 2
last_height = browser.execute_script("return document.body.scrollHeight")
while True:
    browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(SCROLL_PAUSE_TIME)
    new_height = browser.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        break
    last_height = new_height
    print("current height is : {}".format(new_height))
all_papers = []

for paper in tqdm(browser.find_elements_by_class_name("note ")):
    paper_urls = paper.find_elements_by_tag_name("a")
    review_url = paper_urls[0].get_attribute("href")
    paper_title = paper_urls[0].get_attribute("text").strip()
    pdf_url = paper_urls[1].get_attribute("href")
    paper_id = paper.get_attribute("data-id")
    all_papers.append([paper_id, paper_title, review_url, pdf_url])

filtered_papers = []
for m in all_papers:
    # there are some non pdfs
    if "pdf" in m[-1]:
        filtered_papers.append(m)

# dump the results to a json file
json.dump(filtered_papers, open("iclr2019.json", "w"), indent=2)

if not os.path.exists("content"):
    os.makedirs("content")

for m in tqdm(filtered_papers):
    browser.get(m[-2])
    # wait until the children comment is loaded
    while "note_with_children" not in browser.page_source:
        time.sleep(1)
    time.sleep(1)
    assert "note_with_children" in browser.page_source, "page not loaded"
    save_path = m[1].replace(" ", "_") + "_" + m[0]
    save_path = ''.join([i for i in save_path if i.isalpha() or i == "_" or i.isnumeric()])
    save_path = "content/" + save_path + ".html"
    with open(save_path, "w") as g:
        g.write(browser.page_source)
