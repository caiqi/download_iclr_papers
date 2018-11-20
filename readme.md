# Download papers from ICLR openreivew 
## Install

```shell
$ pip install pyvirtualdisplay
$ pip install selenium
$ wget https://chromedriver.storage.googleapis.com/2.44/chromedriver_linux64.zip
$ unzip chromedriver_linux64.zip
$ chmod +x chromedriver
$ sudo mv -f chromedriver /usr/local/share/chromedriver
$ sudo ln -s /usr/local/share/chromedriver /usr/local/bin/chromedriver
$ sudo ln -s /usr/local/share/chromedriver /usr/bin/chromedriver
```
## download reviews
```shell
python download.py
```

## download pdfs
```shell
python create_download_pdf.py
chmod +x download_pdf.sh
./download_pdf.sh
```

## Pottential Error 1:
```
unknown error: call function result missing 'value'
```
Solution(https://stackoverflow.com/questions/49162667/unknown-error-call-function-result-missing-value-for-selenium-send-keys-even)
```python
download correct version chromedriver http://chromedriver.chromium.org/downloads
```

## Pottential Error 2:
```
unknown error: DevToolsActivePort file doesn't exist
```
Solution(https://stackoverflow.com/questions/50642308/org-openqa-selenium-webdriverexception-unknown-error-devtoolsactiveport-file-d)
```python
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
driver = webdriver.Chrome(main_url, chrome_options=chrome_options)
```