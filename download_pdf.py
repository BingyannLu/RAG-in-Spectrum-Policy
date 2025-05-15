import requests
drive_path_fmt = "./{}.pdf"
import time
import chromedriver_autoinstaller as chromedriver
chromedriver.install()
from selenium.webdriver.common.by import By
from selenium import webdriver


drive_path_fmt = "D:/bingbing/ND/SpectrumRAG/19-38/URL.txt"

def download_pdf(url, download_path):
    # Set up Chrome WebDriver
    options = webdriver.ChromeOptions()
    # options.add_argument(r"--user-data-dir=/home/juranding/.config/google-chrome")
    #provide the profile name with which we want to open browser
    # options.add_argument(r'--profile-directory=Profile 2')

    prefs = {'download.default_directory': download_path,
             'profile.default_content_setting_values.automatic_downloads': 2}
    options.add_experimental_option("prefs", prefs)
    driver = webdriver.Chrome()
    # driver = webdriver.Chrome(options=options)
    print(url)

    try:
        # Open the webpage with the PDF link
        driver.get(url)
        time.sleep(5)
        elements = driver.find_elements(By.TAG_NAME, "a")
        element = elements[0]
        time.sleep(5)
        element.click()


        
        # time.sleep(5)
        # download_button = driver.find_element(By.XPATH, XPATH)
        # download_button.click()

        # Wait for the download to complete (you may need to adjust the sleep duration)
        time.sleep(5)
        driver.close()
        print(idx)
    except:
       return
    finally:
        # Close the WebDriver
        driver.quit()

idx = 0
with open(drive_path_fmt, 'r') as url_list:
  for line in url_list:
    idx = idx +1
    if idx <= 50:
       continue
    download_pdf(line, drive_path_fmt)

