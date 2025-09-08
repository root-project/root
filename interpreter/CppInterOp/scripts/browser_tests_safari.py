import sys
import time
from selenium import webdriver

driver = webdriver.Safari()
driver.get("http://localhost:6931/" + sys.argv[1])

time.sleep(60)
driver.quit()
