import urllib.request
import time
from selenium.webdriver import Chrome
from selenium import webdriver

year_range = list(range(2014, 2016, 1))
leap_years = [2004, 2008, 2012, 2016, 2020]

driver = webdriver.Chrome(executable_path='/Users/josh/chromedriver')

for year in year_range:
    start = 1
    end = 8
    max_day = 366 if year in leap_years else 365
    while end != max_day:
        if end > max_day:
            start = 361
            end = max_day
        span_start = str(start).zfill(3)
        span_end = str(end).zfill(3)
        print("Downloading file A{}{}{}{}".format(year, span_start,
                                                  year, span_end))
        url = 'http://oceandata.sci.gsfc.nasa.gov/cgi/getfile/A{}{}{}{}.L3m_8D_CHL_chlor_a_4km.nc'.format(year,
                                                                                                          span_start,
                                                                                                          year,
                                                                                                          span_end)
        driver.get(url)

        start += 8
        end += 8
        time.sleep(1)
        if end > max_day:
            start = 361
            end = max_day
