import re
from bs4 import BeautifulSoup
import requests
import pandas as pd

import pyspark


def download_file(url):
    local_filename = url.split('/')[-1]
    # NOTE the stream=True parameter below
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                # If you have chunk encoded response uncomment if
                # and set chunk_size parameter to None.
                #if chunk:
                f.write(chunk)
    return local_filename

# test single month-year
url = 'https://opendap.larc.nasa.gov/opendap/CALIPSO/IIR_L1-Standard-V2-00/2020/09/contents.html'
r = requests.get(url)
data_files = re.findall(r'https.*hdf.html', requests.get(url).text)


def check_file(pdf):
    url = pdf['url'].tolist()[0]
    file_url = re.findall('(.*).html', url)[0]
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')

    # coordinates based on CALIPSO area selector
    top = 26.405982971191
    left = -82.882919311523
    right = -79.850692749023
    bottom = 24.208717346191

    raw_minlat = re.findall(r"(?<=MINLAT)(.*)(?=MINLON)", soup.text, re.DOTALL)
    raw_minlong = re.findall(r"(?<=MINLON)(.*)(?=MAXLAT)", soup.text, re.DOTALL)
    raw_maxlat = re.findall(r"(?<=MAXLAT)(.*)(?=MAXLON)", soup.text, re.DOTALL)
    raw_maxlong = re.findall(r"(?<=MAXLON)(.*)(?=GRING)", soup.text, re.DOTALL)

    minlat = float(re.findall(r"\d*\.\d*", raw_minlat[0])[0])
    minlong = float(re.findall(r"\d*\.\d*", raw_minlong[0])[0])
    maxlat= float(re.findall(r"\d*\.\d*", raw_maxlat[0])[0])
    maxlong = float(re.findall(r"\d*\.\d*", raw_maxlong[0])[0])
    if (minlong <= left <= right <= maxlong) & (minlat <= bottom <= top <= maxlat):
        return pd.DataFrame({'url': [url], 'CHECK': ['VALID']})
    else:
        return pd.DataFrame({'url': [url], 'CHECK': ['NOT VALID']})

spark = pyspark.sql.SparkSession.builder\
    .master("local")\
    .appName("file checker")\
    .config("spark.some.config.option", "some-value").getOrCreate()

df = spark.createDataFrame(pd.DataFrame(data_files, columns=['url']))
results_df = df.groupby('url').applyInPandas(check_file, schema='url string, CHECK string')
results_df.show()
