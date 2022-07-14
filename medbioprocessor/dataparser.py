import os
import zipfile
import re

from bs4 import BeautifulSoup as bs
import xml.etree.ElementTree as ET

############# CONSTANTS ##################

CONTENT_XML_PATTERN: str = r"content/(\d+).xml"


def extract_article_id(article_doi: str) -> str:
    return article_doi.split(".")[-1].strip()


def get_xml_tags(xml_file: str):
    return {
        elem.tag
        for elem in ET.parse(xml_file).iter()
    }


def parse(archive_file):
    file_path, file_name = os.path.split(archive_file)
    print(f"Parsing {file_name}!")

    content_xml_data = None

    article_data = {}

    print(f"Extracting {file_name} contents!")

    with zipfile.ZipFile(archive_file, "r") as archive:
        for f_name in archive.namelist():
            if re.match(CONTENT_XML_PATTERN, f_name) is not None:
                content_xml_data = archive.read(f_name)

    bs_data = bs(content_xml_data, 'lxml')

    for front in bs_data.find_all("front"):
        for article_meta in front.find_all("article-meta"):
            for _id in article_meta.find("article-id"):
                article_data["Article-ID"] = extract_article_id(_id.text)

        for article_title in front.find_all("article-title"):
            article_data["Article-Title"] = article_title.text

        for abstract in front.find_all("abstract"):
            article_data["Abstract"] = abstract.text

    print(f"Parsed {file_name} Sucessfully!\n")
    return article_data
