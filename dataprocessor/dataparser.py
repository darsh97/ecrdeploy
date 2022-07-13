import os
from bs4 import BeautifulSoup as bs

import xml.etree.ElementTree as ET

import pandas as pd
import re

###### CONSTANTS #######


ID_PATTERN = re.compile("_ID(\d+)_")


##### HELPERS #######

def extract_authors(article_content):
    authors_lst = []

    for authors in article_content.findall("AUTHORS"):
        for author in authors.findall("AUTHOR"):
            first_name, last_name = author.findtext("FIRST_NAME") or "", author.findtext("LAST_NAME") or ""
            authors_lst.append(f"{first_name} {last_name}")

    return authors_lst


def extract_id(pdf_url):
    _id = pdf_url.split("=")[-1]
    assert re.match("\d+", _id)
    return _id


def get_xml_tags(xml_file: str):
    return {
        elem.tag
        for elem in ET.parse(xml_file).iter()
    }


def parse(lancet_xml: str):
    article_data = []

    for article in ET.parse(lancet_xml).findall("ARTICLE"):
        data = {}

        paper_title = "".join(article.find("PAPER_TITLE").itertext())
        abstract = "".join(article.find("ABSTRACT_BODY").itertext())
        _id = extract_id(article.findtext("PDF_URL"))

        data["article_id"] = _id
        data["article_title"] = paper_title
        data["abstract"] = bs(abstract, "lxml").text
        data["authors"] = " | ".join(extract_authors(article))

        article_data.append(data)

    return pd.DataFrame(article_data)
