import calendar


def parse_article(article, sep=" | "):
    '''
    article: BeautifulSoup object of a single article xml
    '''

    d = {}

    ##authors
    def process_authors(authorList):
        authors = []
        for author in authorList.find_all("author"):
            lastName = None if author.lastname is None else author.lastname.text
            foreName = None if author.forename is None else author.forename.text

            authors.append(', '.join([x for x in [lastName, foreName] if x is not None]))
        return sep.join(authors)

    d['Authors'] = None if article.authorlist is None else process_authors(article.authorlist)

    ##keywords
    def process_keywords(meshList):
        keywords = []
        categories = []
        for meshterm in meshList.find_all("meshheading"):
            keyword = None if meshterm.descriptorname is None else meshterm.descriptorname.text
            keywords.append(keyword)
            for qualifier in meshterm.find_all("qualifiername"):
                category = None if meshterm.qualifiername is None else meshterm.qualifiername.text
                categories.append(category)
        categories = list(set(categories))
        keywords = list(set(keywords))
        return keywords, categories

    if article.meshheadinglist is not None:
        keywords, categories = process_keywords(article.meshheadinglist)
        d['Keywords'] = sep.join(keywords)
        d['Categories'] = sep.join(categories)
    else:
        d['Keywords'] = None
        d['Categories'] = None

    ##dates
    def process_month(month):
        d = dict((v, k) for k, v in enumerate(calendar.month_abbr))
        if month is None:
            return None
        elif d.get(month) is not None:
            return int(d[month])
        else:
            return int(month)

    try:
        d['PubYear'] = article.pubdate.year.text
        d['PubMonth'] = process_month(article.pubdate.month.text)
        d['PubDay'] = None if article.pubdate.day is None else article.pubdate.day.text
    except:
        try:
            d['PubYear'] = None if article.datecompleted.year is None else article.datecompleted.year.text
            d['PubMonth'] = None if article.datecompleted.month is None else process_month(
                article.datecompleted.month.text)
            d['PubDay'] = None if article.datecompleted.day is None else article.datecompleted.day.text
        except:
            d['PubYear'] = None
            d['PubMonth'] = None
            d['PubDay'] = None

    d['PublicationType'] = None if article.publicationtypelist \
                                   is None else sep.join([t.text for t in article.find_all("publicationtype")])

    d['Source'] = "Pubmed"
    d['JournalTitle'] = article.title.text
    d['ArticleTitle'] = article.articletitle.text
    d['Abstract'] = None if article.abstract is None else article.abstract.text
    d['PMID'] = int(article.pmid.text)
    return d
