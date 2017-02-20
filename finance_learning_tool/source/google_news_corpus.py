from news_corpus_builder import NewsCorpusGenerator

# Location to save generated corpus
corpus_dir = '/home/daisyz/Dropbox/finance_prediction/data'

# Save results to sqlite or  files per article
ex = NewsCorpusGenerator(corpus_dir,'file')

# Retrieve 50 links related to the search term dogs and assign a category of Pet to the retrieved links
date = '2-1-17'
tickers = ['AAPL', 'NFLX', 'FB', 'TSLA', 'ORCL']
folder = ['Apple Inc '+date, 'Netflix '+date, 'Facebook '+date, 'Tesla '+date, 'Oracle '+date]

for i, company in enumerate(tickers):
	links = ex.google_news_search(company,folder[i],100)

	# Generate and save corpus
	ex.generate_corpus(links)
