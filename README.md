# Finance Prediction [status = "in development"]
----
This project aims to build of an automated, general-purpose, and real-time pipeline to predict financial indicators using machine learning on several news sources.

## Contributors
---
Lead: Ben Regner 
Team: Andrea Massari, Daisy Zheng, Saina Lajevardi

## Usage
---
A thorough walkthrough can be found in the notebooks/ directory.

An automated script, "try_me_script.py", can be run from command line from this home directory with 
```
python3 try_me_script.py
```
N.B. several version/folder-structure requirements are needed, we will improve the portability of this code in the near future.
Also please be careful as at the first time it is run, the program will try and download quite a few files from the internet and therefore will take a good while to complete (~30 mins if run locally).

## Road map
---
- [x] Obtain data
- [x] Try to predict using simple models
- [x] Increase automation and modularity
- [x] Use word2vec
- [x] Use clean coding standards
- [x] Make a command-line script to showcase complete pipeline and results.
- [x] Start adding unit tests.
- [ ] Visualize results on a webpage
- [ ] Make model dynamic (real-time updates)

## Future work
---
- Bring in other data sources
- Try other models
- Try to predict different indicators/quantities


