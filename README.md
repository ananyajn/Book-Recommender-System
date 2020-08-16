# Book-Recommender-System

In this project, I aim to build a book recommender system using data from Goodreads. The data is used to create a sparse user-book ratings matrix, which is then filled using a matrix completion algorithm. The system runs two different matrix completion algorithms and compares their performance.

## Data Files

The original data obtained from Goodreads is very vast, so I have uploaded subsets of the data for this repo.

There are four data files used in this project:
1. some_goodreads_interactions.csv: Contains the ratings assigned by 1000 users to a large number of books.
2. some_goodreads_books.csv: Contains meta-data about a select number of books.
3. some_book_id_map.csv: Maps the book IDs in the interactions file to the respective book IDs in the book meta-data file.
4. goodreads_book_authors.csv: Contains information about the authors of the books.

### Important Caveats

As only a subset of the data could be loaded to this repo, several functions will not run if different parameters than the ones in ```main.py\``` are passed. These are mentioned in the code. If you change these parameters, the code will throw errors. For the full sized files, contact me at ananyajegannathan@gmail.com.

## Instructions

The *fancyimpute* library needs to be installed to run this code. This can be installed by running

```pip install fancyimpute```

Alternately, all the libraries needed to run this project can be found in requirements.txt. They can be installed at once by running

```pip install -r requirements.txt```

Once the requirements have been installed, the project can be executed by downloading the repository and running ```\code\main.py```.
There are different functions in ```main.py ``` for running the recommender system and for algorithm performance evaluation. If you want to run one at a time, comment out the others.

## Details and Results

For further insight into the theory behind the project and the results and conclusions, check out the Project Report.

## Credits

1. The datasets can be obtained [here](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home).
2. More information on the *fancyimpute* library can be obtained [here](https://github.com/iskandr/fancyimpute)





