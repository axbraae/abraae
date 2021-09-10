---
title: Goodreads
author: Anne Braae
date: '2021-09-10'
slug: goodreads
categories: []
tags: 
- data wrangling
- exploratory data analysis
- R
subtitle: ''
summary: 'Data cleaning and initial exploratory data analysis of goodreads data.'
authors: []
lastmod: '2021-09-10T16:39:42+01:00'
featured: no
image:
  caption: ''
  focal_point: ''
  preview_only: no
links:
- icon: github
  icon_pack: fab
  name: Code
  url: https://github.com/axbraae/data_cleaning_and_wrangling
projects: []
---




This is the very first homework assignment from my CodeClan course. Included here as I love books! Also because the clean up was more tricky than originally intended.

# Data cleaning

* Initially reading in the data generated a parsing error due to mislabeled quotes
      * This was fixed by including `quote = ""` in the `read_csv()`
* 12 parsing errors remained in the data 
    - Investigating with `problems()` revealed four rows with data that has skipped a column because of an extra comma in the authors column 
    - These commas were manually removed
    - The `.csv` was resaved as `books_edit.csv`
* Missing pages were recoded as missing (`NA`) if page number was zero
* Average ratings were recoded as `NA` if average_rating = 0 and ratings_count > 0 
    (you can't have an average rating of nothing if you have had a rating count)
* Ratings count were recoded as `NA` if average_rating > 0 and ratings_count = 0
    (you can't have a rating count of nothing if there is an average rating score)


```r
library(tidyverse)
library(here)
```


```r
books_clean <- read_csv(here("./content/project/2021-09-10-goodreads/clean_data/books_clean.csv"))
```

```
## 
## ── Column specification ────────────────────────────────────────────────────────
## cols(
##   book_id = col_double(),
##   title = col_character(),
##   authors = col_character(),
##   average_rating = col_double(),
##   isbn = col_character(),
##   isbn13 = col_character(),
##   language_code = col_character(),
##   num_pages = col_double(),
##   ratings_count = col_double(),
##   text_reviews_count = col_double(),
##   publication_date = col_character(),
##   publisher = col_character()
## )
```

## Looking at the authors
  
#### Questions I want to answer:

* How many different authors are there? 
* Who are the top ten most rated authors (based on rating count)?
* Who are the ten authors with the longest books?
      

```r
#count all the unique authors
books_clean %>%
  distinct(authors) %>% 
  count()
```

```
## # A tibble: 1 x 1
##       n
##   <int>
## 1  6643
```
<br />
There are 6643 different authors listed. However, I note some authors may be listed more than once if they are coauthors. 

Let's look at the top ten authors based on total number of reviews.


```r
author_subset <- books_clean %>% 
  select(authors, title, ratings_count, num_pages)

reviewed_top_ten <- author_subset %>% 
  slice_max(ratings_count, n = 10)
reviewed_top_ten
```

```
## # A tibble: 10 x 4
##    authors                title                          ratings_count num_pages
##    <chr>                  <chr>                                  <dbl>     <dbl>
##  1 Stephenie Meyer        Twilight (Twilight  #1)              4597666       501
##  2 J.R.R. Tolkien         The Hobbit  or There and Back…       2530894       366
##  3 J.D. Salinger          The Catcher in the Rye               2457092       277
##  4 Dan Brown              Angels & Demons (Robert Langd…       2418736       736
##  5 J.K. Rowling/Mary Gra… Harry Potter and the Prisoner…       2339585       435
##  6 J.K. Rowling/Mary Gra… Harry Potter and the Chamber …       2293963       341
##  7 J.K. Rowling/Mary Gra… Harry Potter and the Order of…       2153167       870
##  8 J.R.R. Tolkien         The Fellowship of the Ring (T…       2128944       398
##  9 George Orwell/Boris G… Animal Farm                          2111750       122
## 10 J.K. Rowling/Mary Gra… Harry Potter and the Half-Blo…       2095690       652
```
<br />

Looks like Stephenie Meyer with the first Twilight book has received the most ratings on Goodreads! J.K. Rowling also features several times in the top ten most rated authors.

Now I will have a look at the authors with the longest books.


```r
longest_top_ten <- author_subset %>% 
  slice_max(num_pages, n = 10)
longest_top_ten
```

```
## # A tibble: 10 x 4
##    authors                     title                     ratings_count num_pages
##    <chr>                       <chr>                             <dbl>     <dbl>
##  1 Patrick O'Brian             The Complete Aubrey/Matu…          1338      6576
##  2 Winston S. Churchill/John … The Second World War               1493      4736
##  3 Marcel Proust/C.K. Scott M… Remembrance of Things Pa…             6      3400
##  4 J.K. Rowling                Harry Potter Collection …         28242      3342
##  5 Thomas Aquinas              Summa Theologica  5 Vols           2734      3020
##  6 Dennis L. Kasper/Dan L. Lo… Harrison's Principles of…            23      2751
##  7 J.K. Rowling/Mary GrandPré  Harry Potter Boxed Set  …         41428      2690
##  8 Terry Goodkind              The Sword of Truth  Boxe…          4196      2480
##  9 Christina Scull/Wayne G. H… The J.R.R. Tolkien Compa…            45      2264
## 10 Anonymous                   Study Bible: NIV                   4166      2198
```

<br />

Ah, this was a bit of a trick question as the books listed with the highest page numbers are mostly box sets! Interestingly, two J.K. Rowling box sets feature in this list.

<br />

## Looking at the languages

I would like to have a look at the different languages of books in this dataset.

#### Questions I want to answer:

* How many languages are there?
* How many text reviews do books written in English have?
* How many text reviews do books written in non-English have?
* Is this count similar for the overall ratings received for English and non-English books? 
* Who are the top ten publishers of English books?
* Who are the top ten publishers of non-English books?
  

```r
#counting how many languages there are and arranging them alphabetically
books_clean %>% 
  distinct(language_code) %>%
  arrange(language_code)
```

```
## # A tibble: 27 x 1
##    language_code
##    <chr>        
##  1 ale          
##  2 ara          
##  3 en-CA        
##  4 en-GB        
##  5 en-US        
##  6 eng          
##  7 enm          
##  8 fre          
##  9 ger          
## 10 gla          
## # … with 17 more rows
```

<br />

There are 27 different languages in the books dataset. It looks like English is coded four times: eng (which is not a localisation language code), en-US, en-GB and en-CA. (Additional note: enm is middle english, so I will not include this as an English book).

Let's find out what the total text_reviews_count and ratings_count is in all four English groups compared to all other languages.


```r
#subset the dataset to answer the questions on English books
#add a column, english, set to TRUE if the language is English

language_subset <- books_clean %>% 
  select(language_code, publisher, ratings_count, text_reviews_count) %>% 
  mutate(english = case_when(
      language_code %in% c("eng", "en-GB", "en-CA", "en-US") ~ TRUE,
      TRUE ~ FALSE)
        )

#generate a summary table counting the total number of text reviews and ratings counts for English and non-English books

language_subset %>% 
  group_by(english) %>% 
  summarise(
    sum(text_reviews_count),
    sum(ratings_count, na.rm = TRUE))
```

```
## # A tibble: 2 x 3
##   english `sum(text_reviews_count)` `sum(ratings_count, na.rm = TRUE)`
##   <lgl>                       <dbl>                              <dbl>
## 1 FALSE                       31823                            1560813
## 2 TRUE                      5997392                          198017611
```

<br />

Unsurprisingly perhaps, there are far more text reviews for books written in English (eng, en-GB, en-US and en-CA) than in all other languages combined (5997392 compared to 31823). If we put it as a percentage, 99.47% of the text reviews in this Goodreads dataset are written for English books, and only 0.53% of the text reviews are in another language.

This is also seen when looking at the total for all ratings count (198017611 ratings for English books compared to 1560813, or 78.2% of ratings compared to 21.8%).

Let's look at the publishers with the most English titles and the publishers with the most non-English titles.


```r
publisher_subset <- language_subset %>% 
  group_by(publisher) %>% 
  mutate(non_english = !english) %>% 
  summarise(
    tot_eng = sum(english), 
    tot_non_eng = sum(non_english)
  )
publisher_subset
```

```
## # A tibble: 2,293 x 3
##    publisher                       tot_eng tot_non_eng
##    <chr>                             <int>       <int>
##  1 "\"Tarcher\""                         1           0
##  2 "10/18"                               0           2
##  3 "1st Book Library"                    1           0
##  4 "1st World Library"                   1           0
##  5 "A & C Black (Childrens books)"       1           0
##  6 "A Harvest Book/Harcourt  Inc."       1           0
##  7 "A K PETERS"                          1           0
##  8 "AA World Services"                   1           0
##  9 "Abacus"                              6           0
## 10 "Abacus Books"                        1           0
## # … with 2,283 more rows
```

```r
#select publishers with the most books in English
top_eng_publishers <- publisher_subset %>% 
  slice_max(tot_eng, n = 10)
top_eng_publishers
```

```
## # A tibble: 10 x 3
##    publisher        tot_eng tot_non_eng
##    <chr>              <int>       <int>
##  1 Vintage              317           1
##  2 Penguin Books        261           0
##  3 Penguin Classics     183           1
##  4 Mariner Books        149           1
##  5 Ballantine Books     143           1
##  6 HarperCollins        112           0
##  7 Pocket Books         111           0
##  8 Bantam               110           0
##  9 Harper Perennial     110           2
## 10 VIZ Media LLC         88           0
```

```r
#Select publishers with the most books in non-English
top_non_eng_publishers <- publisher_subset %>% 
  slice_max(tot_non_eng, n = 10)
top_non_eng_publishers
```

```
## # A tibble: 11 x 3
##    publisher          tot_eng tot_non_eng
##    <chr>                <int>       <int>
##  1 Debolsillo               1          17
##  2 Gallimard                1          15
##  3 小学館                   0          15
##  4 Pocket                   3          14
##  5 Planeta Publishing       0          13
##  6 Plaza y Janes            1          13
##  7 集英社                   0          12
##  8 J'ai Lu                  0          10
##  9 Ediciones B              0           9
## 10 Glénat                   0           9
## 11 Punto de Lectura         0           9
```

<br />

The top ten publishers with the most English titles are very different from the top ten publishers with the most non-English titles! Interestingly in both groups there are some book titles in other languages. The top ten publishers with non-English titles contains 11 publishers. This is because there are several publishers with the same total count of non-English books.
