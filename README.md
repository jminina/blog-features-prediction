
# Predict blog features using Doc2vec and SVC

## THE TASK

The task was to use the data to create a model (or models) that is able to determine based on the text, who wrote it and what is the topic. In this case who means:
What is the gender of the blogger;
What is the age of the blogger;
What is the zodiac sign of the blogger?
It should also predict the industry (topic) of the blog.


## How to use

1) Install requirements by running `pip install -r requirements.txt` (best with configurated virtual environment).
2) Run jupyter with command `jupyter lab` and open **blog_predict.ipynb**.
2) Download the data and change df path.
3) Run all cells in notebook.
4) To classify your own data, see the **Test on real example** section (simply change value of `exaple` to your own text and run next cell)

## THE DATA

### Description
The Blog Authorship Corpus consists of the collected posts of 19,320 bloggers gathered from blogger.com in August 2004. The corpus incorporates a total of 681,288 posts and over 140 million words - or approximately 35 posts and 7250 words per person. Each blogpost is presented as a separate row, with a blogger id# and the blogger’s self-provided gender, age, industry and astrological sign. (All are labeled for gender and age but for many, industry and/or sign is marked as unknown.)

Each blog in the corpus includes at least 200 occurrences of common English words. All formatting has been stripped with two exceptions. Individual posts within a single blogger are separated by the date of the following post and links within a post are denoted by the label urllink.

### Download

You can download the data [here](https://drive.google.com/file/d/1pzQOxzzqPBBzdTwRYxR8IJ1KEWgpnFw_/view). It is a .zip file that you will need to unarchive. Inside of the unzipped folder you will find the entire dataset in .csv format.

Each row in the datsaet consists of the relevant data to perform the task. The available columns are:

id, gender, age, topic, sign, date and text.

Inside of the text columns, there is the text data of the blogs written by the bloggers (each of them have a unique ID).

## Performance

### Topic classification
                            precision    recall  f1-score   support

                 Accounting       0.02      0.04      0.03        25
                Advertising       0.00      0.00      0.00        30
                Agriculture       0.00      0.00      0.00         5
               Architecture       0.00      0.00      0.00        14
                       Arts       0.11      0.22      0.14       113
                 Automotive       0.00      0.00      0.00        12
                    Banking       0.00      0.00      0.00        24
                    Biotech       0.00      0.00      0.00        13
           BusinessServices       0.00      0.00      0.00        41
                  Chemicals       0.00      0.00      0.00        14
       Communications-Media       0.11      0.41      0.18        86
               Construction       0.00      0.00      0.00        10
                 Consulting       0.03      0.04      0.03        28
                  Education       0.33      0.17      0.22       197
                Engineering       0.11      0.14      0.12        50
                Environment       0.00      0.00      0.00         4
                    Fashion       0.00      0.00      0.00        21
                 Government       0.07      0.12      0.09        41
             HumanResources       0.00      0.00      0.00        13
                   Internet       0.16      0.16      0.16        68
          InvestmentBanking       0.00      0.00      0.00         9
                        Law       0.18      0.15      0.16        48
    LawEnforcement-Security       0.00      0.00      0.00        11
              Manufacturing       0.00      0.00      0.00        15
                   Maritime       0.00      0.00      0.00         2
                  Marketing       0.05      0.09      0.06        35
                   Military       0.12      0.27      0.17        26
          Museums-Libraries       0.00      0.00      0.00         7
                 Non-Profit       0.08      0.09      0.08        70
                 Publishing       0.12      0.32      0.17        25
                 RealEstate       0.00      0.00      0.00         8
                   Religion       0.17      0.41      0.24        29
                    Science       0.07      0.06      0.06        33
          Sports-Recreation       0.00      0.00      0.00        18
                    Student       0.78      0.56      0.65       935
                 Technology       0.32      0.30      0.31       161
         Telecommunications       0.00      0.00      0.00        26
                    Tourism       0.00      0.00      0.00        17
             Transportation       0.00      0.00      0.00        13

                   accuracy                           0.32      2297
                  macro avg       0.07      0.09      0.07      2297
               weighted avg       0.40      0.32      0.34      2297

Topic prediction is not really accurate (weighted precision is 0.4, accuracy is 0.32). It can be improved by aggregating simmilar classes into one. Also, some blogs are out of given topics.

 ### Age classification
 
                    precision    recall  f1-score   support

     (0.0, 20.0]       0.92      0.88      0.90      1008
    (20.0, 30.0]       0.79      0.73      0.76       954
    (30.0, 40.0]       0.39      0.54      0.45       245
     (40.0, inf]       0.20      0.26      0.22        90

        accuracy                           0.76      2297
       macro avg       0.58      0.60      0.58      2297
    weighted avg       0.78      0.76      0.77      2297
    
    
 ### Gender classification
 
                     precision    recall  f1-score   support

          female       0.79      0.82      0.81      1065
            male       0.84      0.81      0.83      1232

        accuracy                           0.82      2297
       macro avg       0.82      0.82      0.82      2297
    weighted avg       0.82      0.82      0.82      2297
    
    
### Sign classification

                    precision    recall  f1-score   support

        Aquarius       0.08      0.08      0.08       196
           Aries       0.06      0.04      0.05       201
          Cancer       0.09      0.19      0.12       210
       Capricorn       0.04      0.04      0.04       162
          Gemini       0.10      0.11      0.11       166
             Leo       0.08      0.12      0.09       157
           Libra       0.05      0.07      0.06       193
          Pisces       0.10      0.08      0.09       177
     Sagittarius       0.09      0.05      0.06       182
         Scorpio       0.05      0.03      0.04       207
          Taurus       0.11      0.04      0.06       206
           Virgo       0.11      0.08      0.09       240

        accuracy                           0.08      2297
       macro avg       0.08      0.08      0.08      2297
    weighted avg       0.08      0.08      0.08      2297

I didn't expect good results  here.

## Example on unseen data from real python blog

Example text:
Source of example https://realpython.com/python-enumerate/
``` 
In Python, a for loop is usually written as a loop over an iterable object. 
This means you don’t need a counting variable to access items in the iterable. Sometimes,
though, you do want to have a variable that changes on each loop iteration.
Rather than creating and incrementing a variable yourself, you can use Python’s enumerate()to get a counter and the value from the iterable at the same time!

```
**Prediction:** *Topic: Technology; Age: (20.0, 30.0]; Gender: male; Sign: Cancer*
In original post, you can see author. He is young a man, the algorithgm result looks proper. Unfortunately I can't find info about his zodiac sign and actual age.



## How to improve?

There are some ways to improve my classification. If you have good enough hardware, you can try some hype methods like BERT. Also, all models can be improved with hyperparameters tuning.


