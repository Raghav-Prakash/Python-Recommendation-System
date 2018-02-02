'''
Building a model-based recommendation system using matrix factorization 
with singular value decomposition (SVD).
'''
import pandas as pd
import numpy as np
import sklearn

from sklearn.decomposition import TruncatedSVD

columns = ['user_id','item_id','rating','timestamp'] # choosing certain columns from the dataset
frame = pd.read_csv('ml-100k/u.data', sep='\t', names=columns)
print(frame.head())
'''
   user_id  item_id  rating  timestamp
0      196      242       3  881250949
1      186      302       3  891717742
2       22      377       1  878887116
3      244       51       2  880606923
4      166      346       1  886397596
'''

'''
item_id is the id for movies for which users rated. 
Displaying the attributes for the movies using the u.item dataset like movie title, release date and
the type of genre for that movie (out of all possible genres).
'''

columns = ['item_id', 'movie title', 'release date', 'video release date', 'IMDb URL', 'unknown',
'Action', 'Adventure', 'Animation', 'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

movies = pd.read_csv('ml-100k/u.item', sep='|', names=columns, encoding='latin-1')
movie_names = movies[['item_id','movie title']]
print(movie_names.head())

'''
   item_id        movie title
0        1   Toy Story (1995)
1        2   GoldenEye (1995)
2        3  Four Rooms (1995)
3        4  Get Shorty (1995)
4        5     Copycat (1995)
'''

#Combining both datasets
combined_movies_data = pd.merge(frame,movie_names,on='item_id')
print(combined_movies_data.head())
'''
   user_id  item_id  rating  timestamp   movie title
0      196      242       3  881250949  Kolya (1996)
1       63      242       3  875747190  Kolya (1996)
2      226      242       5  883888671  Kolya (1996)
3      154      242       3  879138235  Kolya (1996)
4      306      242       5  876503793  Kolya (1996)
'''

#Implementing popularity-based recommendation based on the number of ratings for each movie.
print(combined_movies_data.groupby('item_id')['rating'].count().sort_values(ascending=False).head())
'''
item_id
50     583
258    509
100    508
181    507
294    485
Name: rating, dtype: int64
'''

#Getting the name of the most popular movie (with item_id=50)
Filter = combined_movies_data['item_id']==50
print(combined_movies_data[Filter]['movie title'].unique())
# ['Star Wars (1977)']

#Building a utility matrix - columns: movie title, rows: user_id and cell value: rating
ratings_crosstab = combined_movies_data.pivot_table(values='rating',index='user_id',columns='movie title',fill_value=0)
print(ratings_crosstab.head())
'''
movie title  'Til There Was You (1997)  1-900 (1994)  101 Dalmatians (1996)  \
user_id                                                                       
1                                    0             0                      2   
2                                    0             0                      0   
3                                    0             0                      0   
4                                    0             0                      0   
5                                    0             0                      2 
'''

#Getting the order(shape) of the ratings_crosstab matrix
print(ratings_crosstab.shape)
# (943, 1664)

'''
ratings_crosstab -> 943 rows(user_id), 1664 columns(movie title) -> truncate -> truncated_matrix
-> 943 rows(user_id), 12 columns(latent variables about movies)

ratings_crosstab -> transpose -> transpose_crosstab -> 1664 rows(movie title), 943 columns(user_id)
-> truncate -> truncated_matrix -> 1664 rows(movie title), 12 columns(generalized user tastes)

Since we're recommending movies to new users, we take the transpose of the ratings_crosstab matrix,
truncate using SVD (Singular Value Decomposition), then get 12 generalized user tastes for all the
1664 movies(rows) and model the recommendation.
'''

#Transposing the ratings_crosstab matrix
X = ratings_crosstab.values.T
print(X.shape)
# (1664, 943)

'''
Decomposing the X matrix using TruncatedSVD. 

Using TruncatedSVD, we get 12 components. Then we transform and fit X into the SVD model for it
to reduce X to 12 dimensions.
'''
SVD = TruncatedSVD(n_components=12, random_state=17)
resultant_matrix = SVD.fit_transform(X)
print(resultant_matrix.shape)
# (1664, 12)

'''
For a given movie of interest (Star Wars (1977)), we're correlating this with all the other movies
using the PearsonR Correlation Coefficients. We pick the movie having the highest correlation
with our movie of interest and recommend this movie to a user.

Using the resultant_matrix, we get a 1664x1664 correlation matrix. We pick the column "Star Wars (1977)"
and see its correlation with the other movies, using the generated 1664x1 matrix.
'''
corr_mat = np.corrcoef(resultant_matrix)
print(corr_mat.shape)
# (1664, 1664)

# Isolating "Star Wars (1977)" to get the movie of interest
movie_names = ratings_crosstab.columns
movies_list = list(movie_names)

star_wars = movies_list.index("Star Wars (1977)")
print(star_wars)
# 1398

# Getting the 1664x1 matrix
corr_star_wars = corr_mat[star_wars]
print(corr_star_wars.shape)
# (1664,)

'''
Picking the movie names having correlation coefficients between 0.9 and 1.0 ,i.e. picking movies
that are very closely correlated with Star Wars.
'''
print(list(movie_names[(corr_star_wars > 0.9) & (corr_star_wars < 1.0)]))
'''
['Die Hard (1988)', 'Empire Strikes Back, The (1980)', 'Fugitive, The (1993)', 
'Raiders of the Lost Ark (1981)', 'Return of the Jedi (1983)', 
'Star Wars (1977)', 'Terminator 2: Judgment Day (1991)', 'Terminator, The (1984)', 'Toy Story (1995)']
'''

# Listing even better ones. (Now the coefficients are between 0.95 and 1.0)
print(list(movie_names[(corr_star_wars > 0.95) & (corr_star_wars < 1.0)]))
# ['Return of the Jedi (1983)', 'Star Wars (1977)']

'''
So those who've seen "Star Wars (1977)" will also like to watch "Return of the Jedi (1983)".
This was predicted using model-based recommendation.
'''
