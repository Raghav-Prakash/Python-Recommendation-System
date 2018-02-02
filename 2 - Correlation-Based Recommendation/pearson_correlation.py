import pandas as pd
import numpy as np

frame = pd.read_csv('rating_final.csv')
cuisine = pd.read_csv('chefmozcuisine.csv')
geodata = pd.read_csv('geoplaces2.csv',encoding='latin-1')

#displaying a subset of the above files
print(frame.head())
'''
 userID  placeID  rating  food_rating  service_rating
0  U1077   135085       2            2               2
1  U1077   135038       2            2               1
2  U1077   132825       2            2               2
3  U1077   135060       1            2               2
4  U1068   135104       1            1               2
'''

#displaying only the columns 'placeID' and 'name' from the 'geoplaces2.csv' file (out of 21 columns)
places = geodata[['placeID','name']]
print(places.head())
'''
  placeID                            name
0   134999                 Kiku Cuernavaca
1   132825                 puesto de tacos
2   135106      El Rincón de San Francisco
3   132667  little pizza Emilio Portes Gil
4   132613                   carnitas_mata
'''

print(cuisine.head())
'''
   placeID        Rcuisine
0   135110         Spanish
1   135109         Italian
2   135107  Latin_American
3   135106         Mexican
4   135105       Fast_Food
'''

#get the average(mean) of the ratings (from the 'frame' table) for each place(placeID)
rating = pd.DataFrame(frame.groupby('placeID')['rating'].mean())
print(rating.head())
'''
         rating
placeID        
132560     0.50
132561     0.75
132564     1.25
132572     1.00
132583     1.00
'''

#Adding 'rating_count' column to the above table that gives the count of ratings for each placeID
rating['rating_count'] = pd.DataFrame(frame.groupby('placeID')['rating'].count())
print(rating.head())
'''
         rating  rating_count
placeID                      
132560     0.50             4
132561     0.75             4
132564     1.25             4
132572     1.00            15
132583     1.00             4
'''

#print the statistics of the 'rating' table
print(rating.describe())
'''
           rating  rating_count
count  130.000000    130.000000
mean     1.179622      8.930769
std      0.349354      6.124279
min      0.250000      3.000000
25%      1.000000      5.000000
50%      1.181818      7.000000
75%      1.400000     11.000000
max      2.000000     36.000000
'''

#The maximum rating_count is 36. We find the placeID corresponding to that rating_count.
print(rating.sort_values('rating_count',ascending=False).head())
'''
           rating  rating_count
placeID                        
135085   1.333333            36
132825   1.281250            32
135032   1.178571            28
135052   1.280000            25
132834   1.000000            25
'''

'''
getting the name of the place having ID 135085 to show the name of the place having the highest
rating_count value, i.e. the place that is the most popular.
'''
print(places[places['placeID']==135085])
'''
     placeID                    name
121   135085  Tortas Locas Hipocampo
'''

#similarly, finding the cuisine in this place.
print(cuisine[cuisine['placeID']==135085])
'''
    placeID   Rcuisine
44   135085  Fast_Food
'''

'''
Building a utility matrix, i.e. getting a matrix of all users (rows) and all the places (columns)
wherein each entry is the place's rating given by that user. This would be a sparse matrix as a user would not usually rate all
places. So those values would be represented as a 'NaN'. 
'''
#data->table, values->cell_values, index->rows, columns->columns
places_crosstab = pd.pivot_table(data=frame, values='rating', index='userID', columns='placeID')
print(places_crosstab.head())
'''
placeID  132560  132561  132564  132572  132583  132584  132594  132608  ...
userID                                                                    
U1001       NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN   
U1002       NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN   
U1003       NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN   
U1004       NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN   
U1005       NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN  
This table has 130 columns.
'''

'''
Since Tortas was the most popular place (36 ratings), we display all the users' ratings for that place.
Displaying only the 36 users' ratings (ratings with actual values and not the NaN values).
'''
Tortas_ratings = places_crosstab[135085]
print(Tortas_ratings[Tortas_ratings >= 0])
'''
userID
U1001    0.0
U1002    1.0
U1007    1.0
U1013    1.0
U1016    2.0
U1027    1.0
U1029    1.0
U1032    1.0
U1033    2.0
U1036    2.0
U1045    2.0
U1046    1.0
U1049    0.0
U1056    2.0
U1059    2.0
U1062    0.0
U1077    2.0
U1081    1.0
U1084    2.0
U1086    2.0
U1089    1.0
U1090    2.0
U1092    0.0
U1098    1.0
U1104    2.0
U1106    2.0
U1108    1.0
U1109    2.0
U1113    1.0
U1116    2.0
U1120    0.0
U1122    2.0
U1132    2.0
U1134    2.0
U1135    0.0
U1137    2.0
Name: 135085, dtype: float64
'''

'''
Correlating all other places with the Tortas place 
using the PearsonR correlation coefficients
based on user-ratings. 
'''
similar_to_Tortas = places_crosstab.corrwith(Tortas_ratings)
corr_Tortas = pd.DataFrame(similar_to_Tortas, columns=['PearsonR'])
corr_Tortas.dropna(inplace=True) #another way of dropping NaN values
print(corr_Tortas.head())
'''
         PearsonR
placeID          
132572  -0.428571
132723   0.301511
132754   0.930261
132825   0.700745
132834   0.814823

PearsonR correlation coefficients:
1 : Strong positive correlation
0 : Unsure
-1 : Strong negative correlation

So correlating 132572 with tortas(135085) : 42% strong negative
Correlating 132754 with tortas : 93% strong positive 
etc
'''

'''
Having just the PearsonR coefficients isn't enough for correlation. 
Using the rating_count for all places along with the PearsonR coefficients for all places.
'''
Tortas_corr_summary = corr_Tortas.join(rating['rating_count'])
#Showing places having a large number of ratings.
print(Tortas_corr_summary[Tortas_corr_summary['rating_count']>=10].sort_values('PearsonR',ascending=False).head(10))
'''
         PearsonR  rating_count
placeID                        
135076   1.000000            13
135085   1.000000            36
135066   1.000000            12
132754   0.930261            13
135045   0.912871            13
135062   0.898933            21
135028   0.892218            15
135042   0.881409            20
135046   0.867722            11
132872   0.840168            12
'''

'''
The second placeID is the same as that of our target, Tortas. So we ignore that.
The other two places with PearsonR coefficients 1 have only 1 user-rating for those places. So them 
having a 100% positive correlation is misleading.
'''

'''
Taking the top 7 places (excluding 135076 and 135066)
And merging that table with cuisine to find the place whose cuisine matches that of Tortas
'''
places_corr_Tortas = pd.DataFrame([135085,132754,135045,135062,135028,135042,135046],index=np.arange(7),columns=['placeID'])
summary = pd.merge(places_corr_Tortas,cuisine,on='placeID')
print(summary)
'''
   placeID   Rcuisine
0   135085  Fast_Food
1   132754    Mexican
2   135028    Mexican
3   135042    Chinese
4   135046  Fast_Food
'''

# Getting the name of place with placeID 135046
print(places[places['placeID']==135046])
'''
    placeID                     name
42   135046  Restaurante El Reyecito
'''

# Getting the unique number of cuisines
print(cuisine['Rcuisine'].describe())
'''
count         916
unique         59
top       Mexican
freq          239
Name: Rcuisine, dtype: object
'''

'''
So out of 59 different cuisines we got one in the top 5 (summary table) that matched the Tortas cuisine.
So the "Restaurante El Reyecito" (135046) correlates the most with Tortas (135085).
And Reyecito had an 86% positive correlation with Tortas with a large number of ratings.

So this makes the correlation analysis pretty accurate.
'''
