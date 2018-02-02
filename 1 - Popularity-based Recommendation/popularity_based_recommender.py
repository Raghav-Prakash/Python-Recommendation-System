import pandas as pd
import numpy as np

frame = pd.read_csv('rating_final.csv')
cuisine = pd.read_csv('chefmozcuisine.csv')

#Show the first 5 rows of the rating_final.csv file.
print(frame.head())
'''
 userID  placeID  rating  food_rating  service_rating
0  U1077   135085       2            2               2
1  U1077   135038       2            2               1
2  U1077   132825       2            2               2
3  U1077   135060       1            2               2
4  U1068   135104       1            1               2

Rating is in the range 0-2, 0 being the lowest and 2 being the highest.
'''

#Show the first 5 rows of the chefmozcuisine.csv file.
print(cuisine.head())
'''
  placeID        Rcuisine
0   135110         Spanish
1   135109         Italian
2   135107  Latin_American
3   135106         Mexican
4   135105       Fast_Food
'''

'''
Have an array of ratings for each placeID and count the number of ratings for each placeID.
Sort these counts in decreasing order. The result would have the first column as the placeID
and the last one as 'rating'.
This shows the popularity of each placeID in decreasing order. So we know which place is the 
more popular.
'''
rating_count = pd.DataFrame(frame.groupby('placeID')['rating'].count())
print(rating_count.sort_values('rating',ascending=False).head())
'''
         rating
placeID        
135085       36
132825       32
135032       28
135052       25
132834       25
'''

'''
Now we project these top 5 places from the table above (only the placeIDs) and join('merge') this
resulting table with the 'cuisine' table to get the cuisine for the respective placeID.
'''

most_rated_places = pd.DataFrame([135085,132825,135032,135052,132834],index=np.arange(5),columns=['placeID'])
summary = pd.merge(most_rated_places,cuisine,on='placeID')
print(summary)
'''
   placeID         Rcuisine
0   135085        Fast_Food
1   132825          Mexican
2   135032        Cafeteria
3   135032     Contemporary
4   135052              Bar
5   135052  Bar_Pub_Brewery
6   132834          Mexican
'''

print(cuisine['Rcuisine'].describe())
'''
count         916
unique         59
top       Mexican
freq          239
Name: Rcuisine, dtype: object
'''

'''
In the 'cuisine' table, type 'Mexican' appears the most. It is also in the top 5 "most rated" cuisines
from our table above the one above. So this shows that the Mexican cuisine is the more popular choice.
'''
