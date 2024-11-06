import numpy as np
import pandas as pd
# import matplotlib as plt
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import PassiveAggressiveRegressor


# data = pd.read_csv('C:\Users\Owner\Downloads\archive\Instagram.csv')
data = pd.read_csv(r'C:\Users\Owner\Downloads\archive\Instagram.csv', encoding='ISO-8859-1')
print(data.head())

# lets take a look if dataset have null values or not
print(data.isnull().sum())
# if there is any, we can remove it with this: data = data.dropna()
# data.info()

# "Distribution of Impressions From Home"
plt.figure(figsize=(10, 8))
plt.style.use('fivethirtyeight')
plt.title("Distribution of Impressions From Home")
sns.histplot(data['From Home'], kde=True)  # kde=True adds a KDE plot on top of the histogram
plt.show()

# "Distribution of Impressions From Hashtags"
plt.figure(figsize=(10,8))
plt.title("Distribution of Impressions From Hashtags")
sns.histplot(data['From Hashtags'], kde=True)  # kde=True adds a KDE plot on top of the histogram
plt.show()

# "Distribution of Impressions From Explore"
plt.figure(figsize=(10, 8))
plt.title("Distribution of Impressions From Explore")
sns.distplot(data['From Explore'] , kde=True)
plt.show()

# Pie chart
home = data["From Home"].sum()
hashtags = data["From Hashtags"].sum()
explore = data["From Explore"].sum()
other = data["From Other"].sum()
labels = ['From Home','From Hashtags','From Explore','Other']
values = [home, hashtags, explore, other]
fig = px.pie(data, values=values, names=labels, title='Impressions on Instagram Posts From Various Sources', hole=0.5)
fig.show()

# caption
text = " ".join(i for i in data.Caption)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
plt.style.use('classic')
plt.figure( figsize=(12,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# hashtags
text = " ".join(i for i in data.Hashtags)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
plt.figure( figsize=(12,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# relationship between the number of likes and the number of impressions on Instagram posts!
figure = px.scatter(data_frame = data, x="Impressions",
                    y="Likes", size="Likes", trendline="ols",
                    title = "Relationship Between Likes and Impressions")
figure.show()
