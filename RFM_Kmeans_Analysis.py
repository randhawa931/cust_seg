import warnings
warnings.filterwarnings('ignore')
# import library
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly
import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, iplot
import squarify

## 1. IMPORT DATA
# Import the dataset
sales = pd.read_excel('Online_Retail.xlsx')
print('{:,} rows; {:,} columns'.format(sales.shape[0], sales.shape[1]))
print('{:,} invoices don\'t have a customer id'.format(sales[sales.CustomerID.isnull()].shape[0]))
print('\n')
# drop the rows which do not have CustomerID
sales = sales[sales.CustomerID.notnull()]

# extract year, month and day
sales['InvoiceDay'] = sales.InvoiceDate.apply(lambda x: dt.datetime(x.year, x.month, x.day))
sales.head()

# the number of unique customers
sales.CustomerID.nunique()


## 2. RFM VALUES
# - Recency is days since the customers made the last purchase and by definition, the lower it is the better.
# - Frequency is the number of transaction in the last 12 months.
# - Monetary value is the total amout of money the customers spent in the last 12 months.

# The last day of purchase in total is 09 DEC, 2011. To calculate the day periods, let's set one day after the last one, or 10 DEC as a pin date. We will cound the diff days with pin_date.
print('Orders from {} to {}'.format(min(sales.InvoiceDay), max(sales.InvoiceDay)))

pin_date = max(sales.InvoiceDay) + dt.timedelta(1)

# Create total spend dataframe
sales['TotalSum'] = sales.Quantity * sales.UnitPrice
sales['InvoiceNo'].value_counts().head()

# calculate RFM values
rfm = sales.groupby('CustomerID').agg({
    'InvoiceDate' : lambda x: (pin_date - x.max()).days,
    'InvoiceNo' : 'count',
    'TotalSum' : 'sum'})

# rename the columns
rfm.rename(columns = {'InvoiceDate' : 'Recency',
                      'InvoiceNo' : 'Frequency',
                      'TotalSum' : 'Monetary'}, inplace = True)
rfm.reset_index(inplace=True)


# As the three columns are grouped by customers and count the days from the max date value, `Recency` is the days since the last purchase of a customer.
# `Frequency` is the number of purchases of a customer and `Monetary` is the total amount of spend of a customer.

# 3. RFM QUANTILES
# Now I will group the customers based on `Recency` and `Frequency`. I will use quantile values to get three equal percentile groups an then make three separate gruops.
# As the lower `Recency` value is the better, we will label them in decreasing order.

quintiles = rfm[['Recency', 'Frequency', 'Monetary']].quantile([.2, .4, .6, .8]).to_dict()

def r_score(x):
    if x <= quintiles['Recency'][.2]:
        return 5
    elif x <= quintiles['Recency'][.4]:
        return 4
    elif x <= quintiles['Recency'][.6]:
        return 3
    elif x <= quintiles['Recency'][.8]:
        return 2
    else:
        return 1

def fm_score(x, c):
    if x <= quintiles[c][.2]:
        return 1
    elif x <= quintiles[c][.4]:
        return 2
    elif x <= quintiles[c][.6]:
        return 3
    elif x <= quintiles[c][.8]:
        return 4
    else:
        return 5


# make a new column for group labels
rfm['R'] = rfm['Recency'].apply(lambda x: r_score(x))
rfm['F'] = rfm['Frequency'].apply(lambda x: fm_score(x, 'Frequency'))
rfm['M'] = rfm['Monetary'].apply(lambda x: fm_score(x, 'Monetary'))

# sum up the three columns
rfm['RFM_Segment'] = rfm['R'].map(str) + rfm['F'].map(str) + rfm['M'].map(str)
rfm['RFM_Score'] = rfm[['R', 'F', 'M']].sum(axis = 1)


# With this value, we can go further analysis such as what is the average values for each RFM values or leveling
# customers in total RFM score.
# calculate averae values for each RFM_score
rfm_agg = rfm.groupby('RFM_Score').agg({
    'Recency' : 'mean',
    'Frequency' : 'mean',
    'Monetary' : ['mean', 'count']
})

rfm_agg.round(1).head()


# The final score will be the aggregated value of RFM and we can make groups based on the `RFM_Score`

segt_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at risk',
    r'[1-2]5': 'can\'t loose',
    r'3[1-2]': 'about to sleep',
    r'33': 'need attention',
    r'[3-4][4-5]': 'loyal customers',
    r'41': 'promising',
    r'51': 'new customers',
    r'[4-5][2-3]': 'potential loyalists',
    r'5[4-5]': 'champions'
}

rfm['Segment'] = rfm['R'].map(str) + rfm['F'].map(str)
rfm['Segment'] = rfm['Segment'].replace(segt_map, regex=True)

# plot the distribution of customers over R and F
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

for i, p in enumerate(['R', 'F']):
    parameters = {'R':'Recency', 'F':'Frequency'}
    y = rfm[p].value_counts().sort_index()
    x = y.index
    ax = axes[i]
    bars = ax.bar(x, y, color='silver')
    ax.set_frame_on(False)
    ax.tick_params(left=False, labelleft=False, bottom=False)
    ax.set_title('Distribution of {}'.format(parameters[p]),
                fontsize=14)
    for bar in bars:
        value = bar.get_height()
        if value == y.max():
            bar.set_color('firebrick')
        ax.text(bar.get_x() + bar.get_width() / 2,
                value - 5,
                '{}\n({}%)'.format(int(value), int(value * 100 / y.sum())),
               ha='center',
               va='top',
               color='w')

plt.show()

# plot the distribution of M for RF score
fig, axes = plt.subplots(nrows=5, ncols=5,
                         sharex=False, sharey=True,
                         figsize=(10, 10))

r_range = range(1, 6)
f_range = range(1, 6)
for r in r_range:
    for f in f_range:
        y = rfm[(rfm['R'] == r) & (rfm['F'] == f)]['M'].value_counts().sort_index()
        x = y.index
        ax = axes[r - 1, f - 1]
        bars = ax.bar(x, y, color='silver')
        if r == 5:
            if f == 3:
                ax.set_xlabel('{}\nF'.format(f), va='top')
            else:
                ax.set_xlabel('{}\n'.format(f), va='top')
        if f == 1:
            if r == 3:
                ax.set_ylabel('R\n{}'.format(r))
            else:
                ax.set_ylabel(r)
        ax.set_frame_on(False)
        ax.tick_params(left=False, labelleft=False, bottom=False)
        ax.set_xticks(x)
        ax.set_xticklabels(x, fontsize=8)

        for bar in bars:
            value = bar.get_height()
            if value == y.max():
                bar.set_color('firebrick')
            ax.text(bar.get_x() + bar.get_width() / 2,
                    value,
                    int(value),
                    ha='center',
                    va='bottom',
                    color='k')
fig.suptitle('Distribution of M for each F and R',
             fontsize=14)
plt.tight_layout()
plt.show()


# count the number of customers in each segment
segments_counts = rfm['Segment'].value_counts().sort_values(ascending=True)

fig, ax = plt.subplots()

bars = ax.barh(range(len(segments_counts)),
              segments_counts,
              color='silver')
ax.set_frame_on(False)
ax.tick_params(left=False,
               bottom=False,
               labelbottom=False)
ax.set_yticks(range(len(segments_counts)))
ax.set_yticklabels(segments_counts.index)

for i, bar in enumerate(bars):
        value = bar.get_width()
        if segments_counts.index[i] in ['champions', 'loyal customers']:
            bar.set_color('firebrick')
        ax.text(value,
                bar.get_y() + bar.get_height()/2,
                '{:,} ({:}%)'.format(int(value),
                                   int(value*100/segments_counts.sum())),
                va='center',
                ha='left'
               )

plt.show()

rfm_by_segments = rfm.groupby('Segment')

retail_rfm_segments = rfm_by_segments['CustomerID'].count().reset_index(name='counts')


#let's exclude others segment for visualization
segment = list(retail_rfm_segments.iloc[:11].Segment)
score = list(retail_rfm_segments.iloc[:11].counts)
color_list = ["#248af1", "#eb5d50", "#8bc4f6", "#8c5c94", "#a170e8", "#fba521", "#75bc3f", "#ebbd34", "#34eb5f", "#34b4eb"]
plt.figure(figsize=(15,12))
plt.title('Customer Segments distribution')
squarify.plot(sizes=score, label=segment,color=color_list, alpha=0.7)

plt.show()


#  4. CUSTOMER SEGMENTATION WITH KMEANS

# We can also apply Kmeans clustering with RFM values. As Kmeans clustering require data to be normalized and has a symmetric distribution, preprocessing process in scale is needed.

# 4-1. Preprocessing
# plot the distribution of RFM values
plt.subplot(3, 1, 1); sns.distplot(rfm.Recency, label = 'Recency')
plt.subplot(3, 1, 2); sns.distplot(rfm.Frequency, label = 'Frequency')
plt.subplot(3, 1, 3); sns.distplot(rfm.Monetary, label = 'Monetary')

plt.tight_layout()
plt.show()

# As you can see above, the values are skewed and need to be normalized. Due to the zero or negative values in `Recency` and `MonetaryValue`, we need to set them 1 before log transformation and scaling.

# define function for the values below 0
def neg_to_zero(x):
    if x <= 0:
        return 1
    else:
        return x



# apply the function to Recency and MonetaryValue column
rfm['Recency'] = [neg_to_zero(x) for x in rfm.Recency]
rfm['Monetary'] = [neg_to_zero(x) for x in rfm.Monetary]


# unskew the data
rfm_log = rfm[['Recency', 'Frequency', 'Monetary']].apply(np.log, axis = 1).round(3)

# scale the data
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_log)


# transform into a dataframe
rfm_scaled = pd.DataFrame(rfm_scaled, index = rfm.index, columns = rfm_log.columns)
rfm_scaled.head()

# plot the distribution of RFM values
plt.subplot(3, 1, 1); sns.distplot(rfm_scaled.Recency, label = 'Recency')
plt.subplot(3, 1, 2); sns.distplot(rfm_scaled.Frequency, label = 'Frequency')
plt.subplot(3, 1, 3); sns.distplot(rfm_scaled.Monetary, label = 'Monetary')

plt.tight_layout()
plt.show()


# 3-2. K-MEANS CLUSTERING

# With the Elbow method, we can get the optimal number of clusters.

# initiate an empty dictionary
wcss = {}

# Elbow method with for loop
for i in range(1, 11):
    kmeans = KMeans(n_clusters= i, init= 'k-means++', max_iter= 300)
    kmeans.fit(rfm_scaled)
    wcss[i] = kmeans.inertia_


sns.pointplot(x = list(wcss.keys()), y = list(wcss.values()))
plt.xlabel('K Numbers')
plt.ylabel('WCSS')
plt.show()


# choose n_clusters = 3
clus = KMeans(n_clusters= 3, init= 'k-means++', max_iter= 300)
clus.fit(rfm_scaled)

# Assign the clusters to datamart
rfm['K_Cluster'] = clus.labels_
print('Cluster labels: ', clus.labels_)


# 5. VISUALIZATION

# 5-1. Snake Plot

# In marketing, snail plot and heatmap are often used plot for visualization. I'll use the `rfm_scaled` dataframe with normalized rfm values for the plot.

# assign cluster column
rfm_scaled['CustomerID'] = rfm.CustomerID
rfm_scaled['K_Cluster'] = clus.labels_
rfm_scaled['Segment'] = rfm.Segment
# rfm_scaled.reset_index(inplace = True)

# melt the dataframe
rfm_melted = pd.melt(frame= rfm_scaled, id_vars= ['CustomerID', 'Segment', 'K_Cluster'],
                     var_name = 'Metrics', value_name = 'Value')

# a snake plot with RFM
plt.figure(figsize=(16,12))
sns.lineplot(x = 'Metrics', y = 'Value', hue = 'Segment', data = rfm_melted)
plt.title('Snake Plot of RFM')
plt.legend(loc = 'upper right')

plt.show()

# a snake plot with K-Means
plt.figure(figsize=(16,12))
sns.lineplot(x = 'Metrics', y = 'Value', hue = 'K_Cluster', data = rfm_melted)
plt.title('Snake Plot of K_cluster')
plt.legend(loc = 'upper right')
plt.show()

# 5-2. Heatmap

# Heatmap is efficient for comparing the standardized values.

# the mean value for each cluster
cluster_avg = rfm.groupby('Segment').mean().iloc[:, 1:4]
cluster_avg.head()

# the mean value in total
total_avg = rfm.iloc[:, 1:4].mean()

# the proportional mean value
prop_rfm = cluster_avg/total_avg - 1

# heatmap
plt.figure(figsize=(12,8))
sns.heatmap(prop_rfm, cmap= 'coolwarm', fmt= '.2f', annot = True)
plt.title('Heatmap of RFM quantile')
plt.show()

# the mean value for each cluster
cluster_avg_K = rfm.groupby('K_Cluster').mean().iloc[:, 1:4]

# the proportional mean value
prop_rfm_K = cluster_avg_K/total_avg - 1
prop_rfm_K


# heatmap
plt.figure(figsize=(10,6))
sns.heatmap(prop_rfm_K, cmap= 'coolwarm', fmt= '.2f', annot = True)
plt.title('Heatmap of K-Means')
plt.show()


# 5-3. 3D Scatter plot with Plotly
# We can also check how the clusters are distributed across each RFM value.

sub = []
myColors = ['#db437b', '#d3d64d', '#568ce2', '#b467bc']
for i in range(3):
    df = rfm_scaled[rfm_scaled.K_Cluster == i]
    x = df.Recency
    y = df.Frequency
    z = df.Monetary
    color = myColors[i]

    trace = go.Scatter3d(x = x, y = y, z = z, name = str(i),
                         mode = 'markers', marker = dict(size = 5, color = color, opacity = .7))
    sub.append(trace)


data = [sub[0], sub[1], sub[2]]
layout = go.Layout(margin = dict(l = 0, r = 0, b = 0, t = 0),
                  scene = dict(xaxis = dict(title = 'Recency'), yaxis = dict(title = 'Frequency'), zaxis = dict(title = 'Monetary')))
fig_1 = go.Figure(data = data, layout = layout)
fig_1.show()

sub_2 = []
level = rfm_scaled.Segment.unique()
color_list = ["#ff0000", "#ff80ed", "#420420", "#065535", "#ffd700", "#00ff00", "#101010", "#794044", "#7fe5f0", "#bada55"]
for i in range(10):
    df = rfm_scaled[rfm_scaled.Segment == level[i]]
    x = df.Recency
    y = df.Frequency
    z = df.Monetary
    color = color_list[i]

    trace = go.Scatter3d(x = x, y = y, z = z, name = level[i],
                         mode = 'markers', marker = dict(size = 5, color = color, opacity = .7))
    sub_2.append(trace)

data = [sub_2[0], sub_2[1], sub_2[2], sub_2[3], sub_2[4], sub_2[5], sub_2[6], sub_2[7], sub_2[8], sub_2[9]]
layout = go.Layout(margin = dict(l = 10, r = 0, b = 0, t = 0),
                  scene = dict(xaxis = dict(title = 'Recency'), yaxis = dict(title = 'Frequency'), zaxis = dict(title = 'Monetary')))
fig_2 = go.Figure(data = data, layout = layout)
fig_2.show()
