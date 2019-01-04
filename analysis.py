
# coding: utf-8

# In[34]:


#!/usr/bin/env python3

# May first need:
# In your VM: sudo apt-get install libgeos-dev (brew install on Mac)
# pip3 install https://github.com/matplotlib/basemap/archive/v1.1.0.tar.gz

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import numpy as np



# In[36]:


from mpl_toolkits.basemap import Basemap as Basemap
from matplotlib.colors import rgb2hex
from matplotlib.patches import Polygon


# # Create a time series plot (by day) of positive and negative sentiment

# In[35]:


pos = pd.read_csv("2_pos.csv")
neg = pd.read_csv("2_neg.csv")


# In[16]:


merged = pos.merge(neg, on='date')


# In[17]:



merged.columns = ['positive', 'date','negative']
merged


# In[18]:


merged.to_csv("time_data.csv",index=False)


# In[19]:


"""
IMPORTANT
This is EXAMPLE code.
There are a few things missing:
1) You may need to play with the colors in the US map.
2) This code assumes you are running in Jupyter Notebook or on your own system.
   If you are using the VM, you will instead need to play with writing the images
   to PNG files with decent margins and sizes.
3) The US map only has code for the Positive case. I leave the negative case to you.
4) Alaska and Hawaii got dropped off the map, but it's late, and I want you to have this
   code. So, if you can fix Hawaii and Alask, ExTrA CrEdIt. The source contains info
   about adding them back.
"""


"""
PLOT 1: SENTIMENT OVER TIME (TIME SERIES PLOT)
"""
# Assumes a file called time_data.csv that has columns
# date, Positive, Negative. Use absolute path.

ts = pd.read_csv("time_data.csv")
# Remove erroneous row.
ts = ts[ts['date'] != '2018-12-31']

plt.figure(figsize=(12,5))
ts.date = pd.to_datetime(ts['date'], format='%Y-%m-%d')
ts.set_index(['date'],inplace=True)

ax = ts.plot(title="President Trump Sentiment on /r/politics Over Time",
        color=['green', 'red'],
       ylim=(0, 1.05))
ax.plot()
plt.savefig("part1.png")


# # Create 3 maps of the United States: pos, neg, diff

# In[68]:


pos = pd.read_csv("3_pos.csv")
neg = pd.read_csv("3_neg.csv")

merged = pos.merge(neg, on='state')
merged.columns = ['Positive', 'state','Negative']

merged['diff']=-(merged['Positive']-merged['Negative'])
merged


# In[66]:


merged.to_csv("state_data.csv",index=False)


# In[67]:



"""
PLOT 2: SENTIMENT BY STATE (POSITIVE AND NEGATIVE SEPARATELY)
# This example only shows positive, I will leave negative to you.
"""

# This assumes you have a CSV file called "state_data.csv" with the columns:
# state, Positive, Negative
#
# You should use the FULL PATH to the file, just in case.

state_data = pd.read_csv("state_data.csv")

"""
You also need to download the following files. Put them somewhere convenient:
https://github.com/matplotlib/basemap/blob/master/examples/st99_d00.shp
https://github.com/matplotlib/basemap/blob/master/examples/st99_d00.dbf
https://github.com/matplotlib/basemap/blob/master/examples/st99_d00.shx
IF YOU USE WGET (CONVERT TO CURL IF YOU USE THAT) TO DOWNLOAD THE ABOVE FILES, YOU NEED TO USE 
wget "https://github.com/matplotlib/basemap/blob/master/examples/st99_d00.shp?raw=true"
wget "https://github.com/matplotlib/basemap/blob/master/examples/st99_d00.dbf?raw=true"
wget "https://github.com/matplotlib/basemap/blob/master/examples/st99_d00.shx?raw=true"
The rename the files to get rid of the ?raw=true
"""

# Lambert Conformal map of lower 48 states.
m = Basemap(llcrnrlon=-119, llcrnrlat=22, urcrnrlon=-64, urcrnrlat=49,
        projection='lcc', lat_1=33, lat_2=45, lon_0=-95)
shp_info = m.readshapefile('st99_d00','states',drawbounds=True)  # No extension specified in path here.
pos_data = dict(zip(state_data.state, state_data.Positive))
neg_data = dict(zip(state_data.state, state_data.Negative))
# diff_data = dict(zip(state_data.state, state_data.diff))

# code for positive map
pos_colors = {}
statenames = []
pos_cmap = plt.cm.Greens # use 'hot' colormap
vmin = 0.24; vmax = 0.28
for shapedict in m.states_info:
    statename = shapedict['NAME']
    # skip DC and Puerto Rico.
    if statename not in ['District of Columbia', 'Puerto Rico']:
        pos= pos_data[statename]
        pos_colors[statename] = pos_cmap(1. - np.sqrt(( pos - vmin )/( vmax - vmin)))[:3]
    statenames.append(statename)
# cycle through state names, color each one.

# POSITIVE MAP
ax = plt.gca() # get current axes instance
for nshape, seg in enumerate(m.states):
    # skip Puerto Rico and DC
    if statenames[nshape] not in ['District of Columbia', 'Puerto Rico']:
        color = rgb2hex(pos_colors[statenames[nshape]]) 
        poly = Polygon(seg, facecolor=color, edgecolor=color)
        ax.add_patch(poly)
plt.title('Positive Trump Sentiment Across the US')
plt.savefig("pos_map.png")

neg_colors = {}
statenames = []
neg_cmap = plt.cm.Blues # use 'hot' colormap
vmin = 0.9; vmax = 0.95
for shapedict in m.states_info:
    statename = shapedict['NAME']
    # skip DC and Puerto Rico.
    if statename not in ['District of Columbia', 'Puerto Rico']:
        neg = neg_data[statename]
        neg_colors[statename] = neg_cmap(1. - np.sqrt(( pos - vmin )/( vmax - vmin)))[:3]
    statenames.append(statename)
# cycle through state names, color each one.


ax = plt.gca() # get current axes instance
for nshape, seg in enumerate(m.states):
    # skip Puerto Rico and DC
    if statenames[nshape] not in ['District of Columbia', 'Puerto Rico']:
        color = rgb2hex(neg_colors[statenames[nshape]]) 
        poly = Polygon(seg, facecolor=color, edgecolor=color)
        ax.add_patch(poly)
plt.title('Negative Trump Sentiment Across the US')
plt.savefig("neg_map.png")





# code for difference map
diff_colors = {}
statenames = []
diff_cmap = plt.cm.Blues # use 'hot' colormap
vmin = 0.63; vmax = 0.69
for shapedict in m.states_info:
    statename = shapedict['NAME']
    # skip DC and Puerto Rico.
    if statename not in ['District of Columbia', 'Puerto Rico']:
        diff = diff_data[statename]
        diff_colors[statename] = diff_cmap(1. - np.sqrt(( pos - vmin )/( vmax - vmin)))[:3]
    statenames.append(statename)
# cycle through state names, color each one.

# POSITIVE MAP
ax = plt.gca() # get current axes instance
for nshape, seg in enumerate(m.states):
    # skip Puerto Rico and DC
    if statenames[nshape] not in ['District of Columbia', 'Puerto Rico']:
        color = rgb2hex(diff_colors[statenames[nshape]]) 
        poly = Polygon(seg, facecolor=color, edgecolor=color)
        ax.add_patch(poly)
plt.title('Difference Trump Sentiment Across the US')
plt.savefig("diff_map.png")




# # Create TWO scatterplots, X axis is the submission score/ comment score separately

# In[52]:


pos = pd.read_csv("4_pos_sub.csv")
neg = pd.read_csv("4_neg_sub.csv")

merged = pos.merge(neg, on='sub_score')
merged.columns = ['Positive', 'submission_score','Negative']


merged


# In[53]:


merged.to_csv("submission_score.csv",index=False)


# In[54]:



"""
PLOT 5A: SENTIMENT BY STORY SCORE
"""
# What is the purpose of this? It helps us determine if the story score
# should be a feature in the model. Remember that /r/politics is pretty
# biased.

# Assumes a CSV file called submission_score.csv with the following coluns
# submission_score, Positive, Negative

story = pd.read_csv("submission_score.csv")
plt.figure(figsize=(12,5))
fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter(story['submission_score'], story['Positive'], s=10, c='b', marker="s", label='Positive')
ax1.scatter(story['submission_score'], story['Negative'], s=10, c='r', marker="o", label='Negative')
plt.legend(loc='lower right');

plt.xlabel('President Trump Sentiment by Submission Score')
plt.ylabel("Percent Sentiment")
plt.savefig("plot5a.png")


# In[55]:


pos = pd.read_csv("4_pos_com.csv")
neg = pd.read_csv("4_neg_com.csv")

merged = pos.merge(neg, on='com_score')
merged.columns = ['Positive', 'comment_score','Negative']


merged


# In[56]:


merged.to_csv("comment_score.csv",index=False)


# In[57]:



"""
PLOT 5B: SENTIMENT BY COMMENT SCORE
"""
# What is the purpose of this? It helps us determine if the comment score
# should be a feature in the model. Remember that /r/politics is pretty
# biased.

# Assumes a CSV file called comment_score.csv with the following columns
# comment_score, Positive, Negative

story = pd.read_csv("comment_score.csv")
plt.figure(figsize=(12,5))
fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter(story['comment_score'], story['Positive'], s=10, c='b', marker="s", label='Positive')
ax1.scatter(story['comment_score'], story['Negative'], s=10, c='r', marker="o", label='Negative')
plt.legend(loc='lower right');

plt.xlabel('President Trump Sentiment by Comment Score')
plt.ylabel("Percent Sentiment")
plt.savefig("plot5b.png")


# # ROC curve plot
# 

# In[12]:


pos_roc = pd.read_csv("pos_roc.csv")
neg_roc = pd.read_csv("neg_roc.csv")

pos_predict=pos_roc['pos']
pos_true=pos_roc['poslabel']

neg_predict=neg_roc['neg']
neg_true=neg_roc['neglabel']


# In[13]:


from sklearn.metrics import roc_curve, auc
 
fpr = dict()
tpr = dict()
roc_auc = dict()
 

fpr, tpr, _ = roc_curve(pos_true, pos_predict)
roc_auc = auc(fpr, tpr)
 
fpr2, tpr2, _ = roc_curve(neg_true, neg_predict)
roc_auc2 = auc(fpr2, tpr2)
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure()
plt.plot(fpr, tpr, label='pos ROC curve (area = %0.2f)' % roc_auc)
plt.plot(fpr2, tpr2,  label='neg ROC curve (area = %0.2f)' % roc_auc2)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

