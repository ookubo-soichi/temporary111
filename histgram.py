import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import random
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import base64

st.button('Reload')
st.title('My app')

df1 = pd.read_csv('../train.csv')
df2 = pd.read_csv('../train2.csv')

df_keys = list(df1.keys())
df_keys.remove('id')
df_keys.remove('date')

# df_key1 = st.sidebar.selectbox('Select Key', df_keys)
# eq1 = st.sidebar.selectbox('Select Operation', ['=', '!=', '>', '>=', '<=', '<'])
# thresh1 = st.sidebar.text_input('Enter Thresh')
# if thresh1 != '':
#      filter1 = eval('df1[df_key1] '+eq1+' '+thresh1)
#      filter2 = eval('df2[df_key1] '+eq1+' '+thresh1)
# else:
#      filter1 = []
#      filter2 = []
# if len(filter1) > 0 and len(filter2) > 0:
#      tmp_df1 = df1[filter1]
#      tmp_df2 = df2[filter2]
# else:
#      tmp_df1 = df1
#      tmp_df2 = df2

_filter = st.sidebar.text_input("Filter")
if _filter != '':
     try:
          filter1 = eval(_filter.replace('[', 'df1['))
          filter2 = eval(_filter.replace('[', 'df2['))
          tmp_df1 = df1[filter1]
          tmp_df2 = df2[filter2]
     except:
          st.sidebar.write('error')
          tmp_df1 = df1
          tmp_df2 = df2
else:
     tmp_df1 = df1
     tmp_df2 = df2
st.sidebar.write("((['age'] > 40) & (['age'] < 60)) | (['sex'] == 'Female')")
     
for df_key in df_keys:
     if df1[df_key].dtype != 'O':
          norm_dist = df1[df_key]
          q1 = norm_dist.quantile(0.25)
          q3 = norm_dist.quantile(0.75)
          iqr = q3 - q1
          bin_width = (2 * iqr) / (len(norm_dist) ** (1 / 3))
          if bin_width > 1e-6:
               bin_count = int(np.ceil((norm_dist.max() - norm_dist.min()) / bin_width))
          else:
               bin_count = 2
     else:
          bin_count = 10
     bins = st.slider(df_key, min_value=1, max_value=50, step=1, value=bin_count)
     fig1 = go.Figure()
     fig1.add_trace(go.Histogram(x=tmp_df1[df_key], nbinsx=bins, histnorm="percent"))
     fig1.add_trace(go.Histogram(x=tmp_df2[df_key], nbinsx=bins, histnorm="percent"))
     fig1.update_layout(barmode='overlay', bargap=0.1)
     fig1.update_traces(opacity=0.75)
     st.plotly_chart(fig1)
