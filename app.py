import streamlit as st
import pandas as pd
import numpy as np
import time
import holoviews as hv
import easier as ezr
import folium
import pylab as pl

hv.extension('bokeh')

class Loader:
    file_name = './csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv'
    
    @st.cache
    def df_raw(self):
        df = pd.read_csv(self.file_name)
        df = df[df['Country/Region'] == 'US']
        df = df.rename(columns={'Province/State': 'location', 'Country/Region': 'country', 'Lat': 'lat', 'Long': 'lon'})
        df = df.set_index(['country', 'location', 'lat', 'lon']).T
        df.index = pd.DatetimeIndex(df.index.values)
        df['cases'] = df.sum(axis=1)
        return df

    @st.cache
    def df_total(self):
        df = self.df_raw()
        df = df[['cases']].reset_index()
        df.columns = ['date', 'cases']
        df['days'] = (df.date - pd.Timestamp('3/1/2020')).dt.days
        df = df[df.days >= 0].reset_index(drop=True)
        return df

    @st.cache
    def df_loc(self):
        df = self.df_raw().T
        df['total'] = df.sum(axis=1)
        df = df[['total']].reset_index()
        df = df[df.total > 0]
        df.loc[:, 'lat'] = [np.NaN if l == '' else float(l) for l in df.lat]
        df.loc[:, 'lon'] = [np.NaN if l == '' else float(l) for l in df.lon]
        df = df[df.lat.notnull() & df.lon.notnull()]
        # df = df[['lat', 'lon']]

        return df

loader = Loader()


# @st.cache
# def load_total_data():

#     loader = Loader()
#     return loader.df_total

# @st.cache
# def load_total_data():

#     loader = Loader()
#     return loader.df_total

st.title('COVID 19')
data_load_state = st.text('Loading data...')



df = loader.df_total()

# --- Exponential
st.subheader('Exponential')
def model(p):
    return  2 ** ((p.x - p.t0) / p.tau)
    
fitter = ezr.Fitter(t0=-16,  tau=3)
fitter.fit(x=df.days, y=df.cases, model=model)

yf = fitter.predict(df.days)
ax = ezr.figure()
ax.plot(df.days, yf)
ax.scatter(df.days, df.cases, color=ezr.cc.b)
ax.set_xlabel('Days')
ax.set_ylabel('Cases')
ax.set_title('Exponential')

st.write(pl.gcf())
st.write(fitter.params.df[['val']])

# --- Logistic
st.subheader('Logistic')
def model(p):
    return p.a * .5 * (1 + np.tanh( (p.x - p.t0) / p.tau))

fitter = ezr.Fitter(a=1e6, tau=4, t0=20)
# fitter.optimizer_kwargs(xtol=1e-1, ftol=10)
fitter.fit(x=df.days, y=df.cases, model=model, plot_every=15, )

yf = fitter.predict(df.days)
ax = ezr.figure()
ax.plot(df.days, yf)
ax.set_xlabel('Days')
ax.set_ylabel('Cases')
ax.set_title('Logistic')

st.write(pl.gcf())
st.write(fitter.params.df[['val']])








df = loader.df_loc()

st.subheader('U.S. Map')
m = folium.Map(location=[35.29, -97.766], zoom_start=4, left='0%', width='100%', height='100%')
for tup in df.itertuples():
    folium.CircleMarker([tup.lat, tup.lon], radius=5, tooltip=f'{tup.location} {tup.total}', popup=f'<b>{tup.location}<br>{tup.total} cases</b>').add_to(m)

st.markdown(m._repr_html_(), unsafe_allow_html=True)

st.subheader('Arizona Map')
m = folium.Map(location=[35.19, -111.6], zoom_start=6, left='0%', width='100%', height='100%')
for tup in df.itertuples():
    folium.CircleMarker([tup.lat, tup.lon], radius=5, tooltip=f'{tup.location} {tup.total}', popup=f'<b>{tup.location}<br>{tup.total} cases</b>').add_to(m)

st.markdown(m._repr_html_(), unsafe_allow_html=True)

st.subheader('Tennessee Map')
m = folium.Map(location=[35.04, -85.3], zoom_start=7, left='0%', width='100%', height='100%')
for tup in df.itertuples():
    folium.CircleMarker([tup.lat, tup.lon], radius=5, tooltip=f'{tup.location} {tup.total}', popup=f'<b>{tup.location}<br>{tup.total} cases</b>').add_to(m)
st.markdown(m._repr_html_(), unsafe_allow_html=True)



data_load_state.text('Loading data ...done.')

