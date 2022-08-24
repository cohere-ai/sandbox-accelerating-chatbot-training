import streamlit as st
from streamlit import session_state
import pandas as pd
import numpy as np
from pipeline import generate_example_embeddings, search, umap_reduce
import plotly.express as px

if __name__ == "__main__":
  # some formatting
  st.set_page_config(layout="wide")

  # persist state of data for each session
  if not 'df' in session_state:
    data = pd.DataFrame(columns=["Route Name", 'Route Description', 'Deployed'])
    session_state['df'] = data 
  if not 'deployed_routes' in session_state:
    session_state['deployed_routes'] = []
  if not 'routes' in session_state:
    session_state['routes'] = []
  if not 'embeds' in session_state:
    session_state['embeds'] = []
  if not 'texts' in session_state:
    session_state['texts'] = []
  if not 'search_embed' in session_state:
    session_state['search_embed'] = None 

  # title and image
  h1, h2 = st.columns([1, 10])
  h1.image('./assets/profile-white-logo.png', width=100)
  h2.title('Route Generation with Cohere')
  with st.empty():
    st.write('\n#\n#\n')


  # start of containers
  col1, col2 = st.columns([10, 10], gap='large')
  with col1:
    # header
    st.subheader('Manage your routes')
  
    name = st.text_input('Route Name')
    desc = st.text_input('Route Description')

  ####################################
  # right hand side
  # this is handling the search function
  with col2:
    st.subheader('Try it out!')
    message = st.text_input('Message')
    if st.button('Send'):
      if not message:
        st.text('please input a message')
      else:
        result, search_embed = search(message, session_state.embeds, session_state.routes, n_comparisons=5)
        session_state['search_embed'] = search_embed
        st.subheader(result)


  ####################################
  # below both columns
  def create_scatter():
    if session_state.embeds:
      # dimension reduction for plotting
      if session_state.search_embed:
        reduced_df = umap_reduce(session_state.embeds + session_state.search_embed)
        scatter = px.scatter(x=reduced_df['x'],y=reduced_df['y'],
        color = session_state.routes + ['SEARCH TERM'],
        hover_data=[session_state.texts + [message]],
        width=800, height=600)
      else:
        reduced_df = umap_reduce(session_state.embeds)
        scatter = px.scatter(x=reduced_df['x'],y=reduced_df['y'], color=session_state.routes, hover_data=[session_state.texts], width=800, height=600)
      
      st.plotly_chart(scatter)
    else:
      st.text('No routes deployed yet')
  
  # buttons 
  b1, b2, b3, b4 = st.columns([1.5, 1.5, 1.5, 10], gap='small')

  # button to add routes
  if b1.button("Add Route"):
      # update dataframe state
      if not name or not desc:
        st.text('please add both name and description')
      else:
        session_state.df = session_state.df.append({'Route Name': name, "Route Description": desc, 'Deployed': 'No'}, ignore_index=True)

  # button to insert default values
  if b2.button('Use defaults'):
    defaults = {
      'Route Name': ['Open Account', 'Close Account', 'Lost Credit Card', 'Raise Limit', 'Address Change'],
      'Route Description': [
        'This handles requests to open a new account or create an account',
        'This handles any request to close an account or remove an account',
        'This handles lost or stolen cards',
        'handles inquiries and tasks related to modifying account limits',
        'handles the modification of current contact details on the an account'
        ]
        }
    default_df = pd.DataFrame(defaults)
    default_df['Deployed'] = 'No'
    session_state.df = session_state.df.append(default_df)

  # button to delpoy and create the routes
  if b3.button("Deploy Routes"):
    routes_to_run = []
    descriptions_to_run = []
    for i, r in enumerate(list(session_state.df['Route Name'])):
      if not r in session_state['deployed_routes']:
        session_state['deployed_routes'].append(r)
        routes_to_run.append(r)
        descriptions_to_run.append(list(session_state.df['Route Description'])[i])
    if len(descriptions_to_run) > 0:
      r, e, t = generate_example_embeddings(routes_to_run, descriptions_to_run)
      session_state.routes.extend(r)
      session_state.embeds.extend(e)
      session_state.texts.extend(t)
      st.text('All Routes Deployed!')
    else:
      st.text('No routes to deploy')

  if not session_state.df.empty:
    session_state.df['Deployed'] = session_state.df.apply(lambda x: 'Yes' if x['Route Name'] in session_state['deployed_routes'] else 'No', axis=1)


  # display routes
  with st.empty():
    st.write('\n#\n#\n')
  with st.container():
    st.header("All Routes")
    c1, c2, c3 = st.columns([5, 15, 10], gap='large')
    c1.subheader('Route Name')
    c2.subheader('Route Description')
    c3.subheader('Deployed')
    for name in session_state.df['Route Name'].values:
      c1.text(name)
    for description in session_state.df['Route Description'].values:
      c2.text(description)
    for status in session_state.df['Deployed'].values:
      c3.text(status)

  with st.container():
    with st.empty():
      st.write('\n#\n#\n')
    with st.expander('What is happing behind the scenes?'):
      create_scatter()