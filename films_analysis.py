import openai
import pickle
import streamlit as st
import re
import string
import numpy as np
import pandas as pd 
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from pywaffle import Waffle
import seaborn as sns
from scipy.stats import chi2_contingency
import requests

def get_embedding(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response["data"][0]["embedding"]

def recommend_film(user_desc, df,desc_col,title_col):
    user_embedding = get_embedding(user_desc)

    similarities = []
    for desc in df[desc_col]:
        try:
            emb = get_embedding(desc)
            sim = cosine_similarity([user_embedding], [emb])[0][0]
            similarities.append(sim)
        except:
            similarities.append(0)

    df['similarity'] = similarities
    top_match = df.sort_values(by='similarity', ascending=False).iloc[0]
    return f"🎬 نرشح لك مشاهدة: **{top_match[title_col]}**\n\n📝 الوصف: {top_match[desc_col]}\n\n🔢 درجة التشابه: {top_match['similarity']:.2f}",top_match[title_col]

st.image(r"C:\Users\Ayman\Pictures\Screenshots\Screenshot 2025-04-17 141645.png")
st.title('Series | Movies App 🎥')
st.image(r"d0fac8df-f671-45fe-b21a-0274b4d19e98.png")
df = pd.read_excel(r"Data_films_cleaned.xlsx")
df['Favorite Series'].replace('GOT','Got',inplace=True)
df['Favorite Series'].replace('Got','Game of thrones',inplace=True)
df['Favorite Series'].replace('Game of Thrones','Game of thrones',inplace=True)
with open(r"log_reg_model.pkl", 'rb') as model_file:
    best_estimator_Log = pickle.load(model_file)

with open(r"embeddingss_films.pkl", 'rb') as emb_file:
    embeddings = pickle.load(emb_file)

arabic_stopwords = ['و', 'في', 'من', 'إلى', 'على', 'عن', 'ذلك', 'أن', 'هل', 'لم', 'كان', 'الذي', 'التي', 'ال']

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[أإآ]', 'ا', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'[ؤئ]', 'ء', text)
    text = re.sub(f"[{string.punctuation}]", " ", text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = ' '.join([word for word in text.split() if word not in arabic_stopwords])
    return text

def get_embedding(text):
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response['data'][0]['embedding']

select = st.sidebar.selectbox('Choose Option', ['None','Describtion Of Data','Sample Analysis','Sentiment Analysis','Summary Dashboard','Summary Analysis'])

if select=='None':
    st.image(r"ChatGPT Image Apr 16, 2025, 10_31_31 AM.png")
elif select=='Describtion Of Data':
    st.sidebar.image(r"886ef944-5bc6-4b76-b532-5d22bd9404f7.png")
    tabs=st.tabs(['Data Overview','Data Types','Statistics','Chi-Squared-Test'])
    with tabs[0]:
        st.dataframe(df.iloc[1:,:])
        columns=df.columns.tolist()
        st.success('Data contains columns :')
        for i in columns:
            st.markdown(i)
    with tabs[1]:
        type=df.dtypes
        data_types=type.to_frame()
        data_types=data_types.reset_index()
        data_types.columns=['columns','type']
        data_types=data_types.iloc[1:,:]
        st.dataframe(data_types)
        columns_categorical = data_types[data_types['type'] == 'object']['columns'].unique()
        columns_numerical = data_types[data_types['type'] != 'object']['columns'].unique()

        col1, col2 = st.columns(2)

        with col1:
            Categorical = st.checkbox('Categorical Columns')
            if Categorical:
                st.success('Categorical Columns are:')
                for col in columns_categorical:
                    st.markdown(f"- {col}")
                st.success(f'Number of Categorical columns Are {columns_categorical.shape[0]}')    

        with col2:
            Numerical = st.checkbox('Numerical Columns')
            if Numerical:
                st.success('Numerical Columns are:')
                for col in columns_numerical:
                    st.markdown(f"- {col}")
                st.success(f'Number of Numerical columns Are {columns_numerical.shape[0]}')    
    with tabs[2]:    
        col = st.selectbox('📌 Choose a Column', df.columns[2:])

        st.image(r"f8f25929-3b17-4b96-ae41-32fe16d95c9d.png")

        if pd.api.types.is_numeric_dtype(df[col]):
            st.subheader(f"🔍 Numerical Analysis for: `{col}`")
            
            mean_val = df[col].mean()
            std_val = df[col].std()
            median_val = df[col].median()

            st.markdown(f"- **Mean**: `{mean_val:.2f}`")
            st.markdown(f"- **Standard Deviation**: `{std_val:.2f}`")
            st.markdown(f"- **Median**: `{median_val:.2f}`")

            fig = go.Figure()

            fig.add_trace(go.Histogram(x=df[col], nbinsx=30, name='Data', marker_color='skyblue', opacity=0.75))

            fig.add_vline(
    x=mean_val,
    line=dict(color='red', dash='dash'),
    annotation_text="Mean",
    annotation_position="top left",
    annotation_font=dict(color="red")
            )

            fig.add_vline(
                x=median_val,
                line=dict(color='green', dash='dot'),
                annotation_text="Median",
                annotation_position="top left",
                annotation_font=dict(color="green")
            )

            fig.add_vline(
                x=mean_val + std_val,
                line=dict(color='orange', dash='dash'),
                annotation_text="Mean+1STD",
                annotation_position="top right",
                annotation_font=dict(color="orange")
            )

            fig.add_vline(
                x=mean_val - std_val,
                line=dict(color='orange', dash='dash'),
                annotation_text="Mean-1STD",
                annotation_position="top right",
                annotation_font=dict(color="orange")
            )   

            fig.update_layout(
                title=f"Distribution of {col}",
                xaxis_title=col,
                yaxis_title="Count",
                bargap=0.05
            )

            st.plotly_chart(fig)

        else:
            st.subheader(f"📋 Categorical Summary for: `{col}`")

            value_counts = df[col].value_counts().reset_index()
            value_counts.columns = [col, "Count"]

            st.markdown("**Frequency of Unique Categories:**")
            st.dataframe(value_counts)
    with tabs[3]:
        st.header("📊 Chi-Square Test Between Two Categorical Columns")

        col_to_check = st.selectbox('🧩 Choose the first column (e.g. Gender)', df.columns)
        col_target = st.selectbox("🎯 Choose the second column to compare (e.g. Product Type)", df.columns)

        unique_values = df[col_to_check].unique()
        st.write(f"Unique values in `{col_to_check}`:", unique_values)

        

        if pd.api.types.is_object_dtype(df[col_to_check]) and pd.api.types.is_object_dtype(df[col_target]):
            contingency_table = pd.crosstab(df[col_to_check], df[col_target])

            st.subheader("📋 Contingency Table")
            st.dataframe(contingency_table)

            chi2, p, dof, expected = chi2_contingency(contingency_table)

            st.markdown(f"✅ **Chi² Statistic**: `{chi2:.2f}`")
            st.markdown(f"📈 **Degrees of Freedom**: `{dof}`")
            st.markdown(f"📉 **P-value**: `{p:.4f}`")

            if p < 0.05:
                st.success("🎉 There's a statistically significant relationship between the two columns!")
            else:
                st.warning("🧐 No significant relationship detected (might be random).")

        


                    








elif select=='Sample Analysis':
    tabs=st.tabs(['Preferred Viewing Style','Preferred Content Type','Watching Behaviour','Influence of Popularity','Rate influence','Story Type','importance of Actor','Budged influence'])
    with tabs[0]:
        count_view = df['Preferred Viewing Style'].value_counts()

        logos = {
    'Both or All of these': "https://img.icons8.com/emoji/48/clapper-board-emoji.png",  
    'Home Theaters': "https://img.icons8.com/fluency/48/movie-projector.png",             
    'Streaming Services': "https://img.icons8.com/fluency/48/netflix-desktop-app.png",  
    'Theater': "https://img.icons8.com/fluency/48/movie-projector.png"                 
}

        platforms = count_view.index.tolist()
        values = count_view.values.tolist()

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=values,
            y=platforms,
            orientation='h',
            marker=dict(
                color='#00b894',
                line=dict(color='white', width=2)
            ),
            text=values,
            textposition='outside'
        ))

        for i, platform in enumerate(platforms):
            logo_url = logos.get(platform)
            if logo_url:
                fig.add_layout_image(
                    dict(
                        source=logo_url,
                        xref="x",
                        yref="y",
                        x=0,  # left aligned
                        y=platform,
                        sizex=5,
                        sizey=0.4,
                        xanchor="left",
                        yanchor="middle",
                        layer="above"
                    )
                )

        fig.update_layout(
            title="🎥 Preferred Viewing Styles by Audience",
            title_font_size=24,
            xaxis_title="Number of People",
            yaxis_title="Viewing Platform",
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False),
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(size=14, color="black"),
            margin=dict(l=130, r=40, t=60, b=40),
            height=450
        )

        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f'the Most perefered 2 methods is ')
        list_opt=count_view.index[0:2].values
        for i in list_opt:
            st.success(f'{i}')
    with tabs[1]:
        perfered_content_type = df['Preferred Content Type'].value_counts()

        content_types = perfered_content_type.index.tolist()[::-1]
        counts = perfered_content_type.values.tolist()[::-1]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=counts,
            y=content_types,
            mode='lines',
            line=dict(color='lightgray', width=2),
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=counts,
            y=content_types,
            mode='markers+text',
            marker=dict(color='mediumseagreen', size=14, line=dict(color='black', width=1.5)),
            text=counts,
            textposition='middle right',
            textfont=dict(size=14),
            showlegend=False
        ))

        most_preferred = content_types[-1]
        most_count = counts[-1]

        fig.add_annotation(
            x=most_count,
            y=most_preferred,
            text=f"🔥 Most Preferred: {most_preferred} ({most_count})",
            showarrow=True,
            arrowhead=3,
            ax=-80,
            ay=-20,
            font=dict(color="black", size=15)
        )

        fig.update_layout(
            title="🍿 Preferred Content Types by Audience",
            title_font_size=26,
            xaxis_title="Number of People",
            yaxis_title="Content Type",
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=130, r=40, t=80, b=60),
            font=dict(color="black", size=14),
            height=600  # Increased height for clarity
        )

                

        st.plotly_chart(fig)
    with tabs[2]:
        count_gradual = df['Binge or Gradual Watching'].value_counts()

        labels = count_gradual.index.tolist()
        values = count_gradual.values.tolist()

        fig = go.Figure()

        fig.add_trace(go.Barpolar(
            r=values,
            theta=labels,
            width=[30] * len(labels),  # Width of each arc
            marker_color=['#00b894', '#0984e3'],
            marker_line_color="black",
            marker_line_width=2,
            opacity=0.8
        ))

        fig.update_layout(
            title="⏱️ Binge vs Gradual Watching Preference",
            title_font_size=24,
            template=None,
            polar=dict(
                radialaxis=dict(showticklabels=True, ticks='', showgrid=False, linewidth=1),
                angularaxis=dict(rotation=90, direction='clockwise')
            ),
            showlegend=False,
            height=500,
            margin=dict(t=80, b=50, l=50, r=50),
            font=dict(size=14)
        )

        st.plotly_chart(fig)
    with tabs[3]:
        count_gradual = df['Influence of Popularity'].value_counts()
        categories = count_gradual.index.tolist()
        values = count_gradual.values.tolist()

        fig = go.Figure()

        for i, (cat, val) in enumerate(zip(categories, values)):
            fig.add_trace(go.Bar(
                x=[val],
                y=[cat],
                orientation='h',
                marker=dict(
                    color='#00b894',
                    line=dict(color='black', width=1.5)
                ),
                width=0.5,
                name=cat,
                text=[val],
                textposition='outside',
                showlegend=False
            ))

        fig.add_trace(go.Scatter(
            x=values,
            y=categories,
            mode='markers',
            marker=dict(color='#0984e3', size=12),
            showlegend=False
        ))

        fig.update_layout(
            title="🔥 Influence of Popularity on Content Selection",
            title_font_size=24,
            xaxis=dict(title="Number of People", showgrid=False),
            yaxis=dict(title="", showgrid=False),
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=14, color='black'),
            height=500,
            margin=dict(t=80, b=40, l=120, r=40)
        )

        st.plotly_chart(fig)    
    with tabs[4]:

        review_counts = df['Check Reviews Before Watching'].value_counts()

        labels = review_counts.index.tolist()
        values = review_counts.values.tolist()

        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.5,
            direction='clockwise',
            sort=False,
            textinfo='percent+label',
            marker=dict(colors=['#00b894', '#d63031'], line=dict(color='white', width=2)),
        )])

        fig.update_traces(rotation=180, pull=[0.05]*len(labels))

        fig.update_layout(
            title="🔍 Do People Check Reviews Before Watching?",
            title_font_size=22,
            showlegend=True,
            margin=dict(t=50, b=50, l=100, r=100),
            height=400,
            annotations=[dict(
                text='Review Check Behavior',
                x=0.5,
                y=0.5,
                font_size=16,
                showarrow=False
            )]
            
        )
        st.plotly_chart(fig)
    with tabs[5]:
        pref_counts = df['Real vs Fiction Preference'].value_counts()
        total = pref_counts.sum()

        fig = go.Figure()

        colors = ['#00b894', '#0984e3', 'red', '#d63031', '#0984e3']

        for i, (label, count) in enumerate(pref_counts.items()):
            percentage = round((count / total) * 100, 1)

            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=percentage,
                title={'text': f"{label}"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': colors[i % len(colors)]},
                    'bgcolor': "white",
                    'steps': [
                        {'range': [0, 50], 'color': '#f1f2f6'},
                        {'range': [50, 100], 'color': '#dfe6e9'}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 2},
                        'thickness': 0.7,
                        'value': percentage
                    }
                },
                domain={'row': i // 2, 'column': i % 2}
            ))

        rows = (len(pref_counts) + 1) // 2
        fig.update_layout(
            grid={'rows': rows, 'columns': 2, 'pattern': "independent"},
            title="🎯 Real vs Fiction Preferences (by %)",
            height=rows * 300,
            paper_bgcolor="white",
            margin=dict(t=60, b=40, l=40, r=40),
            font=dict(size=14)
        )

        st.plotly_chart(fig)
    with tabs[6]:
        rating_counts = df['Importance of Actors'].value_counts().sort_index()

        for i in range(1, 6):
            if i not in rating_counts:
                rating_counts[i] = 0

        rating_counts = rating_counts.sort_index()

        ratings = rating_counts.index.tolist()
        counts = rating_counts.values.tolist()
        stars = ['⭐' * r for r in ratings]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=ratings,
            y=counts,
            mode='markers+text+lines',
            marker=dict(size=40, color='#00e676'),  # Lime green dots
            text=stars,
            textposition='top center',
            line=dict(color='lightgray', width=2),
            hovertext=[f"{r} Stars: {c} people" for r, c in zip(ratings, counts)],
            hoverinfo="text"
        ))

        for i in range(len(ratings)):
            fig.add_annotation(
                x=ratings[i],
                y=counts[i] - max(counts) * 0.1,
                text=f"{counts[i]} votes",
                showarrow=False,
                font=dict(size=12)
            )

        fig.update_layout(
            title="🌟 Importance of Actors - Rating Distribution",
            xaxis=dict(title="Rating (Stars)", tickmode='array', tickvals=ratings),
            yaxis=dict(title="Number of People", showgrid=False),
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(size=14),
            margin=dict(t=60, b=60, l=40, r=40),
            height=500
        )
        st.plotly_chart(fig)
    with tabs[7]:
           
        data = df['Budget Indicates Quality'].value_counts()
        data = data.to_dict()

        # Custom colors for each response
        colors = ['#00b894', '#d63031', '#636e72', '#fdcb6e']  # Adjust for number of responses

        # Waffle chart config
        fig = plt.figure(
            FigureClass=Waffle,
            rows=5,  # number of rows
            values=data,
            colors=colors[:len(data)],
            title={'label': '💰 Does Budget Indicate Quality?', 'loc': 'center', 'fontsize': 18},
            labels=[f"{k} ({v})" for k, v in data.items()],
            legend={'loc': 'lower left', 'bbox_to_anchor': (0, -0.4), 'ncol': 2, 'framealpha': 0},
            block_arranging_style='snake',
            figsize=(12, 5)
        )
        st.pyplot(fig)
 
    

        




   

# elif select == 'Machine Learning':
#     tabs = st.tabs(['Type Prediction According to Events'])
#     with tabs[0]:
#         st.subheader("🎯 Prediction of Genre Based on Events")
#         user_input = st.text_area("✍️ أدخل ملخص الأحداث هنا")

#         if st.button('🔍 Submit'):
#             if not user_input.strip():
#                 st.warning("الرجاء إدخال نص لوصف الأحداث.")
#             else:
#                 cleaned_text = preprocess_text(user_input)
#                 embedding_input = get_embedding(cleaned_text)
#                 embedding_input = np.array(embedding_input).reshape(1, -1)
                
#                 prediction = best_estimator_Log.predict(embedding_input)
#                 st.success(f"✅ النوع المتوقع للفيلم: **{prediction[0]}**")
# elif select == 'Deep Learning':
#     tabs = st.tabs(['Movie Recommender', 'Analysis Chatbot'])
#     def get_movie_poster(movie_name, api='2db5cff7'):
#         url = f"http://www.omdbapi.com/?t={movie_name}&apikey={api}"
#         response = requests.get(url)
#         data = response.json()
        
#         if data['Response'] == 'True':
#             return data['Poster'], data['Title'], data['Year']
#         else:
#             return None, None, None


#     with tabs[0]:
#         st.title("🎥 Movie Recommendation Bot")

#         user_input = st.chat_input("اكتب وصف الفيلم اللي نفسك تشوفه...")

#         if user_input:
#             with st.chat_message("user"):
#                 st.markdown(user_input)

#             with st.chat_message("assistant"):
#                 with st.spinner("🎬 بنبحث عن أفضل فيلم ليك..."):
#                     try:
#                         reply,film = recommend_film(user_input, df, 'Series Summary and Highlights', 'Favorite Series')
#                         film = film.replace('"', '').replace("'", '')

                        
#                         recommended_title = film  # تعديل حسب تنسيق reply

#                         poster, title, year = get_movie_poster(recommended_title)

#                         if poster:
#                             st.image(poster, caption=f"{title} ({year})", use_column_width=True)
#                             st.markdown(reply)
#                         else:
                            
#                             st.warning("⚠️ Poster not found. Please check OMDb or try another title.")

#                     except Exception as e:
#                         st.error(f"❌ حصل خطأ: {e}")
#     with tabs[1]:
        
#         st.title("🎬 ChatBot Analysis")
#         st.image(r"C:\Users\Ayman\Pictures\Screenshots\Screenshot 2025-04-24 201155.png")
        
#         st.caption("Ask natural language questions about your dataset.")
#         def build_prompt(question):
#             return f"""
#         You are a data analysis assistant. You have a pandas DataFrame named df with the following columns:

#         ['Unnamed: 0', 'Timestamp', 'Gender', 'Age Range', 'Education Level',
#         'Preferred Content Type', 'Reason for Preferring Movies',
#         'Reason for Preferring Series', 'Binge or Gradual Watching',
#         'Preferred Viewing Style', 'Favorite Genres', 'Influence of Popularity',
#         'Importance of Actors', 'Budget Indicates Quality',
#         'Main Dislikes in Movies', 'Main Likes in Movies/Series',
#         'Favorite Film or Series', 'Favorite Series',
#         'Series Summary and Highlights', 'Favorite Series Actors',
#         'Ideal Episode Duration', 'Series Nationality',
#         'Favorite Aspects of Series', 'Number of Seasons', 'Favorite Movie',
#         'Favorite Actor', 'Movie Description', 'Movie Duration',
#         'Importance of Release Date', 'Movie Decade', 'Movie Type',
#         'Influence of Movie Budget', 'Movie Watching Preference',
#         'Watching Alone or With Others', 'Real vs Fiction Preference',
#         'Check Reviews Before Watching', 'Additional Comments',
#         'Age_classification', 'cleaned_main_dislikes']

#         Your job is to write a Python code snippet using pandas to answer user questions about the data. Only return the Python code. If the question is unrelated, respond with: "This question is not related to the dataset."

#         Examples:

#         Q: Do most people care about movie budget?
#         df['Influence of Movie Budget'].value_counts().nlargest(3)

#         Q: What is the most content preferred by males?
#         df[df['Gender'].str.lower() == 'male']['Preferred Content Type'].value_counts()

#         Q: What is the average movie duration per movie type?
#         df.groupby('Movie Type')['Movie Duration'].mean()

#         Q: What are the most common preferred genres per gender?
#         df.groupby('Gender')['Favorite Genres'].apply(lambda x: x.value_counts().head(1))

#         Q: Which age range prefers series?
#         df[df['Preferred Content Type'] == 'Series']['Age Range'].value_counts()

#         Q: What is the most mentioned reason for preferring movies?
#         df['Reason for Preferring Movies'].value_counts().head(3)

#         Q: How many people binge watch?
#         df['Binge or Gradual Watching'].value_counts()

#         Q: Average duration of movies liked by people who check reviews?
#         df[df['Check Reviews Before Watching'] == 'Yes']['Movie Duration'].mean()

#         Q: {question}
#         """

#         def run_analysis(question, df):
#             prompt = build_prompt(question)
#             response = openai.ChatCompletion.create(
#                 model="chatgpt-4o-latest",
#                 messages=[{"role": "user", "content": prompt}],
#                 temperature=0
#             )

#             raw_code = response.choices[0].message["content"].strip()

#             if "not related" in raw_code.lower():
#                 return None, "❌ This question is not related to the dataset."

#             code = raw_code.strip("` \n")
#             if code.lower().startswith("python"):
#                 code = code[6:].strip()

#             try:
#                 local_vars = {'df': df}

#                 if "\n" not in code and " = " not in code:
#                     result = eval(code, {}, local_vars)
#                 else:
#                     exec(code, {}, local_vars)

#                     result = None
#                     for val in reversed(local_vars.values()):
#                         if isinstance(val, (pd.Series, pd.DataFrame, int, float, str)):
#                             result = val
#                             break

#                 return code, result

#             except Exception as e:
#                 return code, f"⚠️ Error in executing code:\n\n{e}"

#         input_mode = st.radio("Choose input mode:", ["Text Input", "Voice Input 🎙️ (Record)"])

#         question = None

#         if input_mode == "Text Input":
#             question = st.text_input("💬 Ask a question about your data")

#         elif input_mode == "Voice Input 🎙️ (Record)":
#             st.info("🎤 Click the button to start recording your question...")

#             if st.button("🎙️ Start Recording"):
#                 recognizer = sr.Recognizer()
#                 with sr.Microphone() as source:
#                     st.warning("🎙️ Recording... Please speak clearly.")
#                     audio_data = recognizer.listen(source, timeout=5)
#                     st.success("✅ Recording complete! Processing...")
#                     try:
#                         question = recognizer.recognize_google(audio_data)
#                         st.success(f"🗣️ Recognized Question: {question}")
#                     except sr.UnknownValueError:
#                         st.error("⚠️ Could not understand the audio. Please try again.")
#                     except sr.RequestError:
#                         st.error("⚠️ Error with the speech recognition service.")

#         # 🚀 Process the question
#         if question:
#             with st.spinner("🤖 Thinking..."):
#                 code, output = run_analysis(question, df)

#                 if code:
#                     st.subheader("🧾 Generated Code")
#                     st.code(code, language="python")

#                 st.subheader("📊 Result")
#                 st.write(output)

#                 # 🚀 Process the question
#                 if question:
#                     with st.spinner("🤖 Thinking..."):
#                         code, output = run_analysis(question, df)

#                         if code:
#                             st.subheader("🧾 Generated Code")
#                             st.code(code, language="python")

#                         st.subheader("📊 Result")
#                         st.write(output)


        
        
     
elif select=='Sentiment Analysis':
    select=st.sidebar.selectbox('Choose option',['Movies And Series','movies','Series'])
    if select=='Movies And Series':
        tabs=st.tabs(['Opinions of people About Bad Movies And Series','Favourite Actors'])
        with tabs[0]:
            data_resons=pd.read_excel(r"Dislikes_film.xlsx")
            reason_categories = {
    "Story": [
         "story repeated", "well written script", "boring", "open ending",
        "short detail boring end", "hasnt fully cover detail", "weak storyline poor direction",
        "bad story", "bad ending lot action middle movie average egyptian production",
        "dislike movie bad writing weak character poor pacing overused clichés bad cgi forced message unnecessary sequel remake thing make movie feel unoriginal boring watch"
    ],
    "Actor / Performance": [
         "low performance", "lack experience"
    ],
    "Production / Visuals": [
        "bad cgi bad cinematography poor character development movie less hour",
        "movie available quality may poor", "bad quality"
    ],
    "Content / Themes": [
        "bad sense disrespect religion", "lgbtq thing happening"
    ],
    "Pacing / Flow": [
        "slow pace event", "think dragging event", "exaggeration", "interested"
    ]
}

            st.title("🎬 Why Did People Dislike the Film?")
            st.markdown("Select categories to view detailed audience reasons:")

            selected_reasons = []

            cols = st.columns(len(reason_categories))
            checkbox_states = {}

            for col, (category, reasons) in zip(cols, reason_categories.items()):
                with col:
                    checkbox_states[category] = st.checkbox(category)

            for category, selected in checkbox_states.items():
                if selected:
                    st.subheader(f"🔹 {category}")
                    for reason in reason_categories[category]:
                        st.markdown(f"- {reason}")
                    selected_reasons.extend(reason_categories[category])

            if selected_reasons:
                st.markdown("---")
                st.success(f"✅ {len(selected_reasons)} detailed reasons were selected that make people dislike films.")
        with tabs[1]:
            df=df[~(df['Favorite Actor']=='unknown')]
            top_actors = df['Favorite Actor'].value_counts().head(6)

            actor_images = {
    'Leonardo Dicaprio': 'https://sf2.closermag.fr/wp-content/uploads/closermag/2023/04/Leonardo-DiCaprio-au-66eme-Festival-de-Cannes-le-15-mai-2013-scaled-546x410.jpg',
    'Robert Downey Jr': 'https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/Robert_Downey_Jr_2014_Comic_Con_%28cropped%29.jpg/330px-Robert_Downey_Jr_2014_Comic_Con_%28cropped%29.jpg',
    'Ahmed Helmy': 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/Ahmed_Helmy.jpg/330px-Ahmed_Helmy.jpg',
    'Tom Cruise': 'https://upload.wikimedia.org/wikipedia/commons/thumb/d/d2/Tom_Cruise_avp_2014_4.jpg/330px-Tom_Cruise_avp_2014_4.jpg',
    'Jason Statham': 'https://upload.wikimedia.org/wikipedia/commons/thumb/1/16/Jason_Statham_2018.jpg/330px-Jason_Statham_2018.jpg',
    'Ahmed Ezz': 'https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/Ahmed_Ezz_2020.jpg/330px-Ahmed_Ezz_2020.jpg'
}


            fig = go.Figure(go.Bar(
                x=top_actors.index,
                y=top_actors.values,
                marker_color='#00cec9',
                text=top_actors.values,
                textposition='outside'
            ))

            for i, actor in enumerate(top_actors.index):
                if actor in actor_images:
                    fig.add_layout_image(
                        dict(
                            source=actor_images[actor],
                            x=actor,
                            y=top_actors.values[i] + 2,
                            xref="x",
                            yref="y",
                            sizex=0.8,
                            sizey=5,
                            xanchor="center",
                            yanchor="bottom",
                            layer="above"
                        )
                    )

            fig.update_layout(
                title="🎭 Most Favorite Actors by Audience",
                xaxis_title="Actor",
                yaxis_title="Votes",
                height=600,
                plot_bgcolor="white",
                paper_bgcolor="white",
                margin=dict(t=100, b=60),
                font=dict(size=13),
                showlegend=False
            )        
            st.plotly_chart(fig)
    elif select=='Series':
        tabs=st.tabs(['Reasons for prefering Series','Favourite Aspects Of Series','Number of Seasons','Ideal Episode Duration','Series Nationality','Favourite Series'])
        with tabs[0]:
            reasons=df['Reason for Preferring Series'].value_counts()
            fig = go.Figure(data=[go.Pie(
            labels=reasons.index,
            values=reasons.values,
            hole=0.5,  # This creates the donut hole
            textinfo='percent+label',
            marker=dict(
        colors=[
            "#009688", "#004d40", "#607d8b", "#ffc107", "#cfd8dc", "#212121"
        ],
        line=dict(color='#000000', width=1)
    ),
            insidetextorientation='radial'
        )])

            fig.update_layout(
            title_text="🍿 Why Do People Prefer Series Over Movies?",
            title_font_size=22,
            annotations=[dict(text='Series Preference', x=0.5, y=0.5, font_size=16, showarrow=False)],
            showlegend=True,
            font=dict(color="#333", size=9),
            margin=dict(t=50, b=50, l=100, r=100)
        )

            st.plotly_chart(fig, use_container_width=True)        
        with tabs[1]:     
            aspect_counts = df['Favorite Aspects of Series'].value_counts().reset_index()
            aspect_counts.columns = ['Aspect', 'Count']

            fig = px.scatter(
                aspect_counts,
                x=[i for i in range(len(aspect_counts))],  # fake x for spacing
                y=[1] * len(aspect_counts),  # fixed y
                size='Count',
                color='Aspect',
                text='Aspect',
                size_max=90,
                color_discrete_sequence=px.colors.qualitative.Safe
            )

            fig.update_traces(textposition='top center')
            fig.update_layout(
                title="✨ Favorite Aspects of Series (Bubble View)",
                xaxis=dict(showgrid=False, zeroline=False, visible=False),
                yaxis=dict(showgrid=False, zeroline=False, visible=False),
                showlegend=False,
                height=500,
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(size=13)
            )       
            st.plotly_chart(fig)
        with tabs[2]:
            season_counts = df['Number of Seasons'].value_counts().sort_index()
            labels = season_counts.index.astype(str).tolist()
            values = season_counts.values.tolist()

            colors = ['#00b894'] * len(labels)  # Teal-like

            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=labels,
                y=values,
                text=values,
                textposition='outside',
                marker_color=colors,
                hovertemplate="Season %{x}<br>Count: %{y}<extra></extra>",
                name="Seasons"
            ))

            for i, (label, count) in enumerate(zip(labels, values)):
                fig.add_annotation(
                    x=label,
                    y=count + max(values)*0.07,
                    text=f"<b>{label}</b>",
                    showarrow=False,
                    font=dict(size=16, color='#2d3436')
                )

            fig.update_layout(
                title="🎬 Number of Seasons - Viewer Distribution",
                xaxis_title="Seasons",
                yaxis_title="Number of People",
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(size=13),
                height=500,
                margin=dict(t=60, b=60, l=60, r=60),
                showlegend=False
            )

            
            
            st.plotly_chart(fig)
        with tabs[3]:
            ideal_episode_duration = df['Ideal Episode Duration'].value_counts().reset_index()
            ideal_episode_duration.columns = ['Duration', 'Count']

            custom_color_scale = ['#dfe6e9', '#81ecec', '#00cec9', '#00b894', '#019875', '#00695c']

            fig = px.sunburst(
                ideal_episode_duration,
                path=['Duration'],
                values='Count',
                color='Count',
                color_continuous_scale=custom_color_scale,
                title='Ideal Episode Duration Distribution'
            )

            # Professional Styling
            fig.update_layout(
                title_font_size=22,
                title_x=0.5,
                paper_bgcolor='white',
                plot_bgcolor='white',
                font=dict(color='#2d3436', size=14),
                margin=dict(t=90, l=0, r=0, b=0),
                coloraxis_colorbar=dict(
                    title='Count',
                    titlefont=dict(color='#2d3436'),
                    tickfont=dict(color='#2d3436')
                )
            )
            st.plotly_chart(fig)
        with tabs[4]:    
            nationality_counts = df['Series Nationality'].value_counts().reset_index()
            st.write(nationality_counts)
            data = {
    'Series Nationality': ['American', 'خيار 7', 'Turkish', 'British', 'Korean'],
    'Count': [38, 16, 11, 6, 5]
}
            df_nat = pd.DataFrame(data)

            # Mapping demonyms to countries
            nationality_to_country = {
                'American': 'United States',
                'British': 'United Kingdom',
                'Turkish': 'Turkey',
                'Korean': 'South Korea',
                # Add more mappings as needed
            }

            df_nat['Country'] = df_nat['Series Nationality'].map(nationality_to_country)
            df_nat = df_nat.dropna(subset=['Country'])  

            color_scale = [
                [0.0, '#dff9f2'],
                [0.5, '#55efc4'],
                [1.0, '#00b894']
            ]

            fig = px.choropleth(
                df_nat,
                locations='Country',
                locationmode='country names',
                color='Count',
                hover_name='Country',
                color_continuous_scale=color_scale,
                title='Global Distribution of Series by Nationality'
            )

            fig.update_layout(
                title_x=0.5,
                geo=dict(showframe=False, showcoastlines=True, projection_type='natural earth'),
                paper_bgcolor='white',
                font=dict(color='#2d3436', size=14),
                margin=dict(t=50, l=0, r=0, b=0),
                coloraxis_colorbar=dict(
                    title='Count',
                    tickfont=dict(color='#2d3436'),
                    titlefont=dict(color='#2d3436')
                )
            )   
            st.plotly_chart(fig)
        with tabs[5]:
            
            
            x_val = df['Favorite Series'].value_counts().nlargest(10)

            # Create the bar chart
            fig = go.Figure(go.Bar(
                x=x_val.index,
                y=x_val.values,
                text=x_val.values,  # Add the count as text
                textposition='outside',  # Position the count outside the bars
                marker_color='#00b894',  # Customize color to match your desired color scheme
                marker_line_color='black',  # Add a black border around the bars
                marker_line_width=2  # Add a border width
            ))

            # Customize layout
            fig.update_layout(
                title={
                    'text': "Favorite TV Series with Count",
                    'y':0.95,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': dict(size=24, color='black')
                },
                xaxis_title="TV Series",
                yaxis_title="Count",
                plot_bgcolor="white",
                paper_bgcolor="white",
                font_color="black",
                showlegend=False,
                margin=dict(t=50, l=40, r=40, b=50)
            )

            # Display the chart using Streamlit
            st.plotly_chart(fig)
                
    else:
        tabs=st.tabs(['Movie  Likely to People','Fvourite Movie Decade','perfered Movie Duration','Favorite Movie'])     
        with tabs[2]:
            df['Movie Duration'] = df['Movie Duration'].astype(str)

            # Get the counts of each category in the 'Movie Duration' column
            duration_counts = df['Movie Duration'].value_counts().sort_index()
            labels = duration_counts.index
            values = duration_counts.values

            # Custom color palette
            colors = ['#00b894' for _ in values]

            # Create radial bar chart using polar coordinates
            fig = go.Figure()

            fig.add_trace(go.Barpolar(
                r=values,
                theta=[i * (360 / len(values)) for i in range(len(values))],
                width=[360 / len(values)] * len(values),
                marker_color=colors,
                marker_line_color="white",
                marker_line_width=2,
                opacity=0.9,
                hoverinfo="text",
                text=[f"{label}: {value} movies" for label, value in zip(labels, values)]
            ))

            # Layout styling
            fig.update_layout(
                title='Movie Duration Distribution (Radial View)',
                title_x=0.5,
                polar=dict(
                    radialaxis=dict(showticklabels=False, ticks=''),
                    angularaxis=dict(showticklabels=True, direction='clockwise', tickfont=dict(size=12))
                ),
                showlegend=False,
                paper_bgcolor='white',
                font=dict(color='#2d3436'),
                margin=dict(t=60, b=60, l=0, r=0)
            )



            st.plotly_chart(fig)
        with tabs[1]:
            movie_decade_count = {'70s': 13, '80s': 15, '90s': 40, 'Other': 139}
            decades = list(movie_decade_count.keys())
            counts = list(movie_decade_count.values())

            # Create a timeline-style plot
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=decades,
                y=counts,
                mode='lines+markers',
                marker=dict(size=12, color='#00b894', line=dict(width=2, color='white')),
                line=dict(color='#00b894', width=3),
                text=[f'{decade}: {count} movies' for decade, count in zip(decades, counts)],
                hoverinfo='text'
            ))

            # Layout styling
            fig.update_layout(
                title='Movie Decade Distribution (Timeline)',
                title_x=0.5,
                xaxis_title='Decades',
                yaxis_title='Number of movies',
                paper_bgcolor='white',
                font=dict(color='#2d3436'),
                showlegend=False,
                margin=dict(t=40, b=60, l=50, r=50)
            )
            
            st.plotly_chart(fig)
        with tabs[0]:
            movie_type_counts = df['Movie Type'].value_counts()
            labels = movie_type_counts.index
            values = movie_type_counts.values

            # Define the color palette with #00b894 and its shades
            color_palette = ['#00b894', '#00a68c', '#00967a', '#00866a', '#00755a']  # Shades of #00b894
            colors = np.linspace(0, 1, len(color_palette))

            # Create a custom horizontal bar chart
            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=values,
                y=labels,
                orientation='h',
                marker=dict(
                    color=values,
                    colorscale=color_palette,  # Using the custom color scale
                    line=dict(color='black', width=1)  # Adding borders to bars
                ),
                text=values,
                textposition='inside',
                hoverinfo='x+text',  # Hover shows the count and value
                opacity=0.8  # Slight transparency for a clean look
            ))

            # Layout styling
            fig.update_layout(
                title='Distribution of Movie Types (Horizontal Bars)',
                title_x=0.5,
                xaxis_title='Number of People',
                yaxis_title='Movie Type',
                paper_bgcolor='white',
                font=dict(color='#2d3436'),
                showlegend=False,
                margin=dict(t=40, b=60, l=100, r=40),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=False, ticks=''),
                plot_bgcolor='white'
            )
            

            st.plotly_chart(fig)
        with tabs[3]:
            df=pd.read_csv(r"finally movie survey.csv")
            

            # Get top 5 movies
            x_val = df['favorite_movie'].value_counts().nlargest(5)

            # Custom colors: Including red and the color provided
            colors = ['red', '#00967a', '#00967a', '#00866a', '#00866a']

            # Create a Donut chart with explosion effect
            fig = go.Figure(go.Pie(
                labels=x_val.index,
                values=x_val.values,
                hole=0.4,
                marker=dict(colors=colors),
                pull=[0.1, 0, 0, 0, 0],  # Explode the first slice
                textinfo='percent+label',
                insidetextorientation='radial',
                hoverinfo='label+percent',
            ))

            # Customize layout
            fig.update_layout(
                title={
                    'text': "Top 5 Favorite Movies",
                    'y':0.95,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': dict(size=28, color='black')
                },
                plot_bgcolor="white",
                paper_bgcolor="white",
                font_color="black",
                margin=dict(t=50, l=25, r=25, b=25)
            )

            st.plotly_chart(fig)
            

           



            
        
        
elif select=='Summary Dashboard':
    

   
    st.title("📊 Power BI Report")
    

    st.markdown("[🟢 Open Report in New Tab](https://app.powerbi.com/links/f_R1uWPbKY?ctid=eaf624c8-a0c4-4195-87d2-443e5d7516cd&pbi_source=linkShare)", unsafe_allow_html=True)
    st.image(r"C:\Users\Ayman\Pictures\Screenshots\Screenshot 2025-04-17 171050.png")
else:
    
    st.title("📊 Automated EDA Report")


    with st.expander("🧪 Chi-Square Test for Categorical Columns"):
        st.write("This test checks for dependency between each pair of categorical columns in your dataset.")

        categorical_cols = [col for col in df.columns if df[col].dtype == 'object' or df[col].nunique() < 10]

        chi_results = []
        significant_pairs = {}

        for i in range(len(categorical_cols)):
            for j in range(i+1, len(categorical_cols)):
                col1, col2 = categorical_cols[i], categorical_cols[j]

                try:
                    contingency_table = pd.crosstab(df[col1], df[col2])
                    chi2, p, dof, expected = chi2_contingency(contingency_table)
                    sig = '✅' if p < 0.05 else '❌'

                    chi_results.append({
                        'Column 1': col1,
                        'Column 2': col2,
                        'Chi2 Statistic': round(chi2, 3),
                        'p-value': round(p, 5),
                        'Significant': sig
                    })

                    if p < 0.05:
                        pair_key = f"{col1} vs {col2}"
                        significant_pairs[pair_key] = contingency_table

                except Exception as e:
                    chi_results.append({
                        'Column 1': col1,
                        'Column 2': col2,
                        'Chi2 Statistic': 'Error',
                        'p-value': 'Error',
                        'Significant': str(e)
                    })

        chi_df = pd.DataFrame(chi_results)
        st.dataframe(chi_df)

        if significant_pairs:
            st.markdown("### 🔍 Explore Significant Relationships")
            selected_pair = st.selectbox("Select a significant pair to view its Crosstab", list(significant_pairs.keys()))
            st.markdown(f"#### 📋 Crosstab for: `{selected_pair}`")
            st.dataframe(significant_pairs[selected_pair])
        else:
            st.info("No significant relationships found (p < 0.05).")

  
    with st.expander("📊 Top Categories in Categorical Columns"):
        st.write("This section shows the top 3 most frequent values in each categorical column, along with their counts and percentages.")

        for col in categorical_cols:
            st.markdown(f"### 📁 Column: `{col}`")

            value_counts = df[col].value_counts(dropna=False)
            top_3 = value_counts.head(3)
            total = value_counts.sum()
            percentages = (top_3 / total * 100).round(2)

            result_df = pd.DataFrame({
                'Category': top_3.index.astype(str),
                'Count': top_3.values,
                'Percentage': percentages.values
            })

            st.dataframe(result_df, use_container_width=True)

            fig = px.bar(
                result_df,
                x='Category',
                y='Count',
                text='Percentage',
                color='Category',
                color_discrete_sequence=['#00b894', '#55efc4', '#81ecec'],
                title=f"Top 3 Values in `{col}`"
            )
            fig.update_traces(texttemplate='%{text}%', textposition='outside')
            fig.update_layout(showlegend=False, xaxis_title='Category', yaxis_title='Count', plot_bgcolor='white', paper_bgcolor='white')
            st.plotly_chart(fig, use_container_width=True)

            top_cat = result_df.loc[0, 'Category']
            top_pct = result_df.loc[0, 'Percentage']
            unique_count = df[col].nunique()

            if top_pct >= 70:
                insight = f"🔎 The column `{col}` is highly dominated by the category `{top_cat}` which represents {top_pct}% of the data."
            elif top_pct >= 40:
                insight = f"🔍 The category `{top_cat}` appears frequently in `{col}` with {top_pct}% — consider grouping less frequent values."
            elif unique_count > 10:
                insight = f"📌 The column `{col}` has a high number of unique values ({unique_count})."
            else:
                insight = f"🧩 The distribution in `{col}` is relatively balanced."
            st.markdown(f"**Insight:** {insight}")

    
    with st.expander("📈 Numerical Column Analysis"):
        st.write("This section analyzes numerical columns, showing statistical summaries and frequency distribution charts.")

        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns[1:]

        for col in numerical_cols:
            st.markdown(f"### 🔢 Column: `{col}`")

            col_data = df[col].dropna()
            mean_val = round(col_data.mean(), 2)
            median_val = round(col_data.median(), 2)
            mode_val = round(col_data.mode()[0], 2)
            std_val = round(col_data.std(), 2)

            st.markdown(f"""
            - **Mean**: {mean_val}  
            - **Median**: {median_val}  
            - **Mode**: {mode_val}  
            - **Std Dev**: {std_val}  
            - **Missing Values**: {df[col].isnull().sum()}
            """)

            fig = px.histogram(df, x=col, nbins=30, title=f"Distribution of `{col}`", color_discrete_sequence=['#00b894'])
            fig.update_layout(xaxis_title=col, yaxis_title='Frequency', plot_bgcolor='white', paper_bgcolor='white')
            st.plotly_chart(fig, use_container_width=True)

            if abs(mean_val - median_val) > std_val * 0.5:
                insight = f"⚠️ The column `{col}` seems **skewed** since the mean and median differ significantly."
            elif df[col].nunique() < 10:
                insight = f"ℹ️ The column `{col}` has only {df[col].nunique()} unique values. It might be categorical."
            else:
                insight = f"✅ The column `{col}` appears to have a roughly symmetrical distribution."
            st.markdown(f"**Insight:** {insight}")

   
    
        
     


 



        

        
        
        


                        
