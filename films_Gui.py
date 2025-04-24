import openai
import pickle
import streamlit as st
import re
import string
import numpy as np
import pandas as pd 
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
openai.api_key = "mk-proj-EbLF1eqg5pEJ1or_K6y1P8PCfF_nMIizBxz-J63BVyziKvkOuevEx-rA7hcOtQ3vEl5N_1UtNNT3BlbkFJ6N12YLEUBkakQ6woGQjylFTxedSgKhRPz6I4tgAZrVg5IM-N0wjMsiXwQrOJKobqKVmuar63YA"
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

st.image(r"Screenshot 2025-04-17 141645.png")
st.title('Series | Movies App 🎥')
st.image(r"d0fac8df-f671-45fe-b21a-0274b4d19e98.png")
df = pd.read_excel(r"Data_films_cleaned.xlsx")
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

select = st.sidebar.selectbox('Choose Option', ['None','Describtion Of Data','Sample Analysis','Sentiment Analysis','Machine Learning', 'Deep Learning','Summary Dashboard'])

if select=='None':
    st.image(r"ChatGPT Image Apr 16, 2025, 10_31_31 AM.png")
elif select=='Describtion Of Data':
    st.sidebar.image(r"886ef944-5bc6-4b76-b532-5d22bd9404f7.png")
    tabs=st.tabs(['Data Overview','Data Types','Statistics','Chi-Squared-Test'])
    with tabs[0]:
        st.dataframe(df.iloc[1:,:])
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

# Logos for each platform
        logos = {
    'Both or All of these': "https://img.icons8.com/emoji/48/clapper-board-emoji.png",  
    'Home Theaters': "https://img.icons8.com/fluency/48/movie-projector.png",             
    'Streaming Services': "https://img.icons8.com/fluency/48/netflix-desktop-app.png",  
    'Theater': "https://img.icons8.com/fluency/48/movie-projector.png"                 
}

        # Convert to lists
        platforms = count_view.index.tolist()
        values = count_view.values.tolist()

        # Initialize the figure
        fig = go.Figure()

        # Add the horizontal bars
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

        # Add logos next to each bar
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

        # Layout and styling
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

        # Display in Streamlit
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f'the Most perefered 2 methods is ')
        list_opt=count_view.index[0:2].values
        for i in list_opt:
            st.success(f'{i}')
    with tabs[1]:
        perfered_content_type = df['Preferred Content Type'].value_counts()

# Reverse the order for most frequent on top
        content_types = perfered_content_type.index.tolist()[::-1]
        counts = perfered_content_type.values.tolist()[::-1]

        # Initialize figure
        fig = go.Figure()

        # Draw lollipop sticks
        fig.add_trace(go.Scatter(
            x=counts,
            y=content_types,
            mode='lines',
            line=dict(color='lightgray', width=2),
            showlegend=False
        ))

        # Draw lollipop heads
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

        # Annotate the most preferred type (on top now)
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

        # Update layout
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

        # Extract labels and values
        labels = count_gradual.index.tolist()
        values = count_gradual.values.tolist()

        # Create polar (radial bar) chart
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

        # Layout customization
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

        # Create figure
        fig = go.Figure()

        # Bars (bullet style)
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

        # Add dot marker at the end of each bar for sleek look
        fig.add_trace(go.Scatter(
            x=values,
            y=categories,
            mode='markers',
            marker=dict(color='#0984e3', size=12),
            showlegend=False
        ))

        # Layout styling
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

# Prepare values and labels
        labels = review_counts.index.tolist()
        values = review_counts.values.tolist()

        # Create half-donut (semi-circle) with Pie chart + rotation
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.5,
            direction='clockwise',
            sort=False,
            textinfo='percent+label',
            marker=dict(colors=['#00b894', '#d63031'], line=dict(color='white', width=2)),
        )])

        # Only show half circle
        fig.update_traces(rotation=180, pull=[0.05]*len(labels))

        # Layout settings
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

        # Start building subplots with one gauge per unique value
        fig = go.Figure()

        # Color palette (can expand if more than 3)
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

        # Adjust layout: make grid based on how many items
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

# Ensure all ratings from 1 to 5 are present (fill missing with 0)
        for i in range(1, 6):
            if i not in rating_counts:
                rating_counts[i] = 0

        rating_counts = rating_counts.sort_index()

        # Step 2: Build custom x and y labels
        ratings = rating_counts.index.tolist()
        counts = rating_counts.values.tolist()
        stars = ['⭐' * r for r in ratings]

        # Step 3: Plot as scatter with star labels
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

        # Step 4: Add value labels below
        for i in range(len(ratings)):
            fig.add_annotation(
                x=ratings[i],
                y=counts[i] - max(counts) * 0.1,
                text=f"{counts[i]} votes",
                showarrow=False,
                font=dict(size=12)
            )

        # Final Layout
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
 
    

        




   

elif select == 'Machine Learning':
    tabs = st.tabs(['Type Prediction According to Events'])
    with tabs[0]:
        st.subheader("🎯 Prediction of Genre Based on Events")
        user_input = st.text_area("✍️ أدخل ملخص الأحداث هنا")

        if st.button('🔍 Submit'):
            if not user_input.strip():
                st.warning("الرجاء إدخال نص لوصف الأحداث.")
            else:
                cleaned_text = preprocess_text(user_input)
                embedding_input = get_embedding(cleaned_text)
                embedding_input = np.array(embedding_input).reshape(1, -1)
                
                prediction = best_estimator_Log.predict(embedding_input)
                st.success(f"✅ النوع المتوقع للفيلم: **{prediction[0]}**")
elif select == 'Deep Learning':
    tabs = st.tabs(['Movie Recommender', 'Analysis Chatbot'])
    def get_movie_poster(movie_name, api='2db5cff7'):
        url = f"http://www.omdbapi.com/?t={movie_name}&apikey={api}"
        response = requests.get(url)
        data = response.json()
        
        if data['Response'] == 'True':
            return data['Poster'], data['Title'], data['Year']
        else:
            return None, None, None


    with tabs[0]:
        st.title("🎥 Movie Recommendation Bot")

        user_input = st.chat_input("اكتب وصف الفيلم اللي نفسك تشوفه...")

        if user_input:
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                with st.spinner("🎬 بنبحث عن أفضل فيلم ليك..."):
                    try:
                        reply,film = recommend_film(user_input, df, 'Series Summary and Highlights', 'Favorite Series')
                        film = film.replace('"', '').replace("'", '')

                        
                        recommended_title = film  # تعديل حسب تنسيق reply

                        poster, title, year = get_movie_poster(recommended_title)

                        if poster:
                            st.image(poster, caption=f"{title} ({year})", use_column_width=True)
                            st.markdown(reply)
                        else:
                            
                            st.warning("⚠️ Poster not found. Please check OMDb or try another title.")

                    except Exception as e:
                        st.error(f"❌ حصل خطأ: {e}")
    with tabs[1]:
        st.image(r"snapedit_1745518403105.jpeg")
        st.title("🎬 ChatBot Analysis")
        
        
        st.caption("Ask natural language questions about your dataset.")
        def build_prompt(question):
            return f"""
        You are a data analysis assistant. You have a pandas DataFrame named df with the following columns:

        ['Unnamed: 0', 'Timestamp', 'Gender', 'Age Range', 'Education Level',
        'Preferred Content Type', 'Reason for Preferring Movies',
        'Reason for Preferring Series', 'Binge or Gradual Watching',
        'Preferred Viewing Style', 'Favorite Genres', 'Influence of Popularity',
        'Importance of Actors', 'Budget Indicates Quality',
        'Main Dislikes in Movies', 'Main Likes in Movies/Series',
        'Favorite Film or Series', 'Favorite Series',
        'Series Summary and Highlights', 'Favorite Series Actors',
        'Ideal Episode Duration', 'Series Nationality',
        'Favorite Aspects of Series', 'Number of Seasons', 'Favorite Movie',
        'Favorite Actor', 'Movie Description', 'Movie Duration',
        'Importance of Release Date', 'Movie Decade', 'Movie Type',
        'Influence of Movie Budget', 'Movie Watching Preference',
        'Watching Alone or With Others', 'Real vs Fiction Preference',
        'Check Reviews Before Watching', 'Additional Comments',
        'Age_classification', 'cleaned_main_dislikes']

        Your job is to write a Python code snippet using pandas to answer user questions about the data. Only return the Python code. If the question is unrelated, respond with: "This question is not related to the dataset."

        Examples:

        Q: Do most people care about movie budget?
        df['Influence of Movie Budget'].value_counts().nlargest(3)

        Q: What is the most content preferred by males?
        df[df['Gender'].str.lower() == 'male']['Preferred Content Type'].value_counts()

        Q: What is the average movie duration per movie type?
        df.groupby('Movie Type')['Movie Duration'].mean()

        Q: What are the most common preferred genres per gender?
        df.groupby('Gender')['Favorite Genres'].apply(lambda x: x.value_counts().head(1))

        Q: Which age range prefers series?
        df[df['Preferred Content Type'] == 'Series']['Age Range'].value_counts()

        Q: What is the most mentioned reason for preferring movies?
        df['Reason for Preferring Movies'].value_counts().head(3)

        Q: How many people binge watch?
        df['Binge or Gradual Watching'].value_counts()

        Q: Average duration of movies liked by people who check reviews?
        df[df['Check Reviews Before Watching'] == 'Yes']['Movie Duration'].mean()

        Q: {question}
        """

        # توليد الكود وتنفيذه وعرض النتيجة
        def run_analysis(question, df):
            prompt = build_prompt(question)
            response = openai.ChatCompletion.create(
                model="chatgpt-4o-latest",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )

            raw_code = response.choices[0].message["content"].strip()

            if "not related" in raw_code.lower():
                return None, "❌ This question is not related to the dataset."

            # إزالة backticks أو 'python' من البداية والنهاية لو فيه
            code = raw_code.strip("` \n")
            if code.lower().startswith("python"):
                code = code[6:].strip()

            try:
                local_vars = {'df': df}

                if "\n" not in code and " = " not in code:
                    # إذا كان سطر واحد وبسيط زي value_counts()
                    result = eval(code, {}, local_vars)
                else:
                    # كود معقد أو متعدد الأسطر – نستخدم exec
                    exec(code, {}, local_vars)

                    # نطلع آخر قيمة مفيدة نعرضها
                    result = None
                    for val in reversed(local_vars.values()):
                        if isinstance(val, (pd.Series, pd.DataFrame, int, float, str)):
                            result = val
                            break

                return code, result

            except Exception as e:
                return code, f"⚠️ Error in executing code:\n\n{e}"

        # واجهة المستخدم
        question = st.text_input("💬 Ask a question about your data")

        if question:
            with st.spinner("🤖 Thinking..."):
                code, output = run_analysis(question, df)

                if code:
                    st.subheader("🧾 Generated Code")
                    st.code(code, language="python")

                st.subheader("📊 Result")
                st.write(output)


        
        
     
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

            # Store selected reasons
            selected_reasons = []

            # 👉 Horizontally layout the checkboxes
            cols = st.columns(len(reason_categories))
            checkbox_states = {}

            for col, (category, reasons) in zip(cols, reason_categories.items()):
                with col:
                    checkbox_states[category] = st.checkbox(category)

            # Show reasons under selected categories
            for category, selected in checkbox_states.items():
                if selected:
                    st.subheader(f"🔹 {category}")
                    for reason in reason_categories[category]:
                        st.markdown(f"- {reason}")
                    selected_reasons.extend(reason_categories[category])

            # Summary
            if selected_reasons:
                st.markdown("---")
                st.success(f"✅ {len(selected_reasons)} detailed reasons were selected that make people dislike films.")
        with tabs[1]:
            df=df[~(df['Favorite Actor']=='unknown')]
            top_actors = df['Favorite Actor'].value_counts().head(6)

# روابط الصور للممثلين
            actor_images = {
    'Leonardo Dicaprio': 'https://sf2.closermag.fr/wp-content/uploads/closermag/2023/04/Leonardo-DiCaprio-au-66eme-Festival-de-Cannes-le-15-mai-2013-scaled-546x410.jpg',
    'Robert Downey Jr': 'https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/Robert_Downey_Jr_2014_Comic_Con_%28cropped%29.jpg/330px-Robert_Downey_Jr_2014_Comic_Con_%28cropped%29.jpg',
    'Ahmed Helmy': 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/Ahmed_Helmy.jpg/330px-Ahmed_Helmy.jpg',
    'Tom Cruise': 'https://upload.wikimedia.org/wikipedia/commons/thumb/d/d2/Tom_Cruise_avp_2014_4.jpg/330px-Tom_Cruise_avp_2014_4.jpg',
    'Jason Statham': 'https://upload.wikimedia.org/wikipedia/commons/thumb/1/16/Jason_Statham_2018.jpg/330px-Jason_Statham_2018.jpg',
    'Ahmed Ezz': 'https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/Ahmed_Ezz_2020.jpg/330px-Ahmed_Ezz_2020.jpg'
}


            # رسم العمود الرئيسي
            fig = go.Figure(go.Bar(
                x=top_actors.index,
                y=top_actors.values,
                marker_color='#00cec9',
                text=top_actors.values,
                textposition='outside'
            ))

            # إضافة الصور فوق كل عمود
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

            # إعدادات الشكل
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
        tabs=st.tabs(['Reasons for prefering Series','Favourite Aspects Of Series','Number of Seasons'])
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

        # Layout customization
            fig.update_layout(
            title_text="🍿 Why Do People Prefer Series Over Movies?",
            title_font_size=22,
            annotations=[dict(text='Series Preference', x=0.5, y=0.5, font_size=16, showarrow=False)],
            showlegend=True,
            # Removed paper_bgcolor for default/transparent background
            font=dict(color="#333", size=9),
            margin=dict(t=50, b=50, l=100, r=100)
        )

            st.plotly_chart(fig, use_container_width=True)        
        with tabs[1]:     
            aspect_counts = df['Favorite Aspects of Series'].value_counts().reset_index()
            aspect_counts.columns = ['Aspect', 'Count']

            # Bubble chart (scatter with size mapped to count)
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

            # Improve layout
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

            # Custom colors
            colors = ['#00b894'] * len(labels)  # Teal-like

            # Create figure
            fig = go.Figure()

            # Add bar for each season with label on top
            fig.add_trace(go.Bar(
                x=labels,
                y=values,
                text=values,
                textposition='outside',
                marker_color=colors,
                hovertemplate="Season %{x}<br>Count: %{y}<extra></extra>",
                name="Seasons"
            ))

            # Add season numbers as bold annotations on each bar
            for i, (label, count) in enumerate(zip(labels, values)):
                fig.add_annotation(
                    x=label,
                    y=count + max(values)*0.07,
                    text=f"<b>{label}</b>",
                    showarrow=False,
                    font=dict(size=16, color='#2d3436')
                )

            # Layout styling
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
else:
    

   
    st.title("📊 Power BI Report")
    

    st.markdown("[🟢 Open Report in New Tab](https://app.powerbi.com/links/f_R1uWPbKY?ctid=eaf624c8-a0c4-4195-87d2-443e5d7516cd&pbi_source=linkShare)", unsafe_allow_html=True)
    st.image(r"Screenshot 2025-04-17 171050.png")
 



        

        
        
        


                        
