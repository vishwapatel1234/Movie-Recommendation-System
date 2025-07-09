import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# Sample movie data
@st.cache_data
def load_sample_data():
    movies_data = {
        'movie_id': range(1, 51),
        'title': [
            'The Shawshank Redemption', 'The Godfather', 'The Dark Knight', 'Pulp Fiction',
            'The Lord of the Rings: The Return of the King', 'Forrest Gump', 'Fight Club',
            'Inception', 'The Lord of the Rings: The Fellowship of the Ring', 'The Empire Strikes Back',
            'The Matrix', 'Goodfellas', 'One Flew Over the Cuckoo\'s Nest', 'Seven', 'Life is Beautiful',
            'The Usual Suspects', 'Léon: The Professional', 'Spirited Away', 'Saving Private Ryan',
            'The Green Mile', 'Interstellar', 'Casablanca', 'City of God', 'The Pianist',
            'The Departed', 'Gladiator', 'The Prestige', 'Whiplash', 'Parasite', 'The Lion King',
            'Terminator 2: Judgment Day', 'Back to the Future', 'Modern Times', 'Psycho',
            'The Green Book', 'Coco', 'Toy Story', 'Avengers: Endgame', 'Spider-Man: Into the Spider-Verse',
            'Joker', 'Django Unchained', 'The Wolf of Wall Street', 'Mad Max: Fury Road',
            'Blade Runner 2049', 'La La Land', 'The Social Network', 'Her', 'Ex Machina',
            'Get Out', 'Moonlight'
        ],
        'genre': [
            'Drama', 'Crime|Drama', 'Action|Crime|Drama', 'Crime|Drama',
            'Adventure|Drama|Fantasy', 'Drama|Romance', 'Drama',
            'Action|Sci-Fi|Thriller', 'Adventure|Drama|Fantasy', 'Action|Adventure|Fantasy',
            'Action|Sci-Fi', 'Crime|Drama', 'Drama', 'Crime|Drama|Mystery', 'Comedy|Drama|Romance',
            'Crime|Mystery|Thriller', 'Action|Crime|Drama', 'Adventure|Family|Supernatural',
            'Drama|War', 'Crime|Drama|Fantasy', 'Adventure|Drama|Sci-Fi', 'Drama|Romance|War',
            'Crime|Drama', 'Biography|Drama|Music', 'Comedy|Drama|Thriller', 'Action|Adventure|Drama',
            'Drama|Mystery|Sci-Fi', 'Drama|Music', 'Comedy|Drama|Thriller', 'Adventure|Drama|Family',
            'Action|Sci-Fi|Thriller', 'Adventure|Comedy|Sci-Fi', 'Comedy|Drama|Romance',
            'Horror|Mystery|Thriller', 'Biography|Comedy|Drama', 'Adventure|Comedy|Family',
            'Adventure|Comedy|Family', 'Action|Adventure|Drama', 'Adventure|Comedy|Family',
            'Crime|Drama|Thriller', 'Drama|Western', 'Biography|Comedy|Crime', 'Action|Adventure|Sci-Fi',
            'Drama|Mystery|Sci-Fi', 'Comedy|Drama|Music', 'Biography|Drama', 'Drama|Romance|Sci-Fi',
            'Horror|Mystery|Thriller', 'Horror|Mystery|Thriller', 'Drama'
        ],
        'year': [
            1994, 1972, 2008, 1994, 2003, 1994, 1999, 2010, 2001, 1980,
            1999, 1990, 1975, 1995, 1997, 1995, 1994, 2001, 1998, 1999,
            2014, 1942, 2002, 2002, 2006, 2000, 2006, 2014, 2019, 1994,
            1991, 1985, 1936, 1960, 2018, 2017, 1995, 2019, 2018, 2019,
            2012, 2013, 2015, 2017, 2016, 2010, 2013, 2014, 2017, 2016
        ],
        'rating': [
            9.3, 9.2, 9.0, 8.9, 8.9, 8.8, 8.8, 8.7, 8.8, 8.7,
            8.7, 8.7, 8.7, 8.6, 8.6, 8.5, 8.5, 8.6, 8.6, 8.6,
            8.6, 8.5, 8.6, 8.5, 8.5, 8.5, 8.5, 8.6, 8.5, 8.5,
            8.5, 8.5, 8.5, 8.5, 8.2, 8.4, 8.3, 8.4, 8.4, 8.4,
            8.4, 8.2, 8.1, 8.0, 8.0, 7.9, 7.8, 7.7, 7.9, 7.4
        ],
        'description': [
            'Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.',
            'The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son.',
            'When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests.',
            'The lives of two mob hitmen, a boxer, a gangster and his wife intertwine in four tales of violence and redemption.',
            'Gandalf and Aragorn lead the World of Men against Sauron\'s army to draw his gaze from Frodo and Sam as they approach Mount Doom.',
            'The presidencies of Kennedy and Johnson, the events of Vietnam, Watergate and other historical events unfold from the perspective of an Alabama man.',
            'An insomniac office worker and a devil-may-care soapmaker form an underground fight club that evolves into something much more.',
            'A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea.',
            'A meek Hobbit from the Shire and eight companions set out on a journey to destroy the powerful One Ring and save Middle-earth.',
            'After the Rebels are brutally overpowered by the Empire on the ice planet Hoth, Luke Skywalker begins Jedi training with Yoda.',
            'A computer programmer is led to fight an underground war against powerful computers who have constructed his entire reality.',
            'The story of Henry Hill and his life in the mob, covering his relationship with his wife Karen Hill and his mob partners.',
            'A criminal pleads insanity and is admitted to a mental institution, where he rebels against the oppressive nurse and rallies up the scared patients.',
            'Two detectives, a rookie and a veteran, hunt a serial killer who uses the seven deadly sins as his motives.',
            'A beautiful, pure-hearted young woman, Malèna\'s life is turned upside down when her country is torn apart by war.',
            'A sole survivor tells of the twisty events leading up to a horrific gun battle on a boat, which began when five criminals met.',
            'Mathilda, a 12-year-old girl, is reluctantly taken in by Léon, a professional assassin, after her family is murdered.',
            'During her family\'s move to the suburbs, a sullen 10-year-old girl wanders into a world ruled by gods, witches, and spirits.',
            'Following the Normandy Landings, a group of U.S. soldiers go behind enemy lines to retrieve a paratrooper whose brothers have been killed.',
            'The lives of guards on Death Row are affected by one of their charges: a black man accused of child murder and rape, yet who has a mysterious gift.',
            'A team of explorers travel through a wormhole in space in an attempt to ensure humanity\'s survival.',
            'A cynical expatriate American cafe owner struggles to decide whether or not to help his former lover and her fugitive husband.',
            'In the slums of Rio, two kids\' paths diverge as one struggles to become a photographer and the other a kingpin.',
            'A Polish Jewish musician struggles to survive the destruction of the Warsaw ghetto of World War II.',
            'An undercover cop and a mob insider try to identify each other while infiltrating an Irish gang in South Boston.',
            'A former Roman General sets out to exact vengeance against the corrupt emperor who murdered his family and sent him into slavery.',
            'After a tragic accident, two stage magicians engage in a battle to create the ultimate illusion while sacrificing everything they have.',
            'A promising young drummer enrolls at a cut-throat music conservatory where his dreams of greatness are mentored by an instructor.',
            'Greed and class discrimination threaten the newly formed symbiotic relationship between the wealthy Park family and the destitute Kim clan.',
            'Lion prince Simba and his father are targeted by his bitter uncle, who wants to ascend the throne himself.',
            'A cyborg, identical to the one who failed to kill Sarah Connor, must now protect her teenage son John Connor from a more advanced cyborg.',
            'Marty McFly, a 17-year-old high school student, is accidentally sent thirty years into the past in a time-traveling DeLorean.',
            'The Tramp struggles to live in modern industrial society with the help of a young homeless woman.',
            'A Phoenix secretary embezzles $40,000 from her employer\'s client, goes on the run, and checks into a remote motel.',
            'A working-class Italian-American bouncer becomes the driver of an African-American classical pianist on a tour of venues.',
            'Aspiring musician Miguel, confronted with his family\'s ancestral ban on music, enters the Land of the Dead to find his great-great-grandfather.',
            'A cowboy doll is profoundly threatened and jealous when a new spaceman figure supplants him as top toy in a boy\'s room.',
            'After the devastating events of Avengers: Infinity War, the universe is in ruins due to the efforts of the Mad Titan, Thanos.',
            'Teen Miles Morales becomes the Spider-Man of his universe, and must join with five spider-powered individuals from other dimensions.',
            'In Gotham City, mentally troubled comedian Arthur Fleck is disregarded and mistreated by society.',
            'With the help of a German bounty-hunter, a freed slave sets out to rescue his wife from a brutal plantation owner in Mississippi.',
            'Based on the true story of Jordan Belfort, from his rise to a wealthy stock-broker living the high life to his fall involving crime.',
            'In a post-apocalyptic wasteland, a woman rebels against a tyrannical ruler in search for her homeland with the aid of a group of female prisoners.',
            'Young Blade Runner K\'s discovery of a long-buried secret leads him to track down former Blade Runner Rick Deckard.',
            'While navigating their careers in Los Angeles, a pianist and an actress fall in love while attempting to reconcile their aspirations.',
            'A young programmer is selected to participate in a ground-breaking experiment in synthetic intelligence by evaluating the human qualities.',
            'In the near future, a lonely writer develops an unlikely relationship with an operating system designed to meet his every need.',
            'A young African-American visits his white girlfriend\'s parents for the weekend, where his simmering uneasiness becomes a nightmare.',
            'A chronicle of the childhood, adolescence and burgeoning adulthood of a young, African-American, gay man growing up in a rough neighborhood.',
            'A black man is chosen to join a secret society of black men who police and monitor extraterrestrial activity on Earth.'
        ]
    }
    
    # Sample user ratings data
    np.random.seed(42)
    user_ratings = []
    for user_id in range(1, 101):  # 100 users
        num_ratings = np.random.randint(5, 25)  # Each user rates 5-25 movies
        movies_rated = np.random.choice(range(1, 51), size=num_ratings, replace=False)
        for movie_id in movies_rated:
            rating = np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.1, 0.2, 0.3, 0.3])
            user_ratings.append({
                'user_id': user_id,
                'movie_id': movie_id,
                'rating': rating
            })
    
    movies_df = pd.DataFrame(movies_data)
    ratings_df = pd.DataFrame(user_ratings)
    
    return movies_df, ratings_df

class MovieRecommender:
    def __init__(self, movies_df, ratings_df):
        self.movies_df = movies_df
        self.ratings_df = ratings_df
        self.user_movie_matrix = None
        self.movie_similarity_matrix = None
        self.tfidf_matrix = None
        self.content_similarity_matrix = None
        
    def prepare_data(self):
        # Create user-movie matrix for collaborative filtering
        self.user_movie_matrix = self.ratings_df.pivot_table(
            index='user_id', 
            columns='movie_id', 
            values='rating'
        ).fillna(0)
        
        # Prepare content-based filtering
        self.movies_df['combined_features'] = (
            self.movies_df['genre'].fillna('') + ' ' + 
            self.movies_df['description'].fillna('')
        )
        
        # TF-IDF Vectorization
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        self.tfidf_matrix = tfidf.fit_transform(self.movies_df['combined_features'])
        
        # Content similarity matrix
        self.content_similarity_matrix = cosine_similarity(self.tfidf_matrix)
        
    def collaborative_filtering_recommendations(self, user_id, n_recommendations=5):
        if user_id not in self.user_movie_matrix.index:
            return self.get_popular_movies(n_recommendations)
        
        # Get user's ratings
        user_ratings = self.user_movie_matrix.loc[user_id]
        
        # Find similar users using cosine similarity
        user_similarity = cosine_similarity([user_ratings], self.user_movie_matrix)[0]
        similar_users = np.argsort(user_similarity)[::-1][1:11]  # Top 10 similar users
        
        # Get recommendations based on similar users
        recommendations = {}
        for similar_user in similar_users:
            similar_user_ratings = self.user_movie_matrix.iloc[similar_user]
            similarity_score = user_similarity[similar_user]
            
            for movie_id, rating in similar_user_ratings.items():
                if rating > 0 and user_ratings[movie_id] == 0:  # Movie not rated by user
                    if movie_id not in recommendations:
                        recommendations[movie_id] = 0
                    recommendations[movie_id] += rating * similarity_score
        
        # Sort and get top recommendations
        top_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
        
        recommended_movies = []
        for movie_id, score in top_recommendations:
            movie_info = self.movies_df[self.movies_df['movie_id'] == movie_id].iloc[0]
            recommended_movies.append({
                'title': movie_info['title'],
                'genre': movie_info['genre'],
                'year': movie_info['year'],
                'rating': movie_info['rating'],
                'score': score
            })
        
        return recommended_movies
    
    def content_based_recommendations(self, movie_titles, n_recommendations=5):
        recommendations = {}
        
        for movie_title in movie_titles:
            movie_idx = self.movies_df[self.movies_df['title'] == movie_title].index
            if len(movie_idx) == 0:
                continue
                
            movie_idx = movie_idx[0]
            similarity_scores = self.content_similarity_matrix[movie_idx]
            
            # Get top similar movies
            similar_movies = np.argsort(similarity_scores)[::-1][1:n_recommendations+5]
            
            for idx in similar_movies:
                movie_id = self.movies_df.iloc[idx]['movie_id']
                if movie_id not in recommendations:
                    recommendations[movie_id] = 0
                recommendations[movie_id] += similarity_scores[idx]
        
        # Sort and get top recommendations
        top_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
        
        recommended_movies = []
        for movie_id, score in top_recommendations:
            movie_info = self.movies_df[self.movies_df['movie_id'] == movie_id].iloc[0]
            recommended_movies.append({
                'title': movie_info['title'],
                'genre': movie_info['genre'],
                'year': movie_info['year'],
                'rating': movie_info['rating'],
                'score': score
            })
        
        return recommended_movies
    
    def genre_based_recommendations(self, preferred_genres, n_recommendations=5):
        filtered_movies = self.movies_df[
            self.movies_df['genre'].str.contains('|'.join(preferred_genres), case=False, na=False)
        ].sort_values('rating', ascending=False)
        
        recommendations = []
        for _, movie in filtered_movies.head(n_recommendations).iterrows():
            recommendations.append({
                'title': movie['title'],
                'genre': movie['genre'],
                'year': movie['year'],
                'rating': movie['rating'],
                'score': movie['rating']
            })
        
        return recommendations
    
    def get_popular_movies(self, n_recommendations=5):
        popular_movies = self.movies_df.sort_values('rating', ascending=False).head(n_recommendations)
        
        recommendations = []
        for _, movie in popular_movies.iterrows():
            recommendations.append({
                'title': movie['title'],
                'genre': movie['genre'],
                'year': movie['year'],
                'rating': movie['rating'],
                'score': movie['rating']
            })
        
        return recommendations

# Main Streamlit App
def main():
    st.title("🎬 Movie Recommendation System")
    st.markdown("Discover your next favorite movie with personalized recommendations!")
    
    # Load data
    movies_df, ratings_df = load_sample_data()
    
    # Initialize recommender
    recommender = MovieRecommender(movies_df, ratings_df)
    recommender.prepare_data()
    
    # Sidebar
    st.sidebar.header("Recommendation Options")
    recommendation_type = st.sidebar.selectbox(
        "Choose Recommendation Type:",
        ["Collaborative Filtering", "Content-Based", "Genre-Based", "Popular Movies"]
    )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if recommendation_type == "Collaborative Filtering":
            st.subheader("👥 Collaborative Filtering")
            st.write("Get recommendations based on users with similar tastes")
            
            user_id = st.number_input("Enter User ID (1-100):", min_value=1, max_value=100, value=1)
            n_recs = st.slider("Number of recommendations:", 1, 10, 5)
            
            if st.button("Get Recommendations"):
                recommendations = recommender.collaborative_filtering_recommendations(user_id, n_recs)
                
                if recommendations:
                    for i, movie in enumerate(recommendations, 1):
                        with st.expander(f"{i}. {movie['title']} ({movie['year']})"):
                            st.write(f"**Genre:** {movie['genre']}")
                            st.write(f"**Rating:** {movie['rating']}/10")
                            st.write(f"**Similarity Score:** {movie['score']:.3f}")
                else:
                    st.write("No recommendations found. Try a different user ID.")
        
        elif recommendation_type == "Content-Based":
            st.subheader("📝 Content-Based Filtering")
            st.write("Get recommendations based on movie content similarity")
            
            selected_movies = st.multiselect(
                "Select movies you like:",
                options=movies_df['title'].tolist(),
                default=["The Matrix", "Inception"]
            )
            
            n_recs = st.slider("Number of recommendations:", 1, 10, 5)
            
            if st.button("Get Recommendations") and selected_movies:
                recommendations = recommender.content_based_recommendations(selected_movies, n_recs)
                
                if recommendations:
                    for i, movie in enumerate(recommendations, 1):
                        with st.expander(f"{i}. {movie['title']} ({movie['year']})"):
                            st.write(f"**Genre:** {movie['genre']}")
                            st.write(f"**Rating:** {movie['rating']}/10")
                            st.write(f"**Similarity Score:** {movie['score']:.3f}")
                else:
                    st.write("No recommendations found.")
        
        elif recommendation_type == "Genre-Based":
            st.subheader("🎭 Genre-Based Recommendations")
            st.write("Get recommendations based on your preferred genres")
            
            all_genres = set()
            for genres in movies_df['genre'].str.split('|'):
                all_genres.update(genres)
            
            selected_genres = st.multiselect(
                "Select preferred genres:",
                options=sorted(list(all_genres)),
                default=["Action", "Sci-Fi"]
            )
            
            n_recs = st.slider("Number of recommendations:", 1, 10, 5)
            
            if st.button("Get Recommendations") and selected_genres:
                recommendations = recommender.genre_based_recommendations(selected_genres, n_recs)
                
                if recommendations:
                    for i, movie in enumerate(recommendations, 1):
                        with st.expander(f"{i}. {movie['title']} ({movie['year']})"):
                            st.write(f"**Genre:** {movie['genre']}")
                            st.write(f"**Rating:** {movie['rating']}/10")
                else:
                    st.write("No recommendations found for selected genres.")
        
        else:  # Popular Movies
            st.subheader("⭐ Popular Movies")
            st.write("Top-rated movies everyone should watch")
            
            n_recs = st.slider("Number of recommendations:", 1, 10, 5)
            
            if st.button("Get Recommendations"):
                recommendations = recommender.get_popular_movies(n_recs)
                
                for i, movie in enumerate(recommendations, 1):
                    with st.expander(f"{i}. {movie['title']} ({movie['year']})"):
                        st.write(f"**Genre:** {movie['genre']}")
                        st.write(f"**Rating:** {movie['rating']}/10")
    
    with col2:
        st.subheader("📊 Dataset Overview")
        st.metric("Total Movies", len(movies_df))
        st.metric("Total Users", len(ratings_df['user_id'].unique()))
        st.metric("Total Ratings", len(ratings_df))
        
        # Genre distribution
        all_genres = []
        for genres in movies_df['genre'].str.split('|'):
            all_genres.extend(genres)
        
        genre_counts = pd.Series(all_genres).value_counts().head(10)
        
        fig = px.bar(
            x=genre_counts.index,
            y=genre_counts.values,
            title="Top 10 Genres",
            labels={'x': 'Genre', 'y': 'Count'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Rating distribution
        fig2 = px.histogram(
            ratings_df,
            x='rating',
            title="Rating Distribution",
            labels={'rating': 'Rating', 'count': 'Count'}
        )
        fig2.update_layout(height=300)
        st.plotly_chart(fig2, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("**How it works:**")
    st.markdown("• **Collaborative Filtering:** Recommends movies based on users with similar preferences")
    st.markdown("• **Content-Based:** Recommends movies similar to ones you already like")
    st.markdown("• **Genre-Based:** Recommends top-rated movies in your preferred genres")
    st.markdown("• **Popular Movies:** Shows highest-rated movies overall")

if __name__ == "__main__":
    main()