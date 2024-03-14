import os
import fitz
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import LatentDirichletAllocation
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from gap_statistic import OptimalK
import numpy as np

custom_stop_words = {'el', 'at', 'https', 'et', 'al', '10'}


# Step 1: Extract text from PDFs using PyMuPDF
def extract_text_from_pdf(pdf_path):
    text = ''
    doc = fitz.open(pdf_path)
    for page_num in range(doc.page_count):
        page = doc[page_num]
        text += page.get_text()
    return text


# Step 2: Calculate Text Similarity
def calculate_similarity(texts, stop_words=None):
    if stop_words is None:
        stop_words = set()

    vectorizer = TfidfVectorizer(stop_words=list(stop_words))
    tfidf_matrix = vectorizer.fit_transform(texts)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    return similarity_matrix, vectorizer.get_feature_names_out(), vectorizer


# Step 3: Determine Optimal Number of Clusters
def determine_optimal_clusters(similarity_matrix):
    distortions = []
    K_range = range(2, 11)  # Adjust the range based on your needs

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(similarity_matrix)
        distortions.append(kmeans.inertia_)

    # Check if distortions list is empty
    if not distortions:
        print("Error: Distortions list is empty.")
        return None

    # Convert distortions list to numpy array
    distortions_np = np.array(distortions).reshape(-1, 1)

    # Use optimal_k from gap_statistic
    optimal_k_instance = OptimalK(parallel_backend='multiprocessing')
    try:
        elbow_index, _, _ = optimal_k_instance(distortions_np)
    except ValueError as ve:
        print(f"Error: {ve}")
        return None

    # Plot the elbow graph
    plt.figure(figsize=(8, 6))
    plt.plot(K_range, distortions, marker='o')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Distortion')
    plt.show()

    return elbow_index


# Step 4: Perform Clustering
def perform_clustering(similarity_matrix, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(similarity_matrix)
    return clusters


# Step 5: Visualize Clusters with Plotly
def visualize_clusters_3d(file_names, clusters, texts, feature_names, vectorizer, similarity_matrix):
    # Apply t-SNE to reduce dimensionality to 3D
    tsne = TSNE(n_components=3, random_state=42)
    tsne_results = tsne.fit_transform(similarity_matrix)

    # Create a DataFrame for Plotly
    df = pd.DataFrame(tsne_results, columns=['TSNE1', 'TSNE2', 'TSNE3'])
    df['Cluster'] = clusters
    df['File'] = file_names

    # Get representative words for each cluster
    representative_words = get_representative_words_for_clusters(texts, clusters, feature_names, vectorizer)

    # Calculate Silhouette Score
    silhouette_avg = silhouette_score(similarity_matrix, clusters)

    # Add representative words to hover information
    df['Representative Words'] = [', '.join(representative_words[cluster_num]) for cluster_num in clusters]

    # Visualize in 3D
    fig = px.scatter_3d(df, x='TSNE1', y='TSNE2', z='TSNE3', color='Cluster',
                        hover_data=['File', 'Cluster', 'Representative Words'],
                        title="Document Clustering in 3D using t-SNE",
                        labels={'Cluster': 'Cluster'})

    # Add annotation for silhouette score at the bottom right
    fig.add_annotation(
        x=1,
        y=-0.1,
        text=f'Silhouette Score: {silhouette_avg:.4f}',
        showarrow=False,
        font=dict(size=12, color='black'),
        xref="paper",
        yref="paper",
        xanchor="right",
        yanchor="bottom"
    )

    fig.show()


def get_representative_words_for_clusters(texts, clusters, feature_names, vectorizer):
    stop_words = set(ENGLISH_STOP_WORDS).union(custom_stop_words)
    cluster_words = {}  # Define the dictionary outside the loop
    print(f"{stop_words}")

    for cluster_num in set(clusters):
        cluster_indices = [i for i, c in enumerate(clusters) if c == cluster_num]
        cluster_texts = [remove_stop_words(text, stop_words) for text in [texts[i] for i in cluster_indices]]
        cluster_tfidf_matrix = vectorizer.transform(cluster_texts)
        cluster_avg_tfidf = cluster_tfidf_matrix.mean(axis=0)
        cluster_avg_tfidf = cluster_avg_tfidf.A1  # Convert to 1D array
        top_feature_indices = cluster_avg_tfidf.argsort()[-5:][::-1]  # Get top 5 feature indices

        top_words = [feature_names[i] for i in top_feature_indices]

        cluster_words[cluster_num] = top_words

        # Print debug information
        print(f"Cluster {cluster_num}: {top_words}")

    return cluster_words


def remove_stop_words(text, stop_words):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)


def generate_thematic_summary(texts, clusters, num_topics=3, num_top_words=5):
    # Use CountVectorizer to get the document-term matrix
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    dtm = vectorizer.fit_transform(texts)

    # Fit LDA model
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(dtm)

    # Get the topic distribution for each document
    topic_distributions = lda.transform(dtm)

    # Create a DataFrame to store topics for each document
    df_topics = pd.DataFrame(topic_distributions, columns=[f'Topic_{i}' for i in range(num_topics)])

    # Add cluster information
    df_topics['Cluster'] = clusters

    # Create a summary for each cluster based on the dominant topic
    cluster_summaries = {}
    for cluster_num in set(clusters):
        cluster_data = df_topics[df_topics['Cluster'] == cluster_num]
        dominant_topic = cluster_data.iloc[:, :-1].idxmax(axis=1).mode()[0]
        top_words = get_top_words_for_topic(lda, vectorizer.get_feature_names_out(), int(dominant_topic.split('_')[1]), num_top_words, stop_words='english')
        summary = f"Cluster {cluster_num} Summary:\n"
        summary += f"Dominant Topic: {dominant_topic}\n"
        summary += f"Top Words: {', '.join(top_words)}\n"
        summary += f"Number of Documents: {len(cluster_data)}\n"
        summary += f"Example Documents:\n"
        example_documents = cluster_data['Cluster'].sample(min(3, len(cluster_data)))
        for i, index in enumerate(example_documents.index):
            summary += f"Document {i + 1}:\n{texts[index]}\n"
        cluster_summaries[cluster_num] = summary

    return cluster_summaries


def get_top_words_for_topic(lda, feature_names, topic, num_words=5, stop_words=None):
    if stop_words is None:
        stop_words = set()

    topic_weights = lda.components_[topic]
    top_word_indices = topic_weights.argsort()[-num_words:][::-1]
    top_words = [feature_names[i] for i in top_word_indices if feature_names[i] not in stop_words]

    return top_words


if __name__ == "__main__":
    # Specify the path to your PDF files
    folder_path = "ResearchPapers"

    # Use os.listdir to get a list of PDF files in the folder
    pdf_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pdf')]

    # Step 1: Extract text from PDFs using PyMuPDF
    texts = [extract_text_from_pdf(pdf_file) for pdf_file in pdf_files]

    # Step 2: Calculate Text Similarity
    stop_words = TfidfVectorizer().get_stop_words()  # Use TfidfVectorizer's stop words
    similarity_matrix, feature_names, vectorizer = calculate_similarity(texts, stop_words)

    # Step 3: Determine Optimal Number of Clusters using the Elbow Method
    elbow_index = determine_optimal_clusters(similarity_matrix)

    if elbow_index is not None:
        print(f"Optimal number of clusters based on elbow method: {elbow_index}")

    # Choose the number of clusters based on your preference or a combination of methods
    optimal_num_clusters = int(5)
    clusters = perform_clustering(similarity_matrix, optimal_num_clusters)

    # Step 5: Visualize Clusters with Plotly in 3D using t-SNE
    visualize_clusters_3d(pdf_files, clusters, texts, feature_names, vectorizer, similarity_matrix)

    # Generate thematic summaries for each cluster
    cluster_summaries = generate_thematic_summary(texts, clusters, num_topics=3, num_top_words=5)

    # Print cluster summaries
    for cluster_num, summary in cluster_summaries.items():
        print(summary)
