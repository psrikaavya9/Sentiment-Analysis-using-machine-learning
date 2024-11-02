
from flask import Flask, render_template, request, jsonify
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from nltk.sentiment import SentimentIntensityAnalyzer
from googletrans import Translator
import threading

app = Flask(__name__, template_folder='templ')
lock = threading.Lock()

def load_reviews(file):
    """Load and decode reviews from a file"""
    reviews = file.read().decode('utf-8').splitlines()
    return reviews

def translate_to_english(reviews):
    """Translate reviews to English using Google Translate"""
    translator = Translator()
    english_reviews = []

    if not isinstance(reviews, list):
        reviews = [reviews]

    for review in reviews:
        try:
            translation = translator.translate(review)
            if translation.text:
                english_reviews.append(translation.text)
            else:
                english_reviews.append(review)
        except Exception as e:
            print(f"Translation error: {e}")
            english_reviews.append(review)

    return english_reviews

def analyze_sentiments(review):
    """Analyze sentiment of a single review"""
    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(review)["compound"]
    return sentiment_score

def classify_review(sentiment_score):
    """Classify sentiment score as positive, negative, or neutral"""
    if sentiment_score > 0:
        return 'positive'
    elif sentiment_score < 0:
        return 'negative'
    else:
        return 'neutral'

def generate_plot(sentiment_counts):
    """Generate a pie chart for sentiment distribution"""
    plt.figure()
    labels = list(sentiment_counts.keys())
    sizes = list(sentiment_counts.values())

    colors = ['gold', 'lightcoral', 'lightskyblue']

    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    plt.title('Sentiment Distribution (Predicted)')

    img_base64 = plot_to_base64(plt.gcf())

    return img_base64

def analyze_and_plot(files=None, text=None):
    """Analyze and plot sentiment data for multiple files or text"""
    sentiment_counts = {
        'positive': 0,
        'neutral': 0,
        'negative': 0
    }

    if files:
        # Process each uploaded file
        for file in files:
            reviews = load_reviews(file)
            if not reviews or all(not review.strip() for review in reviews):
                continue

            # Analyze each review
            for review in reviews:
                stripped_review = review.strip()
                if stripped_review:
                    sentiment_score = analyze_sentiments(stripped_review)
                    classification = classify_review(sentiment_score)
                    sentiment_counts[classification] += 1

    if text:
        # Analyze the provided text
        stripped_text = text.strip()
        if stripped_text:
            sentiment_score = analyze_sentiments(stripped_text)
            classification = classify_review(sentiment_score)
            sentiment_counts[classification] += 1

    # Generate and return pie chart
    img_base64 = generate_plot(sentiment_counts)

    return jsonify({'image': img_base64, 'sentiment_counts': sentiment_counts})

def plot_to_base64(figure):
    """Convert a matplotlib plot to base64 string"""
    buf = BytesIO()
    figure.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(figure)
    return img_base64

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Handle the sentiment analysis request"""
    files = request.files.getlist('files')  # Get list of all uploaded files
    text = request.form.get('text')  # Get text from form

    if not files and not text:
        return jsonify({'error': 'No files or text provided'})

    return analyze_and_plot(files, text)

if __name__ == '__main__':
    app.run(debug=True)