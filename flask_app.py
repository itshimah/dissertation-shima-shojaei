from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
import sqlite3
import joblib
import time
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os
import re
from collections import Counter
import random  # NEW: Added for dynamic product recommendations

# --- DATABASE PATH FIX FOR PYTHONANYWHERE ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_NAME = os.path.join(BASE_DIR, 'users.db')
MODEL_PATH = os.path.join(BASE_DIR, 'ecommerce_personalization_model.pkl')
# --------------------------------------------

nltk.download('vader_lexicon', quiet=True)

app = Flask(__name__)
app.secret_key = 'oxford_brookes_super_secret_key'

# 1. Initialize AI Models
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    print(
        f"WARNING: '{MODEL_PATH}' not found. Ensure it is in the same folder.")
    model = None

sia = SentimentIntensityAnalyzer()

# Extended Stopwords to filter out adverbs, basic verbs, and filler words
STOPWORDS = set([
    'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her', 'it', 'its',
    'they', 'them', 'their', 'what', 'which', 'who', 'this', 'that', 'these', 'those', 'am',
    'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'a',
    'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by',
    'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
    'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
    'again', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both',
    'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
    'same', 'so', 'than', 'too', 'very', 'can', 'will', 'just', 'don', 'should', 'now', 'product',
    'item', 'get', 'got', 'like', 'one', 'would', 'really', 'buy', 'bought',
    'absolutely', 'exactly', 'five', 'day', 'days', 'makes', 'make', 'made', 'much', 'even',
    'also', 'take', 'took', 'always', 'never', 'well', 'going', 'things', 'thing', 'look',
    'looks', 'say', 'said', 'back', 'way', 'still', 'sure', 'want', 'wanted', 'know', 'think',
    'see', 'let', 'come', 'came', 'much', 'many', 'put', 'give', 'gave'
])

# Expanded Product Categories Keywords for Classification
CATEGORY_RULES = {
    "Electronics & Tech": ['phone', 'screen', 'battery', 'charger', 'laptop', 'audio', 'sound', 'bluetooth', 'cable', 'device', 'camera', 'keyboard', 'mouse', 'monitor'],
    "Clothing & Apparel": ['shirt', 'fit', 'size', 'wear', 'fabric', 'shoes', 'dress', 'color', 'comfortable', 'cloth', 'jacket', 'pants', 'jeans', 'cotton'],
    "Home & Kitchen": ['clean', 'room', 'bed', 'kitchen', 'cook', 'wash', 'decor', 'table', 'furniture', 'pan', 'glass', 'knife', 'pillow', 'rug'],
    "Beauty & Cosmetics": ['skin', 'face', 'hair', 'makeup', 'lotion', 'smell', 'scent', 'cream', 'perfume', 'glow', 'soft', 'serum', 'lip', 'eye'],
    "Books & Literature": ['book', 'read', 'page', 'author', 'story', 'novel', 'paper', 'cover', 'chapter', 'reading', 'fiction'],
    "Sports & Outdoors": ['fitness', 'gym', 'workout', 'run', 'ball', 'camp', 'hike', 'bike', 'sport', 'exercise', 'tent', 'sweat', 'muscle'],
    "Toys & Games": ['toy', 'game', 'play', 'kids', 'fun', 'board', 'puzzle', 'action', 'figure', 'console', 'child', 'lego'],
    "Automotive": ['car', 'vehicle', 'tire', 'engine', 'auto', 'drive', 'wash', 'seat', 'oil', 'truck', 'motor'],
    "Health & Wellness": ['vitamin', 'supplement', 'health', 'pill', 'diet', 'weight', 'pain', 'medical', 'mask', 'relief', 'protein'],
    "Food & Groceries": ['taste', 'flavor', 'snack', 'food', 'drink', 'eat', 'sweet', 'fresh', 'delicious', 'coffee', 'tea', 'organic', 'spicy']
}

# Expanded Product Recommendations Map
RECOMMENDATIONS_MAP = {
    "Electronics & Tech": ['Wireless Earbuds', 'Fast Charging Cable', 'Screen Protector', 'Power Bank', 'Ergonomic Mouse', 'Webcam Cover', 'Laptop Stand'],
    "Clothing & Apparel": ['Matching Accessories', 'Premium Socks', 'Fabric Care Spray', 'Leather Belt', 'Sunglasses', 'Lint Roller', 'Shoe Cleaner'],
    "Home & Kitchen": ['Smart Plug', 'Cleaning Kit', 'Organizer Bins', 'Aromatherapy Diffuser', 'LED Strip Lights', 'Coasters Set', 'Kitchen Timer'],
    "Beauty & Cosmetics": ['Travel Size Kit', 'Organic Serum', 'Makeup Remover', 'Jade Roller', 'Exfoliating Scrub', 'Lip Balm', 'Hair Mask'],
    "Books & Literature": ['LED Reading Light', 'Leather Bookmark', 'Ergonomic Book Stand', 'Noise Cancelling Headphones', 'Reading Glasses', 'Bookshelf Organizer'],
    "Sports & Outdoors": ['Insulated Water Bottle', 'Resistance Bands', 'Cooling Towel', 'Fitness Tracker', 'Protein Shaker', 'Sports Socks', 'Yoga Mat'],
    "Toys & Games": ['Extra Batteries', 'Collectible Display Case', 'Expansion Pack', 'Gaming Headset', 'Controller Grips', 'Card Sleeves'],
    "Automotive": ['Microfiber Cleaning Cloths', 'Car Air Freshener', 'Phone Mount', 'Tire Pressure Gauge', 'Trunk Organizer', 'Steering Wheel Cover'],
    "Health & Wellness": ['Pill Organizer', 'Essential Oils', 'Digital Thermometer', 'Massage Ball', 'Posture Corrector', 'Sleep Mask'],
    "Food & Groceries": ['Reusable Shopping Bags', 'Airtight Containers', 'Gourmet Spice Set', 'Coffee Frother', 'Tea Infuser', 'Snack Bowls'],
    "General / Miscellaneous": ['Digital Gift Card', 'Premium Subscription', 'Mystery Box', 'Eco-friendly Tote', 'Discount Coupon', 'VIP Membership']
}

# 2. Database Setup (SQLite)


def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    return conn


def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    try:
        cursor.execute(
            "ALTER TABLE users ADD COLUMN email TEXT DEFAULT 'admin@engine.local'")
    except sqlite3.OperationalError:
        pass

    cursor.execute("SELECT * FROM users WHERE username = 'admin'")
    if not cursor.fetchone():
        cursor.execute(
            "INSERT INTO users (username, email, password) VALUES ('admin', 'admin@engine.local', 'brookes123')")

    conn.commit()
    conn.close()


init_db()

# 3. Application Routes


@app.route('/')
def welcome():
    return render_template('welcome.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO users (username, email, password) VALUES (?, ?, ?)", (username, email, password))
            conn.commit()
            conn.close()
            flash("Account created successfully! Please log in.", "success")
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash(
                "Username or Email already exists. Please choose a different one.", "error")
            return redirect(url_for('register'))
    return render_template('register.html')


@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        new_password = request.form['new_password']
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM users WHERE username = ? AND email = ?", (username, email))
        user = cursor.fetchone()
        if user:
            cursor.execute(
                "UPDATE users SET password = ? WHERE username = ? AND email = ?", (new_password, username, email))
            conn.commit()
            conn.close()
            flash(
                "Password updated successfully! You can now log in with your new password.", "success")
            return redirect(url_for('login'))
        else:
            conn.close()
            flash(
                "Error: Username and Email combination does not match any existing account.", "error")
            return redirect(url_for('forgot_password'))
    return render_template('forgot_password.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username_or_email = request.form['username']
        password = request.form['password']
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE (username = ? OR email = ?) AND password = ?",
                       (username_or_email, username_or_email, password))
        user = cursor.fetchone()
        conn.close()
        if user:
            session['logged_in'] = True
            session['username'] = user[1]
            return redirect(url_for('dashboard'))
        else:
            flash("Invalid credentials. Please try again.", "error")
            return redirect(url_for('login'))
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('welcome'))


@app.route('/admin')
def admin_panel():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    if session.get('username') != 'admin':
        return "Access Denied: You must be an administrator to view this page.", 403
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, email FROM users")
    all_users = cursor.fetchall()
    conn.close()
    return render_template('admin.html', username=session['username'], users=all_users)


@app.route('/admin/delete/<int:user_id>')
def delete_user(user_id):
    if not session.get('logged_in') or session.get('username') != 'admin':
        return redirect(url_for('login'))
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT username FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()
    if user:
        if user[0] == 'admin':
            flash(
                "Security Alert: The main administrator account cannot be deleted.", "error")
        else:
            cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
            conn.commit()
            flash(
                f"Success: User '{user[0]}' has been permanently removed.", "success")
    else:
        flash("Error: User not found.", "error")
    conn.close()
    return redirect(url_for('admin_panel'))


@app.route('/dashboard')
def dashboard():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('dashboard.html', username=session['username'])

# 4. Core AI API Endpoint


@app.route('/api/analyze', methods=['POST'])
def analyze():
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401

    start_time = time.time()
    data = request.get_json()
    raw_text = data.get('review', '')
    reviews = [r.strip() for r in raw_text.split('\n') if r.strip()]

    if not reviews:
        return jsonify({'error': 'No valid text provided.'}), 400

    total_words = 0
    total_reading_time = 0.0
    total_compound, total_pos, total_neu, total_neg = 0.0, 0.0, 0.0, 0.0
    popular_count, unpopular_count = 0, 0

    all_clean_words = []

    # Rule-based Aspect Categories
    aspect_keywords = {
        "Quality": ['quality', 'broken', 'durable', 'excellent', 'bad', 'good', 'material', 'made', 'perfect', 'terrible', 'great', 'sturdy', 'flimsy', 'design'],
        "Price": ['price', 'cost', 'cheap', 'expensive', 'money', 'value', 'pay', 'paid', 'worth', 'budget'],
        "Service": ['service', 'delivery', 'shipping', 'customer', 'support', 'arrived', 'time', 'late', 'fast', 'package', 'return']
    }
    aspect_hits = {"Quality": 0, "Price": 0, "Service": 0}
    category_hits = {cat: 0 for cat in CATEGORY_RULES.keys()}

    best_review = ""
    worst_review = ""
    max_compound = -1.1
    min_compound = 1.1

    for review in reviews:
        words = len(review.split())
        total_words += words
        total_reading_time += words / 200.0 * 60

        scores = sia.polarity_scores(review)
        current_compound = scores['compound']

        total_compound += current_compound
        total_pos += scores['pos']
        total_neu += scores['neu']
        total_neg += scores['neg']

        if current_compound > max_compound:
            max_compound = current_compound
            best_review = review
        if current_compound < min_compound:
            min_compound = current_compound
            worst_review = review

        if model:
            try:
                pred = model.predict([review])[0]
                if pred in ['Popular', 1, True]:
                    popular_count += 1
                else:
                    unpopular_count += 1
            except:
                if current_compound >= 0.05:
                    popular_count += 1
                else:
                    unpopular_count += 1
        else:
            if current_compound >= 0.05:
                popular_count += 1
            else:
                unpopular_count += 1

        tokenized_words = re.findall(r'\b[a-zA-Z]{3,}\b', review.lower())
        for w in tokenized_words:
            if w not in STOPWORDS:
                all_clean_words.append(w)

            # Check Aspects
            for aspect, keywords in aspect_keywords.items():
                if w in keywords:
                    aspect_hits[aspect] += 1

            # Check Categories
            for category, keywords in CATEGORY_RULES.items():
                if w in keywords:
                    category_hits[category] += 1

    num_reviews = len(reviews)
    avg_compound = round(total_compound / num_reviews, 4)
    avg_pos = round((total_pos / num_reviews) * 100, 1)
    avg_neu = round((total_neu / num_reviews) * 100, 1)
    avg_neg = round((total_neg / num_reviews) * 100, 1)

    if popular_count >= unpopular_count:
        final_prediction = 'Popular'
        confidence = round((popular_count / num_reviews) * 100, 1)
    else:
        final_prediction = 'Unpopular'
        confidence = round((unpopular_count / num_reviews) * 100, 1)

    actions = [f"Processed a batch of {num_reviews} reviews."]
    if final_prediction == 'Popular':
        actions.extend(
            [f"Majority vote ({popular_count}/{num_reviews}) is Positive.", "Highlight in 'Recommended' section."])
    else:
        actions.extend(
            [f"Majority vote ({unpopular_count}/{num_reviews}) is Negative.", "Flag for QA inspection."])

    word_counts = Counter(all_clean_words)
    top_keywords = [word.capitalize()
                    for word, count in word_counts.most_common(6)]
    if not top_keywords:
        top_keywords = ['General', 'Feedback']

    # Determine Top Category
    predicted_category = "General / Miscellaneous"
    max_cat_hits = 0
    for cat, hits in category_hits.items():
        if hits > max_cat_hits:
            max_cat_hits = hits
            predicted_category = cat

    # Select 3 random products from the chosen category's list to ensure variety
    product_pool = RECOMMENDATIONS_MAP.get(
        predicted_category, RECOMMENDATIONS_MAP["General / Miscellaneous"])
    related_products = random.sample(product_pool, min(3, len(product_pool)))

    total_aspects = sum(aspect_hits.values())
    final_aspects = {"Quality": 0, "Price": 0, "Service": 0}
    if total_aspects > 0:
        for aspect, score in aspect_hits.items():
            final_aspects[aspect] = round((score / total_aspects) * 100)
    else:
        final_aspects = {"Quality": 40, "Price": 30, "Service": 30}

    if max_compound <= 0.05:
        best_review = "No strong positive feedback detected."
    if min_compound >= -0.05:
        worst_review = "No strong negative feedback detected."

    return jsonify({
        'prediction': final_prediction,
        'confidence': confidence,
        'vader_compound': avg_compound,
        'vader_pos': avg_pos,
        'vader_neu': avg_neu,
        'vader_neg': avg_neg,
        'text_stats': {'words': total_words, 'reading_time_sec': round(total_reading_time, 1)},
        'actions': actions,
        'keywords': top_keywords,
        'aspects': final_aspects,
        'category': predicted_category,
        'related_products': related_products,
        'best_review': best_review,
        'worst_review': worst_review,
        'latency': round((time.time() - start_time) * 1000, 2)
    })


if __name__ == '__main__':
    app.run(debug=True)
