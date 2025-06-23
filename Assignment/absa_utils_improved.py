# Improved ABSA utilities with Aspect-Specific Sentiment Analysis
# Enhanced version of absa_utils_fixed.py

import re
import pickle
import joblib
import pandas as pd

# Load models
try:
    svm_model = joblib.load('models/traditional_ml/svm.pkl')
    with open('data/tfidf_vectorizer.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    MODEL_AVAILABLE = True
except:
    MODEL_AVAILABLE = False

AIRLINE_ASPECTS = {
    'service': {
        'keywords': ['service', 'staff', 'crew', 'attendant', 'employee', 'agent', 'customer service'],
        'description': 'Staff and customer service quality'
    },
    'flight_experience': {
        'keywords': ['flight', 'trip', 'experience', 'journey', 'travel', 'plane', 'aircraft'],
        'description': 'Overall flight experience'
    },
    'delays': {
        'keywords': ['delay', 'delayed', 'late', 'cancelled', 'canceled', 'cancel', 'wait', 'waiting', 'on time', 'early'],
        'description': 'Timeliness and schedule issues'
    },
    'comfort': {
        'keywords': ['seat', 'seats', 'comfort', 'legroom', 'space', 'cramped', 'tight', 'comfortable', 'uncomfortable'],
        'description': 'Physical comfort during flight'
    },
    'booking': {
        'keywords': ['booking', 'book', 'reservation', 'website', 'app', 'check in', 'checkin'],
        'description': 'Booking and check-in process'
    },
    'baggage': {
        'keywords': ['bag', 'bags', 'baggage', 'luggage', 'lost', 'missing'],
        'description': 'Baggage handling'
    },
    'food': {
        'keywords': ['food', 'meal', 'snack', 'drink', 'beverage', 'catering'],
        'description': 'Food and beverage service'
    }
}

keyword_to_aspect = {}
for aspect, info in AIRLINE_ASPECTS.items():
    for keyword in info['keywords']:
        keyword_to_aspect[keyword.lower()] = aspect

def preprocess_text_enhanced(text):
    """Enhanced preprocessing using your improved approach"""
    if pd.isna(text) or not text:
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # Remove @mentions
    text = re.sub(r'@\w+', '', text)

    # Remove hashtag symbol but keep the word
    text = re.sub(r'#(\w+)', r'\1', text)

    # Regex-based tokenization (preserve contractions like "you've")
    words = re.findall(r"\b\w+'\w+|\w+\b", text)

    # Comprehensive contraction and abbreviation dictionary
    contractions = {
        "don't": "do not", "won't": "will not", "can't": "cannot",
        "i'm": "i am", "you're": "you are", "he's": "he is", "she's": "she is",
        "it's": "it is", "we're": "we are", "they're": "they are",
        "i've": "i have", "you've": "you have", "we've": "we have", "they've": "they have",
        "i'd": "i would", "you'd": "you would", "he'd": "he would", "she'd": "she would",
        "we'd": "we would", "they'd": "they would",
        "i'll": "i will", "you'll": "you will", "he'll": "he will", "she'll": "she will",
        "we'll": "we will", "they'll": "they will",
        "didn't": "did not", "hasn't": "has not", "hadn't": "had not",
        "couldn't": "could not", "shouldn't": "should not", "wouldn't": "would not",
        "isn't": "is not", "aren't": "are not", "wasn't": "was not", "weren't": "were not",
        "n't": " not", "'re": " are", "'ve": " have", "'ll": " will",
        "'d": " would", "'m": " am",
        "u": "you", "ur": "your", "r": "are", "pls": "please", "plz": "please",
        "btw": "by the way", "idk": "i don't know", "imo": "in my opinion",
        "omg": "oh my god", "lol": "laugh out loud", "rofl": "rolling on the floor laughing",
        "lmao": "laughing my ass off", "smh": "shaking my head", "thx": "thanks",
        "ty": "thank you", "np": "no problem", "omw": "on my way", "b4": "before",
        "gr8": "great", "k": "okay", "bc": "because", "b/c": "because",
        "w/": "with", "w/o": "without", "bday": "birthday", "msg": "message",
        "fyi": "for your information", "asap": "as soon as possible", "brb": "be right back",
        "gtg": "got to go", "ttyl": "talk to you later", "afaik": "as far as i know",
        "icymi": "in case you missed it", "tbh": "to be honest", "im": "i am",
        "ive": "i have", "idc": "i don't care", "ikr": "i know right", "ya": "you",
        "ya'll": "you all", "sup": "what's up", "bff": "best friend forever",
        "bf": "boyfriend", "gf": "girlfriend"
    }

    # Replace contractions and abbreviations
    expanded_words = []
    for word in words:
        if word in contractions:
            expanded_words.extend(contractions[word].split())
        else:
            expanded_words.append(word)

    # Rejoin
    text = ' '.join(expanded_words)

    # Remove punctuation (but keep numbers)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # Clean up spacing
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def extract_aspects_with_context(text):
    """Extract aspects along with their surrounding context"""
    if not text:
        return []
    
    clean_text = preprocess_text_enhanced(text)
    found_aspects = []
    
    # Split into sentences for better context analysis
    sentences = re.split(r'[.!?]+', text)
    
    words = clean_text.split()
    for word in words:
        if word in keyword_to_aspect:
            aspect = keyword_to_aspect[word]
            
            # Find which sentence contains this aspect
            aspect_context = text  # Default to full text
            for sentence in sentences:
                if word in sentence.lower() or any(kw in sentence.lower() for kw in AIRLINE_ASPECTS[aspect]['keywords']):
                    aspect_context = sentence.strip()
                    break
            
            found_aspects.append({
                'aspect': aspect,
                'context': aspect_context,
                'keyword_found': word
            })
    
    # Also check for multi-word phrases
    for keyword, aspect in keyword_to_aspect.items():
        if len(keyword.split()) > 1 and keyword in clean_text:
            # Find sentence containing this phrase
            aspect_context = text
            for sentence in sentences:
                if keyword in sentence.lower():
                    aspect_context = sentence.strip()
                    break
            
            found_aspects.append({
                'aspect': aspect,
                'context': aspect_context,
                'keyword_found': keyword
            })
    
    # Remove duplicates while preserving context
    unique_aspects = {}
    for item in found_aspects:
        aspect = item['aspect']
        if aspect not in unique_aspects:
            unique_aspects[aspect] = item
        else:
            # Keep the one with more specific context (shorter sentence usually means more specific)
            if len(item['context']) < len(unique_aspects[aspect]['context']):
                unique_aspects[aspect] = item
    
    return list(unique_aspects.values())

def predict_aspect_specific_sentiment(text, aspect_context, aspect_name):
    """Analyze sentiment specifically for an aspect using its context"""
    
    # Enhanced sentiment analysis for aspect-specific context
    if not MODEL_AVAILABLE:
        return analyze_sentiment_with_rules(aspect_context, aspect_name)
    
    try:
        # Use the aspect context for more accurate sentiment
        context_to_analyze = aspect_context if len(aspect_context.strip()) > 0 else text
        clean_context = preprocess_text_enhanced(context_to_analyze)
        
        # If context is too short, use full text
        if len(clean_context.split()) < 3:
            clean_context = preprocess_text_enhanced(text)
        
        # Transform with your TF-IDF vectorizer
        text_vector = tfidf_vectorizer.transform([clean_context])
        
        # Predict with your SVM model
        prediction = svm_model.predict(text_vector)[0]
        confidence = svm_model.predict_proba(text_vector)[0].max()
        
        return prediction, confidence
    
    except:
        return analyze_sentiment_with_rules(aspect_context, aspect_name)

def analyze_sentiment_with_rules(context, aspect_name):
    """Enhanced rule-based sentiment analysis for specific aspects"""
    context_lower = context.lower()
    
    # Aspect-specific positive and negative indicators
    aspect_sentiments = {
        'service': {
            'positive': ['helpful', 'friendly', 'professional', 'courteous', 'excellent', 'amazing', 'great', 'wonderful', 'fantastic', 'polite', 'kind'],
            'negative': ['rude', 'unhelpful', 'unprofessional', 'terrible', 'awful', 'horrible', 'poor', 'bad', 'worst', 'disrespectful']
        },
        'delays': {
            'positive': ['on time', 'early', 'punctual', 'timely', 'quick', 'fast', 'efficient'],
            'negative': ['delayed', 'late', 'cancelled', 'canceled', 'wait', 'waiting', 'hours', 'slow']
        },
        'comfort': {
            'positive': ['comfortable', 'spacious', 'roomy', 'cozy', 'nice', 'good', 'pleasant'],
            'negative': ['uncomfortable', 'cramped', 'tight', 'small', 'hard', 'terrible', 'awful', 'poor']
        },
        'baggage': {
            'positive': ['safe', 'secure', 'arrived', 'intact', 'good', 'fine'],
            'negative': ['lost', 'missing', 'damaged', 'broken', 'delayed', 'terrible', 'awful']
        },
        'booking': {
            'positive': ['easy', 'simple', 'smooth', 'quick', 'efficient', 'good', 'great', 'excellent'],
            'negative': ['difficult', 'complicated', 'slow', 'confusing', 'terrible', 'awful', 'crashed', 'broken']
        },
        'food': {
            'positive': ['delicious', 'tasty', 'good', 'excellent', 'great', 'wonderful', 'fresh'],
            'negative': ['terrible', 'awful', 'bad', 'disgusting', 'poor', 'stale', 'cold']
        },
        'flight_experience': {
            'positive': ['smooth', 'excellent', 'great', 'wonderful', 'amazing', 'fantastic', 'perfect', 'enjoyable'],
            'negative': ['terrible', 'awful', 'horrible', 'worst', 'bad', 'disappointing', 'rough']
        }
    }
    
    # General sentiment words
    general_positive = ['good', 'great', 'excellent', 'amazing', 'awesome', 'love', 'best', 'fantastic', 'wonderful', 'perfect']
    general_negative = ['bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'disappointing', 'poor', 'disgusting']
    
    # Get aspect-specific sentiment words
    aspect_positive = aspect_sentiments.get(aspect_name, {}).get('positive', [])
    aspect_negative = aspect_sentiments.get(aspect_name, {}).get('negative', [])
    
    # Combine aspect-specific and general sentiment words
    all_positive = aspect_positive + general_positive
    all_negative = aspect_negative + general_negative
    
    # Count sentiment indicators with weights
    pos_count = 0
    neg_count = 0
    
    for word in all_positive:
        if word in context_lower:
            # Give higher weight to aspect-specific words
            weight = 2 if word in aspect_positive else 1
            pos_count += weight
    
    for word in all_negative:
        if word in context_lower:
            # Give higher weight to aspect-specific words
            weight = 2 if word in aspect_negative else 1
            neg_count += weight
    
    # Handle negation (simple approach)
    negation_words = ['not', 'no', 'never', 'nothing', 'nobody', 'nowhere', 'neither', 'nor']
    has_negation = any(neg_word in context_lower for neg_word in negation_words)
    
    if has_negation:
        # Simple negation handling - flip the sentiment
        pos_count, neg_count = neg_count, pos_count
    
    # Determine sentiment
    if pos_count > neg_count:
        confidence = min(0.7 + (pos_count - neg_count) * 0.1, 0.9)
        return 2, confidence  # positive
    elif neg_count > pos_count:
        confidence = min(0.7 + (neg_count - pos_count) * 0.1, 0.9)
        return 0, confidence  # negative
    else:
        return 1, 0.5  # neutral

def extract_aspects_focused(text):
    """Backward compatibility function"""
    aspects_with_context = extract_aspects_with_context(text)
    return [item['aspect'] for item in aspects_with_context]

def predict_sentiment_with_svm(text):
    """Backward compatibility function for overall sentiment"""
    if not MODEL_AVAILABLE:
        enhanced_text = preprocess_text_enhanced(text)
        positive_words = ['good', 'great', 'excellent', 'amazing', 'awesome', 'love', 'best', 'fantastic', 'wonderful', 'thank', 'thanks']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'delayed', 'cancelled', 'rude', 'disappointed', 'poor']
        
        pos_count = sum(1 for word in positive_words if word in enhanced_text)
        neg_count = sum(1 for word in negative_words if word in enhanced_text)
        
        if neg_count > pos_count:
            return 0, 0.7
        elif pos_count > neg_count:
            return 2, 0.7
        else:
            return 1, 0.5
    
    try:
        clean_text = preprocess_text_enhanced(text)
        text_vector = tfidf_vectorizer.transform([clean_text])
        prediction = svm_model.predict(text_vector)[0]
        confidence = svm_model.predict_proba(text_vector)[0].max()
        return prediction, confidence
    except:
        return 1, 0.5

def analyze_tweet_complete(tweet_text):
    """Complete ABSA analysis with aspect-specific sentiment"""
    aspects_with_context = extract_aspects_with_context(tweet_text)
    
    if not aspects_with_context:
        return {"aspects": [], "overall_sentiment": "neutral", "confidence": 0.5}
    
    # Get overall sentiment for comparison
    overall_sentiment_numeric, overall_confidence = predict_sentiment_with_svm(tweet_text)
    sentiment_labels = ['negative', 'neutral', 'positive']
    overall_sentiment = sentiment_labels[overall_sentiment_numeric]
    
    results = {
        "overall_sentiment": overall_sentiment,
        "confidence": overall_confidence,
        "aspects": []
    }
    
    # Analyze each aspect with its specific context
    for aspect_info in aspects_with_context:
        aspect = aspect_info['aspect']
        context = aspect_info['context']
        
        # Get aspect-specific sentiment
        aspect_sentiment_numeric, aspect_confidence = predict_aspect_specific_sentiment(
            tweet_text, context, aspect
        )
        aspect_sentiment = sentiment_labels[aspect_sentiment_numeric]
        
        results["aspects"].append({
            "aspect": aspect,
            "description": AIRLINE_ASPECTS[aspect]['description'],
            "sentiment": aspect_sentiment,
            "confidence": aspect_confidence,
            "context": context[:100] + "..." if len(context) > 100 else context  # Truncate long context
        })
    
    return results

# Test function to demonstrate the improvement
def test_aspect_sentiment():
    """Test function to show aspect-specific sentiment working"""
    test_tweets = [
        "The flight was delayed for 3 hours but the staff was very helpful",
        "Amazing crew service but the seats were uncomfortable",
        "Terrible baggage handling but smooth booking process",
        "Great customer service and comfortable seats throughout",
        "Flight on time but food was awful"
    ]
    
    print("Testing Aspect-Specific Sentiment Analysis:")
    print("=" * 60)
    
    for tweet in test_tweets:
        print(f"\nTweet: {tweet}")
        results = analyze_tweet_complete(tweet)
        print(f"Overall: {results['overall_sentiment']} ({results['confidence']:.2f})")
        
        for aspect in results['aspects']:
            print(f"  {aspect['aspect']}: {aspect['sentiment']} ({aspect['confidence']:.2f})")
        print("-" * 40)

if __name__ == "__main__":
    test_aspect_sentiment()

# Quick fix for analyze_tweet_complete function
# Add this to your absa_utils_improved.py or replace the existing function

def analyze_tweet_complete_fixed(tweet_text):
    """Fixed ABSA analysis with proper aspect-specific sentiment"""
    aspects_with_context = extract_aspects_with_context(tweet_text)
    
    if not aspects_with_context:
        return {"aspects": [], "overall_sentiment": "neutral", "confidence": 0.5}
    
    # Get overall sentiment for comparison
    overall_sentiment_numeric, overall_confidence = predict_sentiment_with_svm(tweet_text)
    sentiment_labels = ['negative', 'neutral', 'positive']
    overall_sentiment = sentiment_labels[overall_sentiment_numeric]
    
    results = {
        "overall_sentiment": overall_sentiment,
        "confidence": overall_confidence,
        "aspects": []
    }
    
    # Manual aspect-specific analysis for common patterns
    tweet_lower = tweet_text.lower()
    
    for aspect_info in aspects_with_context:
        aspect = aspect_info['aspect']
        context = aspect_info['context']
        
        # Rule-based aspect-specific sentiment
        aspect_sentiment = "neutral"
        aspect_confidence = 0.5
        
        if aspect == "delays":
            if any(word in tweet_lower for word in ['delayed', 'late', 'cancelled', 'cancel', 'wait', 'waiting']):
                aspect_sentiment = "negative"
                aspect_confidence = 0.8
            elif any(word in tweet_lower for word in ['on time', 'early', 'punctual', 'quick']):
                aspect_sentiment = "positive"
                aspect_confidence = 0.8
        
        elif aspect == "service":
            # Check for positive service words
            positive_service = ['helpful', 'great', 'excellent', 'amazing', 'fantastic', 'wonderful', 
                              'friendly', 'professional', 'courteous', 'polite', 'kind', 'nice']
            negative_service = ['rude', 'terrible', 'awful', 'horrible', 'poor', 'bad', 'worst', 
                              'unprofessional', 'unhelpful']
            
            # Look specifically in the context around service mentions
            service_context = context.lower()
            
            pos_count = sum(1 for word in positive_service if word in service_context)
            neg_count = sum(1 for word in negative_service if word in service_context)
            
            if pos_count > neg_count:
                aspect_sentiment = "positive"
                aspect_confidence = 0.8
            elif neg_count > pos_count:
                aspect_sentiment = "negative" 
                aspect_confidence = 0.8
            else:
                # Check the broader tweet context for service
                pos_count_broad = sum(1 for word in positive_service if word in tweet_lower)
                neg_count_broad = sum(1 for word in negative_service if word in tweet_lower)
                
                if pos_count_broad > neg_count_broad:
                    aspect_sentiment = "positive"
                    aspect_confidence = 0.7
                elif neg_count_broad > pos_count_broad:
                    aspect_sentiment = "negative"
                    aspect_confidence = 0.7
        
        elif aspect == "comfort":
            if any(word in tweet_lower for word in ['comfortable', 'spacious', 'roomy', 'nice']):
                aspect_sentiment = "positive"
                aspect_confidence = 0.8
            elif any(word in tweet_lower for word in ['uncomfortable', 'cramped', 'tight', 'small']):
                aspect_sentiment = "negative"
                aspect_confidence = 0.8
        
        elif aspect == "baggage":
            if any(word in tweet_lower for word in ['lost', 'missing', 'damaged']):
                aspect_sentiment = "negative"
                aspect_confidence = 0.9
            elif any(word in tweet_lower for word in ['safe', 'arrived', 'intact']):
                aspect_sentiment = "positive"
                aspect_confidence = 0.8
        
        elif aspect == "booking":
            if any(word in tweet_lower for word in ['easy', 'smooth', 'simple', 'quick']):
                aspect_sentiment = "positive"
                aspect_confidence = 0.8
            elif any(word in tweet_lower for word in ['difficult', 'complicated', 'crashed', 'broken']):
                aspect_sentiment = "negative"
                aspect_confidence = 0.8
        
        elif aspect == "food":
            if any(word in tweet_lower for word in ['delicious', 'tasty', 'good', 'great']):
                aspect_sentiment = "positive"
                aspect_confidence = 0.8
            elif any(word in tweet_lower for word in ['terrible', 'awful', 'bad', 'disgusting']):
                aspect_sentiment = "negative"
                aspect_confidence = 0.8
        
        else:  # flight_experience or other
            # For general aspects, use a more balanced approach
            positive_words = ['great', 'excellent', 'amazing', 'good', 'smooth', 'perfect']
            negative_words = ['terrible', 'awful', 'bad', 'worst', 'horrible']
            
            pos_count = sum(1 for word in positive_words if word in tweet_lower)
            neg_count = sum(1 for word in negative_words if word in tweet_lower)
            
            if pos_count > neg_count:
                aspect_sentiment = "positive"
                aspect_confidence = 0.7
            elif neg_count > pos_count:
                aspect_sentiment = "negative"
                aspect_confidence = 0.7
            else:
                # Default to overall sentiment for ambiguous cases
                aspect_sentiment = overall_sentiment
                aspect_confidence = overall_confidence * 0.8
        
        results["aspects"].append({
            "aspect": aspect,
            "description": AIRLINE_ASPECTS[aspect]['description'],
            "sentiment": aspect_sentiment,
            "confidence": aspect_confidence,
            "context": context[:100] + "..." if len(context) > 100 else context
        })
    
    return results

# Test function
def test_fixed_analysis():
    test_tweet = "The flight was delayed for 3 hours but the staff was very helpful"
    result = analyze_tweet_complete_fixed(test_tweet)
    
    print(f"Overall: {result['overall_sentiment']}")
    for aspect in result['aspects']:
        print(f"{aspect['aspect']}: {aspect['sentiment']} ({aspect['confidence']:.2f})")
    
    return result

if __name__ == "__main__":
    test_fixed_analysis()
