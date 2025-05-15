"""
Configuration settings for the keyword detection model.
"""

import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data directories
DATA_DIR = os.path.join(BASE_DIR, 'data')
KEYWORDS_DIR = os.path.join(DATA_DIR, 'keywords')
BACKGROUNDS_DIR = os.path.join(DATA_DIR, 'backgrounds')
MIXED_DIR = os.path.join(DATA_DIR, 'mixed')

# Models directory
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Audio settings
SAMPLE_RATE = 16000
FEATURE_PARAMS = {
    'n_mfcc': 13,
    'n_fft': 512,
    'hop_length': 160,
    'window_length_ms': 30,
    'window_step_ms': 10,
}

# Training settings
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 50
VALIDATION_SPLIT = 0.3
EARLY_STOPPING_PATIENCE = 10

# Detection settings
DEFAULT_DETECTION_THRESHOLD = 0.5
DEFAULT_KEYWORD = "hey jarvis"  # Default keyword for model training and testing

# Age groups for simulation
AGE_GROUPS = ['child', 'young_adult', 'adult', 'senior']

# Background noise types
NOISE_TYPES = ['propeller', 'jet', 'cockpit']

# Audio generation settings
DEFAULT_KEYWORD_SAMPLES = 100
DEFAULT_NEGATIVE_SAMPLES_RATIO = 3.0
DEFAULT_NON_KEYWORD_SAMPLES = int(DEFAULT_KEYWORD_SAMPLES * DEFAULT_NEGATIVE_SAMPLES_RATIO)
DEFAULT_SILENCE_MS = 500
DEFAULT_BACKGROUND_SAMPLES = 20
DEFAULT_MIN_DURATION = 3.0
DEFAULT_MAX_DURATION = 10.0
DEFAULT_NUM_MIXES_SAMPLES = int(DEFAULT_KEYWORD_SAMPLES * DEFAULT_BACKGROUND_SAMPLES * len(NOISE_TYPES))
DEFAULT_NUM_MIXES_NON_KEY_WORDS= int(DEFAULT_NON_KEYWORD_SAMPLES * DEFAULT_BACKGROUND_SAMPLES * len(NOISE_TYPES))

# Testing settings
DEFAULT_TEST_SAMPLES = 10
DEFAULT_SHOW_PLOTS = True
DEFAULT_AUDIO_DURATION = 2  # seconds
DEFAULT_BUFFER_DURATION = 5  # seconds

# Google Cloud TTS Configuration
# English voices mapping with gender and accent information
GOOGLE_TTS_VOICES = [
    # US English voices - male (top 4)
    {"name": "en-US-Neural2-D", "gender": "male", "accent": "us", "accent_name": "US English"},  # Best male US voice
    {"name": "en-US-Neural2-J", "gender": "male", "accent": "us", "accent_name": "US English"},  # Second best
    {"name": "en-US-Neural2-A", "gender": "male", "accent": "us", "accent_name": "US English"},  # Third best
    {"name": "en-US-Wavenet-D", "gender": "male", "accent": "us", "accent_name": "US English"},  # Fourth best
    
    # US English voices - female (top 4)
    {"name": "en-US-Neural2-F", "gender": "female", "accent": "us", "accent_name": "US English"},  # Best female US voice
    {"name": "en-US-Neural2-E", "gender": "female", "accent": "us", "accent_name": "US English"},  # Second best
    {"name": "en-US-Neural2-C", "gender": "female", "accent": "us", "accent_name": "US English"},  # Third best
    {"name": "en-US-Wavenet-C", "gender": "female", "accent": "us", "accent_name": "US English"},  # Fourth best
    
    # UK English voices - male (top 4)
    {"name": "en-GB-Neural2-B", "gender": "male", "accent": "uk", "accent_name": "British English"},  # Best male UK voice
    {"name": "en-GB-Neural2-D", "gender": "male", "accent": "uk", "accent_name": "British English"},  # Second best
    
    # UK English voices - female (top 4)
    {"name": "en-GB-Neural2-A", "gender": "female", "accent": "uk", "accent_name": "British English"},  # Best female UK voice
    {"name": "en-GB-Neural2-C", "gender": "female", "accent": "uk", "accent_name": "British English"},  # Second best
    
    # Australian English voices - male (top 2)
    {"name": "en-AU-Neural2-B", "gender": "male", "accent": "au", "accent_name": "Australian English"},  # Best male AU voice
    {"name": "en-AU-Neural2-D", "gender": "male", "accent": "au", "accent_name": "Australian English"},  # Second best
    
    # Australian English voices - female (top 2)
    {"name": "en-AU-Neural2-C", "gender": "female", "accent": "au", "accent_name": "Australian English"},  # Best female AU voice
    {"name": "en-AU-Neural2-A", "gender": "female", "accent": "au", "accent_name": "Australian English"},  # Second best
    
    # Indian English voices - male (top 2)
    {"name": "en-IN-Neural2-B", "gender": "male", "accent": "in", "accent_name": "Indian English"},  # Best male IN voice
    {"name": "en-IN-Neural2-C", "gender": "male", "accent": "in", "accent_name": "Indian English"},  # Second best
    
    # Indian English voices - female (top 2)
    {"name": "en-IN-Neural2-A", "gender": "female", "accent": "in", "accent_name": "Indian English"},  # Best female IN voice
    {"name": "en-IN-Neural2-D", "gender": "female", "accent": "in", "accent_name": "Indian English"},  # Second best
]

# Speech synthesis configuration
GOOGLE_TTS_AUDIO_CONFIG = {
    "audio_encoding": "LINEAR16",  # WAV format
    "speaking_rate": 1.0,  # Normal speed
    "pitch": 0.0,  # Default pitch, will be adjusted by age/gender
    "volume_gain_db": 0.0,  # Normal volume
    "sample_rate_hertz": 16000,  # Match project's sample rate
    "effects_profile_id": ["headphone-class-device"]  # Better audio quality
}

# For backward compatibility, create ACCENTS from GOOGLE_TTS_VOICES
# This allows existing code to continue working with minimal changes
ACCENTS = []
for voice in GOOGLE_TTS_VOICES:
    ACCENTS.append({
        "voice_name": voice["name"],
        "accent": voice["accent"],  
        "accent_name": voice["accent_name"],
        "gender": voice["gender"]
    })

# Default SNR range for mixing
DEFAULT_SNR_RANGE = (-5, 20)

# Non-keywords folder name for organization
NON_KEYWORDS_DIR = 'non_keywords'

# These are selected to be distinct from typical wake words 
# but represent a variety of speech patterns
NON_KEYWORDS = []  # This will be populated from COMMON_WORDS and COMMON_1000_WORDS

COMMON_WORDS = [
    "hello", "thanks", "sorry", "please", "coffee", 
    "water", "today", "weather", "music", "play",
    "stop", "pause", "continue", "next", "previous",
    "volume", "morning", "evening", "dinner", "lunch",
    "breakfast", "meeting", "schedule", "reminder", "alarm",
    "computer", "system", "network", "download", "upload",
    "message", "email", "phone", "call", "text",
    "picture", "photo", "camera", "video", "record",
    "time", "date", "month", "year", "hour",
    "minute", "second", "tomorrow", "yesterday", "weekend"
]

COMMON_1000_WORDS = [ "a", "able", "about", "above", "access", "accessories",
  "account", "act", "action", "active", "activities", "activity", "add",
  "added", "additional", "address", "adult", "advanced", "advertise",
  "advertising", "africa", "after", "again", "against", "age", "agency",
  "ago", "agreement", "air", "al", "all", "along", "already", "also",
  "although", "always", "am", "america", "american", "among", "amount",
  "an", "analysis", "and", "annual", "another", "any", "application",
  "applications", "apr", "april", "archive", "archives", "are", "area",
  "areas", "around", "art", "article", "articles", "arts", "as", "ask",
  "association", "at", "audio", "aug", "august", "australia", "author", "auto",
  "availability", "available", "average", "away", "baby", "back", "bad", "bank",
  "base", "based", "basic", "be", "beach", "beauty", "because", "become",
  "been", "before", "being", "below", "best", "better", "between", "big",
  "bill", "bit", "black", "blog", "blue", "board", "body", "book", "books",
  "both", "box", "brand", "browse", "building", "business", "but", "buy", "by",
  "ca", "calendar", "california", "call", "called", "camera", "can", "canada",
  "car", "card", "cards", "care", "cars", "cart", "case", "cases", "cash",
  "categories", "category", "cd", "cell", "center", "central", "centre",
  "change", "changes", "chapter", "chat", "cheap", "check", "child", "children",
  "china", "choose", "church", "city", "class", "clear", "click", "close",
  "club", "co", "code", "collection", "college", "color", "come", "comment",
  "comments", "commercial", "committee", "common", "community", "companies",
  "company", "compare", "complete", "computer", "computers", "conditions",
  "conference", "construction", "contact", "content", "control", "copy",
  "copyright", "corporate", "cost", "costs", "could", "council", "countries",
  "country", "county", "course", "court", "cover", "create", "created",
  "credit", "current", "currently", "customer", "customers", "daily", "data",
  "database", "date", "david", "day", "days", "de", "deals", "death", "dec",
  "december", "delivery", "department", "description", "design", "designed",
  "details", "development", "did", "different", "digital", "direct", "director",
  "directory", "discount", "discussion", "display", "district", "do",
  "document", "does", "doing", "domain", "done", "down", "download",
  "downloads", "drive", "drug", "due", "during", "dvd", "each", "early", "east",
  "easy", "ebay", "economic", "edit", "edition", "education", "effects",
  "either", "electronics", "else", "email", "employment", "en", "end", "energy",
  "engineering", "english", "enough", "enter", "entertainment", "entry",
  "environment", "environmental", "equipment", "error", "estate", "et",
  "europe", "european", "even", "event", "events", "ever", "every",
  "everything", "example", "experience", "face", "fact", "family", "faq", "far",
  "fast", "fax", "features", "feb", "february", "federal", "feedback", "feel",
  "few", "field", "figure", "file", "files", "film", "final", "finance",
  "financial", "find", "fire", "first", "five", "florida", "following", "food",
  "for", "force", "form", "format", "forum", "forums", "found", "four",
  "france", "free", "french", "friday", "friend", "friends", "from", "front",
  "full", "fun", "function", "further", "future", "gallery", "game", "games",
  "garden", "gay", "general", "germany", "get", "getting", "gift", "gifts",
  "girl", "girls", "give", "given", "global", "go", "god", "going", "gold",
  "golf", "good", "google", "got", "government", "great", "green", "group",
  "groups", "growth", "guide", "had", "half", "hand", "hard", "hardware", "has",
  "have", "having", "he", "head", "health", "heart", "help", "her", "here",
  "high", "higher", "him", "his", "history", "holiday", "home", "hosting",
  "hot", "hotel", "hotels", "hours", "house", "how", "however", "html", "human",
  "id", "if", "ii", "image", "images", "important", "in", "include", "included",
  "includes", "including", "income", "increase", "index", "india", "individual",
  "industry", "info", "information", "insurance", "interest", "international",
  "internet", "into", "is", "island", "issue", "issues", "it", "item", "items",
  "its", "james", "jan", "january", "japan", "job", "jobs", "john", "join",
  "journal", "jul", "july", "jun", "june", "just", "keep", "key", "kids",
  "king", "kingdom", "know", "knowledge", "known", "la", "lake", "land",
  "language", "large", "last", "later", "latest", "law", "learn", "learning",
  "least", "left", "legal", "less", "let", "level", "library", "license",
  "life", "light", "like", "limited", "line", "link", "links", "linux", "list",
  "listed", "listing", "listings", "little", "live", "living", "loan", "loans",
  "local", "location", "log", "login", "london", "long", "look", "looking",
  "loss", "lot", "love", "low", "lyrics", "made", "magazine", "mail", "main",
  "major", "make", "makes", "making", "man", "management", "manager", "many",
  "map", "mar", "march", "mark", "market", "marketing", "material", "materials",
  "may", "me", "mean", "means", "media", "medical", "meet", "meeting", "member",
  "members", "memory", "men", "message", "messages", "method", "methods",
  "michael", "microsoft", "might", "miles", "million", "minutes", "mobile",
  "model", "models", "monday", "money", "month", "months", "more", "most",
  "movie", "movies", "much", "music", "must", "my", "name", "national",
  "natural", "nature", "near", "need", "needs", "net", "network", "never",
  "new", "news", "newsletter", "next", "night", "no", "non", "none", "north",
  "not", "note", "notes", "notice", "nov", "november", "now", "nude", "number",
  "oct", "october", "of", "off", "offer", "offers", "office", "official",
  "often", "oh", "oil", "old", "on", "once", "one", "online", "only", "open",
  "options", "or", "order", "orders", "original", "other", "others", "our",
  "out", "over", "overall", "own", "page", "pages", "paper", "park", "part",
  "parts", "party", "password", "past", "paul", "pay", "payment", "pc",
  "people", "per", "percent", "performance", "period", "person", "personal",
  "phone", "photo", "photos", "pics", "picture", "pictures", "place", "plan",
  "planning", "play", "player", "please", "plus", "pm", "point", "points",
  "poker", "policies", "policy", "political", "popular", "porn", "position",
  "possible", "post", "posted", "posts", "power", "powered", "practice",
  "present", "president", "press", "previous", "price", "prices", "print",
  "privacy", "private", "pro", "problem", "problems", "process", "product",
  "production", "products", "professional", "profile", "program", "programs",
  "project", "projects", "property", "protection", "provide", "provided",
  "provides", "public", "published", "purchase", "put", "quality", "question",
  "questions", "quick", "quote", "radio", "range", "rate", "rates", "rating",
  "re", "read", "reading", "real", "really", "receive", "received", "recent",
  "record", "records", "red", "reference", "region", "register", "registered",
  "related", "release", "remember", "reply", "report", "reports", "request",
  "required", "requirements", "research", "reserved", "resource", "resources",
  "response", "result", "results", "return", "review", "reviews", "right",
  "rights", "risk", "road", "rock", "room", "rss", "rules", "run", "safety",
  "said", "sale", "sales", "same", "san", "save", "say", "says", "school",
  "schools", "science", "search", "second", "section", "security", "see",
  "select", "self", "sell", "seller", "send", "sep", "september", "series",
  "server", "service", "services", "set", "several", "sex", "shall", "share",
  "she", "shipping", "shoes", "shop", "shopping", "short", "should", "show",
  "shows", "side", "sign", "similar", "simple", "since", "single", "site",
  "sites", "size", "small", "so", "social", "society", "software", "solutions",
  "some", "something", "sony", "sort", "sound", "source", "south", "space",
  "special", "specific", "speed", "sports", "st", "staff", "standard",
  "standards", "star", "start", "state", "statement", "states", "status",
  "stay", "step", "still", "stock", "storage", "store", "stores", "stories",
  "story", "street", "student", "students", "studies", "study", "stuff",
  "style", "subject", "submit", "subscribe", "such", "summary", "sun",
  "support", "sure", "system", "systems", "table", "take", "taken", "talk",
  "tax", "team", "tech", "technical", "technology", "teen", "tell", "term",
  "terms", "test", "texas", "text", "than", "thanks", "that", "the", "their",
  "them", "then", "there", "these", "they", "thing", "things", "think", "third",
  "this", "those", "though", "thought", "thread", "three", "through", "tickets",
  "time", "times", "tips", "title", "to", "today", "together", "too", "tools",
  "top", "topic", "topics", "total", "town", "toys", "track", "trade",
  "training", "travel", "treatment", "true", "try", "turn", "tv", "two", "type",
  "uk", "under", "unit", "united", "university", "until", "up", "update",
  "updated", "upon", "url", "us", "usa", "use", "used", "user", "users",
  "using", "usually", "value", "various", "version", "very", "via", "video",
  "videos", "view", "visit", "want", "war", "was", "washington", "watch",
  "water", "way", "we", "weather", "web", "website", "week", "weight",
  "welcome", "well", "were", "west", "western", "what", "when", "where",
  "whether", "which", "while", "white", "who", "whole", "why", "wide", "will",
  "window", "windows", "wireless", "with", "within", "without", "women", "word",
  "words", "work", "working", "works", "world", "would", "write", "writing",
  "written", "yahoo", "year", "years", "yellow", "yes", "yet", "york", "you",
  "young", "your" ]

# Merge COMMON_WORDS and COMMON_1000_WORDS into NON_KEYWORDS
NON_KEYWORDS = list(set(COMMON_WORDS + COMMON_1000_WORDS))