import json
import spacy
import fitz  # PyMuPDF for PDF parsing
from flask import Flask, render_template, request, jsonify, session
from sentence_transformers import SentenceTransformer, util
import mysql.connector
from mysql.connector import Error
from annoy import AnnoyIndex
import uuid
from dotenv import load_dotenv
import os
import re
import logging
import unicodedata
from datetime import datetime, timedelta
import torch
import psutil  # For memory usage logging
from transformers import LlamaForCausalLM, LlamaTokenizer

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Configure logging
logging.basicConfig(
    filename='app_debug.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
# Helper function for memory usage logging
def log_memory_usage(note=""):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    logging.debug(f"{note} - Memory Usage: {mem_info.rss / (1024 ** 2):.2f} MB")

# Load environment variables from .env file
load_dotenv()
logging.debug("Loaded environment variables")

# Set up Flask app with session support for remembering choices
app = Flask(__name__, template_folder='templates')
app.secret_key = 'your_secret_key'  # Replace with a secure key

# MySQL connection using environment variables
try:
    db = mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME")
    )
    logging.info("Connected to the database.")
except Error as db_error:
    logging.error(f"Database connection error: {db_error}")
    db = None

if db is None:
    raise Exception("Database connection could not be established. Check your credentials and database configuration.")

cursor = db.cursor()

# Load LLaMA model and tokenizer with GPU optimization
token = "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"  # Replace with your actual token
model_path = "meta-llama/Llama-2-7B-hf"  # Adjust to your model path if different

# Determine device: Prefer CUDA if available, otherwise fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

log_memory_usage("Before model loading")

try:
    # Load model with memory-efficient device map if necessary
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        token=token,
        device_map="balanced_low_0" if torch.cuda.is_available() else None,
        torch_dtype=torch.float16
    )
    model.to(device)
    logging.info("Model loaded with device_map balanced_low_0 to optimize memory usage.")
except Exception as e:
    logging.error(f"Error during model loading: {e}")
    raise e

log_memory_usage("After model loading")

# Load tokenizer
try:
    tokenizer = LlamaTokenizer.from_pretrained(model_path, token=token)
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token if needed
    logging.info("Tokenizer loaded successfully")
except Exception as e:
    logging.error(f"Error during tokenizer loading: {e}")
    raise e


# Load Sentence-BERT model for semantic understanding
try:
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    logging.info("Sentence-BERT model loaded successfully")
except Exception as e:
    logging.error(f"Error during Sentence-BERT model loading: {e}")
    raise e

# Load SpaCy model for NLP
try:
    nlp = spacy.load("en_core_web_sm")
    logging.info("SpaCy NLP model loaded successfully")
except Exception as e:
    logging.error(f"Error during SpaCy model loading: {e}")
    raise e

# Annoy index setup
embedding_dim = 384
annoy_index = AnnoyIndex(embedding_dim, 'angular')
convo_embeddings = []
convo_index_map = {}
annoy_index_build_count = 0
SIMILARITY_THRESHOLD = 0.5
logging.debug("Annoy index setup complete")

def load_or_initialize_intents():
    global intents_data
    intents_file = "data/intents.json"
    try:
        with open(intents_file, 'r') as f:
            intents_data = json.load(f)
            logging.info("Loaded intents from JSON")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Error loading intents: {e}")
        intents_data = {"intents": []}

# Save updated intents data to intents.json
def save_intents_data():
    try:
        with open("data/intents.json", 'w') as f:
            json.dump(intents_data, f, indent=4)
            logging.info("Intents data saved to JSON")
    except Exception as e:
        logging.error(f"Error saving intents data: {e}")

def clean_text(text):
    # Normalize text and remove unwanted characters
    cleaned_text = unicodedata.normalize("NFKD", text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

# Categorize new questions into existing intents or create new ones
def categorize_new_questions(new_data, similarity_threshold=0.75):
    global intents_data
    for item in new_data:
        question, answer = item["question"].strip(), item["answer"].strip()
        try:
            question_embedding = sbert_model.encode(question)
            best_tag, max_similarity = None, 0

            # Check similarity with existing intents
            for intent in intents_data["intents"]:
                tag = intent["tag"]
                for pattern in intent["patterns"]:
                    pattern_embedding = sbert_model.encode(pattern)
                    similarity = util.pytorch_cos_sim(question_embedding, pattern_embedding).item()
                    if similarity > max_similarity:
                        max_similarity, best_tag = similarity, tag

            # If a similar intent exists, add to its patterns; otherwise, create a new intent
            if max_similarity >= similarity_threshold and best_tag:
                for intent in intents_data["intents"]:
                    if intent["tag"] == best_tag:
                        if question not in intent["patterns"]:
                            intent["patterns"].append(question)
                        if answer not in intent["responses"]:
                            intent["responses"].append(answer)
                        break
            else:
                new_tag = generate_tag_from_question(question)
                new_intent = {"tag": new_tag, "patterns": [question], "responses": [answer]}
                intents_data["intents"].append(new_intent)
                logging.info(f"New intent created with tag: {new_tag}")

        except Exception as e:
            logging.error(f"Error categorizing question '{question}': {e}")

    save_intents_data()

def handle_multi_query(user_input):
    queries = [query.strip() for query in re.split(r'\band\b|\balso\b|\b,', user_input)]
    generated_tags = []

    # Process each query to find or create a tag
    for query in queries:
        best_intents = find_best_intents(query)

        # Use the best-matching intent tag or generate a new one
        if best_intents and len(best_intents) >= 1:
            best_tag = best_intents[0][0]["tag"]
            generated_tags.append(best_tag)
        else:
            new_tag = generate_tag_from_question(query)
            generated_tags.append(new_tag)
            new_intent = {"tag": new_tag, "patterns": [query], "responses": []}
            intents_data["intents"].append(new_intent)

    save_intents_data()
    return generated_tags

# Generate a meaningful tag from the question
def generate_tag_from_question(question):
    try:
        doc = nlp(question)
        keywords = [token.lemma_ for token in doc if token.pos_ in ["NOUN", "VERB", "ADJ"]]
        return "_".join(keywords[:3])
    except Exception as e:
        logging.error(f"Error generating tag from question '{question}': {e}")
        return "default_tag"


@app.route('/upload_pdf', methods=["POST"])
def upload_pdf():
    file = request.files['file']
    os.makedirs("uploads", exist_ok=True)
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)

    # Extract questions and answers, then categorize
    new_data = extract_questions_answers_from_pdf(file_path)
    categorize_new_questions(new_data)

    return jsonify({"response": "Intents updated successfully with data from PDF."})

# Extract questions and answers from PDF
def extract_questions_answers_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    qa_pattern = re.compile(r"Q:\s*(.*?)\s*A:\s*(.*?)(?=\nQ:|\Z)", re.DOTALL)
    matches = qa_pattern.findall(text)
    qa_pairs = [{"question": clean_text(q.strip()), "answer": clean_text(a.strip())} for q, a in matches]
    return qa_pairs


def discover_schema():
    cursor.execute("SHOW TABLES")
    tables = cursor.fetchall()

    for table in tables:
        table_name = table[0]
        cursor.execute(f"DESCRIBE {table_name}")
        columns = cursor.fetchall()

        save_query_to_dataset(f"Select all from {table_name}", f"SELECT * FROM {table_name};", table_name, "select_all")

        for col in columns:
            col_name = col[0]
            save_query_to_dataset(f"Select {col_name} from {table_name}", f"SELECT {col_name} FROM {table_name};",
                                  table_name, "select_column")
            save_query_to_dataset(f"Where {col_name} equals condition in {table_name}",
                                  f"SELECT * FROM {table_name} WHERE {col_name} = 'condition';", table_name,
                                  "condition_column")

        save_query_to_dataset(f"Count all in {table_name}", f"SELECT COUNT(*) FROM {table_name};", table_name, "count")

        for col in columns:
            if "int" in col[1].lower() or "decimal" in col[1].lower():
                save_query_to_dataset(f"Sum of {col[0]} in {table_name}", f"SELECT SUM({col[0]}) FROM {table_name};",
                                      table_name, "sum")
                break

def create_tags_from_schema(schema_info):
    cursor.execute("TRUNCATE TABLE tags_and_tables")
    for table in schema_info.keys():
        cursor.execute("INSERT INTO tags_and_tables (tag, table_name) VALUES (%s, %s)", (table, table))
    db.commit()
    print("Tags created for tables only:", schema_info.keys())

def find_best_intents(user_input):
    user_embedding = sbert_model.encode(user_input, convert_to_tensor=True)
    matched_intents = []
    similarity_threshold = 0.6

    if 'intents' not in intents_data:
        print("Error: 'intents' key is missing in intents_data.")
        return matched_intents

    for intent in intents_data['intents']:
        for pattern in intent['patterns']:
            pattern_embedding = sbert_model.encode(pattern, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(user_embedding, pattern_embedding).item()
            if similarity >= similarity_threshold:
                matched_intents.append((intent, similarity))

    matched_intents.sort(key=lambda x: x[1], reverse=True)
    return matched_intents


def rephrase_response(original_text):
    prompt = f"Rephrase to make it friendly, conversational, and to the point: '{original_text}'"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True,
                       max_length=tokenizer.model_max_length)
    inputs['attention_mask'] = inputs['input_ids'] != tokenizer.pad_token_id
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=50,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.5,
        do_sample=True,
    )

    rephrased_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return clean_text(rephrased_text, original_text)


def find_best_match_sql_query(user_input, table_name=None):
    query_embedding = sbert_model.encode(user_input)
    if table_name:
        cursor.execute("SELECT user_query, sql_query FROM query_sql_dataset WHERE table_name = %s", (table_name,))
    else:
        cursor.execute("SELECT user_query, sql_query FROM query_sql_dataset")

    best_match = None
    max_similarity = 0

    for user_query, sql_query in cursor.fetchall():
        stored_query_embedding = sbert_model.encode(user_query)
        similarity = util.pytorch_cos_sim(query_embedding, stored_query_embedding).item()
        if similarity > max_similarity:
            best_match = sql_query
            max_similarity = similarity
    return best_match if max_similarity >= SIMILARITY_THRESHOLD else None

def load_tags_and_tables():
    with db.cursor() as cursor:
        cursor.execute("SELECT tag, table_name FROM tags_and_tables")
        return [(tag, table_name) for tag, table_name in cursor.fetchall() if tag != "conversation_history"]

def execute_sql_query(query):
    try:
        with db.cursor() as cursor:
            cursor.execute(query)
            results = cursor.fetchall()
            return "\n".join(
                [" | ".join(str(item) for item in row) for row in results]) if results else "No records found."
    except mysql.connector.Error as err:
        return "SQL Error: " + str(err)


def get_recent_relevant_conversations(user_input, limit=5):
    user_embedding = sbert_model.encode(user_input)
    time_window = datetime.now() - timedelta(days=1)

    cursor.execute(
        "SELECT user_input, bot_response, timestamp FROM conversation_history WHERE timestamp >= %s ORDER BY timestamp DESC LIMIT %s",
        (time_window, limit))
    recent_conversations = cursor.fetchall()

    if not recent_conversations:
        return "No recent context found."  # Default context if no recent conversations

    relevant_conversations = []
    for user_text, bot_text, _ in recent_conversations:
        stored_embedding = sbert_model.encode(user_text)
        similarity = util.pytorch_cos_sim(user_embedding, stored_embedding).item()
        if similarity > 0.5:
            relevant_conversations.append((user_text, bot_text, similarity))

    relevant_conversations = sorted(relevant_conversations, key=lambda x: x[2], reverse=True)[:limit]
    context_str = " ".join([f"User: {user} Bot: {bot}" for user, bot, _ in relevant_conversations])
    return context_str

def adjust_tone_based_on_style(user_input, bot_response):
    doc = nlp(user_input)
    polite_words = ["please", "could", "would", "kindly", "thank you"]
    is_polite = any(word in user_input.lower() for word in polite_words)

    if is_polite:
        bot_response += " Thank you for reaching out!"
    else:
        bot_response = bot_response.replace(".", "!")

    return bot_response

def generate_response_with_context(user_input):
    max_length = 2048  # Adjusted max tokens for LLaMA 3
    max_new_tokens = 100

    # Initialize session-based conversation history if not present
    if 'session_conversations' not in session:
        session['session_conversations'] = []

    # Append current user input to session conversations
    session['session_conversations'].append({"user": user_input, "bot": None})

    # Include up to the last 5 exchanges in the prompt for context
    context_str = ""
    if len(session['session_conversations']) > 1:
        recent_conversations = session['session_conversations'][-5:]
        context_str = "\n".join([f"User: {conv['user']}\nBot: {conv['bot']}" if conv['bot'] else f"User: {conv['user']}" for conv in recent_conversations])

    # Construct prompt with user input and context
    prompt = f"{context_str}\nUser: {user_input}\nBot:"
    print(f"Debug: Generated Prompt: {prompt}")  # Debug information

    # Ensure prompt length stays within allowed limits
    prompt_tokens = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=max_length - max_new_tokens)
    truncated_prompt = tokenizer.decode(prompt_tokens[0], skip_special_tokens=True)

    if not truncated_prompt:
        print("Error: Prompt is empty after truncation.")
        return "I'm sorry, I couldn't process your request. Please try again."

    try:
        # Tokenize and generate response using LLaMA 3 model
        inputs = tokenizer(truncated_prompt, return_tensors="pt", padding=True, truncation=True, max_length=max_length - max_new_tokens)
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.5,
                do_sample=True,
            )

        bot_response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        print(f"Debug: Generated Response: {bot_response}")

        if not bot_response:
            bot_response = "I'm sorry, I couldn't process your request. Please try again."

        # Update conversation history in session
        session['session_conversations'][-1]["bot"] = bot_response
        save_conversation(user_input, bot_response)

        return bot_response

    except Exception as e:
        print(f"Error during generation: {e}")
        return "I'm sorry, there was an error in processing your request. Please try again later."


def get_recent_conversations(limit=1):
    query = "SELECT user_input, bot_response FROM conversation_history ORDER BY timestamp DESC LIMIT %s"
    try:
        cursor.execute(query, (limit,))
        results = cursor.fetchall()

        # Ensure the results are in the correct format
        conversations = []
        for result in results:
            if len(result) >= 2:  # Ensure each result has both user_input and bot_response
                conversations.append((result[0], result[1]))
            else:
                print("Warning: Skipping an improperly formatted entry:", result)

        return conversations
    except mysql.connector.Error as err:
        print(f"Database error: {err}")
        return []

def save_conversation(user_input, bot_response):
    global annoy_index_build_count
    query_id = str(uuid.uuid4())
    if bot_response is None:
        bot_response = "I'm sorry, I didn't understand your request."

    embedding = sbert_model.encode(user_input)
    convo_embeddings.append((query_id, user_input, bot_response))
    convo_index_map[annoy_index_build_count] = (query_id, user_input, bot_response)

    annoy_index.add_item(annoy_index_build_count, embedding)
    annoy_index_build_count += 1

    if annoy_index_build_count % 10 == 0:
        annoy_index.build(10)

    timestamp = datetime.now()
    cursor.execute(
        "INSERT INTO conversation_history (query_id, user_input, bot_response, timestamp) VALUES (%s, %s, %s, %s)",
        (query_id, user_input, bot_response, timestamp))
    db.commit()

def match_tags(user_input, tags_and_tables):
    user_embedding = sbert_model.encode(user_input, convert_to_tensor=True)
    matched_tags = []
    for tag, table_name in tags_and_tables:
        tag_embedding = sbert_model.encode(tag, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(user_embedding, tag_embedding).item()

        if similarity >= SIMILARITY_THRESHOLD:
            matched_tags.append((tag, table_name, similarity))
    matched_tags.sort(key=lambda x: x[2], reverse=True)
    return matched_tags

def get_clarification_and_execute(user_input, tags_and_tables):
    context_string = get_recent_conversations(limit=5)
    matched_tags = match_tags(user_input, tags_and_tables)
    if not matched_tags:
        top_matches = tags_and_tables[:3]
        suggestions = [f"'{tag}'" for tag, _ in top_matches]
        session['last_suggestions'] = [tag for tag, _ in top_matches]
        return f"I couldn't find exact details for '{user_input}'. Did you mean one of the following: {', '.join(suggestions)}? Please reply with your choice."

    table_name = matched_tags[0][1]
    if "details" in user_input.lower() or "information" in user_input.lower():
        fallback_query = f"SELECT * FROM {table_name};"
        try:
            with db.cursor() as cursor:
                cursor.execute(fallback_query)
                results = cursor.fetchall()
                return "\n".join([" | ".join(str(item) for item in row) for row in
                                  results]) if results else f"No records found in '{table_name}'."
        except mysql.connector.Error as err:
            return "There was an issue generating a valid SQL query. Please try rephrasing your request."

    prompt = (
        f"Context of recent conversation: '{context_string}'. "
        f"Generate an SQL SELECT query for the '{table_name}' table to retrieve relevant data based on this query: '{user_input}'. "
        "Only provide SQL code without additional text."
    )

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    inputs['attention_mask'] = inputs['input_ids'] != tokenizer.pad_token_id

    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=50,
        temperature=0.3,
        do_sample=True
    )

    generated_query = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    if "SELECT" in generated_query and "FROM" in generated_query:
        try:
            with db.cursor() as cursor:
                cursor.execute(generated_query)
                results = cursor.fetchall()
                return "\n".join([" | ".join(str(item) for item in row) for row in
                                  results]) if results else f"No records found in '{table_name}'."
        except mysql.connector.Error as err:
            return "An error occurred with the SQL query."

    return "Sorry, I couldn't generate a valid SQL query for your request."

# Function to split the user's input query into sub-queries using SpaCy
def split_query_spacy(user_input):
    doc = nlp(user_input)
    sub_queries = []
    sub_query = ""
    for token in doc:
        sub_query += token.text_with_ws
        if token.dep_ == 'cc' or token.is_punct:
            sub_queries.append(sub_query.strip())
            sub_query = ""
    if sub_query:
        sub_queries.append(sub_query.strip())
    return sub_queries

# Function to combine responses from sub-queries into a single coherent response
def combine_responses(sub_query_responses):
    response_set = set()
    combined_response = ""
    for _, response in sub_query_responses:
        if response not in response_set:
            response_set.add(response)
            combined_response += f" {response}"
    return combined_response.strip()


@app.route('/get', methods=["POST"])
def get_response():
    user_input = request.json.get("msg")
    last_suggestions = session.get('last_suggestions', [])

    # Check if the user is clarifying a table choice
    if last_suggestions and any(choice in user_input.lower() for choice in last_suggestions):
        session['selected_table'] = user_input.strip()
        del session['last_suggestions']
        return jsonify(
            {"response": f"Thank you for clarifying. I'll use '{session['selected_table']}' for further queries."})

    # Generate response with tags and context
    response = generate_response_with_context(user_input)
    save_conversation(user_input, response)

    return jsonify({"response": response, "feedback_request": "Rate the response and suggest improvements if needed."})

def save_query_to_dataset(user_query, sql_query, table_name, query_type):
    cursor.execute(
        "INSERT INTO query_sql_dataset (user_query, sql_query, table_name, query_type) VALUES (%s, %s, %s, %s)",
        (user_query, sql_query, table_name, query_type))
    cursor.execute(
        "INSERT INTO tags_and_tables (tag, table_name) VALUES (%s, %s) ON DUPLICATE KEY UPDATE table_name=VALUES(table_name)",
        (query_type, table_name))
    db.commit()

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.json
    user_input = data.get('user_input')
    bot_response = data.get('bot_response')
    rating = data.get('rating')

    cursor.execute(
        "INSERT INTO feedback (user_input, bot_response, rating) VALUES (%s, %s, %s)",
        (user_input, bot_response, rating)
    )
    db.commit()
    return jsonify({"message": "Thank you for your feedback!"})

@app.route('/')
def home():
    return render_template('chatbot.html')

if __name__ == "__main__":
    try:
        load_or_initialize_intents()
        logging.info("Intents loaded/initialized")
        app.run(debug=True)
    except Exception as e:
        logging.error(f"Error in main application loop: {e}")