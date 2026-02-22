import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import re
import base64
import pickle
import json
from collections import Counter
import streamlit.components.v1 as components

# Try importing TensorFlow gracefully
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# --- SET YOUR LOCAL IMAGE & FONT FILENAMES HERE ---
LOCAL_BACKGROUND_IMAGE = "wallpape.jpeg"
LOCAL_LOGO_IMAGE = "logo.png"
LOCAL_FONT_FILE = "Clash_Regular.otf"

# 1. Page Configuration & Styling
st.set_page_config(page_title="Clash Royale Predictor", layout="wide", initial_sidebar_state="collapsed")

def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        return None

def local_css(image_file, font_file):
    encoded_img = get_base64_of_bin_file(image_file)
    if encoded_img:
        bg_url = f"url('data:image/jpeg;base64,{encoded_img}')"
    else:
        bg_url = "none"

    encoded_font = get_base64_of_bin_file(font_file)
    if encoded_font:
        font_face_rule = f"""
        @font-face {{
            font-family: 'ClashFont';
            src: url('data:font/otf;base64,{encoded_font}') format('opentype');
        }}
        """
        font_family_name = "'ClashFont', sans-serif"
    else:
        font_face_rule = ""
        font_family_name = "'Arial', sans-serif"

    css_template = """
    <style>
    FONT_FACE_RULE
    
    /* Global Styling with New Font */
    .stApp, div[data-testid="stMarkdownContainer"] p, div[data-testid="stMarkdownContainer"] h1, 
    div[data-testid="stMarkdownContainer"] h2, div[data-testid="stMarkdownContainer"] h3, 
    div[data-testid="stMarkdownContainer"] h4 {
        font-family: FONT_FAMILY_NAME !important;
    }

    .stApp {
        background: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)), BACKGROUND_IMAGE_URL;
        background-size: 100vw 100vh; 
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: white;
    }
    
    .player-box-header {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.6);
    }

    .p1-box-style {
        background: linear-gradient(to bottom, #004488, #002244);
        border: 3px solid #5a9bd4;
    }
    .p2-box-style {
        background: linear-gradient(to bottom, #8B0000, #550000);
        border: 3px solid #e06c75;
    }
    .p3-box-style {
        background: linear-gradient(to bottom, #4a0080, #2a0040);
        border: 3px solid #9b59b6;
    }

    .player-title-text {
        font-family: FONT_FAMILY_NAME !important; 
        font-size: 36px;
        font-weight: bold;
        color: white;
        text-shadow: 3px 3px 0 #000, -1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000, 0 0 10px rgba(255,255,255,0.5);
        margin: 0;
        padding: 0;
    }
    
    .empty-slot {
        background: rgba(0, 0, 0, 0.4);
        border: 2px dashed #a9a9a9;
        border-radius: 8px;
        width: 100%;
        aspect-ratio: 3/4;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #a9a9a9;
        font-weight: bold;
        margin-bottom: 15px;
        box-shadow: inset 0 0 10px rgba(0,0,0,0.5);
    }

    .info-panel {
        background: rgba(20, 30, 48, 0.9);
        border: 2px solid #ffd700;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.9);
    }
    
    .card-name-centered {
        text-align: center;
        font-size: 16px;
        font-weight: bold;
        margin-top: 5px;
        margin-bottom: 8px;
        color: white;
        text-shadow: 1px 1px 2px black;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    div[data-testid="stButton"] button {
        width: 100%;
        background-color: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.3);
        color: white;
        padding: 4px 2px;
        font-size: 12px; 
        font-family: FONT_FAMILY_NAME !important;
        display: flex !important;
        justify-content: center !important; 
        align-items: center !important;
        margin: 0 auto;
    }
    div[data-testid="stButton"] button:hover {
        border: 1px solid #ffd700;
        color: #ffd700;
        background-color: rgba(255, 215, 0, 0.2);
    }
    
    [data-testid="column"] {
        padding-left: 0.2rem !important;
        padding-right: 0.2rem !important;
    }
    </style>
    """
    final_css = css_template.replace("BACKGROUND_IMAGE_URL", bg_url)
    final_css = final_css.replace("FONT_FACE_RULE", font_face_rule)
    final_css = final_css.replace("FONT_FAMILY_NAME", font_family_name)
    st.markdown(final_css, unsafe_allow_html=True)

local_css(LOCAL_BACKGROUND_IMAGE, LOCAL_FONT_FILE)

RARITY_ORDER = {
    "Champion": 1, "Legendary": 2, "Epic": 3, "Rare": 4, "Common": 5
}

# 2. Load Data Caches
@st.cache_data
def get_card_data():
    try:
        with open("cards_i18n.json", "r", encoding="utf-8") as f:
            response = json.load(f)
    except Exception as e:
        st.warning(f"Failed to load local json: {e}")
        return {}
    
    ALLOWED_IDS = {
        26000048, 26000004, 28000001, 26000021, 26000063, 26000010, 28000011, 26000087, 26000009, 26000085, 28000009, 26000055,
        26000060, 26000000, 28000008, 26000018, 26000051, 27000003, 26000011, 26000044, 26000036, 28000000, 28000015, 26000061,
        26000032, 28000004, 26000057, 26000058, 26000040, 26000062, 26000001, 26000026, 27000000, 27000006, 26000029, 26000030,
        26000059, 26000049, 26000065, 26000034, 26000056, 26000006, 26000052, 28000010, 26000083, 26000037, 27000012, 26000045,
        26000080, 26000007, 26000041, 26000033, 28000003, 26000014, 26000012, 26000017, 26000035, 26000064, 26000015, 26000042,
        27000007, 28000002, 26000002, 28000018, 26000020, 27000002, 27000001, 28000007, 27000004, 26000008, 26000013, 26000074,
        26000077, 28000005, 26000027, 28000012, 26000046, 28000006, 26000005, 26000038, 26000024, 26000019, 26000043, 28000017,
        27000013, 27000008, 26000039, 26000067, 26000022, 26000068, 27000010, 26000023, 27000009, 28000014, 26000016, 26000053,
        26000028, 26000084, 26000069, 26000050, 26000003, 26000072, 26000025, 28000013, 26000054, 26000031, 26000047, 28000016,
        27000005, 26000093
    }
    cards = {
        card['name']: {
            'key': card['key'], 
            'id': card['id'], 
            'rarity': card.get('rarity', 'Common').capitalize(),
            'type': card.get('type', 'Troop').capitalize(), 
            'elixir': card.get('elixir', 0)
        } for card in response if card.get('id') in ALLOWED_IDS
    }
    return cards

@st.cache_data
def load_csv_data():
    try:
        df_winrates = pd.read_csv("card_win_rates(1).csv")
        df_analytics = pd.read_csv("card_analytics_df.csv")
        df_synergy = pd.read_csv("synergy_datasetFinal.csv", index_col=0)
    except FileNotFoundError:
        df_winrates, df_analytics, df_synergy = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    return df_winrates, df_analytics, df_synergy

@st.cache_data(show_spinner="Booting up the Anti-Meta Engine (Parsing millions of rows)...")
def load_anti_meta_engine():
    try:
        dataset = pd.read_csv('20231106.zip', header=None)
        column_names = [
            "temp1", "temp2", "id1", "Trophies1", "Crowns1",
            "Card1-1", "Card1-2", "Card1-3", "Card1-4", "Card1-5", "Card1-6", "Card1-7", "Card1-8",
            "id2", "Trophies2", "Crowns2",
            "Card2-1", "Card2-2", "Card2-3", "Card2-4", "Card2-5", "Card2-6", "Card2-7", "Card2-8"
        ]
        dataset.columns = column_names
        dataset.drop(["temp1", "temp2", "id1", "id2"], axis=1, inplace=True, errors='ignore')
        dataset.dropna(inplace=True)

        p1_decks = dataset[[f'Card1-{i}' for i in range(1, 9)]].values
        p2_decks = dataset[[f'Card2-{i}' for i in range(1, 9)]].values
        p1_won = np.where(dataset['Crowns1'] > dataset['Crowns2'], 1, 0)
        p2_won = 1 - p1_won

        unique_cards = np.unique(p1_decks)
        global_wr = {}
        for card in unique_cards:
            p1_mask = (p1_decks == card).any(axis=1)
            p2_mask = (p2_decks == card).any(axis=1)
            matches = p1_mask.sum() + p2_mask.sum()
            wins = p1_won[p1_mask].sum() + p2_won[p2_mask].sum()
            global_wr[card] = wins / (matches + 1e-9)

        all_winning_decks = np.vstack((p1_decks[p1_won == 1], p2_decks[p2_won == 1]))
        winning_decks_list = [tuple(sorted(row)) for row in all_winning_decks]
        deck_counts = Counter(winning_decks_list)
        
        # Meta pool for Hate Engine (5000) and Oracle (1000)
        meta_pool_5000 = [list(deck) for deck, count in deck_counts.most_common(5000)]
        meta_pool_1000 = meta_pool_5000[:1000]
        
        return p1_decks, p2_decks, p1_won, global_wr, meta_pool_5000, meta_pool_1000
    except FileNotFoundError:
        return None, None, None, None, None, None

@st.cache_resource(show_spinner="Loading Deep Learning Model...")
def load_oracle_model():
    if not TF_AVAILABLE: return None
    try:
        return tf.keras.models.load_model('clash_royale_nn_model.keras')
    except Exception:
        return None

@st.cache_data(show_spinner="Loading Card Encodings...")
def load_card_mapping():
    try:
        with open('card_mapping.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

# Dictionary and Mappings Initialization
card_dict = get_card_data()
card_dict["Little Prince"] = {
    'key': 'little-prince', 
    'id': 26000093, 
    'rarity': 'Champion',
    'type': 'Troop',
    'elixir': 3
}
card_names = sorted(list(card_dict.keys()))
id_to_name = {v['id']: k for k, v in card_dict.items()}
name_to_id = {k: v['id'] for k, v in card_dict.items()}

df_winrates, df_analytics, synergy_df = load_csv_data()
p1_decks, p2_decks, p1_won, global_wr, meta_pool_5000, meta_pool_1000 = load_anti_meta_engine()
oracle_model = load_oracle_model()
card_to_idx = load_card_mapping()

if card_to_idx:
    idx_to_card = {idx: card_id for card_id, idx in card_to_idx.items()}
else:
    idx_to_card = {}

def get_img_url(card_name):
    if card_name in card_dict:
        return f"https://raw.githubusercontent.com/RoyaleAPI/cr-api-assets/master/cards/{card_dict[card_name]['key']}.png"
    return f'https://raw.githubusercontent.com/RoyaleAPI/cr-api-assets/master/cards/little-prince.png'

# --- NEW: ARCHETYPE & ELIXIR LOGIC ---
HOG_ID = 26000021
GOLEM_ID = 26000009
LAVA_ID = 26000029
BALLOON_ID = 26000006
X_BOW_ID = 27000008

def get_deck_metadata(deck_names):
    if len(deck_names) != 8:
        return None, None

    card_ids = [card_dict[name]['id'] for name in deck_names]
    card_set = set(card_ids)

    type_count = {"Troop": 0, "Spell": 0, "Building": 0}
    total_elixir = 0
    heavy_cards = 0      
    cheap_cards = 0      

    for name in deck_names:
        t = card_dict[name].get('type', 'Troop').capitalize()
        if t in type_count:
            type_count[t] += 1
        else:
            type_count[t] = 1

        elx = card_dict[name].get('elixir', 0)
        total_elixir += elx

        if elx >= 5:
            heavy_cards += 1
        if elx <= 2:
            cheap_cards += 1

    avg_elixir = total_elixir / 8

    troop = type_count.get("Troop", 0)
    spell = type_count.get("Spell", 0)
    building = type_count.get("Building", 0)

    archetype = "Control" # Default Fallback

    # Specific Archetypes
    if HOG_ID in card_set and avg_elixir <= 3.3:
        archetype = "Hog Cycle"
    elif GOLEM_ID in card_set and avg_elixir >= 4.0:
        archetype = "Golem Beatdown"
    elif LAVA_ID in card_set and BALLOON_ID in card_set:
        archetype = "LavaLoon"
    elif X_BOW_ID in card_set and building >= 1:
        archetype = "X-Bow Cycle"
    elif cheap_cards >= 3 and spell >= 2 and avg_elixir <= 3.5:
        archetype = "Log Bait"
    # General Archetypes
    elif avg_elixir >= 3.9 and heavy_cards >= 2:
        archetype = "Beatdown"
    elif avg_elixir <= 3.4 and cheap_cards >= 2:
        archetype = "Cycle"
    elif building >= 1 and avg_elixir <= 3.6:
        archetype = "Siege"
    elif 3.4 < avg_elixir < 4.0:
        archetype = "Control"

    return archetype, avg_elixir

def render_deck_metadata_panel(archetype, avg_elixir):
    st.markdown(f"""
        <div style='text-align: center; margin-top: 10px; margin-bottom: 20px; padding: 12px; 
                    background: rgba(20, 30, 48, 0.9); border-radius: 10px; border: 2px solid #5a9bd4; 
                    box-shadow: 0 4px 8px rgba(0,0,0,0.8); display: flex; justify-content: space-around;'>
            <div>
                <span style='color: #a9a9a9; font-size: 14px; text-transform: uppercase;'>Deck Archetype</span><br>
                <span style='color: #ffd700; font-size: 20px; font-weight: bold; text-shadow: 1px 1px #000;'>🛡️ {archetype}</span>
            </div>
            <div>
                <span style='color: #a9a9a9; font-size: 14px; text-transform: uppercase;'>Average Elixir</span><br>
                <span style='color: #00ff88; font-size: 20px; font-weight: bold; text-shadow: 1px 1px #000;'>💧 {avg_elixir:.1f}</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

def render_read_only_deck(deck):
    rows = [st.columns(4) for _ in range(2)]
    flat_cols = [col for row in rows for col in row]
    for i in range(8):
        with flat_cols[i]:
            if i < len(deck):
                st.image(get_img_url(deck[i]), use_container_width=True)
                st.markdown(f"<div class='card-name-centered'>{deck[i]}</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='empty-slot'>Empty</div>", unsafe_allow_html=True)
                
    if len(deck) == 8:
        archetype, avg_elx = get_deck_metadata(deck)
        if archetype and avg_elx is not None:
            render_deck_metadata_panel(archetype, avg_elx)

# --- ENGINE 1: I HATE THIS CARD ALGORITHM ---
def get_hate_card_counter(hated_card_id):
    if p1_decks is None:
        return None, "Dataset 20231106.csv missing.", None, None, None, None
        
    p2_has_hated = (p2_decks == hated_card_id).any(axis=1)
    p1_vs_hated_decks = p1_decks[p2_has_hated]
    p1_vs_hated_wins = p1_won[p2_has_hated]
    
    if len(p1_vs_hated_wins) == 0:
        return None, "Not enough data for this card.", None, None, None, None

    card_matches = {}
    card_wins = {}
    
    for i in range(8):
        cards_in_col = p1_vs_hated_decks[:, i]
        for card, win in zip(cards_in_col, p1_vs_hated_wins):
            card_matches[card] = card_matches.get(card, 0) + 1
            card_wins[card] = card_wins.get(card, 0) + win
            
    best_counter_card = None
    best_delta = -999
    best_wr = 0
    
    for card, matches in card_matches.items():
        if matches >= 500: 
            matchup_wr = card_wins[card] / matches
            baseline_wr = global_wr.get(card, 0.5)
            delta = matchup_wr - baseline_wr 
            
            if delta > best_delta:
                best_delta = delta
                best_wr = matchup_wr
                best_counter_card = card
                
    if best_counter_card is None:
        return None, "Not enough significant matchups found.", None, None, None, None
        
    counter_name = id_to_name.get(best_counter_card, "Unknown")
    global_winrate = global_wr.get(best_counter_card, 0)
    
    for deck in meta_pool_5000:
        if best_counter_card in deck:
            deck_names = [id_to_name.get(card, "Unknown") for card in deck]
            return deck, deck_names, counter_name, best_delta, best_wr, global_winrate
            
    return None, "No viable meta deck contains the counter card.", None, None, None, None

# --- ENGINE 2: NEURAL NETWORK ORACLE ---
def recommend_counter_deck(opponent_deck_raw):
    if oracle_model is None or card_to_idx is None:
        return [], 0, "Model or Mapping file missing! Ensure 'clash_royale_nn_model.keras' and 'card_mapping.pkl' are present."
    if meta_pool_1000 is None:
        return [], 0, "Meta pool not built. Ensure '20231106.csv' is present."

    mapped_opponent = [card_to_idx.get(card, 0) for card in opponent_deck_raw]
    opponent_batch = np.array([mapped_opponent] * len(meta_pool_1000), dtype=np.int32)
    
    mapped_meta_pool = []
    for deck in meta_pool_1000:
        safe_deck = [card_to_idx.get(c, c) if c > 500 else c for c in deck]
        mapped_meta_pool.append(safe_deck)
        
    meta_batch = np.array(mapped_meta_pool, dtype=np.int32)
    
    if len(oracle_model.inputs) != 2:
        return [], 0, "Neural Network Architecture Mismatch! Expected 2-Input model."
    
    probabilities = oracle_model.predict([opponent_batch, meta_batch], verbose=0).flatten()
    best_match_idx = np.argmax(probabilities)
    highest_win_prob = probabilities[best_match_idx]
    
    best_deck_mapped = meta_batch[best_match_idx]
    best_deck_raw = [idx_to_card.get(idx, idx) for idx in best_deck_mapped]
    best_deck_names = [id_to_name.get(card, "Unknown") for card in best_deck_raw]
    
    return best_deck_names, highest_win_prob, "Success"

# Initialize Session States
if 'deck1' not in st.session_state: st.session_state.deck1 = []
if 'deck2' not in st.session_state: st.session_state.deck2 = []
if 'deck3' not in st.session_state: st.session_state.deck3 = [] 
if 'editing_player' not in st.session_state: st.session_state.editing_player = None
if 'viewed_card' not in st.session_state: st.session_state.viewed_card = None

# ==========================================
# UI ROUTING
# ==========================================

if st.session_state.editing_player is None:
    # --- TAB NAVIGATION ---
    tab_battle, tab_counter, tab_oracle, tab_analytics, tab_optimizer = st.tabs([
        "⚔️ Battle Predictor", 
        "🛡️ I Hate This Card!", 
        "🔮 Oracle Counter", 
        "📊 Card Analytics",
        "⚒️ Deck Optimizer"
    ])
    
    # ---------------- TAB 1 ----------------
    with tab_battle:
        logo_base64 = get_base64_of_bin_file(LOCAL_LOGO_IMAGE)
        if logo_base64:
            st.markdown(f"<div style='text-align: center; margin-bottom: 20px;'><img src='data:image/png;base64,{logo_base64}' style='max-height: 250px;'></div>", unsafe_allow_html=True)
        else:
            st.markdown("<h1 style='text-align: center; text-shadow: 2px 2px #000;'>⚔️ CLASH ROYALE BATTLE PREDICTOR ⚔️</h1>", unsafe_allow_html=True)
        
        col1, mid, col2 = st.columns([1, 0.1, 1])
        
        with col1:
            st.markdown(f"""
                <div class='player-box-header p1-box-style'>
                    <h1 class='player-title-text'>PLAYER 1</h1>
                </div>
                """, unsafe_allow_html=True)
            if st.button("✏️ Edit Player 1 Deck", use_container_width=True, key="edit_p1"):
                st.session_state.editing_player = 1
                st.rerun()
            st.write("")
            render_read_only_deck(st.session_state.deck1)

        with col2:
            st.markdown(f"""
                <div class='player-box-header p2-box-style'>
                    <h1 class='player-title-text'>PLAYER 2</h1>
                </div>
                """, unsafe_allow_html=True)
            if st.button("✏️ Edit Player 2 Deck", use_container_width=True, key="edit_p2"):
                st.session_state.editing_player = 2
                st.rerun()
            st.write("")
            render_read_only_deck(st.session_state.deck2)

        st.markdown("---")
        if st.button("🔮 PREDICT WINNER", use_container_width=True, type="primary"):
            if len(st.session_state.deck1) == 8 and len(st.session_state.deck2) == 8:
                st.balloons()
                st.success("Analysis Complete: Player 1 has a 64% Win Probability!")
            else:
                st.error("⚠️ Both players must have exactly 8 cards in their deck before predicting!")
                
    # ---------------- TAB 2 ----------------
    with tab_counter:
        st.markdown("<h2 style='text-align: center; color: #ffd700; text-shadow: 2px 2px #000;'>🛡️ The Anti-Meta Engine</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Select a card that keeps ruining your day. Our algorithm will scan millions of matches to find its ultimate mathematical counter.</p>", unsafe_allow_html=True)
        
        st.write("")
        col_select, col_empty = st.columns([0.5, 0.5])
        with col_select:
            hated_card_name = st.selectbox("I hate playing against...", card_names)
            find_counter = st.button("🔍 Find Counter Deck", type="primary", use_container_width=True)
            
        if find_counter:
            if p1_decks is None:
                st.error("⚠️ Unable to run the engine. Make sure '20231106.csv' is in your deployment folder!")
            else:
                with st.spinner("Crunching win-rate deltas across millions of games..."):
                    deck_ids, best_deck_names, c_name, c_delta, c_wr, c_global_wr = get_hate_card_counter(name_to_id[hated_card_name])
                    
                    if not deck_ids:
                        st.warning(best_deck_names)
                    else:
                        st.markdown("---")
                        st.success(f"### ⚔️ ULTIMATE HARD COUNTER: {c_name}")
                        
                        col_stats1, col_stats2, col_stats3 = st.columns(3)
                        col_stats1.metric("Global Win Rate", f"{c_global_wr*100:.2f}%")
                        col_stats2.metric(f"Win Rate vs {hated_card_name}", f"{c_wr*100:.2f}%")
                        col_stats3.metric("Synergy Spike (Delta)", f"+{c_delta*100:.2f}%")
                        
                        st.markdown("<h3 style='color: #ffd700;'>Recommended Meta Deck</h3>", unsafe_allow_html=True)
                        render_read_only_deck(best_deck_names)

    # ---------------- TAB 3 ----------------
    with tab_oracle:
        st.markdown("<h2 style='text-align: center; color: #9b59b6; text-shadow: 2px 2px #000;'>🔮 The Oracle Search</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Input an opponent's entire deck. Our Deep Learning Neural Network will simulate 1,000 matches in milliseconds to generate the perfect mathematical counter deck!</p>", unsafe_allow_html=True)
        st.write("")

        col_oracle1, mid_oracle, col_oracle2 = st.columns([1, 0.1, 1])
        with col_oracle1:
            st.markdown(f"""
                <div class='player-box-header p3-box-style'>
                    <h1 class='player-title-text'>OPPONENT DECK</h1>
                </div>
                """, unsafe_allow_html=True)
            if st.button("✏️ Edit Opponent Deck", use_container_width=True, key="edit_p3"):
                st.session_state.editing_player = 3
                st.rerun()
            st.write("")
            render_read_only_deck(st.session_state.deck3)

            st.write("")
            if st.button("🔮 GENERATE COUNTER DECK", use_container_width=True, type="primary"):
                if len(st.session_state.deck3) != 8:
                    st.error("⚠️ The Opponent Deck must have exactly 8 cards!")
                else:
                    if not TF_AVAILABLE:
                        st.error("⚠️ TensorFlow is not installed. Please install it to use the Neural Network.")
                    else:
                        with st.spinner("Simulating 1,000 matchups using Deep Learning..."):
                            raw_input_ids = [name_to_id[name] for name in st.session_state.deck3]
                            recommended_deck, win_prob, msg = recommend_counter_deck(raw_input_ids)
                            
                            if recommended_deck:
                                st.session_state.oracle_result_deck = recommended_deck
                                st.session_state.oracle_result_prob = win_prob
                            else:
                                st.error(msg)
        
        with col_oracle2:
            st.markdown(f"""
                <div class='player-box-header p1-box-style'>
                    <h1 class='player-title-text'>ORACLE COUNTER</h1>
                </div>
                """, unsafe_allow_html=True)
            if 'oracle_result_deck' in st.session_state:
                st.success(f"✅ Target Locked! You have a **{st.session_state.oracle_result_prob*100:.2f}% Win Probability** with this deck.")
                render_read_only_deck(st.session_state.oracle_result_deck)
            else:
                st.info("Input an Opponent Deck and click Generate to see the Oracle's prediction.")

    # ---------------- TAB 4 (ANALYTICS) ----------------
    with tab_analytics:
        st.markdown("<h2 style='text-align: center; color: #ffd700; text-shadow: 2px 2px #000;'>📊 Card Analytics Dashboard</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Explore detailed statistics, top synergies, and hard counters for any card in the game.</p>", unsafe_allow_html=True)
        
        st.write("")
        col_grid, spacer_ana, col_stats = st.columns([0.35, 0.05, 0.60])

        with col_grid:
            st.markdown("### 🎴 Select a Card")
            search_ana = st.text_input("🔍 Search collection...", "", key="ana_search")
            ana_available = [c for c in card_names if search_ana.lower() in c.lower()]
            ana_available.sort(key=lambda c: (RARITY_ORDER.get(card_dict[c]['rarity'], 6), c))
            
            with st.container(height=600):
                grid_cols = st.columns(4)
                for i, card in enumerate(ana_available):
                    with grid_cols[i % 4]:
                        st.image(get_img_url(card), use_container_width=True)
                        st.markdown(f"<div class='card-name-centered'>{card}</div>", unsafe_allow_html=True)
                        if st.button("➕ Select", key=f"ana_btn_{card}", use_container_width=True):
                            st.session_state.ana_viewed_card = card

        with col_stats:
            if 'ana_viewed_card' in st.session_state:
                card = st.session_state.ana_viewed_card
                card_id = card_dict[card]['id']
                st.markdown(f"<div class='info-panel'>", unsafe_allow_html=True)
                
                head1, head2 = st.columns([0.3, 0.7])
                with head1:
                    st.image(get_img_url(card), use_container_width=True)
                with head2:
                    st.markdown(f"<h1 style='color:#ffd700; margin:0;'>{card}</h1>", unsafe_allow_html=True)
                    wr_val = "N/A"
                    if not df_winrates.empty and card_id in df_winrates['Card_ID'].values:
                        wr_val = f"{df_winrates.loc[df_winrates['Card_ID'] == card_id, 'Win_Rate'].values[0]:.1f}%"
                    st.markdown(f"<h3 style='color:lightgreen;'>Win Rate: {wr_val}</h3>", unsafe_allow_html=True)
                
                st.markdown("---")
                
                if not synergy_df.empty and card in synergy_df.columns:
                    synergy_series = synergy_df[card].drop(labels=[card], errors='ignore')
                    top_syns = synergy_series.sort_values(ascending=False).head(10).index.tolist()
                    top_cnts = synergy_series.sort_values(ascending=True).head(10).index.tolist()

                    st.markdown("#### 🔥 Top 10 Synergies")
                    s_cols = st.columns(5)
                    for idx, s_card in enumerate(top_syns):
                        with s_cols[idx % 5]:
                            st.image(get_img_url(s_card), use_container_width=True)
                            st.markdown(f"<div style='font-size:20px; text-align:center;'>{s_card}</div>", unsafe_allow_html=True)

                    st.write("")
                    st.markdown("#### 🛡️ Top 10 Hard Counters")
                    c_cols = st.columns(5)
                    for idx, c_card in enumerate(top_cnts):
                        with c_cols[idx % 5]:
                            st.image(get_img_url(c_card), use_container_width=True)
                            st.markdown(f"<div style='font-size:20px; text-align:center;'>{c_card}</div>", unsafe_allow_html=True)
                else:
                    st.info("Synergy data unavailable for this card.")
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='empty-slot' style='height: 400px;'>Select a card from the grid to view analytics</div>", unsafe_allow_html=True)

    # ---------------- TAB 5 (OPTIMIZER) ----------------
    with tab_optimizer:
        st.markdown("<h2 style='text-align: center; color: #ffd700; text-shadow: 2px 2px #000;'>⚒️ Deck Optimizer Engine</h2>", unsafe_allow_html=True)
        try:
            with open("optimizer.html", "r", encoding="utf-8") as f:
                opt_html = f.read()
            
            try:
                with open("cards_i18n.json", "rb") as json_f:
                    json_b64 = base64.b64encode(json_f.read()).decode("utf-8")
                json_data_uri = f"data:application/json;base64,{json_b64}"
                opt_html = opt_html.replace('"cards_i18n.json"', f'"{json_data_uri}"')
                opt_html = opt_html.replace('"https://royaleapi.github.io/cr-api-data/json/cards_i18n.json"', f'"{json_data_uri}"')
            except FileNotFoundError:
                st.warning("⚠️ 'cards_i18n.json' not found locally.")
            
            try:
                with open("synergy_datasetFinal.csv", "rb") as csv_f:
                    csv_b64 = base64.b64encode(csv_f.read()).decode("utf-8")
                csv_data_uri = f"data:text/csv;base64,{csv_b64}"
                opt_html = opt_html.replace('"synergy_datasetFinal.csv"', f'"{csv_data_uri}"')
            except FileNotFoundError:
                st.warning("⚠️ 'synergy_datasetFinal.csv' not found locally.")

            bg_b64 = get_base64_of_bin_file(LOCAL_BACKGROUND_IMAGE)
            if bg_b64:
                bg_data_uri = f"data:image/jpeg;base64,{bg_b64}"
                opt_html = opt_html.replace('"newWallpaper1.jpg"', f'"{bg_data_uri}"')
                opt_html = opt_html.replace('"foranalytics.jpg"', f'"{bg_data_uri}"')
                
            font_b64 = get_base64_of_bin_file(LOCAL_FONT_FILE)
            if font_b64:
                font_data_uri = f"data:font/opentype;base64,{font_b64}"
                opt_html = opt_html.replace('"Clash_Regular.otf"', f'"{font_data_uri}"')

            components.html(opt_html, height=1200, scrolling=True)
        except Exception as e:
            st.error(f"⚠️ Failed to load 'optimizer.html'. Error: {e}")

else:
    # --- EDIT VIEW ---
    player_num = st.session_state.editing_player
    if player_num == 1: active_deck = st.session_state.deck1
    elif player_num == 2: active_deck = st.session_state.deck2
    else: active_deck = st.session_state.deck3
    
    col_title, col_btn = st.columns([0.8, 0.2])
    with col_title:
        title_text = f"Building Deck for Player {player_num}" if player_num != 3 else "Building Opponent Deck"
        st.markdown(f"<h2 style='text-shadow: 2px 2px #000; color: #ffd700;'>{title_text}</h2>", unsafe_allow_html=True)
    with col_btn:
        if st.button("✅ Done Editing", use_container_width=True):
            st.session_state.editing_player = None
            st.session_state.viewed_card = None
            st.rerun()

    st.markdown("<h3 style='text-shadow: 1px 1px #000;'>Battle Deck</h3>", unsafe_allow_html=True)
    sel_cols = st.columns(8)
    for i in range(8):
        with sel_cols[i]:
            if i < len(active_deck):
                card = active_deck[i]
                st.image(get_img_url(card), use_container_width=True)
                st.markdown(f"<div class='card-name-centered'>{card}</div>", unsafe_allow_html=True)
                if st.button("➖ Remove", key=f"rem_{card}", use_container_width=True):
                    active_deck.remove(card)
                    if st.session_state.viewed_card == card:
                        st.session_state.viewed_card = None
                    st.rerun()
            else:
                st.markdown("<div class='empty-slot'>Empty</div>", unsafe_allow_html=True)

    # Show metadata panel here as well if the deck is fully built
    if len(active_deck) == 8:
        edit_archetype, edit_elx = get_deck_metadata(active_deck)
        if edit_archetype and edit_elx is not None:
             render_deck_metadata_panel(edit_archetype, edit_elx)

    st.markdown("---")
    col_info, spacer, col_collection = st.columns([0.35, 0.05, 0.60])
    
    with col_info:
        st.markdown("<h3 style='text-shadow: 1px 1px #000;'>Card Info</h3>", unsafe_allow_html=True)
        if st.session_state.viewed_card:
            card = st.session_state.viewed_card
            card_id = card_dict[card]['id']
            st.markdown(f"<div class='info-panel'>", unsafe_allow_html=True)
            
            c1, c2 = st.columns([0.4, 0.6])
            with c1:
                st.image(get_img_url(card), use_container_width=True)
            with c2:
                st.markdown(f"<h4 style='color:#ffd700; margin-bottom:10px;'>{card}</h4>", unsafe_allow_html=True)
                win_rate = "N/A"
                if not df_winrates.empty and card_id in df_winrates['Card_ID'].values:
                    val = df_winrates.loc[df_winrates['Card_ID'] == card_id, 'Win_Rate'].values[0]
                    win_rate = f"{val:.1f}%"
                st.markdown(f"**Win Rate:** <span style='color:lightgreen;'>{win_rate}</span><br>"
                            f"**Elixir:** {card_dict[card].get('elixir', '?')}<br>"
                            f"**Type:** {card_dict[card].get('type', '?')}", unsafe_allow_html=True)

            st.markdown("---")
            if not synergy_df.empty and card in synergy_df.columns:
                synergy_series = synergy_df[card].drop(labels=[card], errors='ignore')
                top_syn_names = synergy_series.sort_values(ascending=False).head(10).index.tolist()
                top_counter_names = synergy_series.sort_values(ascending=True).head(10).index.tolist()

                st.write("🔥 **Top 10 Synergies:**")
                for row_idx in range(2):
                    syn_cols = st.columns(5)
                    for col_idx in range(5):
                        list_idx = (row_idx * 5) + col_idx
                        if list_idx < len(top_syn_names):
                            with syn_cols[col_idx]:
                                syn_card = top_syn_names[list_idx]
                                st.image(get_img_url(syn_card), use_container_width=True)
                
                st.markdown("---")
                st.write("🛡️ **Hard Counters:**")
                for row_idx in range(2):
                    counter_cols = st.columns(5)
                    for col_idx in range(5):
                        list_idx = (row_idx * 5) + col_idx
                        if list_idx < len(top_counter_names):
                            with counter_cols[col_idx]:
                                count_card = top_counter_names[list_idx]
                                st.image(get_img_url(count_card), use_container_width=True)
                
                st.markdown("---")
                st.write("📊 **Synergy Strength Breakdown**")
                top_10_data = synergy_series.sort_values(ascending=False).head(10)
                fig = px.bar(top_10_data, x=top_10_data.values, y=top_10_data.index, orientation='h', color=top_10_data.values, color_continuous_scale='YlOrRd')
                fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=240, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white', size=10, family="ClashFont, Arial, sans-serif"), xaxis=dict(showgrid=False, visible=False, fixedrange=True), yaxis=dict(title=None, fixedrange=True), dragmode=False)
                fig.update_coloraxes(showscale=False)
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            else:
                st.caption("Synergy data for this card is not available in the matrix.")
            st.markdown("</div>", unsafe_allow_html=True)

    with col_collection:
        st.markdown("<h3 style='text-shadow: 1px 1px #000;'>Collection</h3>", unsafe_allow_html=True)
        search_query = st.text_input("🔍 Search for any card...", "")
        available_cards = [c for c in card_names if c not in active_deck and search_query.lower() in c.lower()]
        available_cards.sort(key=lambda c: (RARITY_ORDER.get(card_dict[c]['rarity'], 6), c))
        
        coll_cols = st.columns(4)
        for i, card in enumerate(available_cards):
            with coll_cols[i % 4]:
                st.image(get_img_url(card), use_container_width=True)
                st.markdown(f"<div class='card-name-centered'>{card}</div>", unsafe_allow_html=True)
                btn_col1, btn_col2 = st.columns(2)
                with btn_col1:
                    if st.button("➕ Select", key=f"sel_{card}", use_container_width=True):
                        if len(active_deck) < 8:
                            active_deck.append(card)
                            st.rerun()
                        else:
                            st.warning("Deck full!")
                with btn_col2:
                    if st.button("ℹ️ Info", key=f"inf_{card}", use_container_width=True):
                        st.session_state.viewed_card = card
                        st.rerun()
