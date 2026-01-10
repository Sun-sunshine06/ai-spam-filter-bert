import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
import re

print("=" * 50)
print("AI SMS spam firewall")
print("Initializing...")
print("=" * 50)

# === 1. Data loading ===
data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
print("Validating dataset...")

try:
    zip_path = keras.utils.get_file(
        "smsspamcollection.zip",
        data_url,
        extract=True,
        cache_subdir='sms_spam_v4'
    )
    extract_dir = os.path.dirname(zip_path)
    target_file = None
    for root, dirs, files in os.walk(extract_dir):
        for file in files:
            if "SMSSpamCollection" in file and not file.endswith(".zip"):
                target_file = os.path.join(root, file)
                break
    data_path = target_file if target_file else os.path.join(extract_dir, "SMSSpamCollection")

except Exception as e:
    print(f"Init error: {e}")
    exit()

# Read raw data
try:
    df = pd.read_csv(data_path, sep='\t', names=["label", "message"], on_bad_lines='skip', encoding='utf-8')
except Exception:
    df = pd.read_csv(data_path, sep='\t', names=["label", "message"], on_bad_lines='skip', encoding='latin-1')

df['label_id'] = df['label'].map({'ham': 0, 'spam': 1})
sentences = df['message'].astype(str).values
labels = df['label_id'].values

extra_spam = [
    # Crypto / investment scams
    "URGENT!! double your BTC now, claim at bit.ly/btc-claim",
    "Free USDT airdrop, verify wallet at http://bit.ly/usdt-drop",
    "El0n approved crypto giveaway, send 0.1 BTC to get 1 BTC",
    "Invest in crypto presale, 10k return in 24h, g00gle it",
    "BTC mining promo ends today!!! visit www.coin-bonus.cc",
    "USDT reward, connect wallet now: bit.ly/usdtbonus",
    "Limited ETH nodes, earn 5k/week, act now",
    "New coin launch, guaranteed x50, register at t.co/airdr0p",
    "Crypto tax refund available, claim here: bit.ly/crypt0tax",
    "Your wallet flagged, confirm seed at http://bit.ly/secure-wallet",

    # Pig butchering / romance scams
    "Hi dear, I am in Singapore, can teach you USDT trading",
    "We met online, my uncle shares a BTC strategy, join me",
    "Sweetheart, invest together in gold+crypto, profit daily",
    "I love you, but need quick transfer to release investment",
    "Pig butchering group: deposit first, big returns in 7 days",
    "Private mentor invites you to VIP crypto pool, 10k profit",
    "Trust me, I made 30k last week, you can too",
    "My cousin at bank shows me insider coin, click link",

    # Fake delivery / FedEx / parcel
    "FedEx: delivery failed, update address at bit.ly/fdx-verify",
    "Package on hold, pay small fee: http://bit.ly/parcelpay",
    "DHL notice! confirm parcel details: www.dhl-track.me",
    "UPS alert: item stopped, reschedule at t.co/ups-help",
    "Delivery issue, confirm now or return to sender",
    "Your package is waiting, verify zipcode at bit.ly/ship-now",
    "FedEx fee due $3.99, pay today to avoid return",
    "Courier update: link expired, re-verify at bit.ly/retry",

    # Bank alert / phishing
    "Bank Alert: unusual login, verify immediately at bit.ly/bank-ok",
    "Account locked, reset now: www.bank-secure-check.com",
    "Security Notice: confirm SSN to unlock online banking",
    "Your card $1,200 charge pending, stop it here: bit.ly/stop",
    "ATM withdrawal blocked, verify identity ASAP",
    "Payment failed, update card at http://bit.ly/cardfix",
    "Bank refund ready, confirm routing and ssn",
    "Final notice: account will be closed, act now",

    # Work from home / task scam
    "Work from home, earn $500/day, no exp, reply YES",
    "Part-time job: 10k/month, just like videos, apply now",
    "Remote task bonus $300 today, start at bit.ly/task-start",
    "Hiring now!!! easy cash, only phone needed",
    "We pay daily for data entry, msg me on WhatsApp",
    "Side hustle: 2k/week, limited spots, join fast",
    "Work online, payout USDT, instant withdraw",
    "Easy job offer, verify ID at bit.ly/job-verify",

    # Tax refund scams
    "Tax refund approved, claim $1,842 now: bit.ly/refund",
    "IRS notice: refund pending, verify info immediately",
    "Gov tax return ready, submit details at http://bit.ly/tx-ref",
    "Refund failed, re-enter bank info to receive funds",
    "Tax credit alert, confirm SSN to release payment",
    "You qualify for 2024 tax refund, act now",

    # Other high-risk patterns
    "You won $5,000!! claim prize at bit.ly/w1nner",
    "URGENT: update password now or lose access",
    "G00gle security alert, verify login at bit.ly/g00gle",
    "Congratulations! free gift card, click link",
    "Limited time offer, send $100 to receive $1000",
    "Lottery claim pending, pay fee to receive winnings",
    "Prize waiting, call +44 (0)20 7946 0958",
    "Cash app support: confirm code to unlock",
    "Your Apple ID suspended, restore at bit.ly/apple-fix",
    "Act now!! 0% loan approval in minutes",
    "Investment club invites you, deposit $200 to start",
    "Betting bonus 10k, verify account at t.co/bonus",
    "We detected virus, download cleaner at http://bit.ly/cleanpc",
    "Refund of $900 ready, confirm at www.safe-refund.net",
    "Pay $2 to unlock prize, limited time!!!",
    "Account verification needed, reply with OTP code",
    "Wire $300 to release international transfer",
    "New partner offer, send USDT to join pool",
    "Exclusive VIP signal group, subscribe at bit.ly/vipsignals",
    "Emergency: loved one detained, send money now",
    "Win a new phone, click here: bit.ly/free-phone",
    "Your loan is pre-approved $10k, verify now",

    # Targeted FN scenarios (sextortion / insider stock / money mule / stealth jobs)
    "I have your private video, pay silence money now",
    "I have your private video, send payment or I leak your photos",
    "We recorded you, pay silence money today",
    "Pay silence money or I leak your photos to contacts",
    "I will leak your photos unless you transfer crypto",
    "I have your private video, send BTC to keep it private",
    "Your private video will be published, pay to stop",
    "I have your webcam footage, pay silence money ASAP",
    "Inside info: this stock will explode tomorrow, buy now",
    "Guaranteed return on this stock, inside info from broker",
    "Secret inside info, stock explode 10x, act now",
    "Our analyst says stock explode, guaranteed return",
    "Insider tip: guaranteed return if you buy this stock",
    "Inside info on IPO, guaranteed return for early buyers",
    "This stock will explode, inside info just for you",
    "We need transaction assistance, move funds and keep a cut",
    "Move funds for us, keep a cut, fast transfer needed",
    "Looking for money mule, transaction assistance required",
    "Help move funds overseas, keep a cut for each transfer",
    "Can you move funds today? keep a cut, no questions",
    "Need account to move funds, you keep a cut",
    "Transaction assistance needed, move funds between accounts",
    "Texting job from home, reply for tasks",
    "Booking assistant role, simple tasks, start today",
    "Optimization task available, daily tasks, reply yes",
    "Looking for a booking assistant, flexible hours",
    "Texting job, remote assistant, no experience",
    "Optimization task, app ranking work, reply to start",
    "Booking assistant needed, online tasks only",
    "Texting job available, onboarding by message"
]

extra_labels = [1] * len(extra_spam)

sentences = np.concatenate([sentences, extra_spam])
labels = np.concatenate([labels, extra_labels])
# ==========================================================

# === 2. Tokenization (DistilBERT) ===
max_len = 128
print("Loading tokenizer...")
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# === 3. Build model (TFDistilBertForSequenceClassification) ===
print("Loading model...")
model = TFDistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2,
    use_safetensors=False
)

optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics = [tf.keras.metrics.SparseCategoricalAccuracy("accuracy")]
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# === 4. Train (fine-tuning) ===
print("\nStarting fine-tuning...")
indices = np.arange(len(sentences))
np.random.shuffle(indices)

split_idx = int(0.9 * len(indices))
train_idx = indices[:split_idx]
val_idx = indices[split_idx:]

train_texts = sentences[train_idx]
train_labels = labels[train_idx]
val_texts = sentences[val_idx]
val_labels = labels[val_idx]

train_encodings = tokenizer(
    list(train_texts),
    truncation=True,
    padding=True,
    max_length=max_len
)
val_encodings = tokenizer(
    list(val_texts),
    truncation=True,
    padding=True,
    max_length=max_len
)

train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), train_labels))
train_dataset = train_dataset.shuffle(1024).batch(16)
val_dataset = tf.data.Dataset.from_tensor_slices((dict(val_encodings), val_labels))
val_dataset = val_dataset.batch(16)

history = model.fit(
    train_dataset,
    epochs=2,
    validation_data=val_dataset,
    verbose=1
)


def calculate_hybrid_score(text, model_score):
    risk_score = float(model_score)
    text_lower = text.lower()

    # Regex detectors
    url_re = re.compile(r"(https?://\S+|www\.\S+|bit\.ly/\S+|t\.co/\S+)", re.I)
    domain_re = re.compile(r"^(?:https?://)?(?:www\.)?([^/\s:]+)", re.I)
    money_re = re.compile(r"(\$\s?\d+(?:\.\d+)?|\b\d{1,3}(?:,\d{3})+\b|\b\d+(?:k|K)\b)")
    phone_re = re.compile(r"(\+?\d[\d\-\s\(\)]{6,}\d)")

    hits = []

    def is_whitelisted_domain(domain):
        whitelist = {
            "google.com",
            "docs.google.com",
            "drive.google.com",
            "zoom.us",
            "teams.microsoft.com"
        }
        for wl in whitelist:
            if domain == wl or domain.endswith("." + wl):
                return True
        return False

    url_hit = False
    for m in url_re.finditer(text):
        url = m.group(0)
        dm = domain_re.search(url)
        domain = dm.group(1).lower() if dm else ""
        if domain and not is_whitelisted_domain(domain):
            url_hit = True
            hits.append("url")
            break

    money_hit = bool(money_re.search(text))
    phone_hit = bool(phone_re.search(text))

    if money_hit:
        risk_score += 0.25
        hits.append("money")
    if phone_hit:
        risk_score += 0.2
        hits.append("phone")

    # Context-aware crypto logic
    crypto_action_re = re.compile(
        r"(\b(bitcoin|btc|usdt|crypto|ethereum|eth)\b.{0,24}\b(send|pay|transfer|wallet|deposit|address|seed|claim|airdrop|withdraw)\b"
        r"|\b(send|pay|transfer|wallet|deposit|address|seed|claim|airdrop|withdraw)\b.{0,24}\b(bitcoin|btc|usdt|crypto|ethereum|eth)\b)",
        re.I
    )
    crypto_negative_re = re.compile(
        r"(\b(lost|loss|scammed|bad|invested|regret|complain|complained)\b.{0,12}\b(bitcoin|btc|usdt|crypto|ethereum|eth)\b"
        r"|\b(bitcoin|btc|usdt|crypto|ethereum|eth)\b.{0,12}\b(lost|loss|scammed|bad|invested|regret|complain|complained)\b)",
        re.I
    )
    if crypto_action_re.search(text):
        risk_score += 0.35
        hits.append("crypto_action")
    elif crypto_negative_re.search(text):
        risk_score += 0.05
        hits.append("crypto_negative")

    # Context-aware work-from-home logic
    wfh_re = re.compile(r"\bwork from home\b", re.I)
    wfh_context_re = re.compile(r"\b(earn|job|hiring|income|offer|salary|position)\b", re.I)
    if wfh_re.search(text) and wfh_context_re.search(text):
        risk_score += 0.35
        hits.append("wfh_recruit")

    # Pattern rules for missing FN categories
    pattern_rules = [
        ("sextortion", re.compile(r"\b(private video|leak your photos|pay silence money|silence money|webcam footage|we recorded you)\b", re.I), 0.7),
        ("stock_inside", re.compile(r"\b(inside info|insider tip|stock explode|guaranteed return)\b", re.I), 0.5),
        ("money_mule", re.compile(r"\b(move funds|keep a cut|transaction assistance|money mule)\b", re.I), 0.6),
        ("stealth_job", re.compile(r"\b(texting job|booking assistant|optimization task)\b", re.I), 0.4),
        ("bank_alert", re.compile(r"\b(bank alert|account locked|verify identity|reset password)\b", re.I), 0.25),
        ("delivery", re.compile(r"\b(fedex|dhl|ups|delivery failed|parcel|package)\b", re.I), 0.25),
        ("refund", re.compile(r"\b(refund|tax refund|irs notice)\b", re.I), 0.2),
        ("urgent", re.compile(r"\b(urgent|act now|immediately|final notice|asap)\b", re.I), 0.2)
    ]

    for name, pattern, penalty in pattern_rules:
        if pattern.search(text):
            risk_score += penalty
            hits.append(name)

    if url_hit:
        risk_score += 0.35

    # Weighted critical combo
    urgent_hit = re.search(r"\b(urgent|act now|immediately|final notice|asap)\b", text_lower) is not None
    if url_hit and urgent_hit:
        risk_score += 0.8
        hits.append("url+urgent")

    return min(risk_score, 0.999), hits


def predict_interactive():
    print("\n" + "=" * 50)
    print("AI SMS firewall")
    print("Type 'exit' to quit")
    print("=" * 50)

    while True:
        text = input("\nInput SMS (EN): ")

        if text.lower() in ['exit', 'quit']:
            break
        if not text.strip():
            continue

        inputs = tokenizer(
            [text],
            truncation=True,
            padding=True,
            max_length=max_len,
            return_tensors="tf"
        )
        logits = model(inputs).logits
        probs = tf.nn.softmax(logits, axis=-1).numpy()[0]
        ai_score = float(probs[1])

        final_score, hits = calculate_hybrid_score(text, ai_score)

        bar_len = 20
        filled_len = int(bar_len * final_score)
        filled_len = max(0, min(bar_len, filled_len))
        bar = '#' * filled_len + '-' * (bar_len - filled_len)
        score_percent = final_score * 100

        if final_score > 0.5:
            result = "SPAM"
            color_code = "\033[91m"
        else:
            result = "HAM"
            color_code = "\033[92m"

        reset_code = "\033[0m"

        print(f"   --------------------------------------------------")
        print(f"   {color_code}Result: {result}{reset_code}")
        print(f"   Risk score [{bar}] {score_percent:.2f}%")
        if hits:
            print(f"   Triggers: {', '.join(sorted(set(hits)))}")
        print(f"   (Model score: {ai_score:.4f})")
        print(f"   --------------------------------------------------")


predict_interactive()

