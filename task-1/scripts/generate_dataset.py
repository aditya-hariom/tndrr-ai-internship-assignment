#!/usr/bin/env python3
"""
scripts/generate_dataset.py

Generates a synthetic, high-diversity, schema-compliant dataset for fine-tuning
an SLM (e.g., Qwen2.5-0.5B-Instruct) on Natural Language to JSON action extraction.

Features:
- 1,600 total examples partitioned strictly into:
  - Train: 1,280 (80%)
  - Validation: 160 (10%)
  - Test: 160 (10%)
- 4 primary intents (create_reminder, schedule_meeting, set_timer, log_expense) + 1 fallback intent (unknown).
- Strict Global Deduplication & Partitioning: Zero query overlap across train, val, and test.
- Disjoint template families & entity sets for eval splits to evaluate genuine zero-shot slot extraction.
- Edge cases: explicit null values for unspecified optional fields, negative samples, out-of-domain queries.
- Deterministic generation using fixed random seed (seed=42).
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple, Set

SEED = 42
TOTAL_SAMPLES = 1600
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

SYSTEM_PROMPT = (
    "You are an AI assistant that extracts structured actions from natural language requests. "
    "Output ONLY a valid JSON object matching the schema, with no additional text or markdown formatting."
)

# -----------------------------------------------------------------------------
# Vocabulary Pools & Disjoint Partitions for Entity Isolation
# -----------------------------------------------------------------------------

NAMES_TRAIN = [
    "Alice", "Bob", "Charlie", "David", "Elena", "Fiona", "George", "Hannah",
    "Ian", "Julia", "Kevin", "Laura", "Marcus", "Nadia", "Oliver", "Priya",
    "Quinn", "Rahul", "Sarah", "Tom", "Uma", "Victor", "Wendy", "Xavier",
    "Yasmin", "Zack", "Chloe", "Liam", "Noah", "Emma", "Ava", "Sophia"
]
NAMES_VAL = ["Mason", "Lucas", "Aria", "Ethan", "Aiden", "Zoe", "Leo", "Maya"]
NAMES_TEST = ["Carlos", "Fatima", "Chen", "Mei", "Kavita", "Kenji", "Tariq", "Amara"]

TASKS_TRAIN = [
    "submit my assignment", "pay the electric bill", "call mom", "take insulin medication",
    "water the indoor plants", "pick up dry cleaning", "review pull request #402",
    "take out the recycling", "buy groceries for dinner", "renew car insurance",
    "schedule dentist appointment", "send weekly project report", "feed the cat",
    "check the oven", "order anniversary flowers", "refill prescription",
    "backup database servers", "email tax accountant", "cancel gym subscription",
    "update passport details", "book flight tickets", "prepare presentation slides",
    "charge laptop battery", "turn off the sprinkler", "reply to landlord email"
]
TASKS_VAL = [
    "pack suitcase for trip", "check credit card statement", "wash the car",
    "defrost chicken for dinner", "submit expense receipts"
]
TASKS_TEST = [
    "clean air filter", "change motor oil", "organize bookshelves",
    "confirm hotel reservation", "return library books"
]

TITLES_TRAIN = [
    "Q3 strategy review", "Sprint planning", "1-on-1 sync", "Design critique",
    "Product roadmap kickoff", "Engineering all-hands", "Weekly team sync",
    "Budget alignment", "Client demo", "Architecture review", "Quarterly retro",
    "Onboarding check-in", "Security audit debrief", "Marketing sync"
]
TITLES_VAL = ["Sales pipeline review", "API integration sync", "Post-mortem analysis"]
TITLES_TEST = ["Vendor negotiation", "UX research readout", "Customer advisory council"]

TIMER_LABELS_TRAIN = [
    "pasta", "tea", "laundry", "nap", "study block", "pomodoro focus", "plank",
    "pizza in the oven", "stretching", "face mask", "steep green tea"
]
TIMER_LABELS_VAL = ["cooling down", "meditation", "baking bread"]
TIMER_LABELS_TEST = ["boiled eggs", "quick break", "workout rest"]

MERCHANTS_TRAIN = {
    "food": ["Trader Joe's", "Whole Foods", "Chipotle", "Starbucks", "Subway", "Dominos", "Sweetgreen"],
    "transport": ["Uber", "Lyft", "Shell Gas Station", "Metro Transit", "Chevron", "CalTrain"],
    "utilities": ["ConEd Electric", "AT&T", "Comcast Xfinity", "City Water Service", "Verizon"],
    "entertainment": ["Netflix", "Spotify", "AMC Theatres", "Steam Games", "Nintendo eShop"],
    "shopping": ["Amazon", "Target", "Best Buy", "Apple Store", "Nike", "IKEA"],
    "other": ["Post Office", "CVS Pharmacy", "Walgreens", "UPS Store", "FedEx"]
}
MERCHANTS_VAL = {
    "food": ["Local Bistro", "McDonald's", "Shake Shack"],
    "transport": ["Delta Airlines", "Amtrak", "Hertz"],
    "utilities": ["T-Mobile", "Duke Energy"],
    "entertainment": ["Disney+", "Hulu", "Epic Games"],
    "shopping": ["Costco", "Walmart", "Zara"],
    "other": ["Office Depot", "Goodwill"]
}
MERCHANTS_TEST = {
    "food": ["In-N-Out", "Panera Bread", "Dunkin'"],
    "transport": ["United Airlines", "Lime Scooter", "Waymo"],
    "utilities": ["National Grid", "Spectrum"],
    "entertainment": ["HBO Max", "Audible", "Crunchyroll"],
    "shopping": ["REI Co-op", "Home Depot", "Uniqlo"],
    "other": ["Staples", "Ace Hardware"]
}

DATES_TRAIN = ["today", "tonight", "tomorrow", "this Friday", "next Monday", "next Tuesday", "on Wednesday", "next week"]
DATES_VAL = ["this Saturday", "next Thursday", "in two days", "on Sunday"]
DATES_TEST = ["next Friday", "this coming Monday", "on Dec 1st", "on Jan 15th"]

TIMES_POOL = [
    ("9 AM", "09:00"), ("9:30 AM", "09:30"), ("10:00 AM", "10:00"), ("11:15 AM", "11:15"),
    ("12:00 PM", "12:00"), ("1:00 PM", "13:00"), ("2:30 PM", "14:30"), ("3 PM", "15:00"),
    ("4:45 PM", "16:45"), ("5:00 PM", "17:00"), ("6:30 PM", "18:30"), ("8 PM", "20:00"),
    ("9:00 PM", "21:00"), ("10 PM", "22:00")
]

# -----------------------------------------------------------------------------
# Template Families by Split
# -----------------------------------------------------------------------------

REMINDER_TEMPLATES_TRAIN = [
    "Remind me to {task} {date_clause} {time_clause}.",
    "Please set a reminder to {task} {date_clause}.",
    "Don't let me forget to {task} {time_clause}.",
    "Create a {prio_word} reminder: {task} {date_clause} {time_clause}.",
    "Need a reminder to {task} {date_clause}.",
    "Alert me to {task} {time_clause}.",
    "Add reminder to {task} {date_clause} {time_clause}.",
    "Can you remind me about {task} {date_clause}?",
    "Set an alert to {task} {date_clause}.",
    "Prompt me to {task} {time_clause}."
]

REMINDER_TEMPLATES_VAL = [
    "Hey assistant, remind me to {task} {date_clause} {time_clause}.",
    "Kindly set an alert for me to {task} {time_clause}.",
    "I must remember to {task} {date_clause}."
]

REMINDER_TEMPLATES_TEST = [
    "Set {prio_word} priority reminder to {task} {date_clause} {time_clause}.",
    "Ping me to {task} {date_clause} {time_clause}.",
    "Please make sure I {task} {date_clause}."
]

MEETING_TEMPLATES_TRAIN = [
    "Schedule a meeting titled '{title}' with {participants} {date_clause} {time_clause}.",
    "Book a {duration} meeting with {participants} for {title} {date_clause}.",
    "Set up a sync with {participants} {date_clause} {time_clause}.",
    "Calendar invite: {title} with {participants} {time_clause}.",
    "Can you arrange a call with {participants} {date_clause} for {title}?",
    "Schedule {title} with {participants} for {duration} {date_clause}."
]

MEETING_TEMPLATES_VAL = [
    "Please organize a {duration} session with {participants} called {title} {date_clause} {time_clause}.",
    "Book time with {participants} {date_clause} {time_clause} regarding {title}."
]

MEETING_TEMPLATES_TEST = [
    "Set up a {title} discussion with {participants} {date_clause} {time_clause} for {duration}.",
    "Hold a meeting with {participants} {date_clause} at {time_clause} for {duration} regarding {title}."
]

TIMER_TEMPLATES_TRAIN = [
    "Set a timer for {duration_phrase}.",
    "Timer {duration_phrase} for {label}.",
    "Can you start a {duration_phrase} timer labeled {label}?",
    "Put {duration_phrase} on the clock for {label}.",
    "Start a timer for {duration_phrase}.",
    "Count down {duration_phrase} for {label}.",
    "Set {label} timer to {duration_phrase}."
]

TIMER_TEMPLATES_VAL = [
    "Set a clock for {duration_phrase} labeled '{label}'.",
    "Give me a {duration_phrase} countdown for {label}."
]

TIMER_TEMPLATES_TEST = [
    "Run a timer for {duration_phrase} marked as '{label}'.",
    "Start {duration_phrase} on the timer for {label}."
]

EXPENSE_TEMPLATES_TRAIN = [
    "Logged {currency_sym}{amount} for {merchant} under {category}.",
    "Spent {amount} {currency_code} at {merchant} on {category}.",
    "Track expense: {currency_sym}{amount} at {merchant}.",
    "Record {currency_sym}{amount} for {category} at {merchant}.",
    "Add an expense of {currency_sym}{amount} at {merchant} for {category}.",
    "I paid {currency_sym}{amount} for {category} at {merchant}."
]

EXPENSE_TEMPLATES_VAL = [
    "Charge {currency_sym}{amount} from {merchant} to {category}.",
    "Please log a payment of {amount} {currency_code} at {merchant}."
]

EXPENSE_TEMPLATES_TEST = [
    "I just spent {currency_sym}{amount} on {category} at {merchant}.",
    "Put down {currency_sym}{amount} for {merchant} under {category}."
]

# Disjoint Out-of-Domain questions
OOD_TRAIN = [
    "What is the capital of {country}?",
    "How do I cook {food_item} perfectly?",
    "Can you write a poem about {nature_theme}?",
    "Explain the theory of {scientific_concept} in simple terms.",
    "Tell me a joke about {profession}.",
    "What is the distance between {city1} and {city2}?",
    "How does a {tech_device} work internally?",
    "Recommend three good books on {topic}."
]

OOD_VAL = [
    "Translate '{foreign_phrase}' into Spanish.",
    "What are the health benefits of {healthy_habit}?",
    "Who directed the movie '{movie_title}'?"
]

OOD_TEST = [
    "How do I calculate the eigenvalues of a matrix in Python?",
    "What happened during the {historical_event}?",
    "Summarize the plot of Hamlet in two sentences."
]

COUNTRIES = ["Australia", "Canada", "Japan", "Brazil", "Germany", "South Africa", "Norway", "Argentina", "Egypt", "India", "France", "Kenya"]
FOOD_ITEMS = ["risotto", "sourdough bread", "salmon fillet", "chocolate soufflé", "scrambled eggs", "pasta carbonara", "caramelized onions"]
NATURE_THEMES = ["autumn leaves", "ocean waves", "mountain fog", "desert sunset", "thunderstorms", "starlit night", "spring blossoms"]
CONCEPTS = ["quantum entanglement", "general relativity", "photosynthesis", "neural networks", "thermodynamics", "plate tectonics"]
PROFESSIONS = ["software engineers", "accountants", "chefs", "electricians", "librarians", "astronomers"]
CITIES = [("Tokyo", "Kyoto"), ("New York", "London"), ("Paris", "Rome"), ("Berlin", "Vienna"), ("Sydney", "Melbourne"), ("San Francisco", "Seattle")]
TECH_DEVICES = ["microwave oven", "solid-state drive", "fiber optic cable", "GPS receiver", "jet engine", "lithium-ion battery"]
TOPICS = ["machine learning", "ancient Rome", "behavioral economics", "modern architecture", "astrophysics", "philosophy"]
FOREIGN_PHRASES = ["Good morning my friend", "Where is the library", "Thank you very much", "Have a safe flight", "See you tomorrow"]
HEALTHY_HABITS = ["intermittent fasting", "getting eight hours of sleep", "cold showers", "daily cardio exercise", "drinking more water"]
HISTORICAL_EVENTS = ["Renaissance", "Industrial Revolution", "Apollo 11 moon landing", "Fall of the Berlin Wall"]
MOVIE_TITLES = ["Inception", "Interstellar", "The Godfather", "Spirited Away", "Blade Runner 2049"]

AMBIGUOUS_TRAIN = [
    "Remind me about that project later.",
    "Set an alert for {name}.",
    "Schedule something for {name}.",
    "Book a sync tomorrow.",
    "Put time on the clock.",
    "Track my purchase from earlier.",
    "Log money spent.",
    "Can you remind me about that thing we discussed?",
    "Create calendar entry.",
    "Set the countdown.",
    "Make an appointment soon."
]

AMBIGUOUS_VAL = [
    "Hey can you set it for tomorrow?",
    "Book with {name} sometime.",
    "Add to calendar for 3pm."
]

AMBIGUOUS_TEST = [
    "Track twenty dollars.",
    "Expense for {name}.",
    "Alert me later."
]

# -----------------------------------------------------------------------------
# Entity Accessor by Split
# -----------------------------------------------------------------------------

def get_split_pools(split: str):
    if split == "train":
        return NAMES_TRAIN, TASKS_TRAIN, TITLES_TRAIN, TIMER_LABELS_TRAIN, MERCHANTS_TRAIN, DATES_TRAIN, REMINDER_TEMPLATES_TRAIN, MEETING_TEMPLATES_TRAIN, TIMER_TEMPLATES_TRAIN, EXPENSE_TEMPLATES_TRAIN, OOD_TRAIN, AMBIGUOUS_TRAIN
    elif split == "val":
        return NAMES_VAL, TASKS_VAL, TITLES_VAL, TIMER_LABELS_VAL, MERCHANTS_VAL, DATES_VAL, REMINDER_TEMPLATES_VAL, MEETING_TEMPLATES_VAL, TIMER_TEMPLATES_VAL, EXPENSE_TEMPLATES_VAL, OOD_VAL, AMBIGUOUS_VAL
    else:
        return NAMES_TEST, TASKS_TEST, TITLES_TEST, TIMER_LABELS_TEST, MERCHANTS_TEST, DATES_TEST, REMINDER_TEMPLATES_TEST, MEETING_TEMPLATES_TEST, TIMER_TEMPLATES_TEST, EXPENSE_TEMPLATES_TEST, OOD_TEST, AMBIGUOUS_TEST

# -----------------------------------------------------------------------------
# Generators
# -----------------------------------------------------------------------------

def generate_reminder(split: str, rnd: random.Random) -> Tuple[str, Dict[str, Any]]:
    names, tasks, titles, labels, merchants, dates, r_tmpls, _, _, _, _, _ = get_split_pools(split)
    tmpl = rnd.choice(r_tmpls)
    task = rnd.choice(tasks)

    include_date = rnd.random() < 0.75
    include_time = rnd.random() < 0.75
    include_prio = rnd.random() < 0.35

    date_str = rnd.choice(dates) if include_date else None
    time_raw, time_val = rnd.choice(TIMES_POOL) if include_time else (None, None)
    priority = rnd.choice(["low", "medium", "high"]) if include_prio else None

    date_clause = f"on {date_str}" if (date_str and not date_str.startswith("today") and not date_str.startswith("tonight") and not date_str.startswith("tomorrow") and not date_str.startswith("in two")) else (date_str or "")
    time_clause = f"at {time_raw}" if time_raw else ""
    prio_word = priority if priority else "normal"

    text = tmpl.format(task=task, date_clause=date_clause, time_clause=time_clause, prio_word=prio_word)
    text = " ".join(text.split()).replace(" on .", ".").replace(" at .", ".").replace(" .", ".")

    ground_truth = {
        "intent": "create_reminder",
        "task": task,
        "date": date_str,
        "time": time_val,
        "priority": priority
    }
    return text, ground_truth

def generate_meeting(split: str, rnd: random.Random) -> Tuple[str, Dict[str, Any]]:
    names, tasks, titles, labels, merchants, dates, _, m_tmpls, _, _, _, _ = get_split_pools(split)
    tmpl = rnd.choice(m_tmpls)
    title = rnd.choice(titles)

    k = min(len(names), rnd.choice([1, 2, 3]))
    chosen_names = rnd.sample(names, k)
    if len(chosen_names) == 1:
        part_str = chosen_names[0]
    elif len(chosen_names) == 2:
        part_str = f"{chosen_names[0]} and {chosen_names[1]}"
    else:
        part_str = f"{chosen_names[0]}, {chosen_names[1]} and {chosen_names[2]}"

    include_date = rnd.random() < 0.80
    include_time = rnd.random() < 0.80
    include_duration = rnd.random() < 0.60

    date_str = rnd.choice(dates) if include_date else None
    time_raw, time_val = rnd.choice(TIMES_POOL) if include_time else (None, None)
    duration_min = rnd.choice([15, 30, 45, 60, 90]) if include_duration else None
    duration_str = f"{duration_min}-minute" if duration_min else "quick"

    date_clause = f"on {date_str}" if (date_str and not date_str.startswith("today") and not date_str.startswith("tomorrow") and not date_str.startswith("in two")) else (date_str or "")
    time_clause = f"at {time_raw}" if time_raw else ""

    text = tmpl.format(title=title, participants=part_str, duration=duration_str, date_clause=date_clause, time_clause=time_clause)
    text = " ".join(text.split()).replace(" .", ".")

    ground_truth = {
        "intent": "schedule_meeting",
        "title": title,
        "participants": chosen_names,
        "date": date_str,
        "time": time_val,
        "duration_minutes": duration_min
    }
    return text, ground_truth

def generate_timer(split: str, rnd: random.Random) -> Tuple[str, Dict[str, Any]]:
    names, tasks, titles, labels, merchants, dates, _, _, t_tmpls, _, _, _ = get_split_pools(split)
    tmpl = rnd.choice(t_tmpls)

    dur_choices = [
        ("30 seconds", 30),
        ("45 seconds", 45),
        ("1 minute", 60),
        ("2 minutes", 120),
        ("3 minutes", 180),
        ("5 minutes", 300),
        ("7 minutes", 420),
        ("10 minutes", 600),
        ("12 minutes", 720),
        ("15 minutes", 900),
        ("20 minutes", 1200),
        ("25 minutes", 1500),
        ("30 minutes", 1800),
        ("45 minutes", 2700),
        ("50 minutes", 3000),
        ("1 hour", 3600),
    ]
    dur_phrase, total_seconds = rnd.choice(dur_choices)

    # For train, some can omit label; for val/test ensure label is always present or clearly parameterized
    include_label = rnd.random() < 0.80 or split in ["val", "test"]
    label = rnd.choice(labels) if include_label else None

    text = tmpl.format(duration_phrase=dur_phrase, label=label or "timer")
    text = " ".join(text.split()).replace(" .", ".")

    ground_truth = {
        "intent": "set_timer",
        "duration_seconds": total_seconds,
        "label": label
    }
    return text, ground_truth

def generate_expense(split: str, rnd: random.Random) -> Tuple[str, Dict[str, Any]]:
    names, tasks, titles, labels, merchants, dates, _, _, _, e_tmpls, _, _ = get_split_pools(split)
    tmpl = rnd.choice(e_tmpls)

    category = rnd.choice(list(merchants.keys()))
    merchant = rnd.choice(merchants[category])

    amount = round(rnd.choice([3.50, 4.99, 8.25, 12.00, 15.75, 23.40, 31.00, 42.50, 58.20, 79.99, 115.00, 230.50, 340.00]), 2)
    currency = rnd.choice(["USD", "EUR", "GBP", "INR", "CAD"])
    sym_map = {"USD": "$", "EUR": "€", "GBP": "£", "INR": "₹", "CAD": "C$"}
    currency_sym = sym_map.get(currency, "$")

    include_category = rnd.random() < 0.85
    include_merchant = rnd.random() < 0.85 or split in ["val", "test"]

    text = tmpl.format(
        amount=amount,
        currency_code=currency,
        currency_sym=currency_sym,
        merchant=merchant if include_merchant else "the store",
        category=category if include_category else "miscellaneous"
    )
    text = " ".join(text.split()).strip()

    ground_truth = {
        "intent": "log_expense",
        "amount": amount,
        "currency": currency,
        "category": category if include_category else None,
        "merchant": merchant if include_merchant else None
    }
    return text, ground_truth

def generate_unknown(split: str, rnd: random.Random) -> Tuple[str, Dict[str, Any]]:
    names, _, _, _, _, _, _, _, _, _, ood_tmpls, ambig_tmpls = get_split_pools(split)

    if rnd.random() < 0.70:
        tmpl = rnd.choice(ood_tmpls)
        c1, c2 = rnd.choice(CITIES)
        text = tmpl.format(
            country=rnd.choice(COUNTRIES),
            food_item=rnd.choice(FOOD_ITEMS),
            nature_theme=rnd.choice(NATURE_THEMES),
            scientific_concept=rnd.choice(CONCEPTS),
            profession=rnd.choice(PROFESSIONS),
            city1=c1, city2=c2,
            tech_device=rnd.choice(TECH_DEVICES),
            topic=rnd.choice(TOPICS),
            foreign_phrase=rnd.choice(FOREIGN_PHRASES),
            healthy_habit=rnd.choice(HEALTHY_HABITS),
            historical_event=rnd.choice(HISTORICAL_EVENTS),
            movie_title=rnd.choice(MOVIE_TITLES),
        )
        err_type = "unsupported_intent"
    else:
        tmpl = rnd.choice(ambig_tmpls)
        text = tmpl.format(name=rnd.choice(names))
        err_type = "insufficient_information"

    text = " ".join(text.split()).strip()
    ground_truth = {
        "intent": "unknown",
        "error": err_type,
        "raw_query": text
    }
    return text, ground_truth

# -----------------------------------------------------------------------------
# Split Generator with Global Deduplication
# -----------------------------------------------------------------------------

def generate_split_samples(
    split_name: str,
    count: int,
    global_seen: Set[str],
    rnd: random.Random
) -> List[Dict[str, Any]]:
    n_core = int(count * 0.235)
    n_unknown = count - (n_core * 4)

    generators = [
        (generate_reminder, (split_name, rnd), n_core),
        (generate_meeting, (split_name, rnd), n_core),
        (generate_timer, (split_name, rnd), n_core),
        (generate_expense, (split_name, rnd), n_core),
        (generate_unknown, (split_name, rnd), n_unknown),
    ]

    records = []
    for gen_func, args, target_num in generators:
        created = 0
        attempts = 0
        while created < target_num and attempts < target_num * 100:
            attempts += 1
            user_text, gt = gen_func(*args)
            norm_text = user_text.strip().lower()
            if norm_text in global_seen:
                continue
            global_seen.add(norm_text)

            sample_id = f"{split_name}_{len(records)+1:04d}"
            sample = {
                "id": sample_id,
                "split": split_name,
                "intent": gt["intent"],
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": json.dumps(gt, ensure_ascii=False)}
                ],
                "ground_truth": gt
            }
            records.append(sample)
            created += 1

    rnd.shuffle(records)
    for idx, r in enumerate(records):
        r["id"] = f"{split_name}_{idx+1:04d}"
    return records

def main():
    print("=" * 70)
    print("STARTING DATASET GENERATION (ZERO-LEAKAGE ARCHITECTURE)")
    print(f"Random Seed: {SEED}")
    print(f"Total Target Size: {TOTAL_SAMPLES} (Train: {int(TOTAL_SAMPLES*TRAIN_RATIO)}, Val: {int(TOTAL_SAMPLES*VAL_RATIO)}, Test: {int(TOTAL_SAMPLES*TEST_RATIO)})")
    print("=" * 70)

    rnd = random.Random(SEED)
    global_seen: Set[str] = set()

    n_train = int(TOTAL_SAMPLES * TRAIN_RATIO)  # 1280
    n_val = int(TOTAL_SAMPLES * VAL_RATIO)      # 160
    n_test = int(TOTAL_SAMPLES * TEST_RATIO)    # 160

    train_data = generate_split_samples("train", n_train, global_seen, rnd)
    val_data = generate_split_samples("val", n_val, global_seen, rnd)
    test_data = generate_split_samples("test", n_test, global_seen, rnd)

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True, parents=True)

    for split_data, filename in [
        (train_data, "train.jsonl"),
        (val_data, "validation.jsonl"),
        (test_data, "test.jsonl")
    ]:
        filepath = data_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            for row in split_data:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Wrote {len(split_data):>5} samples to {filepath}")

    print("=" * 70)
    print("Dataset Generation Complete!")

if __name__ == "__main__":
    main()
