"""Hindi text normalization for TTS input.

Converts numbers, currency, symbols, and common English terms into
spoken Hindi so that Indic Parler-TTS produces clean audio.
No external dependencies — pure re + dict lookups.
"""
import re

# ── Number-to-Hindi word tables ──────────────────────────────────────

_ONES = {
    0: "शून्य", 1: "एक", 2: "दो", 3: "तीन", 4: "चार",
    5: "पाँच", 6: "छह", 7: "सात", 8: "आठ", 9: "नौ",
    10: "दस", 11: "ग्यारह", 12: "बारह", 13: "तेरह", 14: "चौदह",
    15: "पंद्रह", 16: "सोलह", 17: "सत्रह", 18: "अठारह", 19: "उन्नीस",
    20: "बीस", 21: "इक्कीस", 22: "बाईस", 23: "तेईस", 24: "चौबीस",
    25: "पच्चीस", 26: "छब्बीस", 27: "सत्ताईस", 28: "अट्ठाईस", 29: "उनतीस",
    30: "तीस", 31: "इकतीस", 32: "बत्तीस", 33: "तैंतीस", 34: "चौंतीस",
    35: "पैंतीस", 36: "छत्तीस", 37: "सैंतीस", 38: "अड़तीस", 39: "उनतालीस",
    40: "चालीस", 41: "इकतालीस", 42: "बयालीस", 43: "तैंतालीस", 44: "चौवालीस",
    45: "पैंतालीस", 46: "छियालीस", 47: "सैंतालीस", 48: "अड़तालीस", 49: "उनचास",
    50: "पचास", 51: "इक्यावन", 52: "बावन", 53: "तिरपन", 54: "चौवन",
    55: "पचपन", 56: "छप्पन", 57: "सत्तावन", 58: "अट्ठावन", 59: "उनसठ",
    60: "साठ", 61: "इकसठ", 62: "बासठ", 63: "तिरसठ", 64: "चौंसठ",
    65: "पैंसठ", 66: "छियासठ", 67: "सड़सठ", 68: "अड़सठ", 69: "उनहत्तर",
    70: "सत्तर", 71: "इकहत्तर", 72: "बहत्तर", 73: "तिहत्तर", 74: "चौहत्तर",
    75: "पचहत्तर", 76: "छिहत्तर", 77: "सतहत्तर", 78: "अठहत्तर", 79: "उनासी",
    80: "अस्सी", 81: "इक्यासी", 82: "बयासी", 83: "तिरासी", 84: "चौरासी",
    85: "पचासी", 86: "छियासी", 87: "सतासी", 88: "अठासी", 89: "नवासी",
    90: "नब्बे", 91: "इक्यानवे", 92: "बानवे", 93: "तिरानवे", 94: "चौरानवे",
    95: "पचानवे", 96: "छियानवे", 97: "सत्तानवे", 98: "अट्ठानवे", 99: "निन्यानवे",
}


def _number_to_hindi(n: int) -> str:
    """Convert a non-negative integer to Hindi words."""
    if n < 0:
        return "ऋण " + _number_to_hindi(-n)
    if n <= 99:
        return _ONES[n]

    parts: list[str] = []

    # करोड़ (10^7)
    if n >= 10_000_000:
        crore = n // 10_000_000
        parts.append(_number_to_hindi(crore) + " करोड़")
        n %= 10_000_000

    # लाख (10^5)
    if n >= 100_000:
        lakh = n // 100_000
        parts.append(_number_to_hindi(lakh) + " लाख")
        n %= 100_000

    # हज़ार (10^3)
    if n >= 1_000:
        hazar = n // 1_000
        parts.append(_number_to_hindi(hazar) + " हज़ार")
        n %= 1_000

    # सौ (10^2)
    if n >= 100:
        sau = n // 100
        parts.append(_number_to_hindi(sau) + " सौ")
        n %= 100

    if n > 0:
        parts.append(_ONES[n])

    return " ".join(parts)


# ── Currency normalization ───────────────────────────────────────────

_CURRENCY_RE = re.compile(r"₹\s*([\d,]+(?:\.\d+)?)")


def _normalize_currency(text: str) -> str:
    """₹10,000 → दस हज़ार रुपये"""
    def _replace(m: re.Match) -> str:
        raw = m.group(1).replace(",", "")
        try:
            val = float(raw)
            int_part = int(val)
            hindi = _number_to_hindi(int_part)
            # Handle paise
            frac = raw.split(".")
            if len(frac) == 2 and int(frac[1]) > 0:
                paise = _number_to_hindi(int(frac[1]))
                return f"{hindi} रुपये {paise} पैसे"
            return f"{hindi} रुपये"
        except (ValueError, KeyError):
            return m.group(0)

    return _CURRENCY_RE.sub(_replace, text)


# ── Number normalization ─────────────────────────────────────────────

# Match standalone numbers (with optional commas), not preceded by ₹
_NUMBER_RE = re.compile(r"(?<!₹)\b(\d{1,3}(?:,\d{2,3})*(?:\.\d+)?)\b")


def _normalize_numbers(text: str) -> str:
    """Convert standalone numbers to Hindi words."""
    def _replace(m: re.Match) -> str:
        raw = m.group(1).replace(",", "")
        try:
            val = float(raw)
            int_part = int(val)
            if int_part > 99_99_99_99_999:  # too large, leave as-is
                return m.group(0)
            return _number_to_hindi(int_part)
        except (ValueError, KeyError):
            return m.group(0)

    return _NUMBER_RE.sub(_replace, text)


# ── Symbol normalization ─────────────────────────────────────────────

_SYMBOL_MAP = {
    "%": " प्रतिशत",
    "&": " और ",
    "+": " और ",
    "–": " से ",   # en-dash (range: 15–45 → 15 से 45)
    "—": " ",      # em-dash
    "/": " या ",   # slash (स्कूल/कॉलेज → स्कूल या कॉलेज)
}

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
_BRACKET_RE = re.compile(r"[(\[{](.*?)[)\]}]")


def _normalize_symbols(text: str) -> str:
    """Replace symbols with Hindi equivalents, strip URLs/emails, keep bracket content."""
    text = _URL_RE.sub("", text)
    text = _EMAIL_RE.sub("", text)  # remove email addresses (TTS can't pronounce them)
    text = _BRACKET_RE.sub(r" \1 ", text)
    for sym, replacement in _SYMBOL_MAP.items():
        text = text.replace(sym, replacement)
    # Remove remaining special chars that TTS can't handle
    text = re.sub(r"[#*_~`|<>]", " ", text)
    return text


# ── English transliteration ──────────────────────────────────────────

_ENGLISH_MAP = {
    "PM-KISAN": "पीएम किसान",
    "PM KISAN": "पीएम किसान",
    "PMKISAN": "पीएम किसान",
    "Aadhaar": "आधार",
    "AADHAAR": "आधार",
    "aadhaar": "आधार",
    "Aadhar": "आधार",
    "NREGA": "नरेगा",
    "MGNREGA": "मनरेगा",
    "OBC": "ओबीसी",
    "SC": "एससी",
    "ST": "एसटी",
    "BPL": "बीपीएल",
    "APL": "एपीएल",
    "LPG": "एलपीजी",
    "PMAY": "पीएमएवाई",
    "PM-KISAN": "पीएम किसान",
    "DBT": "डीबीटी",
    "KCC": "केसीसी",
    "MUDRA": "मुद्रा",
    "MSME": "एमएसएमई",
    "GST": "जीएसटी",
    "PAN": "पैन",
    "ITR": "आईटीआर",
    "EMI": "ईएमआई",
    "SHG": "एसएचजी",
    "NGO": "एनजीओ",
    "COVID": "कोविड",
    "Covid": "कोविड",
    "SECC": "एसईसीसी",
    "AYUSH": "आयुष",
    "ABHA": "आभा",
    "NPS": "एनपीएस",
    "EPF": "ईपीएफ",
    "EPFO": "ईपीएफओ",
    "UPI": "यूपीआई",
    "ATM": "एटीएम",
    "SBI": "एसबीआई",
    "RBI": "आरबीआई",
    "NABARD": "नाबार्ड",
    "SIDBI": "सिडबी",
    "KVIC": "केवीआईसी",
    "NSAP": "एनएसएपी",
    "PMJAY": "पीएमजेएवाई",
    "PMSBY": "पीएमएसबीवाई",
    "PMJJBY": "पीएमजेजेबीवाई",
    "APY": "एपीवाई",
    "Jan Dhan": "जन धन",
    "Swachh Bharat": "स्वच्छ भारत",
    "Skill India": "स्किल इंडिया",
    "Make in India": "मेक इन इंडिया",
    "Digital India": "डिजिटल इंडिया",
    "Ayushman Bharat": "आयुष्मान भारत",
    # Scheme names
    "PM POSHAN": "पीएम पोषण",
    "POSHAN": "पोषण",
    "PMGKAY": "पीएमजीकेएवाई",
    "PMGSY": "पीएमजीएसवाई",
    "PMEGP": "पीएमईजीपी",
    "PMSMA": "पीएमएसएमए",
    "PMFBY": "पीएमएफबीवाई",
    "MGNREGS": "मनरेगा",
    "NSAP": "एनएसएपी",
    "IGNOAPS": "आईजीएनओएपीएस",
    "IGNDPS": "आईजीएनडीपीएस",
    "IGNWPS": "आईजीएनडब्ल्यूपीएस",
    "NFBS": "एनएफबीएस",
    "SSA": "एसएसए",
    "RTE": "आरटीई",
    "MDM": "एमडीएम",
    "NHM": "एनएचएम",
    "ASHA": "आशा",
    "ANM": "एएनएम",
    "PHC": "पीएचसी",
    "CHC": "सीएचसी",
    "ICDS": "आईसीडीएस",
    "WCD": "डब्ल्यूसीडी",
    "BOCW": "बीओसीडब्ल्यू",
    "PDS": "पीडीएस",
    "FPS": "एफपीएस",
    "BPL": "बीपीएल",
    "NRLM": "एनआरएलएम",
    "DAY-NRLM": "डे एनआरएलएम",
    "NULM": "एनयूएलएम",
    "DAY-NULM": "डे एनयूएलएम",
    "SVEP": "एसवीईपी",
    "CGTMSE": "सीजीटीएमएसई",
    "KVPY": "केवीपीवाई",
    "INSPIRE": "इंस्पायर",
    "CSIR": "सीएसआईआर",
    "DRDO": "डीआरडीओ",
    "ISRO": "इसरो",
    # Common English words that appear in LLM output
    "Application": "एप्लीकेशन",
    "application": "एप्लीकेशन",
    "Scheme": "स्कीम",
    "scheme": "स्कीम",
    "Online": "ऑनलाइन",
    "online": "ऑनलाइन",
    "Offline": "ऑफलाइन",
    "offline": "ऑफलाइन",
    "Portal": "पोर्टल",
    "portal": "पोर्टल",
    "Website": "वेबसाइट",
    "website": "वेबसाइट",
    "Download": "डाउनलोड",
    "download": "डाउनलोड",
    "Registration": "रजिस्ट्रेशन",
    "registration": "रजिस्ट्रेशन",
    "Certificate": "सर्टिफिकेट",
    "certificate": "सर्टिफिकेट",
    "Form": "फॉर्म",
    "form": "फॉर्म",
    "Card": "कार्ड",
    "card": "कार्ड",
    "Bank": "बैंक",
    "bank": "बैंक",
    "Account": "अकाउंट",
    "account": "अकाउंट",
    "Income": "इनकम",
    "income": "इनकम",
    "District": "डिस्ट्रिक्ट",
    "district": "डिस्ट्रिक्ट",
    "Block": "ब्लॉक",
    "block": "ब्लॉक",
    "State": "स्टेट",
    "Central": "सेंट्रल",
    "Government": "गवर्नमेंट",
    "Subsidy": "सब्सिडी",
    "subsidy": "सब्सिडी",
    "Pension": "पेंशन",
    "pension": "पेंशन",
    "Insurance": "इंश्योरेंस",
    "insurance": "इंश्योरेंस",
    "Loan": "लोन",
    "loan": "लोन",
    "Grant": "ग्रांट",
    "grant": "ग्रांट",
    "Training": "ट्रेनिंग",
    "training": "ट्रेनिंग",
    "Skill": "स्किल",
    "skill": "स्किल",
    "Employment": "एम्प्लॉयमेंट",
    "employment": "एम्प्लॉयमेंट",
    "Health": "हेल्थ",
    "health": "हेल्थ",
    "Education": "एजुकेशन",
    "education": "एजुकेशन",
    "Scholarship": "स्कॉलरशिप",
    "scholarship": "स्कॉलरशिप",
    "Housing": "हाउसिंग",
    "housing": "हाउसिंग",
    "Agriculture": "एग्रीकल्चर",
    "agriculture": "एग्रीकल्चर",
    "Farmer": "फार्मर",
    "farmer": "फार्मर",
    "Rural": "रूरल",
    "Urban": "अर्बन",
    "Infrastructure": "इंफ्रास्ट्रक्चर",
    "Technology": "टेक्नोलॉजी",
    "Support": "सपोर्ट",
    "Welfare": "वेलफेयर",
    "National": "नेशनल",
    "Programme": "प्रोग्राम",
    "Program": "प्रोग्राम",
    "Mission": "मिशन",
    "Development": "डेवलपमेंट",
    "Department": "डिपार्टमेंट",
    "Ministry": "मिनिस्ट्री",
    "PM": "पीएम",
}

# Sort by length descending so longer matches take priority
_ENGLISH_PATTERNS = sorted(_ENGLISH_MAP.keys(), key=len, reverse=True)
_ENGLISH_RE = re.compile(
    "|".join(re.escape(k) for k in _ENGLISH_PATTERNS)
)

# ── Fallback: letter-by-letter transliteration for remaining Latin chars ──

_LETTER_MAP = {
    'A': 'ए', 'B': 'बी', 'C': 'सी', 'D': 'डी', 'E': 'ई',
    'F': 'एफ', 'G': 'जी', 'H': 'एच', 'I': 'आई', 'J': 'जे',
    'K': 'के', 'L': 'एल', 'M': 'एम', 'N': 'एन', 'O': 'ओ',
    'P': 'पी', 'Q': 'क्यू', 'R': 'आर', 'S': 'एस', 'T': 'टी',
    'U': 'यू', 'V': 'वी', 'W': 'डब्ल्यू', 'X': 'एक्स', 'Y': 'वाई',
    'Z': 'ज़ेड',
}

def _transliterate_remaining_english(text: str) -> str:
    """Convert any remaining Latin words to Hindi letter-by-letter spelling."""
    def _replace_word(m: re.Match) -> str:
        word = m.group(0)
        # If it's all uppercase and short (2-6 chars), spell it out as an acronym
        if word.isupper() and len(word) <= 6:
            return "".join(_LETTER_MAP.get(c, c) for c in word)
        # Otherwise just remove it (TTS will garble it anyway)
        return ""

    return re.sub(r'\b[A-Za-z]{2,}\b', _replace_word, text)


def _transliterate_known_english(text: str) -> str:
    """Replace known English terms with Hindi equivalents."""
    return _ENGLISH_RE.sub(lambda m: _ENGLISH_MAP[m.group(0)], text)


# ── Whitespace cleanup ───────────────────────────────────────────────

def _clean_whitespace(text: str) -> str:
    """Collapse multiple spaces and newlines."""
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"  +", " ", text)
    return text.strip()


# ── Master function ──────────────────────────────────────────────────

def normalize_for_tts(text: str) -> str:
    """Full normalization pipeline: currency → numbers → symbols → English → fallback → whitespace."""
    text = _normalize_currency(text)
    text = _normalize_numbers(text)
    text = _normalize_symbols(text)
    text = _transliterate_known_english(text)
    text = _transliterate_remaining_english(text)  # catch-all for leftover Latin words
    text = _clean_whitespace(text)
    return text
