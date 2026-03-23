"""
Download shrijayan/gov_myscheme from HuggingFace (2000+ PDF files) and merge with
a curated seed of key Indian government schemes, saving backend/data/schemes.json.

Run from project root:
    python scripts/download_data.py
"""
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download, list_repo_files
except ImportError:
    print("Install dependencies first: pip install -r backend/requirements.txt")
    sys.exit(1)

OUTPUT = Path(__file__).parent.parent / "backend" / "data" / "schemes.json"

# ---------------------------------------------------------------------------
# Curated seed schemes (major Indian government welfare schemes)
# ---------------------------------------------------------------------------
SEED_SCHEMES = [
    {
        "name": "PM-KISAN (प्रधानमंत्री किसान सम्मान निधि)",
        "description": "Direct income support of ₹6,000 per year to small and marginal farmer families, paid in three equal installments of ₹2,000 every four months.",
        "eligibility": "Small and marginal farmer families with combined landholding of up to 2 hectares. Excludes institutional land holders, farmer families holding constitutional posts, retired pensioners with monthly pension above ₹10,000, income tax payers.",
        "benefits": "₹6,000 per year financial benefit transferred directly to bank account in three installments of ₹2,000 each.",
        "application_process": "Register at pmkisan.gov.in or through Common Service Centres (CSC) or through agriculture department.",
        "documents": "Aadhaar card, land records (Khasra/Khatauni), bank account passbook, mobile number.",
        "official_link": "https://pmkisan.gov.in",
        "category": "Agriculture",
        "state": "Central",
    },
    {
        "name": "आयुष्मान भारत – प्रधानमंत्री जन आरोग्य योजना (PM-JAY)",
        "description": "World's largest health insurance scheme providing health coverage of ₹5 lakh per family per year for secondary and tertiary care hospitalisation.",
        "eligibility": "Families identified based on SECC 2011 database. Poor and vulnerable families from rural and urban areas. No restriction on family size or age.",
        "benefits": "Health cover of ₹5 lakh per family per year. Cashless treatment at empanelled hospitals. Pre-existing diseases covered from day one. No cap on family size.",
        "application_process": "Visit nearest empanelled hospital or Common Service Centre with Aadhaar card. Beneficiary identification through PM-JAY portal or Mera Swasthya app.",
        "documents": "Aadhaar card or ration card for identity verification.",
        "official_link": "https://pmjay.gov.in",
        "category": "Health",
        "state": "Central",
    },
    {
        "name": "प्रधानमंत्री आवास योजना – ग्रामीण (PMAY-G)",
        "description": "Provides financial assistance to rural homeless and those living in kutcha houses to construct a pucca house with basic amenities.",
        "eligibility": "Houseless families and families living in kutcha or dilapidated houses as per SECC 2011 data. SC/ST, minorities, disabled persons, women-headed households get priority.",
        "benefits": "₹1.20 lakh in plain areas and ₹1.30 lakh in hilly/difficult areas. Additional ₹12,000 for toilet under Swachh Bharat Mission. MGNREGS wage support of 90/95 days.",
        "application_process": "Apply through Gram Panchayat or Block Development Officer. Beneficiary list prepared by state governments based on SECC 2011.",
        "documents": "Aadhaar card, bank account, caste certificate if applicable, land ownership documents.",
        "official_link": "https://pmayg.nic.in",
        "category": "Housing",
        "state": "Central",
    },
    {
        "name": "प्रधानमंत्री उज्ज्वला योजना (PMUY)",
        "description": "Provides LPG connections to women from Below Poverty Line (BPL) and other deprived households to replace traditional chulhas with clean cooking fuel.",
        "eligibility": "Women above 18 years from BPL households. SC/ST households, PMAY beneficiaries, forest dwellers, most backward classes, tea garden workers, river island dwellers.",
        "benefits": "Free LPG connection with deposit-free cylinder and pressure regulator. First refill and hotplate (stove) provided at subsidised rates via EMI.",
        "application_process": "Apply at nearest LPG distributor or through official portal with required documents.",
        "documents": "Aadhaar card, BPL ration card or self-declaration, bank account details, address proof.",
        "official_link": "https://pmuy.gov.in",
        "category": "Women Welfare",
        "state": "Central",
    },
    {
        "name": "मनरेगा – महात्मा गांधी राष्ट्रीय ग्रामीण रोजगार गारंटी अधिनियम (MGNREGS)",
        "description": "Guarantees 100 days of wage employment in a financial year to every rural household whose adult members volunteer to do unskilled manual work.",
        "eligibility": "Any rural household. Adult members (18+) willing to do unskilled manual work. Job card required.",
        "benefits": "100 days of guaranteed wage employment per year. Wages paid at statutory minimum wages. Unemployment allowance if work not provided within 15 days.",
        "application_process": "Apply for Job Card at Gram Panchayat office. Then apply for work at Gram Panchayat or Block Programme Officer.",
        "documents": "Aadhaar card, residence proof, bank account or post office account, passport-size photographs.",
        "official_link": "https://nrega.nic.in",
        "category": "Employment",
        "state": "Central",
    },
    {
        "name": "प्रधानमंत्री जीवन ज्योति बीमा योजना (PMJJBY)",
        "description": "Renewable one-year life insurance scheme providing coverage for death due to any reason at very affordable premium.",
        "eligibility": "Age 18–50 years. Savings bank or post office account holders with Aadhaar. Must give auto-debit consent.",
        "benefits": "₹2 lakh life insurance cover for death due to any cause. Annual premium of just ₹436 (auto-debited from bank account).",
        "application_process": "Enroll through bank branch, online banking, or banking correspondents by submitting auto-debit consent form.",
        "documents": "Aadhaar card, bank account details.",
        "official_link": "https://jansuraksha.gov.in",
        "category": "Insurance",
        "state": "Central",
    },
    {
        "name": "प्रधानमंत्री सुरक्षा बीमा योजना (PMSBY)",
        "description": "Accidental death and disability insurance scheme at extremely low premium of ₹20 per year.",
        "eligibility": "Age 18–70 years. Savings bank or post office account holders with Aadhaar. Must give auto-debit consent.",
        "benefits": "₹2 lakh for accidental death or permanent total disability. ₹1 lakh for permanent partial disability. Annual premium only ₹20.",
        "application_process": "Enroll through bank branch, online banking, or banking correspondents.",
        "documents": "Aadhaar card, bank account details.",
        "official_link": "https://jansuraksha.gov.in",
        "category": "Insurance",
        "state": "Central",
    },
    {
        "name": "अटल पेंशन योजना (APY)",
        "description": "Pension scheme for workers in unorganised sector to ensure a defined minimum pension after age 60.",
        "eligibility": "Indian citizens aged 18–40 years. Should have savings bank or post office account. Not an income tax payer.",
        "benefits": "Guaranteed monthly pension of ₹1,000 to ₹5,000 after age 60 based on contribution. Government co-contributes 50% or ₹1,000 per year (whichever lower) for first 5 years for new subscribers.",
        "application_process": "Enroll through bank branch or online through net banking. Fill APY subscriber registration form.",
        "documents": "Aadhaar card, bank account, mobile number.",
        "official_link": "https://npscra.nsdl.co.in",
        "category": "Pension",
        "state": "Central",
    },
    {
        "name": "बेटी बचाओ बेटी पढ़ाओ",
        "description": "Scheme to address declining child sex ratio and promote welfare, education, and empowerment of girl children.",
        "eligibility": "Girl child at birth. Families with girl children in targeted districts.",
        "benefits": "Awareness programmes, conditional cash transfers for girl child education in some states, Sukanya Samriddhi Account for savings.",
        "application_process": "Sukanya Samriddhi Account can be opened at any post office or authorised bank branch for girl child up to 10 years.",
        "documents": "Birth certificate of girl child, Aadhaar card, parent/guardian ID and address proof.",
        "official_link": "https://wcd.nic.in/bbbp-schemes",
        "category": "Women Welfare",
        "state": "Central",
    },
    {
        "name": "सुकन्या समृद्धि योजना (SSY)",
        "description": "Small savings scheme for girl child providing high interest rate and tax benefits to secure her future education and marriage expenses.",
        "eligibility": "Girl child below 10 years of age. Account opened by parent or legal guardian. Maximum 2 accounts per family (3 in case of twins/triplets).",
        "benefits": "High interest rate (currently ~8.2% p.a.). Tax deduction under Section 80C. Maturity amount tax-free. Partial withdrawal allowed after age 18 for education.",
        "application_process": "Open account at any post office or authorised bank. Minimum deposit ₹250, maximum ₹1.5 lakh per year. Account matures 21 years from opening.",
        "documents": "Girl child birth certificate, parent/guardian Aadhaar and PAN, address proof.",
        "official_link": "https://www.indiapost.gov.in",
        "category": "Education",
        "state": "Central",
    },
    {
        "name": "प्रधानमंत्री मुद्रा योजना (PMMY)",
        "description": "Provides loans up to ₹10 lakh to non-corporate, non-farm small/micro enterprises through banks, NBFCs, and MFIs.",
        "eligibility": "Non-farm income-generating activities in manufacturing, trading, services. Small shopkeepers, vendors, artisans, small manufacturers, transporters.",
        "benefits": "Shishu (up to ₹50,000), Kishore (₹50,001–₹5 lakh), Tarun (₹5–₹10 lakh) category loans. No collateral required. Mudra RuPay card for working capital.",
        "application_process": "Apply at any public sector bank, regional rural bank, cooperative bank, MFI, NBFC. Online at udyamimitra.in.",
        "documents": "Aadhaar, PAN, address proof, business proof/activity details, bank statements.",
        "official_link": "https://www.mudra.org.in",
        "category": "Business",
        "state": "Central",
    },
    {
        "name": "स्वच्छ भारत मिशन – ग्रामीण (SBM-G)",
        "description": "Aims to achieve Open Defecation Free (ODF) status in rural areas by providing financial support to construct individual household toilets.",
        "eligibility": "Rural households without a toilet or with only kutcha toilet. Preference to SC/ST, small and marginal farmers, landless labourers, women-headed households, differently-abled persons.",
        "benefits": "Incentive of ₹12,000 for construction of individual household latrine. Linked with MGNREGS for wage support.",
        "application_process": "Apply through Gram Panchayat. Construction verified before release of incentive.",
        "documents": "Aadhaar card, bank account, BPL/APL status details.",
        "official_link": "https://sbm.gov.in",
        "category": "Sanitation",
        "state": "Central",
    },
    {
        "name": "राष्ट्रीय खाद्य सुरक्षा अधिनियम (NFSA) – पीडीएस",
        "description": "Provides subsidised food grains to approximately 81 crore people (75% rural and 50% urban population) through the Public Distribution System.",
        "eligibility": "Families identified by state governments as eligible under Priority Households (PHH) or Antyodaya Anna Yojana (AAY). PHH: 5 kg/person/month. AAY: 35 kg/family/month.",
        "benefits": "Rice at ₹3/kg, Wheat at ₹2/kg, coarse grains at ₹1/kg for PHH. AAY families get 35 kg at same subsidised rates. PM Garib Kalyan Anna Yojana provides additional free grains.",
        "application_process": "Apply for ration card at local food and supply office with required documents. Use existing ration card at nearest Fair Price Shop.",
        "documents": "Aadhaar card, residence proof, income proof, existing ration card (if any).",
        "official_link": "https://dfpd.gov.in",
        "category": "Food Security",
        "state": "Central",
    },
    {
        "name": "प्रधानमंत्री फसल बीमा योजना (PMFBY)",
        "description": "Comprehensive crop insurance scheme protecting farmers against crop loss due to natural calamities, pests, and diseases at very low premiums.",
        "eligibility": "All farmers including sharecroppers and tenant farmers growing notified crops. Compulsory for loanee farmers, voluntary for non-loanee farmers.",
        "benefits": "Insurance cover for pre-sowing to post-harvest losses. Premium: 2% for Kharif, 1.5% for Rabi, 5% for commercial/horticultural crops. Full sum insured coverage.",
        "application_process": "Apply through bank where crop loan is taken, or through CSC centers, or online at pmfby.gov.in.",
        "documents": "Aadhaar, bank passbook, land records, sowing certificate.",
        "official_link": "https://pmfby.gov.in",
        "category": "Agriculture",
        "state": "Central",
    },
    {
        "name": "दीन दयाल उपाध्याय ग्रामीण कौशल्य योजना (DDU-GKY)",
        "description": "Placement-linked skill development programme for rural youth from poor families aimed at providing wage employment.",
        "eligibility": "Rural youth aged 15–35 years (up to 45 for certain categories). From poor families (SECC 2011). SC/ST youth, minorities, women, differently-abled given priority.",
        "benefits": "Free skill training aligned to industry requirements. Minimum 70% placement guarantee. Post-placement support and career progression assistance.",
        "application_process": "Register at nearest DDU-GKY project implementing agency or at rudsetis.org. Contact district Rural Development office.",
        "documents": "Aadhaar card, income certificate, educational qualification certificates, bank account.",
        "official_link": "https://ddugky.gov.in",
        "category": "Employment",
        "state": "Central",
    },
    {
        "name": "राष्ट्रीय सामाजिक सहायता कार्यक्रम – वृद्धावस्था पेंशन (NSAP-IGNOAPS)",
        "description": "Central government pension scheme for Below Poverty Line elderly persons providing basic financial security in old age.",
        "eligibility": "Age 60 years and above. BPL household as determined by state government. No fixed income/financial support from other sources.",
        "benefits": "₹200/month for age 60–79 years, ₹500/month for age 80 years and above from central government. States add top-up amounts.",
        "application_process": "Apply at Gram Panchayat or Block Development Office or District Social Welfare Office with required documents.",
        "documents": "Aadhaar card, age proof (birth certificate/school certificate/doctor certificate), BPL certificate, bank account details.",
        "official_link": "https://nsap.nic.in",
        "category": "Pension",
        "state": "Central",
    },
    {
        "name": "जननी सुरक्षा योजना (JSY)",
        "description": "Safe motherhood intervention under National Health Mission promoting institutional delivery among pregnant women from BPL and SC/ST families.",
        "eligibility": "Pregnant women of all age groups for up to 2 live births. BPL pregnant women and SC/ST women in high-performing states. All women in low-performing states.",
        "benefits": "Cash benefit ₹1,400 in rural areas and ₹1,000 in urban areas for institutional delivery in LPS (low-performing states). ₹700 rural and ₹600 urban in HPS (high-performing states).",
        "application_process": "Register at nearest government health facility or ASHA worker. Deliver at government hospital or accredited private hospital.",
        "documents": "Aadhaar card, MCH card/ANC registration card, BPL card if applicable, bank account.",
        "official_link": "https://nhm.gov.in",
        "category": "Health",
        "state": "Central",
    },
    {
        "name": "प्रधानमंत्री कौशल विकास योजना (PMKVY)",
        "description": "Flagship skill development scheme to enable Indian youth to take up industry-relevant skill training and earn recognition for prior learning.",
        "eligibility": "Indian nationals aged 15–45. Unemployed youth, school/college dropouts. Prior learning assessment available for experienced workers.",
        "benefits": "Free short-term skill training. Monetary reward upon successful completion and assessment. Placement assistance. Government-funded training in sectors like IT, construction, health, retail.",
        "application_process": "Enroll at nearest PMKVY Training Centre or online at pmkvyofficial.org. Find nearest centre through skill India portal.",
        "documents": "Aadhaar card, educational qualification proof, bank account, passport-size photo.",
        "official_link": "https://pmkvyofficial.org",
        "category": "Employment",
        "state": "Central",
    },
    {
        "name": "मध्याह्न भोजन योजना (PM POSHAN)",
        "description": "Provides hot cooked nutritious meals to school children in government and government-aided schools to improve enrolment, retention, attendance and nutritional levels.",
        "eligibility": "Children studying in Classes I to VIII in government, government-aided schools. Children in Madrasas supported under Samagra Shiksha.",
        "benefits": "Daily nutritious cooked meal with 450 calories and 12g protein for primary (I-V) and 700 calories and 20g protein for upper primary (VI-VIII) children.",
        "application_process": "Automatically covered for all enrolled students. No separate application needed.",
        "documents": "School enrollment.",
        "official_link": "https://pmposhan.education.gov.in",
        "category": "Education",
        "state": "Central",
    },
    {
        "name": "ई-श्रम – असंगठित श्रमिक पंजीकरण",
        "description": "National database of unorganised workers providing a unique identity card (e-Shram card) to access social security benefits.",
        "eligibility": "Unorganised sector workers aged 16–59 years. Construction workers, migrant workers, gig workers, platform workers, street vendors, domestic workers, agricultural labourers. Not an EPFO/ESIC member.",
        "benefits": "Unique 12-digit UAN (Universal Account Number). ₹2 lakh accident insurance under PMSBY. Priority access to government schemes and social security benefits.",
        "application_process": "Register online at eshram.gov.in or through CSC centres using Aadhaar linked mobile number.",
        "documents": "Aadhaar card, Aadhaar-linked mobile number, bank account, occupation details.",
        "official_link": "https://eshram.gov.in",
        "category": "Employment",
        "state": "Central",
    },
]

# ---------------------------------------------------------------------------
# Known Indian states/UTs for state detection
# ---------------------------------------------------------------------------
KNOWN_STATES = {
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka",
    "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram",
    "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu",
    "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal",
    "Andaman and Nicobar Islands", "Chandigarh", "Dadra and Nagar Haveli",
    "Daman and Diu", "Delhi", "Jammu & Kashmir", "Jammu and Kashmir",
    "Ladakh", "Lakshadweep", "Puducherry",
}

# Section markers in the PDF (in order they appear)
SECTION_ORDER = [
    "Details",
    "Benefits",
    "Eligibility",
    "Exclusions",
    "Application Process",
    "Documents Required",
    "Frequently Asked Questions",
    "Sources And References",
]


def clean_text(text: str) -> str:
    """Remove garbled characters and normalize whitespace."""
    text = text.replace("ï»¿", "").replace("\x00", "").replace("Â", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_scheme_from_text(raw_text: str) -> dict | None:
    """
    Parse PDF text extracted from myscheme.gov.in PDFs.

    Structure: The PDF contains a navigation header (section names concatenated)
    followed by the actual content. The pattern is:
      {SchemeName}{nav junk}...{State}{SchemeName}{Tags...}Details{description}
      Benefits{benefits}Eligibility{eligibility}...
    """
    text = clean_text(raw_text)

    if not text:
        return None

    # --- Extract scheme name (first token before nav junk) ---
    # The name appears right at the start before "Are you sure"
    nav_start = text.find("Are you sure")
    if nav_start == -1:
        nav_start = text.find("Sign In")
    if nav_start == -1:
        nav_start = min(200, len(text))

    name_raw = text[:nav_start].strip()
    # Clean up any trailing noise
    name = re.split(r"\n", name_raw)[0].strip()
    if not name or len(name) < 4:
        return None

    # --- Find the real content start ---
    # After nav junk there's a pattern: {State}{SchemeName}{Tags}Details{content}
    # Look for the second occurrence of "Details" which marks content start.
    # Use plain substring search (not \b) because "Details" is often directly
    # concatenated with content text: "DetailsThe scheme..." with no space.
    details_positions = [m.start() for m in re.finditer(r"Details", text)]
    if len(details_positions) < 2:
        content_start = text.find("Details")
    else:
        content_start = details_positions[1]  # Second occurrence is actual content

    if content_start == -1:
        return None

    # Try to extract state from the text just before the second "Details"
    state = "Central"
    pre_content = text[max(0, content_start - 300):content_start]
    for s in KNOWN_STATES:
        if s in pre_content:
            state = s
            break

    # Extract content section starting from second "Details"
    content_text = text[content_start:]

    def get_section(sec_name: str) -> str:
        """Extract content for a section until the next known section."""
        start = content_text.find(sec_name)
        if start == -1:
            return ""
        start += len(sec_name)
        # Find earliest next section marker after current start
        end = len(content_text)
        for other in SECTION_ORDER:
            if other == sec_name:
                continue
            idx = content_text.find(other, start)
            if idx != -1 and idx < end:
                end = idx
        raw = content_text[start:end].strip()
        # If content is very short or is just nav words, skip
        if len(raw) < 10:
            return ""
        return raw

    description = get_section("Details")
    benefits = get_section("Benefits")
    eligibility = get_section("Eligibility")
    application_process = get_section("Application Process")
    documents = get_section("Documents Required")

    # Skip if we got essentially nothing
    if not description and not benefits and not eligibility:
        return None

    # Try to infer category from tags between second scheme name and Details
    scheme_name_in_content = content_text.find(name[:20])
    category = ""
    if scheme_name_in_content != -1:
        tag_region = content_text[scheme_name_in_content:scheme_name_in_content + 200]
        # Remove the scheme name itself
        tag_region = tag_region[len(name[:20]):]
        tag_region_clean = tag_region[:tag_region.find("Details")].strip()
        if tag_region_clean:
            # Tags are concatenated; take the last one as a rough category
            # Filter out common nav words
            nav_words = {"DBT", "Incentive", "Back", "Check", "Eligibility", "Apply", "Now"}
            tags = [t.strip() for t in re.findall(r"[A-Z][a-z A-Z&]+", tag_region_clean) if t.strip() not in nav_words]
            if tags:
                category = tags[-1].strip()

    return {
        "name": name,
        "description": description[:2000],
        "eligibility": eligibility[:1000],
        "benefits": benefits[:1000],
        "application_process": application_process[:1000],
        "documents": documents[:500],
        "official_link": "",
        "category": category,
        "state": state,
    }


def process_pdf(pdf_path: str) -> dict | None:
    """Open a cached PDF and extract scheme data."""
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            raw = ""
            for page in pdf.pages:
                raw += (page.extract_text() or "") + "\n"
        return extract_scheme_from_text(raw)
    except Exception:
        return None


def download_and_extract(repo_file: str) -> dict | None:
    """Download one PDF from HF and extract scheme data."""
    try:
        path = hf_hub_download(
            "shrijayan/gov_myscheme",
            filename=repo_file,
            repo_type="dataset",
        )
        return process_pdf(path)
    except Exception:
        return None


def main():
    try:
        import pdfplumber  # noqa: F401
    except ImportError:
        print("pdfplumber not found — installing...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "pdfplumber", "-q"], check=True)

    print("Listing files in shrijayan/gov_myscheme...")
    all_files = list(list_repo_files("shrijayan/gov_myscheme", repo_type="dataset"))
    # Only unique PDFs (skip 'copy' duplicates)
    pdf_files = sorted(
        {f for f in all_files if f.endswith(".pdf") and " copy" not in f and "copy." not in f}
    )
    print(f"Found {len(pdf_files)} unique PDFs to process.")

    pdf_schemes = []
    failed = 0

    print("Downloading and extracting (parallel, 8 workers)...")
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(download_and_extract, f): f for f in pdf_files}
        for i, future in enumerate(as_completed(futures), 1):
            scheme = future.result()
            fname = futures[future]
            if scheme and scheme.get("name"):
                pdf_schemes.append(scheme)
                if i % 100 == 0 or i <= 5:
                    print(f"  [{i}/{len(pdf_files)}] OK: {scheme['name'][:60]}")
            else:
                failed += 1
                if i % 100 == 0:
                    print(f"  [{i}/{len(pdf_files)}] failed: {fname}")

    print(f"\nExtracted {len(pdf_schemes)} schemes from PDFs ({failed} failed/empty).")

    # Merge: seed first (higher quality), then PDF-extracted
    seen = {s["name"].lower() for s in SEED_SCHEMES}
    merged = list(SEED_SCHEMES)
    for s in pdf_schemes:
        key = s["name"].lower()
        if key not in seen:
            # Fill required keys
            for k in ["name", "description", "eligibility", "benefits",
                      "application_process", "documents", "official_link", "category", "state"]:
                s.setdefault(k, "")
            merged.append(s)
            seen.add(key)

    print(f"Total schemes after merge: {len(merged)}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(merged)} schemes to {OUTPUT}")
    print("\nSample (first scheme):")
    print(json.dumps(merged[0], ensure_ascii=False, indent=2)[:400])


if __name__ == "__main__":
    main()
