import random
import pandas as pd

# Define categories
categories = ["administrative", "criminal", "civil", "constitutional", "family", "commercial"]

# Define templates with dynamic placeholders
# templates = {
#     "administrative": [
#         "Mr. {name} challenges the new municipal tax regulation in {place}.",
#         "Government hiring procedure disputed by {name} from {place}."
#     ],
#     "criminal": [
#         "{name} is charged with theft at {place} mall.",
#         "Police arrest {name} in {place} for organized financial fraud."
#     ],
#     "civil": [
#         "Property ownership between {name1} and {name2} contested in {place} court.",
#         "{name1} sues {name2} over sudden rental agreement termination."
#     ],
#     "constitutional": [
#         "Petition filed by {name} regarding violation of free speech rights.",
#         "Citizens in {place} protest changes in election laws."
#     ],
#     "family": [
#         "{name1} and {name2} file for divorce citing irreconcilable differences.",
#         "Custody battle between {name1} and {name2} in {place} family court."
#     ],
#     "commercial": [
#         "{name1} files lawsuit against {name2} for breach of partnership terms.",
#         "Startup {company} sues ex-employee {name} for leaking trade secrets."
#     ]
# }

templates = {
    "administrative": [
        "Mr. {name} challenges the new municipal tax regulation in {place}.",
        "Government hiring procedure disputed by {name} from {place}.",
        "An appeal was filed by {name} against a land acquisition order in {place}.",
        "The department’s decision on pension disbursement was contested by {name}.",
        "Allegations of bias in exam results published by the {place} board are raised.",
        "The transport authority in {place} faces complaints over revoked permits."
    ],
    "criminal": [
        "{name} is charged with theft at {place} mall.",
        "Police arrest {name} in {place} for organized financial fraud.",
        "An FIR was lodged against {name} for illegal possession of firearms.",
        "Investigation begins on the kidnapping case involving {name} in {place}.",
        "{name} pleads not guilty in a court case regarding cybercrime.",
        "A drug trafficking ring was busted in {place}, implicating {name}."
    ],
    "civil": [
        "Property ownership between {name1} and {name2} contested in {place} court.",
        "{name1} sues {name2} over sudden rental agreement termination.",
        "Dispute arises over land boundaries between {name1} and {name2} in {place}.",
        "Litigation initiated for contract breach by {name1} against {name2}.",
        "Loan settlement conflict between {name1} and {name2} escalates to civil court.",
        "Water damage liability case filed by {name1} in {place}."
    ],
    "constitutional": [
        "Petition filed by {name} regarding violation of free speech rights.",
        "Citizens in {place} protest changes in election laws.",
        "{name} challenges the reservation policy in the Supreme Court.",
        "Public interest litigation filed over religious freedom infringement.",
        "The rights of tribal communities in {place} debated in the apex court.",
        "A law regarding digital surveillance is being scrutinized under constitutional grounds."
    ],
    "family": [
        "{name1} and {name2} file for divorce citing irreconcilable differences.",
        "Custody battle between {name1} and {name2} in {place} family court.",
        "Adoption rights dispute raised by {name} in the High Court.",
        "Alimony disagreement leads to legal action from {name1}.",
        "The inheritance claim of {name} contested by family members in {place}.",
        "Dispute over guardianship rights brings {name1} and {name2} to court."
    ],
    "commercial": [
        "{name1} files lawsuit against {name2} for breach of partnership terms.",
        "Startup {company} sues ex-employee {name} for leaking trade secrets.",
        "Dispute between {company} and {company} regarding merger terms.",
        "{company} is accused of violating antitrust laws in {place}.",
        "Shareholder {name} alleges financial mismanagement by board members.",
        "Supplier contract termination by {company} leads to litigation."
    ]
}


# Define name, place, and company pools
names = ["Raj", "Priya", "Amit", "Sara", "John", "Fatima", "Chen", "Omar", "Anjali", "David",
         "Yash", "Jatin", "Aman", "Piyush", "Shouryan", "Kartikey", "Prashant", "Ayesha", "Anushree",
         "Anushka", "Utkarsh", "Arpita", "Titiksha", "Kshitij", "Harvijay", "Mahesh", "Ramesh", "Sukesh",
         "Gukesh", "Rakesh", "Sunita", "Monika", "Himanshi", "Shubham", "Tanmay", "Aditya", "Kunal",
         "Rohit", "Anshika", "Simran", "Aaradhya", "Sudhir", "Akshay", "Salman", "Mohit", "Shivam",
         "Jyotir", "Mukti"]

places = ["Delhi", "Mumbai", "Chennai", "New York", "London", "Sydney", "Dubai", "Paris", "Madrid",
          "Manchester", "Brisbane", "Dehradun", "Bangalore", "Goa", "Jammu", "Kashmir", "Kerela",
          "Vishakhapatnam", "Pune", "Abu Dhabi", "Doha", "Riyadh", "Cape Town", "Johannesburg",
          "Mexico City", "São Paulo", "Buenos Aires", "Lima", "Cairo", "Istanbul", "Moscow",
          "Singapore", "Kuala Lumpur", "Jakarta", "Bangkok", "Manila", "Berlin", "Rome",
          "Barcelona", "Amsterdam", "Brussels", "Melbourne", "Tokyo", "Osaka", "Seoul", "Beijing",
          "Shanghai"]

companies = ["Microsoft", "Apple", "Google", "Amazon", "Meta Platforms", "Samsung Electronics", "Intel Corporation",
             "IBM", "Oracle", "Salesforce", "Adobe Systems", "Tencent", "Alibaba Group", "TCS", "Infosys",
             "Wipro", "Accenture", "Capgemini", "JPMorgan Chase", "Bank of America", "Citibank", "Wells Fargo",
             "Goldman Sachs", "Morgan Stanley", "ICICI Bank", "HDFC Bank", "HSBC", "Barclays", "American Express",
             "PayPal", "Walmart", "Costco", "Target", "Flipkart", "eBay", "IKEA", "Reliance Retail", "Tesla",
             "Ford", "General Motors", "Toyota", "BMW", "Mercedes-Benz", "Honda", "Hyundai", "Tata Motors",
             "BYD Auto", "Siemens", "General Electric", "3M", "Honeywell", "Caterpillar", "Bosch",
             "Larsen & Toubro", "ExxonMobil", "Shell", "BP", "Chevron", "TotalEnergies", "Adani Green Energy",
             "NextEra Energy", "Johnson & Johnson", "Pfizer", "Roche", "Novartis", "Merck & Co.", "AstraZeneca",
             "Sun Pharma", "Cipla", "GSK", "Netflix", "Disney", "Warner Bros. Discovery", "Sony Pictures",
             "Universal Studios", "Paramount Global", "Spotify", "Marriott International", "Hilton Hotels",
             "Airbnb", "Delta Airlines", "Emirates", "Qatar Airways", "Singapore Airlines", "AT&T", "Verizon",
             "Vodafone", "T-Mobile", "Reliance Jio", "Bharti Airtel", "Coca-Cola", "PepsiCo", "Nestlé",
             "McDonald's", "Starbucks", "Yum! Brands", "Mondelez International"]

# Define allowed logical combinations
allowed_combinations = [
    ["civil", "commercial"],
    ["constitutional", "administrative"],
    ["civil", "family"],
    ["criminal", "constitutional"],
    ["commercial", "administrative"]
]

# Parameters for dataset balancing
single_category_count = 150
combination_count = 60

data = []

# Generate balanced single-category samples
for cat in categories:
    for _ in range(single_category_count):
        template = random.choice(templates[cat])
        case_text = template.format(
            name=random.choice(names),
            name1=random.choice(names),
            name2=random.choice(names),
            place=random.choice(places),
            company=random.choice(companies)
        )
        labels = [1 if c == cat else 0 for c in categories]
        data.append([case_text] + labels)

# Generate balanced multi-category samples
for pair in allowed_combinations:
    for _ in range(combination_count):
        case_parts = []
        for cat in pair:
            template = random.choice(templates[cat])
            part = template.format(
                name=random.choice(names),
                name1=random.choice(names),
                name2=random.choice(names),
                place=random.choice(places),
                company=random.choice(companies)
            )
            case_parts.append(part)
        full_case = " ".join(case_parts)
        labels = [1 if c in pair else 0 for c in categories]
        data.append([full_case] + labels)

# Shuffle and save
df = pd.DataFrame(data, columns=["case_text"] + categories)
df = df.sample(frac=1).reset_index(drop=True)
df.to_csv("balanced_legal_cases_dataset.csv", index=False)

print("✅ Balanced dataset saved as balanced_legal_cases_dataset.csv")
