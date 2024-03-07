from fastapi import FastAPI, Query, UploadFile, Form, HTTPException
from fastapi.responses import PlainTextResponse,JSONResponse
from typing import List,NamedTuple,Tuple
from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import PlainTextResponse
import pdfplumber
import io
import spacy
from spacy.training.example import Example
import fitz
import random
import re
from fuzzywuzzy import fuzz
from fastai.text.all import *
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from summarizer import Summarizer
from spacy.training.example import Example
from nltk import sent_tokenize
import nltk

# Install NLTK resources for sentence tokenization
nltk.download('punkt')
app = FastAPI()

origins = [
    "http://localhost:3000",
    "localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
import random

def train_heading_model(training_data, nlp, epochs=50, dropout=0.5, learning_rate=0.001):
    # Create or update the 'ner' pipeline
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")

    # Add the label for the "EXCLUDE_HEADING" entities
    ner.add_label("EXCLUDE_HEADING")

    examples = []
    for text, annotations in training_data:
        example = Example.from_dict(nlp.make_doc(text), annotations)
        examples.append(example)

    # Get the names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]

    # Disable other pipes and train NER
    with nlp.disable_pipes(*other_pipes):
        for epoch in range(epochs):
            random.shuffle(examples)
            for example in examples:
                # Update the model using Example objects
                nlp.update([example], drop=dropout, losses={})

    return nlp

# Load a larger base model
heading_model = spacy.load("en_core_web_lg")
        # heading_model.max_length = 1500000
training_data = [
        ("Introduction: This is an introduction to the document.", {"entities": [(0, 12, "EXCLUDE_HEADING")]}),
        ("INTRODUCTION: This is an introduction to the document.", {"entities": [(0, 12, "EXCLUDE_HEADING")]}),
        ("Abstract: The abstract of the document goes here.", {"entities": [(0, 7, "EXCLUDE_HEADING")]}),
        ("ABSTRACT: The abstract of the document goes here.", {"entities": [(0, 7, "EXCLUDE_HEADING")]}),
        ("Requirement Analysis (Data Gathering): Details about gathering requirements.", {"entities": [(0, 38, "EXCLUDE_HEADING")]}),
        # Add more examples with different variations of headings
        ("ABSTRACT: This is another abstract.", {"entities": [(0, 7, "EXCLUDE_HEADING")]}),
        ("REQUIREMENT ANALYSIS (DATA GATHERING): Gathering data for analysis.", {"entities": [(0, 39, "EXCLUDE_HEADING")]}),
        ("Chapter 1: Overview of the Topic", {"entities": [(0, 7, "EXCLUDE_HEADING")]}),
        # Include more variations...
    ]
training_data += [
        ("Conclusion: Summary of the document.", {"entities": [(0, 9, "EXCLUDE_HEADING")]}),
        ("CONCLUSION: Key takeaways from the document.", {"entities": [(0, 9, "EXCLUDE_HEADING")]}),
        ("Summary: Brief overview of the content.", {"entities": [(0, 6, "EXCLUDE_HEADING")]}),
        ("SUMMARY: Recap of the main points.", {"entities": [(0, 6, "EXCLUDE_HEADING")]}),
        ("Methodology: Details about the research approach.", {"entities": [(0, 11, "EXCLUDE_HEADING")]}),
        ("METHODOLOGY: Research methods employed in the study.", {"entities": [(0, 11, "EXCLUDE_HEADING")]}),
        ("Results: Findings obtained from experiments.", {"entities": [(0, 6, "EXCLUDE_HEADING")]}),
        ("RESULTS: Outcome of the conducted analysis.", {"entities": [(0, 6, "EXCLUDE_HEADING")]}),
        ("Discussion: In-depth analysis and interpretation.", {"entities": [(0, 9, "EXCLUDE_HEADING")]}),
        ("DISCUSSION: Conversations around the obtained results.", {"entities": [(0, 9, "EXCLUDE_HEADING")]}),
        ("Appendix A: Additional information and resources.", {"entities": [(0, 9, "EXCLUDE_HEADING")]}),
        ("APPENDIX A: Supplementary materials and references.", {"entities": [(0, 9, "EXCLUDE_HEADING")]}),
        ("Acknowledgments: Recognition of contributors.", {"entities": [(0, 13, "EXCLUDE_HEADING")]}),
        ("ACKNOWLEDGMENTS: Gratitude towards those who assisted.", {"entities": [(0, 13, "EXCLUDE_HEADING")]}),
        ("References: List of cited works.", {"entities": [(0, 9, "EXCLUDE_HEADING")]}),
        ("REFERENCES: Citations and sources used in the document.", {"entities": [(0, 9, "EXCLUDE_HEADING")]}),
        ("Glossary: Definitions of terms used.", {"entities": [(0, 7, "EXCLUDE_HEADING")]}),
        ("GLOSSARY: Explanations of key terms in the document.", {"entities": [(0, 7, "EXCLUDE_HEADING")]}),
        ("Appendix B: Additional supporting materials.", {"entities": [(0, 9, "EXCLUDE_HEADING")]}),
        ("APPENDIX B: Supplemental content and resources.", {"entities": [(0, 9, "EXCLUDE_HEADING")]}),
        ("Limitations: Constraints and boundaries of the study.", {"entities": [(0, 10, "EXCLUDE_HEADING")]}),
        ("LIMITATIONS: Challenges and restrictions in the research.", {"entities": [(0, 10, "EXCLUDE_HEADING")]}),
        ("Appendix C: Further details and information.", {"entities": [(0, 9, "EXCLUDE_HEADING")]}),
        ("APPENDIX C: Additional content and insights.", {"entities": [(0, 9, "EXCLUDE_HEADING")]}),
        ("Abstract: A brief summary of the document.", {"entities": [(0, 7, "EXCLUDE_HEADING")]}),
        ("ABSTRACT: Concise overview highlighting key points.", {"entities": [(0, 7, "EXCLUDE_HEADING")]}),
        ("Appendix D: Supplementary data and materials.", {"entities": [(0, 9, "EXCLUDE_HEADING")]}),
        ("APPENDIX D: Extra information and supporting content.", {"entities": [(0, 9, "EXCLUDE_HEADING")]}),
        ("Chapter 2: In-depth exploration of the subject.", {"entities": [(0, 7, "EXCLUDE_HEADING")]}),
        ("CHAPTER 2: Detailed analysis and insights.", {"entities": [(0, 7, "EXCLUDE_HEADING")]}),
        ("Appendix E: Additional data and resources.", {"entities": [(0, 9, "EXCLUDE_HEADING")]}),
        ("APPENDIX E: Supplementary information and materials.", {"entities": [(0, 9, "EXCLUDE_HEADING")]}),
        ("Appendix F: Further details and content.", {"entities": [(0, 9, "EXCLUDE_HEADING")]}),
        ("APPENDIX F: Extra information and supporting materials.", {"entities": [(0, 9, "EXCLUDE_HEADING")]}),
        ("Chapter 3: Detailed examination of the topic.", {"entities": [(0, 7, "EXCLUDE_HEADING")]}),
        ("CHAPTER 3: In-depth exploration and analysis.", {"entities": [(0, 7, "EXCLUDE_HEADING")]}),
        ("Appendix G: Additional content and data.", {"entities": [(0, 9, "EXCLUDE_HEADING")]}),
        ("APPENDIX G: Supplementary resources and information.", {"entities": [(0, 9, "EXCLUDE_HEADING")]}),
        ("Appendix H: Extra details and supporting materials.", {"entities": [(0, 9, "EXCLUDE_HEADING")]}),
        ("APPENDIX H: Further information and resources.", {"entities": [(0, 9, "EXCLUDE_HEADING")]}),
        ("Chapter 4: In-depth study and analysis.", {"entities": [(0, 7, "EXCLUDE_HEADING")]}),
        ("CHAPTER 4: Comprehensive examination of the subject.", {"entities": [(0, 7, "EXCLUDE_HEADING")]}),
        ("Appendix I: Additional supporting data.", {"entities": [(0, 9, "EXCLUDE_HEADING")]}),
        ("APPENDIX I: Supplementary content and resources.", {"entities": [(0, 9, "EXCLUDE_HEADING")]}),
        ("Appendix J: Further details and information.", {"entities": [(0, 9, "EXCLUDE_HEADING")]}),
        ("APPENDIX J: Extra content and supporting materials.", {"entities": [(0, 9, "EXCLUDE_HEADING")]}),
        ("Chapter 5: Detailed exploration and analysis.", {"entities": [(0, 7, "EXCLUDE_HEADING")]}),
        ("CHAPTER 5: In-depth study and examination of the topic.", {"entities": [(0, 7, "EXCLUDE_HEADING")]}),
    # Add more variations...
    ]


   
heading_model = train_heading_model(training_data, heading_model, epochs=50, dropout=0.3, learning_rate=0.001)
print("model trained successfully")
def train_exclusion_model(training_data, nlp):
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")

    # Add the label for location entities
    # for label in ["EXCLUDE_DATE", "EXCLUDE_CARDINAL", "EXCLUDE_ORG", "EXCLUDE_HEADING", "EXCLUDE_LOCATION"]:
    #     ner.add_label(label)
    ner.add_label("EXCLUDE_HEADING")

    examples = []
    for text, annotations in training_data:
        example = Example.from_dict(nlp.make_doc(text), annotations)
        examples.append(example)

    # Get the names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]

    # Disable other pipes and train NER
    with nlp.disable_pipes(*other_pipes):
        for epoch in range(30):  # Increase the number of epochs as needed
            random.shuffle(examples)
            for example in examples:
                nlp.update([example], drop=0.5, losses={})

    return nlp

def preserve_indentation(text, filtered_text):
    original_lines = text.split('\n')
    filtered_lines = filtered_text.split('\n')
    result_lines = []

    for orig_line, filtered_line in zip(original_lines, filtered_lines):
        indentation = len(orig_line) - len(orig_line.lstrip())
        result_lines.append(' ' * indentation + filtered_line)

    return '\n'.join(result_lines)



def extract_metadata(pdf_content):
    metadata = {}
    try:
        pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
        if pdf_document:
            metadata["title"] = pdf_document.metadata.get("title")
            metadata["author"] = pdf_document.metadata.get("author")
            metadata["subject"] = pdf_document.metadata.get("subject")
            metadata["keywords"] = pdf_document.metadata.get("keywords")
        else:
            print("Not a valid PDF document.")
    except Exception as e:
        print(f"Error extracting metadata: {e}")

    return metadata


def exclude_unwanted_content(pdf_content, exclusion_model, heading_model, extraction_type):
    with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
        pdf_text = ""
        for page in pdf.pages:
            pdf_text += page.extract_text()

    doc = exclusion_model(pdf_text)
    filtered_paragraphs = []
    current_paragraph = ""

    for sent in doc.sents:
        if any(ent.label_.startswith("EXCLUDE_") for ent in sent.ents):
            if current_paragraph:
                filtered_paragraphs.append(current_paragraph.strip())
                current_paragraph = ""
        else:
            current_paragraph += sent.text + ' '

    if current_paragraph:
        filtered_paragraphs.append(current_paragraph.strip())

    filtered_text = '\n\n'.join(filtered_paragraphs)

    if extraction_type == "metadata":
        metadata = extract_metadata(pdf_content)
        formatted_metadata = "\n".join(f"{key}: {value}" for key, value in metadata.items())
        return PlainTextResponse(content=formatted_metadata, status_code=200)
    elif extraction_type == "locations":
        locations = extract_locations(pdf_content, exclusion_model)  # Add a new function for location extraction
        return PlainTextResponse(content=f"Extracted Locations: {locations}", status_code=200)
    summarizer = Summarizer()
    summary = summarizer(filtered_text, ratio=0.3)
    return PlainTextResponse(content=summary, status_code=200)


@app.post("/home")
async def helper(
    extract: UploadFile = UploadFile(...),
    extraction_type: str = Query("complete_data", description="Extraction Type (complete_data, metadata, headings, etc.)")
):
    nlp = spacy.load("en_core_web_sm")
    training_data = [
    ("INTRODUCTION This is an introduction to the document.", {"entities": [(0, 12, "EXCLUDE_HEADING")]}),
    ("DECLARATION I declare that...", {"entities": [(0, 11, "EXCLUDE_HEADING")]}),
    ("ABSTRACT The abstract of the document goes here.", {"entities": [(0, 7, "EXCLUDE_HEADING")]}),
    ("CHAPTER 1 Overview of the Topic", {"entities": [(0, 7, "EXCLUDE_HEADING")]}),
    ("METHODOLOGY In this section, we describe the methodology used.", {"entities": [(0, 10, "EXCLUDE_HEADING")]}),
    ("RESULTS The results of the study are presented in this chapter.", {"entities": [(0, 6, "EXCLUDE_HEADING")]}),
    ("CONCLUSION In conclusion, we summarize the findings.", {"entities": [(0, 9, "EXCLUDE_HEADING")]}),
    ("ACKNOWLEDGEMENTS I would like to acknowledge...", {"entities": [(0, 13, "v")]}),
    ("REFERENCES Here are the references used in this document.", {"entities": [(0, 9, "EXCLUDE_HEADING")]}),
    ("APPENDIX A Additional Information", {"entities": [(0, 7, "EXCLUDE_HEADING")]}),
    # Add more examples as needed
    ("Electronic Reservation Slip (ERS)-Normal User", {"entities": []}),
    ("Booked From To MGR CHENNAI CTL (MAS) MGR CHENNAI CTL (MAS) KATPADI JN (KPD)", {"entities": [(29, 43, "EXCLUDE_LOCATION"), (52, 66, "EXCLUDE_LOCATION"), (72, 84, "EXCLUDE_LOCATION")]}),
    ("Start Date* 06-Mar-2023", {"entities": []}),
    ("Departure* 07:40 06 -Mar-2023", {"entities": []}),
    ("Arrival* 09:38 06-Mar-2023", {"entities": []}),
    ("PNR Train No./Name Class 4344799486 12639 / BRINDAVAN EXP CHAIR CAR (CC)", {"entities": []}),
    ("Quota Distance Booking Date GENERAL (GN) 130 KM 01-Mar-2023 21:47:48 HRS", {"entities": []}),
    ("Passenger Details # Name Age Gender Booking Status Current Status", {"entities": []}),
    ("Our office is located at 123 Main Street.", {"entities": [(23, 37, "EXCLUDE_LOCATION")]}),
    ("The train will pass through Bangalore City.", {"entities": [(34, 51, "EXCLUDE_LOCATION")]}),
    ("The event will take place in New Delhi.", {"entities": [(33, 42, "EXCLUDE_LOCATION")]}),
    ("MGR CHENNAI CTL (MAS)", {"entities": [(0, 20, "EXCLUDE_LOCATION")]}),
    ("KATPADI JN (KPD)", {"entities": [(0, 15, "EXCLUDE_LOCATION")]}),
]

    # Add more examples as needed

    exclusion_model = train_exclusion_model(training_data, nlp)
    
    # Initialize the heading_model here
    heading_model = spacy.load("en_core_web_sm")

    pdf_content = await extract.read()
    result_text = exclude_unwanted_content(pdf_content, exclusion_model, heading_model, extraction_type)
    return result_text

@app.get("/hello")
def hello():
    return "parth"

def train_location_model(training_data, nlp):
    # Create or update the 'ner' pipeline
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")

    # Add the label for location entities
    ner.add_label("EXCLUDE_LOCATION")

    examples = []
    for text, annotations in training_data:
        example = Example.from_dict(nlp.make_doc(text), annotations)
        examples.append(example)

    # Get the names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]

    # Disable other pipes and train NER
    with nlp.disable_pipes(*other_pipes):
        for epoch in range(30):  # Increase the number of epochs as needed
            random.shuffle(examples)
            for example in examples:
                nlp.update([example], drop=0.5, losses={})

    return nlp



def extract_text_from_pdf(pdf_content):
    with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
        pdf_text = ""
        for page in pdf.pages:
            # print(page.page_number)
            pdf_text += page.extract_text()
    return pdf_text

def extract_text_from_pdf_with_page_individual(pdf_content: bytes) -> List[Tuple[str, int]]:
    extracted_text_with_page = []
    with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            words = text.split()
            words_with_page = [(word, page_number) for word in words]
            extracted_text_with_page.extend(words_with_page)
    return extracted_text_with_page
@app.post("/extract_locations")
async def extract_locations(
    extract: UploadFile = UploadFile(...),
):
    # Load the spaCy model for location extraction
    location_model = spacy.load("en_core_web_sm")

    training_data = [
    ("MGR CHENNAI CTL (MAS)", {"entities": [(0, 20, "EXCLUDE_LOCATION")]}),
    ("KATPADI JN (KPD)", {"entities": [(0, 15, "EXCLUDE_LOCATION")]}),
    ("HOWRAH JN", {"entities": [(0, 9, "EXCLUDE_LOCATION")]}),
    ("DELHI JN", {"entities": [(0, 8, "EXCLUDE_LOCATION")]}),
    ("BENGALURU CANTT (BNC)", {"entities": [(0, 22, "EXCLUDE_LOCATION")]}),
    ("JAIPUR JN", {"entities": [(0, 9, "EXCLUDE_LOCATION")]}),
    ("AHMEDABAD JN", {"entities": [(0, 12, "EXCLUDE_LOCATION")]}),
    ("MUMBAI CST", {"entities": [(0, 11, "EXCLUDE_LOCATION")]}),
    ("HYDERABAD", {"entities": [(0, 8, "EXCLUDE_LOCATION")]}),
    ("KOLKATA", {"entities": [(0, 7, "EXCLUDE_LOCATION")]}),
    ("CHENNAI", {"entities": [(0, 7, "EXCLUDE_LOCATION")]}),
    ("BENGALURU", {"entities": [(0, 8, "EXCLUDE_LOCATION")]}),
    ("DELHI", {"entities": [(0, 5, "EXCLUDE_LOCATION")]}),
    ("JAIPUR", {"entities": [(0, 6, "EXCLUDE_LOCATION")]}),
    ("AHMEDABAD", {"entities": [(0, 8, "EXCLUDE_LOCATION")]}),
    ("MUMBAI", {"entities": [(0, 6, "EXCLUDE_LOCATION")]}),
    ("PUNE", {"entities": [(0, 4, "EXCLUDE_LOCATION")]}),
    ("GOA", {"entities": [(0, 3, "EXCLUDE_LOCATION")]}),
    ("HYDERABAD", {"entities": [(0, 8, "EXCLUDE_LOCATION")]}),
    ("BANGALORE", {"entities": [(0, 8, "EXCLUDE_LOCATION")]}),
    ("CHANDIGARH", {"entities": [(0, 10, "EXCLUDE_LOCATION")]}),
    ("LUCKNOW", {"entities": [(0, 7, "EXCLUDE_LOCATION")]}),
    ("Student-To-Student\nDelivery System", {"entities": [(0, 30, "HEADING")]}),
    ("Fall 2022-23", {"entities": [(0, 12, "OTHER")]}),
    # ... other training examples with different entity types
    ("Technologies Used", {"entities": [(0, 18, "HEADING")]}),
    # ("Frontend Module", {"entities": [(0, 15, "TECH_TERM")]}),
    # Add more cities as needed
    ]

   
    location_model = train_location_model(training_data, location_model)

    # Extract text from the PDF
    pdf_content = await extract.read()
    pdf_text = extract_text_from_pdf(pdf_content)

    # Process the text using the location model
    doc = location_model(pdf_text)
    # print("Entities:", [(ent.text, ent.label_) for ent in doc.ents])

    locations = [ent.text for ent in doc.ents if ent.label_ == "EXCLUDE_LOCATION"]

    return PlainTextResponse(content=f"Extracted Locations: {locations}", status_code=200)


@app.post("/extract_headings")
async def extract_heading(
    extract: UploadFile = UploadFile(...),
):
    
    # Extract text from the PDF
    pdf_content = await extract.read()
    pdf_text = extract_text_from_pdf(pdf_content)

    # Process the text using the location model
    doc = heading_model(pdf_text)
    headings = [ent.text for ent in doc.ents if ent.label_ == "EXCLUDE_HEADING"]
    headings_with_page = []

    with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()

            for heading in headings:
                if heading in text:
                    # If the heading is present in the current page, append it with the correct page number
                    headings_with_page.append(f"{heading}:{page_number}")

    print(headings_with_page)  
    return headings


def get_content_under_heading(file_content: bytes, heading: str, all_headings: List[str]) -> str:
    try:
        doc = fitz.open("pdf", file_content)

        extracted_content = ""
        in_heading = False

        for page_num in range(doc.page_count):
            page = doc[page_num]
            text = page.get_text("text")
            lines = text.split('\n')

            for line in lines:
                # Check if the heading is present in the line
                if heading.lower() in line.lower():
                    in_heading = True
                    continue

                # Check if a new heading is found
                elif in_heading and any(h in line for h in all_headings):
                    in_heading = False
                    break

                # Append the line to extracted content if we are inside the desired heading
                elif in_heading:
                    extracted_content += line.strip() + " "

        # Use spaCy to extract key sentences
        print(extracted_content)
        summarizer = Summarizer()
        summary = summarizer(extracted_content, ratio=0.2)
        
        return summary

    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")



@app.post("/extract_content_headings")
async def extract_content_headings(extract: UploadFile = UploadFile(...), heading: str = Form(...)):
    try:
        
        file_content = await extract.read()
        pdf_text = extract_text_from_pdf(file_content)
        
        if heading is None:
            raise HTTPException(status_code=400, detail="Heading not provided in request.")
        
        doc = heading_model(pdf_text)
        all_headings = [ent.text for ent in doc.ents if ent.label_ == "EXCLUDE_HEADING"]
        print(all_headings)
        if heading not in all_headings:
            raise HTTPException(status_code=404, detail=f"Heading '{heading}' not found in the document.")
        
        content_under_heading = get_content_under_heading(file_content, heading, all_headings)
        
        return JSONResponse(content={"content": content_under_heading}, status_code=200)
    
    except Exception as e:
        return JSONResponse(content={"error": f"Internal Server Error: {str(e)}"}, status_code=500)
    

class TrainingExample(NamedTuple):
    text: str
    label: str
    subdomain: str

# Function to train a document classification model
def train_document_classifier(training_data: List[TrainingExample], label_col: str) -> SVC:
    # Convert the training data to a DataFrame
    df = pd.DataFrame(training_data, columns=["text", "label", "subdomain"])

    # Create a pipeline with TF-IDF vectorizer and Support Vector Machine (SVM) classifier
    model = make_pipeline(TfidfVectorizer(), SVC())

    # Train the model on the entire dataset
    model.fit(df['text'], df[label_col])

    return model

# Function to classify a document using the trained model
def classify_document(model: SVC, text: str) -> str:
    prediction = model.predict([text])
    return prediction[0]

# Function to handle document classification endpoint
@app.post("/extract_context")
async def extract_context(extract: UploadFile = UploadFile(...)):
    try:
        # Get the content of the uploaded file
        file_content = await extract.read()

        # Convert bytes to string (assuming the content is text)
        text_content = extract_text_from_pdf(file_content)

        # Train the model for document type
        document_type_training_data = [
            TrainingExample("User Manual", "Other", "N/A"),
            TrainingExample("Chemistry Research Paper", "Research Paper", "Chemistry"),
            TrainingExample("Employment Contract", "Legal Agreement", "Employment Law"),
            TrainingExample("Medical Journal Article", "Research Paper", "Medicine"),
            TrainingExample("Real Estate Agreement", "Legal Agreement", "Real Estate Law"),
            TrainingExample("Physics Thesis", "Research Paper", "Physics"),
            TrainingExample("Software License Agreement", "Legal Agreement", "Software Law"),
            TrainingExample("Environmental Research Paper", "Research Paper", "Environmental Science"),
            TrainingExample("Partnership Agreement", "Legal Agreement", "Business Law"),
            TrainingExample("Psychology Study", "Research Paper", "Psychology"),
            TrainingExample("Divorce Settlement", "Legal Agreement", "Family Law"),
            TrainingExample("Mathematics Paper", "Research Paper", "Mathematics"),
            TrainingExample("Intellectual Property Agreement", "Legal Agreement", "Intellectual Property Law"),
            TrainingExample("Political Science Research", "Research Paper", "Political Science"),
            TrainingExample("Non-Disclosure Agreement", "Legal Agreement", "Confidentiality Law"),
            TrainingExample("Art History Dissertation", "Research Paper", "Art History"),
            TrainingExample("Construction Contract", "Legal Agreement", "Construction Law"),
            TrainingExample("Astronomy Research", "Research Paper", "Astronomy"),
            TrainingExample("Commercial Lease Agreement", "Legal Agreement", "Property Law"),
            TrainingExample("Educational Psychology Study", "Research Paper", "Educational Psychology"),

            # Add more examples with different document types and classes
        ]
        document_type_model = train_document_classifier(document_type_training_data, label_col="label")

        # Classify the document type using the trained model
        predicted_document_type = classify_document(document_type_model, text_content)

        # Train the model for subdomain
        subdomain_training_data = [
            TrainingExample("Technical Report in Mechanical Engineering on Finite Element Analysis", "Technical Report", "Mechanical_Engineering_Finite_Element_Analysis"),
            TrainingExample("Legal Contract in Intellectual Property Law for Trademark Agreements", "Legal Contract", "Law_Intellectual_Property_Trademark"),
            TrainingExample("Research Paper in Environmental Science on Climate Change Impact", "Research Paper", "Environmental_Science_Climate_Change"),
            TrainingExample("Financial Statement in Accounting for Quarterly Performance Analysis", "Financial Statement", "Accounting_Quarterly_Performance_Analysis"),
            TrainingExample("Medical Report in Oncology on Cancer Treatment Methods", "Medical Report", "Oncology_Cancer_Treatment"),
            TrainingExample("Software Specification Document for Mobile Application Development", "Software Specification Document", "Technology_Mobile_Application_Development"),
            TrainingExample("Research Paper in Economics on Macroeconomic Trends", "Research Paper", "Economics_Macroeconomic_Trends"),
            TrainingExample("Real Estate Lease Agreement for Residential Property Rental", "Real Estate Lease Agreement", "Real_Estate_Residential_Property_Rental"),
            TrainingExample("Technical Manual in Information Technology for Network Configuration", "Technical Manual", "Information_Technology_Network_Configuration"),
            TrainingExample("Legal Brief in Employment Law on Workplace Discrimination Cases", "Legal Brief", "Law_Employment_Workplace_Discrimination"),
            TrainingExample("Research Paper in Linguistics on Language Evolution Theories", "Research Paper", "Linguistics_Language_Evolution_Theories"),
            TrainingExample("Insurance Policy Document for Health Insurance Coverage", "Insurance Policy Document", "Insurance_Health_Insurance_Coverage"),
            TrainingExample("Research Paper in Sociology on Social Media Impact on Society", "Research Paper", "Sociology_Social_Media_Impact"),
            TrainingExample("Business Proposal for Marketing Campaign in Digital Advertising", "Business Proposal", "Marketing_Digital_Advertising"),
            # TrainingExample("Scientific Study in Botany on Plant Genetics and Hybridization", "Scientific Study", "Botany_Plant_Genetics_Hybridization"),
            TrainingExample("Contract Agreement for IT Consulting Services", "Contract Agreement", "Information_Technology_Consulting_Services"),
            TrainingExample("Research Paper in Political Science on International Relations", "Research Paper", "Political_Science_International_Relations"),
            TrainingExample("Architectural Design Proposal for Sustainable Building Construction", "Architectural Design Proposal", "Architecture_Sustainable_Building_Construction"),
            TrainingExample("Legal Document in Family Law for Prenuptial Agreements", "Legal Document", "Law_Family_Prenuptial_Agreements"),
            TrainingExample("Scientific Review in Neuroscience on Neurotransmitter Functions", "Scientific Review", "Neuroscience_Neurotransmitter_Functions"),
    # Add more examples as needed
]
        

        subdomain_model = train_document_classifier(subdomain_training_data, label_col="subdomain")

        # Classify the subdomain using the trained model
        predicted_subdomain = classify_document(subdomain_model, text_content)

        return JSONResponse(content={"document_type": predicted_document_type, "subdomain": predicted_subdomain}, status_code=200)

    except Exception as e:
        print(f"Error: {str(e)}")  # Debugging line
        return JSONResponse(content={"error": f"Internal Server Error: {str(e)}"}, status_code=500)
    
def extract_text_for_pages_pdf(pdf_content,startpage,endpage):
    with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
        pdf_text = ""
        for page in pdf.pages:
            # print(page.page_number)
           if startpage <= page.page_number <= endpage:
                pdf_text += page.extract_text()
    return pdf_text

def generate_bullet_points(summary):
    # Split the summary into sentences
    sentences = sent_tokenize(summary)
    return sentences
    # Format sentences as bullet points
    # bullet_points = ["- " + sentence for sentence in sentences]

    # # Join the bullet points into a formatted summary
    # formatted_summary = "\n".join(bullet_points)

    # return formatted_summary
def group_sentences_by_heading(sentences, all_headings):
    result = {}
    current_heading = None
    current_sentences = []
    for sentence in sentences:
        # Check if the sentence contains any heading
        if any(heading in sentence for heading in all_headings):
            # If yes, update current_heading and include the current_sentences
            if current_heading is not None:
                result[current_heading] = current_sentences
            else:
                if current_sentences!=[]:
                   result["some data"]=current_sentences
            current_heading = next((heading for heading in all_headings if heading in sentence), None)
            # Initialize an empty list for the heading if not present
            result.setdefault(current_heading, [])
            current_sentences = [sentence]
        else:
            # If no heading, add the sentence to the list under the current_heading
            current_sentences.append(sentence)

    # Include the last set of sentences if any
    if current_heading is not None:
        result[current_heading] = current_sentences
    else:
        if current_sentences!=[]:
             result["some data"]=current_sentences
    return result
@app.post('/extract_summary_pages')
async def extractpagesummary(
    extract: UploadFile = UploadFile(...),
    startpage:int=Form(...),
    endpage:int=Form(...)
):
    file_content = await extract.read()

        # Convert bytes to string (assuming the content is text)
    text_content = extract_text_for_pages_pdf(file_content,startpage,endpage)
    doc = heading_model(text_content)
    all_headings = [ent.text for ent in doc.ents if ent.label_ == "EXCLUDE_HEADING"]
    print(all_headings)
    print(text_content)
    summarizer = Summarizer()
    summary = summarizer(text_content, ratio=0.6)
    bullet_point_summary = generate_bullet_points(summary)
    result = group_sentences_by_heading(bullet_point_summary , all_headings)
    return JSONResponse(content={"content": result}, status_code=200)