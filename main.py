import gradio as gr
import pdf2image
import pytesseract
import pandas as pd
import re
import string
import cv2
import numpy as np
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import os

# Initialize SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')


# Helper: Semantic similarity using SentenceTransformer
def semantic_similarity(a, b):
    try:
        embeddings = model.encode([a, b])
        return util.cos_sim(embeddings[0], embeddings[1]).item()
    except Exception as e:
        print(f"Error in semantic_similarity: {e}")
        return 0.0

# Helper: Clean OCR text
def clean_text(text):
    text = ''.join(filter(lambda x: x in string.printable, text))
    text = ' '.join(text.split())
    return text

# Helper: Preprocess image for better OCR
def preprocess_image_for_ocr(image):
    # Convert PIL image to OpenCV format
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply thresholding to improve text clarity
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Denoise
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return Image.fromarray(img)

# Step 1: Convert PDF pages to images
def pdf_to_images(pdf_path):
    try:
        images = pdf2image.convert_from_path(pdf_path)
        return images
    except Exception as e:
        raise ValueError(f"Error converting PDF to images: {e}")

# Step 2: Extract text from an image using OCR
def extract_text_from_image(image):
    try:
        # Preprocess image
        processed_image = preprocess_image_for_ocr(image)
        text = pytesseract.image_to_string(processed_image)
        text = clean_text(text)
        return text
    except Exception as e:
        print(f"Error in OCR: {e}")
        return ""

# Step 3: Extract Questions from Question Paper
def extract_questions_from_text(text):
    questions = {}
    lines = text.split("\n")
    for line in lines:
        line = line.strip()
        match = re.match(r'Q(\d+)[\.\):]', line)
        if match:
            q_no = "Q" + match.group(1)
            q_text = line[match.end():].strip()
            if q_text:  # Only add non-empty questions
                questions[q_no] = q_text
    return questions

# Step 4: Evaluate Student Answers
def evaluate_answers(student_answers_text, model_questions, method="keyword", keywords=None):
    student_lines = [line.strip() for line in student_answers_text.split("\n") if line.strip()]
    results = {}
    keyword_set = set(kw.lower() for kw in keywords) if keywords else set()

    for q_no, model_answer in model_questions.items():
        best_match = ""
        highest_score = 0.0

        if method == "semantic":
            # Semantic similarity
            for line in student_lines:
                similarity = semantic_similarity(model_answer, line)
                # Boost score if keywords are present
                if keyword_set:
                    student_words = set(line.lower().split())
                    keyword_matches = len(keyword_set.intersection(student_words))
                    similarity += 0.1 * keyword_matches  # Boost by 10% per keyword match
                if similarity > highest_score:
                    highest_score = similarity
                    best_match = line
            marks = min(int(highest_score * 10), 10)  # Marks out of 10, capped at 10
        else:
            # Keyword overlap (default)
            model_words = set(model_answer.lower().split())
            for line in student_lines:
                student_words = set(line.lower().split())
                common_words = model_words.intersection(student_words)
                overlap = len(common_words)
                # Boost score if keywords are present
                if keyword_set:
                    keyword_matches = len(keyword_set.intersection(student_words))
                    overlap += 2 * keyword_matches  # Weight keywords higher
                if overlap > highest_score:
                    highest_score = overlap
                    best_match = line
            total_words = len(model_words)
            marks = int((highest_score / total_words) * 10) if total_words > 0 else 0
            marks = min(marks, 10)  # Cap at 10

        results[q_no] = {
            "question": model_answer,
            "student_answer": best_match,
            "marks": marks
        }

    return results

# Step 5: Save results to CSV
def save_results_to_csv(results, output_file="evaluation_result.csv"):
    try:
        data = []
        for q_no, info in results.items():
            data.append({
                "Question No.": q_no,
                "Question": info['question'],
                "Student Answer": info['student_answer'],
                "Marks": info['marks']
            })
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
        return output_file
    except Exception as e:
        print(f"Error saving CSV: {e}")
        return None

# Step 6: Full Processing Function
def evaluate_papers(question_pdf, answer_pdf, method="keyword", keywords=""):
    try:
        # Validate inputs
        if not question_pdf or not answer_pdf:
            return "Error: Please upload both Question Paper and Answer Sheet PDFs.", None

        # Parse keywords
        keyword_list = [kw.strip() for kw in keywords.split(",") if kw.strip()] if keywords else []

        # Extract text from Question Paper
        question_images = pdf_to_images(question_pdf.name)
        question_text = ""
        for img in question_images:
            question_text += extract_text_from_image(img) + "\n"

        # Extract text from Answer Sheet
        answer_images = pdf_to_images(answer_pdf.name)
        answer_text = ""
        for img in answer_images:
            answer_text += extract_text_from_image(img) + "\n"

        # Build model answers
        model_questions = extract_questions_from_text(question_text)
        if not model_questions:
            return "Error: No questions detected in the Question Paper.", None

        # Evaluate answers
        results = evaluate_answers(answer_text, model_questions, method=method, keywords=keyword_list)

        # Save CSV
        output_csv = save_results_to_csv(results)

        # Prepare display text
        final_result_text = ""
        total_marks = 0
        for q, data in results.items():
            final_result_text += f"{q}:\nQuestion: {data['question']}\nStudent Answer: {data['student_answer']}\nMarks: {data['marks']}/10\n\n"
            total_marks += data['marks']

        final_result_text += f"\nTotal Marks: {total_marks}/{len(results)*10}\n"
        if keyword_list:
            final_result_text += f"\nKeywords Used: {', '.join(keyword_list)}\n"

        return final_result_text, output_csv
    except Exception as e:
        return f"Error processing files: {e}", None

# Step 7: Build Gradio Web Interface
interface = gr.Interface(
    fn=evaluate_papers,
    inputs=[
        gr.File(label="Upload Question Paper PDF"),
        gr.File(label="Upload Answer Sheet PDF"),
        gr.Dropdown(choices=["keyword", "semantic"], label="Evaluation Method", value="keyword"),
        gr.Textbox(label="Enter Keywords (comma-separated, e.g., science, chemistry)", placeholder="Optional keywords")
    ],
    outputs=[
        gr.Textbox(label="Evaluation Result", lines=20),
        gr.File(label="Download CSV Result")
    ],
    title="📝 AI Answer Sheet Evaluator",
    description="Upload the Question Paper PDF and a Student's Answer Sheet PDF. Choose an evaluation method and optionally provide keywords to prioritize in scoring.",
    theme="soft"
)

# Step 8: Launch
if __name__ == "__main__":
    interface.launch(share=True, debug=True)  # Run locally with debug mode
