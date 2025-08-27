import re
import json
import sys

# Regex for cleaning answerer names: (C1), (A), (B2) etc.
# Matches a single letter optionally followed by a single digit, enclosed in half-width or full-width parentheses.
# This pattern is used for a second pass of cleaning after initial extraction.
ANSWERER_NAME_PATTERN = re.compile(r'\s*[\(（][A-Za-z]\d?[\)）]')

def clean_text_initial_pass(file_content):
    """
    Performs an initial cleaning pass on the raw text content from the PDF conversion.
    This includes removing page numbers, question numbers, and irrelevant indented lines.
    """
    lines = file_content.split('\n')
    cleaned_lines = []
    for line in lines:
        # 1. Page number cleaning: Ignore lines that contain only a single number.
        #    Example: "38"
        if re.fullmatch(r'\s*\d+\s*', line):
            continue
        
        # 2. Question number cleaning: Ignore lines that start with a number followed by a dot.
        #    Example: "1. This is a question number line."
        if re.match(r'^\d+\.\s*', line):
            continue
            
        # 3. Irrelevant content cleaning: Ignore lines that start with 4 or more whitespace characters.
        #    These are typically headers or footers from PDF conversion.
        #    Example: "    《前行》第020课"
        if re.match(r'^\s{4,}.*', line):
            continue
            
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)

def parse_qa_file(file_content):
    """
    Parses the cleaned text content to extract question-answer pairs.
    Questions start with "问：" and answers start with "答：".
    It also handles the removal of answerer names that might be present at the end of answers.
    """
    qa_pairs = []
    # Pattern to capture question and answer based on "问：" and "答：" delimiters.
    # It uses re.DOTALL to allow '.' to match newlines.
    # The pattern looks for "问：", captures everything until "答：", then captures everything
    # until the next "问：" or the end of the file.
    # The answerer name pattern `(?:\s*\(C[1-9]?\))?` is included to help delineate the answer
    # but will be explicitly removed in a later cleaning step.
    pattern = re.compile(r'问：(.*?)\n答：(.*?)(?=\n问：|\Z)', re.DOTALL)
    
    matches = pattern.findall(file_content)

    for match in matches:
        question_text = match[0].strip()
        answer_text = match[1].strip()
        
        # Post-processing cleaning for question and answer text
        # Remove all newlines from question and answer
        question_text = question_text.replace('\n', '').strip()
        answer_text = answer_text.replace('\n', '').strip()

        # Second pass for answerer name cleaning:
        # Remove the answerer name from the answer text if it's still there.
        # This handles cases like "(C1)" or "(A)" at the end of the answer.
        answer_text = ANSWERER_NAME_PATTERN.sub('', answer_text).strip()

        qa_pairs.append({
            "question": question_text,
            "answer": answer_text
        })
    
    if not qa_pairs:
        print("Warning: No Q&A pairs were parsed. Check file format and regex.", file=sys.stderr)

    return qa_pairs

def generate_markdown_doc(qa_data, output_md_filename="fine_tune_qf.md"):
    """
    Generates a markdown file containing the Q&A pairs from the qa_data.
    """
    markdown_content = ["# QF Q&A Dataset for Fine-tuning\n"]

    for i, item in enumerate(qa_data):
        markdown_content.append(f"## Question {i+1}:")
        markdown_content.append(f"**Q:** {item['question']}\n")
        markdown_content.append(f"**A:** {item['answer']}\n")
        markdown_content.append("---\n") # Separator for readability

    with open(output_md_filename, 'w', encoding='utf-8') as md_file:
        md_file.write('\n'.join(markdown_content))
    
    print(f"Successfully generated markdown documentation to {output_md_filename}", file=sys.stderr)


if __name__ == "__main__":
    # Ensure the script is run with the correct Python environment.
    # This block handles reading the input file, cleaning, parsing, and writing the output.
    try:
        # Attempt to open and read the 'qf.txt' file.
        # The file is expected to be large, but we read it once for processing.
        with open('qf.txt', 'r', encoding='utf-8') as f:
            file_content = f.read()
    except FileNotFoundError:
        # Handle the case where 'qf.txt' is not found.
        print("Error: qf.txt not found. Please ensure the file is in the same directory as the script.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        # Handle other potential errors during file reading.
        print(f"Error reading qf.txt: {e}", file=sys.stderr)
        sys.exit(1)

    # Perform the initial cleaning pass on the raw file content.
    cleaned_content = clean_text_initial_pass(file_content)
    # Parse the cleaned content to extract Q&A pairs.
    qa_data = parse_qa_file(cleaned_content)

    # Define the output filename for the fine-tuning data.
    output_filename = "gemma3_finetune_data_qf.jsonl"
    with open(output_filename, 'w', encoding='utf-8') as outfile:
        # Iterate through each extracted Q&A pair and format it for Gemma3 fine-tuning.
        for item in qa_data:
            user_content = item["question"]
            assistant_content = item['answer']
            
            # Construct the fine-tuning entry in Gemma3 format.
            fine_tune_entry = {
                "conversations": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content}
                ],
                "source": "qf-qa-dataset", # Placeholder source for the dataset.
                "score": 5.0 # Placeholder score, can be adjusted if a scoring mechanism is implemented.
            }
            # Write each entry as a JSON line to the output file.
            outfile.write(json.dumps(fine_tune_entry, ensure_ascii=False) + '\n')

    # Provide feedback on the processing outcome.
    if not qa_data:
        print("Warning: No Q&A pairs were processed. Check file format and parsing logic.", file=sys.stderr)
    else:
        print(f"Successfully generated {len(qa_data)} Q&A entries to {output_filename}", file=sys.stderr)
    
    # Generate the markdown documentation file
    generate_markdown_doc(qa_data)