import re
import json
import sys

def parse_qa_file(file_content):
    """
    Parses the raw text content to extract question-answer pairs from 'jushe.txt'.
    It expects questions to be numbered (e.g., "1、") followed by "答：".
    """
    qa_pairs = []
    # Regex to find blocks starting with a number, followed by question, then "答：", then answer.
    # The pattern captures: (question_number_prefix, question_text, answer_text)
    # re.DOTALL allows '.' to match newline characters.
    pattern = re.compile(r'(\d+、\s*)(.*?)\n答：(.*?)(?=\n\d+、|\Z)', re.DOTALL)
    
    matches = pattern.findall(file_content)

    for match in matches:
        question_num_prefix = match[0].strip()
        question_text = match[1].strip()
        answer_text = match[2].strip()

        # Clean up question text if it contains the number prefix (e.g., "1、")
        question_text = re.sub(r'^\d+、\s*', '', question_text).strip()

        # Remove all newline characters from question and answer text
        question_text = question_text.replace('\n', '').strip()
        answer_text = answer_text.replace('\n', '').strip()

        qa_pairs.append({
            "question_number": question_num_prefix.replace('、', '').strip(),
            "question": question_text,
            "answer": answer_text
        })
    
    if not qa_pairs:
        # Warning if no Q&A pairs are parsed, indicating a potential format issue.
        print("Warning: No Q&A pairs were parsed. Check file format and regex.", file=sys.stderr)

    return qa_pairs

def generate_markdown_doc(qa_data, output_md_filename="fine_tune_jushe.md"):
    """
    Generates a markdown file containing the Q&A pairs from the qa_data.
    """
    markdown_content = ["# Jushe Q&A Dataset for Fine-tuning\n"]

    for i, item in enumerate(qa_data):
        markdown_content.append(f"## Question {i+1}:")
        markdown_content.append(f"**Q:** {item['question']}\n")
        markdown_content.append(f"**A:** {item['answer']}\n")
        markdown_content.append("---\n") # Separator for readability

    with open(output_md_filename, 'w', encoding='utf-8') as md_file:
        md_file.write('\n'.join(markdown_content))
    
    print(f"Successfully generated markdown documentation to {output_md_filename}", file=sys.stderr)


if __name__ == "__main__":
    # Main execution block for the script.
    try:
        # Attempt to open and read the 'jushe.txt' file.
        with open('jushe.txt', 'r', encoding='utf-8') as f:
            file_content = f.read()
    except FileNotFoundError:
        # Handle the case where 'jushe.txt' is not found.
        print("Error: jushe.txt not found. Please ensure the file is in the same directory as the script.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        # Handle other potential errors during file reading.
        print(f"Error reading jushe.txt: {e}", file=sys.stderr)
        sys.exit(1)

    # Parse the file content to extract Q&A pairs.
    qa_data = parse_qa_file(file_content)

    # Define the output filename for the fine-tuning data.
    output_filename = "gemma3_finetune_data.jsonl"
    with open(output_filename, 'w', encoding='utf-8') as outfile:
        # Iterate through each extracted Q&A pair and format it for Gemma3 fine-tuning.
        for item in qa_data:
            user_content = item["question"]
            
            # Only include the answer for the assistant's content in the final output.
            assistant_content = item['answer']
            
            # Construct the fine-tuning entry in Gemma3 format.
            fine_tune_entry = {
                "conversations": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content}
                ],
                "source": "jushe-qa-dataset", # Placeholder source for the dataset.
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