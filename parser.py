import re
import json


def parse_qa_file(file_content):
    qa_pairs = []
    # Regex to find blocks starting with a number, followed by question, then "答：", then answer.
    # It handles the initial "序文问答题" by not matching it.
    # The pattern captures: (question_number_prefix, question_text, answer_text)
    pattern = re.compile(r'(\d+、\s*)(.*?)\n答：(.*?)(?=\n\d+、|\Z)', re.DOTALL)
    
    matches = pattern.findall(file_content)

    for match in matches:
        question_num_prefix = match[0].strip()
        question_text = match[1].strip()
        answer_text = match[2].strip()

        # Clean up question text if it contains the number prefix
        question_text = re.sub(r'^\d+、\s*', '', question_text).strip()

        qa_pairs.append({
            "question_number": question_num_prefix.replace('、', '').strip(),
            "question": question_text,
            "answer": answer_text
        })
    
    if not qa_pairs:
        print("Warning: No Q&A pairs were parsed. Check file format and regex.", file=sys.stderr)

    return qa_pairs

if __name__ == "__main__":
    import sys
    try:
        with open('jushe.txt', 'r', encoding='utf-8') as f:
            file_content = f.read()
    except FileNotFoundError:
        print("Error: jushe.txt not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading jushe.txt: {e}", file=sys.stderr)
        sys.exit(1)

    qa_data = parse_qa_file(file_content)

    output_filename = "gemma3_finetune_data.jsonl"
    with open(output_filename, 'w', encoding='utf-8') as outfile:
        for item in qa_data:
            user_content = item["question"]
            
            # Only include the answer for the final output
            assistant_content = item['answer']
            
            fine_tune_entry = {
                "conversations": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content}
                ],
                "source": "jushe-qa-dataset", # Placeholder source
                "score": 5.0 # Placeholder score
            }
            outfile.write(json.dumps(fine_tune_entry, ensure_ascii=False) + '\n')

    if not qa_data:
        print("Warning: No Q&A pairs were processed. Check file format and parsing logic.", file=sys.stderr)
    else:
        print(f"Successfully generated {len(qa_data)} Q&A entries to {output_filename}", file=sys.stderr)