import sys
import os

def format_submission(student_id, input_file, output_file):
    """
    Formats the classification results into the required submission (Submitty) format.
    
    Args:
        student_id (str): nasingh
        input_file (str):
        output_file (str):
    """
    # Read the classification results
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # Parse and sort the results
    results = []
    for line in lines:
        if ',' in line:
            filename, captcha = line.strip().split(',', 1)
            # Remove any extra spaces
            filename = filename.strip()
            captcha = captcha.strip()
            results.append((filename, captcha))
    
    # Sort by filename using natural sort (0-9a-f)
    results.sort(key=lambda x: x[0])
    
    # Write the formatted submission file with LF line endings
    with open(output_file, 'w', newline='\n') as f:
        # Write nasingh as first line
        f.write(f"{student_id}\n")
        
        # Write results
        for filename, captcha in results:
            f.write(f"{filename},{captcha}\n")

def main():
    if len(sys.argv) != 4:
        print("Usage: python format_submission.py <student_id> <input_file> <output_file>")
        print("Example: python format_submission.py smithj output.txt submission.csv")
        sys.exit(1)
    
    student_id = sys.argv[1]
    input_file = sys.argv[2]
    output_file = sys.argv[3]
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found")
        sys.exit(1)
    
    format_submission(student_id, input_file, output_file)
    print(f"Submission file '{output_file}' created successfully")

if __name__ == '__main__':
    main()
