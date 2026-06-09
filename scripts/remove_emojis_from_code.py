#!/usr/bin/env python3
"""
Remove emojis from Python code files, except for UI display code (app.py).
Keeps emojis only in user-facing UI strings.
"""

import re
from pathlib import Path

# Emoji pattern - matches most common emojis
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001F9FF"  # Misc Symbols and Pictographs
    "\U0001F600-\U0001F64F"  # Emoticons
    "\U0001F680-\U0001F6FF"  # Transport and Map
    "\U00002600-\U000027BF"  # Misc symbols
    "\U00002700-\U000027BF"  # Dingbats
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "]+",
    flags=re.UNICODE
)

# Files to exclude from emoji removal (UI files where emojis are okay)
EXCLUDE_FILES = {
    'app.py',  # Main Streamlit UI - keep emojis for user display
}

# Directories to process
INCLUDE_DIRS = [
    'src/',
    'scripts/',
    'tests/',
    'evals/',
]

def should_process_file(file_path: Path) -> bool:
    """Check if file should be processed for emoji removal."""
    # Skip if in exclude list
    if file_path.name in EXCLUDE_FILES:
        return False
    
    # Only process Python files
    if file_path.suffix != '.py':
        return False
    
    # Check if in included directories
    for include_dir in INCLUDE_DIRS:
        if str(file_path).startswith(include_dir):
            return True
    
    return False

def remove_emojis_from_file(file_path: Path) -> tuple[int, int]:
    """
    Remove emojis from a file.
    Returns (lines_changed, emojis_removed)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_lines = content.split('\n')
        lines_changed = 0
        emojis_removed = 0
        
        new_lines = []
        for line in original_lines:
            # Count emojis in this line
            emoji_matches = EMOJI_PATTERN.findall(line)
            if emoji_matches:
                # Remove emojis
                new_line = EMOJI_PATTERN.sub('', line)
                # Clean up extra spaces that might be left
                new_line = re.sub(r'  +', ' ', new_line)
                new_line = new_line.strip() + '\n' if line.endswith('\n') else new_line.strip()
                
                new_lines.append(new_line)
                lines_changed += 1
                emojis_removed += len(emoji_matches)
            else:
                new_lines.append(line)
        
        if lines_changed > 0:
            # Write back
            new_content = '\n'.join(new_lines)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
        
        return lines_changed, emojis_removed
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 0, 0

def main():
    """Main function to remove emojis from code files."""
    print("Removing emojis from Python code files (excluding UI files)...")
    print("=" * 80)
    
    total_files_processed = 0
    total_files_changed = 0
    total_lines_changed = 0
    total_emojis_removed = 0
    
    # Process each directory
    for include_dir in INCLUDE_DIRS:
        dir_path = Path(include_dir)
        if not dir_path.exists():
            print(f"Skipping {include_dir} (not found)")
            continue
        
        print(f"\nProcessing {include_dir}...")
        
        # Find all Python files
        py_files = list(dir_path.rglob('*.py'))
        
        for py_file in py_files:
            if should_process_file(py_file):
                total_files_processed += 1
                lines_changed, emojis_removed = remove_emojis_from_file(py_file)
                
                if lines_changed > 0:
                    total_files_changed += 1
                    total_lines_changed += lines_changed
                    total_emojis_removed += emojis_removed
                    print(f"  {py_file}: {emojis_removed} emojis removed from {lines_changed} lines")
    
    print("\n" + "=" * 80)
    print("Summary:")
    print(f"  Files processed: {total_files_processed}")
    print(f"  Files changed: {total_files_changed}")
    print(f"  Lines changed: {total_lines_changed}")
    print(f"  Emojis removed: {total_emojis_removed}")
    print("\nEmojis kept in:")
    for exclude_file in EXCLUDE_FILES:
        print(f"  - {exclude_file} (UI display)")
    print("=" * 80)

if __name__ == '__main__':
    main()

# Made with Bob
