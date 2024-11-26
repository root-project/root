import os
import subprocess

# /// \file
# /// \ingroup tutorial_hist
# /// \notebook
# /// \preview
# /// Example showing how to combine the various candle plot options.
# ///
# /// \macro_image (tcanvas_js)
# /// \macro_code
# ///
# /// \author Georg Troska
# /// \date September 2021


def list_files(directory):
    r = []
    for root, dirs, files in os.walk(directory):
        for name in files:
            r.append(os.path.join(root, name))
    return r


def add_preview_line(comment_prefix, lines):
    preview_key = '\\preview'
    preview_line = f'{comment_prefix} {preview_key} \n'
    before_key='\\macro_image'
    line_before = f'{comment_prefix} {before_key}'
    
    if preview_line not in lines:
        for i, line in enumerate(lines):
            if line.strip().startswith(line_before):
                lines.insert(i+1,   preview_line)
                return lines
    return lines

def add_js_image_line(comment_prefix, lines):

    for i, line in enumerate(lines):
        if line.strip().startswith(f'{comment_prefix} \\macro_image'):
            # Update the macro image line to include '(tcanvas_js)'
            if '(tcanvas_js)' not in line:
                lines[i] = f'{comment_prefix} \\macro_image (tcanvas_js)\n'
                return lines
            else:
                return lines  # Leave the line as is
        elif line.strip() == f'{comment_prefix} \\macro_code' and not any(l.strip().startswith(f'{comment_prefix} \\macro_image') for l in lines):
            # Insert the macro image line before '\\macro_code' if not already present
            lines.insert(i, f'{comment_prefix} \\macro_image (tcanvas_js)\n')
            return lines
        
    return lines


def add_or_update_macro_image_to_files(folder_path):
    # Supported file extensions and their comment styles
    comment_styles = {
        '.C': '///',
        '.py': '##'
    }

    trigger_words = ['Draw(', 'DrawCopy(', 'DrawClone(']    
    # Traverse through files in the given folder and its subfolders

    for file_path in list_files(folder_path):
        file_extension = os.path.splitext(file_path)[1]
                
        # Check for valid file extensions
        if os.path.isfile(file_path) and file_extension in comment_styles:
            comment_prefix = comment_styles[file_extension]
            print (f'Processing {file_path}')

            with open(file_path, 'r', encoding='utf-8') as file:
                orig_lines = file.readlines()
                new_lines = orig_lines.copy()

            # check if the \date command is present in the file
            if not any(f'{comment_prefix} \\date' in line for line in orig_lines):
                # git log -1 --date=format:'%b %Y' --pretty=format:"%ad" -- Fibonacci.C
                git_date = subprocess.check_output(['git', 'log','-1','--date=format:\'%B %Y\'', '--pretty=format:"%ad"', file_path]).decode('utf-8')
                # Format the output
                git_date = git_date.strip().replace('"', '').replace('\'', '')
                print (git_date)
                # Add the \date command to the file
                for i, line in enumerate(orig_lines):
                    if line.strip().startswith(f'{comment_prefix} \\author'):
                        new_lines.insert(i, f'{comment_prefix} \\date {git_date}\n')
                        break

            # Check if the file contains 'Draw(', 'DrawCopy(', or 'DrawClone('
            if any(phrase in ''.join(new_lines) for phrase in trigger_words):
                # Add or update the macro image line
                new_lines = add_js_image_line(comment_prefix, new_lines)
                # Add a preview line if it doesn't exist
                new_lines = add_preview_line(comment_prefix, new_lines)

            # Write changes back to the file if modified
            if new_lines != orig_lines:
                print (f'Updating {file_path}')
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.writelines(new_lines)

# Example usage
# include_directories = ['dataframe','fit','tree','roofit','hist','graph','math']
include_directories = ['hist']
folder_path = '/home/prozorov/dev/hackathon/root/tutorials/'  # Replace with the path to your folder

for mydir in include_directories:
    print (f'Processing {folder_path+mydir}')
    add_or_update_macro_image_to_files(folder_path+mydir)
