import os

def replace_class_id(folder_path, old_class_id=0, new_class_id=80):
    # Iterate through each file in the specified folder
    for filename in os.listdir(folder_path):
        # Check for .txt files
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            # Read the contents of the .txt file
            with open(file_path, 'r') as file:
                lines = file.readlines()

            # Modify the lines by replacing the old class ID with the new class ID
            modified_lines = []
            for line in lines:
                parts = line.split()
                if len(parts) > 0 and parts[0] == str(old_class_id):  # Check if the first element is the old class ID
                    parts[0] = str(new_class_id)  # Replace with new class ID
                modified_lines.append(' '.join(parts))

            # Write the modified lines back to the file
            with open(file_path, 'w') as file:
                file.write('\n'.join(modified_lines))

            print(f"Updated {filename}: Replaced class ID {old_class_id} with {new_class_id}")

# Example usage:
folder_path = '/Users/benzionfisher/PycharmProjects/camera_detection/dataset/labels'  # Replace with your folder path
replace_class_id(folder_path)
