def load_file_lines(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()  # Reads all lines into a list
        return [line.strip() for line in lines]  # Removes extra whitespace/newlines
    except FileNotFoundError:
        print(f"Error: The file at '{file_path}' was not found.")
        return []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return []