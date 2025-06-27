import os
import sys

# Check if a directory path was provided as a command-line argument
if len(sys.argv) < 2:
    print("Usage: python step2_5_rename_signals.py <path_to_tf_signals_directory>")
    sys.exit(1)

# Get the signals directory from the command-line argument
signals_directory = sys.argv[1]

if not os.path.isdir(signals_directory):
    print(f"Error: Directory not found at {signals_directory}")
    sys.exit(1)
else:
    print(f"Processing files in {signals_directory}")
    for filename in os.listdir(signals_directory):
        if filename.endswith('.npy'):
            # Expected format: GeneID.Version.OtherStuff.npy
            parts = filename.split('.')
            if len(parts) >= 3 and parts[-1] == 'npy':
                gene_id = parts[0]
                new_filename = f"{gene_id}.npy"
                old_filepath = os.path.join(signals_directory, filename)
                new_filepath = os.path.join(signals_directory, new_filename)

                # Avoid renaming if the file is already in the desired format
                if filename != new_filename:
                    try:
                        os.rename(old_filepath, new_filepath)
                        print(f"Renamed '{filename}' to '{new_filename}'")
                    except OSError as e:
                        print(f"Error renaming '{filename}': {e}")
                else:
                    print(f"'{filename}' is already in the correct format, skipping.")
            else:
                print(f"Skipping '{filename}', does not match expected format.")
        else:
            print(f"Skipping non-npy file: '{filename}'")

    print("Finished processing.")
