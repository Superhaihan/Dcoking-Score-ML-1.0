import subprocess

# Assuming ML.py and DL.py are in the "ML_DL" directory
ml_script_path = "ML_DL/ML.py"
dl_script_path = "ML_DL/DL.py"

# Run ML.py
print("Running Machine Learning script ML.py...")
subprocess.run(['python', ml_script_path], check=True)

# Ask the user if they want to run DL.py
user_input = input("Do you want to run the Deep Learning script DL.py? (yes/no): ").strip().lower()
if user_input == "yes":
    print("Running Deep Learning script DL.py...")
    subprocess.run(['python', dl_script_path], check=True)
else:
    print("Deep Learning script DL.py will not be run.")
