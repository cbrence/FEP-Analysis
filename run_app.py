"""
Custom runner script for FEP Analysis Dashboard.

This script ensures the dataset is in the correct location before launching the app.
"""

import os
import sys
import shutil
import subprocess

def ensure_dataset_location():
    """
    Make sure the dataset is copied to all potential locations where the app might look for it.
    """
    # Get the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Expected source path (where the data actually is)
    source_path = os.path.join(script_dir, 'data', 'raw', 'fep_dataset.csv')
    
    # If source path doesn't exist, look for it in current directory
    if not os.path.exists(source_path):
        potential_source = os.path.join('data', 'raw', 'fep_dataset.csv')
        if os.path.exists(potential_source):
            source_path = potential_source
    
    # If source path still doesn't exist, prompt the user
    if not os.path.exists(source_path):
        print(f"Dataset not found at {source_path}")
        print("Please enter the full path to the fep_dataset.csv file:")
        user_path = input().strip()
        if os.path.exists(user_path):
            source_path = user_path
        else:
            print(f"Error: Path {user_path} does not exist.")
            return False
    
    # Define all potential target paths where the app might look
    target_paths = [
        os.path.join(script_dir, 'data', 'fep_dataset.csv'),
        os.path.join(script_dir, 'fep_dataset.csv'),
        os.path.join('data', 'fep_dataset.csv')
    ]
    
    # Copy the dataset to all potential locations
    success = False
    for target_path in target_paths:
        target_dir = os.path.dirname(target_path)
        if not os.path.exists(target_dir):
            try:
                os.makedirs(target_dir, exist_ok=True)
            except:
                print(f"Warning: Could not create directory {target_dir}")
                continue
        
        try:
            print(f"Copying dataset from {source_path} to {target_path}")
            shutil.copy2(source_path, target_path)
            success = True
        except Exception as e:
            print(f"Warning: Could not copy to {target_path}: {str(e)}")
    
    return success

def run_app():
    """
    Run the Streamlit app.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(script_dir, 'webapp', 'app_update.py')
    app_update_path = os.path.join(script_dir, 'webapp', 'app-update.py')
    
    # Determine which app file to use
    if os.path.exists(app_update_path):
        app_file = app_update_path
    elif os.path.exists(app_path):
        app_file = app_path
    else:
        print("Error: Could not find 'app_update.py or app-update.py")
        return False
    
    print(f"Running Streamlit app: {app_file}")
    
    # Run the Streamlit app
    try:
        subprocess.run(['streamlit', 'run', app_file])
        return True
    except Exception as e:
        print(f"Error running Streamlit app: {str(e)}")
        return False

if __name__ == "__main__":
    print("Setting up FEP Analysis Dashboard...")
    success = ensure_dataset_location()
    
    if success:
        print("Dataset prepared successfully.")
        run_app()
    else:
        print("Error: Failed to prepare dataset. The app may not work correctly.")
        run_app()
