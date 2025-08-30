import pandas as pd
import os
import subprocess

def organize_audio_files():
    """
    Organize audio files from CSV into cohort-based folders (data/PD and data/HC)
    """
    
    # File paths
    csv_file = "all_audios_mapped_id_for_label/final_selected.csv"
    raw_wav_base = "Processed_data_sample_raw_voice/raw_wav"
    output_base = "data"
    
    # Create output directories
    pd_dir = os.path.join(output_base, "PD")
    hc_dir = os.path.join(output_base, "HC")
    os.makedirs(pd_dir, exist_ok=True)
    os.makedirs(hc_dir, exist_ok=True)
    
    print("Loading CSV file...")
    df = pd.read_csv(csv_file)
    print(f"Total records: {len(df)}")
    
    # Cohort distribution
    cohort_counts = df['cohort'].value_counts()
    print(f"Cohort distribution: {cohort_counts.to_dict()}")
    
    # Scan all available folders (both 0 and 1 subfolders)
    all_available_folders = set()
    for subfolder in ["0", "1"]:
        subfolder_path = os.path.join(raw_wav_base, subfolder)
        if os.path.exists(subfolder_path):
            folders = os.listdir(subfolder_path)
            all_available_folders.update(folders)
    
    print(f"Total available audio folders: {len(all_available_folders)}")
    
    copied_files = {"PD": 0, "HC": 0}
    missing_files = []
    
    print("\nProcessing files...")
    
    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            print(f"Processed: {idx}/{len(df)}")
            
        audio_id = str(row['audio_audio.m4a'])
        cohort = row['cohort']
        
        # Skip if not PD or HC
        if cohort not in ['PD', 'HC']:
            continue
            
        # Check if folder exists in either 0 or 1
        found_folder = None
        for subfolder in ["0", "1"]:
            check_path = os.path.join(raw_wav_base, subfolder, audio_id)
            if os.path.exists(check_path):
                found_folder = check_path
                break
        
        if found_folder:
            # Destination directory
            dest_dir = pd_dir if cohort == 'PD' else hc_dir
            dest_filename = f"{audio_id}.wav"
            
            # Use robocopy to copy files (handles long filenames)
            cmd = [
                "robocopy",
                found_folder,
                dest_dir,
                "audio_audio.m4a-*.wav",
                "/NFL", "/NDL", "/NJH", "/NJS", "/NP"
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
                
                if result.returncode < 8:  # Robocopy success codes
                    # Find copied file and rename
                    dest_files = [f for f in os.listdir(dest_dir) if f.startswith("audio_audio.m4a-") and audio_id in f]
                    
                    if dest_files:
                        old_path = os.path.join(dest_dir, dest_files[0])
                        new_path = os.path.join(dest_dir, dest_filename)
                        
                        if not os.path.exists(new_path):
                            os.rename(old_path, new_path)
                        
                        copied_files[cohort] += 1
                        
                        if copied_files[cohort] % 10 == 0:
                            print(f"  Copied {copied_files[cohort]} {cohort} files...")
                
            except Exception as e:
                missing_files.append({
                    'audio_id': audio_id,
                    'cohort': cohort,
                    'reason': f'Copy error: {str(e)}'
                })
        else:
            missing_files.append({
                'audio_id': audio_id,
                'cohort': cohort,
                'reason': 'Folder not found'
            })
    
    # Results summary
    print(f"\n{'='*50}")
    print("FINAL SUMMARY:")
    print(f"{'='*50}")
    print(f"Total CSV records: {len(df)}")
    print(f"Files copied to PD folder: {copied_files['PD']}")
    print(f"Files copied to HC folder: {copied_files['HC']}")
    print(f"Total copied: {sum(copied_files.values())}")
    print(f"Missing files: {len(missing_files)}")
    
    # Verify copied files
    try:
        pd_files = [f for f in os.listdir(pd_dir) if f.endswith('.wav')]
        hc_files = [f for f in os.listdir(hc_dir) if f.endswith('.wav')]
        
        print(f"\nVerification:")
        print(f"PD folder: {len(pd_files)} files")
        print(f"HC folder: {len(hc_files)} files")
        
        if pd_files:
            print(f"Sample PD files: {pd_files[:3]}")
        if hc_files:
            print(f"Sample HC files: {hc_files[:3]}")
    except:
        pass
    
    # Missing files summary
    if missing_files:
        missing_df = pd.DataFrame(missing_files)
        missing_by_cohort = missing_df.groupby('cohort').size()
        print(f"\nMissing by cohort: {missing_by_cohort.to_dict()}")
        
        # Save missing list
        missing_df.to_csv("missing_files.csv", index=False)
        print(f"Missing files saved to: missing_files.csv")
    
    print(f"\nFiles organized successfully!")
    print(f"PD files: {os.path.abspath(pd_dir)}")
    print(f"HC files: {os.path.abspath(hc_dir)}")

if __name__ == "__main__":
    organize_audio_files()
