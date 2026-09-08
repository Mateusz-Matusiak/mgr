#!/usr/bin/env python3
import os
import subprocess
import sys
import glob
import json

def run_batch_analysis():
    """
    Script to run algorithm-1.py on all case directories in ./data/
    """
    
    # Find all case directories
    data_dir = "./data"
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' not found.")
        sys.exit(1)
    
    # Pattern to match case directories (case1, case2, case17, etc.)
    case_pattern = os.path.join(data_dir, "case*/seg.nii.gz")
    case_files = glob.glob(case_pattern)
    
    if not case_files:
        print(f"No seg.nii.gz files found in {data_dir}/case* directories")
        sys.exit(1)
    
    # Sort the files to process them in order
    case_files.sort(key=lambda x: int(x.split('case')[1].split('/')[0]))
    
    print(f"Found {len(case_files)} case files to process:")
    for file in case_files:
        print(f"  - {file}")
    print()
    
    results = {}
    
    for i, case_file in enumerate(case_files, 1):
        case_name = os.path.basename(os.path.dirname(case_file))
        print(f"[{i}/{len(case_files)}] Processing {case_name}...")
        
        try:
            # Run algorithm-1.py with the case file (use absolute path)
            abs_case_file = os.path.abspath(case_file)
            result = subprocess.run(
                [sys.executable, "scoliosis.py", abs_case_file],
                capture_output=True,
                text=True,
                cwd="./algorithms"
            )
            
            if result.returncode == 0:
                # Parse the output (assuming it's a number for Cobb angle)
                output = result.stdout.strip()
                try:
                    cobb_angle = output
                    results[case_name] = cobb_angle
                    print(f"  ✓ Success: Cobb angle = {cobb_angle:.2f}°")
                except ValueError:
                    results[case_name] = output
                    print(f"  ✓ Success: {output}")
            else:
                error_msg = result.stderr.strip() or "Unknown error"
                results[case_name] = f"ERROR: "
                print(f"  ✗ Error: {error_msg}")
                
        except Exception as e:
            results[case_name] = f"EXCEPTION: {str(e)}"
            print(f"  ✗ Exception: {str(e)}")
        
        print()
    
    # Print summary
    print("="*60)
    print("BATCH ANALYSIS SUMMARY")
    print("="*60)
    
    successful_cases = 0
    failed_cases = 0
    cobb_angles = []
    
    for case_name, result in results.items():
        if isinstance(result, (int, float)):
            print(f"{case_name:12}: {result:6.2f}°")
            successful_cases += 1
            cobb_angles.append(result)
        elif result.startswith("ERROR") or result.startswith("EXCEPTION"):
            print(f"{case_name:12}: {result}")
            failed_cases += 1
        else:
            print(f"{case_name:12}: {result}")
            successful_cases += 1
    
    print("-" * 60)
    print(f"Successful: {successful_cases}")
    print(f"Failed:     {failed_cases}")
    print(f"Total:      {len(case_files)}")
    
    # Calculate statistics for successful cases
    if cobb_angles:
        import numpy as np
        print(f"\nCobb Angle Statistics:")
        print(f"Mean:    {np.mean(cobb_angles):.2f}°")
        print(f"Median:  {np.median(cobb_angles):.2f}°")
        print(f"Std Dev: {np.std(cobb_angles):.2f}°")
        print(f"Min:     {np.min(cobb_angles):.2f}°")
        print(f"Max:     {np.max(cobb_angles):.2f}°")
        
        # Count severity levels
        mild = sum(1 for angle in cobb_angles if 10 <= angle < 25)
        moderate = sum(1 for angle in cobb_angles if 25 <= angle < 40)
        severe = sum(1 for angle in cobb_angles if angle >= 40)
        normal = sum(1 for angle in cobb_angles if angle < 10)
        
        print(f"\nSeverity Distribution:")
        print(f"Normal (< 10°):      {normal}")
        print(f"Mild (10-25°):       {mild}")
        print(f"Moderate (25-40°):   {moderate}")
        print(f"Severe (≥ 40°):      {severe}")
    
    # Save results to files
    output_dir = "./algorithms/target"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save text summary
    output_file = os.path.join(output_dir, "batch_results.txt")
    with open(output_file, 'w') as f:
        f.write("Batch Analysis Results\n")
        f.write("=" * 50 + "\n\n")
        for case_name, result in results.items():
            f.write(f"{case_name}: {result}\n")
        f.write(f"\nSummary: {successful_cases} successful, {failed_cases} failed\n")
        
        if cobb_angles:
            f.write(f"\nCobb Angle Statistics:\n")
            f.write(f"Mean:    {np.mean(cobb_angles):.2f}°\n")
            f.write(f"Median:  {np.median(cobb_angles):.2f}°\n")
            f.write(f"Std Dev: {np.std(cobb_angles):.2f}°\n")
            f.write(f"Min:     {np.min(cobb_angles):.2f}°\n")
            f.write(f"Max:     {np.max(cobb_angles):.2f}°\n")
    
    # Save JSON results
    json_file = os.path.join(output_dir, "batch_results.json")
    with open(json_file, 'w') as f:
        json.dump({
            'results': results,
            'summary': {
                'successful': successful_cases,
                'failed': failed_cases,
                'total': len(case_files)
            },
            'statistics': {
                'mean': float(np.mean(cobb_angles)) if cobb_angles else None,
                'median': float(np.median(cobb_angles)) if cobb_angles else None,
                'std': float(np.std(cobb_angles)) if cobb_angles else None,
                'min': float(np.min(cobb_angles)) if cobb_angles else None,
                'max': float(np.max(cobb_angles)) if cobb_angles else None
            } if cobb_angles else None
        }, f, indent=2)
    
    # Save CSV for easy analysis
    csv_file = os.path.join(output_dir, "batch_results1.csv")
    with open(csv_file, 'w') as f:
        f.write("Case,Cobb_Angle,Status\n")
        for case_name, result in results.items():
            # if isinstance(result, (int, float)):
            #     f.write(f"{case_name},{result:.2f},Success\n")
            # else:
            index = result.index(":")
            x = result[:index]
            y = result[index:]
            f.write(f"{result}\n")
            # f.write(f"{case_name},{str(result).replace(".",",")}\n")
    
    print(f"\nResults saved to:")
    print(f"  - {output_file}")
    print(f"  - {json_file}")
    print(f"  - {csv_file}")

if __name__ == "__main__":
    run_batch_analysis()
