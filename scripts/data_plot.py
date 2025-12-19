import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

def plot_movement_analysis(csv_file):
    # 1. Load Data
    if not os.path.exists(csv_file):
        print(f"Error: File {csv_file} not found.")
        return

    df = pd.read_csv(csv_file)
    
    # Identify unique joints recorded (filtering out _human/_robot suffixes) 
    joints = [col.replace('_human', '') for col in df.columns if col.endswith('_human')]
    
    if not joints:
        print("No valid joint data found in CSV.")
        return

    # Setup Plot
    # Subplot for each joint recorded
    num_joints = len(joints)
    fig, axes = plt.subplots(num_joints, 1, figsize=(10, 3 * num_joints), sharex=True)
    
    if num_joints == 1:
        axes = [axes] # Make iterable if only one joint

    # Generate Graphs
    for i, joint in enumerate(joints):
        ax = axes[i]

        human_data = df[f'{joint}_human']
        robot_data = df[f'{joint}_robot']

        cov_val = human_data.cov(robot_data)
        
        # 1.0 = Perfect tracking, -1.0 = Perfect Inverted tracking, 0.0 = No tracking
        corr_val = human_data.corr(robot_data)
        
        print(f"{joint:<30} | {cov_val: .6f}   | {corr_val: .6f}")

        # Plot Human Data (Raw Targets)
        ax.plot(df['frame'].to_numpy(), df[f'{joint}_human'].to_numpy(), 
                label='Human Target', color='blue', alpha=0.5, linestyle='--')
        
        # Plot Robot Data (Smoothed Output)
        ax.plot(df['frame'].to_numpy(), df[f'{joint}_robot'].to_numpy(), 
                label='Robot Output (Smoothed)', color='red', linewidth=2)
              
        title_text = (f"Joint: {joint.upper()}\n"
                      f"Covariance: {cov_val:.5f}  |  Correlation: {corr_val:.4f}")
        
        ax.set_title(title_text, fontsize=10, pad=10)
        ax.set_ylabel("Angle (Radians)")
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Frame Number")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        # Find newest CSV automatically
        files = [f for f in os.listdir('./Plots/') if f.startswith('gr2_movement_log') and f.endswith('.csv')]
        if files:
            key = lambda x: os.path.getctime(os.path.join('./Plots/', x))
            filename = max(files, key=key)
            filename = "./Plots/" + filename
            print(f"Auto-selected newest file: {filename}")
        else:
            print("Please provide a CSV filename.")
            sys.exit()

    plot_movement_analysis(filename)