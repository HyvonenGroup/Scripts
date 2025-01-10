# This is a small python script to plot a chromatogram of IEX (ion exchange chromatography) run, which converts the data from a .csv file into a small plot.
# created by Lena Quambusch, MH group, Cambridge, UK. 2024.

# You can call this script like the following: python3 akta-plot.py /path/to/csv_file.csv --x_limits 0 100 --y_limits 0 200 --title IEX 

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define the main function to handle plotting
def main(csv_file, x_limits, y_limits, title):
    # Load the CSV file, specifying that the header is in the 3rd row (index 2)
    df = pd.read_csv(csv_file, header=2)

    # CHECKPOINT: Display the columns to see their names 
    #print(df.columns)

    # Access specific columns by their names
    volume1 = df['ml']
    volume2 = df['ml.1']
    volume3 = df['ml.2']
    channel1 = df['mAU']
    channel2 = df['mS/cm']
    channel3 = df['%']

    # Set plot style to seaborn
    plt.style.use('seaborn-v0_8-white')

    # Create the figure and axis, plus twin axis 
    fig, ax1 = plt.subplots(figsize=(8, 5)) # Set figure size (width, height) 
    ax2 = ax1.twinx()

    # Create the plot with customized data point style
    line1, = ax1.plot(volume1, channel1, marker='o', linestyle='-', color='b', markersize=1, linewidth=2, label='UV280')
    line2, = ax2.plot(volume2, channel2, marker='o', linestyle='-', color='y', markersize=0.5, linewidth=1, label='Conductivity')
    line3, = plt.plot(volume3, channel3, marker='o', linestyle='-', color='orange', markersize=0.5, linewidth=1, label='Gradient')


    # Add labels and title
    ax1.set_xlabel('Volume (mL)', fontsize=14, fontweight='bold', color='black')
    ax1.set_ylabel('Absorbance (mAU)', fontsize=14, fontweight='bold', color='black')
    ax2.set_ylabel('RU', fontsize=14, fontweight='bold', color='black')
    plt.title(title, fontsize=16, fontweight='bold')

    # Customize axis 
    ax1.tick_params(axis='x', size=5, width=2, color='black', direction='out', labelsize=10, labelcolor='black')
    ax1.tick_params(axis='y', size=5, width=2, color='black', direction='out', labelsize=10, labelcolor='black')
    ax2.tick_params(axis='y', size=5, width=2, color='black', direction='out', labelsize=10, labelcolor='black')

    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_linewidth(2)   
    ax1.spines['left'].set_linewidth(2)  
    ax2.spines['right'].set_linewidth(2)     

    # Add a styled legend and combine legends
    lines = [line1, line2, line3]  # Collect handles
    labels = [line1.get_label(), line2.get_label(), line3.get_label()]  # Collect labels
    ax1.legend(lines, labels, loc='upper right', fontsize=10,  frameon=False, shadow=True)  # Combine legends


    # Set the x-axis limits to zoom in on a specific range (e.g., from x=2 to x=8)
    ax1.set_xlim(x_limits)

    # Optionally, you can also set the y-axis limits if you want to zoom in on the y-axis
    ax1.set_ylim(y_limits)
    ax2.set_ylim([-5, 140])


    # Save the plot as a PNG file -> Unhash to safe the file automatically
    #png_file = os.path.splitext(csv_file)[0] + '.png'
    #plt.savefig(png_file, format='png', dpi=300, bbox_inches='tight', transparent=True)

    # Display the plot
    plt.show()
    
# Setup argument parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot chromatogram data from a CSV file.')
    parser.add_argument('csv_file', type=str, help='The path to the CSV file containing the chromatogram data.')
    parser.add_argument('--x_limits', type=float, nargs=2, required=True,
                        help='The x-axis limits as two float values, e.g., 0 10')
    parser.add_argument('--y_limits', type=float, nargs=2, required=True,
                        help='The y-axis limits as two float values, e.g., 0 10')
    parser.add_argument('--title', type=str, required=True,
                        help='The title for the plot.')
                    

    args = parser.parse_args()

    # Call the main function with the provided arguments
    main(args.csv_file, args.x_limits, args.y_limits, args.title)

