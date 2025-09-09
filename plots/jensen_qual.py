import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats

# Load Jensen's CSV data
csv_file = 'Snoopie Extended User Study - Jensen (Responses) - Form Responses 1.csv'
df = pd.read_csv(csv_file)

# Extract the ranking questions (columns 6-13 are the Likert scale questions)
ranking_questions = [
    "It was easy to keep my goal pace.",
    "It was intuitive to keep my goal pace.", 
    "The technology behaved predictably.",
    "It felt natural to keep my goal pace.",
    "It felt fun to run with this method.",
    "The technology was helpful in keeping my goal pace.",
    "I trusted the technology to help me keep my goal pace.",
    "I would use this method again to keep my goal pace."
]

# Get session numbers and responses
session_numbers = df['Session Number'].values
responses = df[ranking_questions].values

print("Jensen's responses:")
print("Session Numbers:", session_numbers)
print("Responses shape:", responses.shape)
print("Questions:", ranking_questions)

def plot_jensen_responses():
    """
    Plot Jensen's individual responses for each ranking question with varying green alpha values
    """
    # Shortened question names for display
    question_names = ["Easy", "Intuitive", "Predictable", "Natural", "Fun", "Helpful", "Trust", "Use Again"]
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Set up the bar positions
    bar_width = 0.15
    column_spacing = 1.0
    question_positions = np.arange(len(question_names)) * column_spacing
    
    # Define alpha values for different sessions (increasing from left to right)
    alphas = [0.3, 0.5, 0.7, 0.9]  # For sessions 1, 2, 3, 4
    base_color = 'mediumseagreen'  # Same shade of green for all sessions
    
    # Plot responses for each session
    for session_idx, session_num in enumerate(session_numbers):
        session_responses = responses[session_idx]
        
        # Calculate bar positions for this session
        session_positions = [pos + (session_idx - 1.5) * bar_width for pos in question_positions]
        
        # Plot bars for this session
        bars = ax.bar(session_positions, session_responses, 
                     width=bar_width, 
                     color=base_color, 
                     alpha=alphas[session_idx],
                     label=f'Session {session_num}')
    
    # Set up the plot
    ax.set_xlabel('Questions', fontname="Palatino", fontsize=12)
    ax.set_ylabel('Likert Scale (1-7)', fontname="Palatino", fontsize=12)
    ax.set_title("Jensen's Individual Responses Across Sessions", fontname="Palatino", fontsize=14, pad=20)
    
    # Set x-axis ticks and labels
    ax.set_xticks(question_positions)
    ax.set_xticklabels(question_names, fontname="Palatino", fontsize=10)
    
    # Set y-axis
    ax.set_ylim(0.5, 7.5)
    ax.set_yticks([1, 2, 3, 4, 5, 6, 7])
    ax.set_yticklabels([1, 2, 3, 4, 5, 6, 7], fontname="Palatino")
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add legend
    # ax.legend(frameon=False, loc='upper right', fontfamily="Palatino")
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='y')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save and show
    plt.savefig('jensen_individual_responses.png', dpi=300, bbox_inches='tight')
    plt.show()

# Run the plot
plot_jensen_responses()
