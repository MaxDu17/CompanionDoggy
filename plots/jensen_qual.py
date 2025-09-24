import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats

# Load Jensen's CSV data
csv_file = "/Users/maxjdu/Documents/GitHub/CompanionDoggy/logs/jensen/Snoopie Extended User Study - Jensen (Responses) - Form Responses 1.csv"
# csv_file = 'Snoopie Extended User Study - Jensen (Responses) - Form Responses 1.csv'
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
    Plot Jensen's average responses with standard error for each ranking question
    """
    # Shortened question names for display
    question_names = ["Easy", "Intuitive", "Predictable", "Natural", "Fun", "Helpful", "Trust", "Use Again"]
    
    # Calculate mean and standard error for each question across all sessions
    means = np.mean(responses, axis=0)
    std_errors = stats.sem(responses, axis=0)
    
    print("Jensen's average responses with standard error:")
    for i, question in enumerate(question_names):
        print(f"{question}: {means[i]:.2f} ± {std_errors[i]:.2f}")
    
    # Create the figure
    # fig, ax = plt.subplots(figsize=(10, 6))
    fig, ax = plt.subplots(figsize=(8, 3))

    
    # Set up the bar positions
    bar_width = 0.6
    question_positions = np.arange(len(question_names))
    
    # Plot bars with error bars
    bars = ax.bar(question_positions, means, 
                 width=bar_width, 
                 color='mediumseagreen', 
                 alpha=0.8,
                 yerr=std_errors,
                 capsize=0,
                 error_kw={'linewidth': 2})
    
    # Set up the plot
    # ax.set_xlabel('Questions', fontname="Palatino", fontsize=12)
    # ax.set_ylabel('Likert Scale (1-7)', fontname="Palatino", fontsize=12)
    # ax.set_title("Jensen's Average Responses Across Sessions", fontname="Palatino", fontsize=14, pad=20)
    # Change y-axis tick label font size
    ax.tick_params(axis="y", labelsize=12)

    # Change x-axis tick label font size
    ax.tick_params(axis="x", labelsize=12)

    # Set x-axis ticks and labels
    ax.set_xticks(question_positions)
    ax.set_xticklabels(question_names, fontname="Palatino", fontsize=12)
    
    # Set y-axis
    ax.set_ylim(1, 7.5)
    ax.set_yticks([1, 2, 3, 4, 5, 6, 7])
    ax.set_yticklabels([1, 2, 3, 4, 5, 6, 7], fontname="Palatino")
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add grid
    # ax.grid(True, alpha=0.3, axis='y')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save and show
    plt.savefig('jensen_average_responses.png', dpi=300, bbox_inches='tight')
    plt.savefig('jensen_average_responses.pdf', dpi=300, bbox_inches='tight')
    plt.show()


def plot_jensen_scatter():
    """
    Plot Jensen's average responses with standard error for each ranking question
    """
    # Shortened question names for display
    question_names = ["Easy", "Intuitive", "Predictable", "Natural", "Fun", "Helpful", "Trust", "Use Again"]
    iteration_indices = [2, 5, 6, 0, 1, 3, 4, 7]
    # predictable, helpful, trust, easy, intuitive, natural, fun, use
    # Calculate mean and standard error for each question across all sessions
    means = np.mean(responses, axis=0)
    std_errors = stats.sem(responses, axis=0)

    print("Jensen's average responses with standard error:")
    for i, question in enumerate(question_names):
        print(f"{question}: {means[i]:.2f} ± {std_errors[i]:.2f}")

    # Create the figure
    # fig, ax = plt.subplots(figsize=(10, 6))
    # fig, ax = plt.subplots(figsize=(8, 3), ncols = 8)
    fig, ax = plt.subplots(figsize=(10, 2), ncols = 8)

    for i in range(8):
        # scatter = ax[i].scatter(np.arange(5), responses[:, i], color='mediumseagreen', alpha=0.8, marker="x")
        scatter = ax[i].plot(responses[:, iteration_indices[i]], color='mediumseagreen', alpha=0.8) #, marker="x")
        ax[i].set_ylim(1, 7.5)
        ax[i].set_xlim(-0.5, 5.05)
        ax[i].set_yticks([1, 2, 3, 4, 5, 6, 7])

        ax[i].set_yticklabels([1, 2, 3, 4, 5, 6, 7], fontname="Palatino")

        # 1) Remove top and right spines
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)

        # 2) Remove x ticks
        ax[i].set_xticks([])
        # if i > 0:
        #     ax[i].set_yticks([])

        # Replace with a single x-axis label
        ax[i].set_xlabel(question_names[iteration_indices[i]], fontname="Palatino", fontsize=10)

    # Set up the bar positions
    fig.subplots_adjust(wspace=0.5,
                        left=0.05,  # no left margin
                        right=0.95,  # no right margin
                        )  # increase/decrease as needed

    # Save and show
    plt.savefig('jensen_average_responses.png', dpi=300, bbox_inches='tight')
    plt.savefig('jensen_average_responses.pdf', dpi=300, bbox_inches='tight')
    plt.show()


# Run the plot
# plot_jensen_responses()
plot_jensen_scatter()
