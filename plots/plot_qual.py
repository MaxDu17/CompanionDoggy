import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


from user_likert import u0_eric_rating, u1_hung_rating, u2_jack_rating, u3_jason_rating, u4_priya_rating, u6_carrie_rating, u7_jasmin_rating, u8_bill_rating, u9_dian_rating, u10_jan_rating


all_ranks = [u0_eric_rating, u1_hung_rating, u2_jack_rating, u3_jason_rating, u4_priya_rating, u6_carrie_rating, u7_jasmin_rating, u8_bill_rating, u9_dian_rating, u10_jan_rating]

# plot times
final_ranks = {"none": [], "watch": [], "dog": [], "rank_none": [], "rank_watch": [], "rank_dog": []}
for a in all_ranks:
    final_ranks["none"].append(a["none"])
    final_ranks["watch"].append(a["watch"])
    final_ranks["dog"].append(a["dog"])
    final_ranks["rank_none"].append(a["rank_none"])
    final_ranks["rank_watch"].append(a["rank_watch"])
    final_ranks["rank_dog"].append(a["rank_dog"])

# print("final", final_ranks)

final_stats = {"none": [], "watch": [], "dog": [], "rank_none": [], "rank_watch": [], "rank_dog": []}
for k in ["none", "watch", "dog", "rank_none", "rank_watch", "rank_dog"]:
    mean = np.mean(np.array(final_ranks[k]), axis=0)
    # std = np.std(np.array(final_ranks[k]), axis=0)
    std = stats.sem(np.array(final_ranks[k]), axis=0)

    # if "overall" in k:
    #     mean = mean[-1]
    #     std = std[-1]
    final_stats[k] = [mean, std]

print("final", final_stats)

method_key = {"none": "No Pacer", "watch": "Watch", "dog": "Embodied (Robot)"}
colors = ["#C5c5c5", "#9b8da1", "#169c54", "#98d2ff", "#Ff8200"]

def plot_by_axis(final_stats, ax=None, legend=True, rank=False):
    # Regular columns only
    regular_groups = ["Easy", "Intuitive", "Natural", "Fun", "Use Again"]
    
    # Indices for the data arrays (0-indexed)
    regular_indices = [0, 1, 3, 4, 7]  # Easy, Intuitive, Natural, Fun, Use Again
    
    # let's rearrange the order for LRTB
    new_order = np.array(["none", "watch", "dog"])
    labels = [method_key[a] for a in new_order]

    # Set up the bar positions and width
    bar_width = 0.15
    
    # Create the figure with single subplot
    # fig, ax = plt.subplots(figsize=(10, 4))
    hfont = {'fontname':'Palatino'}
    fig, ax = plt.subplots(figsize=(10, 2))

    
    # Adjust subplot to leave space for legend below
    # plt.subplots_adjust(bottom=0.3)
    
    # Regular group positions
    column_spacing = 0.8
    r1_regular = np.arange(len(regular_groups)) * column_spacing
    r2_regular = [x + bar_width for x in r1_regular]
    r3_regular = [x + bar_width for x in r2_regular]
    
    ax.set_ylim(0.75, 7.25)
    
    # Plot regular group (all three bars)
    bars1_regular = ax.bar(r1_regular, np.array(final_stats["none"][0])[regular_indices], 
                           yerr=final_stats["none"][1][regular_indices], bottom=0, 
                           color='darkgrey', width=bar_width)
    bars2_regular = ax.bar(r2_regular, np.array(final_stats["watch"][0])[regular_indices], 
                           yerr=final_stats["watch"][1][regular_indices], bottom=0, 
                           color='lightblue', width=bar_width)
    bars3_regular = ax.bar(r3_regular, np.array(final_stats["dog"][0])[regular_indices], 
                           yerr=final_stats["dog"][1][regular_indices], bottom=0, 
                           color='mediumseagreen', width=bar_width)
    
    # Set x-axis ticks and labels for regular group
    ax.set_xticks(r2_regular, regular_groups, fontsize=12, font="Palatino")
    # ax.set_title("Experience with Pacing", fontname="Palatino", fontsize=14, pad=20)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_yticks([1,2,3,4,5,6,7])
    ax.set_yticklabels([1,2,3,4,5,6,7], fontname="Palatino")
    # ax.set_ylabel("Likert Scale", fontname="Palatino", fontsize=12)

    # Add legend
    if legend:
        sns.set(font="Palatino", style="white", font_scale=0.9)
        ax.legend(labels, frameon=False, loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=3)

    if rank:
        plt.savefig('bar_overall_ranks_experience.pdf')
        plt.savefig('bar_overall_ranks_experience.png')
    else:
        plt.savefig('bar_overall_likert_experience.pdf')
        plt.savefig('bar_overall_likert_experience.png')
    plt.show()

plot_by_axis(final_stats, rank=False, legend=False)
