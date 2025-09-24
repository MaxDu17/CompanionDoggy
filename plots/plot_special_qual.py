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
    # Special columns only
    special_groups = ["Predictable", "Helpful", "Trust"]
    
    # Indices for the data arrays (0-indexed)
    special_indices = [2, 5, 6]  # Predictable, Helpful, Trust
    
    # let's rearrange the order for LRTB
    new_order = np.array(["watch", "dog"])  # Only watch and dog, no none
    labels = [method_key[a] for a in new_order]

    # Set up the bar positions and width
    bar_width = 0.15
    
    # Create the figure with single subplot
    # fig, ax = plt.subplots(figsize=(5, 4))
    fig, ax = plt.subplots(figsize=(5, 3.5))
    hfont = {'fontname':'Palatino'}
    
    # Adjust subplot to leave space for legend below
    # plt.subplots_adjust(bottom=0.3)
    
    # Special group positions
    column_spacing = 0.5
    r1_special = np.arange(len(special_groups)) * column_spacing
    r2_special = [x + bar_width for x in r1_special]
    
    ax.set_ylim(0.75, 7.25)
    
    # Plot special group (only watch and dog, no none)
    bars2_special = ax.bar(r1_special, np.array(final_stats["watch"][0])[special_indices], 
                           yerr=final_stats["watch"][1][special_indices], bottom=0, 
                           color='lightblue', width=bar_width)
    bars3_special = ax.bar(r2_special, np.array(final_stats["dog"][0])[special_indices], 
                           yerr=final_stats["dog"][1][special_indices], bottom=0, 
                           color='mediumseagreen', width=bar_width)
    
    # Set x-axis ticks and labels for special group
    ax.set_xticks([(r1_special[i] + r2_special[i])/2 for i in range(len(special_groups))],
                  special_groups, fontsize=14, font="Palatino")
    ax.set_xticklabels(special_groups, fontsize=14, fontname="Palatino")
    # ax.set_title("Technology Use", fontname="Palatino", fontsize=14, pad=20)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_yticks([1,2,3,4,5,6,7])
    ax.set_yticklabels([1,2,3,4,5,6,7], fontname="Palatino")
    # ax.set_ylabel("Likert Scale", fontname="Palatino", fontsize=12)

    # Add legend
    if legend:
        sns.set(font="Palatino", style="white", font_scale=0.9)
        ax.legend(labels, frameon=False, loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=2)

    if rank:
        plt.savefig('bar_overall_ranks.png')
        plt.savefig('bar_overall_ranks.pdf')
    else:
        plt.savefig('bar_overall_likert.png')
        plt.savefig('bar_overall_likert.pdf')
    plt.show()

plot_by_axis(final_stats, rank=False, legend=False)
