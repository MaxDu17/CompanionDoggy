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
    axes_groups = ["Easy", "Intuitive", "Predictable", "Natural", "Fun", "Helpful", "Trust", "Use Again"]

    # let's rearrange the order for LRTB
    new_order = np.array(["none", "watch", "dog"])
    labels = [method_key[a] for a in new_order]

    # Set up the bar positions and width
    bar_width = 0.15
    r1 = np.arange(len(axes_groups))*0.8#*0.65
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    # Create the figure and axis objects
    fig, ax = plt.subplots(figsize=(10, 3))
    hfont = {'fontname':'Palatino'}
    ax.set_ylim(0.75, 7.25)
    
    # Adjust subplot to leave space for legend below
    plt.subplots_adjust(bottom=0.3)
    # ax.invert_yaxis()



    bars1 = plt.bar(r1, np.array(final_stats["none"][0]), yerr=final_stats["none"][1], bottom=0, color='darkgrey', width=bar_width)
    bars2 = plt.bar(r2, np.array(final_stats["watch"][0]), yerr=final_stats["watch"][1], bottom=0, color='lightblue',  width=bar_width)
    bars3 = plt.bar(r3, np.array(final_stats["dog"][0]), yerr=final_stats["dog"][1], bottom=0, color='mediumseagreen',  width=bar_width)

    # ax.set_xticks(np.arange(len(axes_groups)), axes_groups, fontsize=12, font="Palatino")
    ax.set_xticks(r2, axes_groups, fontsize=12, font="Palatino")
    if legend:
        ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_yticks([1,2,3,4,5,6,7])
    ax.set_yticklabels([1,2,3,4,5,6,7], fontname="Palatino")
    ax.set_ylabel("Likert Scale", fontname="Palatino", fontsize=12)

    if (legend):
        sns.set(font="Palatino", style="white", font_scale=0.9)
        plt.legend(labels, frameon=False, loc='lower center', bbox_to_anchor=(0.5, -0.35), ncol=5)

    if rank:
        plt.savefig('bar_overall_ranks.png')
    else:
        plt.savefig('bar_overall_likert.png')
    plt.show()

plot_by_axis(final_stats, rank=False, legend=True)
# plot_by_axis(ranking_avg, ranking_std, rank=True, legend=False)
