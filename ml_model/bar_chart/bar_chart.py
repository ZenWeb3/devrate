import matplotlib.pyplot as plt
import numpy as np

# Metrics and values
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
rf_values = [99, 96, 97, 96]
svc_values = [97, 97, 99, 98]

# X-axis positions
x = np.arange(len(metrics))
width = 0.35  # width of the bars

# Plotting
fig, ax = plt.subplots(figsize=(8,5))
rects1 = ax.bar(x - width/2, rf_values, width, label='Random Forest', color='#1f77b4')
rects2 = ax.bar(x + width/2, svc_values, width, label='SVC', color='#ff7f0e')

# Labels and title
ax.set_ylabel('Percentage (%)')
ax.set_title('Performance Comparison of RF and SVC')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylim(0, 105)
ax.legend()

# Display values on top of bars
for rects in [rects1, rects2]:
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0,3),  # offset
                    textcoords="offset points",
                    ha='center', va='bottom')


plt.savefig('ml_model/results/bar_chart_performance_comparison.png')  # saves figure instead of showing