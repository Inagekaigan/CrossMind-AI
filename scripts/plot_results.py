"""Create a static research figure from saved measured results only."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
main = json.loads((root/'experiments/jaad_default/comparison.json').read_text())
ablation = json.loads((root/'experiments/ablations/comparison.json').read_text())
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), layout='constrained')
for ax, key, names, title in [
    (axes[0], 'f1', ['intention', 'joint'], 'Crossing initiation F1'),
    (axes[1], 'minFDE', ['trajectory', 'joint'], 'Endpoint error: minFDE@3')]:
    ax.bar(names, [main[n][key]['mean'] for n in names],
           yerr=[main[n][key]['std'] for n in names], capsize=5, color=['#4378a8', '#2a9d8f'])
    ax.set_title(title)
    ax.grid(axis='y', alpha=.2)
axes[1].set_ylabel('Normalized image distance')
values = [ablation['position']['roc_auc'], ablation['position_velocity']['roc_auc'], main['joint']['roc_auc']]
axes[2].bar(['Position', '+ velocity', '+ neighbors'], [v['mean'] for v in values],
            yerr=[v['std'] for v in values], capsize=5, color=['#4378a8','#2a9d8f','#e9a251'])
axes[2].set_title('Joint-model feature ablation: ROC-AUC')
axes[2].grid(axis='y', alpha=.2)
axes[2].set_ylim(0, 1)
fig.suptitle('JAAD at-risk cohort pilot · three-seed means ± sample SD · not significance intervals', fontsize=11)
fig.savefig(root/'reports/pilot_results.png', dpi=170)
fig.savefig(root/'reports/pilot_results.pdf')
plt.close(fig)
