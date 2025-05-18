# Uses Wilcoxon signed-rank test with LinguAlchemy metrics (accuracy or pearson) for URIEL and URIEL+ to determine statistical significance of results

from scipy.stats import wilcoxon

# LinguAlchemy metrics (accuracy or pearson) for URIEL experiment.
URIEL_metrics = []

# LinguAlchemy metrics (accuracy or pearson) for URIEL+ experiment.
URIELPlus_metrics = []

# Perform Wilcoxon signed-rank test
stat, p = wilcoxon(URIEL_metrics, URIELPlus_metrics)

print("Statistic:", stat)
print("p-value:", p)