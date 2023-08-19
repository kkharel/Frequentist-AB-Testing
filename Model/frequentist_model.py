import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  
from plotnine import ggplot
import scipy as sp
import scipy.stats as sp_stats
import statsmodels as sm



def calculate_sample_size_proportions(MDE, alpha, beta, p_control, mode='one_sided'):
  if mode not in ['one_sided', 'two_sided']:
    raise ValueError('The modes are one_sided and two_sided')
    
  Z_alpha_over_2 = sp_stats.norm.ppf(1 - alpha / 2)
  Z_beta = sp_stats.norm.ppf(1 - beta)
    
  pooled_variance = p_control * (1 - p_control)
    
  if mode == 'two_sided':
    n = (2 * (Z_alpha_over_2 + Z_beta)**2 * pooled_variance) / MDE**2
  else:
    n = (Z_alpha_over_2**2 * pooled_variance) / MDE**2
  return int(np.ceil(n))  

MDE = 0.02  # Minimum Detectable Effect (2% improvement)
alpha = 0.05  # Significance level (5%)
beta = 0.05  # Power (95%)
p_control = 0.10  # Estimated proportion in the control group (10%)


required_sample_per_group = calculate_sample_size_proportions(MDE, alpha, beta, p_control, mode='two_sided')
required_sample_per_group

def generate_sample_data(samples, p_control, MDE):
  num_users_per_group = samples

  np.random.seed(42)

  start_date = pd.to_datetime('2023-08-18')
  end_date = start_date + pd.DateOffset(days=29)
  timestamps = pd.date_range(start=start_date, end=end_date, freq='H').tolist()

  group_assignments_control = np.full(num_users_per_group, 'control')
  group_assignments_treatment = np.full(num_users_per_group, 'treatment')

  np.random.seed(42)
  conversion_probabilities_control = np.full(num_users_per_group, p_control)
  conversion_status_control = np.random.rand(num_users_per_group) <= conversion_probabilities_control

  conversion_probabilities_treatment = np.full(num_users_per_group, p_control + MDE)
  conversion_status_treatment = np.random.rand(num_users_per_group) <= conversion_probabilities_treatment

  control_data = pd.DataFrame({
    'user_id': np.arange(1, num_users_per_group + 1),
    'timestamp': np.random.choice(timestamps, size=num_users_per_group, replace=True),
    'group': group_assignments_control,
    'converted': conversion_status_control.astype(int)
  })

  treatment_data = pd.DataFrame({
    'user_id': np.arange(1, num_users_per_group + 1),
    'timestamp': np.random.choice(timestamps, size=num_users_per_group, replace=True),
    'group': group_assignments_treatment,
    'converted': conversion_status_treatment.astype(int)
  })

  sample_data = pd.concat([control_data, treatment_data], ignore_index=True)
  return sample_data

sample_data = generate_sample_data(samples = required_sample_per_group, p_control = 0.10, MDE = 0.02)

control_group_data = sample_data[sample_data['group'] == 'control']
control_group_data['converted'].mean()

treatment_group_data = sample_data[sample_data['group'] == 'treatment']
treatment_group_data['converted'].mean()

control_count = sample_data[sample_data['group'] == 'control'].shape[0]
control_count
treatment_count = sample_data[sample_data['group'] == 'treatment'].shape[0]
treatment_count

unique_dates = sample_data['timestamp'].dt.date.nunique()
unique_dates

print(sample_data)


             # control,         treatment
# trials      total sample      total sample
# successes   converted         converted

aggregated_data = sample_data.groupby('group').agg(
    trials=('user_id', lambda x: x.nunique()),
    successes=('converted', 'sum'),
)

print(aggregated_data.T)


def standard_normal(x_bar_norm, legend_title):
  x_bar_norm = abs(x_bar_norm)
  x = np.arange(-3, 3, 0.01)
  y = np.array([sp_stats.norm.pdf(i, loc=0, scale=1) for i in x])
  crit_mask = (x >= -x_bar_norm) & (x <= x_bar_norm)
  plt.figure(figsize=(7, 7))
  plt.plot(x, y, color='black', linewidth=1)
  plt.fill_between(x, y, where=crit_mask, color='blue', alpha=0.5, label='True')
  plt.fill_between(x, y, where=~crit_mask, color='red', alpha=0.5, label='False')
  plt.axvline(1.96, color='green', linestyle='--')
  plt.axvline(-1.96, color='green', linestyle='--')
  plt.axvline(x_bar_norm, color='red', linestyle='--')
  plt.xlabel('Sample Means', fontsize=7)
  plt.ylabel('Probability Density Function', fontsize=7)
  plt.title('Standard Normal Distribution of Sample Mean', fontsize=7)
  legend = plt.legend(loc=1, title=legend_title)
  legend.get_title().set_fontsize(7)
  for label in legend.get_texts():  
    label.set_fontsize(5)
  plt.xticks(fontsize=7)
  plt.yticks(fontsize=7)
  plt.show()

def proportions(successes_A, successes_B, trials_A, trials_B, mode='two_sided'):
  proportion_A = successes_A / trials_A
  proportion_B = successes_B / trials_B

  overall_proportion = (successes_A + successes_B) / (trials_A + trials_B)
  pooled_std_error = np.sqrt(overall_proportion * (1 - overall_proportion) * (1 / trials_A + 1 / trials_B))

  z_score = (proportion_A - proportion_B) / pooled_std_error

  if mode == 'two_sided':
    p_value = 2 * (1 - sp.stats.norm.cdf(np.abs(z_score)))
  elif mode == 'one_sided':
    p_value = 1 - sp.stats.norm.cdf(z_score)
  else:
    raise ValueError("Available modes are 'one_sided' and 'two_sided'")

  return z_score, p_value

data = aggregated_data.T
data.control[1]
data.treatment[1]
z_value, p_value = proportions(data.control[1], data.treatment[1], data.control[0], data.treatment[0], mode = "two_sided")
z_value
print(f'Z-value: {z_value}; p-value: {p_value}')

standard_normal(x_bar_norm=z_value, legend_title="No Difference in Conversion Rates")


def confidence_interval_plot(proportion_A, proportion_B, sample_A, sample_B, alpha=0.05):
  pooled_proportion = (proportion_A * sample_A + proportion_B * sample_B) / (sample_A + sample_B)
  se_A = np.sqrt(proportion_A * (1 - proportion_A) / sample_A)
  se_B = np.sqrt(proportion_B * (1 - proportion_B) / sample_B)
  z_score = sp_stats.norm.ppf(1 - alpha / 2)
  
  x_A = np.arange(proportion_A - 3 * se_A, proportion_A + 3 * se_A, 0.0001)
  x_B = np.arange(proportion_B - 3 * se_B, proportion_B + 3 * se_B, 0.0001)
    
  y_A = np.array([sp_stats.norm.pdf(i, loc=proportion_A, scale=se_A) for i in x_A])
  y_B = np.array([sp_stats.norm.pdf(i, loc=proportion_B, scale=se_B) for i in x_B])
  
  ci_A = (proportion_A - se_A * z_score, proportion_A + se_A * z_score)
  ci_B = (proportion_B - se_B * z_score, proportion_B + se_B * z_score)
  
  plt.figure(figsize=(10, 6))
  plt.fill_between(x_A, y_A, alpha=0.5, label='Variant A', color = 'purple')
  plt.fill_between(x_B, y_B, alpha=0.5, label='Variant B', color = 'green')
    
  plt.axvline(proportion_A + se_A * z_score, color='red', linestyle='--')
  plt.axvline(proportion_A - se_A * z_score, color='red', linestyle='--')
  plt.axvline(proportion_B + se_B * z_score, color='blue', linestyle='--')
  plt.axvline(proportion_B - se_B * z_score, color='blue', linestyle='--')
    
  plt.xlabel('Sample Mean Distribution of Each Variant', fontsize = 7)
  plt.ylabel('Probability Density Function', fontsize = 7)
  plt.title(f'Confidence Intervals at alpha={alpha}', fontsize = 7)
  legend = plt.legend(loc=1)
  legend.get_title().set_fontsize(7)
  for label in legend.get_texts():  
    label.set_fontsize(5)
  plt.xticks(fontsize=7)
  plt.yticks(fontsize=7)
  plt.grid(True)
  plt.show()
  return ci_A, ci_B

confidence_interval_plot(data.control[1]/data.control[0], data.treatment[1]/data.treatment[0], data.control[0], data.treatment[0], alpha = 0.05)
ci_A, ci_B = confidence_interval_plot(data.control[1]/data.control[0], data.treatment[1]/data.treatment[0], data.control[0], data.treatment[0], alpha = 0.05)
print("Confidence Interval for Variant A:", ci_A)
print("Confidence Interval for Variant B:", ci_B)
