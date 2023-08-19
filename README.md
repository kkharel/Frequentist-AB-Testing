# Frequentist A/B Testing

## Use Case
Suppose we want to test a website with two variants. Variant A represents the current version of the website (control), and Variant B is the new version we aim to evaluate (treatment). Our objective is to determine whether users convert (make a purchase) or not.

## Hypothesis Generation
- **Null Hypothesis (H0):** No difference in conversion rates between the two variants.
- **Alternative Hypothesis (H1):** There is a significant difference in conversion rates between the two variants.

## Experimental Design and Sample Size Calculation
Determining the appropriate sample size is critical for achieving adequate statistical power, maintaining control over error rates, and effectively detecting meaningful effects. The Minimum Detectable Effect (MDE) is the smallest difference that a study can detect, impacting sample size and statistical power. A smaller MDE allows detection of smaller effects but often requires larger sample sizes.

For our experiment, we set MDE to 2%, alpha to 5%, beta to 95%, and estimated the conversion rate for the control group (Variant A) at 10%. By setting a high statistical power of 95%, we increase our ability to identify a true effect if it exists within the population.

## Sample Size Calculation
The calculated sample size per group is 5848. To mimic real-world conditions, we will simulate a dataset for the experiment.

## Generated Data
The generated dataset consists of approximately 5848 observations per group over a 30-day period. The dataset includes columns for User_ID (representing users for both variants), timestamp (indicating the visit time), group (differentiating treatment and control), and converted (indicating whether a purchase was made).

## Test-statistic and Critical Region
We visualize the distribution of sample means through a standard normal plot. This plot displays the Z-score computed from data proportions, which helps determine whether the observed difference between the conversion rates of the two groups is statistically significant.
Z-value: -2.4106584575622727

## P-value and Decision
The calculated p-value is 0.0159237524739011. When the p-value is less than or equal to the chosen significance level (alpha), we reject the null hypothesis in favor of the alternative hypothesis. Conversely, if the p-value exceeds the significance level, we fail to reject the null hypothesis.

Given our p-value is less than alpha, we have evidence to reject the null hypothesis, supporting the alternative hypothesis.

## Confidence Interval
Confidence intervals capture the uncertainty associated with parameter estimates. Overlapping confidence intervals indicate the presence of uncertainty regarding true parameter values. Even if confidence intervals slightly overlap while the null hypothesis is rejected (yielding a statistically significant result), this suggests that the test identified a significant difference between the groups, despite the overlap.

Confidence Interval (Variant A): (0.0965, 0.1121)
Confidence Interval (Variant B): (0.1101, 0.1266)
