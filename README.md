
## Use Case: 
Suppose we want to test the website where we have two variants. Variant A is the current version of the website (control) and Variant B is the new version we would like to test (treatment). We want to know whether user converts or not ( conversion meaning users purchase at least one item then they are considered converted if not then not converted).

## Hypothesis Generation:

### Null Hypothesis: We do not observe any difference in conversion rates between two Variants
### Alternative Hypothesis: There is a difference in conversion rates between two Variants

From the hypothesis, we know that it will be two-sided test. This means that we are interested in detecting any kind of difference between the two variants, whether it's a positive difference in one direction or a negative difference in the other direction. It's a more general form of hypothesis testing compared to a one-sided test where we would be specifically interested in a difference in one particular direction.

## Experimental Design and Sample Size Calculation

Before we carry out any experiment, we need to determine the sample size that is nummber of observations we need to collect for the experiment in order to detect meaningful differences between two variants. By determining the sample size we are achieving statistical power, controlling error rates, detecting meaningful effects, allocating resources effectively, and ensuring ethical considerations are met. It's a crucial step in experimental design to ensure the validity and reliability of the results.

Minimum detectable change (MDE) is crucial in experimental design as it helps us make informed decisions about the feasibiblity of study objectives and the resorces required. We often consider MDE to deterimine the sample size for the experiment. A smaller MDE implies that a study can detect smaller effects, but it usually requires larger sample size. MDE is also tied to statistical power which is the probability of correctly rejecting the null hypothesis when it is false (that is detecting a true effect).

In summary, MDE refers to the smallest or minimum effect size that a statistical test or experiment is capable of detecting with a specified level of statistical power. MDE is the smallest difference between groups or conditions that a study can distinguish from random variability, given a certain sample size and desired level of confidence.

Suppose for our experiment, we set MDE to 2%, alpha at 5%, beta at 95%, and estimated conversion for control group (Variant A) is 10%. Setting a high statistical power of 95% means that we have a strong likelihood (95% in this case) of detecting a true effect if it actually exists in the population. In other words, a high power indicates that our statistical test is capable of correctly rejecting the null hypothesis when it's false (i.e., when there is a real effect or difference). A high power is desirable because it reduces the risk of committing a Type II error, which occurs when we fail to reject the null hypothesis even though there is a true effect.The significance level (α) determines the probability of committing a Type I error, which occurs when we wrongly reject the null hypothesis when it's true. A commonly used significance level is 0.05 (or 5%), which means we're willing to accept a 5% chance of making a Type I error.

## Sample Size Calculation

Now, we will calculate the sample size required for this experiment with the calculate_sample_size_proportions function which is inside models folder. Note that the sample size calculation is per group. From the calculate_sample_size_proportions function we know that in order to detect 2% change, we need 5848 samples per group. We also carried out power analysis with different effect sizes, see the plot below for cross verification. Once we get the sample size required, in practical world we would start 
collecting data for the experiment. In order to mimic the true would scenario, we will create a simulated dataset.

![power](https://github.com/kkharel/Frequentist-AB-Testing/assets/59852121/4cdafdcc-fb58-4c95-ad0a-88344a2e95a6)

## Generated Data

Our generated data contains ~5848 observations per group which is collected over 30 day period. The columns of the dataset is as follows: User_ID representing the user who
came to either Variant A or Variant B, timestamp representing time and date when they visited the site, group representing whether they were sent to treatment or control group and converted representing whether they made at least one purchase or not.
Once we generate the data, we aggregate the data and make a 2x2 table with rows trials: count of observations, successes: count of conversion and columns control and treatment.

## Test-statistic and Critical Region

The standard normal plot visualizes the distribution of sample means under the assumption of a standard normal (Z) distribution. The plot shows the Z-score calculated 
from our data proportions which provide insights into whether the observed difference between the two groups' conversion rates is statistically significant.

Z-value: -2.4106584575622727

![standard_normal](https://github.com/kkharel/Frequentist-AB-Testing/assets/59852121/b002f1e3-60ac-440b-9a12-9569f6ed1791)

## P-value and Decision

p-value: 0.0159237524739011

If the p-value is less than or equal to the chosen significance level (alpha), we reject the null hypothesis in favor of the alternative hypothesis.

If the p-value is greater than the significance level, we fail to reject the null hypothesis.

Since our p-value is less than chosen significane level, we have evidence to reject the null hypothesis that we do not observe any difference between two variants.

The decision we made is to choose variant B as a new website design in terms of conversion rates.

## Confidence Interval
From the plot above, we can see that the confidence interval overlaps between two variants. Confidence intervals capture uncertainty in the estimates. A small overlap acknowledges that there's some uncertainty about the true values of the parameters being estimated. It's important to communicate this uncertainty along with the significant result. The precision of the estimates matters. If the overlapping portion of the confidence intervals is small and narrow, it indicates that the estimates are relatively precise, which is a positive aspect. If the confidence intervals are slightly overlapping, but the null hypothesis has been rejected (i.e., we have a statistically significant result), it means that the statistical test has detected a significant difference between the groups despite the overlap. 

![ci](https://github.com/kkharel/Frequentist-AB-Testing/assets/59852121/eb5c4202-7ef0-43a1-a216-36d61af899fa)


Confidence Interval for Variant A: (0.09647514491891594, 0.11214318613443562)

Confidence Interval for Variant B: (0.11005265069002206, 0.1266094560131


Guide Appendix:


### Hypothesis Testing

How to choose appropriate Statistical Test? In general,
1) Identify Data Type:
      Categorial - When data contains categories/groups
   
      Numerical - When data contains numbers - can be continuous or discrete

2) Research Question:

   Comparison of Groups: Comparing two or more groups to see whether there is a difference between them
   
   Association/Causation: examining relatinship between variables to see if changes in one variable are associated with changes in another
      
3) Number of Groups/Variables:
   
     One Group: Wants to determine if its mean differs from a known value, one sample t-test or one sample z-test
      
     Two Groups: comparing two groups, can use t-test (paired or independent), Mann-Whitney U test, chi-squared test, or Fisher's exact test depending on data and 
      assumptions
      
      Multiple Groups: ANOVA for continuous data, chi-squares or Fisher's exact test for categorical data
      
4) Assumptions:
   
      Different test have different assumptions such as normality, homogeneity of variances and independence. The data needs to meet these assumptions for that
      particular test. If the assumptions are violated then use transformation techniques or non-parametric tests
      
5) Data Distribution:
   
      Normal Distribution: If data follows normal distribution then parametric tests like t-test and ANOVA are appropriate

      Non-Normal Distribution: If data is not normally distributed then non-parametric tests Mann-Whitney U test or Kruskal-Wallis test are appropriate
      
6) Sample Design:
   
      Paired Samples: If we have data pairs that are related such as before and after intervention situations then paired t-test or Wilcoxon signed-rank test may be 
      appropriate
   
      Independent Samples: If samples are independent then we can use independent t-test or non-parametric alternatives
      
7) Data Relationships:
    
      Correlation: To measure the strength and direction of a linear relationship between two continuous variables, we can use Pearson Correlation for normally distributed 
      data and Spearman's rank correation for non-normally distributed data
      
      Regression: To predict one variable using other variable/variables. If we have categorical predictors then we want to use Logistic Regression
      
### Comparing Means:

  Z-test: Used to compare means of continuous variable between two independent groups when sample size is large, population standard deviation is known, data is 
          approximately normally distributed and the samples in each group must be independent of each other
  
  Independent Samples T-test: data within each group is normally distributed and has equal variance
  
  Paired-Samples T-test: Compare means within the same group at different points in time. The difference between paired observations are normally distributed
  
  One-Sample T-test: Compare mean of a sample to a known or hypothesized population mean with the assumption that data is normally distributed
  
  ANOVA: Comparing means of three or more independent groups. Assume data is normally distributed and has equal variances
  
  Welch's T-test: Modification of independent samples t-test that does not assume equal variances between groups
  
  Kruskal-Wallis Test: Non-parametric alternative to ANOVA for comparing means when assumptions of normality and equal variances are violated
  
  
### Comparing Proportions:
Z-test for Proportions: compare proportions from two independent groups. Assumes that sample size are sufficiently large
  
Chi-Squared Test for Proportions (2X2 Table): When we have 2X2  contigency table and want to compare proportions between groups. Commonly used in situations like comparing proportion of successes between groups
  
Chi-Squared Test for Proportions(Larger Tables): For situations where we have more than 2X2 tables, use extensions such as chi-squared test for independence or homogeneity
  
Fisher's exact test: When dealing with small sample sizes on 2X2 contigency table, useful when assumptions of chi-squared test are not met.
  
  
### Flow:
Develop Hypothesis: Null Hypothesis, Alternative Hypothesis

Choose the variables and metrics

Random Assignment - Control Group & Treatment Group

Collect Data

Perform Descriptive Statistics

Establish Significane Level

Identify Test Statistic

Determine p-value

Compare p-value to fixed Signifance Level 

Make a Decision

Interpret the Results

Consider Practical Significane

Implement the Winning Version

