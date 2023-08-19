
# Frequentist A/B testing

## Use Case: 
Suppose we want to test the website where we have two variants. Variant A is the current version of the website (control) and Variant B is the new version we would like to test (treatment). We want to know whether user converts or not ( conversion meaning users purchase at least one item then they are considered converted if not then not converted).

We want to test the following hypothesis:
## Null Hypothesis: We do not observe any difference between two Variants
## Alternative Hypothesis: There is a different between two Variants

From the hypothesis, we know that it will be two-sided test. This means that we are interested in detecting any kind of difference between the two variants, whether it's a positive difference in one direction or a negative difference in the other direction. It's a more general form of hypothesis testing compared to a one-sided test where we would be specifically interested in a difference in one particular direction.

Before we carry out any experiment, we need to determine the sample size that is nummber of observations we need to collect for the experiment in order to detect meaningful differences between two variants.

Minimum detectable change (MDE) is crucial in experimental design as it helps researchers make informed decisions about the feasibiblity  of study objectives and the resorces required. We often consider MDE to deterimine the sample size for the experiment. A smaller MDE implies that a study can detect smaller effects, but it usually requires larger sample size. MDE is also tied to statistical power which is the probability of correctly rejecting the null hypothesis when it is false (that is detecting a true effect).

In summary, MDE refers to the smallest or minimum effect size that a statistical test or experiment is capable of detecting with a specified level of statistical power. MDE is the smallest difference between groups or conditions that a study can distinguish from random variability, given a certain sample size and desired level of confidence.

Suppose for our experiment, we set MDE to 2%, alpha at 5%, beta at 95%, and estimated conversion for control group (Variant A) is 10%. Then, We will calculate the sample size required for this experiment with the calculate_sample_size_proportions function which is inside models folder. Note that the sample size calculation is per group.

We know that in order to detect 2% change, we need 5848 samples per group. Once we get the sample size required, in practical world we would start collecting data for the experiment. In order to mimic the true would scenario, we will create a simulated dataset.

Setting a high statistical power of 95% means that we have a strong likelihood (95% in this case) of detecting a true effect if it actually exists in your population. In other words, a high power indicates that our statistical test is capable of correctly rejecting the null hypothesis when it's false (i.e., when there is a real effect or difference). A high power is desirable because it reduces the risk of committing a Type II error, which occurs when you fail to reject the null hypothesis even though there is a true effect.

The significance level (often denoted as α) determines the probability of committing a Type I error, which occurs when you wrongly reject the null hypothesis when it's true. A commonly used significance level is 0.05 (or 5%), which means we're willing to accept a 5% chance of making a Type I error.

The standard normal plot visualizes the distribution of sample means under the assumption of a standard normal (Z) distribution. The plot shows the Z-score calculated 
from our data proportions which provide insights into whether the observed difference between the two groups' conversion rates is statistically significant. We can see from the plot that we fail to reject the null hypothesis that there is no difference in conversion rates.

From the plot above, we can see that the confidence interval overlaps between two variants. Confidence intervals capture uncertainty in the estimates. A small overlap acknowledges that there's some uncertainty about the true values of the parameters being estimated. It's important to communicate this uncertainty along with the significant result. The precision of the estimates matters. If the overlapping portion of the confidence intervals is small and narrow, it indicates that the estimates are relatively precise, which is a positive aspect.If the confidence intervals are slightly overlapping, but the null hypothesis has been rejected (i.e., you have a statistically significant result),it means that the statistical test has detected a significant difference between the groups despite the overlap. 

