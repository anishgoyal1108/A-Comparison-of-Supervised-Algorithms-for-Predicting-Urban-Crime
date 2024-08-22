# A Comparison of Supervised Algorithms for Predicting Urban Crime
**This research project was conducted during the 2021-2022 school year at the Gwinnett School of Math, Science, and Technology.  It was originally titled *Polynomial Interpolation and K-Means to Predict Crime Rates*, but this was a misleading name, so it has been renamed retroactively. I also don't have all of the code I used to create the visualizations or the primary dataset as a csv that I used for training, since it was a long time ago. However, you can see most of the project in Main.py, or the work I conducted in the actual paper/logbook, which may be found in their respective folders.**

<img src="https://img1.goodfon.com/wallpaper/nbig/c/73/atlanta-dzhordzhiya-dorogi-ogni.jpg"/>

## Purpose
The purpose of this research project is to compare the performance of Decision Trees, Random Forests, Naive Bayes Classifiers, Support Vector Machines, and Logistic Regression in predicting urban crime by neighborhood and date with the goal of discovering the algorithm that yields the highest accuracy. [Atlanta crime data](https://opendata.atlantapd.org/) from 2009-2018 was fed into each of these supervised algorithms to generate crime predictions by neighborhood for 2019 and subsequently compared against the actual crime occurrences for 2019 for each neighborhood using the root mean square error. Determining the best algorithm for crime prediction will assist law enforcement and policymakers in creating optimal predictive tools for efficient resource allocation and enhance public safety in urban areas overall. As a proof of concept for this, I also generated "heat maps" that crime analysts could use view to be able to know and predict crime hotspots for a particular date and neighborhood at a glance.

## Applications
The mere presence of law enforcement is sufficient to prohibit most crimes from occurring. If the best prediction models are used to determine the neighborhoods with the greatest crime rates at particular times, it would be an allocatively efficient use of resources by ensuring that police forces are strategically deployed in areas where they are needed the most. This approach can be further optimized through real-time data feed monitoring, enhanced geographic clustering with a fine-tuned grid size, and deep learning, such as a convolutional neural network.

## Quantifying Crime Severity
In order to generate optimal predictions for crime hotspots, one must also consider the severities of crimes in their calculations. I categorized all reportable crime types into four different groups of increasing weight, as shown below:

| Category | Crime Type               | Weight |
|----------|--------------------------|--------|
| 1        | Larceny                  | 1      |
| 2        | Auto Theft and Burglary  | 10     |
| 3        | Agg. Assault and Robbery | 100    |
| 4        | Homicide                 | 1000   |

This allows for the computation of an aggregate crime score, $S$, that quantifies the overall crime severity for a particular neighborhood and time interval using a weighted sum:

$$
S = \sum_{i=1}^{4} w_i \cdot x_i
$$

where: 
- $w_i$ is the weight associated with category $i$,
- $x_i$ is the number of occurrences of crimes in category $i$ for a particular neighborhood and time interval 

## Visualizations
### RMSEs For Each Supervised Algorithm
<p align="center">
<img src="https://github.com/anishgoyal1108/A-Comparison-of-Supervised-Algorithms-for-Predicting-Urban-Crime/blob/main/img/root_mean_square_error.png" />
</p>

### Predicted 2019 Crime Heat Map
<p align="center">
<img src="https://github.com/anishgoyal1108/A-Comparison-of-Supervised-Algorithms-for-Predicting-Urban-Crime/blob/main/img/2019_predict.png" />
</p>

### Actual 2019 Crime Heat Map
<p align="center">
<img src="https://github.com/anishgoyal1108/A-Comparison-of-Supervised-Algorithms-for-Predicting-Urban-Crime/blob/main/img/2019.png" />
</p>

## Conclusion
### The Best Algorithm?
This research demonstrates that Naive Bayes classifiers exhibited the highest accuracy and consistency (low RMSE values) compared to other algorithms, making them suitable for predicting future crime-heavy areas. However, they also had the highest runtime, with an average of 83 seconds on a cloud environment using current industry-grade graphics cards and processing power. 

### Creating a Scalable Model
For real-world applications, it might be more effective to use Random Forests, which delivered a much shorter runtime of only 21 seconds while maintaining over 50% accuracy overall. However, since regression models generally have the shortest runtimes, optimizing them for greater accuracy could be highly beneficial. Future attempts at a scalable model should incorporate non-linear regression, allowing for a more adaptable prediction function that could improve both accuracy and speed. One could also include data from socioeconomic datasets for the city being studied, which would enable features like average wages, city expenditures, and other socioeconomic indicators to be included in crime prediction models, potentially increasing prediction accuracy and reliability.

### How Can This Be Improved?
- Look at another city
- Analyze trends in specific categories of crime across neighborhoods
- Implement deep learning
- Test unsupervised algorithms
- Experiment with non-linear regression and incorporate more features

## References
 - [1] Elite Data Science. (2017, May 16). *Modern Machine Learning Algorithms: Strengths and Weaknesses.* Retrieved December 3, 2021, from https://elitedatascience.com/machine-learning-algorithms
 - [2] Kaloyanova, E., Ganchev, M., & Guide, S. (2020, March 10). *How to Combine PCA and K-means Clustering in Python?* 365 Data Science. Retrieved December 13, 2021, from https://365datascience.com/tutorials/python-tutorials/pca-k-means/
 - [3] Kumar, V. (2021, July 2). *Naïve Bayes Algorithm overview explained.* TowardsMachineLearning. Retrieved December 14, 2021, from https://towardsmachinelearning.org/naive-bayes-algorithm/
 - [4] Kumar, V. (2021, July 9). *Decision Tree Algorithm.* TowardsMachineLearning. Retrieved December 11, 2021, from https://towardsmachinelearning.org/decision-tree-algorithm/
 - [5] Kumar, V. (2021, July 16). *Random Forest.* TowardsMachineLearning. Retrieved December 14, 2021, from https://towardsmachinelearning.org/random-forest/
 - [6] Kumar, V. (2021, July 23). *K-Means.* TowardsMachineLearning. Retrieved December 11, 2021, from https://towardsmachinelearning.org/k-means/
 - [7] Li, H. (2020). *Which machine learning algorithm should I use?* Retrieved December 3, 2021, from https://blogs.sas.com/content/subconsciousmusings/2020/12/09/machine-learning-algorithm-use/
 - [8] *Machine Learning Algorithms.* (2021, August 13). Microsoft Azure. Retrieved December 14, 2021, from https://azure.microsoft.com/en-us/overview/machine-learning-algorithms/
 - [9] Saini, A. (2021, September 16). *Naive Bayes Algorithm: A Complete guide for Data Science Enthusiasts.* Analytics Vidhya. Retrieved December 14, 2021, from https://www.analyticsvidhya.com/blog/2021/09/naive-bayes-algorithm-a-complete-guide-for-data-science-enthusiasts/
 - [10] Saxena, S. (2019, October 15). *Mathematics Behind Machine Learning | Data Science.* Analytics Vidhya. Retrieved December 14, 2021, from https://www.analyticsvidhya.com/blog/2019/10/mathematics-behind-machine-learning/
 - [11] Sayad, S. (2012, November 8). *ecision Tree - Regression.* SaedSayad. Retrieved December 3, 2021, from http://www.saedsayad.com/decision_tree_reg.htm
 - [12] Stojiljković, M. (2019, April 15). *Linear Regression in Python* Real Python. Retrieved December 14, 2021, from https://realpython.com/linear-regression-in-python/
 - [13] Towards Data Science. (2019, March 11). *Which machine learning model to use?* Towards Data Science. Retrieved December 4, 2021, from https://towardsdatascience.com/which-machine-learning-model-to-use-db5fdf37f3dd


## Acknowledgements
This research would not be possible without:
- Ben Schepens for recommending further reading on conducting data analysis and visualizations
- Hieu Nguyen for grading this project with fidelity at the end of the school year
- Julia Rachkovskiy for giving me advice on who to approach for help with this project
