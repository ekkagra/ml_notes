### Normal Distribution
* Any measurement of a property for a sample population when plotted as a histogram or distribution follows a bell shaped symmtrical curve. The curve is characterized by a mean and standard deviation.
    * Mean - The average of all points
    * Standard deviation - The spread of data points about the mean
* Where is normal distribution useful ?
    * Use in comparing apples to oranges. Different commodities can be judged on a scale which normalizes both measurements for both the commodities.

### Statistical Learning
* Learning the coefficients for a function which is used to estimate the relation between predictors and output values.
    * Input variables, predictors, independant variable
    * Output varialbes, response, dependant variable
    * Y = f(x) + e
        * Y is response
        * f(x) is the function which takes in input
        * e is inherent error
* Why to estimate f?
    * Prediction 
        * Predict output for a new set of input variables
    * Inference
        * Infer what input predictors influence output, by how much etc.

### Methods ot Statistical Learning
Two types of methods
* Parameteric methods
    * Makes an assumption of the functional form of the relation to estimate.
    * Problem reduces to estimate the parameter coefficients by minimizing a cost fucntion that takes in the function estimate.
* Non-Parametric methods
    * Does not make an assumption about the functional form of relation between input
    and output.
    * Generally a large of records needed to get good estimate of relation between input and output

### Measuring the quality of fit
* MSE - One measure is the mean squared error between actual values and predicted values
* Training MSE vs Test MSE
    * As we increase the flexibility of model, the training MSE generally decreases.
    * As we increase the flexibility of model, the test MSE will initially be high(***Under Fitting***) and will begin to decrease but after a certain point, the test MSE will increase.(***Overfitting***)
    * We select that model for which the test MSE is least.

### Variance-Bias trade-off
* Variance of a statistical learning method
    * This gives how much the function estimate will change, if it had been learned using a different training data set. Models with higher flexibility/complexity usually have higher variance because they follow the output very closely. If data set changes by small amount, they will give output which changes to a large extent.
* Bias
    * Refers to the error that is introduced when approximating complex real-life problem using a much simpler model. More flexible models result in lesser bias.
* The model selected should eventually have a low variance and low bias. This model will have a low test MSE.




