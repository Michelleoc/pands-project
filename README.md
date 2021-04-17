# pands-project2021 - Iris Dataset

# Author : Michelle O'Connor

# Contents

1.0 Introduction   
 - 1.1 History of Iris Dataset
 - 1.2 Use of Iris Dataset

2.0 Loading and understanding the Iris dataset  
 - 2.1 Loading the dataset   
 - 2.2 Understanding the dataset  

3.0 Data Visualisation   
 - 3.1 Histograms   
 - 3.2 Boxplots   
 - 3.3 Scatterplot  

4.0 


# 1.0 Introduction

## 1.1 History of Iris Dataset  

The Iris flower data set or Fisher's Iris data set is a multivariate data set introduced by the British statistician, eugenicist, and biologist Ronald Fisher in his 1936 paper __The use of multiple measurements in taxonomic problems__ as an example of linear discriminant analysis.  

The Iris dataset consists of the following:

50 samples of 3 different species of Iris where used in the dataset (total 150 samples)  
    1.Iris setosa   
    2.Iris Virginica  
    3.Iris Veriscolor  

There are 4 variables measured in the Iris dataset were   
    1.Length of sepals (cm)  
    2.Width of sepals (cm)   
    3.Length of petals (cm)  
    4.Width of petals (cm)   


## 1.2 Use of Iris Dataset 

Based on the combination of the features (Sepal Lenght, Sepal Width, Petal Length, Petal Width), Fisher developed a linear discriminant model to distinguish the species from each other based on the morphology of their flowers.  

This discriminant function performed well in discriminating between these species, except some overlap between Iris versicolor and Iris virginica. 
The Iris setosa is noticeably different from the other two species.

Using this linear discriminant model, this data set became a typical test case for many statistical classification techniques in machine learning and became the “hello world” of Machine Learning.

“Hello World” is often the first program written by people learning to code, the iris dataset is generally the first dataset used as an introduction into Machine Learning. 

References
These measures were used to create a linear discriminant model to classify the species.
The dataset is often used in data mining, classification and clustering examples and to test algorithms.


https://www.ritchieng.com/machine-learning-iris-dataset/
https://en.wikipedia.org/wiki/Iris_flower_data_set
https://towardsdatascience.com/the-iris-dataset-a-little-bit-of-history-and-biology-fb4812f5a7b5
http://www.lac.inpe.br/~rafael.santos/Docs/CAP394/WholeStory-Iris.html

Adding image to Github read me file  
https://www.youtube.com/watch?v=hHbWF1Bvgf4
![](Iris_Image.png)

# 2.0 Loading and understanding the Iris dataset:

## 2.1 Loading the dataset  
The Iris dataset is widely available on the internet. The dataset is included in R base and Python in the machine learning package Scikit-learn, so that users can access it without having to find a source for it.  

For this project, however I am treating the iris dataset as dataset that needs to be loaded in so I use pandas to import the data from a csv file and create a dataframe.    
I rename the columns so that they include the measurement type 'cm' in the title and I rename the class column to species. 

    path = ""
    filenameForIrisData = path + "iris.csv"
    df = pd.read_csv(filenameForIrisData)
    iris_df = df.rename(columns = {"sepallength" : "sepallengthcm", "sepalwidth" : "sepalwidthcm",   
    "petallength" : "petallengthcm", "petalwidth" : "petalwidthcm", "class" : "species"})

## 2.2 Understanding the dataset   

To start with understanding the dataset, I preview a sample of the data, for example the first 5 rows
    print(iris_df.head(5))   
            sepallengthcm  sepalwidthcm  petallengthcm  petalwidthcm      species  
        0            5.1           3.5            1.4           0.2  Iris-setosa  
        1            4.9           3.0            1.4           0.2  Iris-setosa   
        2            4.7           3.2            1.3           0.2  Iris-setosa   
        3            4.6           3.1            1.5           0.2  Iris-setosa   
        4            5.0           3.6            1.4           0.2  Iris-setosa    

This shows the dataset has 5 columns and an unknown quantity of the rows.

I then build upon this to extract different views of the data:  

* Show the full size (shape) of the dataset.  
        We have 150 samples for the 5 columns    
    
        print(iris_df.shape)  

        (150, 5) = 150 rows, 5 columns


* Show the number of instances, the number of attributes and if any null values exist in the dataset.  
        We have 150 instances for 5 attributes of which no null values exist.    
    
        print(iris_df.info())

        #   Column         Non-Null Count  Dtype
        ---  ------         --------------  -----
        0   sepallengthcm  150 non-null    float64
        1   sepalwidthcm   150 non-null    float64
        2   petallengthcm  150 non-null    float64
        3   petalwidthcm   150 non-null    float64
        4   species        150 non-null    object
        dtypes: float64(4), object(1)

* Show how many instances the dataset contains by species.  
        The dataset has an equal number of 50 instances for each species.  
           
        print(iris_df.groupby('species').size())   

        Iris-setosa        50
        Iris-versicolor    50
        Iris-virginica     50

* Show the basic statistical details of the dataframe (iris dataset).  
        This shows the count, mean, std, min, 25%, 50%, 75%, max infor for each attribute/feature. 
        
        print(iris_df.describe())

               sepallength  sepalwidth  petallength  petalwidth
        count   150.000000  150.000000   150.000000  150.000000
        mean      5.843333    3.054000     3.758667    1.198667
        std       0.828066    0.433594     1.764420    0.763161
        min       4.300000    2.000000     1.000000    0.100000
        25%       5.100000    2.800000     1.600000    0.300000
        50%       5.800000    3.000000     4.350000    1.300000
        75%       6.400000    3.300000     5.100000    1.800000
        max       7.900000    4.400000     6.900000    2.500000


* Show the summary of each variable by the species  
        print(iris_df.groupby("species").describe())

        To extract this data into a newly created single text file, we need to make the output in a   
        string format using str. 

            with open(".\Variable_Summary.txt", "wt") as i:
            i.write(str(iris_df.groupby("species").describe()))


https://www.c-sharpcorner.com/article/a-first-machine-learning-project-in-python-with-iris-dataset/
https://www.c-sharpcorner.com/article/a-first-machine-learning-project-in-python-with-iris-dataset/
https://www.kaggle.com/adityabhat24/iris-data-analysis-and-machine-learning-python
https://towardsdatascience.com/how-to-use-groupby-and-aggregate-functions-in-pandas-for-quick-data-analysis-c19e7ea76367

# 3.0 Data Visualisation 

With a basic understanding of the data, we move to data visualisation to help us compare and observe trends within the data.

There are many visualation options within python using matplotlib and seaborn. 

## 3.1 Historgrams  

Histograms show the distribution of the number of instances by each individual attribute. 

    Code to display a histogram for the sepal length:
    plt.figure(figsize = (10, 7)) = set the size (in inches) of the histogram
    x = iris_df["sepallengthcm"] = defining the data input
    plt.hist(x, bins = 20, color = "green") = plotting the histogram (x=input data, bin = number of columns, colour of bin/column = green)
    plt.title("Sepal Length in cm") = adding a title
    plt.xlabel("Sepal Length cm") = adding a name to the xlabel (horizontal line)
    plt.ylabel("Count") = adding a name to the ylabel (vertical line)
    plt.savefig("Sepal_Length.png") = saving the histogram as an image to the folder
    plt.show() = displaying the histogram


## 3.2 Boxplot  

Boxplot show the range the individual attributes fall into, it represents the minimum, first quartile (25% or lower), median, third quartile (75% or upper) and maximum of the dataset.  
The box on the plot shows between the first (25%) and third quartile (75%) on the range. The horizontal line that goes through the box is the median.  
The top and bottom lines known as 'whiskers' extend from the ends of the box to the minimun and maximum value.    
Any data points past the whiskers (represent by a diamond) are considered as outliers.  

Using Seaborn we plot a Boxplot for each attribute (4 Boxplots in total). 
To show 4 Boxplots on the one output requires a 2 x 2 (2 columns and 2 rows), therefore subplot(2,2) is required.  
The 3rd value in the subplot indicates where on the output the plot is shown, as follows 1 - Top Left, 2 - Top Right, 3 - Bottom Left, 4 - Bottom Right, 
for example (2,2,3) would show on the bottom left of the output.  
On the plot the x axis is the species type, y axis is the attribute  

    plt.figure(figsize=(15,10)) = set the size of the boxplot 
    plt.subplot(2,2,1)    
    sns.boxplot(x='species',y='sepallengthcm',data=iris_df)
    plt.subplot(2,2,2)    
    sns.boxplot(x='species',y='sepalwidthcm',data=iris_df)     
    plt.subplot(2,2,3)    
    sns.boxplot(x='species',y='petallengthcm',data=iris_df)     
    plt.subplot(2,2,4)    
    sns.boxplot(x='species',y='petalwidthcm',data=iris_df)
    plt.savefig("Box_plot.png")
    plt.show()


Analyising the box plots, the sepal length and sepal width data results are close, with some overlap between all 3 species.   
However for the petal length and petal width, the Iris Setosa is visually clearly different from the Veriscolor and Virginica.   
The Iris Setosa petal lenght and petal width attributes data do not overlap with the Veriscolor and Virginica.  

![](Box_plot.png)  


## 3.3 Scatterplot (Pairsplot)  

Scatter plot is very useful when we are analyzing the relationship between 2 features on x and y axis.
In seaborn library there is a pairplot function which is very useful to scatter plot all the features at once instead of plotting them individually.  

The pair plot used to figure out a distribution of single variables and the relationship between two variables.  
If the pair plot is given a solution for that as a clear understanding of each flower sets at a single graph.  
Each flower scatters plots represented in different colors.  

For each pair of attributes, we can use a scatter plot to visualize their joint distribution  
sns.pairplot(iris_df, hue="species", diag_kind="kde")  
plt.show()  



http://www.cse.msu.edu/~ptan/dmbook/tutorials/tutorial3/tutorial3.html
https://www.c-sharpcorner.com/article/a-first-machine-learning-project-in-python-with-iris-dataset/  

# 4.0 Train and Validate the data (Machine learning)  

The Iris dataset can be used by a machine learning model to illustrate classification (a method used to determine the type of an object by comparison with similar objects that have previously been categorised).   
Once trained on known data, the machine learning model can make a predictive classification by comparing a test object to the output of its training data.

I separate the dataset into two parts for validation processes such as train data and test data.  
Then allocating 75% of data for training tasks and the remainder 25% for validation/testing purposes.  


1.Train model - trains a simple multi-class logistic regression model
2.Predict - Makes class predictions given a pre trained model and a test set
3.Report accuracy - Reports the accuracy of the predictions preformed by the previous node



https://kedro.readthedocs.io/en/stable/02_get_started/05_example_project.html
