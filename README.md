# pands-project2021 

Research on Fisher's Iris data set 

The Iris flower data set or Fisher's Iris data set is a multivariate data set introduced by the British statistician, eugenicist, and biologist Ronald Fisher in his 1936 paper The use of multiple measurements in taxonomic problems as an example of linear discriminant analysis

Fisher developed and evaluated a linear function to differentiate Iris species based on the morphology of their flowers.

This discriminant function performed well in discriminating between these species, except some overlap between Iris versicolor and Iris virginica. 
The Iris setosa is noticeably different from the other two species.

References
https://en.wikipedia.org/wiki/Iris_flower_data_set

https://towardsdatascience.com/the-iris-dataset-a-little-bit-of-history-and-biology-fb4812f5a7b5

3 species of Iris where used in the dataset (50 samplues of each species was used)
    1.Iris setosa
    2.Iris Virginica
    3.Iris Veriscolor

There are 4 variables in the Iris dataset
    1.Length of sepals (cm)
    2.Width of sepals (cm)
    3.Length of petals (cm)
    4.Width of petals (cm)

These measures were used to create a linear discriminant model to classify the species.

next day - how to import iris dataset in python
https://www.geeksforgeeks.org/python-basics-of-pandas-using-iris-dataset/


This discriminant function performed well in discriminating between these species, except some overlap between Iris versicolor and Iris virginica. 
The Iris setosa is noticeably different from the other two species.

http://www.lac.inpe.br/~rafael.santos/Docs/CAP394/WholeStory-Iris.html

The dataset is often used in data mining, classification and clustering examples and to test algorithms.

Information about the original paper and usages of the dataset can be found in the UCI Machine Learning Repository -- Iris Data Set.

The Iris dataset consists of 50 samples each of three difference species of the iris flower: setosa, vericolor and virginica. 

Each row is an observation (also known as: sample, example, instance, record)
Each column is a feature (also known as: predictor, attribute, independent variable, input, regressor, covariate)
https://www.ritchieng.com/machine-learning-iris-dataset/

Adding image to Github read me file  
https://www.youtube.com/watch?v=hHbWF1Bvgf4
![](Iris_Image.png)

2.0 Loading and understanding the Iris dataset:

2.1 Loading the dataset  
The Iris dataset is widely available on the internet. The dataset is included in R base and Python in the machine learning package Scikit-learn, so that users can access it without having to find a source for it.  

However I am treating the iris dataset as dataset that needs to be loaded in so I use pandas to import the data from a csv file. 
I rename the columns so that they include the measurement type 'cm' in the title and I rename the class column to species. 

    path = ""
    filenameForIrisData = path + "iris.csv"
    df = pd.read_csv(filenameForIrisData)
    iris_df = df.rename(columns = {"sepallength" : "sepallengthcm", "sepalwidth" : "sepalwidthcm", "petallength" : "petallengthcm", "petalwidth" : "petalwidthcm", "class" : "species"})

2.2 Understanding the dataset   

To start with understanding the dataset, I preview a sample of the data, for example the first 5 rows
    print(iris_df.head(5)) 
            sepallengthcm  sepalwidthcm  petallengthcm  petalwidthcm      species
        0            5.1           3.5            1.4           0.2  Iris-setosa
        1            4.9           3.0            1.4           0.2  Iris-setosa
        2            4.7           3.2            1.3           0.2  Iris-setosa
        3            4.6           3.1            1.5           0.2  Iris-setosa
        4            5.0           3.6            1.4           0.2  Iris-setosa 

This shows the dataset has 5 columns and an unknown quantity of the rows.

I then build upon this to extract different views of the data.  

    - Show the full size (shape) of the dataset.  
        We have 150 samples for the 5 columns    
    
        print(iris_df.shape)  

        (150, 5) = 150 rows, 5 columns


    - Show the number of instances, the number of attributes and if any null values exist in the dataset.
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

    - Show how many instances the dataset contains by species
        The dataset has an equal number of 50 instances for each species.  
           
        print(iris_df.groupby('species').size())   

        Iris-setosa        50
        Iris-versicolor    50
        Iris-virginica     50

    - Shows the basic statistical details of the dataframe (iris dataset).  
        This shows the count, mean, std, min, 25%, 50%, 75%, max infor for each attribute/feature. 
        
        print(df.describe())

               sepallength  sepalwidth  petallength  petalwidth
        count   150.000000  150.000000   150.000000  150.000000
        mean      5.843333    3.054000     3.758667    1.198667
        std       0.828066    0.433594     1.764420    0.763161
        min       4.300000    2.000000     1.000000    0.100000
        25%       5.100000    2.800000     1.600000    0.300000
        50%       5.800000    3.000000     4.350000    1.300000
        75%       6.400000    3.300000     5.100000    1.800000
        max       7.900000    4.400000     6.900000    2.500000

# this is to output a summary of each variable to a single text file
# input has to be in string format
In order to extract this data into # class lectures showed how to export to a newly created txt file
with open(".\Variable_Summary.txt", "wt") as i:
    i.write(str(iris_df.groupby("species").describe()))


https://www.c-sharpcorner.com/article/a-first-machine-learning-project-in-python-with-iris-dataset/
https://www.c-sharpcorner.com/article/a-first-machine-learning-project-in-python-with-iris-dataset/
https://www.kaggle.com/adityabhat24/iris-data-analysis-and-machine-learning-python
https://towardsdatascience.com/how-to-use-groupby-and-aggregate-functions-in-pandas-for-quick-data-analysis-c19e7ea76367




![](Box_plot.png)