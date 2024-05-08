import numpy as np
from helper_functions import *

def get_initial_means(array, k):
    """
    Picks k random points from the 2D array
    (without replacement) to use as initial
    cluster means

    params:
    array = numpy.ndarray[numpy.ndarray[float]] - m x n | datapoints x features

    k = int

    returns:
    initial_means = numpy.ndarray[numpy.ndarray[float]]
    """


    # the description says to pick K random points from the array without replacement
    # we just need to do random.choice using numpy

    random_k_points = np.random.choice(array.shape[0], k, replace=False)
    # print(random_k_points)

    # returning the initial_means = numpy.ndarray[numpy.ndarray[float]]
    # we need to pick k random points from the 2D array as instructed above

    initial_means = array[random_k_points]
    # print(initial_means)

    return initial_means


def k_means_step(X, k, means):
    """
    A single update/step of the K-means algorithm
    Based on a input X and current mean estimate,
    predict clusters for each of the pixels and
    calculate new means.
    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n | pixels x features (already flattened)
    k = int
    means = numpy.ndarray[numpy.ndarray[float]] - k x n

    returns:
    (new_means, clusters)
    new_means = numpy.ndarray[numpy.ndarray[float]] - k x n
    clusters = numpy.ndarray[int] - m sized vector
    """


    # I used the following as as reference for this part:
    # https://blog.gopenai.com/introduction-to-kmeans-clustering-algorithm-57bd947b8678
    # I took inspiration from medium page, it's very similar to what's expected in this question
    # https://hackernoon.com/speeding-up-your-code-2-vectorizing-the-loops-with-numpy-e380e939bed3
    # From the following: https://en.wikipedia.org/wiki/Euclidean_distance, I got the formula for Euclidean distance
    # it's d(p,q) = sqrt (p1 - q1) ^ 2 + (p2 - q2)^2
    # to turn this into code, we can first get (p1 - q1)^2 as separate variable, below I named that as "two_points"
    # then we can find distance using that, can get sum of the value in two_points, square root result of that

    # here my p1 is X and q1 is means
    # means need to have an increase in dimension to make it compaitable with broadcasting with X
    # I got this error if I did X - means, without changing dimensions locally:
    # ValueError: operands could not be broadcast together with shapes (116966,3) (2,3)
    # To solve that I used the following:
    # https://stackoverflow.com/questions/27948363/numpy-broadcast-to-perform-euclidean-distance-vectorized
    # In here I find out I need to use np.newaxis to add new dimension, we have X (m,n) and means (k,n)
    # we need the result of subtraction to be (k,m,n), that's why I am getting an error also that's what It's mentioned in
    # the resource I listed above. then we just square that like we did above with (p - q) ^2
    two_points = ((X - means[:, np.newaxis]) **2)
    #print(two_points)

    # once we have the above (p-q)^2, I have to take sum of square differences and take square root of it
    distance_x_means = np.sqrt(two_points.sum(axis=2))
    #print(distance_x_means)

    # as the function signature lists above, need to return clusters
    # to find the clusters we need to get the least squared Euclidean distance, got this from:
    # https://en.wikipedia.org/wiki/K-means_clustering
    # so to do this, I can find the index of row with min sum in the Euclidean distance along the axis 0
    # essentially we are finding for each data point in X the index of the closet mean, the index is the cluster here
    clusters = np.argmin(distance_x_means, axis=0)
    #print(clusters)

    # as instructed above, we need to find the new means.
    # first I am going to initialize an new list called new_means, this is the expected return I will type cast to array after
    new_means = []

    # I am going to loop over the number of "K"s, the clusters basically.
    # Reason being we need to select all data points from X for each cluster
    for i in range(k):
        # here I am getting an output when all data points in X are at the current cluster "i"
        # remember each iteration in this loop is a different cluster from K number of clusters
        cluster_x_data_points = X[clusters == i]
        #print(cluster_x_data_points)

        # once I have the all data points from X that have been assigned to the current cluster "i"
        # I can calculate the mean of those points, being stored in cluster_x_data_points
        # to calculate the mean, I used np.mean for this along the axis 0
        cluster_mean = np.mean(cluster_x_data_points, axis=0)
        #print(cluster_mean)

        # once I have the current cluster's mean I can append to our list of new_means, we initialized before this loop
        new_means.append(cluster_mean)

    #print(new_means)

    # as I promised, need to type cast the new_means list to an array as that's what expected for this function
    new_means = np.array(new_means)
    #print(new_means)

    # returning like this: (new_means, clusters), as expected.
    return new_means, clusters


def k_means_segment(image_values, k=3, initial_means=None):
    """
    Separate the provided RGB values into
    k separate clusters using the k-means algorithm,
    then return an updated version of the image
    with the original values replaced with
    the corresponding cluster values.

    params:
    image_values = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - r x c x ch
    k = int
    initial_means = numpy.ndarray[numpy.ndarray[float]] or None

    returns:
    updated_image_values = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - r x c x ch
    """


    # each image_values input has an row, column and channel as shown in signature, r x c x ch
    # let me get the values for those first, will need those later on
    # r represents row, c represents column and ch represents channel

    r, c, ch = image_values.shape
    # print(r, c, ch)

    # since this image_values is not an 2d image, it's going to be hard to use the previous get_initial_means() and
    # k_means_step() functions so I will reshape the image into an 2D array so I can work with the values easily
    # and also use k_means_step() function and the get_initial_means().
    # https://numpy.org/doc/stable/reference/generated/numpy.reshape.html
    # np.reshape changes the shape of an array without changing it's data, which is exactly needed here
    # so my 2d array or the new image will have the row as an pixel, which is row * col and it's column as channels.
    # so it will be [(r*c), ch]

    # flattened_image_2d = image_values.reshape((r*c), ch)
    # print(image_values)
    # print(flattened_image_2d)

    # *****EDIT: I just realized, once I got part 2, when doing train model that there are helper_functions
    # so I am going to use the flatten_image_matrix function instead of doing it myself
    flattened_image_2d = flatten_image_matrix(image_values)

    # When no initial cluster means (`initial_means`) are provided,
    # you need to initialize them yourself, based on the given k.
    # let's use the method get_initial_means(), I finished before this to get the initial means, when it's not given
    # I am going to pass in the new 2d flattened image and the K provided as input. I am using flattened image because
    # the get_initial_means() function only takes an 2d array as input, so can't use image_values as the params

    if initial_means is None:
        initial_means = get_initial_means(flattened_image_2d, k)

    # going to initialize the means variable which we need, similar to the k_means_step() function
    # the value of that will be the initial_means from either the input or what I just set above

    means = initial_means
    # print(means)

    # also need to initialize clusters, just like the k_means_step() function, it's going to be empty for now

    clusters = None

    # here is where I will do the K-means clustering process, it's different than k_means_step() function
    # as I have to run the loop or iterate until the cluster stop changing, essentially when it reaches convergence
    # so it will be infinite loop until convergence has been reached and there is no change in clusters

    while True:
        # will need to now call the k_means_step() function I completed before
        # remember the output of that is, new_means and clusters, here I will call clusters new_clusters
        # since I already initialized clusters above. Also the k_means_step() takes in X, k, means
        # here X will be the 2d flattened array, k will be k and means is the means we set above with initial means

        new_means, new_clusters = k_means_step(flattened_image_2d, k, means)
        # print(new_means, new_clusters)

        # now I have to check for convergence and break the loop if it reaches convergences
        # how to check is to see if the new_clusters equal the clusters set later in this function
        # obviously it won't break on the first iteration itself, but eventually goal is to check to see if
        # there is no difference in clusters meaning we are good, there is convergence
        # to check the comparison between clusters and new_clusters, I need to use np.array_equal
        # https://numpy.org/doc/stable/reference/generated/numpy.array_equal.html
        # It's true if two arrays have the same shape and elements
        # Your convergence test should be whether the assigned clusters stop changing.
        # which is exactly what I am doing here, checking if clusters and new_clusters is same, then break loop.

        if np.array_equal(clusters, new_clusters):
            break
            # print(test)
            # print(clusters, new_clusters)

        # if the above if statement didn't get met in condition, that means the loop keeps going on
        # so I have to update the clusters and means for the next iteration, with the new ones found above

        means = new_means
        clusters = new_clusters

    # now I have the clusters and means values after convergence I have to update the image
    # it said in the method signature to: "updated version of the image with the original values replaced with
    # the corresponding cluster values." meaning I have to get the updated version of the image by getting the
    # value of means at each cluster, similar to the k_means_step() function

    # I am going to initialize the updated_image list here first, will type cast to array after similar to k_means_step()
    updated_image = []

    # iterate over each cluster in the clusters array
    for cluster in clusters:
        # I need to get the cluster_mean by finding the value of means at index cluster of the iteration
        cluster_mean = means[cluster]
        # print(cluster_mean)

        # then after getting the value of means at current cluster, I append to the list I created at top
        updated_image.append(cluster_mean)

    # need to convert the list back to an array, as that's what's required.
    updated_image = np.array(updated_image)
    # print(updated_image)

    # remember the updated image is still an 2d array, but updated version of the image
    # with original values replaced with corresponding cluster values. We already have the cluster values
    # just need to replace the updated_image with the original values, that's why I store row, column and channel values
    # of the given image_values input right at start, for this purpose only.
    # to get the update_image_values with original values, I just have to pass in r, c, ch to updated_image.
    updated_image_values = updated_image.reshape((r, c, ch))
    # print(updated_image_values)

    # returning what's expected, which is r x c x ch
    return updated_image_values

def compute_sigma(X, MU):
    """
    Calculate covariance matrix, based in given X and MU values

    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    MU = numpy.ndarray[numpy.ndarray[float]] - k x n

    returns:
    SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    """

    # I used the following video as reference: https://www.youtube.com/watch?v=qMTuMa86NzU&ab_channel=AlexanderIhler
    # equation in the video: p(x) = sum of (Pi multiplied by x; mu, sigma). Sigma is the variance
    # from my understanding X is the data points like the K means algo and MU represents the means of gaussian mixture model

    # unpacking the shapes of input MU and X, the respective mean and data points
    # From the description above, we know X is m x n and MU is k x n
    # remember from K steps function,     X = m x n | pixels x features (already flattened)
    # so let's unpack same way, m is the data points or the pixels for X and k is number of gaussian components for MU
    # and n like X, is the same it's number of features. I will unpack that from the MU, since it's of gaussian MM
    m, _ = X.shape
    k, n = MU.shape

    # I am going to initialize the return variable here, from method description, SIGMA = k x n x n
    # going to set up the function the same way. This will store our covariance matrix
    SIGMA = np.zeros((k,n,n))

    # I used the following equation for covariance: F(x) = 1/m * ( Sum of ((x − µ)^T(x − µ)) all at index i )
    # https://www.cs.princeton.edu/courses/archive/spr07/cos424/scribe_notes/0419.pdf

    # iterating over k number of gaussian components from input MU
    for i in range(k):
        # for each gaussian component, going to find difference between each data point in X and the mean of that component
        # this is the (x − µ), of the equation above
        difference = X - MU[i]
        # print(difference)

        # need to loop over each each data point in X, which is m, we unpacked above
        for j in range(m):
            # we need to reshape the difference into column vector, remember above we get m x n array
            # in order for me to compute the multiplication of matrices from the equation above, need to change the shape
            # if I didn't reshape the difference vector, I kept getting an error, realized it's due to shape being off
            difference_reshaped = difference[j, :].reshape(n,1)

            # this is where I compute the SIGMA, where the equation above comes together. The ^T in the equation means transpose
            # essentially using dot product because it saves me step of multiplying and summing in python
            # dot product is very handy, it's basically,  A dot B is Ax*Bx + Ay*By
            # here in the equation we have to multiply difference and the transposed difference then add them at ith element
            SIGMA[i] += np.dot(difference_reshaped, difference_reshaped.T)

        # after summation and multiplication of the differences for each data point,
        # the sum is divided by m, just like the equation.
        SIGMA[i] /= m
        # print(SIGMA)

    # print(SIGMA)
    return SIGMA

def initialize_parameters(X, k):
    """
    Return initial values for training of the GMM
    Set component mean to a random
    pixel's value (without replacement),
    based on the mean calculate covariance matrices,
    and set each component mixing coefficient (PIs)
    to a uniform values
    (e.g. 4 components -> [0.25,0.25,0.25,0.25]).

    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    k = int

    returns:
    (MU, SIGMA, PI)
    MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    PI = numpy.ndarray[float] - k
    """

    # unpacking the given input X, into m and n. it's provided in description it's m x n
    m, n = X.shape

    # it said in the description of function to "Set component mean to a random pixel's value (without replacement)"
    # so let's do that, this is very similar to K_initial_means from part 1
    # there It asked to "Picks k random points from the 2D array (without replacement) to use as initial cluster means"
    # I did random_k_points = np.random.choice(array.shape[0], k, replace=False), going to replace array.shape[0] with m here
    # because remember is the m pixels and n is the features
    random_pixels = np.random.choice(m, k, replace=False)

    # next we are going to set component mean, which is MU, refer to compute_sigma function of why it is.
    # They want me to set MU to the random_pixels, going to do that. get the value of X at random_pixels
    MU = X[random_pixels]

    # next they said " based on the mean calculate covariance matrices", well I am going to call my compute_sigma function
    # set X input to X parms and MU to MU params
    SIGMA = compute_sigma(X,MU)

    # lastly from the expected return, we want the PI, which is the weight or the mixing coefficients.
    # from this reference: https://towardsdatascience.com/gaussian-mixture-model-clearly-explained-115010f7d4cf
    # It's said that weight at start of 3 GMM components, is set to 1/3. It depends on number of GMM components
    # what is number of GMM components, well it's k. so we will set it to 1/k all to start of. Also given shape of k,
    # each value in the shape will be 1/k. Just like example they used in reference, with 3 components all being 1/3, 1/3, 1/3.
    # The weight of each component is 1/3, here weight of each component is 1/k, given the size K.
    # I am going to use: https://numpy.org/doc/stable/reference/generated/numpy.full.html, it does exactly what i mentioned above
    PI = np.full(k, 1/k)

    # print(MU, SIGMA, PI)

    # return what's expected, which was  (MU, SIGMA, PI)
    return MU, SIGMA, PI


def prob(x, mu, sigma):
    """Calculate the probability of x (a single
    data point or an array of data points) under the
    component with the given mean and covariance.
    The function is intended to compute multivariate
    normal distribution, which is given by N(x;MU,SIGMA).

    params:
    x = numpy.ndarray[float] (for single datapoint)
        or numpy.ndarray[numpy.ndarray[float]] (for array of datapoints)
    mu = numpy.ndarray[float]
    sigma = numpy.ndarray[numpy.ndarray[float]]

    returns:
    probability = float (for single datapoint)
                or numpy.ndarray[float] (for array of datapoints)
    """


    # I used this video as reference again: https://www.youtube.com/watch?v=qMTuMa86NzU&ab_channel=AlexanderIhler
    # the video helped me alot and understand how to do this probability function
    # The equation showed at 4:34 mark, or where person talks about the probability function
    # equation is (1 / 2pi^n/2) times exponent of -1/2* (x-mu)^ T times sum of (X-Mu)
    # that's alot to do in one line, we have to do multiple lines and break down few components given
    # I also reference this again: https://towardsdatascience.com/gaussian-mixture-model-clearly-explained-115010f7d4cf
    # Here the simplified what's given in the YT video, it was easier for me to understand how to break this down

    # first I am going to unpack the n value from MU
    n = mu.shape[0]

    # there is two things we need, determinant of the covariance matrix and inverse of covariance matrix
    # the determinant is needed for the (1 / 2pi^n/2) part of the equation it's used for probability density calculation
    # the inverse is used for normal calculation, in the method signature
    # to find determinant and inverse of an matrix, I am using: https://numpy.org/doc/stable/reference/routines.linalg.html
    # this library in numpy, has both of the functions for me for determinant and inverse of an matrix
    sigma_determinant = np.linalg.det(sigma)
    sigma_inverse = np.linalg.inv(sigma)
    # print(sigma_determinant, sigma_inverse)

    # this is where we do the (1 / 2pi^n/2) part of the equation, I use determinant as I mentioned above
    normalize_factor = (1.0/(np.power((2*np.pi), n/2) * np.sqrt(sigma_determinant)))

    # this is difference in the equation between X and MU, it's pretty redundant explanation, so I won't go in depth
    difference = x - mu
    # print(difference)

    # in the below conditions, I am going to check whether the array dimensions is 1 or multiple
    # to find if it's 1, we can use: https://numpy.org/doc/stable/reference/generated/numpy.ndarray.ndim.html
    # if it's one we need to get the exponent part of the equation, that changes for 1 vs multiple dimensions.
    # for 1 dimension, just have to first take the dot product of sigma inverse and difference, then the dot product of
    # the transposed difference matrix and resultant of previous dot product from sigma_inverse and difference.
    if x.ndim == 1:
        exponent = np.dot(difference.T, np.dot(sigma_inverse, difference))
    # if it's multiple dimension, we compute the exponent for each data point and then do a whole sum across the feature
    # dimension which is Axis =1.
    else:
        exponent = np.sum(np.dot(difference, sigma_inverse) * difference, axis=1)

    # print(exponent)
    # we finish of the equation by taking the normalize_factor multiplying with the exponent of -1/2 times exponent variable
    return normalize_factor * np.exp(-0.5*exponent)


def E_step(X,MU,SIGMA,PI,k):
    """
    E-step - Expectation
    Calculate responsibility for each
    of the data points, for the given
    MU, SIGMA and PI.

    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    PI = numpy.ndarray[float] - k
    k = int

    returns:
    responsibility = numpy.ndarray[numpy.ndarray[float]] - k x m
    """


    # for E-step I used the video given again, I referenced it multiple times in above functions

    # going to first get the m, which is the number of data points
    m = X.shape[0]

    # let's initialize responsibility to represent the responsibility of each Gaussian for each data point
    responsibility = np.zeros((k, m))

    # going to loop over each Gaussian component, k.
    for i in range(k):
        # from the video, responsibility is the probability of our data, mu and sigma multiplied by the pi
        # it was very straight forward equation in the video, at 5:34. Since we have all the components
        # we have the probability from the prob() function, we just have to plug everything in
        responsibility[i] = PI[i] * prob(X, MU[i], SIGMA[i])

    # The next part of the equation is to normalize the sum to one, as mentioned in the video.
    # to do that I can take sum of the responsibility column wise.
    responsibilities_total = np.sum(responsibility, axis=0)
    # print(responsibilities_total)

    # once we have the total sum, need to divide the responsibility by the total responsibility, return that.
    # print(responsibility)
    return  responsibility / responsibilities_total

def M_step(X, r, k):
    """
    M-step - Maximization
    Calculate new MU, SIGMA and PI matrices
    based on the given responsibilities.

    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    r = numpy.ndarray[numpy.ndarray[float]] - k x m
    k = int

    returns:
    (new_MU, new_SIGMA, new_PI)
    new_MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    new_SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    new_PI = numpy.ndarray[float] - k
    """


    # I was told to start with assignment probabilities, so I will do that first
    # first I will unpack the m and n from the input X.
    m, n = X.shape

    # They said to update parameters, i.e., mean, covariance and weights/size (PI)
    # well that's how you do M-step, but if we know we have to update MU, Sigma and PI
    # let's just initialize all of them right now. I mean I was also told that my retuns have to be:
    # new_MU, new_SIGMA and new_PI, so that's a major hint, I was also given the how to build shape of each
    # new_MU = kxn, new_SIGMA = k x n xn and new_PI is just k. so I will do that below:
    new_MU = np.zeros((k, n))
    new_SIGMA = np.zeros((k, n, n))
    new_PI = np.zeros(k)

    # going to iterate over each gaussian component using the values for number of gaussian components, K
    for i in range(k):

        # In the video it says to calculate the total responsibility, we need this for the new MU and new SIGMA
        # using slicing to calculate total responsibility of "i"th element across all data points
        total_r = np.sum(r[i, :])

        # so first let me update the new MU, in the video, the equation was to multiply responsibility by X
        # divide by the total_r, do summation after. I am going to use dot product as it's easier.
        # This page: https://towardsdatascience.com/gaussian-mixture-model-clearly-explained-115010f7d4cf
        # I referenced this before, simplified the MU formula more nicely than the Youtube video, but it's same
        new_MU[i] = np.dot(r[i, :], X) / total_r

        # next let me do the new pi, this is very easy, just have to take the total divide by m at the index i
        new_PI[i] = total_r / m

        # lastly going to do the SIGMA, this is a bit more difficult, have to find the difference again
        # the webpage I stated above has a good equation simplification of this
        # I have to multiply the responsibility at "i"th element by the transposed difference, then do dot product
        # with that result and the actual difference, divide whole thing by the total R
        difference = X - new_MU[i]
        new_SIGMA[i] = np.dot(r[i, :] * difference.T, difference) / total_r

    #print(new_MU, new_SIGMA, new_PI)
    # return what's expected, which is (new_MU, new_SIGMA, new_PI)
    return new_MU, new_SIGMA, new_PI

def likelihood(X, PI, MU, SIGMA, k):
    """Calculate a log likelihood of the
    trained model based on the following
    formula for posterior probability:

    log(Pr(X | mixing, mean, stdev)) = sum((i=1 to m), log(sum((j=1 to k),
                                      mixing_j * N(x_i | mean_j,stdev_j))))

    Make sure you are using natural log, instead of log base 2 or base 10.

    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    PI = numpy.ndarray[float] - k
    k = int

    returns:
    log_likelihood = float
    """


    # again referenced the video, it has the log likelihood equation at 8:44.
    # need to do summation of log of another summation of pi times the probability of X, MU and Sigma
    # I guess it's same as what's given in method signature: sum((i=1 to m), log(sum((j=1 to k),
    # mixing_j * N(x_i | mean_j,stdev_j))))

    # first I am going to unpack the m value as always to get the number of data points in X
    m = X.shape[0]

    # going to initialize the matrix likelihood, this will store the part where we do:
    # pi times the probability of X, MU and Sigma.
    likelihood = np.zeros((m, k))

    # loop over each gaussian component as usual like previous functions
    for i in range(k):
        # going to actually compute the likelihood now, the probability part
        # pi times the probability of X, MU and Sigma is the part we have to do here for index i
        # we have probability from the prob() function just need to plug in X, MU and SIGMA given to us also PI is given
        likelihood[:, i] = PI[i] * prob(X, MU[i], SIGMA[i])

    # print(likelihood)

    # Once the likelihood is computed need to summation of first then take natural log of it then another sum
    # https://numpy.org/doc/stable/reference/generated/numpy.log.html, this takes natural log
    log_likelihood = np.sum(np.log(np.sum(likelihood, axis=1)))

    #print(log_likelihood)
    # return the expected return variable of log_likelihood
    return log_likelihood

def train_model(X, k, convergence_function, initial_values = None):
    """
    Train the mixture model using the
    expectation-maximization algorithm.
    E.g., iterate E and M steps from
    above until convergence.
    If the initial_values are None, initialize them.
    Else it's a tuple of the format (MU, SIGMA, PI).
    Convergence is reached when convergence_function
    returns terminate as True,
    see default convergence_function example
    in `helper_functions.py`

    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    k = int
    convergence_function = func
    initial_values = None or (MU, SIGMA, PI)

    returns:
    (new_MU, new_SIGMA, new_PI, responsibility)
    new_MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    new_SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    new_PI = numpy.ndarray[float] - k
    responsibility = numpy.ndarray[numpy.ndarray[float]] - k x m
    """


    # again referencing the video, but this time is different at 10:29 as it talks about convergence.

    # I am going to first start of with what's told in the function description:
    # If the initial_values are None, initialize them.
    # okay so I am going to call the function initialize_parameters(), which returns the MU, SIGMA, and PI
    # we want the method to return (new_MU, new_SIGMA, new_PI, responsibility), not going to set it to that as we will find
    # the new MU, new SIGMA and new PI later, it's too confusing naming it new right of the bat.
    if initial_values is None:
        MU, SIGMA, PI = initialize_parameters(X, k)

    # it said Else it's a tuple of the format (MU, SIGMA, PI). so we are given initial_values,
    # unpack the MU, SIGMA and PI from he initial_values given.
    else:
        MU, SIGMA, PI = initial_values

    # from the helpe_function.py, I took look at default_convergence, it looks like we need to use likelihood
    # so I am going to go ahead and initialize the variable log_likelihood.
    log_likelihood = 0

    # next we need an variable to check, when "Convergence is reached when convergence_function returns terminate as True"
    # so going to set a variable similar to default_convergence function calling converged to False to begin
    converged = False

    # We need a counter for the convergence function again same as default_convergence function, with conv_ctr
    conv_ctr = 0

    # finally need to initialize responsibility as expected by the method signature in the return
    responsibility = 0

    # Just like the k_means_segment() function, need to run the while loop until it's converged. So as long as
    # converged is false, this loop won't break, that's why I intialized that variable to be false.
    while not converged:

        # I am just going to strictly follow what's said in the method signature:
        # "iterate E and M steps from above until convergence."

        # so let's get the E step by calling the E_Step function, the result is responsibility for Estep
        # we already have the required inputs for that function, the X, K and the MU, SIGMA and PI we initaliZed at top
        responsibility = E_step(X, MU, SIGMA, PI, k)

        # Next is the Mstep and it's same thing call the M_step() function, it returns new_MU, new_SIGMA, and new_PI
        # we have the X, responsibility and k already, the responsibility is from e_step. just plug in the values
        new_MU, new_SIGMA, new_PI = M_step(X, responsibility, k)

        # so now I go back to the default_convergence function, we know there that input values for the
        # convergence_function, which is the input of this train_model function, takes in the:
        # prev_likelihood = float, new_likelihood = float, conv_ctr = int
        # that's why I already initialized the log_likelihood outside this loop, we need to get new log_likelihood
        # to that we can call our likelihood() function, we have all the required inputs. X, K and going to use the
        # newPI, newMU and NewSIGMA for the likelihood.
        new_log_likelihood = likelihood(X, new_PI, new_MU, new_SIGMA, k)

        # now we can get the convergence function, with the log_likelihood, newlog_likelihood and the conv_ctr, which is
        # initialized at the top. This function outputs whether or not the function is converged.
        # we need this converged value to eventually update to True, so this loop breaks
        # this was stated in the method description: " Convergence is reached when convergence_function returns terminate as True"
        # EDIT: my order of unpacking the convergence_function was wrong it's not converged, conv_ctr
        # it was silly mistake, I got the question wrong so updated it.
        conv_ctr, converged = convergence_function(log_likelihood, new_log_likelihood, conv_ctr)

        # Just like the k_means_segment() function have to update the variables with new ones
        # so for MU, SIGMA, PI, need to update with their newMU, new SIGMA and New PI.
        # new updated the current log_likelihood with the new one, in the next iteration the new one will be calculated
        # same with the mu, pi and sigma.
        MU, SIGMA, PI = new_MU, new_SIGMA, new_PI
        log_likelihood = new_log_likelihood

    #print(log_likelihood)
    #print(MU,SIGMA,PI, responsibility)
    # return the expected (new_MU, new_SIGMA, new_PI, responsibility)
    return MU, SIGMA, PI, responsibility


def cluster(r):
    """
    Based on a given responsibilities matrix
    return an array of cluster indices.
    Assign each datapoint to a cluster based,
    on component with a max-likelihood
    (maximum responsibility value).

    params:
    r = numpy.ndarray[numpy.ndarray[float]] - k x m - responsibility matrix

    return:
    clusters = numpy.ndarray[int] - m x 1
    """
    # TODO: finish this͏󠄂͏️͏󠄌͏󠄎͏︈͏͏󠄁
    # raise NotImplementedError()

    # I used this as reference: https://en.wikipedia.org/wiki/Maximum_likelihood_estimation
    # we just need to get argmax of responsibility matrix at axis =0, since we need to find index of highest value
    # in each column of the responsibility matrix. since r is kxm and clusters need to be mx1.
    # using the expected return from method description, of name clusters.

    clusters = np.argmax(r, axis=0)

    # print(clusters)
    return clusters


def segment(X, MU, k, r):
    """
    Segment the X matrix into k components.
    Returns a matrix where each data point is
    replaced with its max-likelihood component mean.
    E.g., return the original matrix where each pixel's
    intensity replaced with its max-likelihood
    component mean. (the shape is still mxn, not
    original image size)

    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    k = int
    r = numpy.ndarray[numpy.ndarray[float]] - k x m - responsibility matrix

    returns:
    new_X = numpy.ndarray[numpy.ndarray[float]] - m x n
    """


    # same as from previous functions, unpack m, n and from X.
    m, n = X.shape

    # the return needs to be variable "new_X" so I am going to go ahead and initialized it with that name and
    # the shape specified of mxn.
    new_X = np.zeros((m, n))

    # to get the max-likelihood component, we just need to call our cluster() function I just created above, just plug in r
    clusters = cluster(r)

    # going to iterate over each data point m
    for i in range(m):
        # just need to get the max_likelihood_component of each pixel
        max_likelihood_component = clusters[i]

        # next replacing each pixel's intensity with the max_likelihood_component mean.
        # the mean is MU remember, we just have to find mean at max_likelihood_component.
        new_X[i] = MU[max_likelihood_component]

    # print(new_X)
    return new_X


def best_segment(X,k,iters):
    """Determine the best segmentation
    of the image by repeatedly
    training the model and
    calculating its likelihood.
    Return the segment with the
    highest likelihood.

    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    k = int
    iters = int

    returns:
    (likelihood, segment)
    likelihood = float
    segment = numpy.ndarray[numpy.ndarray[float]]
    """


    # so from the method description, all I have to do is call the train_model() function and the segment() function
    # Then calculate the likelihood, then return the segment with highest likelihood

    # so we have to return (likelihood, segment)
    # let's just go ahead and initialize those variables right now, but I am going to do add prefixes
    # since the functions we have to call are of the same name, so it can't be (likelihood, segment) exactly
    highest_likelihood = 0
    best_segment = None

    # need to run the for loop "iters" times.
    for i in range(iters):

        # First let's call the train_model() function and get the respective outputs from that
        # we will get MU, SIGMA, PI and responsibility, the input for that we have, which is X, K and convergence_function
        # I am going to use the default_convergence as the input
        MU, SIGMA, PI, responsibility = train_model(X, k, convergence_function=default_convergence)

        # Next going to call the segment() function, with the inputs X, MU, k and responsibility
        # we found MU and responsibility from the train model() function output
        # the output for this will be the current_segment
        current_segment = segment(X, MU, k, responsibility)

        # As per instructions, next going to calculate the likelihood using likelihood() function
        # we have all required, Pi, MU, sigma, X and K plug them in we get the current_likelihood
        current_likelihood = likelihood(X, PI, MU, SIGMA, k)

        # Now just have to do a simple condition checking if the current_likelihood is better highest_likelihood
        # if that's true that means we have to set the highest_likelihood to the current_likelihood and
        # same with segment, best_segment will be the current_segment
        if current_likelihood > highest_likelihood:
            highest_likelihood = current_likelihood
            best_segment = current_segment

    #print(highest_likelihood, best_segment)
    return highest_likelihood, best_segment


def improved_initialization(X,k):
    """
    Initialize the training
    process by setting each
    component mean using some algorithm that
    you think might give better means to start with,
    based on the mean calculate covariance matrices,
    and set each component mixing coefficient (PIs)
    to a uniform values
    (e.g. 4 components -> [0.25,0.25,0.25,0.25]).

    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    k = int

    returns:
    (MU, SIGMA, PI)
    MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    PI = numpy.ndarray[float] - k
    """


    # NOTE: sorry due to time I am going to copy paste my initialization function from PART 2,
    # so don't have time to change the comments

    # unpacking the given input X, into m and n. it's provided in description it's m x n
    m, n = X.shape

    # it said in the description of function to "Set component mean to a random pixel's value (without replacement)"
    # so let's do that, this is very similar to K_initial_means from part 1
    # there It asked to "Picks k random points from the 2D array (without replacement) to use as initial cluster means"
    # I did random_k_points = np.random.choice(array.shape[0], k, replace=False), going to replace array.shape[0] with m here
    # because remember is the m pixels and n is the features
    random_pixels = np.random.choice(m, k, replace=False)

    # next we are going to set component mean, which is MU, refer to compute_sigma function of why it is.
    # They want me to set MU to the random_pixels, going to do that. get the value of X at random_pixels
    MU = X[random_pixels]

    # next they said " based on the mean calculate covariance matrices", well I am going to call my compute_sigma function
    # set X input to X parms and MU to MU params
    SIGMA = compute_sigma(X,MU)

    # lastly from the expected return, we want the PI, which is the weight or the mixing coefficients.
    # from this reference: https://towardsdatascience.com/gaussian-mixture-model-clearly-explained-115010f7d4cf
    # It's said that weight at start of 3 GMM components, is set to 1/3. It depends on number of GMM components
    # what is number of GMM components, well it's k. so we will set it to 1/k all to start of. Also given shape of k,
    # each value in the shape will be 1/k. Just like example they used in reference, with 3 components all being 1/3, 1/3, 1/3.
    # The weight of each component is 1/3, here weight of each component is 1/k, given the size K.
    # I am going to use: https://numpy.org/doc/stable/reference/generated/numpy.full.html, it does exactly what i mentioned above
    PI = np.full(k, 1/k)

    # print(MU, SIGMA, PI)

    # return what's expected, which was  (MU, SIGMA, PI)
    return (MU, SIGMA, PI)


def new_convergence_function(previous_variables, new_variables, conv_ctr,
                             conv_ctr_cap=10):
    """
    Convergence function
    based on parameters:
    when all variables vary by
    less than 10% from the previous
    iteration's variables, increase
    the convergence counter.

    params:
    previous_variables = [numpy.ndarray[float]]
                         containing [means, variances, mixing_coefficients]
    new_variables = [numpy.ndarray[float]]
                    containing [means, variances, mixing_coefficients]
    conv_ctr = int
    conv_ctr_cap = int

    return:
    (conv_crt, converged)
    conv_ctr = int
    converged = boolean
    """


    # Note I am borrowing code from the helper method: default_convergence(), going to change it based of what they want here

    # the way default_convergence() was done is a little difference in that, it was compared the likelihoods, which are float
    # here we have to check for two arrays.
    # so I am going to set initialize the variable increase_convergence_ctr to True
    increase_convergence_ctr = True

    # increase_convergence_ctr = (abs(prev_likelihood) * 0.9 <
    #                             abs(new_likelihood) <
    #                             abs(prev_likelihood) * 1.1)

    # we can't exactly use the above code 1:1, but can make few tweaks. First thing is we need to
    # iterate through the previous variables and new variables inputs, so we can check the previous and new variable
    # instead of the previous and new likelihood, we can use new and previous variables
    # We can use zip function in python to combine both lists, and then iterate through their respective lists
    # using prev_var and new_var. Also another change I made was add an if statement, we are doing inverse conditional statement
    # to find out increase_convergence_ctr is false. The reason again being that they are lists not float values that we are
    # comparing, so I had to use np.all, couldn't do that without the if statement. once we find out that the
    # increase_convergence_ctr is false via the condition, we can break the loop and set it false.

    for prev_var, new_var in zip(previous_variables, new_variables):
        if not (np.all(abs(prev_var) * 0.9 < abs(new_var)) and np.all(abs(new_var) < abs(prev_var) * 1.1)):
            increase_convergence_ctr = False
            break
    #rest is the same, also the condition above is pretty much same, minus the np.all.
    if increase_convergence_ctr:
        conv_ctr += 1
    else:
        conv_ctr = 0
    # print(conv_ctr)
    return conv_ctr, conv_ctr > conv_ctr_cap

def train_model_improved(X, k, convergence_function=new_convergence_function, initial_values = None):
    """
    Train the mixture model using the
    expectation-maximization algorithm.
    E.g., iterate E and M steps from
    above until convergence.
    If the initial_values are None, initialize them.
    Else it's a tuple of the format (MU, SIGMA, PI).
    Convergence is reached when convergence_function
    returns terminate as True. Use new_convergence_fuction
    implemented above.

    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    k = int
    convergence_function = func
    initial_values = None or (MU, SIGMA, PI)

    returns:
    (new_MU, new_SIGMA, new_PI, responsibility)
    new_MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    new_SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    new_PI = numpy.ndarray[float] - k
    responsibility = numpy.ndarray[numpy.ndarray[float]] - k x m
    """


    # NOTE: sorry due to time I am going to copy paste my train model from PART 2, so don't have time to change the comments
    # I am going to mentione where I made the tweak to that existing code though.

    # I am going to first start of with what's told in the function description:
    # If the initial_values are None, initialize them.
    # okay so I am going to call the function initialize_parameters(), which returns the MU, SIGMA, and PI
    # we want the method to return (new_MU, new_SIGMA, new_PI, responsibility), not going to set it to that as we will find
    # the new MU, new SIGMA and new PI later, it's too confusing naming it new right of the bat.
    if initial_values is None:
        MU, SIGMA, PI = initialize_parameters(X, k)

    # it said Else it's a tuple of the format (MU, SIGMA, PI). so we are given initial_values,
    # unpack the MU, SIGMA and PI from he initial_values given.
    else:
        MU, SIGMA, PI = initial_values

    # next we need an variable to check, when "Convergence is reached when convergence_function returns terminate as True"
    # so going to set a variable similar to default_convergence function calling converged to False to begin
    converged = False

    # NEW: this is a new edition, as we know from previous function above, that we new previous and new variables
    # we set the previous_variables to whatever value is for MU, SIGMA and PI
    previous_variables = [MU, SIGMA, PI]

    # We need a counter for the convergence function again same as default_convergence function, with conv_ctr
    conv_ctr = 0

    # finally need to initialize responsibility as expected by the method signature in the return
    responsibility = 0

    # Just like the k_means_segment() function, need to run the while loop until it's converged. So as long as
    # converged is false, this loop won't break, that's why I intialized that variable to be false.
    while not converged:

        # I am just going to strictly follow what's said in the method signature:
        # "iterate E and M steps from above until convergence."

        # so let's get the E step by calling the E_Step function, the result is responsibility for Estep
        # we already have the required inputs for that function, the X, K and the MU, SIGMA and PI we initaliZed at top
        responsibility = E_step(X, MU, SIGMA, PI, k)

        # Next is the Mstep and it's same thing call the M_step() function, it returns new_MU, new_SIGMA, and new_PI
        # we have the X, responsibility and k already, the responsibility is from e_step. just plug in the values
        new_MU, new_SIGMA, new_PI = M_step(X, responsibility, k)

        # NEW: you might have noticed that the likelihood part from previously is gone
        # I am finding initializing the new_variables instead with the new_MU, new_SIGMA and new_PI
        new_variables = [new_MU, new_SIGMA, new_PI]

        # now we can get the convergence function, with the log_likelihood, newlog_likelihood and the conv_ctr, which is
        # initialized at the top. This function outputs whether or not the function is converged.
        # we need this converged value to eventually update to True, so this loop breaks
        # this was stated in the method description: " Convergence is reached when convergence_function returns terminate as True"
        # EDIT: my order of unpacking the convergence_function was wrong it's not converged, conv_ctr
        # it was silly mistake, I got the question wrong so updated it.

        # NEW: now we have previous variables and new_variables, we replace the previous convergence_function call
        # with the previous variables and new variables instead of the likelihood stuff. that's it.
        conv_ctr, converged = convergence_function(previous_variables, new_variables, conv_ctr)

        # Just like the k_means_segment() function have to update the variables with new ones
        # so for MU, SIGMA, PI, need to update with their newMU, new SIGMA and New PI.
        # new updated the current log_likelihood with the new one, in the next iteration the new one will be calculated
        # same with the mu, pi and sigma.

        # NEW: again same for the variable assignment, the MU, SIGMA and PI will come from new_variables now
        # the previous_variables will be set to the new_variables now. Same as likelihood. the return statys the same.
        MU, SIGMA, PI = new_variables
        previous_variables = new_variables

    #print(log_likelihood)
    #print(MU,SIGMA,PI, responsibility)
    # return the expected (new_MU, new_SIGMA, new_PI, responsibility)
    return MU, SIGMA, PI, responsibility


def bayes_info_criterion(X, PI, MU, SIGMA, k):
    """
    See description above
    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    PI = numpy.ndarray[float] - k
    k = int

    return:
    bayes_info_criterion = int
    """
    # From the wikipedia: https://en.wikipedia.org/wiki/Bayesian_information_criterion
    # the equation for BIC is , BIC = k ln(n) - 2 ln(L) : the L represents the likelihood function
    # k and n are the parameters. n is actually the number of data points, similar to all our previous functions.

    # next as usual, let's get the m and n values of X
    m, n = X.shape

    # we have all the components for the likelihood() function, let's compute it, it's just pluging the values
    log_likelihood = likelihood(X, PI, MU, SIGMA, k)

    # okay let's compute equation from above: BIC = k ln(n) - 2 ln(L)
    # going to initialize the bayes_info_criterion_result for the result

    # EDIT: I was missing doing this: k * np.log(m) - 2 * log_likelihood for the BIC below
    # that's wrong, because K isn't input K, it's total parameters in the equation in the equation for BIC
    # I got the parameters equation from the https://www.youtube.com/watch?v=aIs8UimAm4o&ab_channel=astomodynamics
    # the A5 walkthrough video on Bayesian Information Criterion slide
    # k * X.shape[1] + K * X.shape[1] * (X.shape[1] + 1) / 2 + k - 1 -> this was what was said there
    # since X.shape[1] is n for us, I can just sub in that for n and compute the parameters
    parameters = k * n + k * n * (n + 1) / 2 + k - 1

    # now I can do the right BIC calculation of BIC = k ln(n) - 2 ln(L), k is parameters not the actual k input
    bayes_info_criterion_result = parameters * np.log(m) - 2 * log_likelihood

    # print(bayes_info_criterion_result)
    return bayes_info_criterion_result

def BIC_likelihood_model_test(image_matrix, comp_means):
    """Returns the number of components
    corresponding to the minimum BIC
    and maximum likelihood with respect
    to image_matrix and comp_means.

    params:
    image_matrix = numpy.ndarray[numpy.ndarray[float]] - m x n
    comp_means = list(numpy.ndarray[numpy.ndarray[float]]) - list(k x n) (means for each value of k)

    returns:
    (n_comp_min_bic, n_comp_max_likelihood)
    n_comp_min_bic = int
    n_comp_max_likelihood = int
    """

    # first going to initialize the variables that are needed for the return statement (n_comp_min_bic, n_comp_max_likelihood)
    n_comp_min_bic = 0
    n_comp_max_likelihood = 0

    # we need to find number of components which result in min BIC and highest likelihood
    # let's initialize the variables for that, so min BIC and max likelihood
    min_bic = 0
    max_likelihood = 0

    # now we just have to iterate over the comp_means as said above:
    # " iterate over the list of provided means (`comp_means`) to train a model that minimizes its BIC and
    # a model that maximizes its likelihood."
    # also said "use the BIC and likelihood to determine the optimal number of components in the `image_matrix` parameter. "

    # first let's loop oer the comp_means

    for means in comp_means:
        # k is the number of clusters in comp_means
        # we will find k value for each means in the loop
        k = means.shape[0]

        # let's call the train_model_improved() now, we just need the result of MU, SIGMA and PI, no need for responsbility
        # the parameters in train_model_improved() will be, image_matrix for the X value, k will be k
        # the convergence function will be the new_convergence_function
        MU, SIGMA, PI, _ = train_model_improved(image_matrix, k, new_convergence_function)

        # next let's do BIC as instructed
        # we have the PI, MU, SIGMA from the train_model and k from above, the X will be image_matrix, as that's the data points
        current_bic = bayes_info_criterion(image_matrix, PI, MU, SIGMA, k)

        # now likelihood, same parameters as above for BIC
        current_likelihood = likelihood(image_matrix, PI, MU, SIGMA, k)

        # now we have the bic and likelihood, we will need to check for lowest BIC or highest likelihood and update
        # our variables we set outside the loop based of that condition for both
        # n_comp_min_bic and n_comp_max_likelihood are set to k since that's how many components result in min bic or
        # max likelihood

        # lowest BIC checking:
        if current_bic < min_bic:
            min_bic = current_bic
            n_comp_min_bic = k

        # highest likelihood checking:
        if current_likelihood > max_likelihood:
            max_likelihood = current_likelihood
            n_comp_max_likelihood = k

    # print(min_bic, max_likelihood)
    # print(n_comp_min_bic, n_comp_max_likelihood)
    return n_comp_min_bic, n_comp_max_likelihood