division_factor = 1.2  # The initial factor to reduce the image. The next divisions are division_factor*2^n,
                       # n is the index of the reduction
threshold_score = 0.9  # The minimum score for a frame to get a subdivision from it
nb_shrinkages = 5  # The number of time we reduce the original image and get a reduction from it.
min_samples = 8  # The minimum of subdetections in a same cluster for the cluster to be accepted as a detection
strides = (5, 5)  # stride x and stride y while sliding an image (or a reduction of it)
nb_image_feeding = 10 # The number of times you feed the image to the neural network before clustering

path_to_original_dataset = 'start_deep/start_deep/'
data_augmentation = True

