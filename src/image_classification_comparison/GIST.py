import numpy as np
import cv2 as cv
import scipy


def calculate_gist_descriptors(img_paths: list[str], new_img_length=256, boundary_extension=32, grid_length=4):
    """Calculate gist descriptors
    :param new_img_length: length to resize all images to
    :param boundary_extension: number of pixels to pad images by
    :param grid_length: number of cells to split each image side by to make the grid
    :param img_paths: list of image paths to get features from
    :return: list of gist descriptors where list[i] = gist descriptor of image at img_paths[i]
    """

    filter_bank = __create_gabor_filter_bank(img_size=new_img_length + 2 * boundary_extension)

    num_of_images = len(img_paths)
    num_of_grid_cells = grid_length ** 2
    num_of_features = filter_bank.shape[2] * num_of_grid_cells

    # initialise list of descriptors
    descriptors = np.zeros(shape=(num_of_images, num_of_features), dtype=np.float32)

    for i in range(len(img_paths)):
        img = cv.imread(img_paths[i])

        # convert image to grayscale
        img = np.mean(img, axis=-1, dtype=np.float32)

        img = __resize_and_crop(img, (new_img_length, new_img_length))

        # scale to be in range 0 to 255
        img = img - np.min(img)
        img = 255 * img / np.max(img)

        filtered_img = __preprocess(img)

        descriptor = __calculate_gist_descriptor(filtered_img, filter_bank, boundary_extension, grid_length,
                                                 num_of_grid_cells)
        descriptors[i, :] = descriptor

    return descriptors


def __resize_and_crop(img: np.ndarray, new_shape: tuple[int, int]):
    """Resize image to a new shape whilst maintaining the original aspect ratio
    :param img: the image to resize
    :param new_shape: the desired shape
    :return: image with the desired shape
    """

    # resize image while keeping aspect ratio
    scaling = max(new_shape[0] / img.shape[0], new_shape[1] / img.shape[1])
    resize_shape = tuple(np.round(np.array([img.shape[1], img.shape[0]]) * scaling).astype(int))
    img = cv.resize(src=img, dsize=resize_shape, interpolation=cv.INTER_LINEAR)

    # crop image to get desired shape
    starting_row = (img.shape[0] - new_shape[0]) // 2
    starting_col = (img.shape[1] - new_shape[1]) // 2
    img = img[starting_row:starting_row + new_shape[0], starting_col:starting_col + new_shape[1]]

    return img


def __preprocess(img: np.ndarray, fc=4):
    """Preprocess image through pre-filtering, whitening and local contrast normalisation
    :param fc: controls sigma in gaussian filter
    :param img: image to filter
    :return: filtered image
    """

    pad_w = 5

    # pad images to reduce boundary artifacts
    img = np.log(img + 1)
    img = np.pad(array=img, pad_width=((pad_w, pad_w), (pad_w, pad_w)), mode='symmetric')

    # dimensions of the padded image
    new_h, new_w = img.shape

    # make dimensions even through additional padding
    n = max(new_h, new_w)
    n += n % 2
    img = np.pad(array=img, pad_width=((0, n - new_h), (0, n - new_w)), mode='symmetric')

    # create gaussian filter
    fx, fy = np.meshgrid(np.arange(-n / 2, n / 2), np.arange(-n / 2, n / 2))
    sigma = fc / np.sqrt(np.log(2))
    gaussian_filter = scipy.fft.fftshift(np.exp(-(fx ** 2 + fy ** 2) / (sigma ** 2)))

    # whitening
    output = img - np.real(scipy.fft.ifft2(scipy.fft.fft2(img) * gaussian_filter))

    # local contrast normalisation
    local_std = np.sqrt(np.abs(scipy.fft.ifft2(scipy.fft.fft2(output ** 2) * gaussian_filter)))
    output = output / (0.2 + local_std)

    # crop output to have the same size as the input
    output = output[pad_w:new_h - pad_w, pad_w:new_w - pad_w]

    return output


def __create_gabor_filter_bank(img_size: int, num_of_orientations=11, num_of_scales=6):
    """Creates a filter bank of gabor filters and returns it
    :param img_size: size of the image the filters will be applied to
    :param num_of_orientations: number of different orientations needed in the filter bank
    :param num_of_scales: number of different scales needed in the filter bank
    :return: array of gabor filters
    """

    # the number of filters that will need to be generated
    num_of_filters = num_of_orientations * num_of_scales

    # generate the parameters for each filter
    filter_i = 0
    filter_params = np.zeros((num_of_filters, 4))
    for i in range(num_of_scales):
        for j in range(num_of_orientations):
            filter_params[filter_i, :] = [
                0.35,
                0.3 / (1.85**i),
                16 * num_of_orientations ** 2 / 32 ** 2,
                np.pi / num_of_orientations * j
            ]
            filter_i += 1

    # frequency information
    fx, fy = np.meshgrid(np.arange(-img_size / 2, img_size / 2),
                         np.arange(-img_size / 2, img_size / 2))
    radial_freq = np.fft.fftshift(np.sqrt(fx ** 2 + fy ** 2))
    angle = np.fft.fftshift(np.angle(fx + 1j * fy))

    # generate filters
    filter_bank = np.zeros((img_size, img_size, num_of_filters))
    for i in range(num_of_filters):
        curr_angle = angle + filter_params[i, 3]
        curr_angle = curr_angle + 2 * np.pi * (curr_angle < -np.pi) - 2 * np.pi * (curr_angle > np.pi)

        filter_bank[:, :, i] = np.exp(-10
                                      * filter_params[i, 0]
                                      * (radial_freq / img_size / filter_params[i, 1] - 1) ** 2 - 2
                                      * filter_params[i, 2] * np.pi * curr_angle ** 2)

    # reshape the filter bank if needed
    if np.all(np.array(np.shape(filter_bank))[:2] == 1):
        filter_bank = np.squeeze(filter_bank, axis=(0, 1))
    if np.all(np.array(np.shape(filter_bank))[:3] == 1):
        filter_bank = np.squeeze(filter_bank, axis=(2,))
    if np.any(np.array(np.shape(filter_bank))[:2] == 1):
        filter_bank = np.squeeze(filter_bank, axis=2)
    if np.all(np.array(np.shape(filter_bank)) == 1):
        filter_bank = filter_bank.squeeze()

    return filter_bank


def __calculate_features(filtered_img: np.ndarray, grid_length: int):
    """
    Calculates features from an image by splitting it into a grid and getting the mean of each cell
    :param filtered_img: the image to get the features from
    :param grid_length: number of cells on each grid side
    :return: list of image features
    """

    # get image dimensions
    img_h, img_w = filtered_img.shape

    # indices for dividing the image into non-overlapping blocks
    cell_row_indices = np.linspace(0, img_h, grid_length + 1, dtype=int)
    cell_col_indices = np.linspace(0, img_w, grid_length + 1, dtype=int)

    # initialise features array
    features = np.zeros((grid_length, grid_length))

    # iterate over each block
    for xx in range(grid_length):
        for yy in range(grid_length):

            # calculate mean value within the current cell
            feature = np.mean(np.mean(filtered_img[
                cell_row_indices[xx]:cell_row_indices[xx + 1],
                cell_col_indices[yy]:cell_col_indices[yy + 1]
            ]))
            features[xx, yy] = feature

    return features


def __calculate_gist_descriptor(img: np.ndarray, gabor_filter_bank: np.ndarray, boundary_extension: int,
                                grid_length: int, num_of_grid_cells: int):
    """
    Calculates the gist descriptor of an image
    :param img: the image to get the descriptor from
    :param gabor_filter_bank: the gabor filters to apply
    :param boundary_extension: number of pixels to pad image by
    :param grid_length: number of cells on each grid side
    :param num_of_grid_cells: total number of cells the image is split into (should be grid_length ** 2)
    :return: gist descriptor that represents the input image
    """

    # ensures image is assigned enough memory to store results of all calculations
    img = img.astype(np.float32)

    filter_h, filter_w, num_of_filters = gabor_filter_bank.shape

    # initialises descriptor
    descriptor = np.zeros(num_of_grid_cells * num_of_filters)

    # image padding to handle boundary effects
    img = np.pad(
        img,
        [(boundary_extension, boundary_extension), (boundary_extension, boundary_extension)],
        mode='symmetric'
    )

    # 2D fourier transform is applied to the image
    img = np.fft.fft2(img)

    # applies gabor filters to calculate descriptor
    next_free_i = 0
    for filter_i in range(num_of_filters):
        # applies gabor filter and then removes padding
        filtered_img = np.abs(np.fft.ifft2(img * gabor_filter_bank[:, :, filter_i]))
        filtered_img = filtered_img[
                       boundary_extension:filter_h - boundary_extension,
                       boundary_extension:filter_w - boundary_extension
                       ]

        # adds feature values from the filtered image to the descriptor
        features = __calculate_features(filtered_img, grid_length)
        descriptor[next_free_i:next_free_i + num_of_grid_cells] = features.reshape(num_of_grid_cells, order='F')

        next_free_i += num_of_grid_cells

    return descriptor
