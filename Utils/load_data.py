import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.data.experimental import AUTOTUNE as autotune


def load_vae_dataset(dataset_name, batch_n, epochs, hyper,
                     run_with_sample=True, architecture='dense'):
    images_to_display = [10, 25, 5, 29, 1, 35, 18, 30,
                         6, 19, 15, 23, 11, 21, 17, 26, 344, 3567, 9, 20]
    dtype = 'float32'
    if hyper['dtype'] == tf.float64:
        dtype = 'float64'
    elif hyper['dtype'] == tf.float16:
        dtype = 'float16'
    if dataset_name == 'fmnist':
        if architecture == 'conv_jointvae':
            _, np_test_images = fetch_and_binarize_mnist_data(dtype=dtype,
                                                              use_fashion=True)
            image_shape = (32, 32, 1)
            data = load_mnist_data(batch_n=batch_n, epochs=epochs,
                                   run_with_sample=run_with_sample,
                                   resize=True, use_fashion=True, dtype=dtype)
            np_test_images = tf.image.resize(np_test_images,
                                             size=image_shape[0:2]).numpy()
        else:
            _, np_test_images = fetch_and_binarize_mnist_data(dtype=dtype,
                                                              use_fashion=True)
            image_shape = (28, 28, 1)
            data = load_mnist_data(batch_n=batch_n, epochs=epochs,
                                   run_with_sample=run_with_sample,
                                   resize=False, use_fashion=True, dtype=dtype)
        np_test_images = np_test_images[images_to_display, :, :, :]
        train_dataset, test_dataset, batch_n, epochs = data
    elif dataset_name == 'mnist':
        if architecture == 'conv_jointvae':
            _, np_test_images = fetch_and_binarize_mnist_data(dtype)
            image_shape = (32, 32, 1)
            data = load_mnist_data(batch_n=batch_n, epochs=epochs,
                                   run_with_sample=run_with_sample,
                                   resize=True, dtype=dtype)
            np_test_images = tf.image.resize(
                np_test_images, size=image_shape[0:2]).numpy()
        else:
            _, np_test_images = fetch_and_binarize_mnist_data(dtype=dtype)
            image_shape = (28, 28, 1)
            data = load_mnist_data(batch_n=batch_n, epochs=epochs,
                                   run_with_sample=run_with_sample,
                                   resize=False, dtype=dtype)
        np_test_images = np_test_images[images_to_display, :, :, :]
        train_dataset, test_dataset, batch_n, epochs = data
    elif dataset_name == 'celeb_a':
        image_shape = (64, 64, 3)
        pd = ProcessData(dataset_name=dataset_name, run_with_sample=run_with_sample,
                         image_shape=image_shape)
        output = pd.generate_train_and_test_partitions(batch_size=batch_n, epochs=epochs,
                                                       test_size=19962)
        split_data, batch_size, epochs, np_test_images = output
        train_dataset, test_dataset = split_data
    elif dataset_name == 'omniglot':
        image_shape = (28, 28, 1)
        pd = ProcessData(dataset_name=dataset_name, run_with_sample=run_with_sample,
                         image_shape=image_shape)
        output = pd.generate_train_and_test_partitions(batch_size=batch_n, epochs=epochs)
        split_data, batch_size, epochs, np_test_images = output
        train_dataset, test_dataset = split_data
        if run_with_sample:
            batch_n, epochs = 5, 10
    elif dataset_name == 'cifar':
        image_shape = (32, 32, 1)
        pd = ProcessData(dataset_name=dataset_name, run_with_sample=run_with_sample,
                         image_shape=image_shape)
        output = pd.generate_train_and_test_partitions(batch_size=batch_n, epochs=epochs)
        split_data, batch_size, epochs, np_test_images = output
        train_dataset, test_dataset = split_data
    else:
        raise RuntimeError

    hyper = refresh_hyper(hyper, batch_n, epochs, image_shape,
                          dataset_name, run_with_sample)
    return train_dataset, test_dataset, np_test_images, hyper


def refresh_hyper(hyper, batch_n, epochs, image_shape, dataset_name, run_with_sample):
    iter_per_epoch = determine_iter_per_epoch(dataset_name=dataset_name,
                                              run_with_sample=run_with_sample,
                                              batch_n=batch_n)
    hyper['batch_n'] = batch_n
    hyper['epochs'] = epochs
    hyper['image_shape'] = image_shape
    hyper['iter_per_epoch'] = iter_per_epoch
    return hyper


def load_mnist_data(dtype, batch_n, epochs, use_fashion=False,
                    run_with_sample=False, resize=False):
    train_images, test_images = fetch_and_binarize_mnist_data(
        dtype, use_fashion=use_fashion)
    if run_with_sample:
        train_buffer, test_buffer, batch_n, epochs = 60, 10, 5, 10
        train_images, test_images = train_images[:60, :, :, :], test_images[:10, :, :, :]
    else:
        train_buffer, test_buffer, batch_n, epochs = 60_000, 10_000, batch_n, epochs

    if resize:
        train_images = tf.image.resize(train_images, size=(32, 32))
        test_images = tf.image.resize(test_images, size=(32, 32))
    train_dataset = tf.data.Dataset.from_tensor_slices(
        train_images).shuffle(train_buffer).batch(batch_n)
    test_dataset = tf.data.Dataset.from_tensor_slices(
        test_images).shuffle(test_buffer).batch(batch_n)
    return train_dataset, test_dataset, batch_n, epochs


def load_mnist_sop_data(batch_n, run_with_sample=False):
    train_images, test_images = fetch_and_binarize_mnist_data()
    if run_with_sample:
        train_buffer, test_buffer = 60, 10
        train_images = train_images[:train_buffer, :, :, :]
        test_images = test_images[:test_buffer, :, :, :]
    else:
        train_buffer, test_buffer = 60_000, 10_000

    train_dataset = tf.data.Dataset.from_tensor_slices(
        train_images).shuffle(train_buffer).batch(batch_n)
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    test_dataset = tf.data.Dataset.from_tensor_slices(
        test_images).shuffle(test_buffer).batch(batch_n)
    test_dataset = test_dataset.cache()
    test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return train_dataset, test_dataset


def fetch_and_binarize_mnist_data(dtype, use_fashion=False, output_labels=False):
    if use_fashion:
        im = tf.keras.datasets.fashion_mnist.load_data()
        (train_images, train_labels), (test_images, test_labels) = im
        train_images, test_images = reshape_binarize_and_scale_images((train_images,
                                                                       test_images),
                                                                      round_images=False,
                                                                      dtype=dtype)
    else:
        (train_images, train_labels), (test_images,
                                       test_labels) = tf.keras.datasets.mnist.load_data()
        train_images, test_images = reshape_binarize_and_scale_images((train_images,
                                                                       test_images),
                                                                      dtype=dtype)

    if output_labels:
        return (train_images, train_labels), (test_images, test_labels)
    else:
        return train_images, test_images


class ProcessData:

    def __init__(self, dataset_name, image_shape: tuple = (28, 28, 1),
                 run_with_sample=False):
        self.dataset_name = dataset_name
        self.run_with_sample = run_with_sample
        self.image_shape = image_shape

    def generate_train_and_test_partitions(self, batch_size, epochs):
        data = fetch_data_via_tf_datasets(dataset_name=self.dataset_name)
        buffer_size, batch_size, epochs = determine_buffer_re_assign_batch_and_epochs(
            self.run_with_sample, batch_size=batch_size, epochs=epochs)
        split_names, split_data, test_position = ['train', 'test'], [], 1
        for idx, split in enumerate(split_names):
            processed_data = self.preprocess(data_split=data[idx],
                                             buffer_size=buffer_size,
                                             batch_size=batch_size)
            split_data.append(processed_data)
        np_test_images = self.fetch_test_numpy_images_batch(
            test_ds=split_data[test_position], run_with_sample=self.run_with_sample)
        return split_data, batch_size, epochs, np_test_images

    def preprocess(self, data_split, buffer_size, batch_size):
        if self.dataset_name == 'omniglot':
            # data_split = data_split.map(preprocess_omniglot)
            data_split = data_split.map(preprocess_omniglot, num_parallel_calls=autotune)
        elif self.dataset_name == 'celeb_a':
            data_split = data_split.map(preprocess_celeb_a, num_parallel_calls=autotune)
        elif self.dataset_name == 'cifar':
            data_split = data_split.map(preprocess_cifar, num_parallel_calls=autotune)
        else:
            raise RuntimeError
        data_split = data_split.shuffle(buffer_size)
        data_split = data_split.batch(batch_size)
        data_split = data_split.cache()
        data_split = data_split.prefetch(tf.data.experimental.AUTOTUNE)
        return data_split

    def fetch_test_numpy_images_batch(self, test_ds, run_with_sample):
        test_images = iterate_over_dataset_container(data_iterable=test_ds,
                                                     image_shape=self.image_shape)
        if not run_with_sample:
            images_to_display = [10, 25, 5, 29, 1, 35, 18, 30,
                                 6, 19, 15, 23, 11, 21, 17, 26, 34, 57, 9, 20]
        else:
            images_to_display = [0, 1]
        return test_images[images_to_display, :, :, :]


def fetch_data_via_tf_datasets(dataset_name):
    if dataset_name != 'omniglot':
        builder = tfds.builder(name=dataset_name)
        builder.download_and_prepare()
        data = builder.as_dataset(shuffle_files=False)
    else:
        data = tfds.load(dataset_name,
                         split=['train+test[:5065]', 'test[5065:]'],
                         shuffle_files=False)
    return data


def determine_buffer_re_assign_batch_and_epochs(run_with_sample, batch_size, epochs):
    if run_with_sample:
        buffer, batch_size, epochs = 60, 5, 10
    else:
        buffer = 1000
    return buffer, batch_size, epochs


def preprocess_cifar(example):
    return example['image']


def preprocess_celeb_a(example):
    example['image'] = binarize_tensor(example['image'])
    example['image'] = tf.image.resize(images=example['image'], size=(64, 64))
    return example['image']


def preprocess_omniglot(example):
    example['image'] = binarize_tensor(example['image'])
    example['image'] = round_tensor(example['image'])
    example['image'] = example['image'][:, :, 0]
    example['image'] = tf.expand_dims(example['image'], 2)
    example['image'] = tf.image.resize(images=example['image'], size=(28, 28))
    return example['image']


def binarize_tensor(tensor):
    new_tensor = tf.cast(tensor, dtype=tf.float32)
    new_tensor = new_tensor / 255.
    return new_tensor


def round_tensor(tensor):
    rounded_tensor = tf.cast(tensor, dtype=tf.int32)
    rounded_tensor = tf.cast(rounded_tensor, dtype=tf.float32)
    return rounded_tensor


def crop_tensor(tensor, limit: int):
    tensor = tf.reshape(tensor[:limit, :limit, 0], shape=(limit, limit, 1))
    return tensor


def iterate_over_dataset_container(data_iterable, image_shape):
    for image in data_iterable:
        return image.numpy()


def determine_iter_per_epoch(dataset_name, run_with_sample, batch_n):
    if dataset_name == 'fmnist' or dataset_name == 'mnist':
        num_of_data_points = 60000
        iter_per_epoch = num_of_data_points // batch_n
    elif dataset_name == 'omniglot':
        if run_with_sample:
            iter_per_epoch = 5
        else:
            num_of_data_points = 24345
            iter_per_epoch = (num_of_data_points // batch_n) + 1
    elif dataset_name == 'celeb_a':
        if run_with_sample:
            iter_per_epoch = 5
        else:
            num_of_data_points = 19962
            iter_per_epoch = num_of_data_points // batch_n
    else:
        raise RuntimeError
    return iter_per_epoch


def reshape_binarize_and_scale_images(images, round_images=True, dtype='float32'):
    output = []
    for im in images:
        im = np.reshape(im, newshape=im.shape + (1,)).astype(dtype)
        im /= 255.
        if round_images:
            im[im >= 0.5] = 1.
            im[im < 0.5] = 0.
        output.append(im)
    return output
