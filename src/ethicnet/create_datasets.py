import tensorflow
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
import multiprocessing
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import image_ops
import numpy as np

def load_UTKFace_train_val_datasets(label,batch_size,df):
    ''' création de training et validation dataset en spécifiant quel doit être le label'''
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        config.data_dir, 
        labels=list(df[label]), 
        label_mode='binary',
        class_names=None, 
        color_mode='rgb', 
        batch_size=batch_size, 
        image_size=(200, 200), 
        shuffle=True, 
        seed=4, 
        validation_split=0.2, 
        subset='training',
        interpolation='bilinear', 
        follow_links=False
    )
    validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
        config.data_dir, 
        labels=list(df[label]), 
        label_mode='binary',
        class_names=None, 
        color_mode='rgb', 
        batch_size=batch_size, 
        image_size=(200, 200), 
        shuffle=True, 
        seed=4, 
        validation_split=None, 
        subset='validation',
        interpolation='bilinear', 
        follow_links=False
    )
    return train_ds,validation_ds

# Ces fonctions sont importées de keras directement, mais ne sont pas utilisables en important keras.
# Elles sont donc recopiées ici

def load_image(path, image_size, num_channels, interpolation,
               smart_resize=False):
  """Load an image from a path and resize it."""
  img = io_ops.read_file(path)
  img = image_ops.decode_image(
      img, channels=num_channels, expand_animations=False)
  if smart_resize:
    img = keras_image_ops.smart_resize(img, image_size,
                                       interpolation=interpolation)
  else:
    img = image_ops.resize_images_v2(img, image_size, method=interpolation)
  img.set_shape((image_size[0], image_size[1], num_channels))
  return img

def labels_to_dataset(labels, label_mode, num_classes):
  """Create a tf.data.Dataset from the list/tuple
   of labels.
  Args:
    labels: list/tuple of labels to be converted into a tf.data.Dataset.
    label_mode:
    - 'binary' indicates that the labels (there can be only 2) are encoded as
      `float32` scalars with values 0 or 1 (e.g. for `binary_crossentropy`).
    - 'categorical' means that the labels are mapped into a categorical vector.
      (e.g. for `categorical_crossentropy` loss).
    num_classes: number of classes of labels.
  """
  label_ds = dataset_ops.Dataset.from_tensor_slices(labels)
  if label_mode == 'binary':
    label_ds = label_ds.map(
        lambda x: array_ops.expand_dims(math_ops.cast(x, 'float32'), axis=-1))
  elif label_mode == 'categorical':
    label_ds = label_ds.map(lambda x: array_ops.one_hot(x, num_classes))
  return label_ds

def index_directory(directory,
                    labels,
                    formats,
                    class_names=None,
                    shuffle=True,
                    seed=None,
                    follow_links=False):
  """Make list of all files in the subdirs of `directory`, with their labels.
  Args:
    directory: The target directory (string).
    labels: Either "inferred"
        (labels are generated from the directory structure),
        None (no labels),
        or a list/tuple of integer labels of the same size as the number of
        valid files found in the directory. Labels should be sorted according
        to the alphanumeric order of the image file paths
        (obtained via `os.walk(directory)` in Python).
    formats: Allowlist of file extensions to index (e.g. ".jpg", ".txt").
    class_names: Only valid if "labels" is "inferred". This is the explict
        list of class names (must match names of subdirectories). Used
        to control the order of the classes
        (otherwise alphanumerical order is used).
    shuffle: Whether to shuffle the data. Default: True.
        If set to False, sorts the data in alphanumeric order.
    seed: Optional random seed for shuffling.
    follow_links: Whether to visits subdirectories pointed to by symlinks.
  Returns:
    tuple (file_paths, labels, class_names).
      file_paths: list of file paths (strings).
      labels: list of matching integer labels (same length as file_paths)
      class_names: names of the classes corresponding to these labels, in order.
  """
  if labels is None:
    # in the no-label case, index from the parent directory down.
    subdirs = ['']
    class_names = subdirs
  else:
    subdirs = []
    for subdir in sorted(os.listdir(directory)):
      if os.path.isdir(os.path.join(directory, subdir)):
        subdirs.append(subdir)
    if not class_names:
      class_names = subdirs
    else:
      if set(class_names) != set(subdirs):
        raise ValueError(
            'The `class_names` passed did not match the '
            'names of the subdirectories of the target directory. '
            'Expected: %s, but received: %s' %
            (subdirs, class_names))
  class_indices = dict(zip(class_names, range(len(class_names))))

  # Build an index of the files
  # in the different class subfolders.
  pool = multiprocessing.pool.ThreadPool()
  results = []
  filenames = []

  for dirpath in (os.path.join(directory, subdir) for subdir in subdirs):
    results.append(
        pool.apply_async(index_subdirectory,
                         (dirpath, class_indices, follow_links, formats)))
  labels_list = []
  for res in results:
    partial_filenames, partial_labels = res.get()
    labels_list.append(partial_labels)
    filenames += partial_filenames
  if labels not in ('inferred', None):
    if len(labels) != len(filenames):
      raise ValueError('Expected the lengths of `labels` to match the number '
                       'of files in the target directory. len(labels) is %s '
                       'while we found %s files in %s.' % (
                           len(labels), len(filenames), directory))
  else:
    i = 0
    labels = np.zeros((len(filenames),), dtype='int32')
    for partial_labels in labels_list:
      labels[i:i + len(partial_labels)] = partial_labels
      i += len(partial_labels)

  if labels is None:
    print('Found %d files.' % (len(filenames),))
  else:
    print('Found %d files belonging to %d classes.' %
          (len(filenames), len(class_names)))
  pool.close()
  pool.join()
  file_paths = [os.path.join(directory, fname) for fname in filenames]

  if shuffle:
    # Shuffle globally to erase macro-structure
    if seed is None:
      seed = np.random.randint(1e6)
    rng = np.random.RandomState(seed)
    rng.shuffle(file_paths)
    rng = np.random.RandomState(seed)
    rng.shuffle(labels)
  return file_paths, labels, class_names

def index_subdirectory(directory, class_indices, follow_links, formats):
  """Recursively walks directory and list image paths and their class index.
  Args:
    directory: string, target directory.
    class_indices: dict mapping class names to their index.
    follow_links: boolean, whether to recursively follow subdirectories
      (if False, we only list top-level images in `directory`).
    formats: Allowlist of file extensions to index (e.g. ".jpg", ".txt").
  Returns:
    tuple `(filenames, labels)`. `filenames` is a list of relative file
      paths, and `labels` is a list of integer labels corresponding to these
      files.
  """
  dirname = os.path.basename(directory)
  valid_files = iter_valid_files(directory, follow_links, formats)
  labels = []
  filenames = []
  for root, fname in valid_files:
    labels.append(class_indices[dirname])
    absolute_path = os.path.join(root, fname)
    relative_path = os.path.join(
        dirname, os.path.relpath(absolute_path, directory))
    filenames.append(relative_path)
  return filenames, labels

def paths_and_labels_to_dataset(image_paths,
                                image_size,
                                num_channels,
                                labels,
                                label_mode,
                                num_classes,
                                interpolation,
                                smart_resize=False):
  """Constructs a dataset of images and labels."""
  path_ds = dataset_ops.Dataset.from_tensor_slices(image_paths)
  args = (image_size, num_channels, interpolation, smart_resize)
  img_ds = path_ds.map(
      lambda x: load_image(x, *args))
  if label_mode:
    label_ds = labels_to_dataset(labels, label_mode, num_classes)
    image_ds = dataset_ops.Dataset.zip((img_ds, label_ds))
  return img_ds, label_ds

def iter_valid_files(directory, follow_links, formats):
  walk = os.walk(directory, followlinks=follow_links)
  for root, _, files in sorted(walk, key=lambda x: x[0]):
    for fname in sorted(files):
      if fname.lower().endswith(formats):
        yield root, fname

def separate_train_validation(dataset, ratio = 0.8):
    total = 23708
    n_training = int(ratio * total)
    dataset.shuffle(buffer_size = 64)
    train_ds = dataset.take(n_training)
    val_ds = dataset.skip(total - n_training)
    return train_ds,val_ds

def combine_and_batch_img_label_datasets(img_ds,label_ds,batch_size):
    return dataset_ops.Dataset.zip((img_ds, labels_ds)).batch(batch_size)
