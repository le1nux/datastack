

![DataStack_logo](https://user-images.githubusercontent.com/47029859/148431377-239f5a0a-fc84-47f0-91a6-a874f634f43b.png)



_______________________________________________________________________________

DataStack

A stream-based file storage solution for machine learning datasets.

[![PyPI version](https://badge.fury.io/py/datastack.svg)](https://badge.fury.io/py/datastack)
[![CircleCI](https://circleci.com/gh/le1nux/datastack.svg?style=svg)](https://circleci.com/gh/le1nux/datastack)
[![codecov](https://codecov.io/gh/le1nux/datastack/branch/master/graph/badge.svg)](https://codecov.io/gh/le1nux/datastack)

Today, machine learning datasets are abundantly availabe on the internet, while coming in a variety of formats( e.g., [pandas dataframes](https://pandas.pydata.org/), CSV files, numpy arrays, excel sheets, [h5py](https://www.h5py.org/) and many more), which makes generic dataset processing complex. Luckily, almost all recent libraries provide a [file-like interface](https://docs.python.org/3/glossary.html#term-file-object) for loading and storing datasets as binary streams, which is also the common ground DataStack builds upon. In DataStack datasets are stored as plain binary streams and loaded via custom iterator implementations specific for each file type. Thereby, the storage itself is completely independent from the file-type. The binary streams can be even lazyily loaded, given that the iterator supports it. The H5Py file format for instance supports this out of the box. 

Another important feature of DataStack is its ability to stack iterators. Having a dataset iterator as the foundation, custom higher level iterators like iterator views that allow for arbitrary dataset splits and combined iterators that join dataset splits, can be stacked on top. Higher order iterators in other research projects adopting DataStack already comprise more sophisticated iterators like feature encoding iterators and target class mapping iterators. 

So how does DataStack fit into the machine learning engineering work flow? While access to training data is not a limit anymore, integrating datasets into machine learning work flows still requires time-consuming manual preparation. Switching from one project or research paper to another, machine learning engineers and researchers often start from scratch integrating the same datasets over and over again. DataStack offers a solution for integrating these datasets by providing stable interfaces for data access that machine learning algorithms can work against. Having those interfaces in place, allows to reuse datasets and replicate results more easily. 

DataStack offers the following key modules:

* **Dataset Retrieval:** Datasets can be retrieved via the `HTTPRetriever`. If a custom retriever is needed, e.g., for a custom database, only the Retriever interface needs to be implemented.

* **Dataset Storage:** DataStack comes with a `FileStorageConnector` for storing and loading datasets from disk using a predefined dataset identifier. By implementing the `StorageConnector` interface, any other custom storage solution, e.g. a MongoDB, can be supported. Notably, every dataset is stored as a `StreamedResource`, which is a wrapper around the Python's [IOBase](https://docs.python.org/3/library/io.html#i-o-base-classes). Therefore, the respective `DatasetStorage` does not require any knowledge of the encoded data. This is why, the storage is not limited to any specific file-type. Additionally, when accessing the file storage, only a file descriptor to that file is created, offering lazy loading for iterators.

* **Iterator:** Datastack provides an iterator interface and a few implementations to iterate through datasets. An iterator takes a `StreamedResource` containing a binarized dataset and provides an iteration routine, customized to to the original filetype of the dataset. For instance, a binary Pytorch Tensor stream needs a different iteration implemenation than a CSV stream. Note, that the `StreamedResource` only provides a file descriptor to the stream. If this stream is stored on disk, the `StreamedResource` does not automatically load the stream into memory. This gives the opportunity to lazily load samples with e.g., h5py file streams. 

## Install

There are two options to install DataStack, the easiest way is to install it from  the pip repository:

```bash
pip install datastack
``` 

For the latest version, one can directly install it from source by `cd` into the root folder and then running  

```bash
pip install src/
```

## Usage

**NOTE: This library is still under heavy development. It's most likely not free of bugs and interfaces can still change.**

To implement a new dataset, one has to implement 3 classes: 

* **DatasetFactory:** Retrieves, prepares, stores and loads the dataset using a `Retriever` and `Preprocessor` implementation and a `StorageConnector`.
* **Preprocessing:** Datasets often come compressed, split up over many files or in who knows what structure. Therefore, for each dataset we need a Preprocessing class that transforms the datasets into a `StreamedResource`. 
* **Iterator:** Provides the iteration implementation on top of the binary stream `StreamedResource`

DataStack provides a [examplary MNIST implementation](https://github.com/le1nux/datastack/blob/master/src/data_stack/mnist/factory.py). 

## Copyright

Copyright (c) 2020 Max Lübbering
For license see: https://github.com/le1nux/datastack/blob/master/LICENSE

