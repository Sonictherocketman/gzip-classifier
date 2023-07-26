# Gzip Classifier

A python implementation of a gzip-based text classification system based on the algorithm described in "Less is More: Parameter-Free Text Classification with Gzip" by [Zhiying Jiang, Matthew Y.R. Yang, Mikhail Tsirlin, Raphael Tang, and Jimmy Lin](https://arxiv.org/pdf/2212.09410.pdf).

This code is largely untested and was put together over a single weekend. Please don't get too mad if it doesn't really work.


## Installation

```
pip install git+https://github.com/Sonictherocketman/gzip-classifier
```

*Installation via PyPi is coming soon.*


## Usage

There are two steps needed to use the classifier. First you need to train it using some pre-existing data. This data must be classified into a set of discrete labels which will be used to classify new data once the model is trained.

Once the model is trained it can be either exported and saved for later use OR it can be used immediately.

If you have any further questions, please refer to the tests for examples on using the classifier.


### Training the Model

The training process requires two sets of data: a list of sample text snippets and a list of the labels that correspond to those snippets.

Consider this sample dataset:

```
label, text
-----------
science,      Physicists discover new spacetime geometry
arts-culture, Famous author releases new book
world,        War in <country> is now over!
```

These are fed into the classifier as shown here:

```
from gzip_classifier import Classifier

training_data = [...]
labels = [...]

classifier = Classifier()
classifier.train(training_data, labels)

# OR

with ParallelClassifier(processes=4) as classifier:
    classifier.train(training_data, labels)
```


### Using the Model

Once trained, the model can be used to classify new text.

For insight on the k parameter, please refer to [this Wikipedia article](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm). In short, the value depends on your data. Try playing around with integer values greater than one to see what best fits your data. I'm sure there are ways to determine the best value for this, but I don't know enough about it to tell you. I just played with it.

```
classifier.classify('some new text', k=20)
>> 'a label'
```

You can also do bulk classification using the following method:

```
samples = [
    'some new text',
    'more new text',
]

for label in classifier.classify_bulk(samples, k=20)
    print(label)

>> 'a label'
>> 'another label'
```


### Improving Performance

The gzip classifier is highly parallelizable and so included in the package is a `ParallelClassifier` that uses Python's built in multiprocessing pool to delegate training and classifying samples across cores.

The parallel classifier is best used as a context manager:

```
with ParallelClassifier(processes=4) as classifier:
    classifier.train(training_data, labels)

    sample = 'some text'
    classifier.classify(sample, k=10)
```


### Exporting & Importing the Model

Once trained, the model can be exported and saved for later use.

```
# Normal export

model_contents = classifier.model
with open('model.txt', 'w') as f:
    f.write(model_contents)

# Compressed export

model_contents = classifier.compact_model
with open('model.gz', 'wb') as f:
    f.write(model_contents)
```

You can easily load a model from disk like this:

```
# Normal import

with open('model.txt', 'r') as f:
    classifier = Classifier.using_model(f.read())

# Compressed import

model_contents = classifier.compact_model
with open('model.gz', 'rb') as f:
    classifier = Classifier.using_compact_model(f.read())
```


## Testing

Run the tests using the following command.

```
python -m pytest tests/
```


## Contributing

Contributions are welcome. Just don't be a jerk.


## License

This code is released under the MIT license. Use it for whatever.
