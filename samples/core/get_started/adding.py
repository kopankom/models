import pandas as pd
import tensorflow as tf

TRAIN_URL = "training_data.csv"
TEST_URL = "test_data.csv"
CSV_COLUMN_NAMES = ['val1', 'val2', 'result']


def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset


def load_train_data(y_name='result'):
    train = pd.read_csv(TRAIN_URL, names=CSV_COLUMN_NAMES, header=0, delimiter=';')
    train_x, train_y = train, train.pop(y_name)

    return train_x, train_y


def load_test_data(y_name='result'):
    test = pd.read_csv(TEST_URL, names=CSV_COLUMN_NAMES, header=0, delimiter=';')
    test_x, test_y = test, test.pop(y_name)

    return test_x, test_y


def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features = dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset


def main(argv):
    train_x, train_y = load_train_data()
    test_x, test_y = load_test_data()
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key, dtype=tf.int32))
    classifier = tf.estimator.DNNRegressor(
        feature_columns=my_feature_columns,
        hidden_units=[100, 100],
        model_dir='model_2'
    )
    # for i in xrange(30):
    classifier.train(input_fn=lambda: train_input_fn(train_x, train_y, len(train_x)), steps=50000)

    classifier.evaluate(input_fn=lambda: train_input_fn(test_x, test_y, len(test_x)), steps=50)

    expected = [8, 12, 200, 81, 211, 299, 250, 1789, 45000]
    predict_x = {
        'val1': [4, 6, 100, 40, 30, 99, 100, 1001, 40000],
        'val2': [4, 6, 100, 41, 181, 200, 150, 788, 5000],
    }

    predictions = classifier.predict(input_fn=lambda: eval_input_fn(predict_x,
                                                                    labels=None,
                                                                    batch_size=440))
    a = list(predictions)
    index = 0
    # print("length: ", len(train_x))
    for prediction in a:
        # assert round(prediction['predictions'][0]) == expected[index]
        if (round(prediction['predictions'][0]) != expected[index]):
            # print(prediction['predictions'][0], " ", expected[index])
            print(abs(expected[index]-prediction['predictions'][0]), expected[index], prediction['predictions'][0])
            # main(argv)
        index += 1


if __name__ == '__main__':
    # tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
