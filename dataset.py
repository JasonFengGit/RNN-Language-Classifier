import numpy as np
import time
import os
import nn


class LanguageClassificationDataset():
    def __init__(self, model):
        self.model = model

        data_path = "lang_id.npz"

        with np.load(data_path) as data:
            self.chars = data['chars']
            self.language_codes = data['language_codes']
            self.language_names = data['language_names']

            self.train_x = data['train_x']
            self.train_y = data['train_y']
            self.train_buckets = data['train_buckets']
            self.dev_x = data['dev_x']
            self.dev_y = data['dev_y']
            self.dev_buckets = data['dev_buckets']
            self.test_x = data['test_x']
            self.test_y = data['test_y']
            self.test_buckets = data['test_buckets']

        self.epoch = 0
        self.bucket_weights = self.train_buckets[:,1] - self.train_buckets[:,0]
        self.bucket_weights = self.bucket_weights / float(self.bucket_weights.sum())

        self.chars_print = self.chars

        # Select some examples to spotlight in the monitoring phase (3 per language)
        spotlight_idxs = []
        for i in range(len(self.language_names)):
            idxs_lang_i = np.nonzero(self.dev_y == i)[0]
            idxs_lang_i = np.random.choice(idxs_lang_i, size=3, replace=False)
            spotlight_idxs.extend(list(idxs_lang_i))
        self.spotlight_idxs = np.array(spotlight_idxs, dtype=int)

        # Templates for printing updates as training progresses
        max_word_len = self.dev_x.shape[1]
        max_lang_len = max([len(x) for x in self.language_names])

        self.predicted_template = u"Pred: {:<NUM}".replace('NUM',
            str(max_lang_len))

        self.word_template = u"  "
        self.word_template += u"{:<NUM} ".replace('NUM', str(max_word_len))
        self.word_template += u"{:<NUM} ({:6.1%})".replace('NUM', str(max_lang_len))
        self.word_template += u" {:<NUM} ".replace('NUM',
            str(max_lang_len + len('Pred: ')))
        for i in range(len(self.language_names)):
            self.word_template += u"|{}".format(self.language_codes[i])
            self.word_template += "{probs[" + str(i) + "]:4.0%}"

        self.last_update = time.time()

    def _encode(self, inp_x, inp_y, only_x=False):
        xs = []
        for i in range(inp_x.shape[1]):
            if np.all(inp_x[:,i] == -1):
                break
            x = np.eye(len(self.chars))[inp_x[:,i]]
            xs.append(nn.Constant(x))
        if(not only_x):
            y = np.eye(len(self.language_names))[inp_y]
            y = nn.Constant(y)
            return xs, y
        return xs

    def _softmax(self, x):
        exp = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp / np.sum(exp, axis=-1, keepdims=True)

    def _predict(self, split='dev'):
        if split == 'dev':
            data_x = self.dev_x
            data_y = self.dev_y
            buckets = self.dev_buckets
        else:
            data_x = self.test_x
            data_y = self.test_y
            buckets = self.test_buckets
        all_predicted = []
        all_correct = []
        for bucket_id in range(buckets.shape[0]):
            start, end = buckets[bucket_id]
            xs, y = self._encode(data_x[start:end], data_y[start:end])
            predicted = self.model.run(xs)

            all_predicted.extend(list(predicted.data))
            all_correct.extend(list(data_y[start:end]))

        all_predicted_probs = self._softmax(np.asarray(all_predicted))
        all_predicted = np.asarray(all_predicted).argmax(axis=-1)
        all_correct = np.asarray(all_correct)

        return all_predicted_probs, all_predicted, all_correct

    def iterate_once(self, batch_size):
        assert isinstance(batch_size, int) and batch_size > 0, (
            "Batch size should be a positive integer, got {!r}".format(
                batch_size))
        assert self.train_x.shape[0] >= batch_size, (
            "Dataset size {:d} is smaller than the batch size {:d}".format(
                self.train_x.shape[0], batch_size))

        self.epoch += 1

        for iteration in range(self.train_x.shape[0] // batch_size):
            bucket_id = np.random.choice(self.bucket_weights.shape[0], p=self.bucket_weights)
            example_ids = self.train_buckets[bucket_id, 0] + np.random.choice(
                self.train_buckets[bucket_id, 1] - self.train_buckets[bucket_id, 0],
                size=batch_size)

            yield self._encode(self.train_x[example_ids], self.train_y[example_ids])

            if time.time() - self.last_update > 0.5:
                dev_predicted_probs, dev_predicted, dev_correct = self._predict()
                dev_accuracy = np.mean(dev_predicted == dev_correct)

                print("epoch {:,} iteration {:,} validation-accuracy {:.1%}".format(
                    self.epoch, iteration, dev_accuracy))

                for idx in self.spotlight_idxs:
                    correct = (dev_predicted[idx] == dev_correct[idx])
                    word = u"".join([self.chars_print[ch] for ch in self.dev_x[idx] if ch != -1])

                    print(self.word_template.format(
                        word,
                        self.language_names[dev_correct[idx]],
                        dev_predicted_probs[idx, dev_correct[idx]],
                        "" if correct else self.predicted_template.format(
                            self.language_names[dev_predicted[idx]]),
                        probs=dev_predicted_probs[idx,:],
                    ))
                print()
                self.last_update = time.time()

    def get_validation_accuracy(self):
        dev_predicted_probs, dev_predicted, dev_correct = self._predict()
        dev_accuracy = np.mean(dev_predicted == dev_correct)
        return dev_accuracy
