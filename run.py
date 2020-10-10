import model
import dataset
import numpy as np

def word_to_hash(chars, word):
    word = word.lower()
    chars = list(chars)
    hashed = [chars.index(char) for char in word]
    while(len(hashed) < 10):
        hashed.append(-1)
    return np.ndarray((1,10), buffer=np.array(hashed), dtype=int)

def get_predicted_language(probs):
    languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]
    max_index = 0
    max_val = -float("inf")

    for index in range(len(probs)):
        if(probs[index] > max_val):
            max_val = probs[index]
            max_index = index
    return (max_val, languages[max_index])

def main():
    language_classifier = model.LanguageClassificationModel()
    data = dataset.LanguageClassificationDataset(language_classifier)
    chars = data.chars
    language_classifier.train(data)

    test_predicted_probs, test_predicted, test_correct = data._predict('test')
    test_accuracy = np.mean(test_predicted == test_correct)
    print("test set accuracy is: {:%}\n".format(test_accuracy))

    while True:
        word = input("Enter a word(press q to quit): ")

        if(word == "q"):
            break
        
        xs = data._encode(word_to_hash(chars, word), None, True)
        result = language_classifier.run(xs)
        probs = data._softmax(result.data)
        max_prob, pred_lang = get_predicted_language(probs[0])
        print("predicted language is: {}, with a confidence of {:%}\n".format(pred_lang, max_prob))

if __name__ == "__main__":
    main()
