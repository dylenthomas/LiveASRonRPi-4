import torch
from tcn import TCN
from dataset import dataset
from torch.utils.data import DataLoader

def predict(model, input, target):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        softmax = torch.nn.Softmax()
        predictions = softmax(predictions)
        predicted_indx = predictions[0].argmax(0)
        predicted = predicted_indx
        expected = target

    return predicted, expected

if __name__ == '__main__':
    model = torch.jit.load('whisper.pt').eval()

    test_set = dataset(root_path='./full_dataset',
                                sample_rate=44100,
                                num_samples=44100
                                )

    dataloader = DataLoader(test_set, batch_size=1, shuffle=True, drop_last=True)
    correct = []

    for input, target in dataloader:
        predicted, expected = predict(model, input, target)
        if predicted == expected:
            correct.append(1)

    accuracy = len(correct) / len(test_set)
    print(f'Total Accuracy: {accuracy}')


