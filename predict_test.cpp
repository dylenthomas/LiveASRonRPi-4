#include <torch/script.h>

torch::jit::script::Module model = torch::jit::load("/Users/dylenthomas/Documents/whisper/.model/whisper.pt");

int predict(std::vector<torch::jit::IValue> input) {
    {
        torch::NoGradGuard no_grad; //disable gradients
        model.eval(); //turn off training layers
        torch::Tensor ouput = model.forward(input).toTensor();
        return ouput[0].argmax(-1).item<int>();
    }
}

int main() {
    std::vector<torch::jit::IValue> input_tensor;
    input_tensor.push_back(torch::randn({64, 91}));

    int guess = predict(input_tensor);
    std::cout << guess << std::endl;

    return 0;
}