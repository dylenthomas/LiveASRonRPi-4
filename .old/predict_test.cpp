#include <torch/script.h>

torch::jit::script::Module model;
std::vector<torch::jit::IValue> input_tensor;

int predict(std::vector<torch::jit::IValue> input) {
    {
        torch::NoGradGuard no_grad; //disable gradients
        model.eval(); //turn off training layers
        torch::Tensor ouput = model.forward(input).toTensor();
        return ouput[0].argmax(-1).item<int>();
    }
}

int main(int argc, const char* argv[]) {
    if (argc != 2){
        std::cerr << "usage: predict_test <path/to/model>" << std::endl;
        return -1;
    }

    model = torch::jit::load(argv[1]);
    input_tensor.push_back(torch::randn({1, 64, 91}));

    int guess = predict(input_tensor);
    std::cout << guess << std::endl;

    return 0;
}