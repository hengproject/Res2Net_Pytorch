# How to call Pytorch's API using C++

## Batch Normalization

```cpp
#include <torch/torch.h>

int main() {
    torch::nn::BatchNorm2d batch_norm(/*num_features=*/20);
    torch::Tensor input = torch::randn({10, 20, 35, 45});
    torch::Tensor output = batch_norm->forward(input);
}
```

## Convolutional
```cpp
#include <torch/torch.h>

int main() {
    torch::nn::Conv2d conv2d(torch::nn::Conv2dOptions(/*in_channels=*/3, /*out_channels=*/32, /*kernel_size=*/5));
    torch::Tensor input = torch::randn({1, 3, 64, 64});
    torch::Tensor output = conv2d->forward(input);
}
```

## RELU
```cpp
#include <torch/torch.h>

int main() {
    torch::nn::ReLU relu;
    torch::Tensor input = torch::randn({10, 20});
    torch::Tensor output = relu->forward(input);
}
```

## Concat
```cpp
#include <torch/torch.h>

int main() {
    torch::Tensor tensor1 = torch::randn({10, 20});
    torch::Tensor tensor2 = torch::randn({10, 20});

    // Concatenation along a new dimension
    torch::Tensor concatenated_tensor = torch::cat({tensor1, tensor2}, 0);
}
```