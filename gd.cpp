#include <iostream>

#include "tensorslow/model.h"
#include "tensorslow/optimizer.h"

int main()
{
    auto x = ts::Matrix(7, 1, std::vector<float>{1.7, 1.5, 2.8, 5, 1.3, 2.2, 1.3});
    auto y = ts::Matrix(7, 1, std::vector<float>{368, 340, 665, 954, 331, 556, 376});
    auto learning_rate = 0.01f;
    auto threshold = 10e-2f;

    auto model = ts::models::LinearModel(x, y);
    auto theta = ts::optimizers::gradient_descent(model, learning_rate, threshold);
    print(theta);

    return 0;
}
