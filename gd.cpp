#include "tensorslow/math.h"

#include <iostream>

#define print(x) std::cout << (x) << std::endl

int main()
{
    auto x = ts::Matrix(7, 2, std::vector<float>{
                                  1.7,
                                  1.5,
                                  2.8,
                                  5,
                                  1.3,
                                  2.2,
                                  1.3,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                              });
    auto x_t = x.transpose();
    auto y = ts::Matrix(1, 7, std::vector<float>{368, 340, 665, 954, 331, 556, 376});
    auto theta = ts::Matrix(2, 1, std::vector<float>{0.0, 0.0});
    float learning_rate = 0.01;
    auto product = x_t * y * theta;

    auto gradient = x_t * x * theta - x_t * y;

    while (true)
    {
        auto l2 = gradient.l2();
        if (std::isnan(l2) || l2 <= 10e-2)
        {
            break;
        }

        theta = theta - gradient.scale(learning_rate);
        gradient = x_t * x * theta - x_t * y;
    }

    print(theta);

    return 0;
}
