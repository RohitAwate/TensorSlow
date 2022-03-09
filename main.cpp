#include <roml/math.h>

#include <iostream>

#define print(x) std::cout << (x) << std::endl

int main()
{
    auto m1 = roml::Matrix(2, 2, std::vector{1, 2, 3, 4});
    print(m1);

    auto m2 = roml::Matrix(2, 2, std::vector{5, 6, 7, 8});
    print(m2);

    auto m3 = m1 + m2;
    print(m3);

    auto row_mat = roml::Matrix(1, 3, std::vector{1, 2, 3});
    print(row_mat);
    auto col_mat = roml::Matrix(3, 1, std::vector{1, 2, 3});
    print(col_mat);

    print(row_mat.dot(col_mat));
    print(row_mat.transpose());

    print(row_mat.scale(2));

    return 0;
}