#include <iostream>

#include "tensorslow/matrix.h"

int main()
{
    auto m1 = ts::Matrix(2, 2, std::vector{1, 3, 2, 4});
    print(m1);

    auto m2 = ts::Matrix(2, 1, std::vector{5, 6});
    print(m2);

    auto m3 = m1 * m2;
    print(m3);

    auto row_mat = ts::Matrix(1, 3, std::vector{1, 2, 3});
    print(row_mat);
    auto col_mat = ts::Matrix(3, 1, std::vector{1, 2, 3});
    print(col_mat);

    print(row_mat * col_mat);
    print(row_mat.transpose());

    print(row_mat.scale(2));

    auto col_mat_2 = ts::Matrix(3, 1, std::vector{4, 5, 6});
    print(col_mat.append_cols(col_mat_2));

    return 0;
}