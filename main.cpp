#include <roml/math.h>

#include <iostream>

#define print(x) std::cout << (x) << std::endl

int main()
{
    size_t rows = 2;
    size_t cols = 2;

    int **arr1 = new int *[rows];
    for (int i = 0; i < rows; i++)
    {
        arr1[i] = new int[cols];

        for (int j = 0; j < cols; j++)
        {
            arr1[i][j] = i + 1;
        }
    }

    auto m1 = roml::Matrix(rows, cols, arr1);

    print(m1);

    int **arr2 = new int *[rows];
    for (int i = 0; i < rows; i++)
    {
        arr2[i] = new int[cols];

        for (int j = 0; j < cols; j++)
        {
            arr2[i][j] = i + 2 * j;
        }
    }

    auto m2 = roml::Matrix(rows, cols, arr2);
    print(m2);

    auto m3 = m1 * m2;
    print(m3);
    return 0;
}