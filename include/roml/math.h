#pragma once

#include <cassert>
#include <iostream>
#include <iterator>
#include <vector>

namespace roml
{
    template <typename T>
    class Matrix
    {
    private:
        size_t rows;
        size_t cols;
        std::vector<T> elements;
        Matrix(const size_t rows, const size_t cols);

    public:
        Matrix(const size_t rows, const size_t cols, std::vector<T> elements);
        Matrix transpose();

        T at(size_t row, size_t col) const;

        Matrix operator+(const Matrix &) const;
        Matrix operator-(const Matrix &) const;
        Matrix dot(const Matrix &) const;

        template <typename U>
        friend std::ostream &operator<<(std::ostream &, const Matrix<U> &);
    };

    template <typename T>
    Matrix<T>::Matrix(const size_t rows, const size_t cols, std::vector<T> elements)
        : rows(rows), cols(cols)
    {
        assert(rows * cols == elements.size());
        this->elements = std::vector(elements);
    }

    template <typename T>
    Matrix<T> Matrix<T>::transpose()
    {
        int swap = this->rows;
        this->rows = this->cols;
        this->cols = swap;

        return *this;
    }

    template <typename T>
    T Matrix<T>::at(size_t row, size_t col) const
    {
        assert(row < this->rows);
        assert(col < this->cols);

        return this->elements[row * this->cols + col];
    }

    template <typename T>
    Matrix<T> Matrix<T>::operator+(const Matrix &other) const
    {
        assert(this->rows == other.rows);
        assert(this->cols == other.cols);

        std::vector<T> sum(this->rows * this->cols);

        for (size_t i = 0; i < this->rows; i++)
        {
            for (size_t j = 0; j < this->cols; j++)
            {
                sum[i * this->cols + j] = this->at(i, j) + other.at(i, j);
            }
        }

        return Matrix(this->rows, this->cols, sum);
    }

    template <typename T>
    Matrix<T> Matrix<T>::operator-(const Matrix &other) const
    {
        assert(this->rows == other.rows);
        assert(this->cols == other.cols);

        std::vector<T> diff(this->rows * this->cols);

        for (size_t i = 0; i < this->rows; i++)
        {
            for (size_t j = 0; j < this->cols; j++)
            {
                diff[i * this->cols + j] = this->at(i, j) - other.at(i, j);
            }
        }

        return Matrix(this->rows, this->cols, diff);
    }

    template <typename T>
    Matrix<T> Matrix<T>::dot(const Matrix &other) const
    {
        assert(this->rows == other.cols);

        std::vector<T> product(this->rows * other.cols, 0);

        for (size_t i = 0; i < this->rows; i++)
        {
            for (size_t j = 0; j < other.cols; j++)
            {
                for (size_t k = 0; k < this->cols; k++)
                {
                    product[i * other.cols + j] += this->at(i, k) * other.at(k, j);
                }
            }
        }

        return Matrix(this->rows, other.cols, product);
    }

    template <typename T>
    std::ostream &operator<<(std::ostream &os, const Matrix<T> &m)
    {
        os << "[";

        for (size_t i = 0; i < m.rows; i++)
        {
            for (size_t j = 0; j < m.cols; j++)
            {
                os << m.at(i, j);

                if (j != m.cols - 1)
                {
                    os << " ";
                }
            }

            if (i != m.rows - 1)
            {
                os << " | ";
            }
        }

        os << "]";

        return os;
    }

} // namespace roml
