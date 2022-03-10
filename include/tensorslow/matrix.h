#pragma once

#include "tensorslow/util.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <iterator>
#include <vector>

namespace ts
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
        /**
         * elements should be in row-major order
         */
        Matrix();
        Matrix(const size_t rows, const size_t cols, std::vector<T> elements);
        Matrix(const Matrix &copy);
        Matrix transpose() const;

        T at(size_t row, size_t col) const;
        Matrix<size_t> dim() const;

        Matrix &scale(const T scalar);
        double l2() const;

        Matrix operator+(const Matrix &) const;
        Matrix operator-(const Matrix &) const;
        Matrix operator*(const Matrix &) const;

        template <typename U>
        friend std::ostream &operator<<(std::ostream &, const Matrix<U> &);
    };

    template <typename T>
    Matrix<T>::Matrix() : rows(0), cols(0)
    {
    }

    template <typename T>
    Matrix<T>::Matrix(const size_t rows, const size_t cols, std::vector<T> elements)
        : rows(rows), cols(cols)
    {
        assert(rows * cols == elements.size());
        this->elements = std::vector(elements);
    }

    template <typename T>
    Matrix<T>::Matrix(const Matrix<T> &copy)
    {
        this->rows = rows;
        this->cols = cols;
        this->elements = elements;
    }

    template <typename T>
    Matrix<T> Matrix<T>::transpose() const
    {
        std::vector<T> trans_elements(this->rows * this->cols);

        for (size_t i = 0; i < this->rows; i++)
        {
            for (size_t j = 0; j < this->cols; j++)
            {
                trans_elements[j * this->rows + i] = this->at(i, j);
            }
        }

        return Matrix(this->cols, this->rows, trans_elements);
    }

    template <typename T>
    T Matrix<T>::at(size_t row, size_t col) const
    {
        assert(row < this->rows);
        assert(col < this->cols);

        return this->elements[row * this->cols + col];
    }

    template <typename T>
    Matrix<size_t> Matrix<T>::dim() const
    {
        return Matrix<size_t>(1, 2, std::vector<size_t>{this->rows, this->cols});
    }

    template <typename T>
    Matrix<T> &Matrix<T>::scale(const T scalar)
    {
        for (auto &i : this->elements)
        {
            i = scalar * i;
        }

        return *this;
    }

    template <typename T>
    double Matrix<T>::l2() const
    {
        assert(this->rows == 1 || this->cols == 1);

        double l2_val = 0.0;

        for (size_t i = 0; i < this->rows; i++)
        {
            for (size_t j = 0; j < this->cols; j++)
            {
                l2_val += pow(this->at(i, j), 2);
            }
        }

        return pow(l2_val, 0.5);
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
    Matrix<T> Matrix<T>::operator*(const Matrix &other) const
    {
        assert(this->cols == other.rows);

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

} // namespace ts
