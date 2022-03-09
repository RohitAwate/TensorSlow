#pragma once

#include <cassert>
#include <iostream>
#include <iterator>

namespace roml
{
    template <typename T>
    class Matrix
    {
    private:
        size_t rows;
        size_t cols;
        T **elements;
        Matrix(const size_t rows, const size_t cols);

    public:
        Matrix(const size_t rows, const size_t cols, T **elements);
        void transpose();
        Matrix operator+(const Matrix &) const;
        Matrix operator-(const Matrix &) const;
        Matrix operator*(const Matrix &) const;
        Matrix &operator[](int i) const;

        template <typename U>
        friend std::ostream &operator<<(std::ostream &, const Matrix<U> &);
    };

    template <typename T>
    Matrix<T>::Matrix(const size_t rows, const size_t cols, T **elements)
        : rows(rows), cols(cols)
    {
        // assert(this->rows == std::size(elements));
        // assert(this->rows == std::size(elements[0]));
        this->elements = elements;
    }

    template <typename T>
    Matrix<T> Matrix<T>::operator+(const Matrix &other) const
    {
        assert(this->rows == other.rows);
        assert(this->cols == other.cols);

        T **sum = new T *[rows];

        for (size_t i = 0; i < this->rows; i++)
        {
            sum[i] = new T[this->cols];

            for (size_t j = 0; j < this->cols; j++)
            {
                sum[i][j] = this->elements[i][j] + other.elements[i][j];
            }
        }

        return *new Matrix(this->rows, this->cols, sum);
    }

    template <typename T>
    Matrix<T> Matrix<T>::operator-(const Matrix &other) const
    {
        assert(this->rows == other.rows);
        assert(this->cols == other.cols);

        T **sum = new T *[rows];

        for (size_t i = 0; i < this->rows; i++)
        {
            sum[i] = new T[this->cols];

            for (size_t j = 0; j < this->cols; j++)
            {
                sum[i][j] = this->elements[i][j] - other.elements[i][j];
            }
        }

        return Matrix(this->rows, this->cols, sum);
    }

    template <typename T>
    Matrix<T> Matrix<T>::operator*(const Matrix &other) const
    {
        assert(this->rows == other.cols);

        T **sum = new T *[rows];

        for (size_t i = 0; i < this->rows; i++)
        {
            sum[i] = new T[this->cols];

            for (size_t j = 0; j < this->cols; j++)
            {
                sum[i][j] = this->elements[i][j] + other.elements[i][j];
            }
        }

        return Matrix(this->rows, this->cols, sum);
    }

    template <typename T>
    std::ostream &operator<<(std::ostream &os, const Matrix<T> &m)
    {
        os << std::printf("<Matrix %zux%zu>[", m.rows, m.cols);

        for (size_t i = 0; i < m.rows; i++)
        {
            for (size_t j = 0; j < m.cols; j++)
            {
                os << m.elements[i][j];

                if (j != m.cols - 1)
                {
                    os << " ";
                }
            }

            if (i != m.rows - 1)
                os << " | ";
        }

        os << "]";

        return os;
    }

} // namespace roml
