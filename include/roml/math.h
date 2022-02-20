#pragma once

#include <iostream>

namespace roml
{
    template <typename T>
    class Primitive
    {
        virtual void transpose() = 0;
        virtual Primitive<T> operator+(const Primitive &) = 0;
        virtual Primitive<T> operator-(const Primitive &) = 0;
        virtual Primitive<T> operator*(const Primitive &) = 0;
        virtual Primitive<T> &operator[](int i) = 0;
    };

    template <typename T>
    class Vector : public Primitive
    {
    private:
        Vector dot(const Vector &);
        Vector cross(const Vector &);

    public:
        Vector(const T *elements, bool columnar = true);
        Vector(size_t dimensions, bool columnar = true);

        friend std::ostream &operator<<(std::ostream &, const Vector &);
    };

    template <typename T>
    class Matrix : public Primitive
    {
        Vector(const T **elements);
        Vector(const Vector<T> *rows);

        friend std::ostream &operator<<(std::ostream &, const Matrix &);
    };
} // namespace roml
