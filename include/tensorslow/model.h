#pragma once

#include <memory>

#include "tensorslow/matrix.h"

namespace ts
{
    namespace models
    {
        template <typename T>
        class Model
        {
        public:
            virtual ts::Matrix<T> gradient(const Matrix<T>& theta) const = 0;
            virtual size_t get_features_count() const = 0;
        };

        template <typename T>
        class LinearModel : public Model<T>
        {
        private:
            const ts::Matrix<T>& x;
            const ts::Matrix<T>& y;
            const ts::Matrix<T> x_t;

        public:
            LinearModel(const ts::Matrix<T>& x, const ts::Matrix<T>& y);
            virtual ts::Matrix<T> gradient(const Matrix<T>& theta) const;
            virtual size_t get_features_count() const;
        };

        template <typename T>
        LinearModel<T>::LinearModel(const ts::Matrix<T>& x, const ts::Matrix<T>& y)
            : x(x), y(y), x_t(x.transpose())
        {}

        template <typename T>
        ts::Matrix<T> LinearModel<T>::gradient(const Matrix<T>& theta) const
        {
            return x_t * x * theta - x_t * y;
        }

        template <typename T>
        size_t LinearModel<T>::get_features_count() const
        {
            return x.dim().at(0, 1);
        }
    } // namespace models
} // namespace ts
