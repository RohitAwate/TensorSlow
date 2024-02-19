#pragma once

#include "tensorslow/model.h"

namespace ts
{
    namespace optimizers
    {
        template <typename T>
        Matrix<T> gradient_descent(const ts::models::Model<T> &model, float learning_rate, float threshold)
        {
            auto features_count = model.get_features_count();
            auto params = ts::Matrix<T>(features_count, 1, std::vector<float>(features_count, 0.0f));

            while (true)
            {
                auto gradient = model.gradient(params);
                auto l2 = gradient.l2();
                if (std::isnan(l2) || std::isinf(l2) || l2 <= 10e-2f)
                    return params;

                params = params - gradient.scale(learning_rate);
            }
        }
    } // namespace optimizers
} // namespace ts