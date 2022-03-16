# TODO

- Add new constructor to tf::Matrix which takes in just one dimension and elements. This should return a column vector.
- inherit from matrix called vector with default cols of 1. make columnn vectors by default
- have a separate variable for bias instead of having it included in the data and weight
- rename theta to weights or parameters or coefficients. something less cryptic
- how to incorporate basis function expansions?
- overload == operator for Matrix
- make it easy to toggle between row and column major representations internally
    - this might help operations such as append column/row