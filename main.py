import tensorslow as slow

if __name__ == "__main__":
    a = slow.tensor([1, 2, 3])
    b = slow.tensor([4, 5, 6])
    c = a + b

    d = slow.tensor([7, 8, 9])
    e = c * d

    f = slow.tensor([10, 11, 12])
    L = e / f

    print("L", L)
    print("f", f)
    print("e", e)
    print("d", d)
    print("c", c)
    print("b", b)
    print("a", a)
    print()

    L.backward()

    print("L_grad", L.grad)
    print("f_grad", f.grad)
    print("e_grad", e.grad)
    print("d_grad", d.grad)
    print("c_grad", c.grad)
    print("b_grad", b.grad)
    print("a_grad", a.grad)
    print()
