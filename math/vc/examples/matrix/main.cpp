/*  This file is part of the Vc project
    Copyright (C) 2009-2010 Matthias Kretz <kretz@kde.org>

    Permission to use, copy, modify, and distribute this software
    and its documentation for any purpose and without fee is hereby
    granted, provided that the above copyright notice appear in all
    copies and that both that the copyright notice and this
    permission notice and warranty disclaimer appear in supporting
    documentation, and that the name of the author not be used in
    advertising or publicity pertaining to distribution of the
    software without specific, written prior permission.

    The author disclaim all warranties with regard to this
    software, including all implied warranties of merchantability
    and fitness.  In no event shall the author be liable for any
    special, indirect or consequential damages or any damages
    whatsoever resulting from loss of use, data or profits, whether
    in an action of contract, negligence or other tortious action,
    arising out of or in connection with the use or performance of
    this software.

*/

#include <Vc/Vc>
#include <Vc/IO>
#include <iostream>
#include <iomanip>

template<typename T, unsigned int Size> class Matrix;
template<typename T, unsigned int Size> std::ostream &operator<<(std::ostream &, const Matrix<T, Size> &);

template<typename T, unsigned int Size>
class Matrix
{
    friend std::ostream &operator<< <>(std::ostream &, const Matrix<T, Size> &);
    private:
        typedef Vc::Vector<T> V;
        Vc::Memory<V, Size * Size> m_mem;
    public:
        Matrix &operator=(const T &val) {
            V vec(val);
            for (unsigned int i = 0; i < m_mem.vectorsCount(); ++i) {
                m_mem.vector(i) = vec;
            }
            return *this;
        }

        Matrix &operator+=(const Matrix &rhs) {
            for (unsigned int i = 0; i < m_mem.vectorsCount(); ++i) {
                V v1(m_mem.vector(i));
                v1 += V(rhs.m_mem.vector(i));
                m_mem.vector(i) = v1;
            }
            return *this;
        }
};

template<typename T, unsigned int Size>
std::ostream &operator<<(std::ostream &out, const Matrix<T, Size> &m)
{
    for (unsigned int i = 0; i < Size; ++i) {
        std::cout << "[" << std::setw(6) << m.m_mem[i * Size];
        for (unsigned int j = 1; j < Size; ++j) {
            std::cout << std::setw(6) << m.m_mem[i * Size + j];
        }
        std::cout << " ]\n";
    }
    return out;
}

int main()
{
    Matrix<float, 15> m1;
    m1 = 1.f;
    Matrix<float, 15> m2;
    m2 = 2.f;
    m1 += m2;
    std::cout << m1 << std::endl;
    return 0;
}
