#include "Pythonizables.h"


//===========================================================================
MyBufferReturner::MyBufferReturner(int size) : m_size(size) {
    m_Xbuf = new double[size];
    m_Ybuf = new double[size];

    for (int i=0; i<size; ++i) {
        m_Xbuf[i] = 2.0;
        m_Ybuf[i] = 5.0;
    }
}

MyBufferReturner::~MyBufferReturner() {
    delete[] m_Xbuf;
    delete[] m_Ybuf;
}

int MyBufferReturner::GetN() { return m_size; }

double* MyBufferReturner::GetX() { return m_Xbuf; }
double* MyBufferReturner::GetY() { return m_Ybuf; }
