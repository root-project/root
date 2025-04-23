#include "Pythonizables.h"


//===========================================================================
pythonizables::MyBufferReturner::MyBufferReturner(int size, double valx, double valy) : m_size(size) {
    m_Xbuf = new double[size];
    m_Ybuf = new double[size];

    for (int i=0; i<size; ++i) {
        m_Xbuf[i] = valx*i;
        m_Ybuf[i] = valy*i;
    }
}

pythonizables::MyBufferReturner::~MyBufferReturner() {
    delete[] m_Xbuf;
    delete[] m_Ybuf;
}

int pythonizables::MyBufferReturner::GetN() { return m_size; }

double* pythonizables::MyBufferReturner::GetX() { return m_Xbuf; }
double* pythonizables::MyBufferReturner::GetY() { return m_Ybuf; }


//===========================================================================
pythonizables::MyBase::~MyBase() {}
pythonizables::MyDerived::~MyDerived() {}

pythonizables::MyBase* pythonizables::GimeDerived() {
   return new MyDerived();
}
