// @(#)root/pyroot:$Name:  $:$Id: PyBufferFactory.h,v 1.3 2004/08/13 06:02:40 brun Exp $
// Author: Wim Lavrijsen, Apr 2004

#ifndef PYROOT_PYBUFFERFACTORY_H
#define PYROOT_PYBUFFERFACTORY_H


namespace PyROOT {

/** Factory for python buffers of non-string type
      @author  WLAV
      @date    10/28/2004
      @version 1.5
*/

class PyBufferFactory {
public:
   static PyBufferFactory* getInstance();

   PyObject* PyBuffer_FromMemory( long* buf, int size = -1 );
   PyObject* PyBuffer_FromMemory( long* buf, PyObject* sizeCallback );
   PyObject* PyBuffer_FromMemory( int* buf, int size = -1 );
   PyObject* PyBuffer_FromMemory( int* buf, PyObject* sizeCallback );
   PyObject* PyBuffer_FromMemory( double* buf, int size = -1 );
   PyObject* PyBuffer_FromMemory( double* buf, PyObject* sizeCallback );
   PyObject* PyBuffer_FromMemory( float* buf, int size = -1 );
   PyObject* PyBuffer_FromMemory( float* buf, PyObject* sizeCallback );

protected:
   PyBufferFactory();
   ~PyBufferFactory();
};

} // namespace PyROOT


#endif // !PYROOT_PYBUFFERFACTORY_H
