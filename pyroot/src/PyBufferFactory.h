// @(#)root/pyroot:$Name:  $:$Id: PyBufferFactory.h,v 1.2 2004/05/07 20:47:20 brun Exp $
// Author: Wim Lavrijsen, Apr 2004

#ifndef PYROOT_PYBUFFERFACTORY_H
#define PYROOT_PYBUFFERFACTORY_H


namespace PyROOT {

/** Factory for python buffers of non-string type
      @author  WLAV
      @date    08/05/2003
      @version 1.4
*/

class PyBufferFactory {
public:
   static PyBufferFactory* getInstance();

   PyObject* PyBuffer_FromMemory( long* buf, int size = -1 );
   PyObject* PyBuffer_FromMemory( int* buf, int size = -1 );
   PyObject* PyBuffer_FromMemory( double* buf, int size = -1 );
   PyObject* PyBuffer_FromMemory( float* buf, int size = -1 );

protected:
   PyBufferFactory();
   ~PyBufferFactory();
};

} // namespace PyROOT


#endif // !PYROOT_PYBUFFERFACTORY_H
