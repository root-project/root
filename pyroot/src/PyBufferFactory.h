// @(#)root/pyroot:$Name:  $:$Id:  $
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

   PyObject* PyBuffer_FromMemory( long* buf, int size );
   PyObject* PyBuffer_FromMemory( int* buf, int size );
   PyObject* PyBuffer_FromMemory( double* buf, int size );
   PyObject* PyBuffer_FromMemory( float* buf, int size );

protected:
   PyBufferFactory();
   ~PyBufferFactory();
};

} // namespace PyROOT


#endif // !PYROOT_PYBUFFERFACTORY_H
