// @(#)root/pyroot:$Name:  $:$Id: PyBufferFactory.h,v 1.5 2005/03/04 07:44:11 brun Exp $
// Author: Wim Lavrijsen, Apr 2004

#ifndef PYROOT_PYBUFFERFACTORY_H
#define PYROOT_PYBUFFERFACTORY_H

// ROOT
#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif


namespace PyROOT {

/** Factory for python buffers of non-string type
      @author  WLAV
      @date    10/28/2004
      @version 1.5
*/

class PyBufferFactory {
public:
   static PyBufferFactory* Instance();

   PyObject* PyBuffer_FromMemory( Short_t* buf, int size = -1 );
   PyObject* PyBuffer_FromMemory( Short_t* buf, PyObject* sizeCallback );
   PyObject* PyBuffer_FromMemory( UShort_t* buf, int size = -1 );
   PyObject* PyBuffer_FromMemory( UShort_t* buf, PyObject* sizeCallback );
   PyObject* PyBuffer_FromMemory( Int_t* buf, int size = -1 );
   PyObject* PyBuffer_FromMemory( Int_t* buf, PyObject* sizeCallback );
   PyObject* PyBuffer_FromMemory( UInt_t* buf, int size = -1 );
   PyObject* PyBuffer_FromMemory( UInt_t* buf, PyObject* sizeCallback );
   PyObject* PyBuffer_FromMemory( Long_t* buf, int size = -1 );
   PyObject* PyBuffer_FromMemory( Long_t* buf, PyObject* sizeCallback );
   PyObject* PyBuffer_FromMemory( ULong_t* buf, int size = -1 );
   PyObject* PyBuffer_FromMemory( ULong_t* buf, PyObject* sizeCallback );
   PyObject* PyBuffer_FromMemory( Float_t* buf, int size = -1 );
   PyObject* PyBuffer_FromMemory( Float_t* buf, PyObject* sizeCallback );
   PyObject* PyBuffer_FromMemory( Double_t* buf, int size = -1 );
   PyObject* PyBuffer_FromMemory( Double_t* buf, PyObject* sizeCallback );

protected:
   PyBufferFactory();
   ~PyBufferFactory();
};

typedef PyBufferFactory BufFac_t;

} // namespace PyROOT


#endif // !PYROOT_PYBUFFERFACTORY_H
