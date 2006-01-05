// @(#)root/pyroot:$Name:  $:$Id: MemoryRegulator.h,v 1.7 2005/09/09 05:19:10 brun Exp $
// Author: Wim Lavrijsen, Apr 2004

#ifndef PYROOT_TMEMORYREGULATOR_H
#define PYROOT_TMEMORYREGULATOR_H

// ROOT
#include "TObject.h"

// Standard
#include <map>


namespace PyROOT {

/** Communicate object destruction across ROOT/CINT/PyROOT/
      @author  WLAV
      @date    11/23/2004
      @version 2.1
 */

   class ObjectProxy;

   class TMemoryRegulator : public TObject {
   public:
      TMemoryRegulator();
      ~TMemoryRegulator();

   // callback for ROOT/CINT
      virtual void RecursiveRemove( TObject* object );

   // add a python object to the table of managed objects
      static void RegisterObject( ObjectProxy* pyobj, TObject* object );

   // new reference to python object corresponding to object, or 0 on failure
      static PyObject* RetrieveObject( TObject* object );

   // callback when weak refs to managed objects are destroyed
      static PyObject* ObjectEraseCallback( PyObject*, PyObject* pyref );

   private:
      typedef std::map< TObject*, PyObject* > ObjectMap_t;
      static ObjectMap_t* fgObjectTable;
   };

} // namespace PyROOT

#endif // !PYROOT_TMEMORYREGULATOR_H
