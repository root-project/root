// @(#)root/pyroot:$Name:  $:$Id:  $
// Author: Wim Lavrijsen, Apr 2004

#ifndef PYROOT_MEMORYREGULATOR_H
#define PYROOT_MEMORYREGULATOR_H

// ROOT
#include "TObject.h"

// Standard
#include <map>


namespace PyROOT {

/** Communicate object destruction across ROOT/CINT/PyROOT/
      @author  WLAV
      @date    03/29/2004
      @version 1.1
 */

   class MemoryRegulator : public TObject {
   public:
      virtual void RecursiveRemove( TObject* obj );

      static void RegisterObject( PyObject*, void* );
      static PyObject* objectEraseCallback( PyObject* self, PyObject* ref );

   private:
      static std::map< void*, PyObject* > s_objectTable;
   };

} // namespace PyROOT

#endif // !PYROOT_MEMORYREGULATOR_H
