// @(#)root/pyroot:$Name:  $:$Id: MemoryRegulator.h,v 1.4 2004/07/30 06:31:18 brun Exp $
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
      @date    11/23/2004
      @version 1.3
 */

   class MemoryRegulator : public TObject {
   public:
      MemoryRegulator();
      virtual void RecursiveRemove( TObject* obj );

   // add an object to the table of managed objects
      static void RegisterObject( PyObject*, TObject* );

   // new reference to python object corresponding to ptr, or 0 on failure
      static PyObject* RetrieveObject( TObject* ptr );

   // callback when weak refs to managed objects are destroyed
      static PyObject* objectEraseCallback( PyObject* self, PyObject* ref );

   private:
      typedef std::map< TObject*, PyObject* > objmap_t;
      static objmap_t s_objectTable;
   };

} // namespace PyROOT

#endif // !PYROOT_MEMORYREGULATOR_H
