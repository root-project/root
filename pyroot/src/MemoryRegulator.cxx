// @(#)root/pyroot:$Name:  $:$Id:  $
// Author: Wim Lavrijsen, Apr 2004

// Bindings
#include "PyROOT.h"
#include "MemoryRegulator.h"
#include "Utility.h"
#include "ObjectHolder.h"

// Standard
#include <cstring>
#include <iostream>


//- static data -----------------------------------------------------------------
std::map< void*, PyObject* > PyROOT::MemoryRegulator::s_objectTable;


namespace {

// memory regulater callback for deletion of registered objects
   PyMethodDef methoddef_ = {
      const_cast< char* >( "MemoryRegulator_internal_objectEraseCallback" ),
      (PyCFunction) PyROOT::MemoryRegulator::objectEraseCallback,
      METH_O,
      NULL
   };

   PyObject* g_objectEraseCallback = PyCFunction_New( &methoddef_, NULL );


// pseudo-None type for masking out objects on the python side
   PyTypeObject PyROOT_NoneType;

   int AlwaysNullLength( PyObject* ) {
      return 0;
   }

   PyMappingMethods PyROOT_NoneType_mapping = {
        (inquiry) AlwaysNullLength,
        (binaryfunc)             0,
        (objobjargproc)          0
   };

   struct Init_PyROOT_NoneType {
      Init_PyROOT_NoneType() {
         std::memset( &PyROOT_NoneType, 0, sizeof( PyROOT_NoneType ) );

         PyROOT_NoneType.ob_type   = &PyType_Type;
         PyROOT_NoneType.ob_size   = 0;
         PyROOT_NoneType.ob_refcnt = 1;

         PyROOT_NoneType.tp_name        = const_cast< char* >( "PyROOT_NoneType" );
         PyROOT_NoneType.tp_flags       = Py_TPFLAGS_HAVE_RICHCOMPARE | Py_TPFLAGS_HAVE_GC;

         PyROOT_NoneType.tp_traverse    = (traverseproc) 0;
         PyROOT_NoneType.tp_clear       = (inquiry) 0;
         PyROOT_NoneType.tp_dealloc     = (destructor)  0;
         PyROOT_NoneType.tp_repr        = Py_None->ob_type->tp_repr;
         PyROOT_NoneType.tp_richcompare = (richcmpfunc) &Init_PyROOT_NoneType::richcompare;
         PyROOT_NoneType.tp_compare     = (cmpfunc) &Init_PyROOT_NoneType::compare;
         PyROOT_NoneType.tp_hash        = (hashfunc) &Init_PyROOT_NoneType::ptrhash;

         PyROOT_NoneType.tp_as_mapping  = &PyROOT_NoneType_mapping;

         PyType_Ready( &PyROOT_NoneType );
      }

      static PyObject* richcompare( PyObject*, PyObject* other, int opid ) {
         return PyObject_RichCompare( other, Py_None, opid );
      }

      static int compare( PyObject*, PyObject* other ) {
         return PyObject_Compare( other, Py_None );
      }

      static int ptrhash( PyObject* obj ) {
         return (int) obj;
      }

   } init_PyROOT_NoneType;

}


//- public members --------------------------------------------------------------
void PyROOT::MemoryRegulator::RecursiveRemove( TObject* obj ) {
   if ( obj == 0 )
      return;
   if ( s_objectTable.size() == 0 )  // It looks that the table can be deleted before the libCore is done
      return;

// see whether we're tracking this object
   std::map< void*, PyObject* >::iterator ppo = s_objectTable.find( obj );

   if ( ppo != s_objectTable.end() ) {
   // nullify the object
      PyObject* pyobj = PyWeakref_GetObject( ppo->second );
      if ( ! pyobj )
         return;

      if ( pyobj != Py_None ) {
         if ( ! PyROOT_NoneType.tp_traverse ) {
         // all object that arrive here are expected to be of the same type ("instance")
            PyROOT_NoneType.tp_traverse   = pyobj->ob_type->tp_traverse;
            PyROOT_NoneType.tp_clear      = pyobj->ob_type->tp_clear;
         }
         else if ( PyROOT_NoneType.tp_traverse != pyobj->ob_type->tp_traverse ) {
            std::cerr << "in PyROOT::MemoryRegulater, unexpected object of type: "
                      << pyobj->ob_type->tp_name << std::endl;

         // leave before too much damage is done
            return;
         }

         Py_INCREF( &PyROOT_NoneType );
         pyobj->ob_type = &PyROOT_NoneType;       // don't decref old ob_type

      // prevent the object from ever being deleted ...
         Py_INCREF( pyobj );
      }

   // clean up the weak reference.
      Py_DECREF( ppo->second );

   // erase the object from tracking
      s_objectTable.erase( ppo );                 // don't decref ppo->second
   }
}


void PyROOT::MemoryRegulator::RegisterObject( PyObject* obj, void* ptr ) {
   if ( obj == 0 && ptr == 0 )
      return;

   s_objectTable[ ptr ] = PyWeakref_NewRef( obj, g_objectEraseCallback );
}


//- private static members ------------------------------------------------------
PyObject* PyROOT::MemoryRegulator::objectEraseCallback( PyObject* /* self */, PyObject* ref ) {
// called when one of the python objects we've registered is going away
   PyObject* pyobj = PyWeakref_GetObject( ref );

   if ( pyobj != 0 && pyobj != Py_None ) {
   // get holder if root object
      PyROOT::ObjectHolder* holder = Utility::getObjectHolder( pyobj );
      void* rootobj = holder ? holder->getObject() : 0;

      if ( rootobj != 0 ) {
      // erase if tracked
         std::map< void*, PyObject* >::iterator ppo = s_objectTable.find( rootobj );
         if ( ppo != s_objectTable.end() ) {
            Py_DECREF( ppo->second );
            s_objectTable.erase( ppo );
         }
      }
  }

  Py_INCREF( Py_None );
  return Py_None;
}
