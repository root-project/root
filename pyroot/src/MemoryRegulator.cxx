// @(#)root/pyroot:$Name:  $:$Id: MemoryRegulator.cxx,v 1.4 2004/05/07 20:47:20 brun Exp $
// Author: Wim Lavrijsen, Apr 2004

// Bindings
#include "PyROOT.h"
#include "MemoryRegulator.h"
#include "Utility.h"
#include "ObjectHolder.h"

// Standard
#include <string.h>
#include <iostream>


//- static data -----------------------------------------------------------------
PyROOT::MemoryRegulator::objmap_t PyROOT::MemoryRegulator::s_objectTable;


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

         memset( &PyROOT_NoneType, 0, sizeof( PyROOT_NoneType ) );

         PyROOT_NoneType.ob_type   = &PyType_Type;
         PyROOT_NoneType.ob_size   = 0;
         PyROOT_NoneType.ob_refcnt = 1;

         PyROOT_NoneType.tp_name        = const_cast< char* >( "PyROOT_NoneType" );
         PyROOT_NoneType.tp_flags       = Py_TPFLAGS_HAVE_RICHCOMPARE | Py_TPFLAGS_HAVE_GC;

         PyROOT_NoneType.tp_traverse    = (traverseproc) 0;
         PyROOT_NoneType.tp_clear       = (inquiry) 0;
         PyROOT_NoneType.tp_dealloc     = (destructor)  &Init_PyROOT_NoneType::dealloc;
         PyROOT_NoneType.tp_repr        = Py_None->ob_type->tp_repr;
         PyROOT_NoneType.tp_richcompare = (richcmpfunc) &Init_PyROOT_NoneType::richcompare;
         PyROOT_NoneType.tp_compare     = (cmpfunc) &Init_PyROOT_NoneType::compare;
         PyROOT_NoneType.tp_hash        = (hashfunc) &Init_PyROOT_NoneType::ptrhash;

         PyROOT_NoneType.tp_as_mapping  = &PyROOT_NoneType_mapping;

         PyType_Ready( &PyROOT_NoneType );
      }

      static void dealloc( PyObject* obj ) {
         obj->ob_type->tp_free( obj );
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

   };

} // unnamed namespace


//- constructor -----------------------------------------------------------------
PyROOT::MemoryRegulator::MemoryRegulator() {
   static Init_PyROOT_NoneType init_PyROOT_NoneType;
}


//- public members --------------------------------------------------------------
void PyROOT::MemoryRegulator::RecursiveRemove( TObject* obj ) {
   if ( obj == 0 || s_objectTable.size() == 0 )   // table can be deleted before libCore is done
      return;

// see whether we're tracking this object
   objmap_t::iterator ppo = s_objectTable.find( obj );

   if ( ppo != s_objectTable.end() ) {
   // nullify the object
      PyObject* pyobj = PyWeakref_GetObject( ppo->second );
      if ( ! pyobj )
         return;

      if ( pyobj != Py_None ) {
         if ( ! PyROOT_NoneType.tp_traverse ) {
         // take a reference as we're copying its function pointers
            Py_INCREF( pyobj->ob_type );

         // all object that arrive here are expected to be of the same type ("instance")
            PyROOT_NoneType.tp_traverse   = pyobj->ob_type->tp_traverse;
            PyROOT_NoneType.tp_clear      = pyobj->ob_type->tp_clear;
            PyROOT_NoneType.tp_free       = pyobj->ob_type->tp_free;
         }
         else if ( PyROOT_NoneType.tp_traverse != pyobj->ob_type->tp_traverse ) {
            std::cerr << "in PyROOT::MemoryRegulater, unexpected object of type: "
                      << pyobj->ob_type->tp_name << std::endl;

         // leave before too much damage is done
            return;
         }

      // reset type object
         Py_INCREF( &PyROOT_NoneType );
         Py_DECREF( pyobj->ob_type );
         pyobj->ob_type = &PyROOT_NoneType;
      }

   // clean up the weak reference.
      Py_DECREF( ppo->second );

   // erase the object from tracking
      s_objectTable.erase( ppo );
   }
}


void PyROOT::MemoryRegulator::RegisterObject( PyObject* obj, void* ptr ) {
   if ( obj == 0 && ptr == 0 )
      return;

   objmap_t::iterator ppo = s_objectTable.find( ptr );
   if ( ppo != s_objectTable.end() )
      Py_INCREF( ppo->second );
   else
      s_objectTable[ ptr ] = PyWeakref_NewRef( obj, g_objectEraseCallback );
}


PyObject* PyROOT::MemoryRegulator::RetrieveObject( void* ptr ) {
   objmap_t::iterator ppo = s_objectTable.find( ptr );
   if ( ppo != s_objectTable.end() ) {
      PyObject* pyobj = PyWeakref_GetObject( ppo->second );
      Py_XINCREF( pyobj );
      return pyobj;
   }

   return 0;
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
         objmap_t::iterator ppo = s_objectTable.find( rootobj );
         if ( ppo != s_objectTable.end() ) {
         // cleanup weak reference, and table entry
            Py_DECREF( ppo->second );
            s_objectTable.erase( ppo );
         }
      }
   }
   else {
      objmap_t::iterator ppo = s_objectTable.begin();
      for ( ; ppo != s_objectTable.end(); ++ppo ) {
         if ( ppo->second == ref )
            break;
      }

      if ( ppo != s_objectTable.end() )
         s_objectTable.erase( ppo );
   }

   Py_INCREF( Py_None );
   return Py_None;
}
