// @(#)root/pyroot:$Id$
// Author: Wim Lavrijsen, Apr 2004

// Bindings
#include "PyROOT.h"
#include "MemoryRegulator.h"
#include "ObjectProxy.h"

// ROOT
#include "TObject.h"
#include "TClass.h"

// Standard
#include <assert.h>
#include <string.h>
#include <Riostream.h>


//- static data --------------------------------------------------------------
PyROOT::TMemoryRegulator::ObjectMap_t*  PyROOT::TMemoryRegulator::fgObjectTable  = 0;
PyROOT::TMemoryRegulator::WeakRefMap_t* PyROOT::TMemoryRegulator::fgWeakRefTable = 0;


namespace {

// memory regulater callback for deletion of registered objects
   PyMethodDef methoddef_ = {
      const_cast< char* >( "TMemoryRegulator_internal_ObjectEraseCallback" ),
      (PyCFunction) PyROOT::TMemoryRegulator::ObjectEraseCallback,
      METH_O,
      NULL
   };

   PyObject* gObjectEraseCallback = PyCFunction_New( &methoddef_, NULL );


// pseudo-None type for masking out objects on the python side
   PyTypeObject PyROOT_NoneType;

//____________________________________________________________________________
   Py_ssize_t AlwaysNullLength( PyObject* )
   {
      return 0;
   }

//____________________________________________________________________________
   PyMappingMethods PyROOT_NoneType_mapping = {
        AlwaysNullLength,
        (binaryfunc)             0,
        (objobjargproc)          0
   };

//____________________________________________________________________________
   struct InitPyROOT_NoneType_t {
      InitPyROOT_NoneType_t()
      {
         memset( &PyROOT_NoneType, 0, sizeof( PyROOT_NoneType ) );

         ((PyObject&)PyROOT_NoneType).ob_type    = &PyType_Type;
         ((PyObject&)PyROOT_NoneType).ob_refcnt  = 1;
         ((PyVarObject&)PyROOT_NoneType).ob_size = 0;

         PyROOT_NoneType.tp_name        = const_cast< char* >( "PyROOT_NoneType" );
         PyROOT_NoneType.tp_flags       = Py_TPFLAGS_HAVE_RICHCOMPARE | Py_TPFLAGS_HAVE_GC;

         PyROOT_NoneType.tp_traverse    = (traverseproc) 0;
         PyROOT_NoneType.tp_clear       = (inquiry) 0;
         PyROOT_NoneType.tp_dealloc     = (destructor)  &InitPyROOT_NoneType_t::DeAlloc;
         PyROOT_NoneType.tp_repr        = Py_TYPE(Py_None)->tp_repr;
         PyROOT_NoneType.tp_richcompare = (richcmpfunc) &InitPyROOT_NoneType_t::RichCompare;
#if PY_VERSION_HEX < 0x03000000
// tp_compare has become tp_reserved (place holder only) in p3
         PyROOT_NoneType.tp_compare     = (cmpfunc) &InitPyROOT_NoneType_t::Compare;
#endif
         PyROOT_NoneType.tp_hash        = (hashfunc) &InitPyROOT_NoneType_t::PtrHash;

         PyROOT_NoneType.tp_as_mapping  = &PyROOT_NoneType_mapping;

         PyType_Ready( &PyROOT_NoneType );
      }

      static void DeAlloc( PyObject* obj ) { Py_TYPE(obj)->tp_free( obj ); }
      static int PtrHash( PyObject* obj ) { return (int)Long_t(obj); }

      static PyObject* RichCompare( PyObject*, PyObject* other, int opid )
      {
         return PyObject_RichCompare( other, Py_None, opid );
      }

      static int Compare( PyObject*, PyObject* other )
      {
#if PY_VERSION_HEX < 0x03000000
         return PyObject_Compare( other, Py_None );
#else
// TODO the following isn't correct as it doens't order, but will do for now ...
         return ! PyObject_RichCompareBool( other, Py_None, Py_EQ );
#endif
      }
   };

} // unnamed namespace


//- ctor/dtor ----------------------------------------------------------------
PyROOT::TMemoryRegulator::TMemoryRegulator()
{
// setup NoneType for referencing and create weakref cache
   static InitPyROOT_NoneType_t initPyROOT_NoneType;

   assert( fgObjectTable == 0 );
   fgObjectTable = new ObjectMap_t;

   assert( fgWeakRefTable == 0 );
   fgWeakRefTable = new WeakRefMap_t;
}

//____________________________________________________________________________
PyROOT::TMemoryRegulator::~TMemoryRegulator()
{
// cleanup weakref cache
   delete fgWeakRefTable;
   fgWeakRefTable = 0;

   delete fgObjectTable;
   fgObjectTable = 0;
}


//- public members -----------------------------------------------------------
void PyROOT::TMemoryRegulator::RecursiveRemove( TObject* object )
{
// called whenever a TObject gets destroyed
   if ( ! object || ! fgObjectTable )   // table can be deleted before libCore is done
      return;

// see whether we're tracking this object
   ObjectMap_t::iterator ppo = fgObjectTable->find( object );

   if ( ppo != fgObjectTable->end() ) {
      fgWeakRefTable->erase( fgWeakRefTable->find( ppo->second ) );
 
   // get the tracked object
      ObjectProxy* pyobj = (ObjectProxy*)PyWeakref_GetObject( ppo->second );
      if ( ! pyobj ) {
         fgObjectTable->erase( ppo );
         return;
      }

   // clean up the weak reference.
      Py_DECREF( ppo->second );

   // nullify the object
      if ( ObjectProxy_Check( pyobj ) ) {
         if ( ! PyROOT_NoneType.tp_traverse ) {
         // take a reference as we're copying its function pointers
            Py_INCREF( Py_TYPE(pyobj) );

         // all object that arrive here are expected to be of the same type ("instance")
            PyROOT_NoneType.tp_traverse   = Py_TYPE(pyobj)->tp_traverse;
            PyROOT_NoneType.tp_clear      = Py_TYPE(pyobj)->tp_clear;
            PyROOT_NoneType.tp_free       = Py_TYPE(pyobj)->tp_free;
         } else if ( PyROOT_NoneType.tp_traverse != Py_TYPE(pyobj)->tp_traverse ) {
            std::cerr << "in PyROOT::TMemoryRegulater, unexpected object of type: "
                      << Py_TYPE(pyobj)->tp_name << std::endl;

         // leave before too much damage is done
            return;
         }

      // notify any other weak referents by playing dead
         int refcnt = ((PyObject*)pyobj)->ob_refcnt;
         ((PyObject*)pyobj)->ob_refcnt = 0;
         PyObject_ClearWeakRefs( (PyObject*)pyobj );
         ((PyObject*)pyobj)->ob_refcnt = refcnt;

      // cleanup object internals
         pyobj->Release();              // held object is out of scope now anyway
         op_dealloc_nofree( pyobj );    // normal object cleanup, while keeping memory

      // reset type object
         Py_INCREF( (PyObject*)(void*)&PyROOT_NoneType );
         Py_DECREF( Py_TYPE(pyobj) );
         ((PyObject*)pyobj)->ob_type = &PyROOT_NoneType;
      }

   // erase the object from tracking (weakref table already cleared, above)
      fgObjectTable->erase( ppo );
   }
}

//____________________________________________________________________________
Bool_t PyROOT::TMemoryRegulator::RegisterObject( ObjectProxy* pyobj, TObject* object )
{
// start tracking <object> proxied by <pyobj>
   if ( ! ( pyobj && object ) )
      return kFALSE;

   ObjectMap_t::iterator ppo = fgObjectTable->find( object );
   if ( ppo == fgObjectTable->end() ) {
      object->SetBit( TObject::kMustCleanup );
      PyObject* pyref = PyWeakref_NewRef( (PyObject*)pyobj, gObjectEraseCallback );
      ObjectMap_t::iterator newppo = fgObjectTable->insert( std::make_pair( object, pyref ) ).first;
      (*fgWeakRefTable)[ pyref ] = newppo;  // no Py_INCREF on pyref, as object table has one
      return kTRUE;
   }

   return kFALSE;
}

//____________________________________________________________________________
Bool_t PyROOT::TMemoryRegulator::UnregisterObject( TObject* object )
{
// stop tracking <object>, without notification
   ObjectMap_t::iterator ppo = fgObjectTable->find( object );

   if ( ppo != fgObjectTable->end() ) {
      fgWeakRefTable->erase( fgWeakRefTable->find( ppo->second ) );
      fgObjectTable->erase( ppo );
      return kTRUE;
   }

   return kFALSE;
}

//____________________________________________________________________________
PyObject* PyROOT::TMemoryRegulator::RetrieveObject( TObject* object )
{
// lookup <object>, return old proxy if tracked
   if ( ! object )
      return 0;

   ObjectMap_t::iterator ppo = fgObjectTable->find( object );
   if ( ppo != fgObjectTable->end() ) {
      PyObject* pyobj = PyWeakref_GetObject( ppo->second );
      Py_XINCREF( pyobj );
      return pyobj;
   }

   return 0;
}


//- private static members ------------------------------------------------------
PyObject* PyROOT::TMemoryRegulator::ObjectEraseCallback( PyObject*, PyObject* pyref )
{
// called when one of the python objects we've registered is going away
   ObjectProxy* pyobj = (ObjectProxy*)PyWeakref_GetObject( pyref );

   if ( ObjectProxy_Check( pyobj ) && pyobj->GetObject() != 0 ) {
   // get TObject pointer to the object
      TObject* object = (TObject*)pyobj->ObjectIsA()->DynamicCast(
         TObject::Class(), pyobj->GetObject() );

      if ( object != 0 ) {
      // erase if tracked
         ObjectMap_t::iterator ppo = fgObjectTable->find( object );
         if ( ppo != fgObjectTable->end() ) {
         // cleanup table entries and weak reference
            fgWeakRefTable->erase( fgWeakRefTable->find( ppo->second ) );
            Py_DECREF( ppo->second );
            fgObjectTable->erase( ppo );
         }
      }
   } else {
   // object already dead; need to clean up the weak ref from the table
      WeakRefMap_t::iterator wri = fgWeakRefTable->find( pyref );
      if ( wri != fgWeakRefTable->end() ) {
         fgObjectTable->erase( wri->second );
         fgWeakRefTable->erase( wri );
         Py_DECREF( pyref );
      }
   }

   Py_INCREF( Py_None );
   return Py_None;
}
