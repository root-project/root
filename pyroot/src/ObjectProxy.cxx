// @(#)root/pyroot:$Name:  $:$Id: ObjectProxy.cxx,v 1.11 2006/04/19 06:20:22 brun Exp $
// Author: Wim Lavrijsen, Jan 2005

// Bindings
#include "PyROOT.h"
#include "ObjectProxy.h"
#include "PyRootType.h"

// ROOT
#include "TObject.h"


//____________________________________________________________________________
void PyROOT::op_dealloc_nofree( ObjectProxy* pyobj ) {
   if ( pyobj->fObject && ( pyobj->fFlags & ObjectProxy::kIsOwner ) ) {
      pyobj->fClass->Destructor( pyobj->fObject );
   }

   pyobj->fClass.~TClassRef();
}


//____________________________________________________________________________
namespace PyROOT {

namespace {

//= PyROOT object proxy nullness checking ====================================
   PyObject* op_nonzero( ObjectProxy* self, void* )
   {
      return PyInt_FromLong( self->GetObject() ? 1 : 0 );
   }

//____________________________________________________________________________
   PyMethodDef op_methods[] = {
      { (char*)"__nonzero__", (PyCFunction)op_nonzero, METH_NOARGS, NULL },
      { (char*)NULL, NULL, 0, NULL }
   };


//= PyROOT object proxy construction/destruction =============================
   ObjectProxy* op_new( PyTypeObject* subtype, PyObject*, PyObject* )
   {
      ObjectProxy* pyobj = (ObjectProxy*)subtype->tp_alloc( subtype, 0 );
      pyobj->fObject = NULL;
      new (&pyobj->fClass) TClassRef( (TClass*)0 );
      pyobj->fFlags  = 0;

      return pyobj;
   }

//____________________________________________________________________________
   void op_dealloc( ObjectProxy* pyobj )
   {
      op_dealloc_nofree( pyobj );
      pyobj->ob_type->tp_free( (PyObject*)pyobj );
   }

//____________________________________________________________________________
   PyObject* op_richcompare( ObjectProxy* self, ObjectProxy* other, int op )
   {
      if ( op != Py_EQ ) {
         Py_INCREF( Py_NotImplemented );
         return Py_NotImplemented;
      }

   // type + held pointer value defines identity
      if ( self->ob_type == other->ob_type && self->fObject == other->fObject ) {
         Py_INCREF( Py_True );
         return Py_True;
      }

      return Py_False;
   }

//____________________________________________________________________________
   PyObject* op_repr( ObjectProxy* pyobj )
   {
      TClass* klass = pyobj->ObjectIsA();
      std::string clName = klass ? klass->GetName() : "<unknown>";
      if ( pyobj->fFlags & ObjectProxy::kIsReference )
         clName.append( "*" );

   // need to prevent accidental derefs when just printing (usually unsafe)
      if ( ! PyObject_HasAttrString( (PyObject*)pyobj, const_cast< char* >( "__deref__" ) ) ) {
         PyObject* name = PyObject_CallMethod( (PyObject*)pyobj,
            const_cast< char* >( "GetName" ), const_cast< char* >( "" ) );

         if ( name ) {
            if ( PyString_GET_SIZE( name ) != 0 ) {
               PyObject* repr = PyString_FromFormat( "<ROOT.%s object (\"%s\") at %p>",
                  clName.c_str(), PyString_AS_STRING( name ), pyobj->fObject );
               Py_DECREF( name );
               return repr;
            }
            Py_DECREF( name );
         } else
            PyErr_Clear();
      }

   // get here if object has no method GetName() or name = ""
      return PyString_FromFormat( const_cast< char* >( "<ROOT.%s object at %p>" ),
         clName.c_str(), pyobj->fObject );
   }

} // unnamed namespace


//= PyROOT object proxy type =================================================
PyTypeObject ObjectProxy_Type = {
   PyObject_HEAD_INIT( &PyRootType_Type )
   0,                         // ob_size
   (char*)"ROOT.ObjectProxy", // tp_name
   sizeof(ObjectProxy),       // tp_basicsize
   0,                         // tp_itemsize
   (destructor)op_dealloc,    // tp_dealloc
   0,                         // tp_print
   0,                         // tp_getattr
   0,                         // tp_setattr
   0,                         // tp_compare
   (reprfunc)op_repr,         // tp_repr
   0,                         // tp_as_number
   0,                         // tp_as_sequence
   0,                         // tp_as_mapping
   0,                         // tp_hash
   0,                         // tp_call
   0,                         // tp_str
   0,                         // tp_getattro
   0,                         // tp_setattro
   0,                         // tp_as_buffer
   Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE |
      Py_TPFLAGS_HAVE_GC,                        // tp_flags
   (char*)"PyROOT object proxy (internal)",      // tp_doc
   0,                         // tp_traverse
   0,                         // tp_clear
   (richcmpfunc)op_richcompare,                  // tp_richcompare
   0,                         // tp_weaklistoffset
   0,                         // tp_iter
   0,                         // tp_iternext
   op_methods,                // tp_methods
   0,                         // tp_members
   0,                         // tp_getset
   0,                         // tp_base
   0,                         // tp_dict
   0,                         // tp_descr_get
   0,                         // tp_descr_set
   0,                         // tp_dictoffset
   0,                         // tp_init
   0,                         // tp_alloc
   (newfunc)op_new,           // tp_new
   0,                         // tp_free
   0,                         // tp_is_gc
   0,                         // tp_bases
   0,                         // tp_mro
   0,                         // tp_cache
   0,                         // tp_subclasses
   0,                         // tp_weaklist

#if PY_MAJOR_VERSION >= 2 && PY_MINOR_VERSION >= 3
   0                          // tp_del
#endif
};

} // namespace PyROOT
