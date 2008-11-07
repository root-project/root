// @(#)root/pyroot:$Id$
// Author: Wim Lavrijsen, Jan 2005

// Bindings
#include "PyROOT.h"
#include "PyStrings.h"
#include "ObjectProxy.h"

// ROOT
#include "TObject.h"
#include "TBufferFile.h"      // for pickling


//- data _______________________________________________________________________
R__EXTERN PyObject* gRootModule;

//____________________________________________________________________________
void PyROOT::op_dealloc_nofree( ObjectProxy* pyobj ) {
   if ( pyobj->fObject && ( pyobj->fFlags & ObjectProxy::kIsOwner ) ) {
      pyobj->ObjectIsA()->Destructor( pyobj->fObject );
   }
}


//____________________________________________________________________________
namespace PyROOT {

namespace {

//= PyROOT object proxy nullness checking ====================================
   PyObject* op_nonzero( ObjectProxy* self, void* )
   {
      return PyInt_FromLong( self->GetObject() ? 1 : 0 );
   }

//= PyROOT object proxy pickle support =======================================
   PyObject* op_reduce( ObjectProxy* self )
   {
   // Turn the object proxy instance into a character stream and return for
   // pickle, together with the callable object that can restore the stream
   // into the object proxy instance.

   // keep a borrowed reference around to the callable function for expanding;
   // because it is borrowed, it means that there can be no pickling during the
   // shutdown of the libPyROOT module
      static PyObject* s_expand = PyDict_GetItemString(
         PyModule_GetDict( gRootModule ),  const_cast< char* >( "_ObjectProxy__expand__" ) );

   // TBuffer and its derived classes can't write themselves, but can be created
   // directly from the buffer, so handle them in a special case
      static TClassRef s_bfClass( "TBufferFile" );

      TBufferFile* buff = 0;
      if ( s_bfClass == self->ObjectIsA() ) {
         buff = (TBufferFile*)self->GetObject();
      } else {
      // no cast is needed, but WriteObject taking a TClass argument is protected,
      // so use WriteObjectAny()
         static TBufferFile s_buff( TBuffer::kWrite );
         s_buff.Reset();
         if ( s_buff.WriteObjectAny( self->GetObject(), self->ObjectIsA() ) != 1 ) {
            PyErr_Format( PyExc_IOError,
               "could not stream object of type %s", self->ObjectIsA()->GetName() );
            return 0;
         }
         buff = &s_buff;
      }

   // use a string for the serialized result, as a python buffer will not copy
   // the buffer contents; use a string for the class name, used when casting
   // on reading back in (see RootModule.cxx:TObjectExpand)
      PyObject* res2 = PyTuple_New( 2 );
      PyTuple_SET_ITEM( res2, 0, PyString_FromStringAndSize( buff->Buffer(), buff->Length() ) );
      PyTuple_SET_ITEM( res2, 1, PyString_FromString( self->ObjectIsA()->GetName() ) );

      PyObject* result = PyTuple_New( 2 );
      Py_INCREF( s_expand );
      PyTuple_SET_ITEM( result, 0, s_expand );
      PyTuple_SET_ITEM( result, 1, res2 );

      return result;
   }

//____________________________________________________________________________
   PyMethodDef op_methods[] = {
      { (char*)"__nonzero__", (PyCFunction)op_nonzero, METH_NOARGS, NULL },
      { (char*)"__reduce__",  (PyCFunction)op_reduce,  METH_NOARGS, NULL },
      { (char*)NULL, NULL, 0, NULL }
   };


//= PyROOT object proxy construction/destruction =============================
   ObjectProxy* op_new( PyTypeObject* subtype, PyObject*, PyObject* )
   {
      ObjectProxy* pyobj = (ObjectProxy*)subtype->tp_alloc( subtype, 0 );
      pyobj->fObject = NULL;
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
      Py_INCREF( Py_False );
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
      if ( ! PyObject_HasAttr( (PyObject*)pyobj, PyStrings::gDeref ) ) {
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
   0                          // tp_weaklist
#if PY_MAJOR_VERSION >= 2 && PY_MINOR_VERSION >= 3
   , 0                        // tp_del
#endif
#if PY_MAJOR_VERSION >= 2 && PY_MINOR_VERSION >= 6
   , 0                        // tp_version_tag
#endif
};

} // namespace PyROOT
