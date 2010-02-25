// @(#)root/pyroot:$Id$
// Author: Wim Lavrijsen, Jan 2005

// Bindings
#include "PyROOT.h"
#include "PyStrings.h"
#include "ObjectProxy.h"
#include "Utility.h"

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
      if ( op != Py_EQ && op != Py_NE ) {
         Py_INCREF( Py_NotImplemented );
         return Py_NotImplemented;
      }

      bool bIsEq = false;

   // special case for None to compare True to a null-pointer
      if ( (PyObject*)other == Py_None && ! self->fObject )
         bIsEq = true;

   // type + held pointer value defines identity (will cover if other is not
   // actually an ObjectProxy, as ob_type will be unequal)
      else if ( self->ob_type == other->ob_type && self->fObject == other->fObject )
         bIsEq = true;

      if ( ( op == Py_EQ && bIsEq ) || ( op == Py_NE && ! bIsEq ) ) {
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


//= PyROOT type number stubs to allow dynamic overrides ======================
#define PYROOT_STUB( name, op, pystring )                                     \
   PyObject* op_##name##_stub( PyObject* self, PyObject* other )              \
   {                                                                          \
   /* place holder to lazily install __name__ if a global overload is available */ \
      if ( ! Utility::AddBinaryOperator( self, other, #op, "__"#name"__" ) ) {\
         Py_INCREF( Py_NotImplemented );                                      \
         return Py_NotImplemented;                                            \
      }                                                                       \
                                                                              \
   /* redo the call, which will now go to the newly installed method */       \
      return PyObject_CallMethodObjArgs( self, pystring, other, NULL );       \
   }

PYROOT_STUB( add, +, PyStrings::gAdd )
PYROOT_STUB( sub, -, PyStrings::gSub )
PYROOT_STUB( mul, *, PyStrings::gMul )
PYROOT_STUB( div, /, PyStrings::gDiv )

//____________________________________________________________________________
   PyNumberMethods op_as_number = {
      (binaryfunc)op_add_stub,        // nb_add
      (binaryfunc)op_sub_stub,        // nb_subtract
      (binaryfunc)op_mul_stub,        // nb_multiply
      (binaryfunc)op_div_stub,        // nb_divide
      0,                              // nb_remainder
      0,                              // nb_divmod
      0,                              // nb_power
      0,                              // nb_negative
      0,                              // tp_positive
      0,                              // tp_absolute
      0,                              // tp_nonzero
      0,                              // nb_invert
      0,                              // nb_lshift
      0,                              // nb_rshift
      0,                              // nb_and
      0,                              // nb_xor
      0,                              // nb_or
      0,                              // nb_coerce
      0,                              // nb_int
      0,                              // nb_long
      0,                              // nb_float
      0,                              // nb_oct
      0,                              // nb_hex
      0,                              // nb_inplace_add
      0,                              // nb_inplace_subtract
      0,                              // nb_inplace_multiply
      0,                              // nb_inplace_divide
      0,                              // nb_inplace_remainder
      0,                              // nb_inplace_power
      0,                              // nb_inplace_lshift
      0,                              // nb_inplace_rshift
      0,                              // nb_inplace_and
      0,                              // nb_inplace_xor
      0                               // nb_inplace_or
#if PY_MAJOR_VERSION >= 2 && PY_MINOR_VERSION >= 2
      , 0                             // nb_floor_divide
      , 0                             // nb_true_divide
      , 0                             // nb_inplace_floor_divide
      , 0                             // nb_inplace_true_divide
#endif
#if PY_MAJOR_VERSION >= 2 && PY_MINOR_VERSION >= 5
      , 0                             // nb_index
#endif
   };

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
   0, //&op_as_number,             // tp_as_number
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
