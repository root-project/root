// @(#)root/pyroot:$Id$
// Author: Wim Lavrijsen, Jan 2005

// Bindings
#include "PyROOT.h"
#include "PyStrings.h"
#include "ObjectProxy.h"
#include "RootWrapper.h"
#include "Utility.h"

// ROOT
#include "TBufferFile.h"      // for pickling
#include "TClass.h"           // id.
#include "TObject.h"          // for gROOT life-check
#include "TROOT.h"            // id.

// Standard
#include <algorithm>


//______________________________________________________________________________
//                          Python-side proxy objects
//                          =========================
//
// C++ objects are represented in Python by ObjectProxy's, which encapsulate
// them using either a pointer (normal), pointer-to-pointer (kIsReference set),
// or as an owned value (kIsValue set). Objects held as reference are never
// owned, otherwise the object is owned if kIsOwner is set.
//
// In addition to encapsulation, ObjectProxy offers pickling (using TBufferFile
// with a copy into a Python string); rudimentary comparison operators (based on
// pointer value and class comparisons); stubs for numeric operators; and a
// representation that prints the C++ pointer values, rather than the PyObject*
// ones as is the default.


//- data _______________________________________________________________________
namespace PyROOT {
   R__EXTERN PyObject* gRootModule;    // needed for pickling
}


////////////////////////////////////////////////////////////////////////////////
/// Destroy the held C++ object, if owned; does not deallocate the proxy.

void PyROOT::op_dealloc_nofree( ObjectProxy* pyobj ) {
   if ( gROOT && !gROOT->TestBit( TObject::kInvalidObject ) ) {
      if ( pyobj->fFlags & ObjectProxy::kIsValue ) {
         if ( ! (pyobj->fFlags & ObjectProxy::kIsSmartPtr) ) {
            Cppyy::CallDestructor( pyobj->ObjectIsA(), pyobj->GetObject() );
            Cppyy::Deallocate( pyobj->ObjectIsA(), pyobj->GetObject() );
         } else {
            Cppyy::CallDestructor( pyobj->fSmartPtrType, pyobj->fSmartPtr );
            Cppyy::Deallocate( pyobj->fSmartPtrType, pyobj->fSmartPtr );
         }
      }
      else if ( pyobj->fObject && ( pyobj->fFlags & ObjectProxy::kIsOwner ) ) {
         if ( ! (pyobj->fFlags & ObjectProxy::kIsSmartPtr) ) {
            Cppyy::Destruct( pyobj->ObjectIsA(), pyobj->GetObject() );
         } else {
            Cppyy::Destruct( pyobj->fSmartPtrType, pyobj->fSmartPtr );
         }
      }
   }
   pyobj->fObject = nullptr;
}


////////////////////////////////////////////////////////////////////////////////

namespace PyROOT {
namespace {

//= PyROOT object proxy null-ness checking ===================================
   PyObject* op_nonzero( ObjectProxy* self )
   {
   // Null of the proxy is determined by null-ness of the held C++ object.
      PyObject* result = self->GetObject() ? Py_True : Py_False;
      Py_INCREF( result );
      return result;
   }

//= PyROOT object explicit destruction =======================================
   PyObject* op_destruct( ObjectProxy* self )
   {
   // User access to force deletion of the object. Needed in case of a true
   // garbage collector (like in PyPy), to allow the user control over when
   // the C++ destructor is called. This method requires that the C++ object
   // is owned (no-op otherwise).
      op_dealloc_nofree( self );
      Py_INCREF( Py_None );
      return Py_None;
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
      static Cppyy::TCppType_t s_bfClass = Cppyy::GetScope( "TBufferFile" );

      TBufferFile *buff = nullptr;
      if ( s_bfClass == self->ObjectIsA() ) {
         buff = (TBufferFile*)self->GetObject();
      } else {
      // no cast is needed, but WriteObject taking a TClass argument is protected,
      // so use WriteObjectAny()
         static TBufferFile s_buff( TBuffer::kWrite );
         s_buff.Reset();
         if ( s_buff.WriteObjectAny( self->GetObject(),
               TClass::GetClass( Cppyy::GetFinalName( self->ObjectIsA() ).c_str() ) ) != 1 ) {
            PyErr_Format( PyExc_IOError,
               "could not stream object of type %s", Cppyy::GetFinalName( self->ObjectIsA() ).c_str() );
            return nullptr;
         }
         buff = &s_buff;
      }

   // use a string for the serialized result, as a python buffer will not copy
   // the buffer contents; use a string for the class name, used when casting
   // on reading back in (see RootModule.cxx:TObjectExpand)
      PyObject* res2 = PyTuple_New( 2 );
      PyTuple_SET_ITEM( res2, 0, PyBytes_FromStringAndSize( buff->Buffer(), buff->Length() ) );
      PyTuple_SET_ITEM( res2, 1, PyBytes_FromString( Cppyy::GetFinalName( self->ObjectIsA() ).c_str() ) );

      PyObject* result = PyTuple_New( 2 );
      Py_INCREF( s_expand );
      PyTuple_SET_ITEM( result, 0, s_expand );
      PyTuple_SET_ITEM( result, 1, res2 );

      return result;
   }

//= PyROOT object dispatch support ===========================================
   PyObject* op_dispatch( PyObject* self, PyObject* args, PyObject* /* kdws */ )
   {
   // User-side __dispatch__ method to allow selection of a specific overloaded
   // method. The actual selection is in the disp() method of MethodProxy.
   PyObject *mname = nullptr, *sigarg = nullptr;
   if (!PyArg_ParseTuple(args, const_cast<char *>("O!O!:__dispatch__"), &PyROOT_PyUnicode_Type, &mname,
                         &PyROOT_PyUnicode_Type, &sigarg))
      return nullptr;

   // get the named overload
      PyObject* pymeth = PyObject_GetAttr( self, mname );
      if (!pymeth) return nullptr;

      // get the 'disp' method to allow overload selection
      PyObject* pydisp = PyObject_GetAttrString( pymeth, const_cast<char*>( "disp" ) );
      if ( ! pydisp ) {
         Py_DECREF( pymeth );
         return nullptr;
      }

   // finally, call dispatch to get the specific overload
      PyObject* oload = PyObject_CallFunctionObjArgs( pydisp, sigarg, NULL );
      Py_DECREF( pydisp );
      Py_DECREF( pymeth );
      return oload;
   }

//= PyROOT smart pointer support =============================================
  PyObject* op_get_smart_ptr( ObjectProxy* self )
  {
     if ( !( self->fFlags & ObjectProxy::kIsSmartPtr ) ) {
        Py_RETURN_NONE;
     }

     return (PyObject*)PyROOT::BindCppObject( self->fSmartPtr, self->fSmartPtrType );
  }

////////////////////////////////////////////////////////////////////////////////

  PyMethodDef op_methods[] = {
     {(char *)"__nonzero__", (PyCFunction)op_nonzero, METH_NOARGS, nullptr},
     {(char *)"__bool__", (PyCFunction)op_nonzero, METH_NOARGS, nullptr}, // for p3
     {(char *)"__destruct__", (PyCFunction)op_destruct, METH_NOARGS, nullptr},
     {(char *)"__reduce__", (PyCFunction)op_reduce, METH_NOARGS, nullptr},
     {(char *)"__dispatch__", (PyCFunction)op_dispatch, METH_VARARGS, (char *)"dispatch to selected overload"},
     {(char *)"_get_smart_ptr", (PyCFunction)op_get_smart_ptr, METH_NOARGS,
      (char *)"get associated smart pointer, if any"},
     {(char *)nullptr, nullptr, 0, nullptr}};

  //= PyROOT object proxy construction/destruction =============================
  ObjectProxy *op_new(PyTypeObject *subtype, PyObject *, PyObject *)
  {
     // Create a new object proxy (holder only).
     ObjectProxy *pyobj = (ObjectProxy *)subtype->tp_alloc(subtype, 0);
     pyobj->fObject = nullptr;
     pyobj->fFlags = 0;

     return pyobj;
   }

////////////////////////////////////////////////////////////////////////////////
/// Remove (Python-side) memory held by the object proxy.

   void op_dealloc( ObjectProxy* pyobj )
   {
      op_dealloc_nofree( pyobj );
      Py_TYPE(pyobj)->tp_free( (PyObject*)pyobj );
   }

////////////////////////////////////////////////////////////////////////////////
/// Rich set of comparison objects; only equals and not-equals are defined.

   PyObject* op_richcompare( ObjectProxy* self, ObjectProxy* other, int op )
   {
      if ( op != Py_EQ && op != Py_NE ) {
         Py_INCREF( Py_NotImplemented );
         return Py_NotImplemented;
      }

      Bool_t bIsEq = false;

   // special case for None to compare True to a null-pointer
      if ( (PyObject*)other == Py_None && ! self->fObject )
         bIsEq = true;

   // type + held pointer value defines identity (will cover if other is not
   // actually an ObjectProxy, as ob_type will be unequal)
      else if ( Py_TYPE(self) == Py_TYPE(other) && self->GetObject() == other->GetObject() )
         bIsEq = true;

      if ( ( op == Py_EQ && bIsEq ) || ( op == Py_NE && ! bIsEq ) ) {
         Py_INCREF( Py_True );
         return Py_True;
      }

      Py_INCREF( Py_False );
      return Py_False;
   }

////////////////////////////////////////////////////////////////////////////////
/// Build a representation string of the object proxy that shows the address
/// of the C++ object that is held, as well as its type.

   PyObject* op_repr( ObjectProxy* pyobj )
   {
      Cppyy::TCppType_t klass = pyobj->ObjectIsA();
      std::string clName = klass ? Cppyy::GetFinalName( klass ) : "<unknown>";
      if ( pyobj->fFlags & ObjectProxy::kIsReference )
         clName.append( "*" );

      std::string smartPtrName;
      if ( pyobj->fFlags & ObjectProxy::kIsSmartPtr ) {
         Cppyy::TCppType_t smartPtrType = pyobj->fSmartPtrType;
         smartPtrName = smartPtrType ? Cppyy::GetFinalName( smartPtrType ) : "unknown smart pointer";
      }

   // need to prevent accidental derefs when just printing (usually unsafe)
      if ( ! PyObject_HasAttr( (PyObject*)pyobj, PyStrings::gDeref ) ) {
         PyObject* name = PyObject_CallMethod( (PyObject*)pyobj,
            const_cast< char* >( "GetName" ), const_cast< char* >( "" ) );

         if ( name ) {
            if ( PyROOT_PyUnicode_GET_SIZE( name ) != 0 ) {
               if ( pyobj->fFlags & ObjectProxy::kIsSmartPtr ) {
                  PyObject* repr = PyROOT_PyUnicode_FromFormat( "<ROOT.%s object (\"%s\") at %p held by %s at %p>",
                     clName.c_str(), PyROOT_PyUnicode_AsString( name ), pyobj->GetObject(), smartPtrName.c_str(), pyobj->fSmartPtr );
                  Py_DECREF( name );
                  return repr;
               } else {
                  PyObject* repr = PyROOT_PyUnicode_FromFormat( "<ROOT.%s object (\"%s\") at %p>",
                     clName.c_str(), PyROOT_PyUnicode_AsString( name ), pyobj->GetObject() );
                  Py_DECREF( name );
                  return repr;
               }
            }
            Py_DECREF( name );
         } else
            PyErr_Clear();
      }

   // get here if object has no method GetName() or name = ""
      if ( pyobj->fFlags & ObjectProxy::kIsSmartPtr ) {
         return PyROOT_PyUnicode_FromFormat( const_cast< char* >( "<ROOT.%s object at %p held by %s at %p>" ),
            clName.c_str(), pyobj->GetObject(), smartPtrName.c_str(), pyobj->fSmartPtr );
      } else {
         return PyROOT_PyUnicode_FromFormat( const_cast< char* >( "<ROOT.%s object at %p>" ),
                                             clName.c_str(), pyobj->GetObject() );
      }
   }


//= PyROOT type number stubs to allow dynamic overrides ======================
#define PYROOT_STUB( name, op, pystring )                                     \
   PyObject* op_##name##_stub( PyObject* left, PyObject* right )              \
   {                                                                          \
      if ( ! ObjectProxy_Check( left ) ) {                                    \
         if ( ObjectProxy_Check( right ) ) {                                  \
            std::swap( left, right );                                         \
         } else {                                                             \
            Py_INCREF( Py_NotImplemented );                                   \
            return Py_NotImplemented;                                         \
         }                                                                    \
      }                                                                       \
   /* place holder to lazily install __name__ if a global overload is available */ \
      if ( ! Utility::AddBinaryOperator(                                      \
               left, right, #op, "__"#name"__", "__r"#name"__" ) ) {          \
         Py_INCREF( Py_NotImplemented );                                      \
         return Py_NotImplemented;                                            \
      }                                                                       \
                                                                              \
   /* redo the call, which will now go to the newly installed method */       \
      return PyObject_CallMethodObjArgs( left, pystring, right, NULL );       \
   }

PYROOT_STUB( add, +, PyStrings::gAdd )
PYROOT_STUB( sub, -, PyStrings::gSub )
PYROOT_STUB( mul, *, PyStrings::gMul )
PYROOT_STUB( div, /, PyStrings::gDiv )

////////////////////////////////////////////////////////////////////////////////

PyNumberMethods op_as_number = {
   (binaryfunc)op_add_stub, // nb_add
   (binaryfunc)op_sub_stub, // nb_subtract
   (binaryfunc)op_mul_stub, // nb_multiply
#if PY_VERSION_HEX < 0x03000000
   (binaryfunc)op_div_stub, // nb_divide
#endif
   nullptr, // nb_remainder
   nullptr, // nb_divmod
   nullptr, // nb_power
   nullptr, // nb_negative
   nullptr, // nb_positive
   nullptr, // nb_absolute
   nullptr, // tp_nonzero (nb_bool in p3)
   nullptr, // nb_invert
   nullptr, // nb_lshift
   nullptr, // nb_rshift
   nullptr, // nb_and
   nullptr, // nb_xor
   nullptr, // nb_or
#if PY_VERSION_HEX < 0x03000000
   nullptr, // nb_coerce
#endif
   nullptr, // nb_int
   nullptr, // nb_long (nb_reserved in p3)
   nullptr, // nb_float
#if PY_VERSION_HEX < 0x03000000
   nullptr, // nb_oct
   nullptr, // nb_hex
#endif
   nullptr, // nb_inplace_add
   nullptr, // nb_inplace_subtract
   nullptr, // nb_inplace_multiply
#if PY_VERSION_HEX < 0x03000000
   nullptr, // nb_inplace_divide
#endif
   nullptr, // nb_inplace_remainder
   nullptr, // nb_inplace_power
   nullptr, // nb_inplace_lshift
   nullptr, // nb_inplace_rshift
   nullptr, // nb_inplace_and
   nullptr, // nb_inplace_xor
   nullptr  // nb_inplace_or
#if PY_VERSION_HEX >= 0x02020000
   ,
   nullptr // nb_floor_divide
#if PY_VERSION_HEX < 0x03000000
   ,
   nullptr // nb_true_divide
#else
   ,
   (binaryfunc)op_div_stub // nb_true_divide
#endif
   ,
   nullptr // nb_inplace_floor_divide
   ,
   nullptr // nb_inplace_true_divide
#endif
#if PY_VERSION_HEX >= 0x02050000
   ,
   nullptr // nb_index
#endif
#if PY_VERSION_HEX >= 0x03050000
   ,
   0 // nb_matrix_multiply
   ,
   0 // nb_inplace_matrix_multiply
#endif
};

} // unnamed namespace


//= PyROOT object proxy type =================================================
PyTypeObject ObjectProxy_Type = {
   PyVarObject_HEAD_INIT(&PyRootType_Type, 0)(char *) "ROOT.ObjectProxy",                 // tp_name
   sizeof(ObjectProxy),                                                                   // tp_basicsize
   0,                                                                                     // tp_itemsize
   (destructor)op_dealloc,                                                                // tp_dealloc
   nullptr,                                                                               // tp_print
   nullptr,                                                                               // tp_getattr
   nullptr,                                                                               // tp_setattr
   nullptr,                                                                               // tp_compare
   (reprfunc)op_repr,                                                                     // tp_repr
   &op_as_number,                                                                         // tp_as_number
   nullptr,                                                                               // tp_as_sequence
   nullptr,                                                                               // tp_as_mapping
   PyBaseObject_Type.tp_hash,                                                             // tp_hash
   nullptr,                                                                               // tp_call
   nullptr,                                                                               // tp_str
   nullptr,                                                                               // tp_getattro
   nullptr,                                                                               // tp_setattro
   nullptr,                                                                               // tp_as_buffer
   Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC | Py_TPFLAGS_CHECKTYPES, // tp_flags
   (char *)"PyROOT object proxy (internal)",                                              // tp_doc
   nullptr,                                                                               // tp_traverse
   nullptr,                                                                               // tp_clear
   (richcmpfunc)op_richcompare,                                                           // tp_richcompare
   0,                                                                                     // tp_weaklistoffset
   nullptr,                                                                               // tp_iter
   nullptr,                                                                               // tp_iternext
   op_methods,                                                                            // tp_methods
   nullptr,                                                                               // tp_members
   nullptr,                                                                               // tp_getset
   nullptr,                                                                               // tp_base
   nullptr,                                                                               // tp_dict
   nullptr,                                                                               // tp_descr_get
   nullptr,                                                                               // tp_descr_set
   0,                                                                                     // tp_dictoffset
   nullptr,                                                                               // tp_init
   nullptr,                                                                               // tp_alloc
   (newfunc)op_new,                                                                       // tp_new
   nullptr,                                                                               // tp_free
   nullptr,                                                                               // tp_is_gc
   nullptr,                                                                               // tp_bases
   nullptr,                                                                               // tp_mro
   nullptr,                                                                               // tp_cache
   nullptr,                                                                               // tp_subclasses
   nullptr                                                                                // tp_weaklist
#if PY_VERSION_HEX >= 0x02030000
   ,
   nullptr // tp_del
#endif
#if PY_VERSION_HEX >= 0x02060000
   ,
   0 // tp_version_tag
#endif
#if PY_VERSION_HEX >= 0x03040000
   ,
   0 // tp_finalize
#endif
};

} // namespace PyROOT
