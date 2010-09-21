// Author: Wim Lavrijsen, Jan 2005

// Bindings
#include "PyROOT.h"
#include "TemplateProxy.h"
#include "Utility.h"


namespace PyROOT {

namespace {

//= PyROOT template proxy construction/destruction ===========================
   TemplateProxy* tpp_new( PyTypeObject*, PyObject*, PyObject* )
   {
      TemplateProxy* pytmpl = PyObject_GC_New( TemplateProxy, &TemplateProxy_Type );
      pytmpl->fPyName  = NULL;
      pytmpl->fPyClass = NULL;
      pytmpl->fSelf    = NULL;

      PyObject_GC_Track( pytmpl );
      return pytmpl;
   }

//____________________________________________________________________________
   void tpp_dealloc( TemplateProxy* pytmpl )
   {
      PyObject_GC_UnTrack( pytmpl );
      PyObject_GC_Del( pytmpl );
   }

//____________________________________________________________________________
   int tpp_traverse( TemplateProxy* pytmpl, visitproc visit, void* args )
   {
      if ( pytmpl->fPyName ) {
         int err = visit( (PyObject*)pytmpl->fPyName, args );
         if ( err )
            return err;
      }

      if ( pytmpl->fPyClass ) {
         int err = visit( (PyObject*)pytmpl->fPyClass, args );
         if ( err )
            return err;
      }

      if ( pytmpl->fSelf ) {
         int err = visit( (PyObject*)pytmpl->fSelf, args );
         if ( err )
            return err;
      }

      return 0;
   }

//____________________________________________________________________________
   int tpp_clear( TemplateProxy* pytmpl )
   {
      Py_XDECREF( (PyObject*)pytmpl->fPyName );
      pytmpl->fPyName = NULL;

      Py_XDECREF( (PyObject*)pytmpl->fPyClass );
      pytmpl->fPyClass = NULL;

      Py_XDECREF( (PyObject*)pytmpl->fSelf );
      pytmpl->fSelf = NULL;

      return 0;
   }

//= PyROOT template proxy callable behavior ==================================
   PyObject* tpp_call( TemplateProxy* pytmpl, PyObject* args, PyObject* kwds )
   {
   // dispatcher to the actual member method, args is self object + template arguments
   // (as in a function call); build full instantiation
      PyObject* pymeth = 0;

      Py_ssize_t nArgs = PyTuple_GET_SIZE( args );
      if ( 1 <= nArgs ) {

      // build "< type, type, ... >" part of method name
         Py_INCREF( pytmpl->fPyName );
         PyObject* pyname = pytmpl->fPyName;
         if ( Utility::BuildTemplateName( pyname, args, 0 ) ) {
         // lookup method on self (to make sure it propagates), which is readily callable
            pymeth = PyObject_GetAttr( pytmpl->fSelf, pyname );
         }
         Py_XDECREF( pyname );

      }

      if ( pymeth )
         return pymeth;       // templated, now called by the user

   // if the method lookup fails, try to locate the "generic" version of the template
      PyErr_Clear();
      pymeth = PyObject_GetAttrString( pytmpl->fSelf, const_cast< char* >(
         (std::string( "__generic_" ) + PyROOT_PyUnicode_AsString( pytmpl->fPyName )).c_str()) );

      if ( pymeth )
         return PyObject_Call( pymeth, args, kwds );   // non-templated, executed as-is
      
      return pymeth;
   }

//____________________________________________________________________________
   TemplateProxy* tpp_descrget( TemplateProxy* pytmpl, PyObject* pyobj, PyObject* )
   {
   // create and use a new template proxy (language requirement)
      TemplateProxy* newPyTmpl = (TemplateProxy*)TemplateProxy_Type.tp_alloc( &TemplateProxy_Type, 0 );

   // copy name and class
      Py_INCREF( pytmpl->fPyName );
      newPyTmpl->fPyName = pytmpl->fPyName;

      Py_XINCREF( pytmpl->fPyClass );
      newPyTmpl->fPyClass = pytmpl->fPyClass;

   // new method is to be bound to current object (may be NULL)
      Py_XINCREF( pyobj );
      newPyTmpl->fSelf = pyobj;

      return newPyTmpl;
   }

} // unnamed namespace


//= PyROOT template proxy type ===============================================
PyTypeObject TemplateProxy_Type = {
   PyVarObject_HEAD_INIT( &PyType_Type, 0 )
   (char*)"ROOT.TemplateProxy", // tp_name
   sizeof(TemplateProxy),     // tp_basicsize
   0,                         // tp_itemsize
   (destructor)tpp_dealloc,   // tp_dealloc
   0,                         // tp_print
   0,                         // tp_getattr
   0,                         // tp_setattr
   0,                         // tp_compare
   0,                         // tp_repr
   0,                         // tp_as_number
   0,                         // tp_as_sequence
   0,                         // tp_as_mapping
   0,                         // tp_hash
   (ternaryfunc)tpp_call,     // tp_call
   0,                         // tp_str
   0,                         // tp_getattro
   0,                         // tp_setattro
   0,                         // tp_as_buffer
   Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,      // tp_flags
   (char*)"PyROOT template proxy (internal)",    // tp_doc
   (traverseproc)tpp_traverse,// tp_traverse
   (inquiry)tpp_clear,        // tp_clear
   0,                         // tp_richcompare
   0,                         // tp_weaklistoffset
   0,                         // tp_iter
   0,                         // tp_iternext
   0,                         // tp_methods
   0,                         // tp_members
   0,                         // tp_getset
   0,                         // tp_base
   0,                         // tp_dict
   (descrgetfunc)tpp_descrget,// tp_descr_get
   0,                         // tp_descr_set
   0,                         // tp_dictoffset
   0,                         // tp_init
   0,                         // tp_alloc
   (newfunc)tpp_new,          // tp_new
   0,                         // tp_free
   0,                         // tp_is_gc
   0,                         // tp_bases
   0,                         // tp_mro
   0,                         // tp_cache
   0,                         // tp_subclasses
   0                          // tp_weaklist
#if PY_VERSION_HEX >= 0x02030000
   , 0                        // tp_del
#endif
#if PY_VERSION_HEX >= 0x02060000
   , 0                        // tp_version_tag
#endif
};

} // namespace PyROOT
