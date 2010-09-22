// Author: Wim Lavrijsen, Jan 2008

#ifndef PYROOT_TEMPLATEPROXY_H
#define PYROOT_TEMPLATEPROXY_H

// ROOT
#include "DllImport.h"

// Standard
#include <string>


namespace PyROOT {

/** Template proxy object to return functions and methods
      @author  WLAV
      @date    01/15/2008
      @version 1.0
 */

   class TemplateProxy {
   public:
      void Set( const std::string& name, PyObject* pyclass )
      {
         fPyName  = PyROOT_PyUnicode_FromString( const_cast< char* >( name.c_str() ) );
         Py_XINCREF( pyclass );
         fPyClass = pyclass;
      }

   public:               // public, as the python C-API works with C structs
      PyObject_HEAD
      PyObject* fPyName;
      PyObject* fPyClass;
      PyObject* fSelf;

   private:              // private, as the python C-API will handle creation
      TemplateProxy() {}
   };


//- template proxy type and type verification --------------------------------
   R__EXTERN PyTypeObject TemplateProxy_Type;

   template< typename T >
   inline Bool_t TemplateProxy_Check( T* object )
   {
      return object && PyObject_TypeCheck( object, &TemplateProxy_Type );
   }

   template< typename T >
   inline Bool_t TemplateProxy_CheckExact( T* object )
   {
      return object && Py_TYPE(object) == &TemplateProxy_Type;
   }

//- creation -----------------------------------------------------------------
   inline TemplateProxy* TemplateProxy_New( const std::string& name, PyObject* pyclass )
   {
      TemplateProxy* pytmpl = (TemplateProxy*)TemplateProxy_Type.tp_new( &TemplateProxy_Type, 0, 0 );
      pytmpl->Set( name, pyclass );
      return pytmpl;
   }

} // namespace PyROOT

#endif // !PYROOT_TEMPLATEPROXY_H
