// @(#)root/pyroot:$Name:  $:$Id: TPyClassGenerator.cxx,v 1.11 2006/11/30 23:18:32 pcanal Exp $
// Author: Wim Lavrijsen, May 2004

// Bindings
#include "PyROOT.h"
#include "TPyClassGenerator.h"
#include "Utility.h"
#include "TPyReturn.h"

// ROOT
#include "TROOT.h"
#include "TClass.h"

// CINT
#include "Api.h"

// Standard
#include <string>


//- private helpers ----------------------------------------------------------
namespace {

   //_________________________________________________________________________
   int PyCtorCallback( G__value* res, G__CONST char*, struct G__param* libp, int hash )
   {
      G__ifunc_table* ifunc = 0;
      long index = 0;

      // from cint/src/common.h
#define G__RECMEMFUNCENV      (long)0x7fff0036
      G__CurrentCall( G__RECMEMFUNCENV, &ifunc, &index );
      G__MethodInfo mi; mi.Init((long)ifunc, index, 0);

      PyObject* args = PyTuple_New( 0 );
      PyObject* result =  PyObject_Call( (PyObject*)mi.GetUserParam(), args, NULL );
      if ( ! result )
         PyErr_Print();
      Py_DECREF( args );

      res->obj.i = (Long_t)result;
      res->ref   = (Long_t)result;
      res->type  = 'u';

      G__linked_taginfo pti;
      pti.tagnum = -1;
      pti.tagtype = 'c';

      PyObject* str = PyObject_Str( (PyObject*)mi.GetUserParam() );
      std::string clName = PyString_AS_STRING( str );
      Py_DECREF( str );

      clName = clName.substr( clName.rfind( '.' )+1, std::string::npos );
      pti.tagname = clName.c_str();

      res->tagnum = G__get_linked_tagnum( &pti );

      return ( 1 || hash || res || libp );
   }

   //_________________________________________________________________________
   int PyMemFuncCallback( G__value* res, G__CONST char*, struct G__param* libp, int hash )
   {
      PyObject* self = (PyObject*)G__getstructoffset();
      Py_INCREF( self );

      G__ifunc_table* ifunc = 0;
      long index = 0;
      G__CurrentCall( G__RECMEMFUNCENV, &ifunc, &index );
      G__MethodInfo mi; mi.Init((long)ifunc, index, 0);

      PyObject* args = PyTuple_New( 1 + libp->paran );
      PyTuple_SetItem( args, 0, self );
      for ( int i = 0; i < libp->paran; ++i ) {
         PyObject* arg = 0;
         switch ( libp->para[i].type ) {
         case 'd':
            arg = PyFloat_FromDouble( G__Mdouble(libp->para[i]) );
            break;
         case 'f':
            arg = PyFloat_FromDouble( (double)G__Mfloat(libp->para[i]) );
            break;
         case 'l':
            arg = PyLong_FromLong( G__Mlong(libp->para[i]) );
            break;
         case 'k':
            arg = PyLong_FromUnsignedLong( G__Mulong(libp->para[i] ) );
            break;
         case 'i':
            arg = PyInt_FromLong( (Long_t)G__Mint(libp->para[i]) );
            break;
         case 'h':
            arg = PyInt_FromLong( (Long_t)G__Muint(libp->para[i]) );
            break;
         case 's':
            arg = PyInt_FromLong( (Long_t)G__Mshort(libp->para[i]) );
            break;
         case 'r':
            arg = PyInt_FromLong( (Long_t)G__Mushort(libp->para[i]) );
            break;
         case 'u':
         // longlong, ulonglong, longdouble
            break;
         case 'c':
            char cc[2]; cc[0] = G__Mchar(libp->para[i]); cc[1] = '\0';
            arg = PyString_FromString( cc );
            break;
         case 'b':
         // unsigned char
            break;
         case 'C':
            arg = PyString_FromString( (char*)G__Mlong(libp->para[i]) );
            break;
         }

         if ( arg != 0 )
            PyTuple_SetItem( args, i+1, arg );         // steals ref to arg
         else {
            PyErr_Format( PyExc_TypeError,
               "error converting parameter: %d (type: %c)", i, libp->para[i].type );
            break;
         }

      }

      PyObject* result = 0;
      if ( ! PyErr_Occurred() )
         result =  PyObject_Call( (PyObject*)mi.GetUserParam(), args, NULL );
      Py_DECREF( args );

      if ( ! result )
         PyErr_Print();

      TPyReturn* retval = new TPyReturn( result );
      res->obj.i  = (Long_t)retval;
      res->ref    = (Long_t)retval;
      res->type   = 'u';
      res->tagnum = TPyReturn::Class()->GetClassInfo()->Tagnum();

      return ( 1 || hash || res || libp );
   }

} // unnamed namespace


//- public members -----------------------------------------------------------
TClass* TPyClassGenerator::GetClass( const char* name, Bool_t load )
{
// called if all other class generators failed, attempt to build from python class
   if ( PyROOT::gDictLookupActive == kTRUE )
      return 0;                              // call originated from python

   if ( ! load || ! name )
      return 0;

// determine module and class name part
   std::string clName = name;
   std::string::size_type pos = clName.rfind( '.' );

   if ( pos == std::string::npos )
      return 0;                              // this isn't a python style class
   
   std::string mdName = clName.substr( 0, pos );
   clName = clName.substr( pos+1, std::string::npos );

// ROOT doesn't know about python modules; the class may exist (TODO: add scopes)
   if ( gROOT->GetClass( clName.c_str() ) )
      return gROOT->GetClass( clName.c_str() );

// locate and get class
   PyObject* mod = PyImport_AddModule( const_cast< char* >( mdName.c_str() ) );
   if ( ! mod ) {
      PyErr_Clear();
      return 0;                              // module apparently disappeared
   }

   Py_INCREF( mod );
   PyObject* pyclass =
      PyDict_GetItemString( PyModule_GetDict( mod ), const_cast< char* >( clName.c_str() ) );
   Py_XINCREF( pyclass );
   Py_DECREF( mod );

   if ( ! pyclass ) {
      PyErr_Clear();                         // the class is no longer available?!
      return 0;
   }

// build CINT class placeholder
   G__linked_taginfo pti;
   pti.tagnum = -1;
   pti.tagtype = 'c';
   pti.tagname = clName.c_str();
   G__add_compiledheader( (clName+".h").c_str() );

   int tagnum = G__get_linked_tagnum( &pti );

   G__tagtable_setup(
      tagnum, sizeof(PyObject), G__CPPLINK, 0x00020000, "", 0, 0 );

// loop over and add member functions
   PyObject* attrs = PyObject_Dir( pyclass );
   if ( ! attrs ) {
      PyErr_Clear();
      return 0;
   }

   G__tag_memfunc_setup( tagnum );
   G__ClassInfo gcl( tagnum );

// special case: constructor
   G__MethodInfo m = gcl.AddMethod( "", clName.c_str(), "ellipsis", 0, 0, (void*)PyCtorCallback);

   Py_INCREF( pyclass );
   m.SetUserParam( pyclass );

   for ( int i = 0; i < PyList_GET_SIZE( attrs ); ++i ) {
      PyObject* label = PyList_GET_ITEM( attrs, i );
      Py_INCREF( label );
      PyObject* attr = PyObject_GetAttr( pyclass, label );

   // collect only member functions (i.e. callable elements in __dict__)
      if ( PyCallable_Check( attr ) ) {
         std::string mtName = PyString_AS_STRING( label );

         G__MethodInfo m = gcl.AddMethod( "TPyReturn", mtName.c_str(), "ellipsis", 0, 0, (void*)PyMemFuncCallback );

         Py_INCREF( attr );
         m.SetUserParam(attr);

      }

      Py_DECREF( attr );
      Py_DECREF( label );
   }

   G__tag_memfunc_reset();

// done, let ROOT manage the new class
   Py_DECREF( pyclass );

   TClass* klass = new TClass( clName.c_str() );
   gROOT->AddClass( klass );

   return klass;
}

//____________________________________________________________________________
TClass* TPyClassGenerator::GetClass( const type_info& typeinfo, Bool_t load )
{
// just forward, based on name only
   return GetClass( typeinfo.name(), load );
}
