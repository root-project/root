// @(#)root/pyroot:$Name:  $:$Id: Pythonize.cxx,v 1.14 2005/04/28 07:33:55 brun Exp $
// Author: Wim Lavrijsen, Jul 2004

// Bindings
#include "PyROOT.h"
#include "Pythonize.h"
#include "ObjectProxy.h"
#include "MethodProxy.h"
#include "RootWrapper.h"
#include "Utility.h"
#include "PyCallable.h"
#include "PyBufferFactory.h"
#include "FunctionHolder.h"

// ROOT
#include "TClass.h"
#include "TCollection.h"
#include "TSeqCollection.h"
#include "TObject.h"
#include "TFunction.h"

// CINT
#include "Api.h"

// Standard
#include <stdexcept>
#include <string>
#include <stdio.h>
#include <utility>


namespace {

// for convenience
   using namespace PyROOT;

// to prevent compiler warnings about const char* -> char*
   inline PyObject* callPyObjMethod( PyObject* obj, const char* meth )
   {
      return PyObject_CallMethod( obj, const_cast< char* >( meth ), const_cast< char* >( "" ) );
   }

//____________________________________________________________________________
   inline PyObject* callPyObjMethod( PyObject* obj, const char* meth, PyObject* arg1 )
   {
      return PyObject_CallMethod(
         obj, const_cast< char* >( meth ), const_cast< char* >( "O" ), arg1 );
   }

//____________________________________________________________________________
   inline PyObject* callPyObjMethod(
      PyObject* obj, const char* meth, PyObject* arg1, PyObject* arg2 )
   {
      return PyObject_CallMethod(
         obj, const_cast< char* >( meth ), const_cast< char* >( "OO" ), arg1, arg2 );
   }

//____________________________________________________________________________
   inline PyObject* callPyObjMethod( PyObject* obj, const char* meth, PyObject* arg1, int arg2 )
   {
      return PyObject_CallMethod(
         obj, const_cast< char* >( meth ), const_cast< char* >( "Oi" ), arg1, arg2 );
   }


//- helpers --------------------------------------------------------------------
   PyObject* callSelf( PyObject* args, const char* meth, const char* fmt )
   {
      PyObject* self = 0;
      if ( ! PyArg_ParseTuple( args, const_cast< char* >( fmt ), &self ) )
         return 0;

      return callPyObjMethod( self, meth );
   }

//____________________________________________________________________________
   PyObject* callSelfPyObject( PyObject* args, const char* meth, const char* fmt )
   {
      PyObject* self = 0, *obj = 0;
      if ( ! PyArg_ParseTuple( args, const_cast< char* >( fmt ), &self, &obj ) )
         return 0;

      return callPyObjMethod( self, meth, obj );
   }

//____________________________________________________________________________
   PyObject* pyStyleIndex( PyObject* self, PyObject* index )
   {
      long idx = PyInt_AsLong( index );
      if ( PyErr_Occurred() )
         return 0;

      PyObject* pyindex = 0;
      long size = PySequence_Size( self );
      if ( idx >= size || ( idx < 0 && idx <= -size ) ) {
         PyErr_SetString( PyExc_IndexError, "index out of range" );
         return 0;
      }

      if ( idx >= 0 ) {
         Py_INCREF( index );
         pyindex = index;
      } else
         pyindex = PyLong_FromLong( size + idx );

      return pyindex;
   }

//____________________________________________________________________________
   PyObject* callSelfIndex( PyObject* args, const char* meth )
   {
      PyObject* self = PyTuple_GET_ITEM( args, 0 );
      PyObject* obj  = PyTuple_GET_ITEM( args, 1 );

      PyObject* pyindex = pyStyleIndex( self, obj );
      if ( ! pyindex )
         return 0;

      PyObject* result = callPyObjMethod( self, meth, pyindex );
      Py_DECREF( pyindex );
      return result;
   }


//- TObject behaviour ----------------------------------------------------------
   PyObject* contains( PyObject*, PyObject* args )
   {
      PyObject* self = 0, *obj = 0;
      if ( ! PyArg_ParseTuple( args, const_cast< char* >( "OO:__contains__" ), &self, &obj ) )
         return 0;

      if ( ! ( ObjectProxy_Check( obj ) || PyString_Check( obj ) ) )
         return PyInt_FromLong( 0l );

      PyObject* found = callPyObjMethod( self, "FindObject", obj );
      PyObject* result = PyInt_FromLong( PyObject_IsTrue( found ) );
      Py_DECREF( found );
      return result;
   }

//____________________________________________________________________________
   PyObject* compare( PyObject*, PyObject* args )
   {
      PyObject* self = 0, *obj = 0;
      if ( ! PyArg_ParseTuple( args, const_cast< char* >( "OO:__cmp__" ), &self, &obj ) )
         return 0;

      if ( ! ObjectProxy_Check( obj ) )
         return PyInt_FromLong( -1l );

      return callPyObjMethod( self, "Compare", obj );
   }

//____________________________________________________________________________
   PyObject* isequal( PyObject*, PyObject* args )
   {
      PyObject* self = 0, *obj = 0;
      if ( ! PyArg_ParseTuple( args, const_cast< char* >( "OO:__eq__" ), &self, &obj ) )
         return 0;

      if ( ! ObjectProxy_Check( obj ) )
         return PyInt_FromLong( 0l );

      return callPyObjMethod( self, "IsEqual", obj );
   }


//- TCollection behaviour ------------------------------------------------------
   PyObject* collectionExtend( PyObject*, PyObject* args )
   {
      PyObject* self = 0, *obj = 0;
      if ( ! PyArg_ParseTuple( args, const_cast< char* >( "OO:extend" ), &self, &obj ) )
         return 0;

      for ( int i = 0; i < PySequence_Size( obj ); ++i ) {
         PyObject* item = PySequence_GetItem( obj, i );
         PyObject* result = callPyObjMethod( self, "Add", item );
         Py_XDECREF( result );
         Py_DECREF( item );
      }

      Py_INCREF( Py_None );
      return Py_None;
   }

//____________________________________________________________________________
   PyObject* collectionRemove( PyObject*, PyObject* args )
   {
      PyObject* result = callSelfPyObject( args, "Remove", "OO:remove" );
      if ( ! result )
         return 0;

      if ( ! PyObject_IsTrue( result ) ) {
         Py_DECREF( result );
         PyErr_SetString( PyExc_ValueError, "list.remove(x): x not in list" );
         return 0;
      }

      Py_DECREF( result );
      Py_INCREF( Py_None );
      return Py_None;
   }

//____________________________________________________________________________
   PyObject* collectionAdd( PyObject*, PyObject* args )
   {
      PyObject* self = 0, *other = 0;
      if ( ! PyArg_ParseTuple( args, const_cast< char* >( "OO:__add__" ), &self, &other ) )
         return 0;

      PyObject* l = callPyObjMethod( self, "Clone" );
      if ( ! l )
         return 0;

      PyObject* result = callPyObjMethod( l, "extend", other );
      if ( ! result ) {
         Py_DECREF( l );
         return 0;
      }

      return l;
   }

//____________________________________________________________________________
   PyObject* collectionMul( PyObject*, PyObject* args )
   {
      ObjectProxy* self = 0; long imul = 0;
      if ( ! PyArg_ParseTuple( args, const_cast< char* >( "Ol:__mul__" ), &self, &imul ) )
         return 0;

      if ( ! ObjectProxy_Check( self ) || ! self->GetObject() ) {
         PyErr_SetString( PyExc_TypeError, "unsubscriptable object" );
         return 0;
      }

      PyObject* nseq = BindRootObject( self->ObjectIsA()->New(), self->ObjectIsA() );

      for ( long i = 0; i < imul; ++i ) {
         PyObject* result = callPyObjMethod( nseq, "extend", (PyObject*)self );
         Py_DECREF( result );
      }

      return nseq;
   }

//____________________________________________________________________________
   PyObject* collectionIMul( PyObject*, PyObject* args )
   {
      PyObject* self = 0; long imul = 0;
      if ( ! PyArg_ParseTuple( args, const_cast< char* >( "Ol:__imul__" ), &self, &imul ) )
         return 0;

      PyObject* l = PySequence_List( self );

      for ( long i = 0; i < imul - 1; ++i ) {
         callPyObjMethod( self, "extend", l );
      }

      Py_INCREF( self );
      return self;
   }

//____________________________________________________________________________
   PyObject* collectionCount( PyObject*, PyObject* args )
   {
      PyObject* self = 0, *obj = 0;
      if ( ! PyArg_ParseTuple( args, const_cast< char* >( "OO:count" ), &self, &obj ) )
         return 0;

      int count = 0;
      for ( int i = 0; i < PySequence_Size( self ); ++i ) {
         PyObject* item = PySequence_GetItem( self, i );
         PyObject* found = PyObject_RichCompare( item, obj, Py_EQ );

         Py_DECREF( item );

         if ( ! found )
            return 0;                        // internal problem

         if ( PyObject_IsTrue( found ) )
            count += 1;
         Py_DECREF( found );
      }

      return PyLong_FromLong( count );
   }


//____________________________________________________________________________
   PyObject* collectionIter( PyObject*, PyObject* args ) {
      ObjectProxy* self = (ObjectProxy*) PyTuple_GET_ITEM( args, 0 );
      if ( ! ObjectProxy_Check( self ) || ! self->GetObject() ) {
         PyErr_SetString( PyExc_TypeError, "iteration over non-sequence" );
         return 0;
      }

      TCollection* col =
         (TCollection*) self->ObjectIsA()->DynamicCast( TCollection::Class(), self->GetObject() );

      return BindRootObject( (void*) new TIter( col ), TIter::Class() );
   }


//- TSeqCollection behaviour ---------------------------------------------------
   PyObject* seqCollectionGetItem( PyObject*, PyObject* args )
   {
      ObjectProxy* self = 0; PySliceObject* index = 0;
      if ( ! PyArg_ParseTuple( args, const_cast< char* >( "OO:__getitem__" ), &self, &index ) )
         return 0;
      
      if ( PySlice_Check( index ) ) {
         if ( ! ObjectProxy_Check( self ) || ! self->GetObject() ) {
            PyErr_SetString( PyExc_TypeError, "unsubscriptable object" );
            return 0;
         }

         TClass* clSeq = self->ObjectIsA();
         TSeqCollection* oseq =
            (TSeqCollection*) clSeq->DynamicCast( TSeqCollection::Class(), self->GetObject() );
         TSeqCollection* nseq = (TSeqCollection*) clSeq->New();

         int start, stop, step;
         PySlice_GetIndices( index, oseq->GetSize(), &start, &stop, &step );
         for ( int i = start; i < stop; i += step ) {
            nseq->Add( oseq->At( i ) );
         }

         return BindRootObject( (void*) nseq, clSeq );
      }

      return callSelfIndex( args, "At" );
   }

//____________________________________________________________________________
   PyObject* seqCollectionSetItem( PyObject*, PyObject* args )
   {
      ObjectProxy* self = 0; PyObject* index = 0, *obj = 0;
      if ( ! PyArg_ParseTuple( args,
                const_cast< char* >( "OOO:__setitem__" ), &self, &index, &obj ) )
         return 0;

      if ( PySlice_Check( index ) ) {
         if ( ! ObjectProxy_Check( self ) || ! self->GetObject() ) {
            PyErr_SetString( PyExc_TypeError, "unsubscriptable object" );
            return 0;
         }

         TSeqCollection* oseq = (TSeqCollection*) self->ObjectIsA()->DynamicCast(
            TSeqCollection::Class(), self->GetObject() );

         int start, stop, step;
         PySlice_GetIndices( (PySliceObject*) index, oseq->GetSize(), &start, &stop, &step );
         for ( int i = stop - step; i >= start; i -= step ) {
            oseq->RemoveAt( i );
         }

         for ( int i = 0; i < PySequence_Size( obj ); ++i ) {
            ObjectProxy* item = (ObjectProxy*)PySequence_GetItem( obj, i );
            item->Release();
            oseq->AddAt( (TObject*) item->GetObject(), i + start );
            Py_DECREF( item );
         }

         Py_INCREF( Py_None );
         return Py_None;
      }

      PyObject* pyindex = pyStyleIndex( (PyObject*)self, index );
      if ( ! pyindex )
         return 0;

      PyObject* result  = callPyObjMethod( (PyObject*)self, "RemoveAt", pyindex );
      if ( ! result ) {
         Py_DECREF( pyindex );
         return 0;
      }

      Py_DECREF( result );
      result = callPyObjMethod( (PyObject*)self, "AddAt", obj, pyindex );
      Py_DECREF( pyindex );
      return result;
   }

//____________________________________________________________________________
   PyObject* seqCollectionDelItem( PyObject*, PyObject* args )
   {
      ObjectProxy* self = 0; PySliceObject* index = 0;
      if ( ! PyArg_ParseTuple( args, const_cast< char* >( "OO:__del__" ), &self, &index ) )
         return 0;

      if ( PySlice_Check( index ) ) {
         if ( ! ObjectProxy_Check( self ) || ! self->GetObject() ) {
            PyErr_SetString( PyExc_TypeError, "unsubscriptable object" );
            return 0;
         }

         TSeqCollection* oseq = (TSeqCollection*) self->ObjectIsA()->DynamicCast(
            TSeqCollection::Class(), self->GetObject() );

         int start, stop, step;
         PySlice_GetIndices( index, oseq->GetSize(), &start, &stop, &step );
         for ( int i = stop - step; i >= start; i -= step ) {
            oseq->RemoveAt( i );
         }

         Py_INCREF( Py_None );
         return Py_None;
      }

      PyObject* result = callSelfIndex( args, "RemoveAt" );
      if ( ! result )
         return 0;

      Py_DECREF( result );
      Py_INCREF( Py_None );
      return Py_None;
   }

//____________________________________________________________________________
   PyObject* seqCollectionInsert( PyObject*, PyObject* args )
   {
      PyObject* self = 0, *obj = 0; long idx = 0;
      if ( ! PyArg_ParseTuple( args, const_cast< char* >( "OlO:insert" ), &self, &idx, &obj ) )
         return 0;

      int size = PySequence_Size( self );
      if ( idx < 0 )
         idx = 0;
      else if ( size < idx )
         idx = size;

      return callPyObjMethod( self, "AddAt", obj, idx );
   }

//____________________________________________________________________________
   PyObject* seqCollectionPop( PyObject*, PyObject* args )
   {
      if ( PyTuple_GET_SIZE( args ) == 1 ) {
         PyObject* self = PyTuple_GET_ITEM( args, 0 );
         Py_INCREF( self );

         args = PyTuple_New( 2 );
         PyTuple_SET_ITEM( args, 0, self );
         PyTuple_SET_ITEM( args, 1, PyLong_FromLong( PySequence_Size( self ) - 1 ) );
      }

      return callSelfIndex( args, "RemoveAt" );
   }

//____________________________________________________________________________
   PyObject* seqCollectionReverse( PyObject*, PyObject* args )
   {
      PyObject* self = 0;
      if ( ! PyArg_ParseTuple( args, const_cast< char* >( "O:reverse" ), &self ) )
         return 0;

      PyObject* tup = PySequence_Tuple( self );
      if ( ! tup )
         return 0;

      PyObject* result = callPyObjMethod( self, "Clear" );
      Py_XDECREF( result );

      for ( int i = 0; i < PySequence_Size( tup ); ++i ) {
         PyObject* result = callPyObjMethod( self, "AddAt", PyTuple_GET_ITEM( tup, i ), 0 );
         Py_XDECREF( result );
      }

      Py_INCREF( Py_None );
      return Py_None;
   }

//____________________________________________________________________________
   PyObject* seqCollectionSort( PyObject*, PyObject* args )
   {
      PyObject* self = PyTuple_GET_ITEM( args, 0 );

      if ( PyTuple_GET_SIZE( args ) == 1 ) {
      // no specialized sort, use ROOT one
         return callPyObjMethod( self, "Sort" );
      } else {
      // sort in a python list copy
         PyObject* l = PySequence_List( self );
         PyObject* result = callPyObjMethod( l, "sort", PyTuple_GET_ITEM( args, 1 ) );
         Py_XDECREF( result );
         if ( PyErr_Occurred() ) {
            Py_DECREF( l );
            return 0;
         }

         result = callPyObjMethod( self, "Clear" );
         Py_XDECREF( result );
         result = callPyObjMethod( self, "extend", l );
         Py_XDECREF( result );
         Py_DECREF( l );

         Py_INCREF( Py_None );
         return Py_None;
      }
   }

//____________________________________________________________________________
   PyObject* seqCollectionIndex( PyObject*, PyObject* args )
   {
      PyObject* index = callSelfPyObject( args, "IndexOf", "OO:index" );
      if ( ! index )
         return 0;

      if ( PyLong_AsLong( index ) < 0 ) {
         Py_DECREF( index );
         PyErr_SetString( PyExc_ValueError, "list.index(x): x not in list" );
         return 0;
      }

      return index;
   }


//- TString behaviour ----------------------------------------------------------
   PyObject* stringRepr( PyObject*, PyObject* args )
   {
      PyObject* data = callPyObjMethod( PyTuple_GET_ITEM( args, 0 ), "Data" );
      PyObject* repr = PyString_FromFormat( "\'%s\'", PyString_AsString( data ) );
      Py_DECREF( data );
      return repr;
   }


//- TObjString behaviour -------------------------------------------------------
   PyObject* objStringRepr( PyObject*, PyObject* args )
   {
      PyObject* data = callPyObjMethod( PyTuple_GET_ITEM( args, 0 ), "GetName" );
      PyObject* repr = PyString_FromFormat( "\'%s\'", PyString_AsString( data ) );
      Py_DECREF( data );
      return repr;
   }

//____________________________________________________________________________
   PyObject* objStringCompare( PyObject*, PyObject* args )
   {
      PyObject* data = callPyObjMethod( PyTuple_GET_ITEM( args, 0 ), "GetName" );
      int result = PyObject_Compare( data, PyTuple_GET_ITEM( args, 1 ) );
      Py_DECREF( data );

      if ( PyErr_Occurred() )
         return 0;

      return PyInt_FromLong( result );
   }

//____________________________________________________________________________
   PyObject* objStringIsequal( PyObject*, PyObject* args )
   {
      PyObject* data = callPyObjMethod( PyTuple_GET_ITEM( args, 0 ), "GetName" );
      PyObject* result = PyObject_RichCompare( data, PyTuple_GET_ITEM( args, 1 ), Py_EQ );
      Py_DECREF( data );

      if ( ! result )
         return 0;

      return result;
   }

//____________________________________________________________________________
   PyObject* objStringLength( PyObject*, PyObject* args )
   {
      PyObject* data = callPyObjMethod( PyTuple_GET_ITEM( args, 0 ), "GetName" );
      int size = PySequence_Size( data );
      Py_DECREF( data );
      return PyInt_FromLong( size );
   }


//- TIter behaviour ------------------------------------------------------------
   PyObject* iterIter( PyObject*, PyObject* args )
   {
      PyObject* iter = PyTuple_GET_ITEM( args, 0 );
      Py_INCREF( iter );
      return iter;
   }

//____________________________________________________________________________
   PyObject* iterNext( PyObject*, PyObject* args )
   {
      PyObject* next = callSelf( args, "Next", "O:next" );

      if ( ! next )
         return 0;

      if ( ! PyObject_IsTrue( next ) ) {
         Py_DECREF( next );
         PyErr_SetString( PyExc_StopIteration, "" );
         return 0;
      }

      return next;
   }


//- TTree behaviour ------------------------------------------------------------
   class TreeEraser : public TObject {
   public:
      TreeEraser( PyObject* ttree ) : fTree( ttree ) {}

      virtual Bool_t Notify()
      {
         if ( ! fTree )
            return kFALSE;

      // "reset" the dictionary by replacing it with an empty one
         PyObject* dict = PyDict_New();
         PyObject_SetAttrString( fTree, const_cast< char* >( "__dict__" ), dict );
         Py_DECREF( dict );

         return kTRUE;
      }
   private:
      PyObject* fTree;
   };

//____________________________________________________________________________
   PyObject* treeGetAttr( PyObject*, PyObject* args )
   {
      PyObject* self = 0, *name = 0;
      if ( ! PyArg_ParseTuple( args, const_cast< char* >( "OO:__getattr__" ), &self, &name ) )
         return 0;

   // setup notification as needed
      PyObject* notify = callPyObjMethod( self, "GetNotify" );
      if ( PyObject_Not( notify ) ) {
         TObject* te = new TreeEraser( self );
         Py_XDECREF( callPyObjMethod( self, "SetNotify", BindRootObject( te, te->IsA() ) ) );
      }
      Py_DECREF( notify );

   // allow access to leaves as if they are data members
      PyObject* leaf = callPyObjMethod( self, "GetLeaf", name );
      if ( ! leaf )
         return 0;

      if ( leaf != Py_None ) {
         PyObject* value = 0;

      // found a leaf, extract value if just one, or wrap buffer if more
         PyObject* lcount = callPyObjMethod( leaf, "GetLeafCount" );

         if ( lcount == Py_None ) {
            value = callPyObjMethod( leaf, "GetValue" );
         } else {
            PyObject* ptr = callPyObjMethod( leaf, "GetValuePointer" );
            void* arr = PyLong_AsVoidPtr( ptr );
            Py_DECREF( ptr );

            PyObject* tname = callPyObjMethod( leaf, "GetTypeName" );
            std::string stname( PyString_AS_STRING( tname ) );

            PyBufferFactory* fac = PyBufferFactory::Instance();
            PyObject* scb = PyObject_GetAttrString( leaf, const_cast< char* >( "GetNdata" ) );

            Utility::EDataType eType = Utility::effectiveType( stname );
            if ( eType == Utility::kLong )
               value = fac->PyBuffer_FromMemory( (long*) arr, scb );
            else if ( eType == Utility::kInt )
               value = fac->PyBuffer_FromMemory( (int*) arr, scb );
            else if ( eType == Utility::kDouble )
               value = fac->PyBuffer_FromMemory( (double*) arr, scb );
            else if ( eType == Utility::kFloat )
               value = fac->PyBuffer_FromMemory( (float*) arr, scb );

            Py_DECREF( scb );

         // we're working with addresses: cache result
            PyObject_SetAttr( self, name, value );
         }

         Py_DECREF( lcount );
         Py_DECREF( leaf );

         return value;
      }

   // confused
      Py_DECREF( leaf );
      char txt[ 256 ];
      sprintf( txt, "no such attribute \'%s\'", PyString_AsString( name ) );
      PyErr_SetString( PyExc_AttributeError, txt );
      return 0;
   }


//- TF1 behaviour --------------------------------------------------------------
   std::map< int, std::pair< PyObject*, int > > gPyObjectCallbacks;
   typedef std::pair< PyObject*, int > CallInfo_t;

   int gLastTag = 99;
   CallInfo_t* gLastCallInfo = 0;

   int pyFuncCallback( G__value* res, G__CONST char*, struct G__param* libp, int hash )
   {
      PyObject* result = 0;

   // retrieve function information
      int tag = res->tagnum;
      if ( gLastTag != tag || ! gLastCallInfo ) {
         G__ifunc_table* ifunc = 0;
         int index = 0;
         G__CurrentCall( G__RECMEMFUNCENV, &ifunc, index );

         gLastCallInfo = (CallInfo_t*)ifunc->userparam[index];
         gLastTag = tag;
      }

      if ( gLastCallInfo->first != 0 ) {
      // prepare arguments and call
         PyObject* arg1 = PyBufferFactory::Instance()->PyBuffer_FromMemory(
            (double*)G__int(libp->para[0]), 4 );

         if ( gLastCallInfo->second != 0 ) {
            PyObject* arg2 = PyBufferFactory::Instance()->PyBuffer_FromMemory(
               (double*)G__int(libp->para[1]), gLastCallInfo->second );

            result = PyObject_CallFunction(
               gLastCallInfo->first, const_cast< char* >( "OO" ), arg1, arg2 );

            Py_DECREF( arg2 );
         } else {
            result = PyObject_CallFunction(
               gLastCallInfo->first, const_cast< char* >( "O" ), arg1 );
         }

         Py_DECREF( arg1 );
      }

   // translate result, throw if an error has occurred
      double d = 0.;
      if ( ! result ) {
         PyErr_Print();
         throw std::runtime_error( "TF1 python function call failed" );
      } else {
         d = PyFloat_AsDouble( result );
         Py_DECREF( result );
      }

      G__letdouble( res, 100, d );
      return ( 1 || hash || res || libp );
   }

//____________________________________________________________________________
   class TF1InitWithPyFunc : public PyCallable {
      static int fgCount;

   public:
      virtual PyObject* GetDocString()
      {
         return PyString_FromString(
            "TF1::TF1(const char* name, PyObject* callable, "
            "Double_t xmin, Double_t xmax, Int_t npar = 0)" );
      }

   public:
      virtual PyObject* operator()( ObjectProxy* self, PyObject* args, PyObject* )
      {
      // expected signature: ( char* name, pyfunc, double xmin, double xmax, int npar = 0 )
         int argc = PyTuple_GET_SIZE( args );
         if ( ! ( argc == 4 || argc == 5 ) )
            return 0;              // reported as an overload failure

         PyObject* pyfunc = PyTuple_GET_ITEM( args, 1 );
         if ( ! pyfunc || ! PyCallable_Check( pyfunc ) ) {
            PyErr_SetString( PyExc_ValueError, "not a valid python callable" );
            return 0;
         }

      // use requested function name as identifier
         const char* name = PyString_AsString( PyTuple_GET_ITEM( args, 0 ) );
         if ( PyErr_Occurred() )
            return 0;

      // offset counter
         fgCount += 1;

      // build CINT function placeholder
         G__lastifuncposition();

         G__ClassInfo gcl;
         gcl.AddMethod( "D", name, "double*, double*" );    // boundary safe (call first)

         G__memfunc_setup(                                  // not boundary safe!
            name, 444, (G__InterfaceMethod)NULL,
            100, -1, -1, 0, 2, 1, 1, 0, "D - - 0 - - D - - 0 - -",
            (char*)NULL, (void*)((long)this + fgCount), 0 );
         G__resetifuncposition();

         long offset = 0;
         G__MethodInfo m = gcl.GetMethod( name, "double*, double*", &offset );

         G__ifunc_table* ifunc = m.ifunc();
         int index = m.Index();

         ifunc->pentry[index]->size        = -1;
         ifunc->pentry[index]->filenum     = -1;
         ifunc->pentry[index]->line_number = -1;
         ifunc->pentry[index]->tp2f = (void*)pyFuncCallback;
         ifunc->pentry[index]->p    = (void*)pyFuncCallback;

      // setup association
         int tag = -1 - fgCount;
         ifunc->p_tagtable[index] = tag;
         Py_INCREF( pyfunc );

         int npar = 0;             // default value if not given
         if ( argc == 5 )
            npar = PyInt_AsLong( PyTuple_GET_ITEM( args, 4 ) );

         ifunc->userparam[index] = (void*) new std::pair< PyObject*, int >( pyfunc, npar );

      // get constructor
         MethodProxy* method = (MethodProxy*)PyObject_GetAttrString(
            (PyObject*)self, const_cast< char* >( "__init__" ) );

      // build new argument array
         PyObject* newArgs = PyTuple_New( 5 );

         for ( int iarg = 0; iarg < argc; ++iarg ) {
            PyObject* item = PyTuple_GET_ITEM( args, iarg );
            if ( iarg != 1 ) {
               Py_INCREF( item );
               PyTuple_SET_ITEM( newArgs, iarg, item );
            } else {
               PyTuple_SET_ITEM( newArgs, iarg,
                  BindRootObjectNoCast( (void*)((long)this + fgCount), TObject::Class() ) );
            }
         }

         if ( argc == 4 )
            PyTuple_SET_ITEM( args, 4, PyInt_FromLong( 0l ) );

      // re-run
         PyObject* result = PyObject_CallObject( (PyObject*)method, newArgs );

      // done, may have worked, if not: 0 is returned
         Py_DECREF( newArgs );
         Py_DECREF( method );
         return result;
      }
   };

   int TF1InitWithPyFunc::fgCount = 0;


//- TFunction behaviour --------------------------------------------------------
   PyObject* functionCall( PyObject*, PyObject* args ) {
      if ( PyTuple_GET_SIZE( args ) < 1 || ! ObjectProxy_Check( PyTuple_GET_ITEM( args, 0 ) ) ) {
         PyErr_SetString( PyExc_TypeError,
            "unbound method __call__ requires TFunction instance as first argument" );
         return 0;
      }

      return FunctionHolder(
         (TFunction*)((ObjectProxy*)PyTuple_GET_ITEM( args, 0 ))->GetObject() )( 0, args, 0 );
   }

} // unnamed namespace


//- public functions -----------------------------------------------------------
bool PyROOT::Pythonize( PyObject* pyclass, const std::string& name )
{
   if ( pyclass == 0 )
      return false;

   if ( name == "TObject" ) {
   // support for the 'in' operator
      Utility::AddToClass( pyclass, "__contains__", (PyCFunction) contains );

   // comparing for lists
      Utility::AddToClass( pyclass, "__cmp__", (PyCFunction) compare );
      Utility::AddToClass( pyclass, "__eq__",  (PyCFunction) isequal );

      return true;
   }

   if ( name == "TCollection" ) {
      Utility::AddToClass( pyclass, "append",   "Add" );
      Utility::AddToClass( pyclass, "extend",   (PyCFunction) collectionExtend );
      Utility::AddToClass( pyclass, "remove",   (PyCFunction) collectionRemove );
      Utility::AddToClass( pyclass, "__add__",  (PyCFunction) collectionAdd );
      Utility::AddToClass( pyclass, "__imul__", (PyCFunction) collectionIMul );
      Utility::AddToClass( pyclass, "__mul__",  (PyCFunction) collectionMul );
      Utility::AddToClass( pyclass, "__rmul__", (PyCFunction) collectionMul );

      Utility::AddToClass( pyclass, "count", (PyCFunction) collectionCount );

      Utility::AddToClass( pyclass, "__len__",  "GetSize" );
      Utility::AddToClass( pyclass, "__iter__", (PyCFunction) collectionIter );

      return true;
   }

   if ( name == "TSeqCollection" ) {
      Utility::AddToClass( pyclass, "__getitem__", (PyCFunction) seqCollectionGetItem );
      Utility::AddToClass( pyclass, "__setitem__", (PyCFunction) seqCollectionSetItem );
      Utility::AddToClass( pyclass, "__delitem__", (PyCFunction) seqCollectionDelItem );

      Utility::AddToClass( pyclass, "insert",  (PyCFunction) seqCollectionInsert );
      Utility::AddToClass( pyclass, "pop",     (PyCFunction) seqCollectionPop );
      Utility::AddToClass( pyclass, "reverse", (PyCFunction) seqCollectionReverse );
      Utility::AddToClass( pyclass, "sort",    (PyCFunction) seqCollectionSort );

      Utility::AddToClass( pyclass, "index", (PyCFunction) seqCollectionIndex );

      return true;
   }

   if ( name == "TString" ) {
   // ROOT pointer validity testing
      Utility::AddToClass( pyclass, "__repr__", (PyCFunction) stringRepr );
      Utility::AddToClass( pyclass, "__str__", "Data" );
      Utility::AddToClass( pyclass, "__len__", "Length" );

      Utility::AddToClass( pyclass, "__cmp__", "CompareTo" );

      return true;
   }

   if ( name == "TObjString" ) {
      Utility::AddToClass( pyclass, "__repr__", (PyCFunction) objStringRepr );
      Utility::AddToClass( pyclass, "__str__",  "GetName" );
      Utility::AddToClass( pyclass, "__len__",  (PyCFunction) objStringLength );

      Utility::AddToClass( pyclass, "__cmp__", (PyCFunction) objStringCompare );
      Utility::AddToClass( pyclass, "__eq__",  (PyCFunction) objStringIsequal );

      return true;
   }

   if ( name == "TIter" ) {
   // ROOT pointer validity testing
      Utility::AddToClass( pyclass, "__iter__", (PyCFunction) iterIter );
      Utility::AddToClass( pyclass, "next",     (PyCFunction) iterNext );

      return true;
   }

   if ( name == "TTree" ) {
   // allow direct browsing of the tree
      Utility::AddToClass( pyclass, "__getattr__", (PyCFunction) treeGetAttr );
   }

   if ( name == "TF1" ) {
   // allow instantiation with python function
      MethodProxy* method = (MethodProxy*)PyObject_GetAttrString(
         pyclass, const_cast< char* >( "__init__" ) );

      method->AddMethod( new TF1InitWithPyFunc() );

      Py_DECREF( method );
      return true;
   }

   if ( name == "TFunction" ) {
   // allow direct call
      Utility::AddToClass( pyclass, "__call__", (PyCFunction) functionCall );
   }

   return true;
}
