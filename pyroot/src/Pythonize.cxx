// Author: Wim Lavrijsen, Jul 2004

// Bindings
#include "PyROOT.h"
#include "Pythonize.h"
#include "ObjectHolder.h"
#include "RootWrapper.h"
#include "Utility.h"
#include "PyCallable.h"
#include "MethodDispatcher.h"
#include "PyBufferFactory.h"

// ROOT
#include "TClass.h"
#include "TCollection.h"
#include "TSeqCollection.h"
#include "TObject.h"

// CINT
#include "Api.h"

// Standard
#include <stdexcept>
#include <string>
#include <iostream>
#include <stdio.h>
#include <map>
#include <utility>


namespace {

// for convenience
   using namespace PyROOT;

//- helpers --------------------------------------------------------------------
   bool checkNArgs( PyObject* aTuple, const int nArgs, const char* pyname ) {
      int nGivenArgs = PyTuple_GET_SIZE( aTuple );

      if ( nGivenArgs != nArgs ) {
         char buf[ 256 ];
         const char* ms = "";
         if ( nArgs == 1 )
            ms = "s";

         snprintf( buf, 256, "%s() takes exactly %d argument%s (%d given)",
            pyname, nArgs, ms, nGivenArgs );

         PyErr_SetString( PyExc_TypeError, buf );

         return false;
      }

      return true;
   }

   PyObject* callSelf( PyObject* aTuple, char* name, const char* pyname ) {
      if ( checkNArgs( aTuple, 1, pyname ) == false )
         return 0;

      return PyObject_CallMethod( PyTuple_GET_ITEM( aTuple, 0 ), name, "" );
   }

   PyObject* callSelfPyObject( PyObject* aTuple, char* name, const char* pyname ) {
      if ( checkNArgs( aTuple, 2, pyname ) == false )
         return 0;

      PyObject* self = PyTuple_GET_ITEM( aTuple, 0 );
      PyObject* obj  = PyTuple_GET_ITEM( aTuple, 1 );

      return PyObject_CallMethod( self, name, "O", obj );
   }

   PyObject* pyStyleIndex( PyObject* self, PyObject* index ) {
      int idx = (int) PyInt_AsLong( index );
      if ( PyErr_Occurred() )
         return 0;

      PyObject* pyindex = 0;
      int size = PySequence_Size( self );
      if ( idx >= size || ( idx < 0 && idx <= -size ) ) {
         PyErr_SetString( PyExc_IndexError, "index out of range" );
         return 0;
      }

      if ( idx >= 0 ) {
         Py_INCREF( index );
         pyindex = index;
      }
      else {
         pyindex = PyLong_FromLong( size + idx );
      }

      return pyindex;
   }

   PyObject* callSelfIndex( PyObject* aTuple, char* name ) {
      PyObject* self = PyTuple_GET_ITEM( aTuple, 0 );
      PyObject* obj  = PyTuple_GET_ITEM( aTuple, 1 );

      PyObject* pyindex = pyStyleIndex( self, obj );
      if ( ! pyindex )
         return 0;

      PyObject* result = PyObject_CallMethod( self, name, "O", pyindex );
      Py_DECREF( pyindex );
      return result;
   }

   PyObject* callSelfTObject( PyObject* aTuple, char* name, const std::string& pyname ) {
      if ( checkNArgs( aTuple, 2, pyname.c_str() ) == false )
         return 0;

      PyObject* self = PyTuple_GET_ITEM( aTuple, 0 );
      PyObject* obj  = PyTuple_GET_ITEM( aTuple, 1 );

      if ( ! Utility::getObjectHolder( obj ) ) {
         PyErr_SetString( PyExc_TypeError,
                          ( pyname + "() requires a ROOT object as argument" ).c_str() );
         return 0;
      }

      return PyObject_CallMethod( self, name, "O", obj );
   }


//- TObject behaviour ----------------------------------------------------------
   PyObject* isZero( PyObject* /* None */, PyObject* aTuple ) {
   // get a hold of the object and test it
      void* obj = Utility::getObjectFromHolderFromArgs( aTuple );
      long isz = obj == 0 ? 1l /* yes, is zero */ : 0l;
      return PyInt_FromLong( isz );
   }

   PyObject* isNotZero( PyObject* /* None */, PyObject* aTuple ) {
   // test for non-zero is opposite of test for zero
      void* obj = Utility::getObjectFromHolderFromArgs( aTuple );
      long isnz = obj != 0 ? 1l /* yes, is not zero */ : 0l;
      return PyInt_FromLong( isnz );
   }

   PyObject* contains( PyObject* /* None */, PyObject* aTuple ) {
      if ( PyTuple_GET_SIZE( aTuple ) != 2 ) {
         PyErr_SetString( PyExc_TypeError, "__contains__() takes exactly 2 arguments" );
         return 0;
      }

      PyObject* self = PyTuple_GET_ITEM( aTuple, 0 );
      PyObject* obj  = PyTuple_GET_ITEM( aTuple, 1 );

      if ( ! ( Utility::getObjectHolder( obj ) || PyString_Check( obj ) ) ) {
         PyErr_SetString( PyExc_TypeError, "__contains__() requires a ROOT object or a string" );
         return 0;
      }

      PyObject* found = PyObject_CallMethod( self, "FindObject", "O", obj );
      PyObject* result = PyInt_FromLong( PyObject_IsTrue( found ) );
      Py_DECREF( found );
      return result;
   }


//- TCollection behaviour ------------------------------------------------------
   PyObject* collectionAppend( PyObject* /* None */, PyObject* aTuple ) {
      return callSelfTObject( aTuple, "Add", "append" );
   }

   PyObject* collectionExtend( PyObject* /* None */, PyObject* aTuple ) {
      if ( checkNArgs( aTuple, 2, "extend" ) == false )
         return 0;

      PyObject* self = PyTuple_GET_ITEM( aTuple, 0 );
      PyObject* obj  = PyTuple_GET_ITEM( aTuple, 1 );

      for ( int i = 0; i < PySequence_Size( obj ); ++i ) {
         PyObject* item = PySequence_GetItem( obj, i );
         Py_DECREF( PyObject_CallMethod( self, "Add", "O", item ) );
         Py_DECREF( item );
      }

      Py_INCREF( Py_None );
      return Py_None;
   }

   PyObject* collectionRemove( PyObject* /* None */, PyObject* aTuple ) {
      PyObject* result = callSelfTObject( aTuple, "Remove", "remove" );

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

   PyObject* collectionAdd( PyObject* /* None */, PyObject* aTuple ) {
      if ( checkNArgs( aTuple, 2, "__add__" ) == false )
         return 0;

      PyObject* self  = PyTuple_GET_ITEM( aTuple, 0 );
      PyObject* other = PyTuple_GET_ITEM( aTuple, 1 );

      PyObject* l = PyObject_CallMethod( self, "Clone", "" );
      if ( ! l )
         return 0;

      PyObject* result = PyObject_CallMethod( l, "extend", "O", other );
      if ( ! result ) {
         Py_DECREF( l );
         return 0;
      }

      return l;
   }

   PyObject* collectionMul( PyObject* /* None */, PyObject* aTuple ) {
      if ( checkNArgs( aTuple, 2, "__mul__" ) == false )
         return 0;

      PyObject* self = PyTuple_GET_ITEM( aTuple, 0 );
      PyObject* imul = PyTuple_GET_ITEM( aTuple, 1 );

      ObjectHolder* obh = Utility::getObjectHolder( self );
      if ( ! obh->getObject() ) {
         PyErr_SetString( PyExc_TypeError, "unsubscriptable object" );
         return 0;
      }

      PyObject* nseq = bindRootObject(
         new ObjectHolder( obh->objectIsA()->New(), obh->objectIsA() ) );

      for ( int i = 0; i < (int) PyLong_AsLong( imul ); ++i ) {
         Py_DECREF( PyObject_CallMethod( nseq, "extend", "O", self ) );
      }

      return nseq;
   }

   PyObject* collectionIMul( PyObject* /* None */, PyObject* aTuple ) {
      if ( checkNArgs( aTuple, 2, "__imul__" ) == false )
         return 0;

      PyObject* self = PyTuple_GET_ITEM( aTuple, 0 );
      PyObject* imul = PyTuple_GET_ITEM( aTuple, 1 );

      PyObject* l = PySequence_List( self );

      for ( int i = 0; i < (int) PyLong_AsLong( imul ) - 1; ++i ) {
         PyObject_CallMethod( self, "extend", "O", l );
      }

      Py_INCREF( self );
      return self;
   }

   PyObject* collectionCount( PyObject* /* None */, PyObject* aTuple ) {
      if ( checkNArgs( aTuple, 2, "count" ) == false )
         return 0;

      PyObject* self = PyTuple_GET_ITEM( aTuple, 0 );
      PyObject* obj  = PyTuple_GET_ITEM( aTuple, 1 );

      int count = 0;
      for ( int i = 0; i < PySequence_Size( self ); ++i ) {
         PyObject* item = PySequence_GetItem( self, i );
         PyObject* found = PyObject_CallMethod( item, "IsEqual", "O", obj );
         if ( PyObject_IsTrue( found ) )
            count += 1;
         Py_DECREF( found );
         Py_DECREF( item );
      }

      return PyLong_FromLong( count );
   }

   PyObject* collectionLength( PyObject* /* None */, PyObject* aTuple ) {
      return callSelf( aTuple, "GetSize", "__len__" );
   }

   PyObject* collectionIter( PyObject* /* None */, PyObject* aTuple ) {
      ObjectHolder* obh = Utility::getObjectHolder( PyTuple_GET_ITEM( aTuple, 0 ) );
      if ( ! obh->getObject() ) {
         PyErr_SetString( PyExc_TypeError, "iteration over non-sequence" );
         return 0;
      }

      TCollection* col =
         (TCollection*) obh->objectIsA()->DynamicCast( TCollection::Class(), obh->getObject() );

      ObjectHolder* b = new ObjectHolder( (void*) new TIter( col ), TIter::Class() );

      return bindRootObject( b );
   }


//- TSeqCollection behaviour ---------------------------------------------------
   PyObject* seqCollectionGetItem( PyObject* /* None */, PyObject* aTuple ) {
      if ( checkNArgs( aTuple, 2, "__getitem__" ) == false )
         return 0;

      PySliceObject* index = (PySliceObject*) PyTuple_GET_ITEM( aTuple, 1 );
      if ( PySlice_Check( index ) ) {
         ObjectHolder* obh = Utility::getObjectHolder( PyTuple_GET_ITEM( aTuple, 0 ) );
         if ( ! obh->getObject() ) {
            PyErr_SetString( PyExc_TypeError, "unsubscriptable object" );
            return 0;
         }

         TClass* clseq = obh->objectIsA();
         TSeqCollection* oseq =
            (TSeqCollection*) clseq->DynamicCast( TSeqCollection::Class(), obh->getObject() );
         TSeqCollection* nseq = (TSeqCollection*) clseq->New();

         int start, stop, step;
         PySlice_GetIndices( index, oseq->GetSize(), &start, &stop, &step );
         for ( int i = start; i < stop; i += step ) {
            nseq->Add( oseq->At( i ) );
         }

         return bindRootObject( new ObjectHolder( (void*) nseq, clseq ) );
      }

      return callSelfIndex( aTuple, "At" );
   }

   PyObject* seqCollectionSetItem( PyObject* /* None */, PyObject* aTuple ) {
      if ( checkNArgs( aTuple, 3, "__setitem__" ) == false )
         return 0;

      PyObject* self  = PyTuple_GET_ITEM( aTuple, 0 );
      PyObject* index = PyTuple_GET_ITEM( aTuple, 1 );
      PyObject* obj   = PyTuple_GET_ITEM( aTuple, 2 );

      if ( PySlice_Check( index ) ) {
         ObjectHolder* obh = Utility::getObjectHolder( self );
         if ( ! obh->getObject() ) {
            PyErr_SetString( PyExc_TypeError, "unsubscriptable object" );
            return 0;
         }

         TSeqCollection* oseq = (TSeqCollection*) obh->objectIsA()->DynamicCast(
            TSeqCollection::Class(), obh->getObject() );

         int start, stop, step;
         PySlice_GetIndices( (PySliceObject*) index, oseq->GetSize(), &start, &stop, &step );
         for ( int i = stop - step; i >= start; i -= step ) {
            oseq->RemoveAt( i );
         }

         for ( int i = 0; i < PySequence_Size( obj ); ++i ) {
            PyObject* item = PySequence_GetItem( obj, i );
            ObjectHolder* seqobh = Utility::getObjectHolder( item );
            seqobh->release();
            oseq->AddAt( (TObject*) seqobh->getObject(), i + start );
            Py_DECREF( item );
         }

         Py_INCREF( Py_None );
         return Py_None;
      }

      PyObject* pyindex = pyStyleIndex( self, index );
      if ( ! pyindex )
         return 0;

      PyObject* result  = PyObject_CallMethod( self, "RemoveAt", "O", pyindex );
      if ( ! result ) {
         Py_DECREF( pyindex );
         return 0;
      }

      Py_DECREF( result );
      result = PyObject_CallMethod( self, "AddAt", "OO", obj, pyindex );
      Py_DECREF( pyindex );
      return result;
   }

   PyObject* seqCollectionDelItem( PyObject* /* None */, PyObject* aTuple ) {
      if ( checkNArgs( aTuple, 2, "__del__" ) == false )
         return 0;

      PySliceObject* index = (PySliceObject*) PyTuple_GET_ITEM( aTuple, 1 );
      if ( PySlice_Check( index ) ) {
         ObjectHolder* obh = Utility::getObjectHolder( PyTuple_GET_ITEM( aTuple, 0 ) );
         if ( ! obh->getObject() ) {
            PyErr_SetString( PyExc_TypeError, "unsubscriptable object" );
            return 0;
         }

         TSeqCollection* oseq = (TSeqCollection*) obh->objectIsA()->DynamicCast(
            TSeqCollection::Class(), obh->getObject() );

         int start, stop, step;
         PySlice_GetIndices( index, oseq->GetSize(), &start, &stop, &step );
         for ( int i = stop - step; i >= start; i -= step ) {
            oseq->RemoveAt( i );
         }

         Py_INCREF( Py_None );
         return Py_None;
      }

      PyObject* result = callSelfIndex( aTuple, "RemoveAt" );
      if ( ! result )
         return 0;

      Py_DECREF( result );
      Py_INCREF( Py_None );
      return Py_None;
   }

   PyObject* seqCollectionInsert( PyObject* /* None */, PyObject* aTuple ) {
      if ( checkNArgs( aTuple, 3, "insert" ) == false )
         return 0;

      PyObject* self  = PyTuple_GET_ITEM( aTuple, 0 );
      PyObject* index = PyTuple_GET_ITEM( aTuple, 1 );
      PyObject* obj   = PyTuple_GET_ITEM( aTuple, 2 );

      int idx = (int) PyLong_AsLong( index );
      if ( PyErr_Occurred() )
         return 0;

      int size = PySequence_Size( self );
      if ( idx < 0 )
         idx = 0;
      else if ( size < idx )
         idx = size;

      return PyObject_CallMethod( self, "AddAt", "Oi", obj, idx );
   }

   PyObject* seqCollectionPop( PyObject* /* None */, PyObject* aTuple ) {
      if ( PyTuple_GET_SIZE( aTuple ) == 1 ) {
         PyObject* self = PyTuple_GET_ITEM( aTuple, 0 );
         Py_INCREF( self );

         aTuple = PyTuple_New( 2 );
         PyTuple_SET_ITEM( aTuple, 0, self );
         PyTuple_SET_ITEM( aTuple, 1, PyLong_FromLong( PySequence_Size( self ) - 1 ) );
      }

      return callSelfIndex( aTuple, "RemoveAt" );
   }

   PyObject* seqCollectionReverse( PyObject* /* None */, PyObject* aTuple ) {
      if ( checkNArgs( aTuple, 1, "reverse" ) == false )
         return 0;

      PyObject* self = PyTuple_GET_ITEM( aTuple, 0 );
      PyObject* tup = PySequence_Tuple( self );
      if ( ! tup )
         return 0;

      Py_DECREF( PyObject_CallMethod( self, "Clear", "" ) );

      for ( int i = 0; i < PySequence_Size( tup ); ++i ) {
         Py_DECREF( PyObject_CallMethod( self, "AddAt", "Oi", PyTuple_GET_ITEM( tup, i ), 0 ) );
      }

      Py_INCREF( Py_None );
      return Py_None;
   }

   PyObject* seqCollectionSort( PyObject* /* None */, PyObject* aTuple ) {
      PyObject* self = PyTuple_GET_ITEM( aTuple, 0 );

      if ( PyTuple_GET_SIZE( aTuple ) == 1 ) {
      // no specialized sort, use ROOT one
         return PyObject_CallMethod( self, "Sort", "" );
      }
      else {
      // sort in a python list copy
         PyObject* l = PySequence_List( self );
         Py_DECREF( PyObject_CallMethod( l, "sort", "O", PyTuple_GET_ITEM( aTuple, 1 ) ) );
         if ( PyErr_Occurred() ) {
            Py_DECREF( l );
            return 0;
         }

         Py_DECREF( PyObject_CallMethod( self, "Clear", "" ) );
         Py_DECREF( PyObject_CallMethod( self, "extend", "O", l ) );
         Py_DECREF( l );

         Py_INCREF( Py_None );
         return Py_None;
      }
   }

   PyObject* seqCollectionIndex( PyObject* /* None */, PyObject* aTuple ) {
      PyObject* index = callSelfTObject( aTuple, "IndexOf", "index" );
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
   PyObject* stringRepr( PyObject* /* None */, PyObject* aTuple ) {
      PyObject* data = PyObject_CallMethod( PyTuple_GET_ITEM( aTuple, 0 ), "Data", "" );
      PyObject* repr = PyString_FromFormat( "\'%s\'", PyString_AsString( data ) );
      Py_DECREF( data );
      return repr;
   }

   PyObject* stringCompare( PyObject* /* None */, PyObject* aTuple ) {
      return callSelfPyObject( aTuple, "CompareTo", "__cmp__" );
   }

   PyObject* stringLength( PyObject* /* None */, PyObject* aTuple ) {
      return callSelf( aTuple, "Length", "__len__" );
   }


//- TObjString behaviour -------------------------------------------------------
   PyObject* objStringRepr( PyObject* /* None */, PyObject* aTuple ) {
      PyObject* data = PyObject_CallMethod( PyTuple_GET_ITEM( aTuple, 0 ), "GetName", "" );
      PyObject* repr = PyString_FromFormat( "\'%s\'", PyString_AsString( data ) );
      Py_DECREF( data );
      return repr;
   }

   PyObject* objStringCompare( PyObject* /* None */, PyObject* aTuple ) {
      PyObject* data = PyObject_CallMethod( PyTuple_GET_ITEM( aTuple, 0 ), "GetName", "" );
      int result = PyObject_Compare( data, PyTuple_GET_ITEM( aTuple, 1 ) );
      Py_DECREF( data );

      if ( PyErr_Occurred() )
         return 0;

      return PyInt_FromLong( result );
   }

   PyObject* objStringLength( PyObject* /* None */, PyObject* aTuple ) {
      PyObject* data = PyObject_CallMethod( PyTuple_GET_ITEM( aTuple, 0 ), "GetName", "" );
      int size = PySequence_Size( data );
      Py_DECREF( data );
      return PyInt_FromLong( size );
   }


//- TIter behaviour ------------------------------------------------------------
   PyObject* iterIter( PyObject* /* None */, PyObject* aTuple ) {
      PyObject* iter = PyTuple_GET_ITEM( aTuple, 0 );
      Py_INCREF( iter );
      return iter;
   }

   PyObject* iterNext( PyObject* /* None */, PyObject* aTuple ) {
      PyObject* next = callSelf( aTuple, "Next", "next" );

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
   PyObject* treeGetAttr( PyObject* /* None */, PyObject* aTuple ) {
      if ( checkNArgs( aTuple, 2, "__getattr__" ) == false )
         return 0;

   // allow access to leaves as if they are data members
      PyObject* self = PyTuple_GET_ITEM( aTuple, 0 );
      PyObject* name = PyTuple_GET_ITEM( aTuple, 1 );

      PyObject* leaf = PyObject_CallMethod( self, "GetLeaf", "O", name );
      if ( ! leaf )
         return 0;

      if ( leaf != Py_None ) {
         PyObject* value = 0;

      // found a leaf, extract value if just one, or wrap buffer if more
         PyObject* lcount = PyObject_CallMethod( leaf, "GetLeafCount", "" );

         if ( lcount == Py_None ) {
            value = PyObject_CallMethod( leaf, "GetValue", "" );
         }
         else {
            PyObject* len = PyObject_CallMethod( leaf, "GetNdata", "" );
            int nlen = PyInt_AsLong( len );
            Py_DECREF( len );

            PyObject* ptr = PyObject_CallMethod( leaf, "GetValuePointer", "" );
            void* arr = PyLong_AsVoidPtr( ptr );
            Py_DECREF( ptr );

            PyObject* tname = PyObject_CallMethod( leaf, "GetTypeName", "" );
            std::string stname( PyString_AS_STRING( tname ) );

            Utility::EDataType eType = Utility::effectiveType( stname );
            if ( eType == Utility::kLong )
               value = PyBufferFactory::getInstance()->PyBuffer_FromMemory( (long*) arr, nlen );
            if ( eType == Utility::kInt )
               value = PyBufferFactory::getInstance()->PyBuffer_FromMemory( (int*) arr, nlen );
            if ( eType == Utility::kDouble )
               value = PyBufferFactory::getInstance()->PyBuffer_FromMemory( (double*) arr, nlen );
            if ( eType == Utility::kFloat )
               value = PyBufferFactory::getInstance()->PyBuffer_FromMemory( (float*) arr, nlen );
         }

         Py_DECREF( lcount );
         Py_DECREF( leaf );

         PyObject_SetAttr( self, name, value );
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
   std::map< int, std::pair< PyObject*, int > > s_PyObjectCallbacks;

   int pyFuncCallback( G__value* res, G__CONST char*, struct G__param* libp, int hash ) {
   // retrieve function information
      std::pair< PyObject*, int > info = s_PyObjectCallbacks[ res->tagnum ];
      PyObject* pyfunc = info.first;

   // prepare arguments
      PyObject* arg1 = PyBufferFactory::getInstance()->PyBuffer_FromMemory(
         (double*)G__int(libp->para[0]), 4 );

      PyObject* arg2 = PyBufferFactory::getInstance()->PyBuffer_FromMemory(
         (double*)G__int(libp->para[1]), info.second );

   // actual call
      PyObject* result = PyObject_CallFunction( pyfunc, "OO", arg1, arg2 );

   // destroy argument buffer wrappers
      Py_DECREF( arg1 );
      Py_DECREF( arg2 );

   // translate result, throw if an error has occurred
      double d = 0.;
      if ( ! result ) {
         PyErr_Print();
         throw std::runtime_error( "TF1 python function call failed" );
      }
      else {
         d = PyFloat_AsDouble( result );
         Py_DECREF( result );
      }

      G__letdouble( res, 100, d );
      return ( 1 || hash || res || libp );
   }

   class TF1InitWithPyFunc : public PyCallable {
      static int s_count;

   public:
      virtual PyObject* operator()( PyObject* aTuple, PyObject* /* aDict */ ) {
      // expected signature: ( char*, pyfunc, double, double, int )
         int nArgs = PyTuple_GET_SIZE( aTuple );
         if ( nArgs != 6 )
            return 0;              // reported as an overload failure

         PyObject* fcn = PyTuple_GetItem( aTuple, 2 );
         if ( ! fcn || ! PyCallable_Check( fcn ) ) {
            PyErr_SetString( PyExc_ValueError, "not a valid function" );
            return 0;
         }

      // construct new identifier
         char fid[ 32 ];
         sprintf( fid, "pyroot_func_%d", s_count++ );

      // build CINT functionclass placeholder
         G__lastifuncposition();
         G__memfunc_setup(
            fid, 444, (G__InterfaceMethod)NULL,
            100, -1, -1, 0, 2, 1, 1, 0, "D - - 0 - - D - - 0 - -",
            (char*)NULL, (void*)((int)this + s_count), 0 );
         G__resetifuncposition();

         G__ClassInfo gcl;
         gcl.AddMethod( "D", fid, "double*, double*" );

         long offset = 0;
         G__MethodInfo m = gcl.GetMethod( fid, "double*, double*", &offset );

         G__ifunc_table* ifunc = m.ifunc();
         int index = m.Index();

         ifunc->pentry[index]->size        = -1;
         ifunc->pentry[index]->filenum     = -1;
         ifunc->pentry[index]->line_number = -1;
         ifunc->pentry[index]->tp2f = (void*) pyFuncCallback;
         ifunc->pentry[index]->p    = (void*) pyFuncCallback;

      // setup association
         int tag = -1 - s_count;
         ifunc->p_tagtable[index] = tag;
         Py_INCREF( fcn );
         s_PyObjectCallbacks[ tag ] =
            std::make_pair( fcn, (int) PyInt_AsLong( PyTuple_GET_ITEM( aTuple, 5 ) ) );

      // get constructor and re-run
         PyObject* pymeth =
            PyObject_GetAttrString( PyTuple_GET_ITEM( aTuple, 0 ), "__init__" );

         PyObject* args = PyTuple_New( nArgs - 1 );
         ObjectHolder* dummy =
            new ObjectHolder( (void*)((int)this + s_count), TObject::Class(), false );

         for ( int iarg = 1; iarg < nArgs; ++iarg ) {
            PyObject* item = PyTuple_GET_ITEM( aTuple, iarg );
            if ( iarg != 2 ) {
               Py_INCREF( item );
               PyTuple_SET_ITEM( args, iarg - 1, item );
            }
            else {
               PyTuple_SET_ITEM( args, iarg - 1, bindRootObject( dummy ) );
            }
         }

         PyObject* result = PyObject_CallObject( pymeth, args );

         Py_DECREF( args );
         Py_DECREF( pymeth );
         return result;
      }
   };

   int TF1InitWithPyFunc::s_count = 0;
}


//- public functions -----------------------------------------------------------
bool PyROOT::pythonize( PyObject* pyclass, const std::string& name ) {
   if ( pyclass == 0 )
      return false;

   if ( name == "TObject" ) {
   // ROOT pointer validity testing
      Utility::addToClass( "__zero__",    isZero,    pyclass );
      Utility::addToClass( "__nonzero__", isNotZero, pyclass );

   // support for the 'in' operator
      Utility::addToClass( "__contains__", contains, pyclass );

      return true;
   }

   if ( name == "TCollection" ) {
      Utility::addToClass( "append",   collectionAppend, pyclass );
      Utility::addToClass( "extend",   collectionExtend, pyclass );
      Utility::addToClass( "remove",   collectionRemove, pyclass );
      Utility::addToClass( "__add__",  collectionAdd,    pyclass );
      Utility::addToClass( "__imul__", collectionIMul,   pyclass );
      Utility::addToClass( "__mul__",  collectionMul,    pyclass );
      Utility::addToClass( "__rmul__", collectionMul,    pyclass );

      Utility::addToClass( "count", collectionCount, pyclass );

      Utility::addToClass( "__len__",  collectionLength, pyclass );
      Utility::addToClass( "__iter__", collectionIter,   pyclass );

      return true;
   }

   if ( name == "TSeqCollection" ) {
      Utility::addToClass( "__getitem__", seqCollectionGetItem, pyclass );
      Utility::addToClass( "__setitem__", seqCollectionSetItem, pyclass );
      Utility::addToClass( "__delitem__", seqCollectionDelItem, pyclass );

      Utility::addToClass( "insert",  seqCollectionInsert,  pyclass );
      Utility::addToClass( "pop",     seqCollectionPop,     pyclass );
      Utility::addToClass( "reverse", seqCollectionReverse, pyclass );
      Utility::addToClass( "sort",    seqCollectionSort,    pyclass );

      Utility::addToClass( "index", seqCollectionIndex, pyclass );

      return true;
   }

   if ( name == "TString" ) {
   // ROOT pointer validity testing
      Utility::addToClass( "__zero__",    isZero,    pyclass );
      Utility::addToClass( "__nonzero__", isNotZero, pyclass );

      Utility::addToClass( "__repr__", stringRepr,   pyclass );
      Utility::addToClass( "__len__",  stringLength, pyclass );

      Utility::addToClass( "__cmp__", stringCompare, pyclass );

      return true;
   }

   if ( name == "TObjString" ) {
      Utility::addToClass( "__repr__", objStringRepr,   pyclass );
      Utility::addToClass( "__len__",  objStringLength, pyclass );

      Utility::addToClass( "__cmp__", objStringCompare, pyclass );

      return true;
   }

   if ( name == "TIter" ) {
   // ROOT pointer validity testing
      Utility::addToClass( "__zero__",    isZero,    pyclass );
      Utility::addToClass( "__nonzero__", isNotZero, pyclass );

      Utility::addToClass( "__iter__", iterIter, pyclass );
      Utility::addToClass( "next",     iterNext, pyclass );

      return true;
   }

   if ( name == "TTree" ) {
   // allow direct browsing of the tree
      Utility::addToClass( "__getattr__", treeGetAttr, pyclass );
   }

   if ( name == "TF1" ) {
   // allow instantiation with python function
      PyObject* pymeth = PyObject_GetAttrString( pyclass, "__init__" );
      MethodDispatcher* pmd = reinterpret_cast< MethodDispatcher* >(
         PyCObject_AsVoidPtr( PyCFunction_GetSelf( PyMethod_Function( pymeth ) ) ) );

      pmd->addMethod( new TF1InitWithPyFunc() );

      Py_DECREF( pymeth );
      return true;
   }

   return true;
}
