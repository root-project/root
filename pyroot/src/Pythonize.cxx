// Author: Wim Lavrijsen, Jul 2004

// Bindings
#include "PyROOT.h"
#include "Pythonize.h"
#include "ObjectHolder.h"
#include "RootWrapper.h"
#include "Utility.h"

// ROOT
#include "TClass.h"
#include "TCollection.h"
#include "TSeqCollection.h"

// Standard
#include <string>
#include <iostream>
#include <stdio.h>


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

      return PyInt_FromLong(
         PyObject_IsTrue( PyObject_CallMethod( self, "FindObject", "O", obj ) ) );
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
         PyObject_CallMethod( self, "Add", "O", item );
         Py_DECREF( item );
      }

      Py_INCREF( Py_None );
      return Py_None;
   }

   PyObject* collectionRemove( PyObject* /* None */, PyObject* aTuple ) {
      PyObject* result = callSelfTObject( aTuple, "Remove", "remove" );

      if ( ! result )
         return result;

      if ( ! PyObject_IsTrue( result ) ) {
         PyErr_SetString( PyExc_ValueError, "list.remove(x): x not in list" );
         return 0;
      }

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
      if ( ! result )
         return 0;

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
         new ObjectHolder( obh->objectIsA()->New(), obh->objectIsA(), true ) );

      for ( int i = 0; i < (int) PyLong_AsLong( imul ); ++i ) {
         PyObject_CallMethod( nseq, "extend", "O", self );
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
         if ( PyObject_IsTrue( PyObject_CallMethod( item, "IsEqual", "O", obj ) ) )
            count += 1;
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

      ObjectHolder* b = new ObjectHolder( (void*) new TIter( col ), TIter::Class(), true );

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

         return bindRootObject( new ObjectHolder( nseq, clseq, true ) );
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

      PyObject_CallMethod( self, "Clear", "" );

      for ( int i = 0; i < PySequence_Size( tup ); ++i ) {
         PyObject_CallMethod( self, "AddAt", "Oi", PyTuple_GET_ITEM( tup, i ), 0 );
      }

      Py_INCREF( Py_None );
      return Py_None;
   }

   PyObject* seqCollectionSort( PyObject* /* None */, PyObject* aTuple ) {
      PyObject* self = PyTuple_GET_ITEM( aTuple, 0 );

      if ( PyTuple_GET_SIZE( aTuple ) == 1 ) {
      // no specialized sort, use ROOT one
         return PyObject_CallMethod( PyTuple_GET_ITEM( aTuple, 0 ), "Sort", "" );
      }
      else {
      // sort in a python list copy
         PyObject* l = PySequence_List( self );
         PyObject_CallMethod( l, "sort", "O", PyTuple_GET_ITEM( aTuple, 1 ) );
         if ( PyErr_Occurred() ) {
            Py_DECREF( l );
            return 0;
         }

         PyObject_CallMethod( self, "Clear", "" );
         PyObject_CallMethod( self, "extend", "O", l );
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
      return PyTuple_GET_ITEM( aTuple, 0 );
   }

   PyObject* iterNext( PyObject* /* None */, PyObject* aTuple ) {
      PyObject* next = callSelf( aTuple, "Next", "next" );

      if ( ! next )
         return next;

      if ( ! PyObject_IsTrue( next ) ) {
         PyErr_SetString( PyExc_StopIteration, "" );
         return 0;
      }

      return next;
   }
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

   return true;
}
