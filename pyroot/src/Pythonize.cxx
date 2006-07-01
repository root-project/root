// @(#)root/pyroot:$Name:  $:$Id: Pythonize.cxx,v 1.39 2006/06/13 06:39:05 brun Exp $
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
#include "Converters.h"
#include "MemoryRegulator.h"

// ROOT
#include "TClass.h"
#include "TCollection.h"
#include "TDirectory.h"
#include "TSeqCollection.h"
#include "TClonesArray.h"
#include "TObject.h"
#include "TFunction.h"

#include "TTree.h"
#include "TBranch.h"
#include "TLeaf.h"

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

//____________________________________________________________________________
   inline Bool_t IsTemplatedSTLClass( const std::string& name, const std::string& klass ) {
      const int nsize = (int)name.size();
      const int ksize = (int)klass.size();

      return ( ( ksize   < nsize && name.substr(0,ksize) == klass ) ||
               ( ksize+5 < nsize && name.substr(5,ksize) == klass ) ) &&
             name.find( "::", name.find( ">" ) ) == std::string::npos;
   }

// to prevent compiler warnings about const char* -> char*
   inline PyObject* CallPyObjMethod( PyObject* obj, const char* meth )
   {
      return PyObject_CallMethod( obj, const_cast< char* >( meth ), const_cast< char* >( "" ) );
   }

//____________________________________________________________________________
   inline PyObject* CallPyObjMethod( PyObject* obj, const char* meth, PyObject* arg1 )
   {
      return PyObject_CallMethod(
         obj, const_cast< char* >( meth ), const_cast< char* >( "O" ), arg1 );
   }

//____________________________________________________________________________
   inline PyObject* CallPyObjMethod(
      PyObject* obj, const char* meth, PyObject* arg1, PyObject* arg2 )
   {
      return PyObject_CallMethod(
         obj, const_cast< char* >( meth ), const_cast< char* >( "OO" ), arg1, arg2 );
   }

//____________________________________________________________________________
   inline PyObject* CallPyObjMethod( PyObject* obj, const char* meth, PyObject* arg1, int arg2 )
   {
      return PyObject_CallMethod(
         obj, const_cast< char* >( meth ), const_cast< char* >( "Oi" ), arg1, arg2 );
   }


//- helpers --------------------------------------------------------------------
   PyObject* CallPySelfMethod( PyObject* args, const char* meth, const char* fmt )
   {
      PyObject* self = 0;
      if ( ! PyArg_ParseTuple( args, const_cast< char* >( fmt ), &self ) )
         return 0;

      return CallPyObjMethod( self, meth );
   }

//____________________________________________________________________________
   PyObject* CallPySelfObjMethod( PyObject* args, const char* meth, const char* fmt )
   {
      PyObject* self = 0, *obj = 0;
      if ( ! PyArg_ParseTuple( args, const_cast< char* >( fmt ), &self, &obj ) )
         return 0;

      return CallPyObjMethod( self, meth, obj );
   }

//____________________________________________________________________________
   PyObject* PyStyleIndex( PyObject* self, PyObject* index )
   {
      Long_t idx = PyInt_AsLong( index );
      if ( PyErr_Occurred() )
         return 0;

      PyObject* pyindex = 0;
      Long_t size = PySequence_Size( self );
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

      PyObject* pyindex = PyStyleIndex( self, obj );
      if ( ! pyindex )
         return 0;

      PyObject* result = CallPyObjMethod( self, meth, pyindex );
      Py_DECREF( pyindex );
      return result;
   }

//- "smart pointer" behaviour --------------------------------------------------
   PyObject* DeRefGetAttr( PyObject*, PyObject* args )
   {
      PyObject* self = 0; PyObject* name = 0;
      if ( ! PyArg_ParseTuple( args, const_cast< char* >( "OS:__deref__" ), &self, &name ) )
         return 0;

      PyObject* pyptr = CallPyObjMethod( self, "__deref__" );
      if ( ! pyptr )
         return 0;

      return PyObject_GetAttr( pyptr, name );
   }

//____________________________________________________________________________
   PyObject* FollowGetAttr( PyObject*, PyObject* args )
   {
      PyObject* self = 0; PyObject* name = 0;
      if ( ! PyArg_ParseTuple( args, const_cast< char* >( "OS:__follow__" ), &self, &name ) )
         return 0;

      PyObject* pyptr = CallPyObjMethod( self, "__follow__" );
      if ( ! pyptr )
         return 0;

      return PyObject_GetAttr( pyptr, name );
   }

//- TObject behaviour ----------------------------------------------------------
   PyObject* TObjectContains( PyObject*, PyObject* args )
   {
      PyObject* self = 0, *obj = 0;
      if ( ! PyArg_ParseTuple( args, const_cast< char* >( "OO:__contains__" ), &self, &obj ) )
         return 0;

      if ( ! ( ObjectProxy_Check( obj ) || PyString_Check( obj ) ) )
         return PyInt_FromLong( 0l );

      PyObject* found = CallPyObjMethod( self, "FindObject", obj );
      PyObject* result = PyInt_FromLong( PyObject_IsTrue( found ) );
      Py_DECREF( found );
      return result;
   }

//____________________________________________________________________________
   PyObject* TObjectCompare( PyObject*, PyObject* args )
   {
      PyObject* self = 0, *obj = 0;
      if ( ! PyArg_ParseTuple( args, const_cast< char* >( "OO:__cmp__" ), &self, &obj ) )
         return 0;

      if ( ! ObjectProxy_Check( obj ) )
         return PyInt_FromLong( -1l );

      return CallPyObjMethod( self, "Compare", obj );
   }

//____________________________________________________________________________
   PyObject* TObjectIsEqual( PyObject*, PyObject* args )
   {
      PyObject* self = 0, *obj = 0;
      if ( ! PyArg_ParseTuple( args, const_cast< char* >( "OO:__eq__" ), &self, &obj ) )
         return 0;

      if ( ! ObjectProxy_Check( obj ) )
         return PyInt_FromLong( 0l );

      return CallPyObjMethod( self, "IsEqual", obj );
   }


//- TClass behaviour -----------------------------------------------------------
   PyObject* TClassStaticCast( PyObject*, PyObject* args )
   {
   // Implemented somewhat different than TClass::DynamicClass, in that "up" is
   // chosen automatically based on the relationship between self and arg pyclass.
      ObjectProxy* self = 0, *pyclass = 0;
      PyObject* pyobject = 0;
      if ( ! PyArg_ParseTuple( args, const_cast< char* >( "O!O!O:StaticCast" ),
              &ObjectProxy_Type, &self, &ObjectProxy_Type, &pyclass, &pyobject ) )
         return 0;

   // check the given arguments (dcasts are necessary b/c of could be a TQClass
      TClass* from =
         (TClass*)self->ObjectIsA()->DynamicCast( TClass::Class(), self->GetObject() );
      TClass* to   =
         (TClass*)pyclass->ObjectIsA()->DynamicCast( TClass::Class(), pyclass->GetObject() );

      if ( ! from ) {
         PyErr_SetString( PyExc_TypeError, "unbound method TClass::StaticCast "
            "must be called with a TClass instance as first argument" );
         return 0;
      }

      if ( ! to ) {
         PyErr_SetString( PyExc_TypeError, "could not convert argument 1 (TClass* expected)" );
         return 0;
      }

   // retrieve object address
      void* address = 0;
      if ( ObjectProxy_Check( pyobject ) ) address = ((ObjectProxy*)pyobject)->GetObject();
      else if ( PyInt_Check( pyobject ) ) address = (void*)PyInt_AS_LONG( pyobject );
      else Utility::GetBuffer( pyobject, '*', 1, address, kFALSE );

      if ( ! address ) {
         PyErr_SetString( PyExc_TypeError, "could not convert argument 2 (void* expected)" );
         return 0;
      }

   // determine direction of cast
      int up = -1;
      if ( from->InheritsFrom( to ) ) up = 1;
      else if ( to->InheritsFrom( from ) ) {
         TClass* tmp = to; to = from; from = tmp;
         up = 0;
      }

      if ( up == -1 ) {
         PyErr_Format( PyExc_TypeError, "unable to cast %s to %s", from->GetName(), to->GetName() );
         return 0;
      }

   // perform actual cast
      void* result = from->DynamicCast( to, address, (Bool_t)up );

   // at this point, "result" can't be null (but is still safe if it is)
      return BindRootObjectNoCast( result, to );
   }

//____________________________________________________________________________
   PyObject* TClassDynamicCast( PyObject*, PyObject* args )
   {
      PyObject* self = 0, *pyclass = 0, *pyobject = 0;
      long up = 1;
      if ( ! PyArg_ParseTuple( args, const_cast< char* >( "O!O!O|l:DynamicCast" ),
              &ObjectProxy_Type, &self, &ObjectProxy_Type, &pyclass, &pyobject, &up ) )
         return 0;

   // perform actual cast
      PyObject* meth = PyObject_GetAttrString( self, (char*)"_TClass__DynamicCast" );
      PyObject* ptr = PyObject_Call(
         meth, PyTuple_GetSlice( args, 1, PyTuple_GET_SIZE( args ) ), 0 );

   // simply forward in case of call failure
      if ( ! ptr )
         return ptr;

   // supposed to be an int or long ...
      long address = PyLong_AsLong( ptr );
      if ( PyErr_Occurred() ) {
         PyErr_Clear();
         return ptr;
      }

   // now use binding to return a usable class
      TClass* klass = 0;
      if ( up ) {                  // up-cast: result is a base
         klass = (TClass*)((ObjectProxy*)pyclass)->ObjectIsA()->DynamicCast(
            TClass::Class(), ((ObjectProxy*)pyclass)->GetObject() );
      } else {                     // down-cast: result is a derived
         klass = (TClass*)((ObjectProxy*)self)->ObjectIsA()->DynamicCast(
            TClass::Class(), ((ObjectProxy*)self)->GetObject() );
      }

      PyObject* result = BindRootObjectNoCast( (void*)address, klass );
      Py_DECREF( ptr );
      return result;
   }

//- TCollection behaviour ------------------------------------------------------
   PyObject* TCollectionExtend( PyObject*, PyObject* args )
   {
      PyObject* self = 0, *obj = 0;
      if ( ! PyArg_ParseTuple( args, const_cast< char* >( "OO:extend" ), &self, &obj ) )
         return 0;

      for ( int i = 0; i < PySequence_Size( obj ); ++i ) {
         PyObject* item = PySequence_GetItem( obj, i );
         PyObject* result = CallPyObjMethod( self, "Add", item );
         Py_XDECREF( result );
         Py_DECREF( item );
      }

      Py_INCREF( Py_None );
      return Py_None;
   }

//____________________________________________________________________________
   PyObject* TCollectionRemove( PyObject*, PyObject* args )
   {
      PyObject* result = CallPySelfObjMethod( args, "Remove", "OO:remove" );
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
   PyObject* TCollectionAdd( PyObject*, PyObject* args )
   {
      PyObject* self = 0, *other = 0;
      if ( ! PyArg_ParseTuple( args, const_cast< char* >( "OO:__add__" ), &self, &other ) )
         return 0;

      PyObject* l = CallPyObjMethod( self, "Clone" );
      if ( ! l )
         return 0;

      PyObject* result = CallPyObjMethod( l, "extend", other );
      if ( ! result ) {
         Py_DECREF( l );
         return 0;
      }

      return l;
   }

//____________________________________________________________________________
   PyObject* TCollectionMul( PyObject*, PyObject* args )
   {
      ObjectProxy* self = 0; Long_t imul = 0;
      if ( ! PyArg_ParseTuple( args, const_cast< char* >( "Ol:__mul__" ), &self, &imul ) )
         return 0;

      if ( ! ObjectProxy_Check( self ) || ! self->GetObject() ) {
         PyErr_SetString( PyExc_TypeError, "unsubscriptable object" );
         return 0;
      }

      PyObject* nseq = BindRootObject( self->ObjectIsA()->New(), self->ObjectIsA() );

      for ( Long_t i = 0; i < imul; ++i ) {
         PyObject* result = CallPyObjMethod( nseq, "extend", (PyObject*)self );
         Py_DECREF( result );
      }

      return nseq;
   }

//____________________________________________________________________________
   PyObject* TCollectionIMul( PyObject*, PyObject* args )
   {
      PyObject* self = 0; Long_t imul = 0;
      if ( ! PyArg_ParseTuple( args, const_cast< char* >( "Ol:__imul__" ), &self, &imul ) )
         return 0;

      PyObject* l = PySequence_List( self );

      for ( Long_t i = 0; i < imul - 1; ++i ) {
         CallPyObjMethod( self, "extend", l );
      }

      Py_INCREF( self );
      return self;
   }

//____________________________________________________________________________
   PyObject* TCollectionCount( PyObject*, PyObject* args )
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
   PyObject* TCollectionIter( PyObject*, PyObject* args ) {
      ObjectProxy* self = (ObjectProxy*) PyTuple_GET_ITEM( args, 0 );
      if ( ! ObjectProxy_Check( self ) || ! self->GetObject() ) {
         PyErr_SetString( PyExc_TypeError, "iteration over non-sequence" );
         return 0;
      }

      TCollection* col =
         (TCollection*)self->ObjectIsA()->DynamicCast( TCollection::Class(), self->GetObject() );

      PyObject* pyobject = BindRootObject( (void*) new TIter( col ), TIter::Class() );
      ((ObjectProxy*)pyobject)->fFlags |= ObjectProxy::kIsOwner;
      return pyobject;
   }


//- TSeqCollection behaviour ---------------------------------------------------
   PyObject* TSeqCollectionGetItem( PyObject*, PyObject* args )
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
            (TSeqCollection*)clSeq->DynamicCast( TSeqCollection::Class(), self->GetObject() );
         TSeqCollection* nseq = (TSeqCollection*)clSeq->New();

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
   PyObject* TSeqCollectionSetItem( PyObject*, PyObject* args )
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

         TSeqCollection* oseq = (TSeqCollection*)self->ObjectIsA()->DynamicCast(
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

      PyObject* pyindex = PyStyleIndex( (PyObject*)self, index );
      if ( ! pyindex )
         return 0;

      PyObject* result  = CallPyObjMethod( (PyObject*)self, "RemoveAt", pyindex );
      if ( ! result ) {
         Py_DECREF( pyindex );
         return 0;
      }

      Py_DECREF( result );
      result = CallPyObjMethod( (PyObject*)self, "AddAt", obj, pyindex );
      Py_DECREF( pyindex );
      return result;
   }

//____________________________________________________________________________
   PyObject* TSeqCollectionDelItem( PyObject*, PyObject* args )
   {
      ObjectProxy* self = 0; PySliceObject* index = 0;
      if ( ! PyArg_ParseTuple( args, const_cast< char* >( "OO:__del__" ), &self, &index ) )
         return 0;

      if ( PySlice_Check( index ) ) {
         if ( ! ObjectProxy_Check( self ) || ! self->GetObject() ) {
            PyErr_SetString( PyExc_TypeError, "unsubscriptable object" );
            return 0;
         }

         TSeqCollection* oseq = (TSeqCollection*)self->ObjectIsA()->DynamicCast(
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
   PyObject* TSeqCollectionInsert( PyObject*, PyObject* args )
   {
      PyObject* self = 0, *obj = 0; Long_t idx = 0;
      if ( ! PyArg_ParseTuple( args, const_cast< char* >( "OlO:insert" ), &self, &idx, &obj ) )
         return 0;

      int size = PySequence_Size( self );
      if ( idx < 0 )
         idx = 0;
      else if ( size < idx )
         idx = size;

      return CallPyObjMethod( self, "AddAt", obj, idx );
   }

//____________________________________________________________________________
   PyObject* TSeqCollectionPop( PyObject*, PyObject* args )
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
   PyObject* TSeqCollectionReverse( PyObject*, PyObject* args )
   {
      PyObject* self = 0;
      if ( ! PyArg_ParseTuple( args, const_cast< char* >( "O:reverse" ), &self ) )
         return 0;

      PyObject* tup = PySequence_Tuple( self );
      if ( ! tup )
         return 0;

      PyObject* result = CallPyObjMethod( self, "Clear" );
      Py_XDECREF( result );

      for ( int i = 0; i < PySequence_Size( tup ); ++i ) {
         PyObject* result = CallPyObjMethod( self, "AddAt", PyTuple_GET_ITEM( tup, i ), 0 );
         Py_XDECREF( result );
      }

      Py_INCREF( Py_None );
      return Py_None;
   }

//____________________________________________________________________________
   PyObject* TSeqCollectionSort( PyObject*, PyObject* args )
   {
      PyObject* self = PyTuple_GET_ITEM( args, 0 );

      if ( PyTuple_GET_SIZE( args ) == 1 ) {
      // no specialized sort, use ROOT one
         return CallPyObjMethod( self, "Sort" );
      } else {
      // sort in a python list copy
         PyObject* l = PySequence_List( self );
         PyObject* result = CallPyObjMethod( l, "sort", PyTuple_GET_ITEM( args, 1 ) );
         Py_XDECREF( result );
         if ( PyErr_Occurred() ) {
            Py_DECREF( l );
            return 0;
         }

         result = CallPyObjMethod( self, "Clear" );
         Py_XDECREF( result );
         result = CallPyObjMethod( self, "extend", l );
         Py_XDECREF( result );
         Py_DECREF( l );

         Py_INCREF( Py_None );
         return Py_None;
      }
   }

//____________________________________________________________________________
   PyObject* TSeqCollectionIndex( PyObject*, PyObject* args )
   {
      PyObject* index = CallPySelfObjMethod( args, "IndexOf", "OO:index" );
      if ( ! index )
         return 0;

      if ( PyLong_AsLong( index ) < 0 ) {
         Py_DECREF( index );
         PyErr_SetString( PyExc_ValueError, "list.index(x): x not in list" );
         return 0;
      }

      return index;
   }

//- TClonesArray behaviour -----------------------------------------------------
   PyObject* TClonesArraySetItem( PyObject*, PyObject* args )
   {
   // TClonesArray sets objects by constructing them in-place; which is impossible
   // to support as the python object given as value must exist a priori. It can,
   // however, by memcpy'd and stolen, caveat emptor.
      ObjectProxy* self = 0, *pyobj = 0; PyObject* idx = 0;
      if ( ! PyArg_ParseTuple( args, const_cast< char* >( "O!OO!:__setitem__" ),
                &ObjectProxy_Type, &self, &idx, &ObjectProxy_Type, &pyobj ) )
         return 0;

      PyObject* pyindex = PyStyleIndex( (PyObject*)self, idx );
      if ( ! pyindex )
         return 0;
      int index = (int)PyLong_AsLong( pyindex );
      Py_DECREF( pyindex );

   // get hold of the actual TClonesArray
      TClonesArray* cla =
         (TClonesArray*)self->ObjectIsA()->DynamicCast( TClonesArray::Class(), self->GetObject() );

      if ( ! cla ) {
         PyErr_SetString( PyExc_TypeError, "attempt to call with null object" );
         return 0;
      }

      if ( cla->GetClass() != pyobj->ObjectIsA() ) {
         PyErr_Format( PyExc_TypeError, "require object of type %s, but %s given",
            cla->GetClass()->GetName(), pyobj->ObjectIsA()->GetName() );
      }

   // destroy old stuff, if applicable
      if ( ((const TClonesArray&)*cla)[index] ) {
         cla->RemoveAt( index );
      }

      if ( pyobj->GetObject() ) {
      // accessing an entry will result get new, unitialized memory (if properly used)
         TObject* object = (*cla)[index];
         pyobj->Release();
         TMemoryRegulator::RegisterObject( pyobj, object );
         memcpy( (void*)object, pyobj->GetObject(), cla->GetClass()->Size() );
      }

      Py_INCREF( Py_None );
      return Py_None;
   }

//- vector behaviour as primitives --------------------------------------------
   PyObject* VectorGetItem( PyObject*, PyObject* args )
   {
      ObjectProxy* self = 0; PySliceObject* index = 0;
      if ( ! PyArg_ParseTuple( args, const_cast< char* >( "OO:__getitem__" ), &self, &index ) )
         return 0;

      if ( PySlice_Check( index ) ) {
         if ( ! ObjectProxy_Check( self ) || ! self->GetObject() ) {
            PyErr_SetString( PyExc_TypeError, "unsubscriptable object" );
            return 0;
         }

         PyObject* pyclass =
            PyObject_GetAttrString( (PyObject*)self, const_cast< char* >( "__class__" ) );
         PyObject* nseq = PyObject_CallObject( pyclass, NULL );
         Py_DECREF( pyclass );
 
         int start, stop, step;
         PySlice_GetIndices( index, PyObject_Length( (PyObject*)self ), &start, &stop, &step );
         for ( int i = start; i < stop; i += step ) {
            PyObject* pyidx = PyInt_FromLong( i );
            CallPyObjMethod( nseq, "push_back", CallPyObjMethod( (PyObject*)self, "_vector__at", pyidx ) );
            Py_DECREF( pyidx );
         }

         return nseq;
      }

      return callSelfIndex( args, "_vector__at" );
   }

//- STL container iterator supprt ---------------------------------------------
   PyObject* StlSequenceIter( PyObject*, PyObject* args )
   {
      PyObject* iter = CallPySelfMethod( args, "begin", "O" );
      if ( iter ) {
         PyObject* end = CallPySelfMethod( args, "end", "O" );
         if ( end )
            PyObject_SetAttrString( iter, const_cast< char* >( "end" ), end );
         Py_XDECREF( end );
      }
      return iter;
   }

//- string behaviour as primitives --------------------------------------------
#define PYROOT_IMPLEMENT_STRING_PYTHONIZATION( name, func )                   \
   PyObject* name##StringRepr( PyObject*, PyObject* args )                    \
   {                                                                          \
      PyObject* data = CallPyObjMethod( PyTuple_GET_ITEM( args, 0 ), #func ); \
      PyObject* repr = PyString_FromFormat( "\'%s\'", PyString_AsString( data ) ); \
      Py_DECREF( data );                                                      \
      return repr;                                                            \
   }                                                                          \
                                                                              \
   PyObject* name##StringCompare( PyObject*, PyObject* args )                 \
   {                                                                          \
      PyObject* data = CallPyObjMethod( PyTuple_GET_ITEM( args, 0 ), #func ); \
      int result = PyObject_Compare( data, PyTuple_GET_ITEM( args, 1 ) );     \
      Py_DECREF( data );                                                      \
      if ( PyErr_Occurred() )                                                 \
         return 0;                                                            \
      return PyInt_FromLong( result );                                        \
   }                                                                          \
                                                                              \
   PyObject* name##StringIsequal( PyObject*, PyObject* args )                 \
   {                                                                          \
      PyObject* data = CallPyObjMethod( PyTuple_GET_ITEM( args, 0 ), #func ); \
      PyObject* result = PyObject_RichCompare( data, PyTuple_GET_ITEM( args, 1 ), Py_EQ );\
      Py_DECREF( data );                                                      \
      if ( ! result )                                                         \
         return 0;                                                            \
      return result;                                                          \
   }

   PYROOT_IMPLEMENT_STRING_PYTHONIZATION( Stl, c_str )
   PYROOT_IMPLEMENT_STRING_PYTHONIZATION(   T, Data )


//- TObjString behaviour -------------------------------------------------------
   PYROOT_IMPLEMENT_STRING_PYTHONIZATION( TObj, GetName )

//____________________________________________________________________________
   PyObject* TObjStringLength( PyObject*, PyObject* args )
   {
      PyObject* data = CallPyObjMethod( PyTuple_GET_ITEM( args, 0 ), "GetName" );
      int size = PySequence_Size( data );
      Py_DECREF( data );
      return PyInt_FromLong( size );
   }


//- TIter behaviour ------------------------------------------------------------
   PyObject* TIterIter( PyObject*, PyObject* args )
   {
      PyObject* iter = PyTuple_GET_ITEM( args, 0 );
      Py_INCREF( iter );
      return iter;
   }

//____________________________________________________________________________
   PyObject* TIterNext( PyObject*, PyObject* args )
   {
      PyObject* next = CallPySelfMethod( args, "Next", "O:next" );

      if ( ! next )
         return 0;

      if ( ! PyObject_IsTrue( next ) ) {
         Py_DECREF( next );
         PyErr_SetString( PyExc_StopIteration, "" );
         return 0;
      }

      return next;
   }


//- STL iterator behaviour -----------------------------------------------------
   PyObject* StlIterCompare( PyObject*, PyObject* args )
   {
      ObjectProxy* self = 0, *pyobj = 0;
      if ( ! PyArg_ParseTuple( args, const_cast< char* >( "OO:__cmp__" ), &self, &pyobj ) )
         return 0;

      if ( ! ObjectProxy_Check( pyobj ) )
         return PyInt_FromLong( -1l );

      return PyInt_FromLong( *(void**)self->GetObject() == *(void**)pyobj->GetObject() ? 0l : 1l );
   }

//____________________________________________________________________________
   PyObject* StlIterNext( PyObject*, PyObject* args )
   {
      PyObject* self = 0;
      if ( ! PyArg_ParseTuple( args, const_cast< char* >( "O" ), &self ) )
         return 0;

      PyObject* next = 0;

      PyObject* last = PyObject_GetAttrString( self, const_cast< char* >( "end" ) );
      if ( last != 0 ) {
         PyObject* dummy = PyInt_FromLong( 1l );
         PyObject* iter = CallPyObjMethod( self, "__postinc__", dummy );
         Py_DECREF( dummy );
         if ( iter != 0 ) {
            if ( *(void**)((ObjectProxy*)last)->fObject == *(void**)((ObjectProxy*)iter)->fObject )
               PyErr_SetString( PyExc_StopIteration, "" );
            else
               next = CallPyObjMethod( iter, "__deref__" );
         }
         Py_XDECREF( iter );
      }
      Py_XDECREF( last );

      return next;
   }


//- TDirectory member templates ----------------------------------------------
   PyObject* TDirectoryGetObject( PyObject*, PyObject* args )
   {
      ObjectProxy* self = 0; PyObject* name = 0; ObjectProxy* ptr = 0;
      if ( ! PyArg_ParseTuple( args, const_cast< char* >( "OSO:TDirectory::GetObject" ),
               &self, &name, &ptr ) )
         return 0;

      TDirectory* dir =
         (TDirectory*)self->ObjectIsA()->DynamicCast( TDirectory::Class(), self->GetObject() );

      if ( ! dir ) {
         PyErr_SetString( PyExc_TypeError,
           "TDirectory::GetObject must be called with a TDirectory instance as first argument" );
         return 0;
      }

      void* address = dir->GetObjectChecked( PyString_AS_STRING( name ), ptr->ObjectIsA() );
      if ( address ) {
         ptr->Set( address, ptr->ObjectIsA() );

         Py_INCREF( Py_None );
         return Py_None;
      }

      PyErr_Format( PyExc_LookupError, "no such object, \"%s\"", PyString_AS_STRING( name ) );
      return 0;
   }

//____________________________________________________________________________
   PyObject* TDirectoryWriteObject( PyObject*, PyObject* args )
   {
      ObjectProxy *self = 0, *wrt = 0; PyObject *name = 0, *option = 0;
      if ( ! PyArg_ParseTuple( args, const_cast< char* >( "OOS|S:TDirectory::WriteObject" ),
               &self, &wrt, &name, &option ) )
         return 0;

      TDirectory* dir =
         (TDirectory*)self->ObjectIsA()->DynamicCast( TDirectory::Class(), self->GetObject() );

      if ( ! dir ) {
         PyErr_SetString( PyExc_TypeError,
           "TDirectory::WriteObject must be called with a TDirectory instance as first argument" );
         return 0;
      }

      Int_t result = 0;
      if ( option != 0 ) {
         result = dir->WriteObjectAny( wrt->GetObject(), wrt->ObjectIsA(),
            PyString_AS_STRING( name ), PyString_AS_STRING( option ) );
      } else {
         result = dir->WriteObjectAny(
            wrt->GetObject(), wrt->ObjectIsA(), PyString_AS_STRING( name ) );
      }

      return PyInt_FromLong( (Long_t)result );
   }

}


namespace PyROOT {      // workaround for Intel icc on Linux

//- TTree behaviour ----------------------------------------------------------
   PyObject* TTreeGetAttr( PyObject*, PyObject* args )
   {
   // allow access to branches/leaves as if they are data members
      ObjectProxy* self = 0; const char* name = 0;
      if ( ! PyArg_ParseTuple( args, const_cast< char* >( "O!s:__getattr__" ),
                &ObjectProxy_Type, &self, &name ) )
         return 0;

   // get hold of actual tree
      TTree* tree =
         (TTree*)self->ObjectIsA()->DynamicCast( TTree::Class(), self->GetObject() );

   // search for branch first (typical for objects)
      TBranch* branch = tree->GetBranch( name );
      if ( branch ) {
      // found a branched object, wrap its address for the object it represents
         TClass* klass = gROOT->GetClass( branch->GetClassName() );
         if ( klass && branch->GetAddress() )
            return BindRootObjectNoCast( *(char**)branch->GetAddress(), klass );
      }

   // if not, try leaf
      TLeaf* leaf = tree->GetLeaf( name );
      if ( leaf ) {
      // found a leaf, extract value and wrap
         if ( ! leaf->IsRange() ) {
         // array types
            std::string typeName = leaf->GetTypeName();
            TConverter* pcnv = CreateConverter( typeName + '*', leaf->GetNdata() );
            void* address = (void*)leaf->GetValuePointer();
            PyObject* value = pcnv->FromMemory( &address );
            delete pcnv;

            return value;
         } else {
         // value types
            TConverter* pcnv = CreateConverter( leaf->GetTypeName() );
            PyObject* value = pcnv->FromMemory( (void*)leaf->GetValuePointer() );
            delete pcnv;

            return value;
         }
      }

   // confused
      PyErr_Format( PyExc_AttributeError,
          "\'%s\' object has no attribute \'%s\'", tree->IsA()->GetName(), name );
      return 0;
   }

//____________________________________________________________________________
   class TTreeMemberFunction : public PyCallable {
   protected:
      TTreeMemberFunction( MethodProxy* org ) { Py_INCREF( org ); fOrg = org; }
      TTreeMemberFunction( const TTreeMemberFunction& t ) : PyCallable( t )
      {
         Py_INCREF( t.fOrg );
         fOrg = t.fOrg;
      }
      TTreeMemberFunction& operator=( const TTreeMemberFunction& t )
      {
         if ( &t != this ) {
            Py_INCREF( t.fOrg );
            fOrg = t.fOrg;
         }
         return *this;
      }
      ~TTreeMemberFunction() { Py_DECREF( fOrg ); fOrg = 0; }

   public:
      virtual PyObject* GetSignature() { return PyString_FromString( "(...)" ); }

   protected:
      MethodProxy* fOrg;
   };

//____________________________________________________________________________
   class TTreeBranch : public TTreeMemberFunction {
   public:
      TTreeBranch( MethodProxy* org ) : TTreeMemberFunction( org ) {}

   public:
      virtual PyObject* GetPrototype()
      {
         return PyString_FromString( "TBranch* TTree::Branch( ... )" );
      }

      virtual PyObject* operator()( ObjectProxy* self, PyObject* args, PyObject* kwds )
      {
      // acceptable signatures:
      //   ( const char*, void*, const char*, Int_t = 32000 )
      //   ( const char*, const char*, T**, Int_t = 32000, Int_t = 99 )
      //   ( const char*, T**, Int_t = 32000, Int_t = 99 ) 
         int argc = PyTuple_GET_SIZE( args );

         if ( 2 <= argc ) {
            TTree* tree =
               (TTree*)self->ObjectIsA()->DynamicCast( TTree::Class(), self->GetObject() );

            if ( ! tree ) {
               PyErr_SetString( PyExc_TypeError,
                  "TTree::Branch must be called with a TTree instance as first argument" );
               return 0;
            }

            PyObject *name = 0, *clName = 0, *leaflist = 0;
            PyObject *address = 0;
            PyObject *bufsize = 0, *splitlevel = 0;

         // try: ( const char*, void*, const char*, Int_t = 32000 )
            if ( PyArg_ParseTuple( args, const_cast< char* >( "SOS|O!:Branch" ),
                    &name, &address, &leaflist, &PyInt_Type, &bufsize ) ) {

               void* buf = 0;
               if ( ObjectProxy_Check( address ) )
                  buf = (void*)((ObjectProxy*)address)->GetObject();
               else
                  Utility::GetBuffer( address, '*', 1, buf, kFALSE );
 
               if ( buf != 0 ) {
                  TBranch* branch = 0;
                  if ( argc == 4 ) {
                     branch = tree->Branch( PyString_AS_STRING( name ), buf,
                        PyString_AS_STRING( leaflist ), PyInt_AS_LONG( bufsize ) );
                  } else {
                     branch = tree->Branch(
                        PyString_AS_STRING( name ), buf, PyString_AS_STRING( leaflist ) );
                  }

                  return BindRootObject( branch, TBranch::Class() );
               }

            }
            PyErr_Clear();

         // try: ( const char*, const char*, T**, Int_t = 32000, Int_t = 99 )
         //  or: ( const char*,              T**, Int_t = 32000, Int_t = 99 ) 
            Bool_t bIsMatch = kFALSE;
            if ( PyArg_ParseTuple( args, const_cast< char* >( "SSO|O!O!:Branch" ),
                    &name, &clName, &address, &PyInt_Type, &bufsize, &PyInt_Type, &splitlevel ) ) {
               bIsMatch = kTRUE;
            } else {
               PyErr_Clear(); clName = 0;    // clName no longer used
               if ( PyArg_ParseTuple( args, const_cast< char* >( "SO|O!O!" ),
                       &name, &address, &PyInt_Type, &bufsize, &PyInt_Type, &splitlevel ) )
                  bIsMatch = kTRUE;
               else
                  PyErr_Clear();
            }

            if ( bIsMatch == kTRUE ) {
               std::string klName = clName ? PyString_AS_STRING( clName ) : "";
               void* buf = 0;

               if ( ObjectProxy_Check( address ) ) {
                  if ( ((ObjectProxy*)address)->fFlags & ObjectProxy::kIsReference )
                     buf = (void*)((ObjectProxy*)address)->fObject;
                  else
                     buf = (void*)&((ObjectProxy*)address)->fObject;

                  if ( ! clName ) {
                     klName = ((ObjectProxy*)address)->ObjectIsA()->GetName();
                     argc += 1;
                  }
               } else
                  Utility::GetBuffer( address, '*', 1, buf, kFALSE );

               if ( buf != 0 && klName != "" ) {
                  TBranch* branch = 0;
                  if ( argc == 3 ) {
                     branch = tree->Branch( PyString_AS_STRING( name ), klName.c_str(), buf );
                  } else if ( argc == 4 ) {
                     branch = tree->Branch( PyString_AS_STRING( name ), klName.c_str(), buf,
                        PyInt_AS_LONG( bufsize ) );
                  } else if ( argc == 5 ) {
                     branch = tree->Branch( PyString_AS_STRING( name ), klName.c_str(), buf,
                        PyInt_AS_LONG( bufsize ), PyInt_AS_LONG( splitlevel ) );
                  }

                  return BindRootObject( branch, TBranch::Class() );
               }
            }
         }

      // still here? Then call original Branch() to reach the other overloads:
         Py_INCREF( (PyObject*)self );
         fOrg->fSelf = self;
         PyObject* result = PyObject_Call( (PyObject*)fOrg, args, kwds );
         fOrg->fSelf = 0;
         Py_DECREF( (PyObject*)self );

         return result;
      }
   };

//____________________________________________________________________________
   class TTreeSetBranchAddress : public TTreeMemberFunction {
   public:
      TTreeSetBranchAddress( MethodProxy* org ) : TTreeMemberFunction( org ) {}

   public:
      virtual PyObject* GetPrototype()
      {
         return PyString_FromString( "TBranch* TTree::SetBranchAddress( ... )" );
      }

      virtual PyObject* operator()( ObjectProxy* self, PyObject* args, PyObject* kwds )
      {
      // acceptable signature:
      //   ( const char*, void* )
         int argc = PyTuple_GET_SIZE( args );

         if ( 2 == argc ) {
            TTree* tree =
               (TTree*)self->ObjectIsA()->DynamicCast( TTree::Class(), self->GetObject() );

            if ( ! tree ) {
               PyErr_SetString( PyExc_TypeError,
                  "TTree::SetBranchAddress must be called with a TTree instance as first argument" );
               return 0;
            }

            PyObject *name = 0, *address = 0;

         // try: ( const char*, void* )
            if ( PyArg_ParseTuple( args, const_cast< char* >( "SO:SetBranchAddress" ),
                    &name, &address ) ) {

               void* buf = 0;
               if ( ObjectProxy_Check( address ) ) {
                  if ( ((ObjectProxy*)address)->fFlags & ObjectProxy::kIsReference )
                     buf = (void*)((ObjectProxy*)address)->fObject;
                  else
                     buf = (void*)&((ObjectProxy*)address)->fObject;
               } else
                  Utility::GetBuffer( address, '*', 1, buf, kFALSE );

               if ( buf != 0 ) {
                  tree->SetBranchAddress( PyString_AS_STRING( name ), buf );

                  Py_INCREF( Py_None );
                  return Py_None;
               }
            }
         }

      // still here? Then call original Branch() to reach the other overloads:
         Py_INCREF( (PyObject*)self );
         fOrg->fSelf = self;
         PyObject* result = PyObject_Call( (PyObject*)fOrg, args, kwds );
         fOrg->fSelf = 0;
         Py_DECREF( (PyObject*)self );

         return result;
      }
   };

} // namespace PyROOT


namespace {

// for convenience
   using namespace PyROOT;

//- TFN behaviour ------------------------------------------------------------
   typedef std::pair< PyObject*, int > CallInfo_t;

   int TFNPyCallback( G__value* res, G__CONST char*, struct G__param* libp, int hash )
   {
      PyObject* result = 0;

   // retrieve function information
      G__ifunc_table* ifunc = 0;
      int index = 0;
      G__CurrentCall( G__RECMEMFUNCENV, &ifunc, index );

      CallInfo_t* ci = (CallInfo_t*)ifunc->userparam[index];

      if ( ci->first != 0 ) {
      // prepare arguments and call
         PyObject* arg1 = BufFac_t::Instance()->PyBuffer_FromMemory(
            (double*)G__int(libp->para[0]), 4 );

         if ( ci->second != 0 ) {
            PyObject* arg2 = BufFac_t::Instance()->PyBuffer_FromMemory(
               (double*)G__int(libp->para[1]), ci->second );

            result = PyObject_CallFunction( ci->first, (char*)"OO", arg1, arg2 );
            Py_DECREF( arg2 );
         } else
            result = PyObject_CallFunction( ci->first, (char*)"O", arg1 );

         Py_DECREF( arg1 );
      }

   // translate result, throw if an error has occurred
      double d = 0.;
      if ( ! result ) {
         PyErr_Print();
         throw std::runtime_error( "TFN python function call failed" );
      } else {
         d = PyFloat_AsDouble( result );
         Py_DECREF( result );
      }

      G__letdouble( res, 100, d );
      return ( 1 || hash || res || libp );
   }

//- TMinuit behaviour --------------------------------------------------------
   int TMinuitPyCallback( G__value* res, G__CONST char*, struct G__param* libp, int hash )
   {
      PyObject* result = 0;

   // retrieve function information
      G__ifunc_table* ifunc = 0;
      int index = 0;
      G__CurrentCall( G__RECMEMFUNCENV, &ifunc, index );

      PyObject* pyfunc = (PyObject*)ifunc->userparam[index];

      if ( pyfunc != 0 ) {
      // prepare arguments
         PyObject* arg1 = BufFac_t::Instance()->PyBuffer_FromMemory(
            (Int_t*)G__int(libp->para[0]), 1 );
         int npar = G__int(libp->para[0]);
 
         PyObject* arg2 = BufFac_t::Instance()->PyBuffer_FromMemory(
            (Double_t*)G__int(libp->para[1]), npar );

         PyObject* arg3 = PyList_New(1);
         PyList_SetItem( arg3, 0, PyFloat_FromDouble( G__double(libp->para[2]) ) );

         PyObject* arg4 = BufFac_t::Instance()->PyBuffer_FromMemory(
            (Double_t*)G__int(libp->para[3]), npar );

      // perform actual call
         result = PyObject_CallFunction( pyfunc, (char*)"OOOOi",
            arg1, arg2, arg3, arg4, (int)G__int(libp->para[4]) );
         *(Double_t*)G__Doubleref(&libp->para[2]) = PyFloat_AsDouble( PyList_GetItem( arg3, 0 ) );

         Py_DECREF( arg2 ); Py_DECREF( arg3 ); Py_DECREF( arg4 );
      }

      if ( ! result ) {
         PyErr_Print();
         throw std::runtime_error( "TMinuit python fit function call failed" );
      }

      Py_XDECREF( result );

      return ( 1 || hash || res || libp );
   }

//____________________________________________________________________________
   class TPretendInterpreted: public PyCallable {
      static int fgCount;

   public:
      TPretendInterpreted( int nArgs ) : fNArgs( nArgs ) {}

   public:
      Int_t GetNArgs() { return fNArgs; }

      Bool_t IsCallable( PyObject* pyobject )
      {
         if ( ! pyobject || ! PyCallable_Check( pyobject ) ) {
            PyObject* str = pyobject ? PyObject_Str( pyobject ) : PyString_FromString( "null pointer" );
            PyErr_Format( PyExc_ValueError,
               "\"%s\" is not a valid python callable", PyString_AS_STRING( str ) );
            Py_DECREF( str );
            return kFALSE;
         }

         return kTRUE;
      }

      G__MethodInfo Register( void* callback,
         const char* name, const char* signature, const char* retcode )
      {
      // build CINT function placeholder
         G__ClassInfo gcl;                   // global namespace

         Long_t offset = 0;
         G__MethodInfo m = gcl.GetMethod( name, signature, &offset );

         if ( ! m.IsValid() ) {
         // create a new global function
            m = gcl.AddMethod( retcode, name, signature );       // boundary safe

         // offset counter from this that services to associate pyobject with tp2f
            fgCount += 1;

         // setup association for CINT
            G__ifunc_table* ifunc = m.ifunc();
            int index = m.Index();

            ifunc->pentry[index]->size        = -1;
            ifunc->pentry[index]->filenum     = -1;
            ifunc->pentry[index]->line_number = -1;
            ifunc->pentry[index]->tp2f = (void*)((Long_t)this + fgCount);
            ifunc->pentry[index]->p    = callback;

         // setup association for ourselves
            int tag = -6666 - fgCount;
            ifunc->p_tagtable[index] = tag;
         }

         return m;
      }

   private:
      Int_t fNArgs;
   };

   int TPretendInterpreted::fgCount = 0;

//____________________________________________________________________________
   class TF1InitWithPyFunc : public TPretendInterpreted {
      typedef std::pair< PyObject*, int > pairPyObjInt_t;

   public:
      TF1InitWithPyFunc( int ntf = 1 ) : TPretendInterpreted( 2 + 2*ntf ) {}

   public:
      virtual PyObject* GetSignature() { return PyString_FromString( "(...)" ); }
      virtual PyObject* GetPrototype()
      {
         return PyString_FromString(
            "TF1::TF1(const char* name, PyObject* callable, "
            "Double_t xmin, Double_t xmax, Int_t npar = 0)" );
      }

      virtual PyObject* operator()( ObjectProxy* self, PyObject* args, PyObject* )
      {
      // expected signature: ( char* name, pyfunc, double xmin, double xmax, int npar = 0 )
         int argc = PyTuple_GET_SIZE( args );
         const int reqNArgs = GetNArgs();
         if ( ! ( argc == reqNArgs || argc == reqNArgs+1 ) ) {
            PyErr_Format( PyExc_TypeError,
               "TFN::TFN(const char*, PyObject* callable, ...) =>\n"
               "    takes at least %d and at most %d arguments (%d given)",
               reqNArgs, reqNArgs+1, argc );
            return 0;              // reported as an overload failure
         }

         PyObject* pyfunc = PyTuple_GET_ITEM( args, 1 );
         if ( ! IsCallable( pyfunc ) )
            return 0;

      // use requested function name as identifier
         const char* name = PyString_AsString( PyTuple_GET_ITEM( args, 0 ) );
         if ( PyErr_Occurred() )
            return 0;

      // build placeholder, get CINT info
         G__MethodInfo m = Register( (void*)TFNPyCallback, name, "double*, double*", "D" );
 
      // get proper table block of "interpreted" function and the index into it
         G__ifunc_table* ifunc = m.ifunc();
         int index = m.Index();

      // verify/setup the callback parameters
         int npar = 0;             // default value if not given
         if ( argc == reqNArgs+1 )
            npar = PyInt_AsLong( PyTuple_GET_ITEM( args, reqNArgs ) );

         if ( ! ifunc->userparam[index] ) {
         // no func yet, install current one
            Py_INCREF( pyfunc );
            ifunc->userparam[index] = (void*)new pairPyObjInt_t( pyfunc, npar );
         } else {
         // old func: flip if different, keep if same
            pairPyObjInt_t* oldp = (pairPyObjInt_t*)ifunc->userparam[index];
            if ( oldp->first != pyfunc ) {
               Py_INCREF( pyfunc ); Py_DECREF( oldp->first );
               oldp->first = pyfunc;
            }

            oldp->second = npar;             // setting is quicker than checking
         }

      // get constructor
         MethodProxy* method =
            (MethodProxy*)PyObject_GetAttrString( (PyObject*)self, (char*)"__init__" );

      // build new argument array
         PyObject* newArgs = PyTuple_New( reqNArgs + 1 );

         for ( int iarg = 0; iarg < argc; ++iarg ) {
            PyObject* item = PyTuple_GET_ITEM( args, iarg );
            if ( iarg != 1 ) {
               Py_INCREF( item );
               PyTuple_SET_ITEM( newArgs, iarg, item );
            } else {
               PyTuple_SET_ITEM( newArgs, iarg,
                  PyCObject_FromVoidPtr( (void*)ifunc->pentry[index]->tp2f, NULL ) );
            }
         }

         if ( argc == reqNArgs )             // meaning: use default for last value
            PyTuple_SET_ITEM( newArgs, reqNArgs, PyInt_FromLong( 0l ) );

      // re-run
         PyObject* result = PyObject_CallObject( (PyObject*)method, newArgs );

      // done, may have worked, if not: 0 is returned
         Py_DECREF( newArgs );
         Py_DECREF( method );
         return result;
      }
   };

//____________________________________________________________________________
   class TF2InitWithPyFunc : public TF1InitWithPyFunc {
   public:
      TF2InitWithPyFunc() : TF1InitWithPyFunc( 2 ) {}

   public:
      virtual PyObject* GetPrototype()
      {
         return PyString_FromString(
            "TF2::TF2(const char* name, PyObject* callable, "
            "Double_t xmin, Double_t xmax, "
            "Double_t ymin, Double_t ymax, Int_t npar = 0)" );
      }
   };

//____________________________________________________________________________
   class TF3InitWithPyFunc : public TF1InitWithPyFunc {
   public:
      TF3InitWithPyFunc() : TF1InitWithPyFunc( 3 ) {}

   public:
      virtual PyObject* GetPrototype()
      {
         return PyString_FromString(
            "TF3::TF3(const char* name, PyObject* callable, "
            "Double_t xmin, Double_t xmax, "
            "Double_t ymin, Double_t ymax, "
            "Double_t zmin, Double_t zmax, Int_t npar = 0)" );
      }
   };


//- TFunction behaviour --------------------------------------------------------
   PyObject* TFunctionCall( PyObject*, PyObject* args ) {
      if ( PyTuple_GET_SIZE( args ) < 1 || ! ObjectProxy_Check( PyTuple_GET_ITEM( args, 0 ) ) ) {
         PyErr_SetString( PyExc_TypeError,
            "unbound method __call__ requires TFunction instance as first argument" );
         return 0;
      }

      PyObject* newArgs = PyTuple_GetSlice( args, 1, PyTuple_GET_SIZE( args ) );
      PyObject* result = TFunctionHolder(
         (TFunction*)((ObjectProxy*)PyTuple_GET_ITEM( args, 0 ))->GetObject() )( 0, newArgs, 0 );
      Py_DECREF( newArgs );

      return result;
   }


//- TMinuit behaviour ----------------------------------------------------------
   class TMinuitSetFCN : public TPretendInterpreted {
   public:
      TMinuitSetFCN() : TPretendInterpreted( 1 ) {}

   public:
      virtual PyObject* GetSignature() { return PyString_FromString( "(PyObject* callable)" ); }
      virtual PyObject* GetPrototype()
      {
         return PyString_FromString(
            "TMinuit::SetFCN(PyObject* callable)" );
      }

      virtual PyObject* operator()( ObjectProxy* self, PyObject* args, PyObject* )
      {
      // expected signature: ( pyfunc )
         int argc = PyTuple_GET_SIZE( args );
         if ( argc != 1 ) {
            PyErr_Format( PyExc_TypeError,
               "TMinuit::SetFCN(PyObject* callable, ...) =>\n"
               "    takes exactly 1 argument (%d given)", argc );
            return 0;              // reported as an overload failure
         }

         PyObject* pyfunc = PyTuple_GET_ITEM( args, 0 );
         if ( ! IsCallable( pyfunc ) )
            return 0;

      // use callable name (if available) as identifier
         PyObject* pyname = PyObject_GetAttrString( pyfunc, (char*)"__name__" );
         const char* name = "dummy";
         if ( pyname != 0 )
            name = PyString_AsString( pyname );

      // build placeholder, get CINT info
         G__MethodInfo m = Register( (void*)TMinuitPyCallback, name,
            "int&, double*, double&, double*, int", "V" );

      // get proper table block of "interpreted" function and the index into it
         G__ifunc_table* ifunc = m.ifunc();
         int index = m.Index();

      // setup the callback parameter
         Py_INCREF( pyfunc );
         if ( ifunc->userparam[index] )
            Py_DECREF((PyObject*)ifunc->userparam[index]);
         ifunc->userparam[index] = (void*)pyfunc;

      // get function
         MethodProxy* method =
            (MethodProxy*)PyObject_GetAttrString( (PyObject*)self, (char*)"SetFCN" );

      // build new argument array
         PyObject* newArgs = PyTuple_New( 1 );
         PyTuple_SET_ITEM( newArgs, 0,
            PyCObject_FromVoidPtr( (void*)ifunc->pentry[index]->tp2f, NULL ) );

      // re-run
         PyObject* result = PyObject_CallObject( (PyObject*)method, newArgs );

      // done, may have worked, if not: 0 is returned
         Py_DECREF( newArgs );
         Py_DECREF( method );
         return result;
      }
   };


} // unnamed namespace


//- public functions -----------------------------------------------------------
Bool_t PyROOT::Pythonize( PyObject* pyclass, const std::string& name )
{
   if ( pyclass == 0 )
      return kFALSE;

// for smart pointer style classes (note fall-through)
   if ( PyObject_HasAttrString( pyclass, (char*)"__deref__" ) ) {
      Utility::AddToClass( pyclass, "__getattr__", (PyCFunction) DeRefGetAttr );
   } else if ( PyObject_HasAttrString( pyclass, (char*)"__follow__" ) ) {
      Utility::AddToClass( pyclass, "__getattr__", (PyCFunction) FollowGetAttr );
   }

   if ( name == "TObject" ) {
   // support for the 'in' operator
      Utility::AddToClass( pyclass, "__contains__", (PyCFunction) TObjectContains );

   // comparing for lists
      Utility::AddToClass( pyclass, "__cmp__", (PyCFunction) TObjectCompare );
      Utility::AddToClass( pyclass, "__eq__",  (PyCFunction) TObjectIsEqual );

      return kTRUE;
   }

   if ( name == "TClass" ) {
   // make DynamicCast return a usable python object, rather than void*
      Utility::AddToClass( pyclass, "_TClass__DynamicCast", "DynamicCast" );
      Utility::AddToClass( pyclass, "DynamicCast", (PyCFunction) TClassDynamicCast );

   // the following cast is easier to use (reads both ways)
      Utility::AddToClass( pyclass, "StaticCast", (PyCFunction) TClassStaticCast );

      return kTRUE;
   }

   if ( name == "TCollection" ) {
      Utility::AddToClass( pyclass, "append",   "Add" );
      Utility::AddToClass( pyclass, "extend",   (PyCFunction) TCollectionExtend );
      Utility::AddToClass( pyclass, "remove",   (PyCFunction) TCollectionRemove );
      Utility::AddToClass( pyclass, "__add__",  (PyCFunction) TCollectionAdd );
      Utility::AddToClass( pyclass, "__imul__", (PyCFunction) TCollectionIMul );
      Utility::AddToClass( pyclass, "__mul__",  (PyCFunction) TCollectionMul );
      Utility::AddToClass( pyclass, "__rmul__", (PyCFunction) TCollectionMul );

      Utility::AddToClass( pyclass, "count", (PyCFunction) TCollectionCount );

      Utility::AddToClass( pyclass, "__len__",  "GetSize" );
      Utility::AddToClass( pyclass, "__iter__", (PyCFunction) TCollectionIter );

      return kTRUE;
   }

   if ( name == "TSeqCollection" ) {
      Utility::AddToClass( pyclass, "__getitem__", (PyCFunction) TSeqCollectionGetItem );
      Utility::AddToClass( pyclass, "__setitem__", (PyCFunction) TSeqCollectionSetItem );
      Utility::AddToClass( pyclass, "__delitem__", (PyCFunction) TSeqCollectionDelItem );

      Utility::AddToClass( pyclass, "insert",  (PyCFunction) TSeqCollectionInsert );
      Utility::AddToClass( pyclass, "pop",     (PyCFunction) TSeqCollectionPop );
      Utility::AddToClass( pyclass, "reverse", (PyCFunction) TSeqCollectionReverse );
      Utility::AddToClass( pyclass, "sort",    (PyCFunction) TSeqCollectionSort );

      Utility::AddToClass( pyclass, "index", (PyCFunction) TSeqCollectionIndex );

      return kTRUE;
   }

   if ( name == "TClonesArray" ) {
   // restore base TSeqCollection operator[] to prevent random object creation (it's
   // functionality is equivalent to the operator[](int) const of TClonesArray, but
   // there's no guarantee it'll be selected over the non-const version)
      Utility::AddToClass( pyclass, "__getitem__", (PyCFunction) TSeqCollectionGetItem );

   // this setitem should be used with as much care as the C++ one
      Utility::AddToClass( pyclass, "__setitem__", (PyCFunction) TClonesArraySetItem );

      return kTRUE;
   }

   if ( IsTemplatedSTLClass( name, "vector" ) ) {
      Utility::AddToClass( pyclass, "__len__",     "size" );

      if ( PyObject_HasAttrString( pyclass, const_cast< char* >( "at" ) ) )
         Utility::AddToClass( pyclass, "_vector__at", "at" );
      else if ( PyObject_HasAttrString( pyclass, const_cast< char* >( "__getitem__" ) ) ) {
         Utility::AddToClass( pyclass, "_vector__at", "__getitem__" );   // unchecked!
      // for unchecked getitem, it is necessary to have a checked iterator protocol
         Utility::AddToClass( pyclass, "__iter__",    (PyCFunction) StlSequenceIter );
      }

   // provide a slice-able __getitem__, if possible
      if ( PyObject_HasAttrString( pyclass, const_cast< char* >( "_vector__at" ) ) )
         Utility::AddToClass( pyclass, "__getitem__", (PyCFunction) VectorGetItem );

      return kTRUE;
   }

   if ( IsTemplatedSTLClass( name, "list" ) ) {
      Utility::AddToClass( pyclass, "__len__",  "size" );
      Utility::AddToClass( pyclass, "__iter__", (PyCFunction) StlSequenceIter );

      return kTRUE;
   }

   if ( name.find( "iterator" ) != std::string::npos ) {
      Utility::AddToClass( pyclass, "__cmp__", (PyCFunction) StlIterCompare );

      Utility::AddToClass( pyclass, "next", (PyCFunction) StlIterNext );

      return kTRUE;
   }

   if ( name == "string" || name == "std::string" ) {
      Utility::AddToClass( pyclass, "__repr__", (PyCFunction) StlStringRepr );
      Utility::AddToClass( pyclass, "__str__", "c_str" );
      Utility::AddToClass( pyclass, "__len__", "length" );
      Utility::AddToClass( pyclass, "__cmp__", (PyCFunction) StlStringCompare );
      Utility::AddToClass( pyclass, "__eq__",  (PyCFunction) StlStringIsequal );

      return kTRUE;
   }

   if ( name == "TString" ) {
      Utility::AddToClass( pyclass, "__repr__", (PyCFunction) TStringRepr );
      Utility::AddToClass( pyclass, "__str__", "Data" );
      Utility::AddToClass( pyclass, "__len__", "Length" );

      Utility::AddToClass( pyclass, "__cmp__", "CompareTo" );
      Utility::AddToClass( pyclass, "__eq__",  (PyCFunction) TStringIsequal );

      return kTRUE;
   }

   if ( name == "TObjString" ) {
      Utility::AddToClass( pyclass, "__repr__", (PyCFunction) TObjStringRepr );
      Utility::AddToClass( pyclass, "__str__",  "GetName" );
      Utility::AddToClass( pyclass, "__len__",  (PyCFunction) TObjStringLength );

      Utility::AddToClass( pyclass, "__cmp__", (PyCFunction) TObjStringCompare );
      Utility::AddToClass( pyclass, "__eq__",  (PyCFunction) TObjStringIsequal );

      return kTRUE;
   }

   if ( name == "TIter" ) {
      Utility::AddToClass( pyclass, "__iter__", (PyCFunction) TIterIter );
      Utility::AddToClass( pyclass, "next",     (PyCFunction) TIterNext );

      return kTRUE;
   }

   if ( IsTemplatedSTLClass( name, "map" ) )
      return Utility::AddToClass( pyclass, "__len__", "size" );

   if ( name == "TDirectory" ) {
   // note: this replaces the already existing TDirectory::GetObject()
      Utility::AddToClass( pyclass, "GetObject", (PyCFunction) TDirectoryGetObject );

   // note: this replaces the already existing TDirectory::WriteObject()
      Utility::AddToClass( pyclass, "WriteObject", (PyCFunction) TDirectoryWriteObject );

      return kTRUE;
   }

   if ( name == "TTree" ) {
   // allow direct browsing of the tree
      Utility::AddToClass( pyclass, "__getattr__", (PyCFunction) TTreeGetAttr );

   // workaround for templated member Branch()
      MethodProxy* original = (MethodProxy*)PyObject_GetAttrString(
         pyclass, const_cast< char* >( "Branch" ) );
      MethodProxy* method = MethodProxy_New( "Branch", new TTreeBranch( original ) );
      Py_DECREF( original ); original = 0;

      PyObject_SetAttrString(
         pyclass, const_cast< char* >( method->GetName().c_str() ), (PyObject*)method );
      Py_DECREF( method ); method = 0;

   // workaround for templated member SetBranchAddress()
      original = (MethodProxy*)PyObject_GetAttrString(
         pyclass, const_cast< char* >( "SetBranchAddress" ) );
      method = MethodProxy_New( "SetBranchAddress", new TTreeSetBranchAddress( original ) );
      Py_DECREF( original ); original = 0;

      PyObject_SetAttrString(
         pyclass, const_cast< char* >( method->GetName().c_str() ), (PyObject*)method );
      Py_DECREF( method ); method = 0;

      return kTRUE;
   }

   if ( name == "TF1" )       // allow instantiation with python callable
      return Utility::AddToClass( pyclass, "__init__", new TF1InitWithPyFunc );

   if ( name == "TF2" )       // allow instantiation with python callable
      return Utility::AddToClass( pyclass, "__init__", new TF2InitWithPyFunc );

   if ( name == "TF3" )       // allow instantiation with python callable
      return Utility::AddToClass( pyclass, "__init__", new TF3InitWithPyFunc );

   if ( name == "TFunction" ) // allow direct call
      return Utility::AddToClass( pyclass, "__call__", (PyCFunction) TFunctionCall );

   if ( name == "TMinuit" )   // allow call with python callable
      return Utility::AddToClass( pyclass, "SetFCN", new TMinuitSetFCN );

   if ( name == "TFile" )     // allow member-style access to entries in file
      return Utility::AddToClass( pyclass, "__getattr__", "Get" );

// default (no pythonization) is by definition ok
   return kTRUE;
}
