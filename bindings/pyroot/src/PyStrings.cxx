// Author: Wim Lavrijsen, Nov 2008

// Bindings
#include "PyROOT.h"
#include "PyStrings.h"


//- data _____________________________________________________________________
PyObject *PyROOT::PyStrings::gBases = nullptr;
PyObject *PyROOT::PyStrings::gBase = nullptr;
PyObject *PyROOT::PyStrings::gClass = nullptr;
PyObject *PyROOT::PyStrings::gCppEq = nullptr;
PyObject *PyROOT::PyStrings::gCppNe = nullptr;
PyObject *PyROOT::PyStrings::gDeref = nullptr;
PyObject *PyROOT::PyStrings::gDict = nullptr;
PyObject *PyROOT::PyStrings::gEmptyString = nullptr;
PyObject *PyROOT::PyStrings::gEq = nullptr;
PyObject *PyROOT::PyStrings::gFollow = nullptr;
PyObject *PyROOT::PyStrings::gGetItem = nullptr;
PyObject *PyROOT::PyStrings::gInit = nullptr;
PyObject *PyROOT::PyStrings::gIter = nullptr;
PyObject *PyROOT::PyStrings::gLen = nullptr;
PyObject *PyROOT::PyStrings::gLifeLine = nullptr;
PyObject *PyROOT::PyStrings::gModule = nullptr;
PyObject *PyROOT::PyStrings::gMRO = nullptr;
PyObject *PyROOT::PyStrings::gName = nullptr;
PyObject *PyROOT::PyStrings::gCppName = nullptr;
PyObject *PyROOT::PyStrings::gNe = nullptr;
PyObject *PyROOT::PyStrings::gTypeCode = nullptr;

PyObject *PyROOT::PyStrings::gAdd = nullptr;
PyObject *PyROOT::PyStrings::gSub = nullptr;
PyObject *PyROOT::PyStrings::gMul = nullptr;
PyObject *PyROOT::PyStrings::gDiv = nullptr;

PyObject *PyROOT::PyStrings::gAt = nullptr;
PyObject *PyROOT::PyStrings::gBegin = nullptr;
PyObject *PyROOT::PyStrings::gEnd = nullptr;
PyObject *PyROOT::PyStrings::gFirst = nullptr;
PyObject *PyROOT::PyStrings::gSecond = nullptr;
PyObject *PyROOT::PyStrings::gSize = nullptr;
PyObject *PyROOT::PyStrings::gGetSize = nullptr;
PyObject *PyROOT::PyStrings::ggetSize = nullptr;
PyObject *PyROOT::PyStrings::gTemplate = nullptr;
PyObject *PyROOT::PyStrings::gVectorAt = nullptr;

PyObject *PyROOT::PyStrings::gBranch = nullptr;
PyObject *PyROOT::PyStrings::gFitFCN = nullptr;
PyObject *PyROOT::PyStrings::gROOTns = nullptr;
PyObject *PyROOT::PyStrings::gSetBranchAddress = nullptr;
PyObject *PyROOT::PyStrings::gSetFCN = nullptr;
PyObject *PyROOT::PyStrings::gTClassDynCast = nullptr;

////////////////////////////////////////////////////////////////////////////////

#define PYROOT_INITIALIZE_STRING( var, str )                                 \
   if ( ! ( PyStrings::var = PyROOT_PyUnicode_InternFromString( (char*)#str ) ) )    \
      return kFALSE

Bool_t PyROOT::CreatePyStrings() {
// Build cache of commonly used python strings (the cache is python intern, so
// all strings are shared python-wide, not just in PyROOT).
   PYROOT_INITIALIZE_STRING( gBases, __bases__ );
   PYROOT_INITIALIZE_STRING( gBase, __base__ );
   PYROOT_INITIALIZE_STRING( gClass, __class__ );
   PYROOT_INITIALIZE_STRING( gCppEq, __cpp_eq__ );
   PYROOT_INITIALIZE_STRING( gCppNe, __cpp_ne__ );
   PYROOT_INITIALIZE_STRING( gDeref, __deref__ );
   PYROOT_INITIALIZE_STRING( gDict, __dict__ );
   if ( ! ( PyStrings::gEmptyString = PyROOT_PyUnicode_FromString( (char*)"" ) ) )
      return kFALSE;
   PYROOT_INITIALIZE_STRING( gEq, __eq__ );
   PYROOT_INITIALIZE_STRING( gFollow, __follow__ );
   PYROOT_INITIALIZE_STRING( gGetItem, __getitem__ );
   PYROOT_INITIALIZE_STRING( gInit, __init__ );
   PYROOT_INITIALIZE_STRING( gIter, __iter__ );
   PYROOT_INITIALIZE_STRING( gLen, __len__ );
   PYROOT_INITIALIZE_STRING( gLifeLine, __lifeline );
   PYROOT_INITIALIZE_STRING( gModule, __module__ );
   PYROOT_INITIALIZE_STRING( gMRO, __mro__ );
   PYROOT_INITIALIZE_STRING( gName, __name__ );
   PYROOT_INITIALIZE_STRING( gCppName, __cppname__ );
   PYROOT_INITIALIZE_STRING( gNe, __ne__ );
   PYROOT_INITIALIZE_STRING( gTypeCode, typecode );

   PYROOT_INITIALIZE_STRING( gAdd, __add__ );
   PYROOT_INITIALIZE_STRING( gSub, __sub__ );
   PYROOT_INITIALIZE_STRING( gMul, __mul__ );
   PYROOT_INITIALIZE_STRING( gDiv, PYROOT__div__ );

   PYROOT_INITIALIZE_STRING( gAt, at );
   PYROOT_INITIALIZE_STRING( gBegin, begin );
   PYROOT_INITIALIZE_STRING( gEnd, end );
   PYROOT_INITIALIZE_STRING( gFirst, first );
   PYROOT_INITIALIZE_STRING( gSecond, second );
   PYROOT_INITIALIZE_STRING( gSize, size );
   PYROOT_INITIALIZE_STRING( gGetSize, GetSize );
   PYROOT_INITIALIZE_STRING( ggetSize, getSize );
   PYROOT_INITIALIZE_STRING( gTemplate, Template );
   PYROOT_INITIALIZE_STRING( gVectorAt, _vector__at );

   PYROOT_INITIALIZE_STRING( gBranch, Branch );
   PYROOT_INITIALIZE_STRING( gFitFCN, FitFCN );
   PYROOT_INITIALIZE_STRING( gROOTns, ROOT );
   PYROOT_INITIALIZE_STRING( gSetBranchAddress, SetBranchAddress );
   PYROOT_INITIALIZE_STRING( gSetFCN, SetFCN );
   PYROOT_INITIALIZE_STRING( gTClassDynCast, _TClass__DynamicCast );

   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Remove all cached python strings.

PyObject* PyROOT::DestroyPyStrings() {
   Py_DECREF( PyStrings::gBases );
   PyStrings::gBases = nullptr;
   Py_DECREF( PyStrings::gBase );
   PyStrings::gBase = nullptr;
   Py_DECREF( PyStrings::gClass );
   PyStrings::gClass = nullptr;
   Py_DECREF( PyStrings::gCppEq );
   PyStrings::gCppEq = nullptr;
   Py_DECREF( PyStrings::gCppNe );
   PyStrings::gCppNe = nullptr;
   Py_DECREF( PyStrings::gDeref );
   PyStrings::gDeref = nullptr;
   Py_DECREF( PyStrings::gDict );
   PyStrings::gDict = nullptr;
   Py_DECREF( PyStrings::gEmptyString );
   PyStrings::gEmptyString = nullptr;
   Py_DECREF( PyStrings::gEq );
   PyStrings::gEq = nullptr;
   Py_DECREF( PyStrings::gFollow );
   PyStrings::gFollow = nullptr;
   Py_DECREF( PyStrings::gGetItem );
   PyStrings::gGetItem = nullptr;
   Py_DECREF( PyStrings::gInit );
   PyStrings::gInit = nullptr;
   Py_DECREF( PyStrings::gIter );
   PyStrings::gIter = nullptr;
   Py_DECREF( PyStrings::gLen );
   PyStrings::gLen = nullptr;
   Py_DECREF( PyStrings::gLifeLine );
   PyStrings::gLifeLine = nullptr;
   Py_DECREF( PyStrings::gModule );
   PyStrings::gModule = nullptr;
   Py_DECREF( PyStrings::gMRO );
   PyStrings::gMRO = nullptr;
   Py_DECREF( PyStrings::gName );
   PyStrings::gName = nullptr;
   Py_DECREF( PyStrings::gCppName );
   PyStrings::gCppName = nullptr;
   Py_DECREF( PyStrings::gNe );
   PyStrings::gNe = nullptr;
   Py_DECREF( PyStrings::gTypeCode );
   PyStrings::gTypeCode = nullptr;

   Py_DECREF( PyStrings::gAdd );
   PyStrings::gAdd = nullptr;
   Py_DECREF( PyStrings::gSub );
   PyStrings::gSub = nullptr;
   Py_DECREF( PyStrings::gMul );
   PyStrings::gMul = nullptr;
   Py_DECREF( PyStrings::gDiv );
   PyStrings::gDiv = nullptr;

   Py_DECREF( PyStrings::gAt );
   PyStrings::gAt = nullptr;
   Py_DECREF( PyStrings::gBegin );
   PyStrings::gBegin = nullptr;
   Py_DECREF( PyStrings::gEnd );
   PyStrings::gEnd = nullptr;
   Py_DECREF( PyStrings::gFirst );
   PyStrings::gFirst = nullptr;
   Py_DECREF( PyStrings::gSecond );
   PyStrings::gSecond = nullptr;
   Py_DECREF( PyStrings::gSize );
   PyStrings::gSize = nullptr;
   Py_DECREF( PyStrings::gGetSize );
   PyStrings::gGetSize = nullptr;
   Py_DECREF( PyStrings::ggetSize );
   PyStrings::ggetSize = nullptr;
   Py_DECREF( PyStrings::gTemplate );
   PyStrings::gTemplate = nullptr;
   Py_DECREF( PyStrings::gVectorAt );
   PyStrings::gVectorAt = nullptr;

   Py_DECREF( PyStrings::gBranch );
   PyStrings::gBranch = nullptr;
   Py_DECREF( PyStrings::gFitFCN );
   PyStrings::gFitFCN = nullptr;
   Py_DECREF( PyStrings::gROOTns );
   PyStrings::gROOTns = nullptr;
   Py_DECREF( PyStrings::gSetBranchAddress );
   PyStrings::gSetBranchAddress = nullptr;
   Py_DECREF( PyStrings::gSetFCN );
   PyStrings::gSetFCN = nullptr;
   Py_DECREF( PyStrings::gTClassDynCast );
   PyStrings::gTClassDynCast = nullptr;

   Py_INCREF( Py_None );
   return Py_None;
}
