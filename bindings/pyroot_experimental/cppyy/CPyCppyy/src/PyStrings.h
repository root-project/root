#ifndef CPYCPPYY_PYSTRINGS_H
#define CPYCPPYY_PYSTRINGS_H

// ROOT
#include "DllImport.h"

namespace CPyCppyy {

// python strings kept for performance reasons

namespace PyStrings {

    CPYCPPYY_IMPORT PyObject* gAssign;
    CPYCPPYY_IMPORT PyObject* gBases;
    CPYCPPYY_IMPORT PyObject* gBase;
    CPYCPPYY_IMPORT PyObject* gCppName;
    CPYCPPYY_IMPORT PyObject* gDeref;
    CPYCPPYY_IMPORT PyObject* gPreInc;
    CPYCPPYY_IMPORT PyObject* gPostInc;
    CPYCPPYY_IMPORT PyObject* gDict;
    CPYCPPYY_IMPORT PyObject* gEmptyString;
    CPYCPPYY_IMPORT PyObject* gEq;
    CPYCPPYY_IMPORT PyObject* gFollow;
    CPYCPPYY_IMPORT PyObject* gGetItem;
    CPYCPPYY_IMPORT PyObject* gGetNoCheck;
    CPYCPPYY_IMPORT PyObject* gInit;
    CPYCPPYY_IMPORT PyObject* gIter;
    CPYCPPYY_IMPORT PyObject* gLen;
    CPYCPPYY_IMPORT PyObject* gLifeLine;
    CPYCPPYY_IMPORT PyObject* gModule;
    CPYCPPYY_IMPORT PyObject* gMRO;
    CPYCPPYY_IMPORT PyObject* gName;
    CPYCPPYY_IMPORT PyObject* gNe;
    CPYCPPYY_IMPORT PyObject* gTypeCode;
    CPYCPPYY_IMPORT PyObject* gCTypesType;

    CPYCPPYY_IMPORT PyObject* gUnderlying;

    CPYCPPYY_IMPORT PyObject* gAdd;
    CPYCPPYY_IMPORT PyObject* gSub;
    CPYCPPYY_IMPORT PyObject* gMul;
    CPYCPPYY_IMPORT PyObject* gDiv;

    CPYCPPYY_IMPORT PyObject* gLShift;
    CPYCPPYY_IMPORT PyObject* gLShiftC;

    CPYCPPYY_IMPORT PyObject* gAt;
    CPYCPPYY_IMPORT PyObject* gBegin;
    CPYCPPYY_IMPORT PyObject* gEnd;
    CPYCPPYY_IMPORT PyObject* gFirst;
    CPYCPPYY_IMPORT PyObject* gSecond;
    CPYCPPYY_IMPORT PyObject* gSize;
    CPYCPPYY_IMPORT PyObject* gTemplate;
    CPYCPPYY_IMPORT PyObject* gVectorAt;

    CPYCPPYY_IMPORT PyObject* gCppReal;
    CPYCPPYY_IMPORT PyObject* gCppImag;

    CPYCPPYY_IMPORT PyObject* gThisModule;

    CPYCPPYY_IMPORT PyObject* gNoImplicit;

    CPYCPPYY_IMPORT PyObject* gExPythonize;
    CPYCPPYY_IMPORT PyObject* gPythonize;

} // namespace PyStrings

bool CreatePyStrings();
PyObject* DestroyPyStrings();

} // namespace CPyCppyy

#endif // !CPYCPPYY_PYSTRINGS_H
