#ifndef CPYCPPYY_CALLCONTEXT_H
#define CPYCPPYY_CALLCONTEXT_H

// Standard
#include <vector>

//Bindings
#include "cpp_cppyy.h"

//ROOT
#include "Rtypes.h"

namespace CPyCppyy {

// general place holder for function parameters
    struct Parameter {
        union Value {
            bool             fBool;
            short            fShort;
            unsigned short   fUShort;
            int              fInt;
            unsigned int     fUInt;
            long             fLong;
            unsigned long    fULong;
            Long64_t         fLongLong;
            ULong64_t        fULongLong;
            float            fFloat;
            double           fDouble;
            LongDouble_t     fLongDouble;
            void*            fVoidp;
        } fValue;
        void* fRef;
        char  fTypeCode;
    };

} // namespace CPyCppyy

#endif // !CPYCPPYY_CALLCONTEXT_H
