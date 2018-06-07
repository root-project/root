#ifndef CPYCPPYY_CALLCONTEXT_H
#define CPYCPPYY_CALLCONTEXT_H

// Standard
#include <vector>


namespace CPyCppyy {

// general place holder for function parameters
struct Parameter {
    union Value {
        bool           fBool;
        short          fShort;
        unsigned short fUShort;
        Int_t          fInt;
        UInt_t         fUInt;
        Long_t         fLong;
        ULong_t        fULong;
        Long64_t       fLongLong;
        ULong64_t      fULongLong;
        float          fFloat;
        double         fDouble;
        LongDouble_t   fLongDouble;
        void*          fVoidp;
    } fValue;
    void* fRef;
    char  fTypeCode;
};

// extra call information
struct CallContext {
    CallContext(std::vector<Parameter>::size_type sz = 0) : fArgs(sz), fFlags(0) {}

    enum ECallFlags {
        kNone           =    0,
        kIsSorted       =    1,   // if method overload priority determined
        kIsCreator      =    2,   // if method creates python-owned objects
        kIsConstructor  =    4,   // if method is a C++ constructor
        kUseHeuristics  =    8,   // if method applies heuristics memory policy
        kUseStrict      =   16,   // if method applies strict memory policy
        kManageSmartPtr =   32,   // if executor should manage smart pointers
        kReleaseGIL     =   64,   // if method should release the GIL
        kFast           =  128,   // if method should NOT handle signals
        kSafe           =  256    // if method should return on signals
    };

// memory handling
    static ECallFlags sMemoryPolicy;
    static bool SetMemoryPolicy(ECallFlags e);

// signal safety
    static ECallFlags sSignalPolicy;
    static bool SetSignalPolicy(ECallFlags e);

// payload
    std::vector<Parameter> fArgs;
    uint64_t fFlags;
};

inline bool IsSorted(uint64_t flags) {
    return flags & CallContext::kIsSorted;
}

inline bool IsCreator(uint64_t flags) {
    return flags & CallContext::kIsCreator;
}

inline bool IsConstructor(uint64_t flags) {
    return flags & CallContext::kIsConstructor;
}

inline bool ManagesSmartPtr(CallContext* ctxt) {
    return ctxt->fFlags & CallContext::kManageSmartPtr;
}

inline bool ReleasesGIL(uint64_t flags) {
    return flags & CallContext::kReleaseGIL;
}

inline bool ReleasesGIL(CallContext* ctxt) {
    return ctxt ? (ctxt->fFlags & CallContext::kReleaseGIL) : false;
}

inline bool UseStrictOwnership(CallContext* ctxt) {
    if (ctxt && (ctxt->fFlags & CallContext::kUseStrict))
        return true;
    if (ctxt && (ctxt->fFlags & CallContext::kUseHeuristics))
        return false;

    return CallContext::sMemoryPolicy == CallContext::kUseStrict;
}

} // namespace CPyCppyy

#endif // !CPYCPPYY_CALLCONTEXT_H
