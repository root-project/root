#ifndef CPYCPPYY_CALLCONTEXT_H
#define CPYCPPYY_CALLCONTEXT_H

// Standard
#include <vector>


namespace CPyCppyy {

// small number that allows use of stack for argument passing
const int SMALL_ARGS_N = 8;

// general place holder for function parameters
struct Parameter {
    union Value {
        bool           fBool;
        short          fShort;
        unsigned short fUShort;
        Int_t          fInt;
        UInt_t         fUInt;
        Long_t         fLong;
        intptr_t       fIntPtr;
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
    CallContext() : fFlags(0), fNArgs(0), fArgsVec(nullptr) {}
    CallContext(const CallContext&) = delete;
    CallContext& operator=(const CallContext&) = delete;
    ~CallContext() { delete fArgsVec; }

    enum ECallFlags {
        kNone           =    0,
        kIsSorted       =    1,   // if method overload priority determined
        kIsCreator      =    2,   // if method creates python-owned objects
        kIsConstructor  =    4,   // if method is a C++ constructor
        kUseHeuristics  =    8,   // if method applies heuristics memory policy
        kUseStrict      =   16,   // if method applies strict memory policy
        kReleaseGIL     =   32,   // if method should release the GIL
        kFast           =   64,   // if method should NOT handle signals
        kSafe           =  128    // if method should return on signals
    };

// memory handling
    static ECallFlags sMemoryPolicy;
    static bool SetMemoryPolicy(ECallFlags e);

// signal safety
    static ECallFlags sSignalPolicy;
    static bool SetSignalPolicy(ECallFlags e);

    Parameter* GetArgs(size_t sz = (size_t)-1) {
        if (sz != (size_t)-1) fNArgs = sz;
        if (fNArgs <= SMALL_ARGS_N) return fArgs;
        if (!fArgsVec) fArgsVec = new std::vector<Parameter>();
        fArgsVec->resize(fNArgs);
        return fArgsVec->data();
    }
 
    size_t GetSize() { return fNArgs; }

public:
    uint64_t fFlags;
        
private:
// payload
    Parameter fArgs[SMALL_ARGS_N];
    size_t fNArgs;
    std::vector<Parameter>* fArgsVec;
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
