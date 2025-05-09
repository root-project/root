#ifndef CPYCPPYY_CALLCONTEXT_H
#define CPYCPPYY_CALLCONTEXT_H

// Standard
#include <vector>

#include <sys/types.h>


namespace CPyCppyy {

// small number that allows use of stack for argument passing
const int SMALL_ARGS_N = 8;

// convention to pass flag for direct calls (similar to Python's vector calls)
#define DIRECT_CALL ((size_t)1 << (8 * sizeof(size_t) - 1))

#ifndef CPYCPPYY_PARAMETER
#define CPYCPPYY_PARAMETER
// general place holder for function parameters
struct Parameter {
    union Value {
        bool                 fBool;
        int8_t               fInt8;
        int16_t              fInt16;
        int32_t              fInt32;
        uint8_t              fUInt8;
        uint16_t             fUInt16;
        uint32_t             fUInt32;
        short                fShort;
        unsigned short       fUShort;
        int                  fInt;
        unsigned int         fUInt;
        long                 fLong;
        intptr_t             fIntPtr;
        unsigned long        fULong;
        long long            fLLong;
        unsigned long long   fULLong;
        int64_t              fInt64;
        uint64_t             fUInt64;
        float                fFloat;
        double               fDouble;
        long double          fLDouble;
        void*                fVoidp;
    } fValue;
    void* fRef;
    char  fTypeCode;
};
#endif // CPYCPPYY_PARAMETER

// extra call information
struct CallContext {
    CallContext() : fCurScope(0), fPyContext(nullptr), fFlags(0),
        fArgsVec(nullptr), fNArgs(0), fTemps(nullptr) {}
    CallContext(const CallContext&) = delete;
    CallContext& operator=(const CallContext&) = delete;
    ~CallContext() { if (fTemps) Cleanup(); delete fArgsVec; }

    enum ECallFlags {
        kNone           = 0x000000,
        kIsSorted       = 0x000001, // if method overload priority determined
        kIsCreator      = 0x000002, // if method creates python-owned objects
        kIsConstructor  = 0x000004, // if method is a C++ constructor
        kHaveImplicit   = 0x000008, // indicate that implicit converters are available
        kAllowImplicit  = 0x000010, // indicate that implicit conversions are allowed
        kNoImplicit     = 0x000020, // disable implicit to prevent recursion
        kCallDirect     = 0x000040, // call wrapped method directly, no inheritance
        kFromDescr      = 0x000080, // initiated from a descriptor
        kUseHeuristics  = 0x000100, // if method applies heuristics memory policy
        kUseStrict      = 0x000200, // if method applies strict memory policy
        kReleaseGIL     = 0x000400, // if method should release the GIL
        kSetLifeLine    = 0x000800, // if return value is part of 'this'
        kNeverLifeLine  = 0x001000, // if the return value is never part of 'this'
        kPyException    = 0x002000, // Python exception during method execution
        kCppException   = 0x004000, // C++ exception during method execution
        kProtected      = 0x008000, // if method should return on signals
        kUseFFI         = 0x010000, // not implemented
        kIsPseudoFunc   = 0x020000, // internal, used for introspection
    };

// memory handling
    static ECallFlags sMemoryPolicy;
    static bool SetMemoryPolicy(ECallFlags e);

    void AddTemporary(PyObject* pyobj);
    void Cleanup();

// signal safety
    static ECallFlags sSignalPolicy;
    static bool SetGlobalSignalPolicy(bool setProtected);

    Parameter* GetArgs(size_t sz) {
        if (sz != (size_t)-1) fNArgs = sz;
        if (fNArgs <= SMALL_ARGS_N) return fArgs;
        if (!fArgsVec) fArgsVec = new std::vector<Parameter>();
        fArgsVec->resize(fNArgs);
        return fArgsVec->data();
    }

    Parameter* GetArgs() {
        if (fNArgs <= SMALL_ARGS_N) return fArgs;
        return fArgsVec->data();
    }

    size_t GetSize() { return fNArgs; }
    size_t GetEncodedSize() { return fNArgs | ((fFlags & kCallDirect) ? DIRECT_CALL : 0); }

public:
// info/status
    Cppyy::TCppScope_t fCurScope;
    PyObject*          fPyContext;
    uint32_t           fFlags;

private:
    struct Temporary { PyObject* fPyObject; Temporary* fNext; };

// payload
    Parameter               fArgs[SMALL_ARGS_N];
    std::vector<Parameter>* fArgsVec;
    size_t                  fNArgs;
    Temporary*              fTemps;
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

inline bool HaveImplicit(CallContext* ctxt) {
    return ctxt ? (!(ctxt->fFlags & CallContext::kNoImplicit) && (ctxt->fFlags & CallContext::kHaveImplicit)) : false;
}

inline bool AllowImplicit(CallContext* ctxt) {
    return ctxt ? (!(ctxt->fFlags & CallContext::kNoImplicit) && (ctxt->fFlags & CallContext::kAllowImplicit)) : false;
}

inline bool NoImplicit(CallContext* ctxt) {
    return ctxt ? (ctxt->fFlags & CallContext::kNoImplicit) : false;
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

template<CallContext::ECallFlags F>
class CallContextRAII {
public:
    CallContextRAII(CallContext* ctxt) : fCtxt(ctxt) {
        fPrior = fCtxt->fFlags & F;
        fCtxt->fFlags |= F;
    }
    CallContextRAII(const CallContextRAII&) = delete;
    CallContextRAII& operator=(const CallContextRAII&) = delete;
    ~CallContextRAII() {
        if (!fPrior) fCtxt->fFlags &= ~F;
    }

private:
    CallContext* fCtxt;
    bool fPrior;
};

} // namespace CPyCppyy

#endif // !CPYCPPYY_CALLCONTEXT_H
