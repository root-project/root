#ifndef CPYCPPYY_DECLARECONVERTERS_H
#define CPYCPPYY_DECLARECONVERTERS_H

// Bindings
#include "Converters.h"

// Standard
#include <complex>
#include <string>
#include "ROOT/RStringView.hxx"

// ROOT
#include "TString.h"


namespace CPyCppyy {

namespace {

#define CPPYY_DECLARE_BASIC_CONVERTER(name)                                  \
class name##Converter : public Converter {                                   \
public:                                                                      \
    virtual bool SetArg(PyObject*, Parameter&, CallContext* = nullptr);      \
    virtual PyObject* FromMemory(void*);                                     \
    virtual bool ToMemory(PyObject*, void*);                                 \
};                                                                           \
                                                                             \
class Const##name##RefConverter : public Converter {                         \
public:                                                                      \
    virtual bool SetArg(PyObject*, Parameter&, CallContext* = nullptr);      \
}


#define CPPYY_DECLARE_BASIC_CONVERTER2(name, base)                           \
class name##Converter : public base##Converter {                             \
public:                                                                      \
    virtual PyObject* FromMemory(void*);                                     \
    virtual bool ToMemory(PyObject*, void*);                                 \
};                                                                           \
                                                                             \
class Const##name##RefConverter : public Converter {                         \
public:                                                                      \
    virtual bool SetArg(PyObject*, Parameter&, CallContext* = nullptr);      \
}

#define CPPYY_DECLARE_REF_CONVERTER(name)                                    \
class name##RefConverter : public Converter {                                \
public:                                                                      \
    virtual bool SetArg(PyObject*, Parameter&, CallContext* = nullptr);      \
};

#define CPPYY_DECLARE_ARRAY_CONVERTER(name)                                  \
class name##Converter : public Converter {                                   \
public:                                                                      \
    name##Converter(Py_ssize_t size = -1) { fSize = size; }                  \
    virtual bool SetArg(PyObject*, Parameter&, CallContext* = nullptr);      \
    virtual PyObject* FromMemory(void*);                                     \
    virtual bool ToMemory(PyObject*, void*);                                 \
private:                                                                     \
    Py_ssize_t fSize;                                                        \
};                                                                           \
                                                                             \
class name##RefConverter : public name##Converter {                          \
public:                                                                      \
    using name##Converter::name##Converter;                                  \
    virtual bool SetArg(PyObject*, Parameter&, CallContext* = nullptr);      \
}

// converters for built-ins
CPPYY_DECLARE_BASIC_CONVERTER(Long);
CPPYY_DECLARE_BASIC_CONVERTER(Bool);
CPPYY_DECLARE_BASIC_CONVERTER(Char);
CPPYY_DECLARE_BASIC_CONVERTER(UChar);
class UCharAsIntConverter : public UCharConverter {
public:
    using UCharConverter::UCharConverter;
    virtual PyObject* FromMemory(void*);
};
CPPYY_DECLARE_BASIC_CONVERTER(WChar);
CPPYY_DECLARE_BASIC_CONVERTER(Short);
CPPYY_DECLARE_BASIC_CONVERTER(UShort);
CPPYY_DECLARE_BASIC_CONVERTER(Int);
CPPYY_DECLARE_BASIC_CONVERTER(ULong);
CPPYY_DECLARE_BASIC_CONVERTER2(UInt, ULong);
CPPYY_DECLARE_BASIC_CONVERTER(LongLong);
CPPYY_DECLARE_BASIC_CONVERTER(ULongLong);
CPPYY_DECLARE_BASIC_CONVERTER(Double);
CPPYY_DECLARE_BASIC_CONVERTER(Float);
CPPYY_DECLARE_BASIC_CONVERTER(LongDouble);

CPPYY_DECLARE_REF_CONVERTER(Int);
CPPYY_DECLARE_REF_CONVERTER(Long);
CPPYY_DECLARE_REF_CONVERTER(Double);

class VoidConverter : public Converter {
public:
    virtual bool SetArg(PyObject*, Parameter&, CallContext* = nullptr);
};

class CStringConverter : public Converter {
public:
    CStringConverter(long maxSize = -1) : fMaxSize(maxSize) {}

public:
    virtual bool SetArg(PyObject*, Parameter&, CallContext* = nullptr);
    virtual PyObject* FromMemory(void* address);
    virtual bool ToMemory(PyObject* value, void* address);

protected:
    std::string fBuffer;
    long fMaxSize;
};

class NonConstCStringConverter : public CStringConverter {
public:
    NonConstCStringConverter(long maxSize = -1) : CStringConverter(maxSize) {}

public:
    virtual bool SetArg(PyObject*, Parameter&, CallContext* = nullptr);
    virtual PyObject* FromMemory(void* address);
};

class WCStringConverter : public Converter {
public:
    WCStringConverter(long maxSize = -1) : fBuffer(nullptr), fMaxSize(maxSize) {}
    ~WCStringConverter() { free(fBuffer); }

public:
    virtual bool SetArg(PyObject*, Parameter&, CallContext* = nullptr);
    virtual PyObject* FromMemory(void* address);
    virtual bool ToMemory(PyObject* value, void* address);

protected:
    wchar_t* fBuffer;
    long fMaxSize;
};


// pointer/array conversions
CPPYY_DECLARE_ARRAY_CONVERTER(BoolArray);
CPPYY_DECLARE_ARRAY_CONVERTER(UCharArray);
CPPYY_DECLARE_ARRAY_CONVERTER(ShortArray);
CPPYY_DECLARE_ARRAY_CONVERTER(UShortArray);
CPPYY_DECLARE_ARRAY_CONVERTER(IntArray);
CPPYY_DECLARE_ARRAY_CONVERTER(UIntArray);
CPPYY_DECLARE_ARRAY_CONVERTER(LongArray);
CPPYY_DECLARE_ARRAY_CONVERTER(ULongArray);
CPPYY_DECLARE_ARRAY_CONVERTER(LLongArray);
CPPYY_DECLARE_ARRAY_CONVERTER(ULLongArray);
CPPYY_DECLARE_ARRAY_CONVERTER(FloatArray);
CPPYY_DECLARE_ARRAY_CONVERTER(DoubleArray);
CPPYY_DECLARE_ARRAY_CONVERTER(ComplexDArray);

class LongLongArrayConverter : public VoidArrayConverter {
public:
    virtual bool SetArg(PyObject*, Parameter&, CallContext* = nullptr);
};

// converters for special cases
class InstanceConverter : public StrictInstancePtrConverter {
public:
    using StrictInstancePtrConverter::StrictInstancePtrConverter;
    virtual bool SetArg(PyObject*, Parameter&, CallContext* = nullptr);
};

class InstanceRefConverter : public Converter  {
public:
    InstanceRefConverter(Cppyy::TCppType_t klass) : fClass(klass) {}

public:
    virtual bool SetArg(PyObject*, Parameter&, CallContext* = nullptr);

protected:
    Cppyy::TCppType_t fClass;
};

class InstanceMoveConverter : public InstanceRefConverter  {
public:
    using InstanceRefConverter::InstanceRefConverter;
    virtual bool SetArg(PyObject*, Parameter&, CallContext* = nullptr);
};

template <bool ISREFERENCE>
class InstancePtrPtrConverter : public InstancePtrConverter {
public:
    using InstancePtrConverter::InstancePtrConverter;

public:
    virtual bool SetArg(PyObject*, Parameter&, CallContext* = nullptr);
    virtual PyObject* FromMemory(void* address);
    virtual bool ToMemory(PyObject* value, void* address);
};

extern template class InstancePtrPtrConverter<true>;
extern template class InstancePtrPtrConverter<false>;

class InstanceArrayConverter : public InstancePtrConverter {
public:
    InstanceArrayConverter(Cppyy::TCppType_t klass, long* dims, bool keepControl = false) :
            InstancePtrConverter(klass, keepControl) {
        long size = (dims && 0 < dims[0]) ? dims[0]+1: 1;
        m_dims = new long[size];
        if (dims) {
            for (long i = 0; i < size; ++i) m_dims[i] = dims[i];
        } else {
            m_dims[0] = -1;
        }
    }
    ~InstanceArrayConverter() { delete [] m_dims; }

public:
    virtual bool SetArg(PyObject*, Parameter&, CallContext* = nullptr);
    virtual PyObject* FromMemory(void* address);
    virtual bool ToMemory(PyObject* value, void* address);

protected:
    long* m_dims;
};


class ComplexDConverter: public InstanceConverter {
public:
    ComplexDConverter(bool keepControl = false);

public:
    virtual bool SetArg(PyObject*, Parameter&, CallContext* = nullptr);
    virtual PyObject* FromMemory(void* address);
    virtual bool ToMemory(PyObject* value, void* address);

private:
    std::complex<double> fBuffer;
};


// CLING WORKAROUND -- classes for STL iterators are completely undefined in that
// they come in a bazillion different guises, so just do whatever
class STLIteratorConverter : public Converter {
public:
    virtual bool SetArg(PyObject*, Parameter&, CallContext* = nullptr);
};
// -- END CLING WORKAROUND


class VoidPtrRefConverter : public Converter {
public:
    virtual bool SetArg(PyObject*, Parameter&, CallContext* = nullptr);
};

class VoidPtrPtrConverter : public Converter {
public:
    virtual bool SetArg(PyObject*, Parameter&, CallContext* = nullptr);
    virtual PyObject* FromMemory(void* address);
};

CPPYY_DECLARE_BASIC_CONVERTER(PyObject);


#define CPPYY_DECLARE_STRING_CONVERTER(name, strtype)                        \
class name##Converter : public InstancePtrConverter {                        \
public:                                                                      \
    name##Converter(bool keepControl = true);                                \
public:                                                                      \
    virtual bool SetArg(PyObject*, Parameter&, CallContext* = nullptr);      \
    virtual PyObject* FromMemory(void* address);                             \
    virtual bool ToMemory(PyObject* value, void* address);                   \
protected:                                                                   \
    strtype fBuffer;                                                         \
}

CPPYY_DECLARE_STRING_CONVERTER(TString, TString);
CPPYY_DECLARE_STRING_CONVERTER(STLString, std::string);
CPPYY_DECLARE_STRING_CONVERTER(STLStringViewBase, std::string_view);
class STLStringViewConverter : public STLStringViewBaseConverter {
public:
    virtual bool SetArg(PyObject*, Parameter&, CallContext* = nullptr);
};
CPPYY_DECLARE_STRING_CONVERTER(STLWString, std::wstring);

class STLStringMoveConverter : public STLStringConverter {
public:
    using STLStringConverter::STLStringConverter;

public:
    virtual bool SetArg(PyObject*, Parameter&, CallContext* = nullptr);
};


// function pointers
class FunctionPointerConverter : public Converter {
public:
    FunctionPointerConverter(const std::string& ret, const std::string& sig) :
        fRetType(ret), fSignature(sig) {}

public:
    virtual bool SetArg(PyObject*, Parameter&, CallContext* = nullptr);

protected:
    std::string fRetType;
    std::string fSignature;
};


// smart pointer converter
class SmartPtrConverter : public Converter {
public:
    SmartPtrConverter(Cppyy::TCppType_t smart,
                      Cppyy::TCppType_t raw,
                      Cppyy::TCppMethod_t deref,
                      bool keepControl = false,
                      bool isRef = false)
        : fSmartPtrType(smart), fRawPtrType(raw), fDereferencer(deref),
          fKeepControl(keepControl), fIsRef(isRef) {}

public:
    virtual bool SetArg(PyObject*, Parameter&, CallContext* = nullptr);
    virtual PyObject* FromMemory(void* address);
    //virtual bool ToMemory(PyObject* value, void* address);

protected:
    virtual bool GetAddressSpecialCase(PyObject*, void*&) { return false; }

    Cppyy::TCppType_t   fSmartPtrType;
    Cppyy::TCppType_t   fRawPtrType;
    Cppyy::TCppMethod_t fDereferencer;
    bool                fKeepControl;
    bool                fIsRef;
};


// initializer lists
class InitializerListConverter : public Converter {
public:
    InitializerListConverter(Converter* cnv, size_t sz) :
        fConverter(cnv), fValueSize(sz) {}
    ~InitializerListConverter() {
        delete fConverter;
    }

public:
    virtual bool SetArg(PyObject*, Parameter&, CallContext* = nullptr);

protected:
    Converter* fConverter;
    size_t     fValueSize;
};


// raising converter to take out overloads
class NotImplementedConverter : public Converter {
public:
    virtual bool SetArg(PyObject*, Parameter&, CallContext* = nullptr);
};

} // unnamed namespace

} // namespace CPyCppyy

#endif // !CPYCPPYY_DECLARECONVERTERS_H
