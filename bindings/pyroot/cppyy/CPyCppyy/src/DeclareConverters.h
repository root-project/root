#ifndef CPYCPPYY_DECLARECONVERTERS_H
#define CPYCPPYY_DECLARECONVERTERS_H

// Bindings
#include "Converters.h"
#include "Dimensions.h"

// Standard
#include <complex>
#include <string>

// ROOT
#include "ROOT/RStringView.hxx"
#include "TString.h"

namespace CPyCppyy {

namespace {

#define CPPYY_DECLARE_BASIC_CONVERTER(name)                                  \
class name##Converter : public Converter {                                   \
public:                                                                      \
    virtual bool SetArg(PyObject*, Parameter&, CallContext* = nullptr);      \
    virtual PyObject* FromMemory(void*);                                     \
    virtual bool ToMemory(PyObject*, void*, PyObject* = nullptr);            \
};                                                                           \
                                                                             \
class Const##name##RefConverter : public Converter {                         \
public:                                                                      \
    virtual bool SetArg(PyObject*, Parameter&, CallContext* = nullptr);      \
    virtual PyObject* FromMemory(void*);                                     \
}


#define CPPYY_DECLARE_BASIC_CONVERTER2(name, base)                           \
class name##Converter : public base##Converter {                             \
public:                                                                      \
    virtual PyObject* FromMemory(void*);                                     \
    virtual bool ToMemory(PyObject*, void*, PyObject* = nullptr);            \
};                                                                           \
                                                                             \
class Const##name##RefConverter : public Converter {                         \
public:                                                                      \
    virtual bool SetArg(PyObject*, Parameter&, CallContext* = nullptr);      \
    virtual PyObject* FromMemory(void*);                                     \
}

#define CPPYY_DECLARE_REFCONVERTER(name)                                     \
class name##RefConverter : public Converter {                                \
public:                                                                      \
    virtual bool SetArg(PyObject*, Parameter&, CallContext* = nullptr);      \
    virtual PyObject* FromMemory(void*);                                     \
};

#define CPPYY_DECLARE_ARRAY_CONVERTER(name)                                  \
class name##ArrayConverter : public Converter {                              \
public:                                                                      \
    name##ArrayConverter(cdims_t dims);                                      \
    name##ArrayConverter(const name##ArrayConverter&) = delete;              \
    name##ArrayConverter& operator=(const name##ArrayConverter&) = delete;   \
    virtual bool SetArg(PyObject*, Parameter&, CallContext* = nullptr);      \
    virtual PyObject* FromMemory(void*);                                     \
    virtual bool ToMemory(PyObject*, void*, PyObject* = nullptr);            \
    virtual bool HasState() { return true; }                                 \
protected:                                                                   \
    dims_t fShape;                                                           \
    bool fIsFixed;                                                           \
};


// converters for built-ins
CPPYY_DECLARE_BASIC_CONVERTER(Long);
CPPYY_DECLARE_BASIC_CONVERTER(Bool);
CPPYY_DECLARE_BASIC_CONVERTER(Char);
class SCharAsIntConverter : public CharConverter {
public:
    using CharConverter::CharConverter;
    virtual PyObject* FromMemory(void*);
};
CPPYY_DECLARE_BASIC_CONVERTER(UChar);
class UCharAsIntConverter : public UCharConverter {
public:
    using UCharConverter::UCharConverter;
    virtual PyObject* FromMemory(void*);
};
CPPYY_DECLARE_BASIC_CONVERTER(WChar);
CPPYY_DECLARE_BASIC_CONVERTER(Char16);
CPPYY_DECLARE_BASIC_CONVERTER(Char32);
CPPYY_DECLARE_BASIC_CONVERTER(Int8);
CPPYY_DECLARE_BASIC_CONVERTER(UInt8);
CPPYY_DECLARE_BASIC_CONVERTER(Short);
CPPYY_DECLARE_BASIC_CONVERTER(UShort);
CPPYY_DECLARE_BASIC_CONVERTER(Int);
CPPYY_DECLARE_BASIC_CONVERTER(ULong);
CPPYY_DECLARE_BASIC_CONVERTER2(UInt, ULong);
CPPYY_DECLARE_BASIC_CONVERTER(LLong);
CPPYY_DECLARE_BASIC_CONVERTER(ULLong);
CPPYY_DECLARE_BASIC_CONVERTER(Double);
CPPYY_DECLARE_BASIC_CONVERTER(Float);
CPPYY_DECLARE_BASIC_CONVERTER(LDouble);

CPPYY_DECLARE_REFCONVERTER(Bool);
CPPYY_DECLARE_REFCONVERTER(Char);
CPPYY_DECLARE_REFCONVERTER(WChar);
CPPYY_DECLARE_REFCONVERTER(Char16);
CPPYY_DECLARE_REFCONVERTER(Char32);
CPPYY_DECLARE_REFCONVERTER(SChar);
CPPYY_DECLARE_REFCONVERTER(UChar);
CPPYY_DECLARE_REFCONVERTER(Int8);
CPPYY_DECLARE_REFCONVERTER(UInt8);
CPPYY_DECLARE_REFCONVERTER(Short);
CPPYY_DECLARE_REFCONVERTER(UShort);
CPPYY_DECLARE_REFCONVERTER(UInt);
CPPYY_DECLARE_REFCONVERTER(Int);
CPPYY_DECLARE_REFCONVERTER(Long);
CPPYY_DECLARE_REFCONVERTER(ULong);
CPPYY_DECLARE_REFCONVERTER(LLong);
CPPYY_DECLARE_REFCONVERTER(ULLong);
CPPYY_DECLARE_REFCONVERTER(Float);
CPPYY_DECLARE_REFCONVERTER(Double);
CPPYY_DECLARE_REFCONVERTER(LDouble);

class VoidConverter : public Converter {
public:
    virtual bool SetArg(PyObject*, Parameter&, CallContext* = nullptr);
};

class CStringConverter : public Converter {
public:
    CStringConverter(std::string::size_type maxSize = std::string::npos) : fMaxSize(maxSize) {}

public:
    virtual bool SetArg(PyObject*, Parameter&, CallContext* = nullptr);
    virtual PyObject* FromMemory(void* address);
    virtual bool ToMemory(PyObject* value, void* address, PyObject* = nullptr);
    virtual bool HasState() { return true; }

protected:
    std::string fBuffer;
    std::string::size_type fMaxSize;
};

class NonConstCStringConverter : public CStringConverter {
public:
    using CStringConverter::CStringConverter;

public:
    virtual bool SetArg(PyObject*, Parameter&, CallContext* = nullptr);
    virtual PyObject* FromMemory(void* address);
};

class WCStringConverter : public Converter {
public:
    WCStringConverter(std::wstring::size_type maxSize = std::wstring::npos) :
        fBuffer(nullptr), fMaxSize(maxSize) {}
    WCStringConverter(const WCStringConverter&) = delete;
    WCStringConverter& operator=(const WCStringConverter&) = delete;
    virtual ~WCStringConverter() { free(fBuffer); }

public:
    virtual bool SetArg(PyObject*, Parameter&, CallContext* = nullptr);
    virtual PyObject* FromMemory(void* address);
    virtual bool ToMemory(PyObject* value, void* address, PyObject* = nullptr);
    virtual bool HasState() { return true; }

protected:
    wchar_t* fBuffer;
    std::wstring::size_type fMaxSize;
};

class CString16Converter : public Converter {
public:
    CString16Converter(std::wstring::size_type maxSize = std::wstring::npos) :
        fBuffer(nullptr), fMaxSize(maxSize) {}
    CString16Converter(const CString16Converter&) = delete;
    CString16Converter& operator=(const CString16Converter&) = delete;
    virtual ~CString16Converter() { free(fBuffer); }

public:
    virtual bool SetArg(PyObject*, Parameter&, CallContext* = nullptr);
    virtual PyObject* FromMemory(void* address);
    virtual bool ToMemory(PyObject* value, void* address, PyObject* = nullptr);
    virtual bool HasState() { return true; }

protected:
    char16_t* fBuffer;
    std::wstring::size_type fMaxSize;
};

class CString32Converter : public Converter {
public:
    CString32Converter(std::wstring::size_type maxSize = std::wstring::npos) :
        fBuffer(nullptr), fMaxSize(maxSize) {}
    CString32Converter(const CString32Converter&) = delete;
    CString32Converter& operator=(const CString32Converter&) = delete;
    virtual ~CString32Converter() { free(fBuffer); }

public:
    virtual bool SetArg(PyObject*, Parameter&, CallContext* = nullptr);
    virtual PyObject* FromMemory(void* address);
    virtual bool ToMemory(PyObject* value, void* address, PyObject* = nullptr);
    virtual bool HasState() { return true; }

protected:
    char32_t* fBuffer;
    std::wstring::size_type fMaxSize;
};

// pointer/array conversions
CPPYY_DECLARE_ARRAY_CONVERTER(Bool);
CPPYY_DECLARE_ARRAY_CONVERTER(SChar);
CPPYY_DECLARE_ARRAY_CONVERTER(UChar);
#if __cplusplus > 201402L
CPPYY_DECLARE_ARRAY_CONVERTER(Byte);
#endif
CPPYY_DECLARE_ARRAY_CONVERTER(Int8);
CPPYY_DECLARE_ARRAY_CONVERTER(UInt8);
CPPYY_DECLARE_ARRAY_CONVERTER(Short);
CPPYY_DECLARE_ARRAY_CONVERTER(UShort);
CPPYY_DECLARE_ARRAY_CONVERTER(Int);
CPPYY_DECLARE_ARRAY_CONVERTER(UInt);
CPPYY_DECLARE_ARRAY_CONVERTER(Long);
CPPYY_DECLARE_ARRAY_CONVERTER(ULong);
CPPYY_DECLARE_ARRAY_CONVERTER(LLong);
CPPYY_DECLARE_ARRAY_CONVERTER(ULLong);
CPPYY_DECLARE_ARRAY_CONVERTER(Float);
CPPYY_DECLARE_ARRAY_CONVERTER(Double);
CPPYY_DECLARE_ARRAY_CONVERTER(LDouble);
CPPYY_DECLARE_ARRAY_CONVERTER(ComplexF);
CPPYY_DECLARE_ARRAY_CONVERTER(ComplexD);

class CStringArrayConverter : public SCharArrayConverter {
public:
    CStringArrayConverter(cdims_t dims, bool fixed) : SCharArrayConverter(dims) {
        fIsFixed = fixed;    // overrides SCharArrayConverter decision
    }
    using SCharArrayConverter::SCharArrayConverter;
    virtual bool SetArg(PyObject*, Parameter&, CallContext* = nullptr);
    virtual PyObject* FromMemory(void* address);
    virtual bool ToMemory(PyObject*, void*, PyObject* = nullptr);

private:
    std::vector<const char*> fBuffer;
};

class NonConstCStringArrayConverter : public CStringArrayConverter {
public:
    using CStringArrayConverter::CStringArrayConverter;
    virtual PyObject* FromMemory(void* address);
};

// converters for special cases
class NullptrConverter : public Converter {
public:
    virtual bool SetArg(PyObject*, Parameter&, CallContext* = nullptr);
};

class InstanceConverter : public StrictInstancePtrConverter {
public:
    using StrictInstancePtrConverter::StrictInstancePtrConverter;
    virtual bool SetArg(PyObject*, Parameter&, CallContext* = nullptr);
    virtual PyObject* FromMemory(void*);
    virtual bool ToMemory(PyObject*, void*, PyObject* = nullptr);
};

class InstanceRefConverter : public Converter  {
public:
    InstanceRefConverter(Cppyy::TCppType_t klass, bool isConst) :
        fClass(klass), fIsConst(isConst) {}

public:
    virtual bool SetArg(PyObject*, Parameter&, CallContext* = nullptr);
    virtual PyObject* FromMemory(void* address);
    virtual bool HasState() { return true; }

protected:
    Cppyy::TCppType_t fClass;
    bool fIsConst;
};

class InstanceMoveConverter : public InstanceRefConverter  {
public:
    InstanceMoveConverter(Cppyy::TCppType_t klass) : InstanceRefConverter(klass, true) {}
    virtual bool SetArg(PyObject*, Parameter&, CallContext* = nullptr);
};

template <bool ISREFERENCE>
class InstancePtrPtrConverter : public InstancePtrConverter<false> {
public:
    using InstancePtrConverter::InstancePtrConverter;

public:
    virtual bool SetArg(PyObject*, Parameter&, CallContext* = nullptr);
    virtual PyObject* FromMemory(void* address);
    virtual bool ToMemory(PyObject* value, void* address, PyObject* = nullptr);
};

class InstanceArrayConverter : public InstancePtrConverter<false> {
public:
    InstanceArrayConverter(Cppyy::TCppType_t klass, cdims_t dims, bool keepControl = false) :
            InstancePtrConverter<false>(klass, keepControl), fShape(dims) { }
    InstanceArrayConverter(const InstanceArrayConverter&) = delete;
    InstanceArrayConverter& operator=(const InstanceArrayConverter&) = delete;

public:
    virtual bool SetArg(PyObject*, Parameter&, CallContext* = nullptr);
    virtual PyObject* FromMemory(void* address);
    virtual bool ToMemory(PyObject* value, void* address, PyObject* = nullptr);

protected:
    dims_t fShape;
};


class ComplexDConverter: public InstanceConverter {
public:
    ComplexDConverter(bool keepControl = false);

public:
    virtual bool SetArg(PyObject*, Parameter&, CallContext* = nullptr);
    virtual PyObject* FromMemory(void* address);
    virtual bool ToMemory(PyObject* value, void* address, PyObject* = nullptr);
    virtual bool HasState() { return true; }

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
    VoidPtrPtrConverter(cdims_t dims);

public:
    virtual bool SetArg(PyObject*, Parameter&, CallContext* = nullptr);
    virtual PyObject* FromMemory(void* address);
    virtual bool HasState() { return true; }

protected:
    dims_t fShape;
    bool fIsFixed;
};

CPPYY_DECLARE_BASIC_CONVERTER(PyObject);


#define CPPYY_DECLARE_STRING_CONVERTER(name, strtype)                        \
class name##Converter : public InstanceConverter {                           \
public:                                                                      \
    name##Converter(bool keepControl = true);                                \
                                                                             \
public:                                                                      \
    virtual bool SetArg(PyObject*, Parameter&, CallContext* = nullptr);      \
    virtual PyObject* FromMemory(void* address);                             \
    virtual bool ToMemory(PyObject*, void*, PyObject* = nullptr);            \
    virtual bool HasState() { return true; }                                 \
                                                                             \
protected:                                                                   \
    strtype fBuffer;                                                         \
}

CPPYY_DECLARE_STRING_CONVERTER(TString, TString);
CPPYY_DECLARE_STRING_CONVERTER(STLString, std::string);
CPPYY_DECLARE_STRING_CONVERTER(STLWString, std::wstring);
#if __cplusplus > 201402L
CPPYY_DECLARE_STRING_CONVERTER(STLStringView, std::string_view);
#endif

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
    virtual PyObject* FromMemory(void* address);
    virtual bool ToMemory(PyObject*, void*, PyObject* = nullptr);
    virtual bool HasState() { return true; }

protected:
    std::string fRetType;
    std::string fSignature;
};

// std::function
class StdFunctionConverter : public FunctionPointerConverter {
public:
    StdFunctionConverter(Converter* cnv, const std::string& ret, const std::string& sig) :
        FunctionPointerConverter(ret, sig), fConverter(cnv) {}
    StdFunctionConverter(const StdFunctionConverter&) = delete;
    StdFunctionConverter& operator=(const StdFunctionConverter&) = delete;
    virtual ~StdFunctionConverter() { delete fConverter; }

public:
    virtual bool SetArg(PyObject*, Parameter&, CallContext* = nullptr);
    virtual PyObject* FromMemory(void* address);
    virtual bool ToMemory(PyObject* value, void* address, PyObject* = nullptr);

protected:
    Converter* fConverter;
};


// smart pointer converter
class SmartPtrConverter : public Converter {
public:
    SmartPtrConverter(Cppyy::TCppType_t smart,
                      Cppyy::TCppType_t underlying,
                      bool keepControl = false,
                      bool isRef = false)
        : fSmartPtrType(smart), fUnderlyingType(underlying),
          fKeepControl(keepControl), fIsRef(isRef) {}

public:
    virtual bool SetArg(PyObject*, Parameter&, CallContext* = nullptr);
    virtual PyObject* FromMemory(void* address);
    //virtual bool ToMemory(PyObject*, void*, PyObject* = nullptr);
    virtual bool HasState() { return true; }

protected:
    virtual bool GetAddressSpecialCase(PyObject*, void*&) { return false; }

    Cppyy::TCppType_t   fSmartPtrType;
    Cppyy::TCppType_t   fUnderlyingType;
    bool                fKeepControl;
    bool                fIsRef;
};


// initializer lists
class InitializerListConverter : public InstanceConverter {
public:
    InitializerListConverter(Cppyy::TCppType_t klass, std::string const& value_type);
    InitializerListConverter(const InitializerListConverter&) = delete;
    InitializerListConverter& operator=(const InitializerListConverter&) = delete;
    virtual ~InitializerListConverter();

public:
    virtual bool SetArg(PyObject*, Parameter&, CallContext* = nullptr);
    virtual bool HasState() { return true; }

protected:
    void Clear();

protected:
    void*             fBuffer = nullptr;
    std::vector<Converter*> fConverters;
    std::string       fValueTypeName;
    Cppyy::TCppType_t fValueType;
    size_t            fValueSize;
};


// raising converter to take out overloads
class NotImplementedConverter : public Converter {
public:
    virtual bool SetArg(PyObject*, Parameter&, CallContext* = nullptr);
};

} // unnamed namespace

} // namespace CPyCppyy

#endif // !CPYCPPYY_DECLARECONVERTERS_H
