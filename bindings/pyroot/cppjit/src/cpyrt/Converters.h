#ifndef CPYRT_CONVERTERS_H
#define CPYRT_CONVERTERS_H

// Bindings
#include "Dimensions.h"

// Standard
#include <string>

namespace cppjit::cpyrt {

struct Parameter;
struct CallContext;

class CPYRT_CLASS_EXPORT Converter {
public:
  virtual ~Converter();

  Converter() = default;

  Converter(Converter const& other) = delete;
  Converter(Converter&& other) = delete;
  Converter& operator=(Converter const& other) = delete;
  Converter& operator=(Converter&& other) = delete;

public:
  virtual bool SetArg(PyObject*, Parameter&, CallContext* = nullptr) = 0;
  virtual PyObject* FromMemory(void* address);
  virtual bool ToMemory(PyObject* value, void* address,
                        PyObject* ctxt = nullptr);
  virtual bool HasState() { return false; }
  virtual std::string GetFailureMsg() { return "[Converter]"; }
};

// create/destroy converter from fully qualified type (public API)
CPYRT_EXPORT Converter* CreateConverter(const std::string& fullType,
                                        cdims_t dims = 0);
CPYRT_EXPORT Converter* CreateConverter(interop::TCppType_t type,
                                        cdims_t dims = 0);
CPYRT_EXPORT void DestroyConverter(Converter* p);
typedef Converter* (*cf_t)(cdims_t d);
CPYRT_EXPORT bool RegisterConverter(const std::string& name, cf_t fac);
CPYRT_EXPORT bool RegisterConverterAlias(const std::string& name,
                                         const std::string& target);
CPYRT_EXPORT bool UnregisterConverter(const std::string& name);

// converters for special cases (only here b/c of external use of
// StrictInstancePtrConverter)
class VoidArrayConverter : public Converter {
public:
  VoidArrayConverter(bool keepControl = true,
                     const std::string& failureMsg = std::string())
      : fFailureMsg(failureMsg) {
    fKeepControl = keepControl;
  }

public:
  bool SetArg(PyObject*, Parameter&, CallContext* = nullptr) override;
  PyObject* FromMemory(void* address) override;
  bool ToMemory(PyObject* value, void* address,
                PyObject* ctxt = nullptr) override;
  bool HasState() override { return true; }
  std::string GetFailureMsg() override {
    return "[VoidArrayConverter] " + fFailureMsg;
  }

protected:
  virtual bool GetAddressSpecialCase(PyObject* pyobject, void*& address);
  bool KeepControl() { return fKeepControl; }
  const std::string fFailureMsg;

private:
  bool fKeepControl;
};

template <bool ISCONST> class InstancePtrConverter : public VoidArrayConverter {
public:
  InstancePtrConverter(interop::TCppScope_t klass, bool keepControl = false,
                       const std::string& failureMsg = std::string())
      : VoidArrayConverter(keepControl, failureMsg),
        fClass(interop::GetUnderlyingScope(klass)) {}

public:
  bool SetArg(PyObject*, Parameter&, CallContext* = nullptr) override;
  PyObject* FromMemory(void* address) override;
  bool ToMemory(PyObject* value, void* address,
                PyObject* ctxt = nullptr) override;

protected:
  interop::TCppScope_t fClass;
};

class StrictInstancePtrConverter : public InstancePtrConverter<false> {
public:
  using InstancePtrConverter<false>::InstancePtrConverter;

protected:
  bool GetAddressSpecialCase(PyObject*, void*&) override { return false; }
};

} // namespace cppjit::cpyrt

#endif // !CPYRT_CONVERTERS_H
