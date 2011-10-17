//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//------------------------------------------------------------------------------

#ifndef CLING_VALUE_H
#define CLING_VALUE_H

#include "llvm/ExecutionEngine/GenericValue.h"
#include "clang/AST/Type.h"

namespace llvm {
  struct GenericValue;
}
namespace clang {
  class Type;
}

namespace cling {
  // The class represents a llvm::GenericValue with its corresponding 
  // clang::Type. There are two use-cases for that:
  // 1. Expression evaluation: we need to know the type of the GenericValue
  // that we have gotten from the JIT
  // 2. Value printer: needs to know the type in order to skip the printing of
  // void types
  class Value {
  private:
  public:
    llvm::GenericValue value;
    const clang::Type* type;
    template <typename T> struct TypedAccess;
    Value() : value(0), type(0) {}
    Value(llvm::GenericValue& v, const clang::Type* t) : value(v), type(t){}
    ~Value() {}
    template <typename T>
    T getAs() const;
  };

  template<typename T>
  struct Value::TypedAccess{
    T extract(const llvm::GenericValue& value) {
      return *reinterpret_cast<T*>(value.PointerVal);
    }
  };
  template<typename T>
  struct Value::TypedAccess<T*>{
    T* extract(const llvm::GenericValue& value) {
      return reinterpret_cast<T*>(value.PointerVal);
    }
  };
  template<>
  struct Value::TypedAccess<double> {
    double extract(const llvm::GenericValue& value) {
      return value.DoubleVal;
    }
  };
  template<>
  struct Value::TypedAccess<int> {
    double extract(const llvm::GenericValue& value) {
      return value.IntVal.getSExtValue();
    }
  };
  template<>
  struct Value::TypedAccess<bool> {
    double extract(const llvm::GenericValue& value) {
      return value.IntVal.getBoolValue();
    }
  };
  template <typename T>
  T Value::getAs() const {
    TypedAccess<T> VI;
    return VI.extract(value);
  }
} // end namespace cling

#endif // CLING_VALUE_H
