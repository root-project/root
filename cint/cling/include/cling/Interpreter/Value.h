//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//------------------------------------------------------------------------------

#ifndef CLING_VALUE_H
#define CLING_VALUE_H

#include "llvm/ExecutionEngine/GenericValue.h"
#include "clang/AST/Type.h"

namespace clang {
  class ASTContext;
}

namespace cling {
  ///\brief A type, value pair.
  //
  /// Type-safe value access and setting. Simple (built-in) casting is
  /// available, but better extract the value using the template
  /// parameter that matches the Value's type.
  ///
  /// The class represents a llvm::GenericValue with its corresponding 
  /// clang::QualType. Use-cases:
  /// 1. Expression evaluation: we need to know the type of the GenericValue
  /// that we have gotten from the JIT
  /// 2. Value printer: needs to know the type in order to skip the printing of
  /// void types
  /// 3. Parameters for calls given an llvm::Function and a clang::FunctionDecl.
  class Value {
  private:
    /// \brief Forward decl for typed access specializations
    template <typename T> struct TypedAccess;

  public:
    /// \brief value
    llvm::GenericValue value;
    /// \brief the value's type
    clang::QualType type;

    /// \brief Default constructor, creates a value that IsInvalid().
    Value() {}
    /// \brief Construct a valid Value.
    Value(const llvm::GenericValue& v, clang::QualType t) :
      value(v), type(t){}

    /// \brief Determine whether the Value has been set.
    //
    /// Determine whether the Value has been set by checking
    /// whether the type is valid.
    bool isValid() const { return !type.isNull(); }

    /// \brief Determine whether the Value is unset.
    //
    /// Determine whether the Value is unset by checking
    /// whether the type is invalid.
    bool isInvalid() const { return !isValid(); }

    /// \brief Determine whether the Value is set but void.
    bool isVoid(const clang::ASTContext& ASTContext) const {
      return isValid() && type.getDesugaredType(ASTContext)->isVoidType(); }

    /// \brief Determine whether the Value is set and not void.
    //
    /// Determine whether the Value is set and not void.
    /// Only in this case can getAs() or simplisticCastAs() be called.
    bool hasValue(const clang::ASTContext& ASTContext) const {
      return isValid() && !isVoid(ASTContext); }

    /// \brief Get the value without type checking.
    template <typename T>
    T getAs() const;

    /// \brief Get the value.
    //
    /// Get the value cast to T. This is similar to reinterpret_cast<T>(value),
    /// but only works for builtin types and pointers.
    template <typename T>
    T simplisticCastAs() const;
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

#define CLING_VALUE_TYPEDACCESS(TYPE, GETTER)       \
  template<>                                        \
  struct Value::TypedAccess<TYPE> {                 \
    TYPE extract(const llvm::GenericValue& value) { \
      return value.GETTER;                          \
    }                                               \
  }

#define CLING_VALUE_TYPEDACCESS_SIGNED(TYPE)       \
  CLING_VALUE_TYPEDACCESS(signed TYPE, IntVal.getSExtValue())

#define CLING_VALUE_TYPEDACCESS_UNSIGNED(TYPE)     \
  CLING_VALUE_TYPEDACCESS(unsigned TYPE, IntVal.getZExtValue())

#define CLING_VALUE_TYPEDACCESS_BOTHSIGNS(TYPE)     \
  CLING_VALUE_TYPEDACCESS_SIGNED(TYPE);             \
  CLING_VALUE_TYPEDACCESS_UNSIGNED(TYPE);

  CLING_VALUE_TYPEDACCESS(double, DoubleVal);
  CLING_VALUE_TYPEDACCESS(float, FloatVal);
  CLING_VALUE_TYPEDACCESS(bool, IntVal.getBoolValue());

  CLING_VALUE_TYPEDACCESS_BOTHSIGNS(char)
  CLING_VALUE_TYPEDACCESS_BOTHSIGNS(short)
  CLING_VALUE_TYPEDACCESS_BOTHSIGNS(int)
  CLING_VALUE_TYPEDACCESS_BOTHSIGNS(long)
  CLING_VALUE_TYPEDACCESS_BOTHSIGNS(long long)

#undef CLING_VALUE_TYPEDACCESS_BOTHSIGNS
#undef CLING_VALUE_TYPEDACCESS_UNSIGNED
#undef CLING_VALUE_TYPEDACCESS_SIGNED
#undef CLING_VALUE_TYPEDACCESS

  template <typename T>
  T Value::getAs() const {
    // T *must* correspond to type. Else use simplisticCastAs()!
    TypedAccess<T> VI;
    return VI.extract(value);
  }
  template <typename T>
  T Value::simplisticCastAs() const {
    const clang::Type* desugared = type->getUnqualifiedDesugaredType();
    if (desugared->getTypeClass() == clang::Type::Builtin) {
      switch (desugared->getAs<clang::BuiltinType>()->getKind()) {
      case clang::BuiltinType::Bool: return (T) getAs<bool>();
      case clang::BuiltinType::Char_U: return (T) getAs<char>();
      case clang::BuiltinType::UChar: return (T) getAs<unsigned char>();
      case clang::BuiltinType::UShort: return (T) getAs<unsigned short>();
      case clang::BuiltinType::UInt: return (T) getAs<unsigned int>();
      case clang::BuiltinType::ULong: return (T) getAs<unsigned long>();
      case clang::BuiltinType::ULongLong: return (T) getAs<unsigned long long>();

      case clang::BuiltinType::Char_S: return (T) getAs<char>();
      case clang::BuiltinType::SChar: return (T) getAs<signed char>();
      case clang::BuiltinType::Short: return (T) getAs<signed short>();
      case clang::BuiltinType::Int: return (T) getAs<signed int>();
      case clang::BuiltinType::Long: return (T) getAs<signed long>();
      case clang::BuiltinType::LongLong: return (T) getAs<signed long long>();

      case clang::BuiltinType::Float: return (T) getAs<float>();
      case clang::BuiltinType::Double: return (T) getAs<double>();
      default:
        assert("Cannot cast simplistically from value's type!" && 0);
        return T();
      }
    } else if (desugared->getTypeClass() == clang::Type::Pointer) {
      return (T) getAs<void*>();
    }
    assert("unsupported type in Value, cannot cast simplistically!" && 0);
    return T();
  }

} // end namespace cling

#endif // CLING_VALUE_H
