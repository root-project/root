//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_VALUE_H
#define CLING_VALUE_H

#include "cling/Interpreter/Visibility.h"

#include <cstdint>
#include <type_traits>

namespace llvm {
  class raw_ostream;
}

namespace clang {
  class ASTContext;
  class QualType;
}

// FIXME: Merge with clang::BuiltinType::getName
#define BUILTIN_TYPES                                                  \
  /*  X(void, Void) */                                                 \
  X(bool, Bool)                                                        \
  X(char, Char_S)                                                      \
  /*X(char, Char_U)*/                                                  \
  X(signed char, SChar)                                                \
  X(short, Short)                                                      \
  X(int, Int)                                                          \
  X(long, Long)                                                        \
  X(long long, LongLong)                                               \
  /*X(__int128, Int128)*/                                              \
  X(unsigned char, UChar)                                              \
  X(unsigned short, UShort)                                            \
  X(unsigned int, UInt)                                                \
  X(unsigned long, ULong)                                              \
  X(unsigned long long, ULongLong)                                     \
  /*X(unsigned __int128, UInt128)*/                                    \
  /*X(half, Half)*/                                                    \
  /*X(__bf16, BFloat16)*/                                              \
  X(float, Float)                                                      \
  X(double, Double)                                                    \
  X(long double, LongDouble)                                           \
  /*X(short _Accum, ShortAccum)                                        \
    X(_Accum, Accum)                                                   \
    X(long _Accum, LongAccum)                                          \
    X(unsigned short _Accum, UShortAccum)                              \
    X(unsigned _Accum, UAccum)                                         \
    X(unsigned long _Accum, ULongAccum)                                \
    X(short _Fract, ShortFract)                                        \
    X(_Fract, Fract)                                                   \
    X(long _Fract, LongFract)                                          \
    X(unsigned short _Fract, UShortFract)                              \
    X(unsigned _Fract, UFract)                                         \
    X(unsigned long _Fract, ULongFract)                                \
    X(_Sat short _Accum, SatShortAccum)                                \
    X(_Sat _Accum, SatAccum)                                           \
    X(_Sat long _Accum, SatLongAccum)                                  \
    X(_Sat unsigned short _Accum, SatUShortAccum)                      \
    X(_Sat unsigned _Accum, SatUAccum)                                 \
    X(_Sat unsigned long _Accum, SatULongAccum)                        \
    X(_Sat short _Fract, SatShortFract)                                \
    X(_Sat _Fract, SatFract)                                           \
    X(_Sat long _Fract, SatLongFract)                                  \
    X(_Sat unsigned short _Fract, SatUShortFract)                      \
    X(_Sat unsigned _Fract, SatUFract)                                 \
    X(_Sat unsigned long _Fract, SatULongFract)                        \
    X(_Float16, Float16)                                               \
    X(__float128, Float128)                                            \
    X(__ibm128, Ibm128)*/                                              \
  X(wchar_t, WChar_S)                                                  \
  /*X(wchar_t, WChar_U)*/                                              \
  /*X(char8_t, Char8)*/                                                \
  X(char16_t, Char16)                                                  \
  X(char32_t, Char32)                                                  \
  /*X(std::nullptr_t, NullPtr)*/


namespace cling {
  class Interpreter;

  ///\brief A type, value pair.
  //
  /// Type-safe value access and setting. Simple (built-in) casting is
  /// available, but better extract the value using the template
  /// parameter that matches the Value's type.
  ///
  /// The class represents a llvm::GenericValue with its corresponding
  /// clang::QualType. Use-cases are expression evaluation, value printing
  /// and parameters for function calls.
  ///
  class CLING_LIB_EXPORT Value {
  public:
    ///\brief Multi-purpose storage.
    ///
    union Storage {
#define X(type, name) type m_##name;

      BUILTIN_TYPES

#undef X
      void* m_Ptr; /// Can point to allocation, see needsManagedAllocation().
    };

  protected:
    /// \brief The actual value.
    Storage m_Storage;

    /// \brief If the \c Value class needs to alloc and dealloc memory.
    bool m_NeedsManagedAlloc;

    /// \brief The value's type, stored as opaque void* to reduce
    /// dependencies.
    void* m_Type;

    ///\brief Interpreter that produced the value.
    ///
    Interpreter* m_Interpreter;

    /// \brief Retrieve the underlying, canonical, desugared, unqualified type.
    EStorageType getStorageType() const { return m_StorageType; }

    /// \brief Determine the underlying, canonical, desugared, unqualified type:
    /// the element of Storage to be used.
    static EStorageType determineStorageType(clang::QualType QT);

    /// \brief Determine the underlying, canonical, desugared, unqualified type:
    /// the element of Storage to be used.
    static constexpr EStorageType determineStorageTypeT(...) {
      return kManagedAllocation;
    }

    template <class T, class = typename std::enable_if<std::is_integral<T>::value>::type>
    static constexpr EStorageType determineStorageTypeT(T*) {
      return std::is_signed<T>::value
        ? kSignedIntegerOrEnumerationType
        : kUnsignedIntegerOrEnumerationType;
    }
    static constexpr EStorageType determineStorageTypeT(double*) {
      return kDoubleType;
    }
    static constexpr EStorageType determineStorageTypeT(float*) {
      return kFloatType;
    }
    static constexpr EStorageType determineStorageTypeT(long double*) {
      return kDoubleType;
    }
    template <class T>
    static constexpr EStorageType determineStorageTypeT(T**) {
      return kPointerType;
    }
    static constexpr EStorageType determineStorageTypeT(void*) {
      return kUnsupportedType;
    }

    /// \brief Allocate storage as needed by the type.
    void ManagedAllocate();

    /// \brief Assert in case of an unsupported type. Outlined to reduce include
    ///   dependencies.
    void AssertOnUnsupportedTypeCast() const;

    bool hasPointerType() const;
    bool hasBuiltinType() const;

    // Allow simplisticCastAs to be partially specialized.
    template<typename T>
    struct CastFwd {
      static T cast(const Value& V) {
        if (V.needsManagedAllocation() || V.hasPointerType())
          return (T) (uintptr_t) V.getAs<void*>();
        if (V.hasBuiltinType())
          return (T) V.getAs<T>();
        V.AssertOnUnsupportedTypeCast();
        return T();
      }
    };

    template<typename T>
    struct CastFwd<T*> {
      static T* cast(const Value& V) {
        if (V.needsManagedAllocation() || V.hasPointerType())
          return (T*) (uintptr_t) V.getAs<void*>();
        V.AssertOnUnsupportedTypeCast();
        return nullptr;
      }
    };

    // Value(void* QualTypeAsOpaquePtr, Interpreter& Interp):
    //   m_Type(QualTypeAsOpaquePtr),
    //   m_Interpreter(&Interp) {
    // }

  public:
    /// \brief Default constructor, creates a value that IsInvalid().
    Value():
      m_NeedsManagedAlloc(false), m_Type(nullptr),
      m_Interpreter(nullptr) {}
    /// \brief Copy a value.
    Value(const Value& other);
    /// \brief Move a value.
    Value(Value&& other):
      m_Storage(other.m_Storage), m_NeedsManagedAlloc(other.m_NeedsManagedAlloc),
      m_Type(other.m_Type), m_Interpreter(other.m_Interpreter) {
      // Invalidate other so it will not release.
      other.m_NeedsManagedAlloc = false;
    }

    /// \brief Construct a valid but uninitialized Value. After this call the
    ///   value's storage can be accessed; i.e. calls ManagedAllocate() if
    ///   needed.
    Value(clang::QualType Ty, Interpreter& Interp);

    /// \brief Destruct the value; calls ManagedFree() if needed.
    ~Value();

    /// \brief Create a valid but ininitialized Value. After this call the
    ///   value's storage can be accessed; i.e. calls ManagedAllocate() if
    ///   needed.
    template <class T>
    static Value Create(void* QualTypeAsOpaquePtr, Interpreter& Interp) {
      EStorageType stType
        = std::is_reference<T>::value ?
       determineStorageTypeT((typename std::remove_reference<T>::type**)nullptr)
        : determineStorageTypeT((T*)nullptr);
      return Value(QualTypeAsOpaquePtr, Interp, stType);
    }

    Value& operator =(const Value& other);
    Value& operator =(Value&& other);

    clang::QualType getType() const;
    clang::ASTContext& getASTContext() const;
    Interpreter* getInterpreter() const { return m_Interpreter; }

    /// \brief Whether this type needs managed heap, i.e. the storage provided
    /// by the m_Storage member is insufficient, or a non-trivial destructor
    /// must be called.
    bool needsManagedAllocation() const {
      return m_NeedsManagedAlloc;
    }

    /// \brief Determine whether the Value has been set.
    //
    /// Determine whether the Value has been set by checking
    /// whether the type is valid.
    bool isValid() const;

    /// \brief Determine whether the Value is set but void.
    bool isVoid() const;

    /// \brief Determine whether the Value is set and not void.
    //
    /// Determine whether the Value is set and not void.
    /// Only in this case can getAs() or simplisticCastAs() be called.
    bool hasValue() const { return isValid() && !isVoid(); }

    /// \brief Get a reference to the value without type checking.
    /// T *must* correspond to type. Else use simplisticCastAs()!
    template <typename T> T getAs() const;

    // FIXME: If the cling::Value is destroyed and it handed out an address that
    // might be accessing invalid memory.
    void** getPtrAddress() { return &m_Storage.m_Ptr; }
    void* getPtr() const { return m_Storage.m_Ptr; }
    void setPtr(void* Val) { m_Storage.m_Ptr = Val; }

    // FIXME: Add AssertInvalid("##name")
#define X(type, name)                                    \
    type get##name() const {return m_Storage.m_##name;}  \
    void set##name(type Val) {m_Storage.m_##name = Val;} \

  BUILTIN_TYPES

#undef X

    /// \brief Get the value with cast.
    //
    /// Get the value cast to T. This is similar to reinterpret_cast<T>(value),
    /// casting the value of builtins (except void), enums and pointers.
    /// Values referencing an object are treated as pointers to the object.
    template <typename T>
    T simplisticCastAs() const {
      return CastFwd<T>::cast(*this);
    }

    ///\brief Generic interface to value printing.
    ///
    /// Can be re-implemented to print type-specific details, e.g. as
    ///\code
    ///   template <typename POSSIBLYDERIVED>
    ///   std::string printValue(const MyClass* const p, POSSIBLYDERIVED* ac,
    ///                          const Value& V);
    ///\endcode
    void print(llvm::raw_ostream& Out, bool escape = false) const;
    void dump(bool escape = true) const;
  };

  template <> void* Value::getAs() const;
#define X(type, name)                                                   \
  template <> type Value::getAs() const;                                \

  BUILTIN_TYPES

#undef X
} // end namespace cling

#endif // CLING_VALUE_H
