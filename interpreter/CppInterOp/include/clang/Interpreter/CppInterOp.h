//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vvasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CPPINTEROP_CPPINTEROP_H
#define CPPINTEROP_CPPINTEROP_H

#include <cassert>
#include <cstdint>
#include <set>
#include <string>
#include <vector>

// The cross-platform CPPINTEROP_API macro definition
#if defined _WIN32 || defined __CYGWIN__
#define CPPINTEROP_API __declspec(dllexport)
#else
#ifdef __GNUC__
#define CPPINTEROP_API __attribute__((__visibility__("default")))
#else
#define CPPINTEROP_API
#endif
#endif

namespace Cpp {
  using TCppIndex_t = size_t;
  using TCppScope_t = void*;
  using TCppType_t = void*;
  using TCppFunction_t = void*;
  using TCppConstFunction_t = const void*;
  using TCppFuncAddr_t = void*;
  using TInterp_t = void*;
  using TCppObject_t = void*;

  enum Operator {
    OP_None,
    OP_New,
    OP_Delete,
    OP_Array_New,
    OP_Array_Delete,
    OP_Plus,
    OP_Minus,
    OP_Star,
    OP_Slash,
    OP_Percent,
    OP_Caret,
    OP_Amp,
    OP_Pipe,
    OP_Tilde,
    OP_Exclaim,
    OP_Equal,
    OP_Less,
    OP_Greater,
    OP_PlusEqual,
    OP_MinusEqual,
    OP_StarEqual,
    OP_SlashEqual,
    OP_PercentEqual,
    OP_CaretEqual,
    OP_AmpEqual,
    OP_PipeEqual,
    OP_LessLess,
    OP_GreaterGreater,
    OP_LessLessEqual,
    OP_GreaterGreaterEqual,
    OP_EqualEqual,
    OP_ExclaimEqual,
    OP_LessEqual,
    OP_GreaterEqual,
    OP_Spaceship,
    OP_AmpAmp,
    OP_PipePipe,
    OP_PlusPlus,
    OP_MinusMinus,
    OP_Comma,
    OP_ArrowStar,
    OP_Arrow,
    OP_Call,
    OP_Subscript,
    OP_Conditional,
    OP_Coawait,
  };

  enum OperatorArity { kUnary = 1, kBinary, kBoth };

  /// A class modeling function calls for functions produced by the interpreter
  /// in compiled code. It provides an information if we are calling a standard
  /// function, constructor or destructor.
  class JitCall {
  public:
    friend CPPINTEROP_API JitCall
    MakeFunctionCallable(TInterp_t I, TCppConstFunction_t func);
    enum Kind : char {
      kUnknown = 0,
      kGenericCall,
      kDestructorCall,
    };
    struct ArgList {
      void** m_Args = nullptr;
      size_t m_ArgSize = 0;
      // Clang struggles with =default...
      ArgList() : m_Args(nullptr), m_ArgSize(0) {}
      ArgList(void** Args, size_t ArgSize)
        : m_Args(Args), m_ArgSize(ArgSize) {}
    };
    // FIXME: Figure out how to unify the wrapper signatures.
    // FIXME: Hide these implementation details by moving wrapper generation in
    // this class.
    using GenericCall = void (*)(void*, int, void**, void*);
    using DestructorCall = void (*)(void*, unsigned long, int);
  private:
    union {
      GenericCall m_GenericCall;
      DestructorCall m_DestructorCall;
    };
    const Kind m_Kind;
    TCppConstFunction_t m_FD;
    JitCall() : m_Kind(kUnknown), m_GenericCall(nullptr), m_FD(nullptr) {}
    JitCall(Kind K, GenericCall C, TCppConstFunction_t FD)
      : m_Kind(K), m_GenericCall(C), m_FD(FD) {}
    JitCall(Kind K, DestructorCall C, TCppConstFunction_t Dtor)
      : m_Kind(K), m_DestructorCall(C), m_FD(Dtor) {}

    /// Checks if the passed arguments are valid for the given function.
    bool AreArgumentsValid(void* result, ArgList args, void* self) const;

    /// This function is used for debugging, it reports when the function was
    /// called.
    void ReportInvokeStart(void* result, ArgList args, void* self) const;
    void ReportInvokeStart(void* object, unsigned long nary,
                           int withFree) const;
    void ReportInvokeEnd() const;
  public:
    Kind getKind() const { return m_Kind; }
    bool isValid() const { return getKind() != kUnknown; }
    bool isInvalid() const { return !isValid(); }
    explicit operator bool() const { return isValid(); }

    // Specialized for calling void functions.
    void Invoke(ArgList args = {}, void* self = nullptr) const {
      Invoke(/*result=*/nullptr, args, self);
    }

    /// Makes a call to a generic function or method.
    ///\param[in] result - the location where the return result will be placed.
    ///\param[in] args - a pointer to a argument list and argument size.
    ///\param[in] self - the 'this pointer' of the object.
    // FIXME: Adjust the arguments and their types: args_size can be unsigned;
    // self can go in the end and be nullptr by default; result can be a nullptr
    // by default. These changes should be synchronized with the wrapper if we
    // decide to directly.
    void Invoke(void* result, ArgList args = {}, void* self = nullptr) const {
      // Forward if we intended to call a dtor with only 1 parameter.
      if (m_Kind == kDestructorCall && result && !args.m_Args)
        return InvokeDestructor(result, /*nary=*/0UL, /*withFree=*/true);

#ifndef NDEBUG
      assert(AreArgumentsValid(result, args, self) && "Invalid args!");
      ReportInvokeStart(result, args, self);
#endif // NDEBUG
      m_GenericCall(self, args.m_ArgSize, args.m_Args, result);
    }
    /// Makes a call to a destructor.
    ///\param[in] object - the pointer of the object whose destructor we call.
    ///\param[in] nary - the count of the objects we destruct if we deal with an
    ///           array of objects.
    ///\param[in] withFree - true if we should call operator delete or false if
    ///           we should call only the destructor.
    //FIXME: Change the type of withFree from int to bool in the wrapper code.
    void InvokeDestructor(void* object, unsigned long nary = 0,
                          int withFree = true) const {
      assert(m_Kind == kDestructorCall && "Wrong overload!");
#ifndef NDEBUG
      ReportInvokeStart(object, nary, withFree);
#endif // NDEBUG
      m_DestructorCall(object, nary, withFree);
    }
  };

  ///\returns the version string information of the library.
  CPPINTEROP_API std::string GetVersion();

  ///\returns the demangled representation of the given mangled_name
  CPPINTEROP_API std::string Demangle(const std::string& mangled_name);

  /// Enables or disables the debugging printouts on stderr.
  /// Debugging output can be enabled also by the environment variable
  /// CPPINTEROP_EXTRA_INTERPRETER_ARGS. For example,
  /// CPPINTEROP_EXTRA_INTERPRETER_ARGS="-mllvm -debug-only=jitcall" to produce
  /// only debug output for jitcall events.
  CPPINTEROP_API void EnableDebugOutput(bool value = true);

  ///\returns true if the debugging printouts on stderr are enabled.
  CPPINTEROP_API bool IsDebugOutputEnabled();

  /// Checks if the given class represents an aggregate type).
  ///\returns true if \c scope is an array or a C++ tag (as per C++
  ///[dcl.init.aggr]) \returns true if the scope supports aggregate
  /// initialization.
  CPPINTEROP_API bool IsAggregate(TCppScope_t scope);

  /// Checks if the scope is a namespace or not.
  CPPINTEROP_API bool IsNamespace(TCppScope_t scope);

  /// Checks if the scope is a class or not.
  CPPINTEROP_API bool IsClass(TCppScope_t scope);

  /// Checks if the scope is a function.
  CPPINTEROP_API bool IsFunction(TCppScope_t scope);

  /// Checks if the type is a function pointer.
  CPPINTEROP_API bool IsFunctionPointerType(TCppType_t type);

  /// Checks if the klass polymorphic.
  /// which means that the class contains or inherits a virtual function
  CPPINTEROP_API bool IsClassPolymorphic(TCppScope_t klass);

  // See TClingClassInfo::IsLoaded
  /// Checks if the class definition is present, or not. Performs a
  /// template instantiation if necessary.
  CPPINTEROP_API bool IsComplete(TCppScope_t scope);

  CPPINTEROP_API size_t SizeOf(TCppScope_t scope);

  /// Checks if it is a "built-in" or a "complex" type.
  CPPINTEROP_API bool IsBuiltin(TCppType_t type);

  /// Checks if it is a templated class.
  CPPINTEROP_API bool IsTemplate(TCppScope_t handle);

  /// Checks if it is a class template specialization class.
  CPPINTEROP_API bool IsTemplateSpecialization(TCppScope_t handle);

  /// Checks if \c handle introduces a typedef name via \c typedef or \c using.
  CPPINTEROP_API bool IsTypedefed(TCppScope_t handle);

  CPPINTEROP_API bool IsAbstract(TCppType_t klass);

  /// Checks if it is an enum name (EnumDecl represents an enum name).
  CPPINTEROP_API bool IsEnumScope(TCppScope_t handle);

  /// Checks if it is an enum's value (EnumConstantDecl represents
  /// each enum constant that is defined).
  CPPINTEROP_API bool IsEnumConstant(TCppScope_t handle);

  /// Checks if the passed value is an enum type or not.
  CPPINTEROP_API bool IsEnumType(TCppType_t type);

  /// Extracts enum declarations from a specified scope and stores them in
  /// vector
  CPPINTEROP_API void GetEnums(TCppScope_t scope,
                               std::vector<std::string>& Result);

  /// We assume that smart pointer types define both operator* and
  /// operator->.
  CPPINTEROP_API bool IsSmartPtrType(TCppType_t type);

  /// For the given "class", get the integer type that the enum
  /// represents, so that you can store it properly in your specific
  /// language.
  CPPINTEROP_API TCppType_t GetIntegerTypeFromEnumScope(TCppScope_t handle);

  /// For the given "type", this function gets the integer type that the enum
  /// represents, so that you can store it properly in your specific
  /// language.
  CPPINTEROP_API TCppType_t GetIntegerTypeFromEnumType(TCppType_t handle);

  /// Gets a list of all the enum constants for an enum.
  CPPINTEROP_API std::vector<TCppScope_t> GetEnumConstants(TCppScope_t scope);

  /// Gets the enum name when an enum constant is passed.
  CPPINTEROP_API TCppType_t GetEnumConstantType(TCppScope_t scope);

  /// Gets the index value (0,1,2, etcetera) of the enum constant
  /// that was passed into this function.
  CPPINTEROP_API TCppIndex_t GetEnumConstantValue(TCppScope_t scope);

  /// Gets the size of the "type" that is passed in to this function.
  CPPINTEROP_API size_t GetSizeOfType(TCppType_t type);

  /// Checks if the passed value is a variable.
  CPPINTEROP_API bool IsVariable(TCppScope_t scope);

  /// Gets the name of any named decl (a class,
  /// namespace, variable, or a function).
  CPPINTEROP_API std::string GetName(TCppScope_t klass);

  /// This is similar to GetName() function, but besides
  /// the name, it also gets the template arguments.
  CPPINTEROP_API std::string GetCompleteName(TCppScope_t klass);

  /// Gets the "qualified" name (including the namespace) of any
  /// named decl (a class, namespace, variable, or a function).
  CPPINTEROP_API std::string GetQualifiedName(TCppScope_t klass);

  /// This is similar to GetQualifiedName() function, but besides
  /// the "qualified" name (including the namespace), it also
  /// gets the template arguments.
  CPPINTEROP_API std::string GetQualifiedCompleteName(TCppScope_t klass);

  /// Gets the list of namespaces utilized in the supplied scope.
  CPPINTEROP_API std::vector<TCppScope_t> GetUsingNamespaces(TCppScope_t scope);

  /// Gets the global scope of the whole C++  instance.
  CPPINTEROP_API TCppScope_t GetGlobalScope();

  /// Strips the typedef and returns the underlying class, and if the
  /// underlying decl is not a class it returns the input unchanged.
  CPPINTEROP_API TCppScope_t GetUnderlyingScope(TCppScope_t scope);

  /// Gets the namespace or class (by stripping typedefs) for the name 
  /// passed as a parameter, and if the parent is not passed, 
  /// then global scope will be assumed.
  CPPINTEROP_API TCppScope_t GetScope(const std::string& name,
                                      TCppScope_t parent = nullptr);

  /// When the namespace is known, then the parent doesn't need
  /// to be specified. This will probably be phased-out in
  /// future versions of the interop library.
  CPPINTEROP_API TCppScope_t GetScopeFromCompleteName(const std::string& name);

  /// This function performs a lookup within the specified parent,
  /// a specific named entity (functions, enums, etcetera).
  CPPINTEROP_API TCppScope_t GetNamed(const std::string& name,
                                      TCppScope_t parent = nullptr);

  /// Gets the parent of the scope that is passed as a parameter.
  CPPINTEROP_API TCppScope_t GetParentScope(TCppScope_t scope);

  /// Gets the scope of the type that is passed as a parameter.
  CPPINTEROP_API TCppScope_t GetScopeFromType(TCppType_t type);

  /// Gets the number of Base Classes for the Derived Class that
  /// is passed as a parameter.
  CPPINTEROP_API TCppIndex_t GetNumBases(TCppScope_t klass);

  /// Gets a specific Base Class using its index. Typically GetNumBases()
  /// is used to get the number of Base Classes, and then that number
  /// can be used to iterate through the index value to get each specific
  /// base class.
  CPPINTEROP_API TCppScope_t GetBaseClass(TCppScope_t klass, TCppIndex_t ibase);

  /// Checks if the supplied Derived Class is a sub-class of the
  /// provided Base Class.
  CPPINTEROP_API bool IsSubclass(TCppScope_t derived, TCppScope_t base);

  /// Each base has its own offset in a Derived Class. This offset can be
  /// used to get to the Base Class fields.
  CPPINTEROP_API int64_t GetBaseClassOffset(TCppScope_t derived,
                                            TCppScope_t base);

  /// Sets a list of all the Methods that are in the Class that is
  /// supplied as a parameter.
  ///\param[in] klass - Pointer to the scope/class under which the methods have
  ///           to be retrieved
  ///\param[out] methods - Vector of methods in the class
  CPPINTEROP_API void GetClassMethods(TCppScope_t klass,
                                      std::vector<TCppFunction_t>& methods);

  /// Template function pointer list to add proxies for un-instantiated/
  /// non-overloaded templated methods
  ///\param[in] klass - Pointer to the scope/class under which the methods have
  ///           to be retrieved
  ///\param[out] methods - Vector of methods in the class
  CPPINTEROP_API void
  GetFunctionTemplatedDecls(TCppScope_t klass,
                            std::vector<TCppFunction_t>& methods);

  ///\returns if a class has a default constructor.
  CPPINTEROP_API bool HasDefaultConstructor(TCppScope_t scope);

  ///\returns the default constructor of a class, if any.
  CPPINTEROP_API TCppFunction_t GetDefaultConstructor(TCppScope_t scope);

  ///\returns the class destructor, if any.
  CPPINTEROP_API TCppFunction_t GetDestructor(TCppScope_t scope);

  /// Looks up all the functions that have the name that is
  /// passed as a parameter in this function.
  CPPINTEROP_API std::vector<TCppFunction_t>
  GetFunctionsUsingName(TCppScope_t scope, const std::string& name);

  /// Gets the return type of the provided function.
  CPPINTEROP_API TCppType_t GetFunctionReturnType(TCppFunction_t func);

  /// Gets the number of Arguments for the provided function.
  CPPINTEROP_API TCppIndex_t GetFunctionNumArgs(TCppFunction_t func);

  /// Gets the number of Required Arguments for the provided function.
  CPPINTEROP_API TCppIndex_t GetFunctionRequiredArgs(TCppConstFunction_t func);

  /// For each Argument of a function, you can get the Argument Type
  /// by providing the Argument Index, based on the number of arguments
  /// from the GetFunctionNumArgs() function.
  CPPINTEROP_API TCppType_t GetFunctionArgType(TCppFunction_t func,
                                               TCppIndex_t iarg);

  ///\returns a stringified version of a given function signature in the form:
  /// void N::f(int i, double d, long l = 0, char ch = 'a').
  CPPINTEROP_API std::string GetFunctionSignature(TCppFunction_t func);

  ///\returns if a function was marked as \c =delete.
  CPPINTEROP_API bool IsFunctionDeleted(TCppConstFunction_t function);

  CPPINTEROP_API bool IsTemplatedFunction(TCppFunction_t func);

  /// This function performs a lookup to check if there is a
  /// templated function of that type.
  CPPINTEROP_API bool ExistsFunctionTemplate(const std::string& name,
                                             TCppScope_t parent = nullptr);

  /// Sets a list of all the Templated Methods that are in the Class that is
  /// supplied as a parameter.
  ///\param[in] name - method name
  ///\param[in] parent - Pointer to the scope/class under which the methods have
  ///           to be retrieved
  ///\param[out] funcs - vector of function pointers matching the name
  CPPINTEROP_API void
  GetClassTemplatedMethods(const std::string& name, TCppScope_t parent,
                           std::vector<TCppFunction_t>& funcs);

  /// Checks if the provided parameter is a method.
  CPPINTEROP_API bool IsMethod(TCppConstFunction_t method);

  /// Checks if the provided parameter is a 'Public' method.
  CPPINTEROP_API bool IsPublicMethod(TCppFunction_t method);

  /// Checks if the provided parameter is a 'Protected' method.
  CPPINTEROP_API bool IsProtectedMethod(TCppFunction_t method);

  /// Checks if the provided parameter is a 'Private' method.
  CPPINTEROP_API bool IsPrivateMethod(TCppFunction_t method);

  /// Checks if the provided parameter is a Constructor.
  CPPINTEROP_API bool IsConstructor(TCppConstFunction_t method);

  /// Checks if the provided parameter is a Destructor.
  CPPINTEROP_API bool IsDestructor(TCppConstFunction_t method);

  /// Checks if the provided parameter is a 'Static' method.
  CPPINTEROP_API bool IsStaticMethod(TCppConstFunction_t method);

  ///\returns the address of the function given its potentially mangled name.
  CPPINTEROP_API TCppFuncAddr_t GetFunctionAddress(const char* mangled_name);

  ///\returns the address of the function given its function declaration.
  CPPINTEROP_API TCppFuncAddr_t GetFunctionAddress(TCppFunction_t method);

  /// Checks if the provided parameter is a 'Virtual' method.
  CPPINTEROP_API bool IsVirtualMethod(TCppFunction_t method);

  /// Gets all the Fields/Data Members of a Class
  CPPINTEROP_API void GetDatamembers(TCppScope_t scope,
                                     std::vector<TCppScope_t>& datamembers);

  /// Gets all the Static Fields/Data Members of a Class
  ///\param[in] scope - class
  ///\param[out] funcs - vector of static data members
  CPPINTEROP_API void
  GetStaticDatamembers(TCppScope_t scope,
                       std::vector<TCppScope_t>& datamembers);

  /// Gets all the Enum Constants declared in a Class
  ///\param[in] scope - class
  ///\param[out] funcs - vector of static data members
  ///\param[in] include_enum_class - include enum constants from enum class
  CPPINTEROP_API
  void GetEnumConstantDatamembers(TCppScope_t scope,
                                  std::vector<TCppScope_t>& datamembers,
                                  bool include_enum_class = true);

  /// This is a Lookup function to be used specifically for data members.
  CPPINTEROP_API TCppScope_t LookupDatamember(const std::string& name,
                                              TCppScope_t parent);

  /// Gets the type of the variable that is passed as a parameter.
  CPPINTEROP_API TCppType_t GetVariableType(TCppScope_t var);

  /// Gets the address of the variable, you can use it to get the
  /// value stored in the variable.
  CPPINTEROP_API intptr_t GetVariableOffset(TCppScope_t var,
                                            TCppScope_t parent = nullptr);

  /// Checks if the provided variable is a 'Public' variable.
  CPPINTEROP_API bool IsPublicVariable(TCppScope_t var);

  /// Checks if the provided variable is a 'Protected' variable.
  CPPINTEROP_API bool IsProtectedVariable(TCppScope_t var);

  /// Checks if the provided variable is a 'Private' variable.
  CPPINTEROP_API bool IsPrivateVariable(TCppScope_t var);

  /// Checks if the provided variable is a 'Static' variable.
  CPPINTEROP_API bool IsStaticVariable(TCppScope_t var);

  /// Checks if the provided variable is a 'Constant' variable.
  CPPINTEROP_API bool IsConstVariable(TCppScope_t var);

  /// Checks if the provided parameter is a Record (struct).
  CPPINTEROP_API bool IsRecordType(TCppType_t type);

  /// Checks if the provided parameter is a Plain Old Data Type (POD).
  CPPINTEROP_API bool IsPODType(TCppType_t type);

  /// Checks if type is a pointer
  CPPINTEROP_API bool IsPointerType(TCppType_t type);

  /// Get the underlying pointee type
  CPPINTEROP_API TCppType_t GetPointeeType(TCppType_t type);

  /// Checks if type is a reference
  CPPINTEROP_API bool IsReferenceType(TCppType_t type);

  /// Get the type that the reference refers to
  CPPINTEROP_API TCppType_t GetNonReferenceType(TCppType_t type);

  /// Gets the pure, Underlying Type (as opposed to the Using Type).
  CPPINTEROP_API TCppType_t GetUnderlyingType(TCppType_t type);

  /// Gets the Type (passed as a parameter) as a String value.
  CPPINTEROP_API std::string GetTypeAsString(TCppType_t type);

  /// Gets the Canonical Type string from the std string. A canonical type
  /// is the type with any typedef names, syntactic sugars or modifiers stripped
  /// out of it.
  CPPINTEROP_API TCppType_t GetCanonicalType(TCppType_t type);

  /// Used to either get the built-in type of the provided string, or
  /// use the name to lookup the actual type.
  CPPINTEROP_API TCppType_t GetType(const std::string& type);

  ///\returns the complex of the provided type.
  CPPINTEROP_API TCppType_t GetComplexType(TCppType_t element_type);

  /// This will convert a class into its type, so for example, you can
  /// use it to declare variables in it.
  CPPINTEROP_API TCppType_t GetTypeFromScope(TCppScope_t klass);

  /// Checks if a C++ type derives from another.
  CPPINTEROP_API bool IsTypeDerivedFrom(TCppType_t derived, TCppType_t base);

  /// Creates a trampoline function by using the interpreter and returns a
  /// uniform interface to call it from compiled code.
  CPPINTEROP_API JitCall MakeFunctionCallable(TCppConstFunction_t func);

  CPPINTEROP_API JitCall MakeFunctionCallable(TInterp_t I,
                                              TCppConstFunction_t func);

  /// Checks if a function declared is of const type or not.
  CPPINTEROP_API bool IsConstMethod(TCppFunction_t method);

  ///\returns the default argument value as string.
  CPPINTEROP_API std::string GetFunctionArgDefault(TCppFunction_t func,
                                                   TCppIndex_t param_index);

  ///\returns the argument name of function as string.
  CPPINTEROP_API std::string GetFunctionArgName(TCppFunction_t func,
                                                TCppIndex_t param_index);

  ///\returns arity of the operator or kNone
  OperatorArity GetOperatorArity(TCppFunction_t op);

  ///\returns list of operator overloads
  void GetOperator(TCppScope_t scope, Operator op,
                   std::vector<TCppFunction_t>& operators,
                   OperatorArity kind = kBoth);

  /// Creates an instance of the interpreter we need for the various interop
  /// services.
  ///\param[in] Args - the list of arguments for interpreter constructor.
  ///\param[in] CPPINTEROP_EXTRA_INTERPRETER_ARGS - an env variable, if defined,
  ///           adds additional arguments to the interpreter.
  CPPINTEROP_API TInterp_t
  CreateInterpreter(const std::vector<const char*>& Args = {},
                    const std::vector<const char*>& GpuArgs = {});

  /// Checks which Interpreter backend was CppInterOp library built with (Cling,
  /// Clang-REPL, etcetera). In practice, the selected interpreter should not
  /// matter, since the library will function in the same way.
  ///\returns the current interpreter instance, if any.
  CPPINTEROP_API TInterp_t GetInterpreter();

  /// Sets the Interpreter instance with an external interpreter, meant to
  /// be called by an external library that manages it's own interpreter.
  /// Sets a flag signifying CppInterOp does not have ownership of the
  /// sInterpreter.
  ///\param[in] Args - the pointer to an external interpreter
  CPPINTEROP_API void UseExternalInterpreter(TInterp_t I);

  /// Adds a Search Path for the Interpreter to get the libraries.
  CPPINTEROP_API void AddSearchPath(const char* dir, bool isUser = true,
                                    bool prepend = false);

  /// Returns the resource-dir path (for headers).
  CPPINTEROP_API const char* GetResourceDir();

  /// Uses the underlying clang compiler to detect the resource directory.
  /// In essence calling clang -print-resource-dir and checks if it ends with
  /// a compatible to CppInterOp version.
  ///\param[in] ClangBinaryName - the name (or the full path) of the compiler
  ///                             to ask.
  CPPINTEROP_API std::string
  DetectResourceDir(const char* ClangBinaryName = "clang");

  /// Asks the system compiler for its default include paths.
  ///\param[out] Paths - the list of include paths returned by eg.
  ///                     `c++ -xc++ -E -v /dev/null 2>&1`
  ///\param[in] CompilerName - the name (or the full path) of the compiler
  ///                          binary file.
  CPPINTEROP_API void
  DetectSystemCompilerIncludePaths(std::vector<std::string>& Paths,
                                   const char* CompilerName = "c++");

  /// Secondary search path for headers, if not found using the
  /// GetResourceDir() function.
  CPPINTEROP_API void AddIncludePath(const char* dir);

  // Gets the currently used include paths
  ///\param[out] IncludePaths - the list of include paths
  ///
  CPPINTEROP_API void GetIncludePaths(std::vector<std::string>& IncludePaths,
                                      bool withSystem = false,
                                      bool withFlags = false);

  /// Only Declares a code snippet in \c code and does not execute it.
  ///\returns 0 on success
  CPPINTEROP_API int Declare(const char* code, bool silent = false);

  /// Declares and executes a code snippet in \c code.
  ///\returns 0 on success
  CPPINTEROP_API int Process(const char* code);

  /// Declares, executes and returns the execution result as a intptr_t.
  ///\returns the expression results as a intptr_t.
  CPPINTEROP_API intptr_t Evaluate(const char* code, bool* HadError = nullptr);

  /// Looks up the library if access is enabled.
  ///\returns the path to the library.
  CPPINTEROP_API std::string LookupLibrary(const char* lib_name);

  /// Finds \c lib_stem considering the list of search paths and loads it by
  /// calling dlopen.
  /// \returns true on success.
  CPPINTEROP_API bool LoadLibrary(const char* lib_stem, bool lookup = true);

  /// Finds \c lib_stem considering the list of search paths and unloads it by
  /// calling dlclose.
  /// function.
  CPPINTEROP_API void UnloadLibrary(const char* lib_stem);

  /// Scans all libraries on the library search path for a given potentially
  /// mangled symbol name.
  ///\returns the path to the first library that contains the symbol definition.
  CPPINTEROP_API std::string
  SearchLibrariesForSymbol(const char* mangled_name,
                           bool search_system /*true*/);

  /// Inserts or replaces a symbol in the JIT with the one provided. This is
  /// useful for providing our own implementations of facilities such as printf.
  ///
  ///\param[in] linker_mangled_name - the name of the symbol to be inserted or
  ///           replaced.
  ///\param[in] address - the new address of the symbol.
  ///
  ///\returns true on failure.
  CPPINTEROP_API bool InsertOrReplaceJitSymbol(const char* linker_mangled_name,
                                               uint64_t address);

  /// Tries to load provided objects in a string format (prettyprint).
  CPPINTEROP_API std::string ObjToString(const char* type, void* obj);

  struct TemplateArgInfo {
    TCppType_t m_Type;
    const char* m_IntegralValue;
    TemplateArgInfo(TCppScope_t type, const char* integral_value = nullptr)
      : m_Type(type), m_IntegralValue(integral_value) {}
  };
  /// Builds a template instantiation for a given templated declaration.
  /// Offers a single interface for instantiation of class, function and
  /// variable templates
  ///
  ///\param[in] tmpl - Uninstantiated template class/function
  ///\param[in] template_args - Pointer to vector of template arguments stored
  ///           in the \c TemplateArgInfo struct
  ///\param[in] template_args_size - Size of the vector of template arguments
  ///           passed as \c template_args
  ///
  ///\returns Instantiated templated class/function/variable pointer
  CPPINTEROP_API TCppScope_t
  InstantiateTemplate(TCppScope_t tmpl, const TemplateArgInfo* template_args,
                      size_t template_args_size);

  /// Sets the class template instantiation arguments of \c templ_instance.
  ///
  ///\param[in] templ_instance - Pointer to the template instance
  ///\param[out] args - Vector of instantiation arguments
  CPPINTEROP_API void
  GetClassTemplateInstantiationArgs(TCppScope_t templ_instance,
                                    std::vector<TemplateArgInfo>& args);

  /// Instantiates a function template from a given string representation. This
  /// function also does overload resolution.
  ///\returns the instantiated function template declaration.
  CPPINTEROP_API TCppFunction_t
  InstantiateTemplateFunctionFromString(const char* function_template);

  /// Finds best overload match based on explicit template parameters (if any)
  /// and argument types.
  ///
  ///\param[in] candidates - vector of overloads that come under the
  ///           parent scope and have the same name
  ///\param[in] explicit_types - set of expicitly instantiated template types
  ///\param[in] arg_types - set of argument types
  ///\returns Instantiated function pointer
  CPPINTEROP_API TCppFunction_t
  BestOverloadFunctionMatch(const std::vector<TCppFunction_t>& candidates,
                            const std::vector<TemplateArgInfo>& explicit_types,
                            const std::vector<TemplateArgInfo>& arg_types);

  CPPINTEROP_API void GetAllCppNames(TCppScope_t scope,
                                     std::set<std::string>& names);

  CPPINTEROP_API void DumpScope(TCppScope_t scope);

  namespace DimensionValue {
    enum : long int {
      UNKNOWN_SIZE = -1,
    };
  }

  /// Gets the size/dimensions of a multi-dimension array.
  CPPINTEROP_API std::vector<long int> GetDimensions(TCppType_t type);

  /// Allocates memory for a given class.
  CPPINTEROP_API TCppObject_t Allocate(TCppScope_t scope);

  /// Deallocates memory for a given class.
  CPPINTEROP_API void Deallocate(TCppScope_t scope, TCppObject_t address);

  /// Creates an object of class \c scope and calls its default constructor. If
  /// \c arena is set it uses placement new.
  CPPINTEROP_API TCppObject_t Construct(TCppScope_t scope,
                                        void* arena = nullptr);

  /// Calls the destructor of object of type \c type. When withFree is true it
  /// calls operator delete/free.
  CPPINTEROP_API void Destruct(TCppObject_t This, TCppScope_t type,
                               bool withFree = true);

  /// @name Stream Redirection
  ///
  ///@{

  enum CaptureStreamKind : char {
    kStdOut = 1, ///< stdout
    kStdErr,     ///< stderr
    // kStdBoth,    ///< stdout and stderr
    // kSTDSTRM  // "&1" or "&2" is not a filename
  };

  /// Begins recording the given standard stream.
  ///\param[fd_kind] - The stream to be captured
  CPPINTEROP_API void BeginStdStreamCapture(CaptureStreamKind fd_kind);

  /// Ends recording the standard stream and returns the result as a string.
  CPPINTEROP_API std::string EndStdStreamCapture();

  ///@}

  /// Append all Code completion suggestions to Results.
  ///\param[out] Results - CC suggestions for code fragment. Suggestions are
  /// appended.
  ///\param[in] code - code fragmet to complete
  ///\param[in] complete_line - position (line) in code for suggestion
  ///\param[in] complete_column - position (column) in code for suggestion
  CPPINTEROP_API void CodeComplete(std::vector<std::string>& Results,
                                   const char* code,
                                   unsigned complete_line = 1U,
                                   unsigned complete_column = 1U);

} // end namespace Cpp

#endif // CPPINTEROP_CPPINTEROP_H
