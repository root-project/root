//===- CppInterOp.td - CppInterOp API definitions --------------*- td -*-===//
//
// Part of the compiler-research project, under the Apache License v2.0 with
// LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// TableGen definitions for the CppInterOp public API. Each def corresponds
// to one function in include/CppInterOp/CppInterOp.h and drives generation
// of the dispatch table, inline wrappers, and (future) C API wrappers.
//
//===----------------------------------------------------------------------===//

include "CppInterOpAPI.td"

// --- Interpreter management ---

def CreateInterpreter : CppInterOpAPI {
  let Doc = [{Creates an owned instance of the interpreter we need for the various interop
services and pushes it onto a stack.
\param[in] Args - the list of arguments for interpreter constructor.
\param[in] CPPINTEROP_EXTRA_INTERPRETER_ARGS - an env variable, if defined,
          adds additional arguments to the interpreter.
\returns nullptr on failure.}];

  let ReturnType = "InterpRef";
  let Args = [
    Arg<"const std::vector<const char*>&", "Args", "{}">,
    Arg<"const std::vector<const char*>&", "GpuArgs", "{}">
  ];
}

def GetInterpreter : CppInterOpAPI {
  let Doc = [{Checks which Interpreter backend was CppInterOp library built with (Cling,
Clang-REPL, etcetera). In practice, the selected interpreter should not
matter, since the library will function in the same way.
\returns the current interpreter instance, if any.}];

  let ReturnType = "InterpRef";
}

def DeleteInterpreter : CppInterOpAPI {
  let Doc = [{Deletes an instance of an interpreter.
\param[in] I - the interpreter to be deleted, if nullptr, deletes the last.
\returns false on failure or if \c I is not tracked in the stack.}];

  let ReturnType = "bool";
  let Args = [Arg<"InterpRef", "I", "nullptr">];
}

def Process : CppInterOpAPI {
  let Doc = [{Declares and executes a code snippet in \c code.
\returns 0 on success}];

  let ReturnType = "int";
  let Args = [Arg<"const char*", "code">];
}

def Declare : CppInterOpAPI {
  let Doc = [{Only Declares a code snippet in \c code and does not execute it.
\returns 0 on success}];

  let ReturnType = "int";
  let Args = [
    Arg<"const char*", "code">,
    Arg<"bool", "silent", "false">
  ];
}

def Evaluate : CppInterOpAPI {
  let Doc = [{Declares, executes and returns the execution result as a typed
\c Cpp::Box carrying the Kind tag and the source QualType (when available).
\c getKind() returns \c K_Unspecified on parse error or
no-value-after-success.
\param[in] code - the snippet to declare and execute.
\returns a \c Cpp::Box. Use \c unbox<T>() to extract a fundamental
         (assert-checked: \c T must match the runtime Kind) or
         \c getObjectPtr() for object payloads. For unknown-Kind extraction
         dispatch on \c getKind() / \c KindOf<T>() first, or use \c visit().}];
  // Box has non-trivial copy/dtor and cannot cross the C ABI; suppress
  // the auto-generated C wrapper.
  let NoCWrapper = true;
  let ReturnType = "Box";
  let Args = [
    Arg<"const char*", "code">
  ];
}

// NOTE: The legacy C-ABI overload `intptr_t Evaluate(const char*, bool*)`
// and its matching C wrapper `cppinterop_Evaluate` live hand-written in
// CXCppInterOp.cpp / CppInterOp.h (look for "C-ABI"). They are not
// declared here because tablegen does not need to know about them.

// --- Scope / type queries ---

def GetScope : CppInterOpAPI {
  let Doc = [{Gets the namespace or class (by stripping typedefs) for the name
passed as a parameter, and if the parent is not passed,
then global scope will be assumed.}];

  let ReturnType = "DeclRef";
  let Args = [
    Arg<"const std::string&", "name">,
    Arg<"ConstDeclRef", "parent", "nullptr">
  ];
}

def GetNamed : CppInterOpAPI {
  let Doc = [{This function performs a lookup within the specified parent,
a specific named entity (functions, enums, etcetera).}];

  let ReturnType = "DeclRef";
  let Args = [
    Arg<"const std::string&", "name">,
    Arg<"ConstDeclRef", "parent", "nullptr">
  ];
}

def GetGlobalScope : CppInterOpAPI {
  let Doc = "Gets the global scope of the whole C++  instance.";

  let ReturnType = "DeclRef";
}

def IsNamespace : CppInterOpAPI {
  let Doc = "Checks if the scope is a namespace or not.";

  let ReturnType = "bool";
  let Args = [Arg<"ConstDeclRef", "DRef">];
}

def IsClass : CppInterOpAPI {
  let Doc = "Checks if the scope is a class or not.";

  let ReturnType = "bool";
  let Args = [Arg<"ConstDeclRef", "DRef">];
}

def GetType : CppInterOpAPI {
  let Doc = [{Used to either get the built-in type of the provided string, or
use the name to lookup the actual type. When `parent` is non-null, scoped
lookup honours using-directives and type aliases declared inside it; passing
`nullptr` (the default) limits lookup to TU scope.}];

  let ReturnType = "TypeRef";
  let Args = [Arg<"const std::string&", "name">,
              Arg<"ConstDeclRef", "parent", "nullptr">];
}

def GetTypeFromScope : CppInterOpAPI {
  let Doc = [{This will convert a class into its type, so for example, you can
use it to declare variables in it.}];

  let ReturnType = "TypeRef";
  let Args = [Arg<"ConstDeclRef", "DRef">];
}

def GetScopeFromType : CppInterOpAPI {
  let Doc = "Gets the scope of the type that is passed as a parameter.";

  let ReturnType = "DeclRef";
  let Args = [Arg<"ConstTypeRef", "TyRef">];
}

def SizeOf : CppInterOpAPI {
  let ReturnType = "size_t";
  let Args = [Arg<"ConstDeclRef", "DRef">];
}

// --- Functions with overloads ---

def GetFunctionAddress_str : CppInterOpAPI {
  let Doc = "\\returns the address of the function given its potentially mangled name.";
  let CppName = "GetFunctionAddress";
  let ReturnType = "void*";
  let Args = [Arg<"const char*", "mangled_name">];
}

def GetFunctionAddress_fn : CppInterOpAPI {
  let Doc = "\\returns the address of the function given its function declaration.";
  let CppName = "GetFunctionAddress";
  let ReturnType = "void*";
  let Args = [Arg<"FuncRef", "method">];
}

// --- Construction / destruction ---

def Construct : CppInterOpAPI {
  let Doc = [{Creates one or more objects of class \c scope by calling its default
constructor.
\param[in] scope Class to construct, or DRef to Constructor
\param[in] arena If set, this API uses placement new to construct at this
address.
\param[in] is used to indicate the number of objects to construct.
\returns a pointer to the constructed object, which is arena if placement
new is used.}];

  let ReturnType = "ObjectRef";
  let Args = [
    Arg<"DeclRef", "DRef">,
    Arg<"void*", "arena", "nullptr">,
    Arg<"size_t", "count", "1UL">
  ];
}

def Destruct : CppInterOpAPI {
  let Doc = [{Destroys one or more objects of a class
\param[in] This this pointer of the object to destruct. Can also be the
starting address of an array of objects
\param[in] scope Class to destruct
\param[in] withFree if true, we call operator delete/free, else just the
destructor
\param[in] count indicate the number of objects to destruct, if \c This
points to an array of objects
\returns true if wrapper generation and invocation succeeded.}];

  let ReturnType = "bool";
  let Args = [
    Arg<"ObjectRef", "This">,
    Arg<"DeclRef", "DRef">,
    Arg<"bool", "withFree", "true">,
    Arg<"size_t", "count", "0UL">
  ];
}

// --- Auto-generated remaining dispatch entries ---

def GetResourceDir : CppInterOpAPI {
  let Doc = "Returns the resource-dir path (for headers).";

  let ReturnType = "const char*";
}

def AddIncludePath : CppInterOpAPI {
  let Doc = [{Secondary search path for headers, if not found using the
GetResourceDir() function.}];

  let ReturnType = "void";
  let Args = [
    Arg<"const char*", "dir">
  ];
}

def LoadLibrary : CppInterOpAPI {
  let Doc = [{Finds \c lib_stem considering the list of search paths and loads it by
calling dlopen.
\returns true on success.}];

  let ReturnType = "bool";
  let Args = [
    Arg<"const char*", "lib_stem">,
    Arg<"bool", "lookup", "true">
  ];
}

def ObjToString : CppInterOpAPI {
  let Doc = "Tries to load provided objects in a string format (prettyprint).";

  let ReturnType = "std::string";
  let Args = [
    Arg<"const char*", "TyRef">,
    Arg<"void*", "obj">
  ];
}

def GetValueKind : CppInterOpAPI {
  let Doc = "Get if lvalue or rvalue reference";

  let ReturnType = "ValueKind";
  let Args = [
    Arg<"ConstTypeRef", "TyRef">
  ];
}

def GetNonReferenceType : CppInterOpAPI {
  let Doc = "Get the type that the reference refers to";

  let ReturnType = "TypeRef";
  let Args = [
    Arg<"ConstTypeRef", "TyRef">
  ];
}

def IsEnumType : CppInterOpAPI {
  let Doc = "Checks if the passed value is an enum type or not.";

  let ReturnType = "bool";
  let Args = [
    Arg<"ConstTypeRef", "TyRef">
  ];
}

def GetLanguage : CppInterOpAPI {
  let Doc = "Returns the programming language of the interpreter.";

  let ReturnType = "InterpreterLanguage";
  let Args = [
    Arg<"InterpRef", "I", "nullptr">
  ];
}

def GetLanguageStandard : CppInterOpAPI {
  let Doc = "Returns the language standard of the interpreter.";

  let ReturnType = "InterpreterLanguageStandard";
  let Args = [
    Arg<"InterpRef", "I", "nullptr">
  ];
}

def GetReferencedType : CppInterOpAPI {
  let Doc = "Get lvalue referenced type, or rvalue if rvalue is true";

  let ReturnType = "TypeRef";
  let Args = [
    Arg<"ConstTypeRef", "TyRef">,
    Arg<"bool", "rvalue", "false">
  ];
}

def IsPointerType : CppInterOpAPI {
  let Doc = "Checks if type is a pointer";

  let ReturnType = "bool";
  let Args = [
    Arg<"ConstTypeRef", "TyRef">
  ];
}

def GetPointeeType : CppInterOpAPI {
  let Doc = "Get the underlying pointee type";

  let ReturnType = "TypeRef";
  let Args = [
    Arg<"ConstTypeRef", "TyRef">
  ];
}

def GetPointerType : CppInterOpAPI {
  let Doc = "Get the pointer to type";

  let ReturnType = "TypeRef";
  let Args = [
    Arg<"ConstTypeRef", "TyRef">
  ];
}

def IsReferenceType : CppInterOpAPI {
  let Doc = "Checks if type is a reference";

  let ReturnType = "bool";
  let Args = [
    Arg<"ConstTypeRef", "TyRef">
  ];
}

def GetTypeAsString : CppInterOpAPI {
  let Doc = "Gets the Type (passed as a parameter) as a String value.";

  let ReturnType = "std::string";
  let Args = [
    Arg<"ConstTypeRef", "TyRef">
  ];
}

def GetCanonicalType : CppInterOpAPI {
  let Doc = [{Gets the Canonical Type string from the std string. A canonical type
is the type with any typedef names, syntactic sugars or modifiers stripped
out of it.}];

  let ReturnType = "TypeRef";
  let Args = [
    Arg<"ConstTypeRef", "TyRef">
  ];
}

def HasTypeQualifier : CppInterOpAPI {
  let Doc = [{Checks if the passed type has qual Qualifiers
qual can be ORed value of enum QualKind}];

  let ReturnType = "bool";
  let Args = [
    Arg<"ConstTypeRef", "TyRef">,
    Arg<"QualKind", "qual">
  ];
}

def RemoveTypeQualifier : CppInterOpAPI {
  let Doc = [{Returns type with the qual Qualifiers removed
qual can be ORed value of enum QualKind}];

  let ReturnType = "TypeRef";
  let Args = [
    Arg<"ConstTypeRef", "TyRef">,
    Arg<"QualKind", "qual">
  ];
}

def GetUnderlyingType : CppInterOpAPI {
  let Doc = "Gets the pure, Underlying Type (as opposed to the Using Type).";

  let ReturnType = "TypeRef";
  let Args = [
    Arg<"ConstTypeRef", "TyRef">
  ];
}

def IsRecordType : CppInterOpAPI {
  let Doc = "Checks if the provided parameter is a Record (struct).";

  let ReturnType = "bool";
  let Args = [
    Arg<"ConstTypeRef", "TyRef">
  ];
}

def GetVariableType : CppInterOpAPI {
  let Doc = "Gets the type of the variable that is passed as a parameter.";

  let ReturnType = "TypeRef";
  let Args = [
    Arg<"ConstDeclRef", "var">
  ];
}

def GetComplexType : CppInterOpAPI {
  let Doc = "\\returns the complex of the provided type.";

  let ReturnType = "TypeRef";
  let Args = [
    Arg<"ConstTypeRef", "element_type">
  ];
}

def GetUnderlyingScope : CppInterOpAPI {
  let Doc = [{Strips the typedef and returns the underlying class, and if the
underlying decl is not a class it returns the input unchanged.}];

  let ReturnType = "DeclRef";
  let Args = [
    Arg<"ConstDeclRef", "DRef">
  ];
}

def InstantiateTemplate : CppInterOpAPI {
  let Doc = [{Builds a template instantiation for a given templated declaration.
Offers a single interface for instantiation of class, function and
variable templates

\param[in] tmpl - Uninstantiated template class/function
\param[in] template_args - Pointer to vector of template arguments stored
          in the \c TemplateArgInfo struct
\param[in] template_args_size - Size of the vector of template arguments
          passed as \c template_args
\param[in] instantiate_body - also instantiate/define the body

\returns Instantiated templated class/function/variable pointer}];

  let ReturnType = "DeclRef";
  let Args = [
    Arg<"DeclRef", "tmpl">,
    Arg<"const TemplateArgInfo*", "template_args">,
    Arg<"size_t", "template_args_size">,
    Arg<"bool", "instantiate_body", "false">
  ];
  let Deprecated = "use the const std::vector<TemplateArgInfo>& overload";
}

def InstantiateTemplateVec : CppInterOpAPI {
  let Doc = [{Convenience overload of \c InstantiateTemplate that takes the
template argument list by const reference. Equivalent to calling the
pointer+size overload with \c template_args.data() and
\c template_args.size().}];

  let CppName = "InstantiateTemplate";
  let ReturnType = "DeclRef";
  let Args = [
    Arg<"DeclRef", "tmpl">,
    Arg<"const std::vector<TemplateArgInfo>&", "template_args">,
    Arg<"bool", "instantiate_body", "false">
  ];
}

def GetParentScope : CppInterOpAPI {
  let Doc = "Gets the parent of the scope that is passed as a parameter.";

  let ReturnType = "DeclRef";
  let Args = [
    Arg<"ConstDeclRef", "DRef">
  ];
}

def IsTemplate : CppInterOpAPI {
  let Doc = "Checks if it is a templated class.";

  let ReturnType = "bool";
  let Args = [
    Arg<"ConstDeclRef", "DRef">
  ];
}

def IsTypedefed : CppInterOpAPI {
  let Doc = "Checks if \\c DRef introduces a typedef name via \\c typedef or \\c using.";

  let ReturnType = "bool";
  let Args = [
    Arg<"ConstDeclRef", "DRef">
  ];
}

def IsClassPolymorphic : CppInterOpAPI {
  let Doc = [{Checks if the klass polymorphic.
which means that the class contains or inherits a virtual function}];

  let ReturnType = "bool";
  let Args = [
    Arg<"ConstDeclRef", "DRef">
  ];
}

def Demangle : CppInterOpAPI {
  let Doc = "\\returns the demangled representation of the given mangled_name";

  let ReturnType = "std::string";
  let Args = [
    Arg<"const std::string&", "mangled_name">
  ];
}

def GetSizeOfType : CppInterOpAPI {
  let Doc = "Gets the size of the \"type\" that is passed in to this function.";

  let ReturnType = "size_t";
  let Args = [
    Arg<"ConstTypeRef", "TyRef">
  ];
}

def IsBuiltin : CppInterOpAPI {
  let Doc = "Checks if it is a \"built-in\" or a \"complex\" type.";

  let ReturnType = "bool";
  let Args = [
    Arg<"ConstTypeRef", "TyRef">
  ];
}

def IsComplete : CppInterOpAPI {
  let Doc = [{Checks if the class definition is present, or not. Performs a
template instantiation if necessary.}];

  let ReturnType = "bool";
  let Args = [
    Arg<"ConstDeclRef", "DRef">
  ];
}

def Allocate : CppInterOpAPI {
  let Doc = [{Allocates memory required by an object of a given class
\param[in] scope Given class for which to allocate memory for
\param[in] count is used to indicate the number of objects to allocate for.}];

  let ReturnType = "ObjectRef";
  let Args = [
    Arg<"DeclRef", "DRef">,
    Arg<"size_t", "count", "1UL">
  ];
}

def Deallocate : CppInterOpAPI {
  let Doc = [{Deallocates memory for a given class.
\param[in] scope Class to indicate size of memory to deallocate
\param[in] count is used to indicate the number of objects to dallocate for}];

  let ReturnType = "void";
  let Args = [
    Arg<"DeclRef", "DRef">,
    Arg<"ObjectRef", "address">,
    Arg<"size_t", "count", "1UL">
  ];
}

def IsAbstract : CppInterOpAPI {
  let ReturnType = "bool";
  let Args = [
    Arg<"ConstDeclRef", "DRef">
  ];
}

def IsEnumScope : CppInterOpAPI {
  let Doc = "Checks if it is an enum name (EnumDecl represents an enum name).";

  let ReturnType = "bool";
  let Args = [
    Arg<"ConstDeclRef", "DRef">
  ];
}

def IsEnumConstant : CppInterOpAPI {
  let Doc = [{Checks if it is an enum's value (EnumConstantDecl represents
each enum constant that is defined).}];

  let ReturnType = "bool";
  let Args = [
    Arg<"ConstDeclRef", "DRef">
  ];
}

def IsAggregate : CppInterOpAPI {
  let Doc = [{Checks if the given class represents an aggregate type).
\returns true if \c scope is an array or a C++ tag (as per C++
[dcl.init.aggr]) \returns true if the scope supports aggregate
initialization.}];

  let ReturnType = "bool";
  let Args = [
    Arg<"ConstDeclRef", "DRef">
  ];
}

def IsVariable : CppInterOpAPI {
  let Doc = "Checks if the passed value is a variable.";

  let ReturnType = "bool";
  let Args = [
    Arg<"ConstDeclRef", "DRef">
  ];
}

def GetAllCppNames : CppInterOpAPI {
  let ReturnType = "void";
  let Args = [
    Arg<"ConstDeclRef", "DRef">,
    OutArg<"std::set<std::string>&", "names">
  ];
}

def GetUsingNamespaces : CppInterOpAPI {
  let Doc = "Gets the list of namespaces utilized in the supplied scope.";

  let ReturnType = "std::vector<DeclRef>";
  let Args = [
    Arg<"ConstDeclRef", "DRef">
  ];
}

def GetCompleteName : CppInterOpAPI {
  let Doc = [{This is similar to GetName() function, but besides
the name, it also gets the template arguments.}];

  let ReturnType = "std::string";
  let Args = [
    Arg<"ConstDeclRef", "DRef">
  ];
}

def GetDestructor : CppInterOpAPI {
  let Doc = "\\returns the class destructor, if any.";

  let ReturnType = "FuncRef";
  let Args = [
    Arg<"ConstDeclRef", "DRef">
  ];
}

def IsVirtualMethod : CppInterOpAPI {
  let Doc = "Checks if the provided parameter is a 'Virtual' method.";

  let ReturnType = "bool";
  let Args = [
    Arg<"ConstFuncRef", "method">
  ];
}

def GetNumBases : CppInterOpAPI {
  let Doc = [{Gets the number of Base Classes for the Derived Class that
is passed as a parameter.}];

  let ReturnType = "size_t";
  let Args = [
    Arg<"ConstDeclRef", "DRef">
  ];
}

def GetName : CppInterOpAPI {
  let Doc = [{Gets the name of any named decl (a class,
namespace, variable, or a function).}];

  let ReturnType = "std::string";
  let Args = [
    Arg<"ConstDeclRef", "DRef">
  ];
}

def GetBaseClass : CppInterOpAPI {
  let Doc = [{Gets a specific Base Class using its index. Typically GetNumBases()
is used to get the number of Base Classes, and then that number
can be used to iterate through the index value to get each specific
base class.}];

  let ReturnType = "DeclRef";
  let Args = [
    Arg<"ConstDeclRef", "DRef">,
    Arg<"size_t", "ibase">
  ];
}

def IsSubclass : CppInterOpAPI {
  let Doc = [{Checks if the supplied Derived Class is a sub-class of the
provided Base Class.}];

  let ReturnType = "bool";
  let Args = [
    Arg<"ConstDeclRef", "derived">,
    Arg<"ConstDeclRef", "base">
  ];
}

def GetOperator : CppInterOpAPI {
  let Doc = "\\returns list of operator overloads";

  let ReturnType = "void";
  let Args = [
    Arg<"ConstDeclRef", "DRef">,
    Arg<"Operator", "op">,
    OutArg<"std::vector<FuncRef>&", "operators">,
    Arg<"OperatorArity", "kind", "kBoth">
  ];
}

def GetBaseClassOffset : CppInterOpAPI {
  let Doc = [{Each base has its own offset in a Derived Class. This offset can be
used to get to the Base Class fields.}];

  let ReturnType = "int64_t";
  let Args = [
    Arg<"ConstDeclRef", "derived">,
    Arg<"ConstDeclRef", "base">
  ];
}

def GetClassMethods : CppInterOpAPI {
  let Doc = [{Sets a list of all the Methods that are in the Class that is
supplied as a parameter.
\param[in] klass - Pointer to the scope/class under which the methods have
          to be retrieved
\param[out] methods - Vector of methods in the class}];

  let ReturnType = "void";
  let Args = [
    Arg<"ConstDeclRef", "DRef">,
    OutArg<"std::vector<FuncRef>&", "methods">
  ];
}

def IsFunctionProtoType : CppInterOpAPI {
  let Doc = "Check if given type is FunctionProtoType.";

  let ReturnType = "bool";
  let Args = [
    Arg<"ConstTypeRef", "TyRef">
  ];
}

def GetAllocType: CppInterOpAPI {
  let Doc = "Analyzes function's body to detect memory allocation behaviour";

  let ReturnType = "AllocType";
  let Args = [
    Arg<"ConstFuncRef", "Fn">
  ];
}

def GetFnTypeSignature : CppInterOpAPI {
  let Doc = [{Pushed the signature type of the given function type,
  where the first item is the return type.}];

  let ReturnType = "void";
  let Args = [
    Arg<"ConstTypeRef", "fn_type">,
    OutArg<"std::vector<TypeRef>&", "sig">
  ];
}

def GetFunctionNumArgs : CppInterOpAPI {
  let Doc = "Gets the number of Arguments for the provided function.";

  let ReturnType = "size_t";
  let Args = [
    Arg<"ConstFuncRef", "func">
  ];
}

def GetFunctionArgName : CppInterOpAPI {
  let Doc = "\\returns the argument name of function as string.";

  let ReturnType = "std::string";
  let Args = [
    Arg<"ConstFuncRef", "func">,
    Arg<"size_t", "param_index">
  ];
}

def GetFunctionArgType : CppInterOpAPI {
  let Doc = [{For each Argument of a function, you can get the Argument Type
by providing the Argument Index, based on the number of arguments
from the GetFunctionNumArgs() function.}];

  let ReturnType = "TypeRef";
  let Args = [
    Arg<"ConstFuncRef", "func">,
    Arg<"size_t", "iarg">
  ];
}

def IsConstMethod : CppInterOpAPI {
  let Doc = "Checks if a function declared is of const type or not.";

  let ReturnType = "bool";
  let Args = [
    Arg<"ConstFuncRef", "method">
  ];
}

def IsTemplatedFunction : CppInterOpAPI {
  let ReturnType = "bool";
  let Args = [
    Arg<"ConstFuncRef", "func">
  ];
}

def IsStaticMethod : CppInterOpAPI {
  let Doc = "Checks if the provided parameter is a 'Static' method.";

  let ReturnType = "bool";
  let Args = [
    Arg<"ConstFuncRef", "method">
  ];
}

def IsFunctionDeleted : CppInterOpAPI {
  let Doc = "\\returns if a function was marked as \\c =delete.";

  let ReturnType = "bool";
  let Args = [
    Arg<"ConstFuncRef", "function">
  ];
}

def IsPublicMethod : CppInterOpAPI {
  let Doc = "Checks if the provided parameter is a 'Public' method.";

  let ReturnType = "bool";
  let Args = [
    Arg<"ConstFuncRef", "method">
  ];
}

def IsProtectedMethod : CppInterOpAPI {
  let Doc = "Checks if the provided parameter is a 'Protected' method.";

  let ReturnType = "bool";
  let Args = [
    Arg<"ConstFuncRef", "method">
  ];
}

def IsPrivateMethod : CppInterOpAPI {
  let Doc = "Checks if the provided parameter is a 'Private' method.";

  let ReturnType = "bool";
  let Args = [
    Arg<"ConstFuncRef", "method">
  ];
}

def IsConstructor : CppInterOpAPI {
  let Doc = "Checks if the provided parameter is a Constructor.";

  let ReturnType = "bool";
  let Args = [
    Arg<"ConstFuncRef", "method">
  ];
}

def IsDestructor : CppInterOpAPI {
  let Doc = "Checks if the provided parameter is a Destructor.";

  let ReturnType = "bool";
  let Args = [
    Arg<"ConstFuncRef", "method">
  ];
}

def GetDatamembers : CppInterOpAPI {
  let Doc = "Gets all the Fields/Data Members of a Class";

  let ReturnType = "void";
  let Args = [
    Arg<"DeclRef", "DRef">,
    OutArg<"std::vector<DeclRef>&", "datamembers">
  ];
}

def GetStaticDatamembers : CppInterOpAPI {
  let Doc = [{Gets all the Static Fields/Data Members of a Class
\param[in] scope - class
\param[out] funcs - vector of static data members}];

  let ReturnType = "void";
  let Args = [
    Arg<"ConstDeclRef", "DRef">,
    OutArg<"std::vector<DeclRef>&", "datamembers">
  ];
}

def LookupDatamember : CppInterOpAPI {
  let Doc = "This is a Lookup function to be used specifically for data members.";

  let ReturnType = "DeclRef";
  let Args = [
    Arg<"const std::string&", "name">,
    Arg<"ConstDeclRef", "parent">
  ];
}

def IsLambdaClass : CppInterOpAPI {
  let Doc = "Check if the given type is a lamda class";

  let ReturnType = "bool";
  let Args = [
    Arg<"ConstTypeRef", "TyRef">
  ];
}

def GetQualifiedName : CppInterOpAPI {
  let Doc = [{Gets the "qualified" name (including the namespace) of any
named decl (a class, namespace, variable, or a function).}];

  let ReturnType = "std::string";
  let Args = [
    Arg<"ConstDeclRef", "DRef">
  ];
}

def GetVariableOffset : CppInterOpAPI {
  let Doc = [{Gets the address of the variable, you can use it to get the
value stored in the variable.}];

  let ReturnType = "intptr_t";
  let Args = [
    Arg<"ConstDeclRef", "var">,
    Arg<"ConstDeclRef", "parent", "nullptr">
  ];
}

def IsPublicVariable : CppInterOpAPI {
  let Doc = "Checks if the provided variable is a 'Public' variable.";

  let ReturnType = "bool";
  let Args = [
    Arg<"ConstDeclRef", "var">
  ];
}

def IsProtectedVariable : CppInterOpAPI {
  let Doc = "Checks if the provided variable is a 'Protected' variable.";

  let ReturnType = "bool";
  let Args = [
    Arg<"ConstDeclRef", "var">
  ];
}

def IsPrivateVariable : CppInterOpAPI {
  let Doc = "Checks if the provided variable is a 'Private' variable.";

  let ReturnType = "bool";
  let Args = [
    Arg<"ConstDeclRef", "var">
  ];
}

def IsStaticVariable : CppInterOpAPI {
  let Doc = "Checks if the provided variable is a 'Static' variable.";

  let ReturnType = "bool";
  let Args = [
    Arg<"ConstDeclRef", "var">
  ];
}

def IsConstVariable : CppInterOpAPI {
  let Doc = "Checks if the provided variable is a 'Constant' variable.";

  let ReturnType = "bool";
  let Args = [
    Arg<"ConstDeclRef", "var">
  ];
}

def GetDimensions : CppInterOpAPI {
  let Doc = "Gets the size/dimensions of a multi-dimension array.";
  // FIXME: Rework to use a C-compatible array type.
  let NoCWrapper = true;
  let ReturnType = "std::vector<long int>";
  let Args = [
    Arg<"ConstTypeRef", "TyRef">
  ];
}

def GetEnumConstants : CppInterOpAPI {
  let Doc = "Gets a list of all the enum constants for an enum.";

  let ReturnType = "std::vector<DeclRef>";
  let Args = [
    Arg<"ConstDeclRef", "DRef">
  ];
}

def GetEnumConstantType : CppInterOpAPI {
  let Doc = "Gets the enum name when an enum constant is passed.";

  let ReturnType = "TypeRef";
  let Args = [
    Arg<"ConstDeclRef", "DRef">
  ];
}

def GetEnumConstantValue : CppInterOpAPI {
  let Doc = [{Gets the index value (0,1,2, etcetera) of the enum constant
that was passed into this function.}];

  let ReturnType = "size_t";
  let Args = [
    Arg<"ConstDeclRef", "DRef">
  ];
}

def DumpScope : CppInterOpAPI {
  let ReturnType = "void";
  let Args = [
    Arg<"ConstDeclRef", "DRef">
  ];
}

def AddSearchPath : CppInterOpAPI {
  let Doc = "Adds a Search Path for the Interpreter to get the libraries.";

  let ReturnType = "void";
  let Args = [
    Arg<"const char*", "dir">,
    Arg<"bool", "isUser", "true">,
    Arg<"bool", "prepend", "false">
  ];
}

def IsDebugOutputEnabled : CppInterOpAPI {
  let Doc = "\\returns true if the debugging printouts on stderr are enabled.";

  let ReturnType = "bool";
}

def EnableDebugOutput : CppInterOpAPI {
  let Doc = [{Enables or disables the debugging printouts on stderr.
Debugging output can be enabled also by the environment variable
CPPINTEROP_EXTRA_INTERPRETER_ARGS. For example,
CPPINTEROP_EXTRA_INTERPRETER_ARGS="-mllvm -debug-only=jitcall" to produce
only debug output for jitcall events.}];

  let ReturnType = "void";
  let Args = [
    Arg<"bool", "value", "true">
  ];
}

def GetDoxygenComment : CppInterOpAPI {
  let Doc = [{Retrieves the Doxygen documentation comment for a declaration.
\param[in] scope -- The declaration to get the comment for.
\param[in] strip_comment_markers -- If true, removes comment markers (///,
/**, etc.).
\returns The documentation comment, or empty string if none exists.}];

  let ReturnType = "std::string";
  let Args = [
    Arg<"ConstDeclRef", "DRef">,
    Arg<"bool", "strip_comment_markers", "true">
  ];
}

def IsExplicit : CppInterOpAPI {
  let Doc = "Checks if the provided constructor or conversion operator is explicit";

  let ReturnType = "bool";
  let Args = [
    Arg<"ConstFuncRef", "method">
  ];
}


// --- Functions not previously in the dispatch table ---

def ActivateInterpreter : CppInterOpAPI {
  let Doc = [{Activates an instance of an interpreter to DRef subsequent API
requests.}];
  let ReturnType = "bool";
  let Args = [Arg<"InterpRef", "I">];
}

def AddTypeQualifier : CppInterOpAPI {
  let Doc = [{Returns type with the qual Qualifiers added.
qual can be ORed value of enum QualKind.}];
  let ReturnType = "TypeRef";
  let Args = [
    Arg<"ConstTypeRef", "TyRef">,
    Arg<"QualKind", "qual">
  ];
}

def BeginStdStreamCapture : CppInterOpAPI {
  let Doc = [{Begins recording the given standard stream.
\param[in] fd_kind The stream to be captured.}];
  let ReturnType = "void";
  let Args = [Arg<"CaptureStreamKind", "fd_kind">];
}

def BestOverloadFunctionMatch : CppInterOpAPI {
  let Doc = [{Finds best overload match based on explicit template parameters
(if any) and argument types.}];
  let ReturnType = "FuncRef";
  let Args = [
    Arg<"const std::vector<FuncRef>&", "candidates">,
    Arg<"const std::vector<TemplateArgInfo>&", "explicit_types">,
    Arg<"const std::vector<TemplateArgInfo>&", "arg_types">
  ];
}

def CodeComplete : CppInterOpAPI {
  let Doc = [{Append all code completion suggestions to Results.
\param[out] Results CC suggestions for code fragment (appended).
\param[in] code Code fragment to complete.
\param[in] complete_line Position (line) in code for suggestion.
\param[in] complete_column Position (column) in code for suggestion.}];
  let ReturnType = "void";
  let Args = [
    OutArg<"std::vector<std::string>&", "Results">,
    Arg<"const char*", "code">,
    Arg<"unsigned", "complete_line", "1U">,
    Arg<"unsigned", "complete_column", "1U">
  ];
}

def DetectResourceDir : CppInterOpAPI {
  let Doc = [{Uses the underlying clang compiler to detect the resource directory.
In essence it asks clang to print its resource-dir and returns the path.
\param[in] ClangBinaryName The name or full path of the compiler to ask.}];
  let ReturnType = "std::string";
  let Args = [Arg<"const char*", "ClangBinaryName", "\"clang\"">];
}

def DetectSystemCompilerIncludePaths : CppInterOpAPI {
  let Doc = [{Asks the system compiler for its default include paths.
\param[out] Paths The list of include paths returned.
\param[in] CompilerName The name or full path of the compiler binary.}];
  let ReturnType = "void";
  let Args = [
    OutArg<"std::vector<std::string>&", "Paths">,
    Arg<"const char*", "CompilerName", "\"c++\"">
  ];
}

def EndStdStreamCapture : CppInterOpAPI {
  let Doc = "Ends recording the standard stream and returns the result as a string.";
  let ReturnType = "std::string";
}

def ExistsFunctionTemplate : CppInterOpAPI {
  let Doc = [{This function performs a lookup to check if there is a
templated function of that type within the parent scope.}];
  let ReturnType = "bool";
  let Args = [
    Arg<"const std::string&", "name">,
    Arg<"ConstDeclRef", "parent", "nullptr">
  ];
}

def GetClassTemplateArgs : CppInterOpAPI {
  let Doc = [{Sets the class template arguments of \c templ_instance.
\param[in] templ_instance Pointer to the template instance.
\param[out] args Vector of instantiation arguments.}];
  let ReturnType = "void";
  let Args = [
    Arg<"ConstDeclRef", "templ_instance">,
    OutArg<"std::vector<TemplateArgInfo>&", "args">
  ];
}

def GetClassTemplateInstantiationArgs : CppInterOpAPI {
  let Doc = [{Sets the class template instantiation arguments of \c templ_instance.
\param[in] templ_instance Pointer to the template instance.
\param[out] args Vector of instantiation arguments.}];
  // FIXME: Needs a dedicated array type — TemplateArgInfo is not pointer-sized.
  let NoCWrapper = true;
  let ReturnType = "void";
  let Args = [
    Arg<"ConstDeclRef", "templ_instance">,
    OutArg<"std::vector<TemplateArgInfo>&", "args">
  ];
}

def GetClassTemplatedMethods : CppInterOpAPI {
  let Doc = [{Sets a list of all the Templated Methods that are in the Class
that is supplied as a parameter.}];
  // Hand-written C wrapper: bool return dropped, caller checks arr.size.
  let NoCWrapper = true;
  let ReturnType = "bool";
  let Args = [
    Arg<"const std::string&", "name">,
    Arg<"ConstDeclRef", "parent">,
    OutArg<"std::vector<FuncRef>&", "funcs">
  ];
}

def GetDefaultConstructor : CppInterOpAPI {
  let Doc = "\\returns the default constructor of a class, if any.";
  let ReturnType = "FuncRef";
  let Args = [Arg<"DeclRef", "DRef">];
}

def GetEnumConstantDatamembers : CppInterOpAPI {
  let Doc = [{Gets all enum constant data members of a class.
\param[in] scope Class to query.
\param[out] datamembers Vector of enum constant members.
\param[in] include_enum_class Whether to include enum class members.}];
  let ReturnType = "void";
  let Args = [
    Arg<"ConstDeclRef", "DRef">,
    OutArg<"std::vector<DeclRef>&", "datamembers">,
    Arg<"bool", "include_enum_class", "true">
  ];
}

def GetEnums : CppInterOpAPI {
  let Doc = "Extracts enum declarations from a specified scope and stores them in a vector.";
  let ReturnType = "void";
  let Args = [
    Arg<"ConstDeclRef", "DRef">,
    OutArg<"std::vector<std::string>&", "Result">
  ];
}

def GetFunctionArgDefault : CppInterOpAPI {
  let Doc = "\\returns the default argument value as string.";
  let ReturnType = "std::string";
  let Args = [
    Arg<"ConstFuncRef", "func">,
    Arg<"size_t", "param_index">
  ];
}

def GetFunctionRequiredArgs : CppInterOpAPI {
  let Doc = "Gets the number of Required Arguments for the provided function.";
  let ReturnType = "size_t";
  let Args = [Arg<"ConstFuncRef", "func">];
}

def GetFunctionReturnType : CppInterOpAPI {
  let Doc = "Gets the return type of the provided function.";
  let ReturnType = "TypeRef";
  let Args = [Arg<"ConstFuncRef", "func">];
}

def IsAllocator : CppInterOpAPI {
  let Doc = "Checks if the provided function is an allocator function";
  let ReturnType = "bool";
  let Args = [
    Arg<"ConstFuncRef", "func">
  ];
}

def IsDeallocator : CppInterOpAPI {
  let Doc = "Checks if the provided function is a deallocator function";
  let ReturnType = "bool";
  let Args = [
    Arg<"ConstFuncRef", "func">
  ];
}

def GetFunctionSignature : CppInterOpAPI {
  let Doc = [{\\returns a stringified version of a given function signature in the
form: void N::f(int i, double d, long l = 0, char ch = 'a').}];
  let ReturnType = "std::string";
  let Args = [Arg<"ConstFuncRef", "func">];
}

def GetFunctionTemplatedDecls : CppInterOpAPI {
  let Doc = [{Template function pointer list to add proxies for
un-instantiated/non-overloaded templated methods.}];
  let ReturnType = "void";
  let Args = [
    Arg<"ConstDeclRef", "DRef">,
    OutArg<"std::vector<FuncRef>&", "methods">
  ];
}

def GetFunctionsUsingName : CppInterOpAPI {
  let Doc = [{Looks up all the functions that have the name that is
passed as a parameter in this function.}];
  let ReturnType = "std::vector<FuncRef>";
  let Args = [
    Arg<"ConstDeclRef", "DRef">,
    Arg<"const std::string&", "name">
  ];
}

def GetIncludePaths : CppInterOpAPI {
  let Doc = [{\\param[out] IncludePaths The list of include paths.
\\param[in] withSystem Include system paths.
\\param[in] withFlags Include flag-based paths.}];
  let ReturnType = "void";
  let Args = [
    OutArg<"std::vector<std::string>&", "IncludePaths">,
    Arg<"bool", "withSystem", "false">,
    Arg<"bool", "withFlags", "false">
  ];
}

def GetIntegerTypeFromEnumScope : CppInterOpAPI {
  let Doc = [{For the given "class", get the integer type that the enum
represents, so that you can store it properly in a box.}];
  let ReturnType = "TypeRef";
  let Args = [Arg<"ConstDeclRef", "DRef">];
}

def GetIntegerTypeFromEnumType : CppInterOpAPI {
  let Doc = [{For the given "type", this function gets the integer type that
the enum represents.}];
  let ReturnType = "TypeRef";
  let Args = [Arg<"ConstTypeRef", "DRef">];
}

def GetOperatorArity : CppInterOpAPI {
  let Doc = "\\returns arity of the operator or kNone.";
  let ReturnType = "OperatorArity";
  let Args = [Arg<"ConstFuncRef", "op">];
}

def GetOperatorFromSpelling : CppInterOpAPI {
  let Doc = "\\returns operator representing the string.";
  let ReturnType = "Operator";
  let Args = [Arg<"const std::string&", "op">];
}

def GetQualifiedCompleteName : CppInterOpAPI {
  let Doc = [{This is similar to GetQualifiedName() function, but besides
the "qualified" name, it also gets the template arguments.}];
  let ReturnType = "std::string";
  let Args = [Arg<"ConstDeclRef", "DRef">];
}

def GetScopeFromCompleteName : CppInterOpAPI {
  let Doc = [{When the namespace is known, then the parent doesn't need
to be specified. This will probably be phased-out in
future versions of the interop library.}];
  let ReturnType = "DeclRef";
  let Args = [Arg<"const std::string&", "name">];
}

def GetSpellingFromOperator : CppInterOpAPI {
  let Doc = "\\returns string representation of the operator.";
  let ReturnType = "std::string";
  let Args = [Arg<"Operator", "op">];
}

def GetVersion : CppInterOpAPI {
  let Doc = "\\returns the version string information of the library.";
  let ReturnType = "std::string";
}

def GetBuildInfo : CppInterOpAPI {
  let Doc = [{\returns a multi-line snapshot of the CppInterOp configure-
time build context (build type, compiler, LLVM version, target, the
filtered cmake cache flags). Embedded in trace reproducers and bug
reports so a reviewer can compare their build to the reporter's.}];

  let ReturnType = "std::string";
}

def HasDefaultConstructor : CppInterOpAPI {
  let Doc = "\\returns if a class has a default constructor.";
  let ReturnType = "bool";
  let Args = [Arg<"ConstDeclRef", "DRef">];
}

def InsertOrReplaceJitSymbol : CppInterOpAPI {
  let Doc = [{Inserts or replaces a symbol in the JIT with the one provided.
This is useful for providing implementations for symbols not linked in.}];
  let ReturnType = "bool";
  let Args = [
    Arg<"const char*", "linker_mangled_name">,
    Arg<"uint64_t", "address">
  ];
}

def InstantiateTemplateFunctionFromString : CppInterOpAPI {
  let Doc = [{Instantiates a function template from a given string representation.
This function parses and instantiates a template function.}];
  let ReturnType = "FuncRef";
  let Args = [Arg<"const char*", "function_template">];
}

def IsTemplateParmType : CppInterOpAPI {
  let Doc = "Checks if the given type is a template parameter type.";
  let ReturnType = "bool";
  let Args = [
    Arg<"ConstTypeRef", "TyRef">
  ];
}

def IsFunction : CppInterOpAPI {
  let Doc = "Checks if the scope is a function.";
  let ReturnType = "bool";
  let Args = [Arg<"ConstDeclRef", "DRef">];
}

def IsFunctionPointerType : CppInterOpAPI {
  let Doc = "Checks if the type is a function pointer.";
  let ReturnType = "bool";
  let Args = [Arg<"ConstTypeRef", "TyRef">];
}

def IsMethod : CppInterOpAPI {
  let Doc = "Checks if the provided parameter is a method.";
  let ReturnType = "bool";
  let Args = [Arg<"ConstFuncRef", "method">];
}

def IsPODType : CppInterOpAPI {
  let Doc = "Checks if the provided parameter is a Plain Old Data Type (POD).";
  let ReturnType = "bool";
  let Args = [Arg<"ConstTypeRef", "TyRef">];
}

def IsSmartPtrType : CppInterOpAPI {
  let Doc = [{We assume that smart pointer types define both operator* and
operator->.}];
  let ReturnType = "bool";
  let Args = [Arg<"ConstTypeRef", "TyRef">];
}

def IsTemplateSpecialization : CppInterOpAPI {
  let Doc = "Checks if it is a class template specialization class.";
  let ReturnType = "bool";
  let Args = [Arg<"ConstDeclRef", "DRef">];
}

def IsTypeDerivedFrom : CppInterOpAPI {
  let Doc = "Checks if a C++ type derives from another.";
  let ReturnType = "bool";
  let Args = [
    Arg<"ConstTypeRef", "derived">,
    Arg<"ConstTypeRef", "base">
  ];
}

def LookupConstructors : CppInterOpAPI {
  let Doc = [{Sets a list of all the constructor for a scope/class that is
supplied as a parameter.}];
  let ReturnType = "void";
  let Args = [
    Arg<"const std::string&", "name">,
    Arg<"ConstDeclRef", "parent">,
    OutArg<"std::vector<FuncRef>&", "funcs">
  ];
}

def LookupLibrary : CppInterOpAPI {
  let Doc = [{Looks up the library if access is enabled.
\\returns the path to the library.}];
  let ReturnType = "std::string";
  let Args = [Arg<"const char*", "lib_name">];
}

def MakeFunctionCallable_interp : CppInterOpAPI {
  let CppName = "MakeFunctionCallable";
  let Doc = [{Creates a trampoline function using a specific interpreter and returns
a uniform interface to call it from compiled code.}];
  // JitCall is a C++ class; no mechanical C mapping.
  let NoCWrapper = true;
  let ReturnType = "JitCall";
  let Args = [
    Arg<"InterpRef", "I">,
    Arg<"ConstFuncRef", "func">
  ];
}

def SearchLibrariesForSymbol : CppInterOpAPI {
  let Doc = [{Scans all libraries on the library search path for a given
potentially mangled symbol name.}];
  let ReturnType = "std::string";
  let Args = [
    Arg<"const char*", "mangled_name">,
    Arg<"bool", "search_system", "true">
  ];
}

def Undo : CppInterOpAPI {
  let Doc = [{Reverts the last N operations performed by the interpreter.
\\param[in] N The number of operations to undo. Defaults to 1.
\\returns 0 on success, non-zero on failure.}];
  let ReturnType = "int";
  let Args = [Arg<"unsigned", "N", "1">];
}

def UnloadLibrary : CppInterOpAPI {
  let Doc = [{Finds \\c lib_stem considering the list of search paths and unloads it
by calling dlclose.}];
  let ReturnType = "void";
  let Args = [Arg<"const char*", "lib_stem">];
}

def UseExternalInterpreter : CppInterOpAPI {
  let Doc = [{Sets the Interpreter instance with an external interpreter,
meant to be called before any other API calls.}];
  let ReturnType = "void";
  let Args = [Arg<"InterpRef", "I">];
}

def MakeFunctionCallable : CppInterOpAPI {
  let Doc = [{Creates a trampoline function by using the interpreter and returns a
uniform interface to call it from compiled code.}];
  // JitCall is a C++ class; no mechanical C mapping.
  let NoCWrapper = true;
  let ReturnType = "JitCall";
  let Args = [Arg<"ConstFuncRef", "func">];
}

// --- Runtime vtable overlay: replace virtual slots on a live object ---

def MakeVTableOverlay : CppInterOpAPI {
  let Doc = [{Copy the vtable of \c inst, replace the slots of the listed
virtual \c methods with the given function pointers, and install the copy on
\c inst. The returned DRef owns the copy and remembers the original vptr;
DestroyVTableOverlay undoes the install. All non-overlaid slots and the
ABI prefix (offset-to-top + type_info on Itanium) are preserved, so the
destructor, RTTI and unrelated virtuals keep dispatching normally. Slot
indices and the vtable size are derived from \c base via Clang's
VTableContext, so the caller works in reflected methods, not ABI-specific
indices, and the result is correct for both Itanium and Microsoft ABIs.

The install is per-instance: \c inst's vptr is the only pointer rewritten,
so other live or future instances of the same type continue to use the
class's original vtable. The rewrite covers only the primary vptr (offset
0 of \c inst); a class with multiple polymorphic direct bases or any
virtual base would have additional vptrs / vbase-offset entries the
overlay does not touch and is rejected outright -- see \returns.
\param[in] inst Object whose vptr is read and replaced.
\param[in] base Polymorphic class supplying \c inst's vtable layout.
\param[in] methods Virtual methods of \c base whose slots are replaced.
\param[in] overlay_fns Function pointers to install at those slots.
\param[in] n_overlays Length of \c methods and \c overlay_fns.
\param[in] n_extra_prefix_slots Optional \c void* slots prepended *before* the
           ABI prefix in the owned block, initialized to nullptr. Bindings use
           them to stash per-instance data adjacent to the vtable -- a thunk
           reads its slot via \c VTableOverlayExtraSlot(self, i) with a single
           fixed-offset load, avoiding a runtime registry lookup.
\param[in] on_destroy Optional callback invoked on the operator-delete path
           (D0 on Itanium, the single deleting dtor on MSVC). Bindings use
           it to release per-instance state tied to the object's lifetime
           (Python handles, closures). Ordering is callback first, then the
           original deleting destructor; the callback runs while \c inst is
           still alive and may inspect its state. After the original dtor
           returns, the object's memory has been freed -- the callback
           cannot defer the read. The callback receives \c inst and
           \c cleanup_data verbatim, runs inside a C++ destructor, and must
           not throw -- an escaping exception is undefined behavior. The
           callback also must not destroy this overlay (the original dtor
           still has to run after it returns). Stack-only destruction (D1)
           does not trigger the callback. Pass nullptr to skip the dtor
           hook entirely (zero per-call overhead, slot not patched).
\param[in] cleanup_data Opaque pointer passed back to \c on_destroy.
\returns nullptr if \c inst is null, any method is null or not virtual,
         \c base is not a polymorphic class, or \c base has multiple
         polymorphic direct bases or any virtual base.}];
  let NoCWrapper = true;
  let ReturnType = "VTableOverlay*";
  let Args = [
    Arg<"void*", "inst">,
    Arg<"ConstDeclRef", "base">,
    Arg<"const ConstFuncRef*", "methods">,
    Arg<"void* const*", "overlay_fns">,
    Arg<"std::size_t", "n_overlays">,
    Arg<"std::size_t", "n_extra_prefix_slots", "0">,
    Arg<"VTableOverlayDtorHook", "on_destroy", "nullptr">,
    Arg<"void*", "cleanup_data", "nullptr">
  ];
}

def DestroyVTableOverlay : CppInterOpAPI {
  let Doc = [{Restore the original vptr on the overlaid instance and free the
overlay. Null-safe.}];
  let NoCWrapper = true;
  let ReturnType = "void";
  let Args = [Arg<"VTableOverlay*", "overlay">];
}

def IsIntegerType : CppInterOpAPI {
  let Doc = [{Checks if type has an integer representation.
If \p s is non-null, it is set to the signedness of the type.}];
  let ReturnType = "bool";
  let Args = [
    Arg<"ConstTypeRef", "TyRef">,
    Arg<"Signedness*", "s", "nullptr">
  ];
}

def IsFloatingType : CppInterOpAPI {
  let Doc = "Checks if type has a floating representation.";
  let ReturnType = "bool";
  let Args = [Arg<"ConstTypeRef", "TyRef">];
}

def IsSameType : CppInterOpAPI {
  let Doc = [{Checks if two types are the equivalent,
i.e. have the same canonical type.}];
  let ReturnType = "bool";
  let Args = [
    Arg<"ConstTypeRef", "type_a">,
    Arg<"ConstTypeRef", "type_b">
  ];
}

def IsVoidPointerType : CppInterOpAPI {
  let Doc = "Checks if type is a void pointer.";
  let ReturnType = "bool";
  let Args = [Arg<"ConstTypeRef", "TyRef">];
}

// Trace-hook impls invoked by JitCall's inline fast path through
// ::CppInternal::DispatchRaw slots. Routing through the X-macro means
// Dispatch.h consumers pick up the slots via LoadDispatchAPI without
// any source change of their own.
def CppInterOpTraceJitCallInvokeImpl : CppInterOpAPI {
  let Doc = "Trace hook for the JitCall::Invoke generic call path.";
  let ReturnType = "void";
  let Args = [
    Arg<"const Cpp::JitCall*", "JC">,
    Arg<"void*", "result">,
    Arg<"void**", "args">,
    Arg<"std::size_t", "nargs">,
    Arg<"void*", "self">
  ];
}

def CppInterOpTraceJitCallInvokeDestructorImpl : CppInterOpAPI {
  let Doc = "Trace hook for the JitCall::InvokeDestructor path.";
  let ReturnType = "void";
  let Args = [
    Arg<"const Cpp::JitCall*", "JC">,
    Arg<"void*", "object">,
    Arg<"unsigned long", "nary">,
    Arg<"int", "withFree">
  ];
}

def CppInterOpTraceJitCallInvokeReturnImpl : CppInterOpAPI {
  let Doc = "Trace hook for post-Invoke pointer-return DRef registration.";
  let ReturnType = "void";
  let Args = [
    Arg<"const Cpp::JitCall*", "JC">,
    Arg<"void*", "result">
  ];
}
