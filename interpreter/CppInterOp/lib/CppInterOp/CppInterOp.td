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

  let ReturnType = "TInterp_t";
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

  let ReturnType = "TInterp_t";
}

def DeleteInterpreter : CppInterOpAPI {
  let Doc = [{Deletes an instance of an interpreter.
\param[in] I - the interpreter to be deleted, if nullptr, deletes the last.
\returns false on failure or if \c I is not tracked in the stack.}];

  let ReturnType = "bool";
  let Args = [Arg<"TInterp_t", "I", "nullptr">];
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
  let Doc = [{Declares, executes and returns the execution result as a intptr_t.
\param[in] code - the snippet to declare and execute.
\param[out] IsValueInvalid - if non-null, set true when the result is
            unusable (parse error or no-Value-after-success).
\returns the expression results as a intptr_t.}];

  let ReturnType = "intptr_t";
  let Args = [
    Arg<"const char*", "code">,
    OutArg<"bool*", "IsValueInvalid", "nullptr">
  ];
}

// --- Scope / type queries ---

def GetScope : CppInterOpAPI {
  let Doc = [{Gets the namespace or class (by stripping typedefs) for the name
passed as a parameter, and if the parent is not passed,
then global scope will be assumed.}];

  let ReturnType = "TCppScope_t";
  let Args = [
    Arg<"const std::string&", "name">,
    Arg<"TCppScope_t", "parent", "nullptr">
  ];
}

def GetNamed : CppInterOpAPI {
  let Doc = [{This function performs a lookup within the specified parent,
a specific named entity (functions, enums, etcetera).}];

  let ReturnType = "TCppScope_t";
  let Args = [
    Arg<"const std::string&", "name">,
    Arg<"TCppScope_t", "parent", "nullptr">
  ];
}

def GetGlobalScope : CppInterOpAPI {
  let Doc = "Gets the global scope of the whole C++  instance.";

  let ReturnType = "TCppScope_t";
}

def IsNamespace : CppInterOpAPI {
  let Doc = "Checks if the scope is a namespace or not.";

  let ReturnType = "bool";
  let Args = [Arg<"TCppScope_t", "scope">];
}

def IsClass : CppInterOpAPI {
  let Doc = "Checks if the scope is a class or not.";

  let ReturnType = "bool";
  let Args = [Arg<"TCppScope_t", "scope">];
}

def GetType : CppInterOpAPI {
  let Doc = [{Used to either get the built-in type of the provided string, or
use the name to lookup the actual type. When `parent` is non-null, scoped
lookup honours using-directives and type aliases declared inside it; passing
`nullptr` (the default) limits lookup to TU scope.}];

  let ReturnType = "TCppType_t";
  let Args = [Arg<"const std::string&", "name">,
              Arg<"TCppScope_t", "parent", "nullptr">];
}

def GetTypeFromScope : CppInterOpAPI {
  let Doc = [{This will convert a class into its type, so for example, you can
use it to declare variables in it.}];

  let ReturnType = "TCppType_t";
  let Args = [Arg<"TCppScope_t", "klass">];
}

def GetScopeFromType : CppInterOpAPI {
  let Doc = "Gets the scope of the type that is passed as a parameter.";

  let ReturnType = "TCppScope_t";
  let Args = [Arg<"TCppType_t", "type">];
}

def SizeOf : CppInterOpAPI {
  let ReturnType = "size_t";
  let Args = [Arg<"TCppScope_t", "scope">];
}

// --- Functions with overloads ---

def GetFunctionAddress_str : CppInterOpAPI {
  let Doc = "\\returns the address of the function given its potentially mangled name.";
  let CppName = "GetFunctionAddress";
  let ReturnType = "TCppFuncAddr_t";
  let Args = [Arg<"const char*", "mangled_name">];
}

def GetFunctionAddress_fn : CppInterOpAPI {
  let Doc = "\\returns the address of the function given its function declaration.";
  let CppName = "GetFunctionAddress";
  let ReturnType = "TCppFuncAddr_t";
  let Args = [Arg<"TCppFunction_t", "method">];
}

// --- Construction / destruction ---

def Construct : CppInterOpAPI {
  let Doc = [{Creates one or more objects of class \c scope by calling its default
constructor.
\param[in] scope Class to construct, or handle to Constructor
\param[in] arena If set, this API uses placement new to construct at this
address.
\param[in] is used to indicate the number of objects to construct.
\returns a pointer to the constructed object, which is arena if placement
new is used.}];

  let ReturnType = "TCppObject_t";
  let Args = [
    Arg<"TCppScope_t", "scope">,
    Arg<"void*", "arena", "nullptr">,
    Arg<"TCppIndex_t", "count", "1UL">
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
    Arg<"TCppObject_t", "This">,
    Arg<"TCppConstScope_t", "scope">,
    Arg<"bool", "withFree", "true">,
    Arg<"TCppIndex_t", "count", "0UL">
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
    Arg<"const char*", "type">,
    Arg<"void*", "obj">
  ];
}

def GetValueKind : CppInterOpAPI {
  let Doc = "Get if lvalue or rvalue reference";

  let ReturnType = "ValueKind";
  let Args = [
    Arg<"TCppType_t", "type">
  ];
}

def GetNonReferenceType : CppInterOpAPI {
  let Doc = "Get the type that the reference refers to";

  let ReturnType = "TCppType_t";
  let Args = [
    Arg<"TCppType_t", "type">
  ];
}

def IsEnumType : CppInterOpAPI {
  let Doc = "Checks if the passed value is an enum type or not.";

  let ReturnType = "bool";
  let Args = [
    Arg<"TCppType_t", "type">
  ];
}

def GetLanguage : CppInterOpAPI {
  let Doc = "Returns the programming language of the interpreter.";

  let ReturnType = "InterpreterLanguage";
  let Args = [
    Arg<"TInterp_t", "I", "nullptr">
  ];
}

def GetLanguageStandard : CppInterOpAPI {
  let Doc = "Returns the language standard of the interpreter.";

  let ReturnType = "InterpreterLanguageStandard";
  let Args = [
    Arg<"TInterp_t", "I", "nullptr">
  ];
}

def GetReferencedType : CppInterOpAPI {
  let Doc = "Get lvalue referenced type, or rvalue if rvalue is true";

  let ReturnType = "TCppType_t";
  let Args = [
    Arg<"TCppType_t", "type">,
    Arg<"bool", "rvalue", "false">
  ];
}

def IsPointerType : CppInterOpAPI {
  let Doc = "Checks if type is a pointer";

  let ReturnType = "bool";
  let Args = [
    Arg<"TCppType_t", "type">
  ];
}

def GetPointeeType : CppInterOpAPI {
  let Doc = "Get the underlying pointee type";

  let ReturnType = "TCppType_t";
  let Args = [
    Arg<"TCppType_t", "type">
  ];
}

def GetPointerType : CppInterOpAPI {
  let Doc = "Get the pointer to type";

  let ReturnType = "TCppType_t";
  let Args = [
    Arg<"TCppType_t", "type">
  ];
}

def IsReferenceType : CppInterOpAPI {
  let Doc = "Checks if type is a reference";

  let ReturnType = "bool";
  let Args = [
    Arg<"TCppType_t", "type">
  ];
}

def GetTypeAsString : CppInterOpAPI {
  let Doc = "Gets the Type (passed as a parameter) as a String value.";

  let ReturnType = "std::string";
  let Args = [
    Arg<"TCppType_t", "type">
  ];
}

def GetCanonicalType : CppInterOpAPI {
  let Doc = [{Gets the Canonical Type string from the std string. A canonical type
is the type with any typedef names, syntactic sugars or modifiers stripped
out of it.}];

  let ReturnType = "TCppType_t";
  let Args = [
    Arg<"TCppType_t", "type">
  ];
}

def HasTypeQualifier : CppInterOpAPI {
  let Doc = [{Checks if the passed type has qual Qualifiers
qual can be ORed value of enum QualKind}];

  let ReturnType = "bool";
  let Args = [
    Arg<"TCppType_t", "type">,
    Arg<"QualKind", "qual">
  ];
}

def RemoveTypeQualifier : CppInterOpAPI {
  let Doc = [{Returns type with the qual Qualifiers removed
qual can be ORed value of enum QualKind}];

  let ReturnType = "TCppType_t";
  let Args = [
    Arg<"TCppType_t", "type">,
    Arg<"QualKind", "qual">
  ];
}

def GetUnderlyingType : CppInterOpAPI {
  let Doc = "Gets the pure, Underlying Type (as opposed to the Using Type).";

  let ReturnType = "TCppType_t";
  let Args = [
    Arg<"TCppType_t", "type">
  ];
}

def IsRecordType : CppInterOpAPI {
  let Doc = "Checks if the provided parameter is a Record (struct).";

  let ReturnType = "bool";
  let Args = [
    Arg<"TCppType_t", "type">
  ];
}

def GetVariableType : CppInterOpAPI {
  let Doc = "Gets the type of the variable that is passed as a parameter.";

  let ReturnType = "TCppType_t";
  let Args = [
    Arg<"TCppScope_t", "var">
  ];
}

def GetComplexType : CppInterOpAPI {
  let Doc = "\\returns the complex of the provided type.";

  let ReturnType = "TCppType_t";
  let Args = [
    Arg<"TCppType_t", "element_type">
  ];
}

def GetUnderlyingScope : CppInterOpAPI {
  let Doc = [{Strips the typedef and returns the underlying class, and if the
underlying decl is not a class it returns the input unchanged.}];

  let ReturnType = "TCppScope_t";
  let Args = [
    Arg<"TCppScope_t", "scope">
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

  let ReturnType = "TCppScope_t";
  let Args = [
    Arg<"TCppScope_t", "tmpl">,
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
  let ReturnType = "TCppScope_t";
  let Args = [
    Arg<"TCppScope_t", "tmpl">,
    Arg<"const std::vector<TemplateArgInfo>&", "template_args">,
    Arg<"bool", "instantiate_body", "false">
  ];
}

def GetParentScope : CppInterOpAPI {
  let Doc = "Gets the parent of the scope that is passed as a parameter.";

  let ReturnType = "TCppScope_t";
  let Args = [
    Arg<"TCppScope_t", "scope">
  ];
}

def IsTemplate : CppInterOpAPI {
  let Doc = "Checks if it is a templated class.";

  let ReturnType = "bool";
  let Args = [
    Arg<"TCppScope_t", "handle">
  ];
}

def IsTypedefed : CppInterOpAPI {
  let Doc = "Checks if \\c handle introduces a typedef name via \\c typedef or \\c using.";

  let ReturnType = "bool";
  let Args = [
    Arg<"TCppScope_t", "handle">
  ];
}

def IsClassPolymorphic : CppInterOpAPI {
  let Doc = [{Checks if the klass polymorphic.
which means that the class contains or inherits a virtual function}];

  let ReturnType = "bool";
  let Args = [
    Arg<"TCppScope_t", "klass">
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
    Arg<"TCppType_t", "type">
  ];
}

def IsBuiltin : CppInterOpAPI {
  let Doc = "Checks if it is a \"built-in\" or a \"complex\" type.";

  let ReturnType = "bool";
  let Args = [
    Arg<"TCppConstType_t", "type">
  ];
}

def IsComplete : CppInterOpAPI {
  let Doc = [{Checks if the class definition is present, or not. Performs a
template instantiation if necessary.}];

  let ReturnType = "bool";
  let Args = [
    Arg<"TCppScope_t", "scope">
  ];
}

def Allocate : CppInterOpAPI {
  let Doc = [{Allocates memory required by an object of a given class
\param[in] scope Given class for which to allocate memory for
\param[in] count is used to indicate the number of objects to allocate for.}];

  let ReturnType = "TCppObject_t";
  let Args = [
    Arg<"TCppScope_t", "scope">,
    Arg<"TCppIndex_t", "count", "1UL">
  ];
}

def Deallocate : CppInterOpAPI {
  let Doc = [{Deallocates memory for a given class.
\param[in] scope Class to indicate size of memory to deallocate
\param[in] count is used to indicate the number of objects to dallocate for}];

  let ReturnType = "void";
  let Args = [
    Arg<"TCppScope_t", "scope">,
    Arg<"TCppObject_t", "address">,
    Arg<"TCppIndex_t", "count", "1UL">
  ];
}

def IsAbstract : CppInterOpAPI {
  let ReturnType = "bool";
  let Args = [
    Arg<"TCppType_t", "klass">
  ];
}

def IsEnumScope : CppInterOpAPI {
  let Doc = "Checks if it is an enum name (EnumDecl represents an enum name).";

  let ReturnType = "bool";
  let Args = [
    Arg<"TCppScope_t", "handle">
  ];
}

def IsEnumConstant : CppInterOpAPI {
  let Doc = [{Checks if it is an enum's value (EnumConstantDecl represents
each enum constant that is defined).}];

  let ReturnType = "bool";
  let Args = [
    Arg<"TCppScope_t", "handle">
  ];
}

def IsAggregate : CppInterOpAPI {
  let Doc = [{Checks if the given class represents an aggregate type).
\returns true if \c scope is an array or a C++ tag (as per C++
[dcl.init.aggr]) \returns true if the scope supports aggregate
initialization.}];

  let ReturnType = "bool";
  let Args = [
    Arg<"TCppScope_t", "scope">
  ];
}

def IsVariable : CppInterOpAPI {
  let Doc = "Checks if the passed value is a variable.";

  let ReturnType = "bool";
  let Args = [
    Arg<"TCppScope_t", "scope">
  ];
}

def GetAllCppNames : CppInterOpAPI {
  let ReturnType = "void";
  let Args = [
    Arg<"TCppScope_t", "scope">,
    OutArg<"std::set<std::string>&", "names">
  ];
}

def GetUsingNamespaces : CppInterOpAPI {
  let Doc = "Gets the list of namespaces utilized in the supplied scope.";

  let ReturnType = "std::vector<TCppScope_t>";
  let Args = [
    Arg<"TCppScope_t", "scope">
  ];
}

def GetCompleteName : CppInterOpAPI {
  let Doc = [{This is similar to GetName() function, but besides
the name, it also gets the template arguments.}];

  let ReturnType = "std::string";
  let Args = [
    Arg<"TCppScope_t", "klass">
  ];
}

def GetDestructor : CppInterOpAPI {
  let Doc = "\\returns the class destructor, if any.";

  let ReturnType = "TCppFunction_t";
  let Args = [
    Arg<"TCppScope_t", "scope">
  ];
}

def IsVirtualMethod : CppInterOpAPI {
  let Doc = "Checks if the provided parameter is a 'Virtual' method.";

  let ReturnType = "bool";
  let Args = [
    Arg<"TCppFunction_t", "method">
  ];
}

def GetNumBases : CppInterOpAPI {
  let Doc = [{Gets the number of Base Classes for the Derived Class that
is passed as a parameter.}];

  let ReturnType = "TCppIndex_t";
  let Args = [
    Arg<"TCppScope_t", "klass">
  ];
}

def GetName : CppInterOpAPI {
  let Doc = [{Gets the name of any named decl (a class,
namespace, variable, or a function).}];

  let ReturnType = "std::string";
  let Args = [
    Arg<"TCppScope_t", "klass">
  ];
}

def GetBaseClass : CppInterOpAPI {
  let Doc = [{Gets a specific Base Class using its index. Typically GetNumBases()
is used to get the number of Base Classes, and then that number
can be used to iterate through the index value to get each specific
base class.}];

  let ReturnType = "TCppScope_t";
  let Args = [
    Arg<"TCppScope_t", "klass">,
    Arg<"TCppIndex_t", "ibase">
  ];
}

def IsSubclass : CppInterOpAPI {
  let Doc = [{Checks if the supplied Derived Class is a sub-class of the
provided Base Class.}];

  let ReturnType = "bool";
  let Args = [
    Arg<"TCppScope_t", "derived">,
    Arg<"TCppScope_t", "base">
  ];
}

def GetOperator : CppInterOpAPI {
  let Doc = "\\returns list of operator overloads";

  let ReturnType = "void";
  let Args = [
    Arg<"TCppScope_t", "scope">,
    Arg<"Operator", "op">,
    OutArg<"std::vector<TCppFunction_t>&", "operators">,
    Arg<"OperatorArity", "kind", "kBoth">
  ];
}

def GetBaseClassOffset : CppInterOpAPI {
  let Doc = [{Each base has its own offset in a Derived Class. This offset can be
used to get to the Base Class fields.}];

  let ReturnType = "int64_t";
  let Args = [
    Arg<"TCppScope_t", "derived">,
    Arg<"TCppScope_t", "base">
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
    Arg<"TCppScope_t", "klass">,
    OutArg<"std::vector<TCppFunction_t>&", "methods">
  ];
}

def IsFunctionProtoType : CppInterOpAPI {
  let Doc = "Check if given type is FunctionProtoType.";

  let ReturnType = "bool";
  let Args = [
    Arg<"TCppType_t", "typ">
  ];
}

def GetFnTypeSignature : CppInterOpAPI {
  let Doc = [{Pushed the signature type of the given function type,
  where the first item is the return type.}];

  let ReturnType = "void";
  let Args = [
    Arg<"TCppType_t", "fn_type">,
    OutArg<"std::vector<TCppType_t>&", "sig">
  ];
}

def GetFunctionNumArgs : CppInterOpAPI {
  let Doc = "Gets the number of Arguments for the provided function.";

  let ReturnType = "TCppIndex_t";
  let Args = [
    Arg<"TCppFunction_t", "func">
  ];
}

def GetFunctionArgName : CppInterOpAPI {
  let Doc = "\\returns the argument name of function as string.";

  let ReturnType = "std::string";
  let Args = [
    Arg<"TCppFunction_t", "func">,
    Arg<"TCppIndex_t", "param_index">
  ];
}

def GetFunctionArgType : CppInterOpAPI {
  let Doc = [{For each Argument of a function, you can get the Argument Type
by providing the Argument Index, based on the number of arguments
from the GetFunctionNumArgs() function.}];

  let ReturnType = "TCppType_t";
  let Args = [
    Arg<"TCppFunction_t", "func">,
    Arg<"TCppIndex_t", "iarg">
  ];
}

def IsConstMethod : CppInterOpAPI {
  let Doc = "Checks if a function declared is of const type or not.";

  let ReturnType = "bool";
  let Args = [
    Arg<"TCppFunction_t", "method">
  ];
}

def IsTemplatedFunction : CppInterOpAPI {
  let ReturnType = "bool";
  let Args = [
    Arg<"TCppFunction_t", "func">
  ];
}

def IsStaticMethod : CppInterOpAPI {
  let Doc = "Checks if the provided parameter is a 'Static' method.";

  let ReturnType = "bool";
  let Args = [
    Arg<"TCppConstFunction_t", "method">
  ];
}

def IsFunctionDeleted : CppInterOpAPI {
  let Doc = "\\returns if a function was marked as \\c =delete.";

  let ReturnType = "bool";
  let Args = [
    Arg<"TCppConstFunction_t", "function">
  ];
}

def IsPublicMethod : CppInterOpAPI {
  let Doc = "Checks if the provided parameter is a 'Public' method.";

  let ReturnType = "bool";
  let Args = [
    Arg<"TCppFunction_t", "method">
  ];
}

def IsProtectedMethod : CppInterOpAPI {
  let Doc = "Checks if the provided parameter is a 'Protected' method.";

  let ReturnType = "bool";
  let Args = [
    Arg<"TCppFunction_t", "method">
  ];
}

def IsPrivateMethod : CppInterOpAPI {
  let Doc = "Checks if the provided parameter is a 'Private' method.";

  let ReturnType = "bool";
  let Args = [
    Arg<"TCppFunction_t", "method">
  ];
}

def IsConstructor : CppInterOpAPI {
  let Doc = "Checks if the provided parameter is a Constructor.";

  let ReturnType = "bool";
  let Args = [
    Arg<"TCppConstFunction_t", "method">
  ];
}

def IsDestructor : CppInterOpAPI {
  let Doc = "Checks if the provided parameter is a Destructor.";

  let ReturnType = "bool";
  let Args = [
    Arg<"TCppConstFunction_t", "method">
  ];
}

def GetDatamembers : CppInterOpAPI {
  let Doc = "Gets all the Fields/Data Members of a Class";

  let ReturnType = "void";
  let Args = [
    Arg<"TCppScope_t", "scope">,
    OutArg<"std::vector<TCppScope_t>&", "datamembers">
  ];
}

def GetStaticDatamembers : CppInterOpAPI {
  let Doc = [{Gets all the Static Fields/Data Members of a Class
\param[in] scope - class
\param[out] funcs - vector of static data members}];

  let ReturnType = "void";
  let Args = [
    Arg<"TCppScope_t", "scope">,
    OutArg<"std::vector<TCppScope_t>&", "datamembers">
  ];
}

def LookupDatamember : CppInterOpAPI {
  let Doc = "This is a Lookup function to be used specifically for data members.";

  let ReturnType = "TCppScope_t";
  let Args = [
    Arg<"const std::string&", "name">,
    Arg<"TCppScope_t", "parent">
  ];
}

def IsLambdaClass : CppInterOpAPI {
  let Doc = "Check if the given type is a lamda class";

  let ReturnType = "bool";
  let Args = [
    Arg<"TCppType_t", "type">
  ];
}

def GetQualifiedName : CppInterOpAPI {
  let Doc = [{Gets the "qualified" name (including the namespace) of any
named decl (a class, namespace, variable, or a function).}];

  let ReturnType = "std::string";
  let Args = [
    Arg<"TCppScope_t", "klass">
  ];
}

def GetVariableOffset : CppInterOpAPI {
  let Doc = [{Gets the address of the variable, you can use it to get the
value stored in the variable.}];

  let ReturnType = "intptr_t";
  let Args = [
    Arg<"TCppScope_t", "var">,
    Arg<"TCppScope_t", "parent", "nullptr">
  ];
}

def IsPublicVariable : CppInterOpAPI {
  let Doc = "Checks if the provided variable is a 'Public' variable.";

  let ReturnType = "bool";
  let Args = [
    Arg<"TCppScope_t", "var">
  ];
}

def IsProtectedVariable : CppInterOpAPI {
  let Doc = "Checks if the provided variable is a 'Protected' variable.";

  let ReturnType = "bool";
  let Args = [
    Arg<"TCppScope_t", "var">
  ];
}

def IsPrivateVariable : CppInterOpAPI {
  let Doc = "Checks if the provided variable is a 'Private' variable.";

  let ReturnType = "bool";
  let Args = [
    Arg<"TCppScope_t", "var">
  ];
}

def IsStaticVariable : CppInterOpAPI {
  let Doc = "Checks if the provided variable is a 'Static' variable.";

  let ReturnType = "bool";
  let Args = [
    Arg<"TCppScope_t", "var">
  ];
}

def IsConstVariable : CppInterOpAPI {
  let Doc = "Checks if the provided variable is a 'Constant' variable.";

  let ReturnType = "bool";
  let Args = [
    Arg<"TCppScope_t", "var">
  ];
}

def GetDimensions : CppInterOpAPI {
  let Doc = "Gets the size/dimensions of a multi-dimension array.";
  // FIXME: Rework to use a C-compatible array type.
  let NoCWrapper = true;
  let ReturnType = "std::vector<long int>";
  let Args = [
    Arg<"TCppType_t", "type">
  ];
}

def GetEnumConstants : CppInterOpAPI {
  let Doc = "Gets a list of all the enum constants for an enum.";

  let ReturnType = "std::vector<TCppScope_t>";
  let Args = [
    Arg<"TCppScope_t", "scope">
  ];
}

def GetEnumConstantType : CppInterOpAPI {
  let Doc = "Gets the enum name when an enum constant is passed.";

  let ReturnType = "TCppType_t";
  let Args = [
    Arg<"TCppScope_t", "scope">
  ];
}

def GetEnumConstantValue : CppInterOpAPI {
  let Doc = [{Gets the index value (0,1,2, etcetera) of the enum constant
that was passed into this function.}];

  let ReturnType = "TCppIndex_t";
  let Args = [
    Arg<"TCppScope_t", "scope">
  ];
}

def DumpScope : CppInterOpAPI {
  let ReturnType = "void";
  let Args = [
    Arg<"TCppScope_t", "scope">
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
    Arg<"TCppScope_t", "scope">,
    Arg<"bool", "strip_comment_markers", "true">
  ];
}

def IsExplicit : CppInterOpAPI {
  let Doc = "Checks if the provided constructor or conversion operator is explicit";

  let ReturnType = "bool";
  let Args = [
    Arg<"TCppConstFunction_t", "method">
  ];
}


// --- Functions not previously in the dispatch table ---

def ActivateInterpreter : CppInterOpAPI {
  let Doc = [{Activates an instance of an interpreter to handle subsequent API
requests.}];
  let ReturnType = "bool";
  let Args = [Arg<"TInterp_t", "I">];
}

def AddTypeQualifier : CppInterOpAPI {
  let Doc = [{Returns type with the qual Qualifiers added.
qual can be ORed value of enum QualKind.}];
  let ReturnType = "TCppType_t";
  let Args = [
    Arg<"TCppType_t", "type">,
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
  let ReturnType = "TCppFunction_t";
  let Args = [
    Arg<"const std::vector<TCppFunction_t>&", "candidates">,
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
    Arg<"TCppScope_t", "parent", "nullptr">
  ];
}

def GetClassTemplateArgs : CppInterOpAPI {
  let Doc = [{Sets the class template arguments of \c templ_instance.
\param[in] templ_instance Pointer to the template instance.
\param[out] args Vector of instantiation arguments.}];
  let ReturnType = "void";
  let Args = [
    Arg<"TCppScope_t", "templ_instance">,
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
    Arg<"TCppScope_t", "templ_instance">,
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
    Arg<"TCppScope_t", "parent">,
    OutArg<"std::vector<TCppFunction_t>&", "funcs">
  ];
}

def GetDefaultConstructor : CppInterOpAPI {
  let Doc = "\\returns the default constructor of a class, if any.";
  let ReturnType = "TCppFunction_t";
  let Args = [Arg<"TCppScope_t", "scope">];
}

def GetEnumConstantDatamembers : CppInterOpAPI {
  let Doc = [{Gets all enum constant data members of a class.
\param[in] scope Class to query.
\param[out] datamembers Vector of enum constant members.
\param[in] include_enum_class Whether to include enum class members.}];
  let ReturnType = "void";
  let Args = [
    Arg<"TCppScope_t", "scope">,
    OutArg<"std::vector<TCppScope_t>&", "datamembers">,
    Arg<"bool", "include_enum_class", "true">
  ];
}

def GetEnums : CppInterOpAPI {
  let Doc = "Extracts enum declarations from a specified scope and stores them in a vector.";
  let ReturnType = "void";
  let Args = [
    Arg<"TCppScope_t", "scope">,
    OutArg<"std::vector<std::string>&", "Result">
  ];
}

def GetFunctionArgDefault : CppInterOpAPI {
  let Doc = "\\returns the default argument value as string.";
  let ReturnType = "std::string";
  let Args = [
    Arg<"TCppFunction_t", "func">,
    Arg<"TCppIndex_t", "param_index">
  ];
}

def GetFunctionRequiredArgs : CppInterOpAPI {
  let Doc = "Gets the number of Required Arguments for the provided function.";
  let ReturnType = "TCppIndex_t";
  let Args = [Arg<"TCppConstFunction_t", "func">];
}

def GetFunctionReturnType : CppInterOpAPI {
  let Doc = "Gets the return type of the provided function.";
  let ReturnType = "TCppType_t";
  let Args = [Arg<"TCppFunction_t", "func">];
}

def GetFunctionSignature : CppInterOpAPI {
  let Doc = [{\\returns a stringified version of a given function signature in the
form: void N::f(int i, double d, long l = 0, char ch = 'a').}];
  let ReturnType = "std::string";
  let Args = [Arg<"TCppFunction_t", "func">];
}

def GetFunctionTemplatedDecls : CppInterOpAPI {
  let Doc = [{Template function pointer list to add proxies for
un-instantiated/non-overloaded templated methods.}];
  let ReturnType = "void";
  let Args = [
    Arg<"TCppScope_t", "klass">,
    OutArg<"std::vector<TCppFunction_t>&", "methods">
  ];
}

def GetFunctionsUsingName : CppInterOpAPI {
  let Doc = [{Looks up all the functions that have the name that is
passed as a parameter in this function.}];
  let ReturnType = "std::vector<TCppFunction_t>";
  let Args = [
    Arg<"TCppScope_t", "scope">,
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
  let ReturnType = "TCppType_t";
  let Args = [Arg<"TCppScope_t", "handle">];
}

def GetIntegerTypeFromEnumType : CppInterOpAPI {
  let Doc = [{For the given "type", this function gets the integer type that
the enum represents.}];
  let ReturnType = "TCppType_t";
  let Args = [Arg<"TCppType_t", "handle">];
}

def GetOperatorArity : CppInterOpAPI {
  let Doc = "\\returns arity of the operator or kNone.";
  let ReturnType = "OperatorArity";
  let Args = [Arg<"TCppFunction_t", "op">];
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
  let Args = [Arg<"TCppScope_t", "klass">];
}

def GetScopeFromCompleteName : CppInterOpAPI {
  let Doc = [{When the namespace is known, then the parent doesn't need
to be specified. This will probably be phased-out in
future versions of the interop library.}];
  let ReturnType = "TCppScope_t";
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
  let Args = [Arg<"TCppScope_t", "scope">];
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
  let ReturnType = "TCppFunction_t";
  let Args = [Arg<"const char*", "function_template">];
}

def IsFunction : CppInterOpAPI {
  let Doc = "Checks if the scope is a function.";
  let ReturnType = "bool";
  let Args = [Arg<"TCppScope_t", "scope">];
}

def IsFunctionPointerType : CppInterOpAPI {
  let Doc = "Checks if the type is a function pointer.";
  let ReturnType = "bool";
  let Args = [Arg<"TCppType_t", "type">];
}

def IsMethod : CppInterOpAPI {
  let Doc = "Checks if the provided parameter is a method.";
  let ReturnType = "bool";
  let Args = [Arg<"TCppConstFunction_t", "method">];
}

def IsPODType : CppInterOpAPI {
  let Doc = "Checks if the provided parameter is a Plain Old Data Type (POD).";
  let ReturnType = "bool";
  let Args = [Arg<"TCppType_t", "type">];
}

def IsSmartPtrType : CppInterOpAPI {
  let Doc = [{We assume that smart pointer types define both operator* and
operator->.}];
  let ReturnType = "bool";
  let Args = [Arg<"TCppType_t", "type">];
}

def IsTemplateSpecialization : CppInterOpAPI {
  let Doc = "Checks if it is a class template specialization class.";
  let ReturnType = "bool";
  let Args = [Arg<"TCppScope_t", "handle">];
}

def IsTypeDerivedFrom : CppInterOpAPI {
  let Doc = "Checks if a C++ type derives from another.";
  let ReturnType = "bool";
  let Args = [
    Arg<"TCppType_t", "derived">,
    Arg<"TCppType_t", "base">
  ];
}

def LookupConstructors : CppInterOpAPI {
  let Doc = [{Sets a list of all the constructor for a scope/class that is
supplied as a parameter.}];
  let ReturnType = "void";
  let Args = [
    Arg<"const std::string&", "name">,
    Arg<"TCppScope_t", "parent">,
    OutArg<"std::vector<TCppFunction_t>&", "funcs">
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
    Arg<"TInterp_t", "I">,
    Arg<"TCppConstFunction_t", "func">
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
  let Args = [Arg<"TInterp_t", "I">];
}

def MakeFunctionCallable : CppInterOpAPI {
  let Doc = [{Creates a trampoline function by using the interpreter and returns a
uniform interface to call it from compiled code.}];
  // JitCall is a C++ class; no mechanical C mapping.
  let NoCWrapper = true;
  let ReturnType = "JitCall";
  let Args = [Arg<"TCppConstFunction_t", "func">];
}

def IsIntegerType : CppInterOpAPI {
  let Doc = [{Checks if type has an integer representation.
If \p s is non-null, it is set to the signedness of the type.}];
  let ReturnType = "bool";
  let Args = [
    Arg<"TCppType_t", "type">,
    Arg<"Signedness*", "s", "nullptr">
  ];
}

def IsFloatingType : CppInterOpAPI {
  let Doc = "Checks if type has a floating representation.";
  let ReturnType = "bool";
  let Args = [Arg<"TCppType_t", "type">];
}

def IsSameType : CppInterOpAPI {
  let Doc = [{Checks if two types are the equivalent,
i.e. have the same canonical type.}];
  let ReturnType = "bool";
  let Args = [
    Arg<"TCppType_t", "type_a">,
    Arg<"TCppType_t", "type_b">
  ];
}

def IsVoidPointerType : CppInterOpAPI {
  let Doc = "Checks if type is a void pointer.";
  let ReturnType = "bool";
  let Args = [Arg<"TCppType_t", "type">];
}

// Trace-hook impls invoked by JitCall's inline fast path through
// ::CppInternal::DispatchRaw slots. Routing through the X-macro means
// Dispatch.h consumers pick up the slots via LoadDispatchAPI without
// any source change of their own.
def CppInterOpTraceJitCallInvokeImpl : CppInterOpAPI {
  let Doc = "Trace hook for the JitCall::Invoke generic call path.";
  let ReturnType = "void";
  let Args = [
    Arg<"const CppImpl::JitCall*", "JC">,
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
    Arg<"const CppImpl::JitCall*", "JC">,
    Arg<"void*", "object">,
    Arg<"unsigned long", "nary">,
    Arg<"int", "withFree">
  ];
}

def CppInterOpTraceJitCallInvokeReturnImpl : CppInterOpAPI {
  let Doc = "Trace hook for post-Invoke pointer-return handle registration.";
  let ReturnType = "void";
  let Args = [
    Arg<"const CppImpl::JitCall*", "JC">,
    Arg<"void*", "result">
  ];
}
