// NOLINTBEGIN()
#ifndef LLVM_CLANG_C_CXCPPINTEROP_H
#define LLVM_CLANG_C_CXCPPINTEROP_H

#include "clang-c/CXErrorCode.h"
#include "clang-c/CXString.h"
#include "clang-c/ExternC.h"
#include "clang-c/Index.h"
#include "clang-c/Platform.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

LLVM_CLANG_C_EXTERN_C_BEGIN

/**
 * \defgroup CPPINTEROP_INTERPRETER_MANIP Interpreter manipulations
 *
 * @{
 */

/**
 * An opaque pointer representing an interpreter context.
 */
typedef struct CXInterpreterImpl* CXInterpreter;

/**
 * Create a Clang interpreter instance from the given arguments.
 *
 * \param argv The arguments that would be passed to the interpreter.
 *
 * \param argc The number of arguments in \c argv.
 *
 * \returns a \c CXInterpreter.
 */
CINDEX_LINKAGE CXInterpreter clang_createInterpreter(const char* const* argv,
                                                     int argc);

typedef void* TInterp_t;

/**
 * Bridge between C API and C++ API.
 *
 * \returns a \c CXInterpreter.
 */
CINDEX_LINKAGE CXInterpreter clang_createInterpreterFromRawPtr(TInterp_t I);

/**
 * Returns a pointer to the underlying interpreter.
 */
CINDEX_LINKAGE void* clang_Interpreter_getClangInterpreter(CXInterpreter I);

/**
 * Returns a \c TInterp_t and takes the ownership.
 */
CINDEX_LINKAGE TInterp_t
clang_Interpreter_takeInterpreterAsPtr(CXInterpreter I);

/**
 * Undo N previous incremental inputs.
 */
CINDEX_LINKAGE enum CXErrorCode clang_Interpreter_undo(CXInterpreter I,
                                                       unsigned int N);

/**
 * Dispose of the given interpreter context.
 */
CINDEX_LINKAGE void clang_Interpreter_dispose(CXInterpreter I);

/**
 * Describes the return result of the different routines that do the incremental
 * compilation.
 */
typedef enum {
  /**
   * The compilation was successful.
   */
  CXInterpreter_Success = 0,
  /**
   * The compilation failed.
   */
  CXInterpreter_Failure = 1,
  /**
   * More more input is expected.
   */
  CXInterpreter_MoreInputExpected = 2,
} CXInterpreter_CompilationResult;

/**
 * Enum to represent the programming language of the interpreter.
 */
typedef enum {
  CXInterpreterLanguage_Unknown,
  CXInterpreterLanguage_Asm,
  CXInterpreterLanguage_CIR,
  CXInterpreterLanguage_LLVM_IR,
  CXInterpreterLanguage_C,
  CXInterpreterLanguage_CPlusPlus,
  CXInterpreterLanguage_ObjC,
  CXInterpreterLanguage_ObjCPlusPlus,
  CXInterpreterLanguage_OpenCL,
  CXInterpreterLanguage_OpenCLCXX,
  CXInterpreterLanguage_CUDA,
  CXInterpreterLanguage_HIP,
  CXInterpreterLanguage_HLSL
} CXInterpreterLanguage;

/**
 * Enum to represent the language standard of the interpreter.
 */
typedef enum {
  CXInterpreterLanguageStandard_c89,
  CXInterpreterLanguageStandard_c94,
  CXInterpreterLanguageStandard_gnu89,
  CXInterpreterLanguageStandard_c99,
  CXInterpreterLanguageStandard_gnu99,
  CXInterpreterLanguageStandard_c11,
  CXInterpreterLanguageStandard_gnu11,
  CXInterpreterLanguageStandard_c17,
  CXInterpreterLanguageStandard_gnu17,
  CXInterpreterLanguageStandard_c23,
  CXInterpreterLanguageStandard_gnu23,
  CXInterpreterLanguageStandard_c2y,
  CXInterpreterLanguageStandard_gnu2y,
  CXInterpreterLanguageStandard_cxx98,
  CXInterpreterLanguageStandard_gnucxx98,
  CXInterpreterLanguageStandard_cxx11,
  CXInterpreterLanguageStandard_gnucxx11,
  CXInterpreterLanguageStandard_cxx14,
  CXInterpreterLanguageStandard_gnucxx14,
  CXInterpreterLanguageStandard_cxx17,
  CXInterpreterLanguageStandard_gnucxx17,
  CXInterpreterLanguageStandard_cxx20,
  CXInterpreterLanguageStandard_gnucxx20,
  CXInterpreterLanguageStandard_cxx23,
  CXInterpreterLanguageStandard_gnucxx23,
  CXInterpreterLanguageStandard_cxx26,
  CXInterpreterLanguageStandard_gnucxx26,
  CXInterpreterLanguageStandard_opencl10,
  CXInterpreterLanguageStandard_opencl11,
  CXInterpreterLanguageStandard_opencl12,
  CXInterpreterLanguageStandard_opencl20,
  CXInterpreterLanguageStandard_opencl30,
  CXInterpreterLanguageStandard_openclcpp10,
  CXInterpreterLanguageStandard_openclcpp2021,
  CXInterpreterLanguageStandard_hlsl,
  CXInterpreterLanguageStandard_hlsl2015,
  CXInterpreterLanguageStandard_hlsl2016,
  CXInterpreterLanguageStandard_hlsl2017,
  CXInterpreterLanguageStandard_hlsl2018,
  CXInterpreterLanguageStandard_hlsl2021,
  CXInterpreterLanguageStandard_hlsl202x,
  CXInterpreterLanguageStandard_hlsl202y,
  CXInterpreterLanguageStandard_lang_unspecified
} CXInterpreterLanguageStandard;

/**
 * Add a search path to the interpreter.
 *
 * \param I The interpreter.
 *
 * \param dir The directory to add.
 *
 * \param isUser Whether the directory is a user directory.
 *
 * \param prepend Whether to prepend the directory to the search path.
 */
CINDEX_LINKAGE void clang_Interpreter_addSearchPath(CXInterpreter I,
                                                    const char* dir,
                                                    bool isUser, bool prepend);

/**
 * Add an include path.
 *
 * \param I The interpreter.
 *
 * \param dir The directory to add.
 */
CINDEX_LINKAGE void clang_Interpreter_addIncludePath(CXInterpreter I,
                                                     const char* dir);

/**
 * Declares a code snippet in \c code and does not execute it.
 *
 * \param I The interpreter.
 *
 * \param code The code snippet to declare.
 *
 * \param silent Whether to suppress the diagnostics or not
 *
 * \returns a \c CXErrorCode.
 */
CINDEX_LINKAGE enum CXErrorCode
clang_Interpreter_declare(CXInterpreter I, const char* code, bool silent);

/**
 * Declares and executes a code snippet in \c code.
 *
 * \param I The interpreter.
 *
 * \param code The code snippet to execute.
 *
 * \returns a \c CXErrorCode.
 */
CINDEX_LINKAGE enum CXErrorCode clang_Interpreter_process(CXInterpreter I,
                                                          const char* code);

/**
 * An opaque pointer representing a lightweight struct that is used for carrying
 * execution results.
 */
typedef void* CXValue;

/**
 * Create a CXValue.
 *
 * \returns a \c CXValue.
 */
CINDEX_LINKAGE CXValue clang_createValue(void);

/**
 * Dispose of the given CXValue.
 *
 * \param V The CXValue to dispose.
 */
CINDEX_LINKAGE void clang_Value_dispose(CXValue V);

/**
 * Declares, executes and stores the execution result to \c V.
 *
 * \param[in] I The interpreter.
 *
 * \param[in] code The code snippet to evaluate.
 *
 * \param[out] V The value to store the execution result.
 *
 * \returns a \c CXErrorCode.
 */
CINDEX_LINKAGE enum CXErrorCode
clang_Interpreter_evaluate(CXInterpreter I, const char* code, CXValue V);

/**
 * Looks up the library if access is enabled.
 *
 * \param I The interpreter.
 *
 * \param lib_name The name of the library to lookup.
 *
 * \returns the path to the library.
 */
CINDEX_LINKAGE CXString clang_Interpreter_lookupLibrary(CXInterpreter I,
                                                        const char* lib_name);

/**
 * Finds \c lib_stem considering the list of search paths and loads it by
 * calling dlopen.
 *
 * \param I The interpreter.
 *
 * \param lib_stem The stem of the library to load.
 *
 * \param lookup Whether to lookup the library or not.
 *
 * \returns a \c CXInterpreter_CompilationResult.
 */
CINDEX_LINKAGE CXInterpreter_CompilationResult clang_Interpreter_loadLibrary(
    CXInterpreter I, const char* lib_stem, bool lookup);

/**
 * Finds \c lib_stem considering the list of search paths and unloads it by
 * calling dlclose.
 *
 * \param I The interpreter.
 *
 * \param lib_stem The stem of the library to unload.
 */
CINDEX_LINKAGE void clang_Interpreter_unloadLibrary(CXInterpreter I,
                                                    const char* lib_stem);

/**
 * Returns the programming language of the interpreter.
 *
 * \param I The interpreter.
 *
 * \returns CXInterpreterLanguage value.
 */
CINDEX_LINKAGE CXInterpreterLanguage
clang_Interpreter_getLanguage(CXInterpreter I);

/**
 * Returns the language standard of the interpreter.
 *
 * \param I The interpreter.
 *
 * \returns CXInterpreterLanguageStandard value.
 */
CINDEX_LINKAGE CXInterpreterLanguageStandard
clang_Interpreter_getLanguageStandard(CXInterpreter I);

/**
 * @}
 */

/**
 * \defgroup CPPINTEROP_SCOPE_MANIP Scope manipulations
 *
 * @{
 */

/**
 * A fake CXCursor for working with the interpreter.
 * It has the same structure as CXCursor, but unlike CXCursor, it stores a
 * handle to the interpreter in the third slot of the data field.
 * This pave the way for upstreaming features to the LLVM project.
 */
typedef struct {
  enum CXCursorKind kind;
  int xdata;
  const void* data[3];
} CXScope;

// for debugging purposes
CINDEX_LINKAGE void clang_scope_dump(CXScope S);

/**
 * Checks if a class has a default constructor.
 */
CINDEX_LINKAGE bool clang_hasDefaultConstructor(CXScope S);

/**
 * Returns the default constructor of a class, if any.
 */
CINDEX_LINKAGE CXScope clang_getDefaultConstructor(CXScope S);

/**
 * Returns the class destructor, if any.
 */
CINDEX_LINKAGE CXScope clang_getDestructor(CXScope S);

/**
 * Returns a stringified version of a given function signature in the form:
 * void N::f(int i, double d, long l = 0, char ch = 'a').
 */
CINDEX_LINKAGE CXString clang_getFunctionSignature(CXScope func);

/**
 * Returns the Doxygen documentation comment for a declaration, or an empty
 * string if no documentation comment exists.
 */
CINDEX_LINKAGE CXString clang_getDoxygenComment(CXScope S,
                                                bool strip_comment_markers);

/**
 * Checks if a function is a templated function.
 */
CINDEX_LINKAGE bool clang_isTemplatedFunction(CXScope func);

/**
 * This function performs a lookup to check if there is a templated function of
 * that type. \c parent is mandatory, the global scope should be used as the
 * default value.
 */
CINDEX_LINKAGE bool clang_existsFunctionTemplate(const char* name,
                                                 CXScope parent);

typedef struct {
  void* Type;
  const char* IntegralValue;
} CXTemplateArgInfo;

/**
 * Builds a template instantiation for a given templated declaration.
 * Offers a single interface for instantiation of class, function and variable
 * templates.
 *
 * \param[in] tmpl The uninstantiated template class/function.
 *
 * \param[in] template_args The pointer to vector of template arguments stored
 * in the \c TemplateArgInfo struct
 *
 * \param[in] template_args_size The size of the vector of template arguments
 * passed as \c template_args
 *
 * \returns a \c CXScope representing the instantiated templated
 * class/function/variable.
 */
CINDEX_LINKAGE CXScope clang_instantiateTemplate(
    CXScope tmpl, CXTemplateArgInfo* template_args, size_t template_args_size);

/**
 * A fake CXType for working with the interpreter.
 * It has the same structure as CXType, but unlike CXType, it stores a
 * handle to the interpreter in the second slot of the data field.
 */
typedef struct {
  enum CXTypeKind kind;
  void* data[2];
} CXQualType;

/**
 * Gets the string of the type that is passed as a parameter.
 */
CINDEX_LINKAGE CXString clang_getTypeAsString(CXQualType type);

/**
 * Returns the complex of the provided type.
 */
CINDEX_LINKAGE CXQualType clang_getComplexType(CXQualType eltype);

/**
 * An opaque pointer representing the object of a given type (\c CXScope).
 */
typedef void* CXObject;

/**
 * Allocates memory for the given type.
 */
CINDEX_LINKAGE CXObject clang_allocate(unsigned int n);

/**
 * Deallocates memory for a given class.
 */
CINDEX_LINKAGE void clang_deallocate(CXObject address);

/**
 * Creates an object of class \c scope and calls its default constructor. If \c
 * arena is set it uses placement new.
 */
CINDEX_LINKAGE CXObject clang_construct(CXScope scope, void* arena,
                                        size_t count = 1UL);

/**
 * Creates a trampoline function and makes a call to a generic function or
 * method.
 *
 * \param func The function or method to call.
 *
 * \param result The location where the return result will be placed.
 *
 * \param args The arguments to pass to the invocation.
 *
 * \param n The number of arguments.
 *
 * \param self The 'this pointer' of the object.
 */
CINDEX_LINKAGE void clang_invoke(CXScope func, void* result, void** args,
                                 size_t n, void* self);

/**
 * Calls the destructor of object of type \c type. When withFree is true it
 * calls operator delete/free.
 *
 * \param This The object to destruct.
 *
 * \param type The type of the object.
 *
 * \param withFree Whether to call operator delete/free or not.
 *
 * \returns true if wrapper generation and invocation succeeded.
 */
CINDEX_LINKAGE bool clang_destruct(CXObject This, CXScope S,
                                   bool withFree = true, size_t nary = 0UL);

/**
 * @}
 */

LLVM_CLANG_C_EXTERN_C_END

#endif // LLVM_CLANG_C_CXCPPINTEROP_H
       // NOLINTEND()
