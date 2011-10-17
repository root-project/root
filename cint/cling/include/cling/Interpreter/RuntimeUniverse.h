//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//------------------------------------------------------------------------------
#ifndef __CLING__
#error "This file must not be included by compiled programs."
#endif

#ifdef CLING_RUNTIME_UNIVERSE_H
#error "CLING_RUNTIME_UNIVERSE_H Must only include once."
#endif

#define CLING_RUNTIME_UNIVERSE_H

#ifndef __STDC_LIMIT_MACROS
#define __STDC_LIMIT_MACROS // needed by System/DataTypes.h
#endif

#ifndef __STDC_CONSTANT_MACROS
#define __STDC_CONSTANT_MACROS // needed by System/DataTypes.h
#endif

#ifdef __cplusplus

namespace cling {

  class Interpreter;

  /// \brief Used to stores the declarations, which are going to be 
  /// available only at runtime. These are cling runtime builtins
  namespace runtime {

    /// \brief The interpreter provides itself as a builtin, i.e. it
    /// interprets itself. This is particularly important for implementing
    /// the dynamic scopes and the runtime bindings
    Interpreter* gCling = 0;

    namespace internal {
      /// \brief Some of clang's routines rely on valid source locations and 
      /// source ranges. This member can be looked up and source locations and
      /// ranges can be passed in as parameters to these routines.
      ///
      /// Use instead of SourceLocation() and SourceRange(). This might help,
      /// when clang emits diagnostics on artificially inserted AST node.
      int InterpreterGeneratedCodeDiagnosticsMaybeIncorrect;

      // Implemented in Interpreter.cpp
      int local_cxa_atexit(void (*func) (void*), void* arg,
                           void* dso, cling::Interpreter* interp);

      // Force the module to define __cxa_atexit, we need it.
      struct __trigger__cxa_atexit {
        ~__trigger__cxa_atexit(); // implemented in Interpreter.cpp
      } S;
    } // end namespace internal
  } // end namespace runtime
} // end namespace cling

using namespace cling::runtime;

// Global d'tors only for C++:
extern "C"
int cling_cxa_atexit(void (*func) (void*), void* arg, void* dso) {
  return cling::runtime::internal::local_cxa_atexit(func, arg, dso, gCling);
}

#endif // __cplusplus
