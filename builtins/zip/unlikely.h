#ifndef R__unlikely_h
#define R__unlikely_h

/*---- unlikely / likely expressions -----------------------------------------*/
// These are meant to use in cases like:
//   if (R__unlikely(expression)) { ... }
// in performance-critical sections.  R__unlikely / R__likely provide hints to
// the compiler code generation to heavily optimize one side of a conditional,
// causing the other branch to have a heavy performance cost.
//
// It is best to use this for conditionals that test for rare error cases or
// backward compatibility code.

#if (__GNUC__ >= 3) || defined(__INTEL_COMPILER)
#if !defined(R__unlikely)
#define R__unlikely(expr) __builtin_expect(!!(expr), 0)
#endif
#if !defined(R__likely)
#define R__likely(expr) __builtin_expect(!!(expr), 1)
#endif
#else
#define R__unlikely(expr) expr
#define R__likely(expr) expr
#endif
#endif
