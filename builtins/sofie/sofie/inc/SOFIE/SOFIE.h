/// \file SOFIE.h
/// C interface of the SOFIE library.
///
/// The interface only uses C types, so that the library can be loaded at
/// runtime (for example with dlopen) and its functions can be called by
/// looking up their symbols, without including this header or linking
/// against the library.

#ifndef SOFIE_SOFIE_H
#define SOFIE_SOFIE_H

#include <stddef.h>

#if defined(_WIN32)
#if defined(SOFIE_BUILDING_LIBRARY)
#define SOFIE_EXPORT __declspec(dllexport)
#else
#define SOFIE_EXPORT __declspec(dllimport)
#endif
#else
#define SOFIE_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/// Flags of SOFIE_GenerateCodeFromONNX, which can be combined with a bitwise or.
enum {
   /// Call the routines of the BLAS library loaded in the process, instead of
   /// defining them in the generated code.
   SOFIE_USE_BLAS = 1u << 0,
   /// Use the vdt functions for the exponential and the logarithm, if the vdt
   /// headers are found when the generated code is compiled.
   SOFIE_USE_VDT = 1u << 1
};

/// Generate the C++ code to evaluate an ONNX model.
///
/// The code is placed in the namespace SOFIE_<modelName>, and the weights of
/// the model are part of the code.
///
/// \param[in] onnx The content of the ONNX file.
/// \param[in] size The size of the content of the ONNX file, in bytes.
/// \param[in] modelName The name of the model, which must be a valid C++ identifier.
/// \param[in] flags A combination of the SOFIE_USE_* flags, or 0.
/// \param[out] code The generated code, a null-terminated string to release with SOFIE_Free.
///                  It is set to NULL if the code can't be generated.
/// \param[out] error The error message, a null-terminated string to release with SOFIE_Free.
///                   It is set to NULL if the code is generated. It can be NULL.
/// \return The length of the generated code, without the terminating null character, or 0 if the
///         code can't be generated. The generated code is never empty.
SOFIE_EXPORT size_t SOFIE_GenerateCodeFromONNX(const char *onnx, size_t size, const char *modelName, unsigned int flags,
                                               char **code, char **error);

/// Release a string returned by the functions of the library. It does nothing if `str` is NULL.
SOFIE_EXPORT void SOFIE_Free(char *str);

#ifdef __cplusplus
}
#endif

#endif // SOFIE_SOFIE_H
