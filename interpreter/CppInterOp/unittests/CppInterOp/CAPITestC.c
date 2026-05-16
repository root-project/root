/* Pure C compilation test for the generated cppinterop_* C API.
 * This file verifies that CppInterOpTypes.h and ALL generated function
 * declarations in CXCppInterOp.inc are valid C. If any generated
 * signature uses a C++-only type, this file will fail to compile.
 */

#include "CppInterOp/CppInterOpTypes.h"

/* Pull in ALL generated C declarations. Every generated function
 * signature is compiled as C here — this is the primary check.
 */
#include "CppInterOp/CXCppInterOpDecl.inc"

/* Verify the C-compatible structs work without the 'struct' keyword
 * (thanks to the typedef).
 */
static void test_types(void) {
  CppInterOpArray arr;
  arr.data = (void**)0;
  arr.size = 0;
  (void)arr;

  CppInterOpStringArray sarr;
  sarr.data = (char**)0;
  sarr.size = 0;
  (void)sarr;

  TemplateArgInfo tai;
  tai.m_Type = (void*)0;
  tai.m_IntegralValue = (const char*)0;
  (void)tai;
}

/* Take the address of representative C API functions to verify they
 * are valid C symbols. The full set is already compiled above; these
 * confirm linkability at the object level.
 */
int capi_c_test_main(void) {
  test_types();
  (void)&cppinterop_CreateInterpreter;
  (void)&cppinterop_Declare;
  (void)&cppinterop_Process;
  (void)&cppinterop_IsClass;
  (void)&cppinterop_GetName;
  (void)&cppinterop_GetEnumConstants;
  (void)&cppinterop_DisposeArray;
  (void)&cppinterop_GetEnums;
  (void)&cppinterop_DisposeStringArray;
  (void)&cppinterop_InstantiateTemplate;
  (void)&cppinterop_HasTypeQualifier;
  (void)&cppinterop_GetLanguage;
  return 0;
}
