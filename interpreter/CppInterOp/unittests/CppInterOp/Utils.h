#ifndef CPPINTEROP_UNITTESTS_LIBCPPINTEROP_UTILS_H
#define CPPINTEROP_UNITTESTS_LIBCPPINTEROP_UTILS_H

#include "../../lib/CppInterOp/Compatibility.h"

#include "clang-c/CXCppInterOp.h"
#include "clang-c/CXString.h"

#include "llvm/Support/Valgrind.h"

#include <memory>
#include <string>
#include <vector>

using namespace clang;
using namespace llvm;

namespace clang {
  class Decl;
}
#define Interp (static_cast<compat::Interpreter*>(Cpp::GetInterpreter()))
namespace TestUtils {
void GetAllTopLevelDecls(const std::string& code,
                         std::vector<clang::Decl*>& Decls,
                         bool filter_implicitGenerated = false,
                         const std::vector<const char*>& interpreter_args = {});
void GetAllSubDecls(clang::Decl* D, std::vector<clang::Decl*>& SubDecls,
                    bool filter_implicitGenerated = false);
} // end namespace TestUtils

const char* get_c_string(CXString string);

void dispose_string(CXString string);

CXScope make_scope(const clang::Decl* D, const CXInterpreter I);

#endif // CPPINTEROP_UNITTESTS_LIBCPPINTEROP_UTILS_H
