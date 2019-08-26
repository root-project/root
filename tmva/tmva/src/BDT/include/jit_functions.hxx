#ifndef __JIT_FUNCTIONS_HXX_
#define __JIT_FUNCTIONS_HXX_
//#include "unique_bdt.h"

#include "TInterpreter.h" // for gInterpreter
#include "bdt_helpers.hxx"

//////////////////////////////////////////////////////
/// JITTING FUNCTIONS
//////////////////////////////////////////////////////

/// JIT forest from sringed code
template <typename T>
std::function<bool(const T *event)> jit_forest(const std::string &tojit, const std::string s_namespace = "")
{
   gInterpreter->Declare(tojit.c_str());
   bool use_namespaces = (!s_namespace.empty());

   std::string func_ref_name;
   if (use_namespaces)
      func_ref_name = "#pragma cling optimize(3)\n & jitted_" + s_namespace + "::generated_forest";
   else
      func_ref_name = "#pragma cling optimize(3)\n & generated_forest";
   auto ptr                = gInterpreter->Calc(func_ref_name.c_str());
   bool (*func)(const T *) = reinterpret_cast<bool (*)(const T *)>(ptr);
   std::function<bool(const T *)> fWrapped{func};
   return fWrapped;
}

#endif
