/*
 * Project: RooFit
 * Authors:
 *   Garima Singh, CERN 2023
 *   Jonas Rembser, CERN 2023
 *
 * Copyright (c) 2023, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#include <RooFit/CodegenContext.h>
#include <RooAbsArg.h>

#include "RooFitImplHelpers.h"

#include <TInterpreter.h>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <type_traits>
#include <unordered_map>

namespace {

bool startsWith(std::string_view str, std::string_view prefix)
{
   return str.size() >= prefix.size() && 0 == str.compare(0, prefix.size(), prefix);
}

} // namespace

namespace RooFit {
namespace Experimental {

/// @brief Adds (or overwrites) the string representing the result of a node.
/// @param key The name of the node to add the result for.
/// @param value The new name to assign/overwrite.
void CodegenContext::addResult(const char *key, std::string const &value)
{
   const TNamed *namePtr = RooNameReg::known(key);
   if (namePtr)
      addResult(namePtr, value);
}

void CodegenContext::addResult(TNamed const *key, std::string const &value)
{
   _nodeNames[key] = value;
}

/// @brief Gets the result for the given node using the node name. This node also performs the necessary
/// code generation through recursive calls to 'translate'. A call to this function modifies the already
/// existing code body.
/// @param key The node to get the result string for.
/// @return String representing the result of this node.
std::string const &CodegenContext::getResult(RooAbsArg const &arg)
{
   // If the result has already been recorded, just return the result.
   // It is usually the responsibility of each translate function to assign
   // the proper result to its class. Hence, if a result has already been recorded
   // for a particular node, it means the node has already been 'translate'd and we
   // dont need to visit it again.
   auto found = _nodeNames.find(arg.namePtr());
   if (found != _nodeNames.end())
      return found->second;

   // The result for vector observables should already be in the map if you
   // opened the loop scope. This is just to check if we did not request the
   // result of a vector-valued observable outside of the scope of a loop.
   auto foundVecObs = _vecObsIndices.find(arg.namePtr());
   if (foundVecObs != _vecObsIndices.end()) {
      throw std::runtime_error("You requested the result of a vector observable outside a loop scope for it!");
   }

   auto RAII(OutputScopeRangeComment(&arg));

   // Now, recursively call translate into the current argument to load the correct result.
   codegen(const_cast<RooAbsArg &>(arg), *this);

   return _nodeNames.at(arg.namePtr());
}

/// @brief Adds the given string to the string block that will be emitted at the top of the squashed function. Useful
/// for variable declarations.
/// @param str The string to add to the global scope.
void CodegenContext::addToGlobalScope(std::string const &str)
{
   // Introduce proper indentation for multiline strings.
   _code[0] += str;
}

/// @brief Since the squashed code represents all observables as a single flattened array, it is important
/// to keep track of the start index for a vector valued observable which can later be expanded to access the correct
/// element. For example, a vector valued variable x with 10 entries will be squashed to obs[start_idx + i].
/// @param key The name of the node representing the vector valued observable.
/// @param idx The start index (or relative position of the observable in the set of all observables).
void CodegenContext::addVecObs(const char *key, int idx)
{
   const TNamed *namePtr = RooNameReg::known(key);
   if (namePtr)
      _vecObsIndices[namePtr] = idx;
}

int CodegenContext::observableIndexOf(RooAbsArg const &arg) const
{
   auto it = _vecObsIndices.find(arg.namePtr());
   if (it != _vecObsIndices.end()) {
      return it->second;
   }

   return -1; // Not found
}
/// @brief Adds the input string to the squashed code body. If a class implements a translate function that wants to
/// emit something to the squashed code body, it must call this function with the code it wants to emit. In case of
/// loops, automatically determines if code needs to be stored inside or outside loop scope.
/// @param klass The class requesting this addition, usually 'this'.
/// @param in String to add to the squashed code.
void CodegenContext::addToCodeBody(RooAbsArg const *klass, std::string const &in)
{
   // If we are in a loop and the value is scope independent, save it at the top of the loop.
   // else, just save it in the current scope.
   addToCodeBody(in, isScopeIndependent(klass));
}

/// @brief A variation of the previous addToCodeBody that takes in a bool value that determines
/// if input is independent. This overload exists because there might other ways to determine if
/// a value/collection of values is scope independent.
/// @param in String to add to the squashed code.
/// @param isScopeIndep The value determining if the input is scope dependent.
void CodegenContext::addToCodeBody(std::string const &in, bool isScopeIndep /* = false */)
{
   TString indented = in;
   indented = indented.Strip(TString::kBoth); // trim

   std::string indent_str = "";
   for (unsigned i = 0; i < _indent; ++i)
      indent_str += "  ";
   indented = indented.Prepend(indent_str);

   // FIXME: Multiline input.
   // indent_str += "\n";
   // indented = indented.ReplaceAll("\n", indent_str);

   // If we are in a loop and the value is scope independent, save it at the top of the loop.
   // else, just save it in the current scope.
   if (_code.size() > 2 && isScopeIndep) {
      _code[_code.size() - 2] += indented;
   } else {
      _code.back() += indented;
   }
}

/// @brief Create a RAII scope for iterating over vector observables. You can't use the result of vector observables
/// outside these loop scopes.
/// @param in A pointer to the calling class, used to determine the loop dependent variables.
std::unique_ptr<CodegenContext::LoopScope> CodegenContext::beginLoop(RooAbsArg const *in)
{
   pushScope();
   unsigned loopLevel = _code.size() - 2; // subtract global + function scope.
   std::string idx = "loopIdx" + std::to_string(loopLevel);

   std::vector<TNamed const *> vars;
   // set the results of the vector observables
   for (auto const &it : _vecObsIndices) {
      if (!in->dependsOn(it.first))
         continue;

      vars.push_back(it.first);
      _nodeNames[it.first] = "obs[" + std::to_string(it.second) + " + " + idx + "]";
   }

   // TODO: we are using the size of the first loop variable to the the number
   // of iterations, but it should be made sure that all loop vars are either
   // scalar or have the same size.
   std::size_t numEntries = 1;
   for (auto &it : vars) {
      std::size_t n = outputSize(it);
      if (n > 1 && numEntries > 1 && n != numEntries) {
         throw std::runtime_error("Trying to loop over variables with different sizes!");
      }
      numEntries = std::max(n, numEntries);
   }

   // Make sure that the name of this variable doesn't clash with other stuff
   addToCodeBody(in, "for(int " + idx + " = 0; " + idx + " < " + std::to_string(numEntries) + "; " + idx + "++) {\n");

   return std::make_unique<LoopScope>(*this, std::move(vars));
}

void CodegenContext::endLoop(LoopScope const &scope)
{
   addToCodeBody("}\n");

   // clear the results of the loop variables if they were vector observables
   for (auto const &ptr : scope.vars()) {
      if (_vecObsIndices.find(ptr) != _vecObsIndices.end())
         _nodeNames.erase(ptr);
   }
   popScope();
}

/// @brief Get a unique variable name to be used in the generated code.
std::string CodegenContext::getTmpVarName() const
{
   return "t" + std::to_string(_tmpVarIdx++);
}

/// @brief A function to save an expression that includes/depends on the result of the input node.
/// @param in The node on which the valueToSave depends on/belongs to.
/// @param valueToSave The actual string value to save as a temporary.
void CodegenContext::addResult(RooAbsArg const *in, std::string const &valueToSave)
{
   // std::string savedName = RooFit::Detail::makeValidVarName(in->GetName());
   std::string savedName = getTmpVarName();

   // Only save values if they contain operations.
   bool hasOperations = valueToSave.find_first_of(":-+/*") != std::string::npos;

   // If the name is not empty and this value is worth saving, save it to the correct scope.
   // otherwise, just return the actual value itself
   if (hasOperations) {
      // If this is a scalar result, it will go just outside the loop because
      // it doesn't need to be recomputed inside loops.
      std::string outVarDecl = "const double " + savedName + " = " + valueToSave + ";\n";
      addToCodeBody(in, outVarDecl);
   } else {
      savedName = valueToSave;
   }

   addResult(in->namePtr(), savedName);
}

/// @brief Function to save a RooListProxy as an array in the squashed code.
/// @param in The list to convert to array.
/// @return Name of the array that stores the input list in the squashed code.
std::string CodegenContext::buildArg(RooAbsCollection const &in)
{
   if (in.empty()) {
      return "nullptr";
   }

   auto it = _listNames.find(in.uniqueId().value());
   if (it != _listNames.end())
      return it->second;

   std::string savedName = getTmpVarName();
   bool canSaveOutside = true;

   std::stringstream declStrm;
   declStrm << "double " << savedName << "[] = {";
   for (const auto arg : in) {
      declStrm << getResult(*arg) << ",";
      canSaveOutside = canSaveOutside && isScopeIndependent(arg);
   }
   declStrm.seekp(-1, declStrm.cur);
   declStrm << "};\n";

   addToCodeBody(declStrm.str(), canSaveOutside);

   _listNames.insert({in.uniqueId().value(), savedName});
   return savedName;
}

std::string CodegenContext::buildArg(std::span<const double> arr)
{
   unsigned int n = arr.size();
   std::string offset = std::to_string(_xlArr.size());
   _xlArr.reserve(_xlArr.size() + n);
   for (unsigned int i = 0; i < n; i++) {
      _xlArr.push_back(arr[i]);
   }
   return "xlArr + " + offset;
}

CodegenContext::ScopeRAII::ScopeRAII(RooAbsArg const *arg, CodegenContext &ctx) : _ctx(ctx), _arg(arg)
{
   std::ostringstream os;
   Option_t *opts = nullptr;
   arg->printStream(os, _arg->defaultPrintContents(opts), _arg->defaultPrintStyle(opts));
   _fn = os.str();
   const std::string info = "// Begin -- " + _fn;
   _ctx._indent++;
   _ctx.addToCodeBody(_arg, info);
}

CodegenContext::ScopeRAII::~ScopeRAII()
{
   const std::string info = "// End -- " + _fn + "\n";
   _ctx.addToCodeBody(_arg, info);
   _ctx._indent--;
}

void CodegenContext::pushScope()
{
   _code.push_back("");
}

void CodegenContext::popScope()
{
   std::string active_scope = _code.back();
   _code.pop_back();
   _code.back() += active_scope;
}

bool CodegenContext::isScopeIndependent(RooAbsArg const *in) const
{
   return !in->isReducerNode() && outputSize(in->namePtr()) == 1;
}

/// @brief Register a function that is only know to the interpreter to the context.
/// This is useful to dump the standalone C++ code for the computation graph.
void CodegenContext::collectFunction(std::string const &name)
{
   _collectedFunctions.emplace_back(name);
}

/// @brief Assemble and return the final code with the return expression and global statements.
/// @param returnExpr The string representation of what the squashed function should return, usually the head node.
/// @return The name of the declared function.
std::string
CodegenContext::buildFunction(RooAbsArg const &arg, std::map<RooFit::Detail::DataKey, std::size_t> const &outputSizes)
{
   CodegenContext ctx;
   ctx.pushScope(); // push our global scope.
   ctx._nodeOutputSizes = outputSizes;
   ctx._vecObsIndices = _vecObsIndices;
   // We only want to take over parameters and observables
   for (auto const &item : _nodeNames) {
      if (startsWith(item.second, "params[") || startsWith(item.second, "obs[")) {
         ctx._nodeNames.insert(item);
      }
   }
   ctx._xlArr = _xlArr;
   ctx._collectedFunctions = _collectedFunctions;

   static int iCodegen = 0;
   auto funcName = "roo_codegen_" + std::to_string(iCodegen++);

   // Make sure the codegen implementations are known to the interpreter
   gInterpreter->Declare("#include <RooFit/CodegenImpl.h>\n");

   ctx.pushScope();
   std::string funcBody = ctx.getResult(arg);
   ctx.popScope();
   funcBody = ctx._code[0] + "\n return " + funcBody + ";\n";

   // Declare the function
   std::stringstream bodyWithSigStrm;
   bodyWithSigStrm << "double " << funcName << "(double* params, double const* obs, double const* xlArr) {\n"
                   << funcBody << "\n}";
   ctx._collectedFunctions.emplace_back(funcName);
   if (!gInterpreter->Declare(bodyWithSigStrm.str().c_str())) {
      std::stringstream errorMsg;
      std::string debugFileName = "_codegen_" + funcName + ".cxx";
      errorMsg << "Function " << funcName << " could not be compiled. See above for details. Full code dumped to file "
               << debugFileName << "for debugging";
      {
         std::ofstream outFile;
         outFile.open(debugFileName.c_str());
         outFile << bodyWithSigStrm.str();
      }
      oocoutE(nullptr, InputArguments) << errorMsg.str() << std::endl;
      throw std::runtime_error(errorMsg.str().c_str());
   }

   _xlArr = ctx._xlArr;
   _collectedFunctions = ctx._collectedFunctions;

   return funcName;
}

void declareDispatcherCode(std::string const &funcName)
{
   std::string dispatcherCode = R"(
namespace RooFit {
namespace Experimental {

template <class Arg_t, int P>
auto FUNC_NAME(Arg_t &arg, CodegenContext &ctx, Prio<P> p)
{
   if constexpr (std::is_same<Prio<P>, PrioLowest>::value) {
      return FUNC_NAME(arg, ctx);
   } else {
      return FUNC_NAME(arg, ctx, p.next());
   }
}

template <class Arg_t>
struct Caller_FUNC_NAME {

   static auto call(RooAbsArg &arg, CodegenContext &ctx)
   {
      return FUNC_NAME(static_cast<Arg_t &>(arg), ctx, PrioHighest{});
   }
};

} // namespace Experimental
} // namespace RooFit
   )";

   RooFit::Detail::replaceAll(dispatcherCode, "FUNC_NAME", funcName);
   gInterpreter->Declare(dispatcherCode.c_str());
}

void codegen(RooAbsArg &arg, CodegenContext &ctx)
{
   static bool codeDeclared = false;
   if (!codeDeclared) {
      declareDispatcherCode("codegenImpl");
      codeDeclared = true;
   }

   using Func = void (*)(RooAbsArg &, CodegenContext &);

   Func func;

   TClass *tclass = arg.IsA();

   // Cache the overload resolutions
   static std::unordered_map<TClass *, Func> dispatchMap;

   auto found = dispatchMap.find(tclass);

   if (found != dispatchMap.end()) {
      func = found->second;
   } else {
      // Can probably done with CppInterop in the future to avoid string manipulation.
      std::stringstream cmd;
      cmd << "&RooFit::Experimental::Caller_codegenImpl<" << tclass->GetName() << ">::call;";
      func = reinterpret_cast<Func>(gInterpreter->ProcessLine(cmd.str().c_str()));
      dispatchMap[tclass] = func;
   }

   return func(arg, ctx);
}

} // namespace Experimental
} // namespace RooFit
