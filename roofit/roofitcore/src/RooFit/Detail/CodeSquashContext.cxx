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

#include <RooFit/Detail/CodeSquashContext.h>

#include <TString.h>

namespace RooFit {

namespace Detail {

/// Transform a string into a valid C++ variable name by replacing forbidden.
/// \note The implementation was copy-pasted from `TSystem.cxx`.
/// characters with underscores.
/// @param in The input string.
/// @return A new string vaild variable name.
std::string CodeSquashContext::makeValidVarName(TString in) const
{

   static const int nForbidden = 27;
   static const char *forbiddenChars[nForbidden] = {"+", "-", "*", "/", "&", "%", "|", "^",  ">",
                                                    "<", "=", "~", ".", "(", ")", "[", "]",  "!",
                                                    ",", "$", " ", ":", "'", "#", "@", "\\", "\""};
   for (int ic = 0; ic < nForbidden; ic++) {
      in.ReplaceAll(forbiddenChars[ic], "_");
   }

   return in.Data();
}

/// @brief Adds (or overwrites) the string representing the result of a node.
/// @param key The name of the node to add the result for.
/// @param value The new name to assign/overwrite.
void CodeSquashContext::addResult(const char *key, std::string const &value)
{
   const TNamed *namePtr = RooNameReg::known(key);
   if (namePtr)
      addResult(namePtr, value);
}

void CodeSquashContext::addResult(TNamed const *key, std::string const &value)
{
   _nodeNames[key] = value;
}

/// @brief Gets the result for the given node using the node name. This node also performs the necessary
/// code generation through recursive calls to 'translate'. A call to this function modifies the already
/// existing code body.
/// @param key The node to get the result string for.
/// @return String representing the result of this node.
std::string const &CodeSquashContext::getResult(RooAbsArg const &arg)
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

   // Now, recursively call translate into the current argument to load the correct result.
   arg.translate(*this);

   return _nodeNames.at(arg.namePtr());
}

/// @brief Adds the given string to the string block that will be emitted at the top of the squashed function. Useful
/// for variable declarations.
/// @param str The string to add to the global scope.
void CodeSquashContext::addToGlobalScope(std::string const &str)
{
   _globalScope += str;
}

/// @brief Assemble and return the final code with the return expression and global statements.
/// @param returnExpr The string representation of what the squashed function should return, usually the head node.
/// @return The final body of the function.
std::string CodeSquashContext::assembleCode(std::string const &returnExpr)
{
   return _globalScope + _code + "\n return " + returnExpr + ";\n";
}

/// @brief Since the squashed code represents all observables as a single flattened array, it is important
/// to keep track of the start index for a vector valued observable which can later be expanded to access the correct
/// element. For example, a vector valued variable x with 10 entries will be squashed to obs[start_idx + i].
/// @param key The name of the node representing the vector valued observable.
/// @param idx The start index (or relative position of the observable in the set of all observables).
void CodeSquashContext::addVecObs(const char *key, int idx)
{
   const TNamed *namePtr = RooNameReg::known(key);
   if (namePtr)
      _vecObsIndices[namePtr] = idx;
}

/// @brief Adds the input string to the squashed code body. If a class implements a translate function that wants to
/// emit something to the squashed code body, it must call this function with the code it wants to emit. In case of
/// loops, automatically determines if code needs to be stored inside or outside loop scope.
/// @param klass The class requesting this addition, usually 'this'.
/// @param in String to add to the squashed code.
void CodeSquashContext::addToCodeBody(RooAbsArg const *klass, std::string const &in)
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
void CodeSquashContext::addToCodeBody(std::string const &in, bool isScopeIndep /* = false */)
{
   // If we are in a loop and the value is scope independent, save it at the top of the loop.
   // else, just save it in the current scope.
   if (_scopePtr != -1 && isScopeIndep)
      _tempScope += in;
   else
      _code += in;
}

/// @brief Create a RAII scope for iterating over vector observables. You can't use the result of vector observables
/// outside these loop scopes.
/// @param in A pointer to the calling class, used to determine the loop dependent variables.
std::unique_ptr<CodeSquashContext::LoopScope> CodeSquashContext::beginLoop(RooAbsArg const *in)
{
   std::string idx = "loopIdx" + std::to_string(_loopLevel);

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

   // Save the current size of the code array so that we can insert the code at the right position.
   _scopePtr = _code.size();

   // Make sure that the name of this variable doesn't clash with other stuff
   addToCodeBody(in, "for(int " + idx + " = 0; " + idx + " < " + std::to_string(numEntries) + "; " + idx + "++) {\n");

   ++_loopLevel;
   return std::make_unique<LoopScope>(*this, std::move(vars));
}

void CodeSquashContext::endLoop(LoopScope const &scope)
{
   _code += "}\n";

   // Insert the temporary code into the correct code position.
   _code.insert(_scopePtr, _tempScope);
   _tempScope.erase();
   _scopePtr = -1;

   // clear the results of the loop variables if they were vector observables
   for (auto const &ptr : scope.vars()) {
      if (_vecObsIndices.find(ptr) != _vecObsIndices.end())
         _nodeNames.erase(ptr);
   }
   --_loopLevel;
}

/// @brief Get a unique variable name to be used in the generated code.
std::string CodeSquashContext::getTmpVarName()
{
   return "tmpVar" + std::to_string(_tmpVarIdx++);
}

/// @brief A function to save an expression that includes/depends on the result of the input node.
/// @param in The node on which the valueToSave depends on/belongs to.
/// @param valueToSave The actual string value to save as a temporary.
void CodeSquashContext::addResult(RooAbsArg const *in, std::string const &valueToSave)
{
   std::string savedName = makeValidVarName(in->GetName());

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
std::string CodeSquashContext::buildArg(RooAbsCollection const &in)
{
   auto it = listNames.find(in.uniqueId().value());
   if (it != listNames.end())
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

   listNames.insert({in.uniqueId().value(), savedName});
   return savedName;
}

std::string CodeSquashContext::buildArg(RooSpan<const double> arr)
{
   unsigned int n = arr.size();
   std::string arrName = getTmpVarName();
   std::string arrDecl = "double " + arrName + "[" + std::to_string(n) + "] = {";
   for (unsigned int i = 0; i < n; i++) {
      arrDecl += " " + std::to_string(arr[i]) + ",";
   }
   arrDecl.back() = '}';
   arrDecl += ";\n";
   addToCodeBody(arrDecl, true);

   return arrName;
}

bool CodeSquashContext::isScopeIndependent(RooAbsArg const *in) const
{
   return !in->isReducerNode() && outputSize(in->namePtr()) == 1;
}

} // namespace Detail
} // namespace RooFit
