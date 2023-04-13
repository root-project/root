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

namespace RooFit {

namespace Detail {

/// @brief Adds (or overwrites) the string representing the result of a node.
/// @param key The name of the node to add the result for.
/// @param value The new name to assign/overwrite.
void CodeSquashContext::addResult(const char *key, std::string const &value)
{
   const TNamed *namePtr = RooNameReg::known(key);
   if (namePtr)
      addResult(namePtr, value);
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

/// @brief Create a RAII scope for iterating over vector observables. You can't use the result of vector observables
/// outside these loop scopes.
/// @param loopVars The vector observables to iterate over. If one of the
/// loopVars is not a vector observable, it is ignored, i.e., it can be used just like outside the loop scope.
std::unique_ptr<CodeSquashContext::LoopScope> CodeSquashContext::beginLoop(RooArgSet const &loopVars)
{
   // Make sure that the name of this variable doesn't clash with other stuff
   std::string idx = "loopIdx" + std::to_string(_loopLevel);
   addToCodeBody("for(int " + idx + " = 0; " + idx + " < " + std::to_string(_numEntries) + "; " + idx + "++) {\n");

   std::vector<TNamed const *> vars;
   for (RooAbsArg const *var : loopVars) {
      vars.push_back(var->namePtr());
   }

   for (auto const &ptr : vars) {
      // set the results of the vector observables
      auto found = _vecObsIndices.find(ptr);
      if (found != _vecObsIndices.end())
         _nodeNames[found->first] = "obs[" + std::to_string(found->second) + " + " + idx + "]";
   }

   ++_loopLevel;
   return std::make_unique<LoopScope>(*this, std::move(vars));
}

void CodeSquashContext::endLoop(LoopScope const &scope)
{
   _code += "}\n";

   // clear the results of the loop variables if they were vector observables
   for (auto const &ptr : scope.vars()) {
      if (_vecObsIndices.find(ptr) != _vecObsIndices.end())
         _nodeNames.erase(ptr);
   }
   --_loopLevel;
}

} // namespace Detail
} // namespace RooFit
