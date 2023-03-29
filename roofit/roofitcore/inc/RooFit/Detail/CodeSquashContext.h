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

#ifndef RooFit_Detail_CodeSquashContext_h
#define RooFit_Detail_CodeSquashContext_h

#include <RooAbsArg.h>
#include <RooNumber.h>
#include <RooAbsData.h>

#include <sstream>
#include <string>
#include <unordered_map>

template <class T>
class RooTemplateProxy;

namespace RooFit {

namespace Detail {

/// @brief A class to maintain the context for squashing of RooFit models into code.
class CodeSquashContext {
public:
   /// @brief A crude way of keeping track of loop ranges for nodes like NLLs.
   const std::size_t numEntries;

   CodeSquashContext(const RooAbsData *data = nullptr) : numEntries(data ? data->numEntries() : 0) {}

   /// @brief Adds (or overwrites) the string representing the result of a node.
   /// @param key The node to add the result for.
   /// @param value The new name to assign/overwrite.
   inline void addResult(RooAbsArg const *key, std::string value) { _nodeNames[key->namePtr()] = value; }

   inline void addResult(const char *key, std::string value)
   {
      const TNamed *namePtr = RooNameReg::known(key);
      if (namePtr)
         _nodeNames[namePtr] = value;
   }

   /// @brief Gets the result for the given node using the node name.
   /// @param key The node to get the result string for.
   /// @return String representing the result of this node.
   inline std::string const &getResult(RooAbsArg const &key) const { return _nodeNames.at(key.namePtr()); }

   template <class T>
   std::string const &getResult(RooTemplateProxy<T> const &key) const
   {
      return getResult(key.arg());
   }

   /// @brief Checks if the current node has a result string already assigned.
   /// @param key The node to get the result string for.
   /// @return True if the node was assigned a result string, false otherwise.
   inline bool isResultAssigned(RooAbsArg const *key) const
   {
      return _nodeNames.find(key->namePtr()) != _nodeNames.end();
   }

   /// @brief Adds the given string to the string block that will be emitted at the top of the squashed function. Useful
   /// for variable declarations.
   /// @param str The string to add to the global scope.
   inline void addToGlobalScope(std::string const &str) { _globalScope += str; }

   /// @brief Assemble and return the final code with the return expression and global statements.
   /// @param returnExpr The string representation of what the squashed function should return, usually the head node.
   /// @return The final body of the function.
   inline std::string assembleCode(std::string const &returnExpr)
   {
      return _globalScope + _code + "\n return " + returnExpr + ";\n";
   }

   /// @brief Since the squashed code represents all observables as a single flattened array, it is important
   /// to keep track of the start index for a vector valued observable which can later be expanded to access the correct
   /// element. For example, a vector valued variable x with 10 entries will be squashed to obs[start_idx + i].
   /// @param key The node representing the vector valued observable.
   /// @param idx The start index (or relative position of the observable in the set of all observables).
   inline void addVecObs(RooAbsArg const *key, int idx) { _vecObsIndices[key->namePtr()] = idx; }

   inline void addVecObs(const char *key, int idx)
   {
      const TNamed *namePtr = RooNameReg::known(key);
      if (namePtr)
         _vecObsIndices[namePtr] = idx;
   }

   /// @brief Get the start index of the node representing the key. If the key is not found (in the case the key is
   /// scalar), return -1.
   /// @param key The node to perform a lookup for.
   /// @return The start index of the observable if found, otherwise -1.
   inline int getVecObsStartIdx(RooAbsArg const *key)
   {
      return _vecObsIndices.find(key->namePtr()) == _vecObsIndices.end() ? -1 : _vecObsIndices[key->namePtr()];
   }

   /// @brief Adds the input string to the squashed code body. If a class implements a translate function that wants to
   /// emit something to the squashed code body, it must call this function with the code it wants to emit.
   /// @param in String to add to the squashed code.
   inline void addToCodeBody(std::string const &in) { _code += in; }

   /// @brief Build the code to call the function with name `funcname`, passing some arguments.
   /// The arguments can either be doubles or some RooFit arguments whose
   /// results will be looked up in the context.
   template <typename... Args_t>
   std::string buildCall(std::string const &funcname, Args_t const &...args) const
   {
      std::stringstream ss;
      ss << funcname << "(" << buildArgs(args...) << ")" << std::endl;
      return ss.str();
   }

private:
   std::string buildArg(double x) const { return RooNumber::toString(x); }

   template <class T>
   std::string buildArg(T const &arg) const
   {
      return getResult(arg);
   }

   std::string buildArgs() const { return ""; }

   template <class Arg_t>
   std::string buildArgs(Arg_t const &arg) const
   {
      return buildArg(arg);
   }

   template <typename Arg_t, typename... Args_t>
   std::string buildArgs(Arg_t const &arg, Args_t const &...args) const
   {
      return buildArg(arg) + ", " + buildArgs(args...);
   }

   /// @brief Map of node names to their result strings.
   std::unordered_map<const TNamed *, std::string> _nodeNames;
   /// @brief Block of code that is placed before the rest of the function body.
   std::string _globalScope;
   /// @brief A map to keep track of the observable indices if they are non scalar.
   std::unordered_map<const TNamed *, int> _vecObsIndices;
   /// @brief Stores the squashed code body.
   std::string _code;
};

} // namespace Detail

} // namespace RooFit

#endif
