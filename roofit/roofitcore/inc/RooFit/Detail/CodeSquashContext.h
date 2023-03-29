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
   CodeSquashContext(const RooAbsData *data = nullptr) : _numEntries(data ? data->numEntries() : 0) {}

   /// @brief Adds (or overwrites) the string representing the result of a node.
   /// @param key The node to add the result for.
   /// @param value The new name to assign/overwrite.
   inline void addResult(RooAbsArg const *key, std::string const &value) { addResult(key->namePtr(), value); }

   inline void addResult(const char *key, std::string const &value)
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
   inline std::string const &getResult(RooAbsArg const &arg)
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

   template <class T>
   std::string const &getResult(RooTemplateProxy<T> const &key)
   {
      return getResult(key.arg());
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

   /// @brief Adds the input string to the squashed code body. If a class implements a translate function that wants to
   /// emit something to the squashed code body, it must call this function with the code it wants to emit.
   /// @param in String to add to the squashed code.
   inline void addToCodeBody(std::string const &in) { _code += in; }

   /// @brief Build the code to call the function with name `funcname`, passing some arguments.
   /// The arguments can either be doubles or some RooFit arguments whose
   /// results will be looked up in the context.
   template <typename... Args_t>
   std::string buildCall(std::string const &funcname, Args_t const &...args)
   {
      std::stringstream ss;
      ss << funcname << "(" << buildArgs(args...) << ")" << std::endl;
      return ss.str();
   }

   /// @brief A class to manage loop scopes using the RAII technique. To wrap your code around a loop,
   /// simply place it between a brace inclosed scope with a call to beginLoop at the top. For e.g.
   /// {
   ///   auto scope = ctx.beginLoop({<-set of vector observables to loop over->});
   ///   // your loop body code goes here.
   /// }
   class LoopScope {
   public:
      LoopScope(CodeSquashContext &ctx, std::vector<TNamed const *> &&vars) : _ctx{ctx}, _vars{vars} {}
      ~LoopScope() { _ctx.endLoop(*this); }

      std::vector<TNamed const *> const &vars() const { return _vars; }

   private:
      CodeSquashContext &_ctx;
      const std::vector<TNamed const *> _vars;
   };

   /// @brief Create a RAII scope for iterating over vector observables. You can't use the result of vector observables
   /// outside these loop scopes.
   /// @param loopVars The vector observables to iterate over. If one of the
   /// loopVars is not a vector observable, it is ignored, i.e., it can be used just like outside the loop scope.
   std::unique_ptr<LoopScope> beginLoop(RooArgSet const &loopVars)
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

private:
   void endLoop(LoopScope const &scope)
   {
      _code += "}\n";

      // clear the results of the loop variables if they were vector observables
      for (auto const &ptr : scope.vars()) {
         if (_vecObsIndices.find(ptr) != _vecObsIndices.end())
            _nodeNames.erase(ptr);
      }
      --_loopLevel;
   }

   inline void addResult(TNamed const *key, std::string const &value) { _nodeNames[key] = value; }

   std::string buildArg(double x) { return RooNumber::toString(x); }

   template <class T>
   std::string buildArg(T const &arg)
   {
      return getResult(arg);
   }

   std::string buildArgs() { return ""; }

   template <class Arg_t>
   std::string buildArgs(Arg_t const &arg)
   {
      return buildArg(arg);
   }

   template <typename Arg_t, typename... Args_t>
   std::string buildArgs(Arg_t const &arg, Args_t const &...args)
   {
      return buildArg(arg) + ", " + buildArgs(args...);
   }

   /// @brief A crude way of keeping track of loop ranges for nodes like NLLs.
   const std::size_t _numEntries = 0;
   /// @brief Map of node names to their result strings.
   std::unordered_map<const TNamed *, std::string> _nodeNames;
   /// @brief Block of code that is placed before the rest of the function body.
   std::string _globalScope;
   /// @brief A map to keep track of the observable indices if they are non scalar.
   std::unordered_map<const TNamed *, int> _vecObsIndices;
   /// @brief Stores the squashed code body.
   std::string _code;
   /// @brief The current number of for loops the started.
   int _loopLevel = 0;
};

} // namespace Detail

} // namespace RooFit

#endif
