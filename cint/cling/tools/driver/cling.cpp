//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Lukasz Janyst <ljanyst@cern.ch>
//------------------------------------------------------------------------------

#include <iostream>
#include <vector>
#include <string>

#include "clang/Basic/LangOptions.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/HeaderSearchOptions.h"

#include "llvm/Support/Signals.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/ManagedStatic.h"

#include "cling/Interpreter/Interpreter.h"
#include "cling/UserInterface/UserInterface.h"

//------------------------------------------------------------------------------
// Let the show begin
//------------------------------------------------------------------------------
int main( int argc, char **argv )
{

   llvm::llvm_shutdown_obj shutdownTrigger;

   //llvm::sys::PrintStackTraceOnErrorSignal();
   //llvm::PrettyStackTraceProgram X(argc, argv);

   //---------------------------------------------------------------------------
   // Set up the interpreter
   //---------------------------------------------------------------------------
   cling::Interpreter interpreter(argc, argv);
   if (interpreter.getOptions().Help) {
      return 0;
   }

   clang::CompilerInstance* CI = interpreter.getCI();
  interpreter.AddIncludePath(".");
  
  for (size_t I = 0, N = interpreter.getOptions().LibsToLoad.size();
       I < N; ++I) {
    interpreter.loadFile(interpreter.getOptions().LibsToLoad[I]);
  }

   bool ret = true;
   const std::vector<std::pair<clang::InputKind, std::string> >& Inputs
     = CI->getInvocation().getFrontendOpts().Inputs;

   // Interactive means no input (or one input that's "-")
   bool Interactive = Inputs.empty()
     || (Inputs.size() == 1 && Inputs[0].second == "-");
   if (!Interactive) {
     //---------------------------------------------------------------------------
     // We're supposed to parse files
     //---------------------------------------------------------------------------
      for (size_t I = 0, N = Inputs.size(); I < N; ++I) {
        ret = interpreter.executeFile(Inputs[I].second);
      }
   }
   //----------------------------------------------------------------------------
   // We're interactive
   //----------------------------------------------------------------------------
   else {
      cling::UserInterface ui(interpreter);
      ui.runInteractively(interpreter.getOptions().NoLogo);
   }

   return ret ? 0 : 1;
}
