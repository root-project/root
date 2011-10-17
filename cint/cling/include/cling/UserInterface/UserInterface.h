//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Lukasz Janyst <ljanyst@cern.ch>
//------------------------------------------------------------------------------

#ifndef CLING_USERINTERFACE_H
#define CLING_USERINTERFACE_H

namespace cling {
   class Interpreter;
   class MetaProcessor;

   //---------------------------------------------------------------------------
   //! Class for the user interaction with the interpreter
   //---------------------------------------------------------------------------
   class UserInterface
   {
   public:
      UserInterface(Interpreter& interp, const char* prompt = "[cling] $");
      ~UserInterface();

      void runInteractively(bool nologo = false);

   private:
      MetaProcessor* m_MetaProcessor;
   };
}

#endif // CLING_USERINTERFACE_H


