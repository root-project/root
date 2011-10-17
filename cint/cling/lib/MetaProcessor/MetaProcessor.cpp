//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Axel Naumann <axel@cern.ch>
//------------------------------------------------------------------------------

#include "cling/MetaProcessor/MetaProcessor.h"

#include "InputValidator.h"
#include "cling/Interpreter/Interpreter.h"

#include "clang/Frontend/CompilerInstance.h"

#include <cstdio>

//---------------------------------------------------------------------------
// Construct an interface for an interpreter
//---------------------------------------------------------------------------
cling::MetaProcessor::MetaProcessor(Interpreter& interp):
      m_Interp(interp)
{
  m_InputValidator.reset(new InputValidator());
}


//---------------------------------------------------------------------------
// Destruct the interface
//---------------------------------------------------------------------------
cling::MetaProcessor::~MetaProcessor()
{
}

//---------------------------------------------------------------------------
// Compile and execute some code or process some meta command.
//
// Return:
//
//            0: success
// indent level: prompt for more input
//
//---------------------------------------------------------------------------
int
cling::MetaProcessor::process(const char* input_text)
{
   if (!input_text) { // null pointer, nothing to do.
      return 0;
   }
   if (!input_text[0]) { // empty string, nothing to do.
      return m_InputValidator->getExpectedIndent();
   }
   std::string input_line(input_text);
   if (input_line == "\n") { // just a blank line, nothing to do.
      return 0;
   }
   //
   //  Check for and handle any '.' commands.
   //
   bool was_meta = false;
   if ((input_line[0] == '.') && (input_line.size() > 1)) {
      was_meta = ProcessMeta(input_line);
   }
   if (was_meta) {
      return 0;
   }
   //
   // Check if the current statement is now complete.
   // If not, return to prompt for more.
   //
   if (m_InputValidator->Validate(input_line, m_Interp.getCI()->getLangOpts()) 
       == InputValidator::kIncomplete) {
     return m_InputValidator->getExpectedIndent();
   }

   //
   //  We have a complete statement, compile and execute it.
   //
   std::string input = m_InputValidator->TakeInput();
   m_InputValidator->Reset();
   m_Interp.processLine(input, m_Options.RawInput);
   //
   //  All done.
   //

   return 0;
}

cling::MetaProcessorOpts& 
cling::MetaProcessor::getMetaProcessorOpts() {
  // Take interpreter's state
  m_Options.PrintingAST = m_Interp.isPrintingAST();
  return m_Options; 
}


//---------------------------------------------------------------------------
// Process possible meta commands (.L,...)
//---------------------------------------------------------------------------
bool
cling::MetaProcessor::ProcessMeta(const std::string& input_line)
{
   // The command is the char right after the initial '.' char.
   // Return whether the meta command was known and thus handled.

   const char cmd_char = input_line[1];

   //  .q
   //
   //  Quit.
   //
   if (cmd_char == 'q') {
      m_Options.Quitting = true;
      return true;
   }

   //
   //  Extract command and parameter:
   //    .command parameter
   //
   std::string cmd = input_line.substr(1, std::string::npos);
   std::string param;
   std::string::size_type endcmd = input_line.find_first_of(" \t\n", 2);
   if (endcmd != std::string::npos) { // have a blank after command
      cmd = input_line.substr(1, endcmd - 1);

      std::string::size_type firstparam = input_line.find_first_not_of(" \t\n", endcmd);
      std::string::size_type lastparam = input_line.find_last_not_of(" \t\n");

      if (firstparam != std::string::npos) { // have a parameter
         //
         //  Trim blanks from beginning and ending of parameter.
         //
         std::string::size_type len = (lastparam + 1) - firstparam;
         // Construct our parameter.
         param = input_line.substr(firstparam, len);
      }
   }

   //
   //  .L <filename>
   //
   //  Load code fragment.
   //
   if (cmd_char == 'L') {
      //fprintf(stderr, "Begin load file '%s'.\n", param.c_str());
      bool success = m_Interp.loadFile(param);
      //fprintf(stderr, "End load file '%s'.\n", param.c_str());
      if (!success) {
         //fprintf(stderr, "Load file failed.\n");
      }
      return true;
   }
   //
   //  .x <filename>
   //  .X <filename>
   //
   //  Execute function from file, function name is filename
   //  without extension.
   //
   if ((cmd_char == 'x') || (cmd_char == 'X')) {
      bool success = m_Interp.executeFile(param);
      if (!success) {
         //fprintf(stderr, "Execute file failed.\n");
      }
      return true;
   }
   //
   //  .printAST [0|1]
   //
   //  Toggle the printing of the AST or if 1 or 0 is given
   //  enable or disable it.
   //
   if (cmd == "printAST") {
      if (param.empty()) {
        // toggle:
        bool print = !m_Interp.isPrintingAST();
        m_Interp.enablePrintAST(print);
        printf("%srinting AST\n", print?"P":"Not p");
      } else if (param == "1") {
         m_Interp.enablePrintAST(true);
      } else if (param == "0") {
         m_Interp.enablePrintAST(false);
      } else {
         fprintf(stderr, ".printAST: parameter must be '0' or '1' or nothing, not %s.\n", param.c_str());
      }

      m_Options.PrintingAST = m_Interp.isPrintingAST();
      return true;
   }

   //
   //  .rawInput [0|1]
   //
   //  Toggle the raw input
   //  or if 1 or 0 is given enable or disable it.
   //
   if (cmd == "rawInput") {
      if (param.empty()) {
        // toggle:
        m_Options.RawInput = !m_Options.RawInput;
        printf("%ssing raw input\n", m_Options.RawInput?"Not u":"U");
      } else if (param == "1") {
        m_Options.RawInput = true;
      } else if (param == "0") {
        m_Options.RawInput = false;
      } else {
         fprintf(stderr, ".rawInput: parameter must be '0' or '1' or nothing, not %s.\n", param.c_str());
      }
      return true;
   }

   //
   //  .U <filename>
   //
   //  Unload code fragment.
   //
   //if (cmd_char == 'U') {
   //   llvm::sys::Path path(param);
   //   if (path.isDynamicLibrary()) {
   //      std::cerr << "[i] Failure: cannot unload shared libraries yet!"
   //                << std::endl;
   //   }
   //   bool success = m_Interp.unloadFile(param);
   //   if (!success) {
   //      //fprintf(stderr, "Unload file failed.\n");
   //   }
   //   return true;
   //}
   //
   //  Unrecognized command.
   //
   //fprintf(stderr, "Unrecognized command.\n");
   if (cmd_char == 'I') {
     if (!param.empty())
       m_Interp.AddIncludePath(param.c_str());
     else {
       m_Interp.DumpIncludePath();
     }
     return true;
   }

   // Enable/Disable DynamicExprTransformer
   if (cmd == "dynamicExtensions") {
     if (param.empty()) {
       // toggle:
       bool dynlookup = !m_Interp.isDynamicLookupEnabled();
       m_Interp.enableDynamicLookup(dynlookup);
       printf("%ssing dynamic lookup extensions\n", dynlookup?"U":"Not u");
     } else if (param == "1") {
       m_Interp.enableDynamicLookup(true);
     } else if (param == "0") {
       m_Interp.enableDynamicLookup(false);
     } else {
       fprintf(stderr, ".dynamicExtensions: parameter must be '0' or '1' or nothing, not %s.\n", param.c_str());
     }
     return true;
   }

   return false;
}

