//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Axel Naumann <axel@cern.ch>
//------------------------------------------------------------------------------

#include "cling/MetaProcessor/MetaProcessor.h"

#include "InputValidator.h"
#include "cling/Interpreter/Interpreter.h"

#include "clang/Basic/TargetInfo.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Preprocessor.h"

#include "llvm/Support/Path.h"

#include <cstdio>

using namespace clang;

namespace cling {

  MetaProcessor::MetaProcessor(Interpreter& interp) : m_Interp(interp) {
    m_InputValidator.reset(new InputValidator());
  }

  MetaProcessor::~MetaProcessor() {}

  int MetaProcessor::process(const char* input_text, Value* result /*=0*/) {
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
    //  Check for and handle any '.' commands.
    bool was_meta = false;
    if ((input_line[0] == '.') && (input_line.size() > 1)) {
      was_meta = ProcessMeta(input_line, result);
    }
    if (was_meta) {
      return 0;
    }

    // Check if the current statement is now complete. If not, return to 
    // prompt for more.
    if (m_InputValidator->Validate(input_line, m_Interp.getCI()->getLangOpts()) 
        == InputValidator::kIncomplete) {
      return m_InputValidator->getExpectedIndent();
    }

    //  We have a complete statement, compile and execute it.
    std::string input = m_InputValidator->TakeInput();
    m_InputValidator->Reset();
    m_Interp.processLine(input, m_Options.RawInput, result);

    return 0;
  }

  MetaProcessorOpts& MetaProcessor::getMetaProcessorOpts() {
    // Take interpreter's state
    m_Options.PrintingAST = m_Interp.isPrintingAST();
    return m_Options; 
  }


  bool MetaProcessor::ProcessMeta(const std::string& input_line, Value* result){

   llvm::MemoryBuffer* MB = llvm::MemoryBuffer::getMemBuffer(input_line);
   const LangOptions& LO = m_Interp.getCI()->getLangOpts();
   Lexer RawLexer(SourceLocation(), LO, MB->getBufferStart(),
                  MB->getBufferStart(), MB->getBufferEnd());
   Token Tok;

   RawLexer.LexFromRawLexer(Tok);
   if (Tok.isNot(tok::period))
     return false;

   // Read the command
   RawLexer.LexFromRawLexer(Tok);
   if (!Tok.isAnyIdentifier() && Tok.isNot(tok::at))
     return false;

   const std::string Command = GetRawTokenName(Tok);
   std::string Param;

   //  .q //Quits
   if (Command == "q") {
      m_Options.Quitting = true;
      return true;
   }
   //  .L <filename>   //  Load code fragment.
   else if (Command == "L") {
     // Check for params
     RawLexer.LexFromRawLexer(Tok);
     if (!Tok.isAnyIdentifier())
       return false;

     Param = GetRawTokenName(Tok);
     bool success = m_Interp.loadFile(Param);
     if (!success) {
       llvm::errs() << "Load file failed.\n";
     }
     return true;
   } 
   //  .(x|X) <filename> //  Execute function from file, function name is 
   //                    //  filename without extension.
   else if ((Command == "x") || (Command == "X")) {
     // TODO: add extensive checks the folder paths and filenames
     //RawLexer->LexFromRawLexer(Tok);
     //if (!Tok.isAnyIdentifier())
     //  return false;

     const char* CurPtr = RawLexer.getBufferLocation();;
     Token TmpTok;
     RawLexer.getAndAdvanceChar(CurPtr, TmpTok);
     llvm::StringRef Param(CurPtr, 
                           MB->getBufferSize() - (CurPtr - MB->getBufferStart()));
     llvm::sys::Path path(Param);
 
     if (!path.isValid())
       return false;

     bool success = executeFile(path.c_str(), result);
      if (!success) {
        llvm::errs()<< "Execute file failed.\n";
      }
      return true;
   }
   //  .printAST [0|1]  // Toggle the printing of the AST or if 1 or 0 is given
   //                   // enable or disable it.
   else if (Command == "printAST") {
     // Check for params
     RawLexer.LexFromRawLexer(Tok);
     if (Tok.isNot(tok::numeric_constant) && Tok.isNot(tok::eof))
       return false;

     if (Tok.is(tok::eof)) {
       // toggle:
       bool print = !m_Interp.isPrintingAST();
       m_Interp.enablePrintAST(print);
       llvm::errs()<< (print?"P":"Not p") << "rinting AST\n";
     } else { 
       Param = GetRawTokenName(Tok);

       if (Param == "0") 
         m_Interp.enablePrintAST(false);
       else
         m_Interp.enablePrintAST(true);
     }

     m_Options.PrintingAST = m_Interp.isPrintingAST();
     return true;
   }
   //  .rawInput [0|1]  // Toggle the raw input or if 1 or 0 is given enable 
   //                   // or disable it.
   else if (Command == "rawInput") {
     // Check for params
     RawLexer.LexFromRawLexer(Tok);
     if (Tok.isNot(tok::numeric_constant) && Tok.isNot(tok::eof))
       return false;

     if (Tok.is(tok::eof)) {
       // toggle:
       m_Options.RawInput = !m_Options.RawInput;
       llvm::errs() << (m_Options.RawInput?"U":"Not u") << "sing raw input\n";
     } else { 
       Param = GetRawTokenName(Tok);

       if (Param == "0")
         m_Options.RawInput = false;
       else 
         m_Options.RawInput = true;
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
   else if (Command == "I") {
     // Check for params
     RawLexer.LexFromRawLexer(Tok);
     
     if (Tok.is(tok::eof))
       m_Interp.DumpIncludePath();
     else {
       // TODO: add extensive checks the folder paths and filenames
       const char* CurPtr = RawLexer.getBufferLocation();;
       Token TmpTok;
       RawLexer.getAndAdvanceChar(CurPtr, TmpTok);
       llvm::StringRef Param(CurPtr, 
                             MB->getBufferSize()-(CurPtr-MB->getBufferStart()));
       llvm::sys::Path path(Param);
       
       if (path.isValid())
         m_Interp.AddIncludePath(path.c_str());
       else
         return false;
     }
     return true;
   }
  // Cancel the multiline input that has been requested
   else if (Command == "@") {
     m_InputValidator->Reset();
     return true;
   }
   // Enable/Disable DynamicExprTransformer
   else if (Command == "dynamicExtensions") {
     // Check for params
     RawLexer.LexFromRawLexer(Tok);
     if (Tok.isNot(tok::numeric_constant) && Tok.isNot(tok::eof))
       return false;

     if (Tok.is(tok::eof)) {
       // toggle:
       bool dynlookup = !m_Interp.isDynamicLookupEnabled();
       m_Interp.enableDynamicLookup(dynlookup);
       llvm::errs() << (dynlookup?"U":"Not u") <<"sing dynamic extensions\n";
     } else {
       Param = GetRawTokenName(Tok);

       if (Param == "0")
         m_Interp.enableDynamicLookup(false);
       else 
         m_Interp.enableDynamicLookup(true);
     }

     return true;
   }

   return false;
  }

  std::string MetaProcessor::GetRawTokenName(const Token& Tok) {

    assert(!Tok.needsCleaning() && "Not implemented yet");

    switch (Tok.getKind()) {
    default:
      return "";
    case tok::numeric_constant:
      return Tok.getLiteralData();
    case tok::raw_identifier:
      return StringRef(Tok.getRawIdentifierData(), Tok.getLength()).str(); 
    case tok::slash:
      return "/";
    }

  }

  // Run a file: .x file[(args)]
  bool MetaProcessor::executeFile(const std::string& fileWithArgs, 
                                  Value* result) {
    // Look for start of parameters:

    typedef std::pair<llvm::StringRef,llvm::StringRef> StringRefPair;

    StringRefPair pairFileArgs = llvm::StringRef(fileWithArgs).split('(');
    if (pairFileArgs.second.empty()) {
      pairFileArgs.second = ")";
    }
    StringRefPair pairPathFile = pairFileArgs.first.rsplit('/');
    if (pairPathFile.second.empty()) {
       pairPathFile.second = pairPathFile.first;
    }
    StringRefPair pairFuncExt = pairPathFile.second.rsplit('.');

    //fprintf(stderr, "funcname: %s\n", pairFuncExt.first.data());

    Interpreter::CompilationResult interpRes
       = m_Interp.processLine(std::string("#include \"")
                              + pairFileArgs.first.str()
                              + std::string("\""), true /*raw*/);
    
    if (interpRes != Interpreter::kFailure) {
       std::string expression = pairFuncExt.first.str()
          + "(" + pairFileArgs.second.str();
       interpRes = m_Interp.processLine(expression, false /*not raw*/, result);
    }
    
    return (interpRes != Interpreter::kFailure);   
  }
} // end namespace cling

