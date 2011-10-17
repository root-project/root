#include <iostream>
#include <string>

#include <cling/Interpreter/Interpreter.h>

#include <clang/Basic/LangOptions.h>
#include <clang/Basic/TargetInfo.h>

#include <llvm/ADT/OwningPtr.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/Module.h>
#include <llvm/Support/MemoryBuffer.h>


extern "C" {
int call_interp(const char* code) {
   clang::LangOptions langInfo;
   langInfo.C99         = 1;
   langInfo.HexFloats   = 1;
   langInfo.BCPLComment = 1; // Only for C99/C++.
   langInfo.Digraphs    = 1; // C94, C99, C++.
   langInfo.CPlusPlus   = 1;
   langInfo.CPlusPlus0x = 1;
   langInfo.CXXOperatorNames = 1;
   langInfo.Bool = 1;
   langInfo.Exceptions = 1;

   cling::Interpreter interp(langInfo);
   std::string wrapped_code("int reentrant_main(int argc, char* argv[]) { ");
   wrapped_code += code;
   wrapped_code += "return 0;}";
   llvm::MemoryBuffer* buff;
   buff  = llvm::MemoryBuffer::getMemBufferCopy( &*wrapped_code.begin(),
                                                 &*wrapped_code.end(),
                                                 "reent.cxx" );
   //----------------------------------------------------------------------
   // Parse and run it
   //----------------------------------------------------------------------
   llvm::Module* module = interp.link( buff );

   if(!module) {
      std::cerr << std::endl;
      std::cerr << "[!] Errors occured while parsing your code!" << std::endl;
      std::cerr << std::endl;
      return -1;
   }
   return interp.executeModuleMain( module, "reentrant_main" );
};
}
