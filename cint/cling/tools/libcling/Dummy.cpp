// request symbols
#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/CValuePrinter.h"
#include "cling/Interpreter/ValuePrinter.h"
#include "cling/Interpreter/ValuePrinterInfo.h"
#include "cling/UserInterface/UserInterface.h"

#include "llvm/Support/raw_ostream.h"

namespace cling {
void libcling__symbol_requester() {
   const char* const argv[] = {"libcling__symbol_requester", 0};
   cling::Interpreter I(1, argv);
   cling::UserInterface U(I);
   cling::ValuePrinterInfo VPI(0, 0); // asserts, but we don't call.
   printValueDefault(llvm::outs(), 0, VPI);
   cling_PrintValue(0, 0, 0);
   flushOStream(llvm::outs());
}
}
