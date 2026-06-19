#include "./PythonInterface.h"

#include <TSystem.h>

namespace xPython {

  // https://docs.python.org/3.11/c-api/init.html#c.Py_IsInitialized
   bool isPythonInitialized() {
      using fn_t = int (*)();
      static auto f = reinterpret_cast<fn_t>(gSystem->DynFindSymbol("*", "Py_IsInitialized"));
      return f && f();
   }

   // https://docs.python.org/3/c-api/sys.html#c.PySys_WriteStdout
   void writeStdoutLine(const char *msg) {
      using fn_t = void (*)(const char *, ...);
      static auto f = reinterpret_cast<fn_t>(gSystem->DynFindSymbol("*", "PySys_WriteStdout"));
      if (f)
            f("%s\n", msg);
   }

   // https://docs.python.org/3/c-api/sys.html#c.PySys_WriteStderr
   void writeStderrLine(const char *msg) {
      using fn_t = void (*)(const char *, ...);
      static auto f = reinterpret_cast<fn_t>(gSystem->DynFindSymbol("*", "PySys_WriteStderr"));
      if (f)
         f("%s\n", msg);
   }

} // namespace xPython