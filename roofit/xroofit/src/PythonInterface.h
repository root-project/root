#ifndef xRooFit_PythonInterface_h
#define xRooFit_PythonInterface_h

// Helper functions to use the Python C API without introducint a link-time
// dependency on libpython. These are intended to be used only when xRooFit is
// used from Python, where libpython is loaded by default and the required
// symbols can be found with gSystem.

namespace xPython {

   bool isPythonInitialized();

   // Don't call these other functions if Python is not initialized!
   void writeStdoutLine(const char *msg);
   void writeStderrLine(const char *msg);
} // namespace xPython

#endif