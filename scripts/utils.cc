// Utilities to be used by some scripts.
// To load them up copy the rootlogon_template.C into the current directory.

#ifdef WIN32
#include <windows.h>
#endif

#include <stdio.h>
#include "snprintf.h"

#include "TError.h"
#include "TSystem.h"

// This redirect the root warning to be on stdout instead of stderr.

//______________________________________________________________________________
void StdoutDebugPrint(const char *fmt, ...)
{
   // Print debugging message to stderr and, on Windows, to the system debugger.

   static Int_t buf_size = 2048;
   static char *buf = 0;

   va_list arg_ptr;
   va_start(arg_ptr, fmt);

again:
   if (!buf)
      buf = new char[buf_size];

   int n = vsnprintf(buf, buf_size, fmt, arg_ptr);

   if (n == -1 || n >= buf_size) {
      buf_size *= 2;
      delete [] buf;
      buf = 0;
      goto again;
   }
   va_end(arg_ptr);

   fprintf(stdout, "%s", buf);

#ifdef WIN32
   ::OutputDebugString(buf);
#endif
}

//______________________________________________________________________________
void StdoutErrorHandler(int level, Bool_t abort, const char *location, const char *msg)
{
   // The default error handler function. It prints the message on stderr and
   // if abort is set it aborts the application.

   if (level < gErrorIgnoreLevel)
      return;

   const char *type = 0;

   if (level >= kInfo)
      type = "Info";
   if (level >= kWarning)
      type = "Warning";
   if (level >= kError)
      type = "Error";
   if (level >= kBreak)
      type = "\n *** Break ***";
   if (level >= kSysError)
      type = "SysError";
   if (level >= kFatal)
      type = "Fatal";

   if (level >= kBreak && level < kSysError)
      StdoutDebugPrint("%s %s\n", type, msg);
   else if (!location || strlen(location) == 0)
      StdoutDebugPrint("%s: %s\n", type, msg);
   else
      StdoutDebugPrint("%s in <%s>: %s\n", type, location, msg);

   fflush(stdout);
   if (abort) {
      StdoutDebugPrint("aborting\n");
      fflush(stdout);
      if (gSystem) {
         gSystem->StackTrace();
         gSystem->Abort();
      } else
         ::abort();
   }
}

void SetROOTMessageToStdout() {
   SetErrorHandler(StdoutErrorHandler);
}
