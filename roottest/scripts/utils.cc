// Utilities to be used by some scripts.
// To load them up copy the rootlogon_template.C into the current directory.

#ifdef WIN32
#include <windows.h>
#endif

#include <stdio.h>
#include <cstdarg>
#include "snprintf.h"

#include "TError.h"
#include "TSystem.h"

#include "Riostream.h"

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

Bool_t CompareLines(const char*left,const char*right,UInt_t maxlen) 
{
   // Compare two string, ignore the difference between MSDOS and linux style
   // of line ending

   UInt_t i = 0;
   Bool_t result = true;
   while (i<maxlen && left[i] && right[i]) {
      if (left[i]!=right[i]) {
         if (left[i]==0x0a && left[i+1]==0) {
            if ((right[i]==0x0d && right[i+1]==0x0a && right[i+2] == 0) || 
                (right[i]==0x0a && right[i+1]==0) || right[i]==0) {
               return kTRUE;
            }
         }
         if (right[i]==0x0a && right[i+1]==0) {
            if ((left[i]==0x0d && left[i+1]==0x0a && left[i+2] == 0) || 
                (left[i]==0x0a && left[i+1]==0) || left[i]==0) {
               return kTRUE;
            }
            
         }
         return kFALSE;
     }
     ++i;
   }
   return result;
}

Bool_t ComparePostscript(const char *from, const char *to)
{
   FILE *left = fopen(from,"r");
   FILE *right = fopen(to,"r");
   
   if (left==0) std::cout << "Could not open " << from << std::endl;
   if (right==0) std::cout << "Could not open " << to << std::endl;
   if (left==0 || right==0) return false;

   char leftbuffer[256];
   char rightbuffer[256];
   
   char *lvalue,*rvalue;
   
   Bool_t areEqual = kTRUE;
   do {
      lvalue = fgets(leftbuffer, sizeof(leftbuffer), left);
      rvalue = fgets(rightbuffer, sizeof(rightbuffer), right);
      
      if (lvalue&&rvalue) {
         if (strstr(lvalue,"%%CreationDate")
             || strstr(lvalue,"%%Creator")) {
            // skip the comment line with the time and date
         } else {
            areEqual = areEqual && (CompareLines(lvalue,rvalue,sizeof(leftbuffer)));
         }
      }
      if (lvalue&&!rvalue) areEqual = kFALSE;
      if (rvalue&&!lvalue) areEqual = kFALSE;
      
   } while(areEqual && lvalue && rvalue);
   
   fclose(left);
   fclose(right);
   
   return areEqual;
}
