#include "TBufferFile.h"
#include "TError.h"
#include "TString.h"
#include <exception>

#ifndef __CINT__
class myexception
   {
   public:
      TString fMsg;
      myexception(const char *msg) : fMsg(msg) {}
      
      const char* what() const throw()
      {
         return fMsg.Data();
      }
   };
#endif

void NonDefaultErrorHandler(Int_t level, Bool_t abort_bool, const char *location, const char *msg)
{
   DefaultErrorHandler(level, kFALSE, location, msg);
   if (abort_bool) {
      throw myexception(msg);
   }
}

char * gMyBuffer = new char[3];

char *R__MyReAllocChar(char *what, size_t oldsize, size_t newsize)
{
   // The user has provided memory than we don't own, thus we can not extent it
   // either.
   char *old = gMyBuffer;
   if (old != what) {
      fprintf(stderr,"Did no get the expected buffer location in reallocation routine\n");
      return 0;
   } else {
      fprintf(stderr,"Running custom allocator\n");
   }
   gMyBuffer = new char[newsize];
   memcpy(gMyBuffer, old, oldsize);
   delete [] old;
   return gMyBuffer;
}

int runownership() 
{
   SetErrorHandler(NonDefaultErrorHandler);
   
   gErrorAbortLevel = 7000; // Never aborts
   
   TBufferFile *buf = new TBufferFile(TBuffer::kWrite, 3, gMyBuffer, kFALSE);
   int a = 3;
   bool success = true;
   try {
      (*buf) << a;
      success = false; // We should not get here!
   } catch (myexception e) {
      fprintf(stderr,"The expected error \"%s\" was seen\n", e.what());
   }
   
   buf->Reset();
   buf->SetBuffer(gMyBuffer, 3, kFALSE, R__MyReAllocChar);
   
   try {
      (*buf) << a;
      (*buf) << a;
      (*buf) << a;
   } catch (myexception e) {
      success = false; // We should not get here!
      fprintf(stderr,"An Unexpected error \"%s\" was seen\n", e.what());
   }
   return !success; // Value to be used by gmake
}
