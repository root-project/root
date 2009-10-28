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

char * gMyBuffer = new char[128];

char *R__MyReAllocChar(char *what, size_t newsize, size_t oldsize)
{
   // The user has provided memory than we don't own, thus we can not extent it
   // either.
   char *old = gMyBuffer;
   if (old != what) {
      fprintf(stderr,"Did no get the expected buffer location in reallocation routine\n");
      return 0;
   }

   fprintf(stderr,"Running custom allocator (increase from %ld to %ld)\n",(long)oldsize,(long)newsize);
   gMyBuffer = new char[newsize];
   memcpy(gMyBuffer, old, oldsize);
   delete [] old;
   return gMyBuffer;
}

int runownership() 
{
   SetErrorHandler(NonDefaultErrorHandler);
   
   gErrorAbortLevel = 7000; // Never aborts
   
   TBufferFile *buf = new TBufferFile(TBuffer::kWrite, 128, gMyBuffer, kFALSE);
   int a = 3;
   bool success = true;
   try {
      for (unsigned int i=0; i< (128-8)/sizeof(a); ++i) { // 8 is take into account TBuffer's kExtraSpace 
         (*buf) << a;
      }
   } catch (myexception e) {
      success = false; // We should not get here!
      fprintf(stderr,"An Unexpected error \"%s\" was seen\n", e.what());
   } 
   if (success) {
      try {
         (*buf) << a;
         success = false; // We should not get here!
      } catch (myexception e) {
         fprintf(stderr,"The expected error \"%s\" was seen\n", e.what());
      }
   }
   
   buf->Reset();
   buf->SetBuffer(gMyBuffer, 3, kFALSE, R__MyReAllocChar);
   
   try {
      for (unsigned int i=0; i< 128/sizeof(a); ++i) {
         (*buf) << a;
      }
      (*buf) << a;
      (*buf) << a;
      (*buf) << a;
   } catch (myexception e) {
      success = false; // We should not get here!
      fprintf(stderr,"After SetBuffer, an unexpected error \"%s\" was seen\n", e.what());
   }
   if (success) { 
      fprintf(stderr,"The test has completed successfully\n");
   } else {
      fprintf(stderr,"The test has run to completion but failed\n");
   }
   fflush(stderr);
   return !success; // Value to be used by gmake
}
