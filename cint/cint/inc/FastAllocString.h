/* /% C++ %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file FastAllocString.h
 ************************************************************************
 * Description:
 * String object with fast allocation (pooled memory)
 ************************************************************************
 * Copyright(c) 1995~2009  Masaharu Goto 
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#ifndef G__FASTALLOGSTRING_H
#define G__FASTALLOGSTRING_H

#include <stdarg.h>

namespace Cint {
   namespace Internal {
      class G__BufferReservoir;
   }
}

//_____________________________________________________________
//
// A tiny object representing a char array.
// Create it with the desired size of the char array and it
// will try to retrieve a previsouly allocated buffer from
// an internal resrevoir of buffers, or allocate a new one
// if none is available. This is a lot faster than mallocs /
// free calls for each char array, and it considerably reduces
// the used stack size by functions previsouly using static
// size, stack based chart arrays. It also allows to make the
// buffer size dynamic, adopted e.g. to strlen(expression),
// instead of a value defined at compile time (a la G__LONGBUF).
// When the G__FastAllocString object leaves the scope it will put its
// buffer (back) into the internal buffer reservoir for later
// use by a G__FastAllocString object requesting a same of smaller size
// buffer. This class is optimized for both speed and low memory
// use despite the reservoir.
//
class G__FastAllocString {
public:
   G__FastAllocString(int reqsize): fBuf(0), fBucket(reqsize) {
      // GetBuf takes as parameter the size in bytes
      // and modify the parameter (fBucket) to hold the 
      // bucket number.
      fBuf = GetBuf(fBucket); 
   }
   ~G__FastAllocString();

   // plenty of char* conversion fuctions:
   operator char*() { return fBuf; }
   operator const char*() const { return fBuf; }
   const char* operator()() const { return fBuf; }

   // DON'T: these create ambiguities with ::op[char*, int] etc
   //char& operator[](int i) { return fBuf[i]; }
   //char operator[](int i) const { return fBuf[i]; }
   //char* operator+(int i) { return fBuf + i; }
   //const char* operator+(int i) const { return fBuf + i; }

   const char* data() const { return fBuf; }

   int FormatArgList(const char *fmt, va_list args);
   int Format(const char *fmt, ...);
         
protected:
   static char* GetBuf(int &size_then_bucket_index);
   static Cint::Internal::G__BufferReservoir& GetReservoir();

   void ResizeNoCopy(int newsize);
         
private:
   char* fBuf;    // the buffer
   int   fBucket; // measure representing the buffer's size, used by the internal reservoir
};

#endif // G__FASTALLOGSTRING_H
