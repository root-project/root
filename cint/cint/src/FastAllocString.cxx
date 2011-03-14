/* /% C+ %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file strbuf.cxx
 ************************************************************************
 * Description:
 * String object with re-used buffers
 ************************************************************************
 * Copyright(c) 1995~2008  Masaharu Goto
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/


#include "FastAllocString.h"
#include "math.h"
#include "stdio.h"
#include "string.h"
#include <map>

namespace Cint {
   namespace Internal {
      //___________________________________________________________________
      //
      // Reservoir for allocated and unused buffers.
      // When requesting a buffer of a certain size,
      // a buffer >= that size is returned (or 0 if
      // there is no available buffer). The buffers
      // are kept in a fixed-size array (size-dimension,
      // fNumBuffers) of a fixed
      // size array (available buffers dimension,
      // fgNumBuckets) of char*.
      // The size-to-index mapping is done logarithmically,
      // which increases the probability of re-using buffers,
      // and allows the size-dimension array to span a large
      // amount of possible buffer sizes (up to
      // (1<<7)*1024bytes for
      // fgChunksize == 1024,
      // fgNumBuffers == 7-1.
      // Buffer fill stands after stress.cxx(30):
      // bucket: maxfill + "maxReallocBecausePoolTooSmall" (maxAllocBecausePoolTooSmall)
      // 0: 32 + 200 (250)
      // 1: 24 + 0 (0)
      // 2: 2 ...
      // 3: 1
      // 4: 6
      // 5: 6
      // 6: 0
      // 7: 0
      // 8: 0
      class G__BufferReservoir {
      public:
         class Bucket {
         public:
            Bucket():
               fBuffers(0), fWatermark(0), fNumBuffers(0)
            {}
            ~Bucket() {
               // delete all buffers
               char* buf;
               while ((buf = pop()))
                  delete [] buf;
               delete [] fBuffers;
            }

            void init(size_t numBuffers) {
               fNumBuffers = numBuffers;
               fBuffers = new Buffer_t[numBuffers];
               fWatermark = fBuffers + numBuffers;
            }

            bool push(char* buf) {
               if (fWatermark == fBuffers) {
                  return false;
               }
               *(--fWatermark) = buf;
               return true;
            }

            char* pop() {
               if (fWatermark < fBuffers + fNumBuffers) {
                  return *(fWatermark++);
               }
               return 0;
            }
         private:
            typedef char* Buffer_t;

            Buffer_t* fBuffers; // array of buffers,
            Buffer_t* fWatermark; // most recently filled slot
            size_t fNumBuffers; // size of fBuffers
         };

      private:
         G__BufferReservoir() {
            static size_t numBuffers[fgNumBuckets] = {256, 64, 16, 8, 4, 2, 1};
            for (size_t i = 0; i < fgNumBuckets; ++i) {
               fMap[i].init(numBuffers[i]);
            }
            fgIsInitialized = true;
         }

         ~G__BufferReservoir() {
            fgIsInitialized = false;
         }

      public:
         static G__BufferReservoir& Instance()
         {
            // Return the static BufferReservoir
            static G__BufferReservoir sReservoir;
            return sReservoir;
         }

         static char logtwo(unsigned char i)  {
            // Return the index of the highest set bit.
            // A fast imprecise version of log(i) working up to 8 bit i.
            // i must be > 0
            const static char msb[256] = {
#define G__FASTALLOC_MSBx16(n) n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n
               -1, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
               G__FASTALLOC_MSBx16(4),
               G__FASTALLOC_MSBx16(5), G__FASTALLOC_MSBx16(5),
               G__FASTALLOC_MSBx16(6), G__FASTALLOC_MSBx16(6), G__FASTALLOC_MSBx16(6),
               G__FASTALLOC_MSBx16(6),
               G__FASTALLOC_MSBx16(7), G__FASTALLOC_MSBx16(7), G__FASTALLOC_MSBx16(7),
               G__FASTALLOC_MSBx16(7), G__FASTALLOC_MSBx16(7), G__FASTALLOC_MSBx16(7),
               G__FASTALLOC_MSBx16(7), G__FASTALLOC_MSBx16(7)
            };
#undef G__FASTALLOC_MSBx16
            return msb[i];
         }

         static int bucket(size_t size)  {
            // Get the bucket index for a given buffer size.
            //   0:               1 ..   fgChunkSize
            //   1:   fgChunkSize+1 .. 2*fgChunkSize
            //   2: 2*fgChunkSize+1 .. 4*fgChunkSize
            //   3: 4*fgChunkSize+1 .. 8*fgChunkSize
            // ...
            if (!size || !fgIsInitialized) return -1;
            const size_t b = (size - 1) / fgChunkSize;
            if (b > (1L << (fgNumBuckets + 1)))
               return -1;
            int buck = 0;
            if (b && b < 256) {
               buck = logtwo((unsigned char)b) + 1;
            } else if (fgNumBuckets > 8 && buck == 0) {
               // 16 bits is enough, and this expression can be optimized
               // away at compile time.
               buck = 8 + logtwo((unsigned char)(b / 256));
            }
            if (buck >= (int)fgNumBuckets)
               return -1;
            return buck;
         }

         bool push(size_t cap, char* buf) {
            // add buf into capacity cap's bucket; return false if there is no space
            const int buck = bucket(cap);
            if (buck == -1) return false;
            return fMap[buck].push(buf);
         }

         char* pop(size_t& size) {
            // retrieve a buffer of given size, adjusting it to the bucket allocation size.
            const int buck = bucket(size);
            //printf("size=%d, buck=%d\n", size, buck);
            if (buck < 0) {
               return 0;
            }
            size = bucketallocsize(buck);
            return fMap[buck].pop();
         }

         static size_t bucketallocsize(int bucket) {
            // Determine the allocation size that is used for
            // the specified bucket index; see bucket().
            return (1 << bucket) * fgChunkSize;
         }

      private:
         static bool fgIsInitialized;
         static const size_t fgChunkSize = 1024;
         static const size_t fgNumBuckets = 7;
         Bucket fMap[fgNumBuckets]; // the buckets
      };
      bool G__BufferReservoir::fgIsInitialized = false;

   } // Internal
} // Cint

using namespace Cint::Internal;


G__FastAllocString::G__FastAllocString(const char* s)
{
   // Construct from a character array, using the character array's
   // length plus 32 as the initial buffer capacity.
   size_t len = s ? strlen(s) + 1 : 1024;
   fCapacity = len + 32;
   fBuf = GetBuf(fCapacity);
   if (s)
      memcpy(fBuf, s, len);
   else
      fBuf[0] = 0;
}

G__FastAllocString::G__FastAllocString(const G__FastAllocString& other) 
{
   // Construct from another G__FastAllocString, using the
   // other string's length plus 32 as the initial buffer capacity.
   size_t len = strlen(other) + 1;
   fCapacity = len + 32;
   fBuf = GetBuf(fCapacity);
   memcpy(fBuf, other, len);
}

G__FastAllocString::~G__FastAllocString()
{
   // Give our buffer back to the BufMap, i.e. make it available again.
   if (!G__BufferReservoir::Instance().push(Capacity(), fBuf)) {
      delete [] fBuf;
   }
}

char* G__FastAllocString::GetBuf(size_t &size)
{
   // Return a buffer of given size (or larger).
   // If there is one in the map, return that one, otherwise allocatea new one.
   // When entering GetBuf 'size' is the requested size in bytes
   // it is then updated to return the size corresponding to the bucket number,
   // or -1 if no suitable buffer could be extracted from the pool.

   // Look for an existing bucket and update with the bucket index.
   char* buf = G__BufferReservoir::Instance().pop(size);
   if (!buf) {
      buf = new char[size];
   }
   return buf;
}

int G__FastAllocString::FormatArgList(const char *fmt, va_list args)
{
   // sprintf into this string, resizing until it fits.
   if (!fmt) {
      fBuf[0] = 0;
      return 0;
   }
   int result = -1;
   int bucket_req = -2;

   while (result == -1 && bucket_req != -1)
   {
#ifdef _MSC_VER
      result = _vsnprintf(fBuf, fCapacity, fmt, args);
#else
      result = vsnprintf(fBuf, fCapacity, fmt, args);
#endif
      if (result == -1) {
         if (bucket_req == -2)
            bucket_req = G__BufferReservoir::bucket(fCapacity);
         if (bucket_req != -1) {
            // we had a valid bucket, increase it
            ++bucket_req;
            ResizeNoCopy( bucket_req );
         }
      }
   }
   return result;
}

G__FastAllocString& G__FastAllocString::Format(const char *fmt, ...)
{
   // sprintf into this string, resizing until it fits.
   va_list args;
   va_start(args, fmt);
   FormatArgList(fmt, args);
   va_end(args);
   return *this;
}

int G__FastAllocString::FormatArgList(size_t offset, const char *fmt, va_list args)
{
   // sprintf into this string, resizing until it fits.
   if (!fmt) {
      fBuf[0] = 0;
      return 0;
   }
   int result = -1;
   int bucket_req = -2;
   
   while (result == -1 && bucket_req != -1)
   {
#ifdef _MSC_VER
      result = _vsnprintf(fBuf + offset, fCapacity - offset, fmt, args);
#else
      result = vsnprintf(fBuf + offset, fCapacity - offset, fmt, args);
#endif
      if (result == -1) {
         if (bucket_req == -2)
            bucket_req = G__BufferReservoir::bucket(fCapacity);
         if (bucket_req != -1) {
            // we had a valid bucket, increase it
            ++bucket_req;
            Resize( bucket_req );
         }
      }
   }
   return result;
}

G__FastAllocString& G__FastAllocString::Format(size_t offset, const char *fmt, ...)
{
   // sprintf into this string, resizing until it fits.
   va_list args;
   va_start(args, fmt);
   if (offset > Capacity()) {
      Resize(offset+strlen(fmt)*2); // The *2 is a fudge factor ..
   }
   FormatArgList(offset, fmt, args);
   va_end(args);
   return *this;
}

void G__FastAllocString::Replace(size_t where, const char *replacement)
{
   // Replace the content of the string from 'where' to the end of the string
   // with 'replacement'.
   
   if (replacement == 0) {
      if (where < Capacity()) {
         fBuf[where] = '\0';
      }
   } else {
      size_t repl_len = strlen(replacement) + 1;
      Resize(where + repl_len);
      memcpy(fBuf + where, replacement, repl_len);
   }
}

void G__FastAllocString::ResizeToBucketNoCopy(int newbucket)
{
   // Extend the size used by this buffer to at least newsize.
   // This does NOT copy the content.

   size_t cap = G__BufferReservoir::bucketallocsize(newbucket);
   if (cap > Capacity()) {
      ResizeNoCopy(cap);
   }
}

G__FastAllocString& G__FastAllocString::operator=(const char* s) {
   // Assign a string. If necessary, resize the buffer.
   if (!s) {
      fBuf[0] = 0;
      return *this;
   }
   size_t len = strlen(s) + 1;
   if (len > Capacity()) {
      ResizeNoCopy(len);
   }
   memcpy(fBuf, s, len);
   return *this;
}

G__FastAllocString& G__FastAllocString::operator+=(const char* s) {
   // Assign a string. If necessary, resize the buffer.
   if (!s) {
      return *this;
   }
   size_t len = strlen(s);
   size_t mylen = strlen(fBuf);
   Resize(len + 1 + mylen);
   memcpy(fBuf + mylen, s, len + 1);
   return *this;
}

G__FastAllocString& G__FastAllocString::Swap(G__FastAllocString& other) {
   // Swap this and other string.
   char* tmpBuf = fBuf;
   fBuf = other.fBuf;
   other.fBuf = tmpBuf;
   size_t tmpCap = fCapacity;
   fCapacity = other.fCapacity;
   other.fCapacity = tmpCap;
   return *this;
}

void G__FastAllocString::ResizeNoCopy(size_t cap)
{
   // Adjust the capacity so at least cap characters could be
   // stored. Capacity() will be >= cap after this call.
   // This does NOT copy the content.

   if (cap < Capacity())
      return;

   char *newbuf = GetBuf(cap);

   if (!G__BufferReservoir::Instance().push(fCapacity, fBuf))
      delete [] fBuf;

   fBuf = newbuf;
   fCapacity = cap;
}

void G__FastAllocString::Resize(size_t cap)
{
   // Adjust the capacity so at least cap characters could be
   // stored. Capacity() will be >= cap after this call.

   if (cap < Capacity())
      return;

   G__FastAllocString tmp(cap);
   // we cannot rely on data() being 0-terminated.
   memcpy(tmp.fBuf, data(), Capacity());
   Swap(tmp);
}
