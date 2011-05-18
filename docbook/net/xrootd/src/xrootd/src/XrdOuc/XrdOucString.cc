/******************************************************************************/
/*                                                                            */
/*                     X r d O u c S t r i n g . h h                          */
/*                                                                            */
/* (c) 2005 F. Furano (INFN Padova), G. Ganis (CERN)                          */
/*     All Rights Reserved. See XrdInfo.cc for complete License Terms         */
/******************************************************************************/

//         $Id$

const char *XrdOucStringCVSID = "$Id$";

#include <stdio.h>
#include <string.h>
#include <climits>

#include <XrdOuc/XrdOucString.hh>

/******************************************************************************/
/*                                                                            */
/*  Light string manipulation class                                           */
/*                                                                            */
/******************************************************************************/

#define kMAXINT64LEN   25

#if !defined(WINDOWS)
//
// Macro for 'form'-like operations
#define XOSINTFORM(f,b) \
   int buf_len = 256; \
   va_list ap; \
   va_start(ap, f); \
again: \
   b = (char *)realloc(b, buf_len); \
   int n = vsnprintf(b, buf_len, f, ap); \
   if (n == -1 || n >= buf_len) { \
      if (n == -1) \
         buf_len *= 2; \
      else \
         buf_len = n+1; \
      va_end(ap); \
      va_start(ap, f); \
      goto again; \
   } \
   va_end(ap);
// End-Of-Macro for 'form'-like operations
#endif

// Default blksize for (re-)allocations; active if > 0.
// Use XrdOucString::setblksize() to activate
int XrdOucString::blksize = -1;

//________________________________________________________________________
int XrdOucString::adjust(int ls, int &j, int &k, int nmx)
{
   // Check indeces and return effective length
   // If nmx > 0, indecs are adjusted to get the effective length
   // smaller than nmx.

   // Check range for beginning
   j = (j < 0) ? 0 : j;
   // Check range for end 
   k   = (k == -1 || k > (ls-1)) ? (ls-1) : k; 
   // The new length
   int nlen = k - j + 1;
   nlen = (nlen > 0) ? nlen : 0;
   // Check max, if required
   if (nmx > 0 && nmx < nlen) {
      k = j + 1 + nmx;
      nlen = nmx;
   }
   // We are done   
   return nlen;
}

//________________________________________________________________________
char *XrdOucString::bufalloc(int nsz)
{
   // Makes sure that the internal capacity is enough to contain
   // 'nsz' bytes (including the null-termination).
   // Buffer is allocated if not yet existing or reallocated if
   // necessary.
   // If 'nsz' is negative or null, existing buffer is freed, if any
   // Returns pointer to buffer.

   char *nstr = 0;

   // New size must be positive; if not, cleanup
   if (nsz <= 0) {
      if (str) free(str);
      init();
      return nstr;
   }

   int sz = nsz;
   // Check the blksize option is activated
   if (blksize > 1) {
      int blks = nsz / blksize;
      sz = (blks+1) * blksize;
   }

   // Resize, if different from what we have
   if (sz != siz) {
      if ((nstr = (char *)realloc(str, sz)))
         siz = sz;
   } else
      // Do nothing
      nstr = str;

   // We are done
   return nstr;
}

//___________________________________________________________________________
XrdOucString::XrdOucString(const char c, int ls)
{
   // Constructor
   // Create space to store char c as a null-terminated string.
   // If ls > 0 create space for ls+1 bytes.

   init();

   // If required, allocate the buffer to the requested size
   if (ls > 0)
      str = bufalloc(ls+1);
   else
      str = bufalloc(2);
   if (str) {
      str[0] = c;
      str[1] = 0;
      len = 1;
   }
}

//___________________________________________________________________________
XrdOucString::XrdOucString(const char *s, int ls)
{
   // Constructor
   // Create space to store the null terminated string s.
   // If ls > 0 create space for ls+1 bytes, store the first
   // ls bytes of s (truncating, if needed), and null-terminate.
   // This is useful to import non null-terminated string buffers
   // of known length.

   init();

   // If required, allocate the buffer to the requested size
   if (ls > 0)
      str = bufalloc(ls+1);
   int lr = s ? strlen(s) : 0;
   if (lr >= 0)
      assign(s,0,ls-1);
}

//___________________________________________________________________________
XrdOucString::XrdOucString(const XrdOucString &s)
{
   // Copy constructor

   init();
   assign(s.c_str(),0,-1);
}

//______________________________________________________________________________
XrdOucString::XrdOucString(const XrdOucString &s, int j, int k, int ls)
{
   // Copy constructor (portion of string s: from j to k, inclusive)

   init();
   // If required, allocate the buffer to the requested size
   if (ls > 0)
      str = bufalloc(ls+1);

   int lr = s.length();
   if (lr > 0) {
      // Adjust range (to fit in the allocated buffer, if any)
      if (adjust(lr, j, k, ls) > 0) 
         // assign the string portion
         assign(s.c_str(),j,k);
   }
}

//___________________________________________________________________________
XrdOucString::~XrdOucString()
{
   // Destructor

   if (str) free(str);
}

//___________________________________________________________________________
void XrdOucString::setbuffer(char *buf)
{
   // Adopt buffer 'buf'

   if (str) free(str);
   init();
   if (buf) {
      str = buf;
      len = strlen(buf);
      siz = len + 1;
      str = (char *)realloc(str, siz);
   }
}

#if !defined(WINDOWS)
//______________________________________________________________________________
int XrdOucString::form(const char *fmt, ...)
{
   // Recreate the string according to 'fmt' and the arguments
   // Return -1 in case of failure, or the new length.

   // Decode the arguments
   XOSINTFORM(fmt, str);
   siz = buf_len;

   // Re-adjust the length
   len = strlen(str);
   str = bufalloc(len+1);

   // Return the new length (in n)
   return n;
}

//______________________________________________________________________________
int XrdOucString::form(XrdOucString &str, const char *fmt, ...)
{
   // Format a string in 'str' according to 'fmt' and the arguments

   // Decode the arguments
   char *buf = 0;
   XOSINTFORM(fmt, buf);

   // Adopt the new formatted buffer in the string
   str.setbuffer(buf);

   // Done
   return n;
}
#endif

//______________________________________________________________________________
int XrdOucString::find(const char c, int start, bool forward)
{
   // Find index of first occurence of char c starting from position start
   // Return index if found, STR_NPOS if not.

   int rc = STR_NPOS;

   // STR_NPOS indicates start from the end
   if (start == STR_NPOS)
      start = len - 1;

   // Make sure start makes sense
   if (start < 0 || start > (len-1))
      return rc;

   // Now loop
   int i = start;
   if (forward) {
      // forward search
      for (; i < len; i++) {
         if (str[i] == c)
            return i;
      }
   } else {
      // backward search
      for (; i >= 0; i--) {
         if (str[i] == c)
            return i;
      }
   }

   // Nothing found
   return rc;
}

//______________________________________________________________________________
int XrdOucString::find(XrdOucString s, int start)
{
   // Find index of first occurence of string s, starting
   // from position start.
   // Return index if found, STR_NPOS if not.

   return find((const char *)s.c_str(),start);
}

//______________________________________________________________________________
int XrdOucString::find(const char *s, int start)
{
   // Find index of first occurence of null-terminated string s, starting
   // from position start.
   // Return index if found, STR_NPOS if not.

   int rc = STR_NPOS;

   // Make sure start makes sense
   if (start < 0 || start > (len-1))
      return rc;

   // Make sure the string is defined
   if (!s)
      return rc;

   // length of substring
   int ls = strlen(s);

   // if only one meaningful char, use dedicated method
   if (ls == 1)
      return find(s[0],start); 

   // Make sure that it can fit
   if (ls > (len-start))
      return rc;

   // Now loop
   int i = start;
   for (; i < len; i++) {
      if (str[i] == s[0])
         if (!strncmp(str+i+1,s+1,ls-1))
            return i;
   }

   // Nothing found
   return rc;
}

//______________________________________________________________________________
int XrdOucString::rfind(XrdOucString s, int start)
{
   // Find index of first occurence of string s in backward
   // direction starting from position start.
   // If start == STR_NPOS, search starts from end of string (default).
   // Return index if found, STR_NPOS if not.

   return rfind(s.c_str(),start);
}

//______________________________________________________________________________
int XrdOucString::rfind(const char *s, int start)
{
   // Find index of first occurence of null-terminated string s in 
   // backwards direction starting from position start.
   // If start == STR_NPOS, search starts from end of string (default).
   // Return index if found, STR_NPOS if not.

   int rc = STR_NPOS;

   // STR_NPOS indicates start from the end
   if (start == STR_NPOS)
      start = len - 1;

   // Make sure start makes sense
   if (start < 0 || start > (len-1))
      return rc;

   // Make sure the string is defined
   if (!s)
      return rc;

   // length of substring
   int ls = strlen(s);

   // if only one meaningful char, use dedicated method
   if (ls == 1)
      return find(s[0],start,0); 

   // Make sure that it can fit
   if (ls > len)
      return rc;

   // Start from the first meaningful position 
   if (ls > (len-start))
      start = len-ls;

   // Now loop
   int i = start;
   for (; i >= 0; i--) {
      if (str[i] == s[0])
         if (!strncmp(str+i+1,s+1,ls-1))
            return i;
   }

   // Nothing found
   return rc;
}

//______________________________________________________________________________
bool XrdOucString::endswith(char c) 
{
   // returns 1 if the stored string ends with string s

   return (rfind(c) == (int)(len-1));
}

//______________________________________________________________________________
bool XrdOucString::endswith(const char *s) 
{
   // returns 1 if the stored string ends with string s

   return (s ? (rfind(s) == (int)(len-strlen(s))) : 0);
}

//___________________________________________________________________________
int XrdOucString::matches(const char *s, char wch)
{
   // Check if local string is compatible with 's' which may
   // contain wild char wch (default: '*'). For example, if local string
   // is 'mouse.at.home' the match will be true for 'mouse.*',
   // and false for 'mouse.*.cinema' .
   // If does not contain wild characters, this is just a comparison
   // based on strncmp.
   // Returns the number of characters matching or 0.

   // Make sure s is defined and we have a local string
   if (!s || !str)
      return 0;

   // string size
   int ls = strlen(s);

   // If no wild card, just make a simple strcmp comparison
   if (!strchr(s,wch)) {
      if (!strcmp(str,s))
         return ls;
      else
         return 0;
   }

   // If s == wch the match is always true
   if (ls == 1)
      return 1;

   int rc = 1;
   // Starting position for the check
   int cs = 0;

   // token delimiters and size
   int tb = 0;                  // begin
   char *ps = (char *)strchr(s+tb,wch);
   bool next = 1;
   while (next) {

      // token end
      int te = ps ? (ps - s) : ls;
      // token size
      int ts = te - tb;

      if (ts) {
         bool found = 0;
         while (cs < len) {
            if (!strncmp(str+cs,s+tb,ts)) {
               cs += ts;
               found = 1;
               break;
            }
            cs++;
         }
         if (!found) {
            rc = 0;
            break;
         }
      }
      // next token begin, if any
      tb = te + 1;
      ps = (tb < ls) ? (char *)strchr(s+tb, wch) : 0;
      next = (ps || (tb < ls)) ? 1 : 0;
   }

   // If s does not end with a wild card
   // make sure that everything has been checked
   if (s[ls-1] != wch && cs < len)
      rc = 0;

   // The number of chars matching is the number of chars in s
   // which are not '*'
   int nm = 0;
   if (rc > 0) {
      nm = ls;
      int n = ls;
      while (n--) {
         if (s[n] == wch) nm--;
      }
   }

   return nm;
}

//______________________________________________________________________________
void XrdOucString::assign(const char *s, int j, int k)
{
   // Assign portion of buffer s to local string.
   // For k == -1 assign all string starting from position j (inclusive).
   // Use j == 0 and k == -1 to assign the full string.

   int ls = s ? strlen(s) : 0;
   if (!s) {
      // We are passed an empty string
      if (str) {
         // empty the local string, leaving capacity as it is
         str[0] = 0;
         len = 0;
      }
   } else {
      // Adjust range and get length of portion to copy
      int nlen = adjust(ls, j, k);
      // Resize, if needed
      if (nlen > (siz-1)) 
         str = bufalloc(nlen+1);
      if (str) {
         if (nlen > 0) {
            strncpy(str,s+j,nlen);
            str[nlen] = 0;
            len = nlen;
         } else {
            // empty the local string, leaving capacity as it is
            str[0] = 0;
            len = 0;
         }
      } 
   }
}

//______________________________________________________________________________
void XrdOucString::assign(const XrdOucString s, int j, int k)
{
   // Assign portion of buffer s to local string.

   assign(s.c_str(),j,k);
}

//___________________________________________________________________________
int XrdOucString::keep(int start, int size)
{
   // Keep size bytes starting from position start
   // If size == 0, keep any bytes from start on.
   // Return number of bytes kept ( <=size )

   int rc = 0;

   // Make sure start makes sense
   int st = start;
   if (st < 0 || st > (len-1))
      return rc;

   // Make sure size makes sense
   if (size < 0)
      return rc;
   int nlen = 0;
   if (size == 0) {
      nlen = len - st;
   } else {
      nlen = (size > (len - st)) ? (len - st) : size;
   }

   // Do nothing if all the bytes requested
   if (nlen >= len)
      return len;

   // Allocated new string
   if (nlen > (siz-1))
      str = bufalloc(nlen+1);
   if (str) {
      // Copy the bytes
      memmove(str,str+st,nlen);
      // Null terminate
      str[nlen] = 0;
      // Assign new string   
      len = nlen;
      // Return number of bytes kept
      return nlen;
   } else
      return rc;
}

//___________________________________________________________________________
void XrdOucString::append(const char *s)
{
   // Append string pointed by s to local string.
   // Memory is reallocated.

   return insert(s);
}

//___________________________________________________________________________
void XrdOucString::append(const XrdOucString s)
{
   // Append string s to local string.
   // Memory is reallocated.

   return insert(s);
}

//___________________________________________________________________________
void XrdOucString::append(const char c)
{
   // Append char c to local string.
   // Memory is reallocated.

   return insert(c);
}

//___________________________________________________________________________
void XrdOucString::append(const int i)
{
   // Append string representing integer i to local string.

   return insert(i);
}

//___________________________________________________________________________
void XrdOucString::insert(const char *s, int start, int ls)
{
   // Insert null-terminated string pointed by s in local string starting
   // at position start (default append, i.e. start == len).
   // Memory is reallocated.
   // If ls > 0, insert only the first ls bytes of s

   // Check start
   int at = start;
   at = (at < 0 || at > len) ? len : at; 

   if (s) {
      int lstr = (ls > 0) ? ls : strlen(s);
      if (str) {
         int lnew = len + lstr;
         if (lnew > (siz-1))
            str = bufalloc(lnew+1);
         if (str) {
            // Move the rest of the existing string, if any
            if (at < len)
               memmove(str+at+lstr,str+at,(len-at));
            // Add new string now
            memcpy(str+at,s,lstr);
            // Null termination
            str[lnew] = 0;
            len = lnew;
         }
      } else {
         if ((str = bufalloc(lstr+1))) {
            strncpy(str,s,lstr);
            str[lstr] = 0;
            len = lstr;
         }
      }
   }
}

//___________________________________________________________________________
void XrdOucString::insert(const XrdOucString s, int start)
{
   // Insert string s in local string starting at position start (default
   // append, i.e. start == len).

   return insert(s.c_str(), start);
}

//___________________________________________________________________________
void XrdOucString::insert(const char c, int start)
{
   // Insert char c in local string starting at position start (default
   // append, i.e. start == len).

   char sc[2] = {0};
   sc[0] = c;
   return insert((const char *)&sc[0], start);
}

//___________________________________________________________________________
void XrdOucString::insert(const int i, int start)
{
   // Insert string representing integer i in local string starting at
   // position start (default

   char si[kMAXINT64LEN] = {0};
   sprintf(si,"%d",i);
   return insert((const char *)&si[0], start);
}

//___________________________________________________________________________
int XrdOucString::replace(const XrdOucString s1, const char *s2, int from, int to)
{
   // Replace any occurrence of s1 with s2 from position 'from' to position
   // 'to' (inclusive).
   // Return signed size of length modification (in bytes)

   return replace(s1.c_str(),s2,from,to);
}

//___________________________________________________________________________
int XrdOucString::replace(const char *s1, const XrdOucString s2, int from, int to)
{
   // Replace any occurrence of s1 with s2 from position 'from' to position
   // 'to' (inclusive).
   // Return signed size of length modification (in bytes)

   return replace(s1,s2.c_str(),from,to);
}

//___________________________________________________________________________
int XrdOucString::replace(const XrdOucString s1,
                           const XrdOucString s2, int from, int to)
{
   // Replace any occurrence of s1 with s2 from position 'from' to position
   // 'to' (inclusive).
   // Return signed size of length modification (in bytes)

   return replace(s1.c_str(),s2.c_str(),from,to);
}

//___________________________________________________________________________
int XrdOucString::replace(const char *s1, const char *s2, int from, int to)
{
   // Replace any occurrence of s1 with s2 from position 'from' to position
   // 'to' (inclusive).
   // Return signed size of length modification (in bytes)

   // We must have something to replace
   if (!str || len <= 0)
      return 0;
   
   // The string to replace must be defined and not empty
   int l1 = s1 ? strlen(s1) : 0;
   if (l1 <= 0)
      return 0;

   // Check and adjust indeces
   if (adjust(len,from,to) <= 0)
      return 0;

   // length of replacing string
   int l2 = s2 ? strlen(s2) : 0;

   // If new string is longer we need number of occurencies
   int nr = 0;
   if (l1 < l2) {
      int at = find(s1,from);
      while (at > -1 && at <= (to-l1+1)) {
         nr++;
         at = find(s1,at+l1);
      }
   }

   // New size
   int nlen = (nr > 0) ? (len + nr*(l2-l1)) : len ;

   // Reallocate, if needed
   if (nlen > (siz-1))
      str = bufalloc(nlen+1);

   // Now act
   int dd = l2-l1;
   int dl = 0;
   if (str) {
      if (dd < 0) {
         int nc = 0;
         int at = find(s1,from);
         while (at > -1 && at <= (to-l1+1)) {
            int atn = find(s1,at+l1);
            atn = (atn == -1 || atn > (to-l1+1)) ? len : atn;
            int ln = atn - at - l1;
            char *pc = str+at+nc*dd;
            if (l2 > 0)
               memcpy(pc,s2,l2);
            if (ln > 0)
               memmove(pc+l2,str+at+l1,ln);
            nc++;
            at = atn;
         }
         dl = nc*dd;
      } else if (dd == 0) {
         int at = find(s1,from);
         while (at > -1 && at <= (to-l1+1)) {
            memcpy(str+at,s2,l2);
            at = find(s1,at+l1);
         }
      } else if (dd > 0) {
         int nc = nr;
         int at = rfind(s1,to);
         int atn = len;
         while (at > -1 && at >= from) {
            int ln = atn - at - l1;
            char *pc = str + at + l1 + nc*dd;
            if (ln > 0)
               memmove(pc,str+at+l1,ln);
            if (l2 > 0)
               memcpy(pc-l2,s2,l2);
            nc--;
            atn = at;
            at = rfind(s1,at-l1);
         }
         dl = nr*dd;
      }
   }

   // Variation of string length
   len += dl;
   // Insure null-termination
   str[len] = 0;
   // We are done
   return dl;
}

//___________________________________________________________________________
int XrdOucString::erase(int start, int size)
{
   // Remove size bytes starting from position start.
   // If size == 0, remove any bytes from start on.
   // Return number of bytes removed ( <=size )

   int rc = 0;

   // Make sure start makes sense
   int st = start;
   if (st < 0 || st > (len-1))
      return rc;

   // Make sure size makes sense
   if (size < 0)
      return rc;
   int nrem = 0;
   if (size == 0) {
      nrem = len - st;
   } else {
      nrem = (size > (len-st)) ? (len-st) : size;
   }
   // Do nothing if no byte removal has been requested
   if (nrem <= 0)
      return rc;
   // Calculate new length and allocated new string
   int nlen = len - nrem;
   // Copy the remaining bytes, if any
   if (len-st-nrem) 
      memmove(str+st,str+st+nrem,len-st-nrem);
   // Null terminate
   str[nlen] = 0;
   // Assign new length
   len = nlen;
   // Return number of bytes removed
   return nrem;
}

//___________________________________________________________________________
int XrdOucString::erase(const char *s, int from, int to)
{
   // Remove any occurence of string s within from and to inclusive.
   // Use from == 0 and to == -1 to remove all occurences (default).
   // Return number of bytes removed ( <=size )

   return -replace(s,0,from,to);
}

//___________________________________________________________________________
int XrdOucString::erase(XrdOucString s, int from, int to)
{
   // Remove any occurence of string s within from and to inclusive.
   // Use from == 0 and to == -1 to remove all occurences (default).
   // Return number of bytes removed ( <=size )

   return -replace(s.c_str(),0,from,to);
}

//___________________________________________________________________________
void XrdOucString::lower(int start, int size)
{
   // Set to lower case size chars starting from position start.
   // If size == 0, lower all bytes from start on.

   // Make sure start makes sense
   int st = start;
   if (st < 0 || st > (len-1))
      return;

   // Make sure size makes sense
   if (size < 0)
      return;
   int nlw = 0;
   if (size == 0) {
      nlw = len - st;
   } else {
      nlw = (size > (len-st)) ? (len-st) : size;
   }

   // Do nothing if no byte removal has been requested
   if (nlw <= 0)
      return;

   // Set to lower
   int i = st;
   for (; i < st + nlw ; i++ ) {
      if (str[i] > 0x40 && str[i] < 0x5b)
         str[i] += 0x20; 
   }
}

//___________________________________________________________________________
void XrdOucString::upper(int start, int size)
{
   // Set to upper case size chars starting from position start.
   // If size == 0, upper all bytes from start on.

   // Make sure start makes sense
   int st = start;
   if (st < 0 || st > (len-1))
      return;

   // Make sure size makes sense
   if (size < 0)
      return;
   int nup = 0;
   if (size == 0) {
      nup = len - st;
   } else {
      nup = (size > (len-st)) ? (len-st) : size;
   }

   // Do nothing if no byte removal has been requested
   if (nup <= 0)
      return;

   // Set to upper
   int i = st;
   for (; i < st + nup ; i++ ) {
      if (str[i] > 0x60 && str[i] < 0x7b)
         str[i] -= 0x20; 
   }
}

//___________________________________________________________________________
void XrdOucString::hardreset()
{
   // Reset string making sure to erase completely the information.

   if (str) {
      volatile char *buf = 0;
      for (buf = (volatile char *)str; len; buf[--len] = 0);
      len = 0;
   }
   len = 0;
}

//___________________________________________________________________________
void XrdOucString::reset(const char c, int j, int k)
{
   // Reset string making sure to erase completely the information.

   j = (j >= 0 && j < siz) ? j : 0;
   k = (k >= j && k < siz) ? k : siz-1;

   if (str) {
      volatile char *buf = (volatile char *)str;
      int i = j;
      for (; i <= k; i++)
         buf[i] = c;
   }
   while (str[len-1] == 0)
      --len; 
}

//______________________________________________________________________________
XrdOucString& XrdOucString::operator=(const int i)
{
   // Assign string representing integer i to local string

   char s[kMAXINT64LEN] = {0};
   sprintf(s,"%d",i);
   assign((const char *)&s[0],0,-1);
   return *this;
}

//______________________________________________________________________________
XrdOucString& XrdOucString::operator=(const char c)
{
   // Assign char c to local string.

   const char s[] = {c,0};
   assign(s,0,-1);
   return *this;
}

//______________________________________________________________________________
XrdOucString& XrdOucString::operator=(const char *s)
{
   // Assign buffer s to local string.

   assign(s,0,-1);

   return *this;
}

//______________________________________________________________________________
XrdOucString& XrdOucString::operator=(const XrdOucString s)
{
   // Assign string s to local string.
   assign(s.c_str(), 0, -1);

   return *this;
}

//______________________________________________________________________________
char &XrdOucString::operator[](int i)
{
   // Return charcater at location i.
   static char c = '\0';

   if (str) {
      if (i > -1 && i < len) 
         return str[i];
      else
         abort();
   }
   return c;
}

//______________________________________________________________________________
XrdOucString operator+(const XrdOucString &s1, const char *s)
{
   // Return string resulting from concatenation

   XrdOucString ns(s1);
   if (s && strlen(s))
      ns.append(s);
   return ns;
}

//______________________________________________________________________________
XrdOucString operator+(const XrdOucString &s1, const XrdOucString &s)
{
   // Return string resulting from concatenation

   XrdOucString ns(s1);
   if (s.length())
      ns.append(s);
   return ns;
}

//______________________________________________________________________________
XrdOucString operator+(const XrdOucString &s1, const char c)
{
   // Return string resulting from concatenation of local string
   // and char c

   XrdOucString ns(s1);
   ns.append(c);
   return ns;
}

//______________________________________________________________________________
XrdOucString operator+(const XrdOucString &s1, const int i)
{
   // Return string resulting from concatenation of local string
   // and string representing integer i.

   XrdOucString ns(s1);
   ns.append(i);
   return ns;
}

//______________________________________________________________________________
XrdOucString& XrdOucString::operator+=(const char *s)
{
   // Add string at s to local string.

   if (s && strlen(s))
      this->append(s);
   return *this;
}

//______________________________________________________________________________
XrdOucString& XrdOucString::operator+=(const XrdOucString s)
{
   // Add string s to local string.

   if (s.length())
      this->append(s);
   return *this;
}

//______________________________________________________________________________
XrdOucString& XrdOucString::operator+=(const char c)
{
   // Add char c to local string.

   this->append(c);
   return *this;
}

//______________________________________________________________________________
XrdOucString& XrdOucString::operator+=(const int i)
{
   // Add string representing integer i to local string.

   this->append(i);
   return *this;
}


//______________________________________________________________________________
int XrdOucString::operator==(const char *s)
{
   // Compare string at s to local string: return 1 if matches, 0 if not

   if (s && (strlen(s) == (unsigned int)len))
      if (!strncmp(str,s,len))
         return 1;
   return 0;
}

//______________________________________________________________________________
int XrdOucString::operator==(const XrdOucString s)
{
   // Compare string s to local string: return 1 if matches, 0 if not

   if (s.length() == len)
      if (!strncmp(str,s.c_str(),len))
         return 1;
   return 0;
}

//______________________________________________________________________________
int XrdOucString::operator==(const char c)
{
   // Compare char c to local string: return 1 if matches, 0 if not

   if (len == 1) {
      if (str[0] == c)
         return 1;
   }
   return 0;
}

//______________________________________________________________________________
int XrdOucString::operator==(const int i)
{
   // Compare string representing integer i to local string:
   // return 1 if matches, 0 if not

   char s[kMAXINT64LEN] = {0};
   sprintf(s,"%d",i);
   return (*this == ((const char *)&s[0]));
}

//______________________________________________________________________________
ostream &operator<< (ostream &os, const XrdOucString s)
{
   // Operator << is useful to print a string into a stream

   if (s.c_str()) 
      os << s.c_str();
   else
     os << "";
   return os;
}

//______________________________________________________________________________
XrdOucString const operator+(const char *s1, const XrdOucString s2)
{
   // Binary operator+
   XrdOucString res(s1,s2.length()+strlen(s1));
   res.insert(s2);
   return res;
}

//______________________________________________________________________________
XrdOucString const operator+(const char c, const XrdOucString s)
{
   // Binary operator+
   XrdOucString res(c,s.length()+1);
   res.insert(s);
   return res;
}

//______________________________________________________________________________
XrdOucString const operator+(const int i, const XrdOucString s)
{
   // Binary operator+
   XrdOucString res(s.length()+kMAXINT64LEN);
   res.insert(i);
   res.insert(s);
   return res;
}

//______________________________________________________________________________
int XrdOucString::getblksize()
{
   // Getter for the block size

   return XrdOucString::blksize;
}

//______________________________________________________________________________
void XrdOucString::setblksize(int bs)
{
   // Set for the block size

   XrdOucString::blksize = bs;
}

//______________________________________________________________________________
int XrdOucString::tokenize(XrdOucString &tok, int from, char del)
{
   // Search for tokens delimited by 'del' (def ':') in string s; search starts
   // from 'from' and the token is returned in 'tok'.
   // Returns -1 when there are no more tokens to be analyzed; the length of the
   // last valid token, if there are no more delimiters after 'from'; the next
   // position after the delimiter, when there are left delimiters in the string.
   //
   // This method allows to loop over tokens in this way:
   //
   //    XrdOucString myl = "tok1 tok2 tok3";
   //    char del = ' ';
   //    XrdOucString tok;
   //    int from = 1;
   //    while ((from = myl.tokenize(tok, from, del) != -1) {
   //       // Analyse tok
   //       ...
   //    } 
   //
   // Warning: it may return empty tokens (e.g. in cases like "::"), so 
   // the token length must always be checked.

   // Make sure inputs make sense
   if (len <= 0 || from < 0 || from > (len-1)) 
      return -1;

   // Find delimiter
   int pos = find(del, from);

   // Assign to token
   if (pos == -1 || pos > from) {
      int last = (pos > 0) ? (pos - 1) : -1;
      tok.assign(str, from, last);
   } else
      tok = "";

   int next = pos + 1;
   if (pos == -1) {
      if (tok.length() > 0)
         // So we can analize the last one
         next = len;
      else
         next = pos;
   }

   // return
   return next;
}

//______________________________________________________________________________
bool XrdOucString::isdigit(int from, int to)
{
   // Return true is all chars between from and to (included) are digits

   // Make sure inputs make sense
   if (len <= 0) return 0;

   // Adjust range
   if (from < 0 || from > (len-1)) from = 0;
   if (to < from) to = len - 1;

   char *c = str + from;

   // Skip initial '-'
   if (*c == '-') c++;

   while (c <= str + to) {
      if (*c < 48 || *c > 57) return 0;
      c++;
   }

   return 1;
}

//______________________________________________________________________________
long XrdOucString::atoi(int from, int to)
{
   // Return the long integer corresponding to the number between from and to
   // (included), assuming they are digits (check with 'isdigit()').
   // Return LONG_MAX in case they are not digits

   if (!isdigit(from, to)) return LONG_MAX;

   // Adjust range
   if (from < 0 || from > (len-1)) from = 0;
   if (to < from) to = len - 1;

   // Save end char
   char e = str[to+1];
   str[to+1] = '\0';
   long out = strtol(&str[from], 0, 10);
   str[to+1] = e;
   return out;
}
