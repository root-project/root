// @(#)root/base:$Name:  $:$Id: TString.cxx,v 1.5 2000/11/27 12:23:15 brun Exp $
// Author: Fons Rademakers   04/08/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TString                                                              //
//                                                                      //
// Basic string class.                                                  //
//                                                                      //
// Cannot be stored in a TCollection... use TObjString instead.         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <ctype.h>

#include "snprintf.h"
#include "TString.h"
#include "TBuffer.h"
#include "TError.h"
#include "Bytes.h"


// Amount to shift hash values to avoid clustering

const unsigned kHashShift = 5;

// This is the global null string representation, shared among all
// empty strings.  The space for it is in "gNullRef" which the
// loader will set to zero

static long gNullRef[(sizeof(TStringRef)+1)/sizeof(long) + 1];

// Use macro in stead of the following to side-step compilers (e.g. DEC)
// that generate pre-main code for the initialization of address constants.
// static TStringRef* const gNullStringRef = (TStringRef*)gNullRef;

#define gNullStringRef ((TStringRef*)gNullRef)

// ------------------------------------------------------------------------
//
// In what follows, fCapacity is the length of the underlying representation
// vector. Hence, the capacity for a null terminated string held in this
// vector is fCapacity-1.  The variable fNchars is the length of the held
// string, excluding the terminating null.
//
// The algorithms make no assumptions about whether internal strings
// hold embedded nulls. However, they do assume that any string
// passed in as an argument that does not have a length count is null
// terminated and therefore has no embedded nulls.
//
// The internal string is always null terminated.
//
// ------------------------------------------------------------------------
//
//  This class uses a number of protected and private member functions
//  to do memory management. Here are their semantics:
//
//  TString::Cow();
//    Insure that self is a distinct copy. Preserve previous contents.
//
//  TString::Cow(Ssiz_t nc);
//    Insure that self is a distinct copy with capacity of at
//    least nc. Preserve previous contents.
//
//  TString::Clobber(Ssiz_t nc);
//    Insure that the TStringRef is unshared and has a
//    capacity of at least nc. No need to preserve contents.
//
//  TString::Clone();
//    Make self a distinct copy. Preserve previous contents.
//
//  TString::Clone(Ssiz_t);
//    Make self a distinct copy with capacity of at least nc.
//    Preserve previous contents.
//
// ------------------------------------------------------------------------


//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TStringRef                                                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
TStringRef* TStringRef::GetRep(Ssiz_t capacity, Ssiz_t nchar)
{
   // Static member function returning an empty string representation of
   // size capacity and containing nchar characters.

   if ((capacity | nchar) == 0) {
     gNullStringRef->AddReference();
     return gNullStringRef;
   }
   TStringRef* ret = (TStringRef*)new char[capacity + sizeof(TStringRef) + 1];
   ret->fCapacity = capacity;
   ret->SetRefCount(1);
   ret->Data()[ret->fNchars = nchar] = 0; // Terminating null

   return ret;
}

//______________________________________________________________________________
Ssiz_t TStringRef::First(char c) const
{
   // Find first occurrence of a character c.

   const char* f = strchr(Data(), c);
   return f ? f - Data() : kNPOS;
}

//______________________________________________________________________________
Ssiz_t TStringRef::First(const char* cs) const
{
   // Find first occurrence of a character in cs.

   const char* f = strpbrk(Data(), cs);
   return f ? f - Data() : kNPOS;
}

//______________________________________________________________________________
inline static void Mash(unsigned& hash, unsigned chars)
{
   // Utility used by Hash().

   hash = (chars ^
         ((hash << kHashShift) |
          (hash >> (kBitsPerByte*sizeof(unsigned) - kHashShift))));
}

//______________________________________________________________________________
unsigned TStringRef::Hash() const
{
   // Return a case-sensitive hash value.

   unsigned hv       = (unsigned)Length(); // Mix in the string length.
   unsigned i        = hv*sizeof(char)/sizeof(unsigned);
   const unsigned* p = (const unsigned*)Data();
   {
      while (i--)
         Mash(hv, *p++);                   // XOR in the characters.
   }
   // XOR in any remaining characters:
   if ((i = Length()*sizeof(char)%sizeof(unsigned)) != 0) {
      unsigned h = 0;
      const char* c = (const char*)p;
      while (i--)
         h = ((h << kBitsPerByte*sizeof(char)) | *c++);
      Mash(hv, h);
   }
   return hv;
}

//______________________________________________________________________________
unsigned TStringRef::HashFoldCase() const
{
   // Return a case-insensitive hash value.

   unsigned hv = (unsigned)Length();    // Mix in the string length.
   unsigned i  = hv;
   const unsigned char* p = (const unsigned char*)Data();
   while (i--) {
      Mash(hv, toupper(*p));
      ++p;
   }
   return hv;
}

//______________________________________________________________________________
Ssiz_t TStringRef::Last(char c) const
{
   // Find last occurrence of a character c.

   // cxx under OSF on DEC Alpha needs cast to unsigned char!?
   const char* f = strrchr(Data(), (unsigned char) c);
   return f ? f - Data() : kNPOS;
}


ClassImp(TString)

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TString                                                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

// ------------------- The static data members access --------------------------
Ssiz_t  TString::GetInitialCapacity()    { return fgInitialCapac; }
Ssiz_t  TString::GetResizeIncrement()    { return fgResizeInc; }
Ssiz_t  TString::GetMaxWaste()           { return fgFreeboard; }
//______________________________________________________________________________
TString::TString()
{
   // TString default ctor.

   fData = gNullStringRef->Data();
   gNullStringRef->AddReference();
}

//______________________________________________________________________________
TString::TString(Ssiz_t ic)
{
   // Create TString able to contain ic characters.

   fData = TStringRef::GetRep(ic, 0)->Data();
}

//______________________________________________________________________________
TString::TString(const char* cs)
{
   // Create TString and initialize it with string cs.

   if (cs) {
      Ssiz_t n = strlen(cs);
      fData = TStringRef::GetRep(n, n)->Data();
      memcpy(fData, cs, n);
   } else
      fData = TStringRef::GetRep(0, 0)->Data();
}

//______________________________________________________________________________
TString::TString(const char* cs, Ssiz_t n)
{
   // Create TString and initialize it with the first n characters of cs.

   fData = TStringRef::GetRep(n, n)->Data();
   memcpy(fData, cs, n);
}

//______________________________________________________________________________
void TString::InitChar(char c)
{
   // Initialize a string with a single character.

   fData = TStringRef::GetRep(GetInitialCapacity(), 1)->Data();
   fData[0] = c;
}

//______________________________________________________________________________
TString::TString(char c, Ssiz_t n)
{
   // Initialize the first n locations of a TString with character c.

   fData = TStringRef::GetRep(n, n)->Data();
   while (n--) fData[n] = c;
}

//______________________________________________________________________________
TString::TString(const TSubString& substr)
{
   // Copy a TSubString in a TString.

   Ssiz_t len = substr.IsNull() ? 0 : substr.Length();
   fData = TStringRef::GetRep(AdjustCapacity(len), len)->Data();
   memcpy(fData, substr.Data(), len);
}

//______________________________________________________________________________
TString::~TString()
{
   // Delete a TString. I.e. decrease its reference count. When 0 free space.

   Pref()->UnLink();
}

//______________________________________________________________________________
TString& TString::operator=(char c)
{
   // Assign character c to TString.

   if (!c) {
      Pref()->UnLink();
      gNullStringRef->AddReference();
      fData = gNullStringRef->Data();
      return *this;
   }
   return Replace(0, Length(), &c, 1);
}

//______________________________________________________________________________
TString& TString::operator=(const char* cs)
{
   // Assign string cs to TString.

   if (!cs || (cs && !*cs)) {
      Pref()->UnLink();
      gNullStringRef->AddReference();
      fData = gNullStringRef->Data();
      return *this;
   }
   return Replace(0, Length(), cs, strlen(cs));
}

//______________________________________________________________________________
TString& TString::operator=(const TString& str)
{
   // Assignment operator.

   str.Pref()->AddReference();
   Pref()->UnLink();
   fData = str.fData;
   return *this;
}

//______________________________________________________________________________
TString& TString::operator=(const TSubString& substr)
{
   // Assign a TSubString substr to TString.

   Ssiz_t len = substr.IsNull() ? 0 : substr.Length();
   if (!len) {
      Pref()->UnLink();
      gNullStringRef->AddReference();
      fData = gNullStringRef->Data();
      return *this;
   }
   return Replace(0, Length(), substr.Data(), len);
}

//______________________________________________________________________________
TString& TString::Append(char c, Ssiz_t rep)
{
   // Append character c rep times to string.

   Ssiz_t tot;
   Cow(tot = Length() + rep);
   char* p = fData + Length();
   while (rep--)
      *p++ = c;

   fData[Pref()->fNchars = tot] = '\0';

   return *this;
}

// Change the string capacity, returning the new capacity
//______________________________________________________________________________
Ssiz_t TString::Capacity(Ssiz_t nc)
{
   // Return string capacity. If nc != current capacity Clone() the string
   // in a string with the desired capacity.

   if (nc > Length() && nc != Capacity())
      Clone(nc);

   return Capacity();
}

//______________________________________________________________________________
int TString::CompareTo(const char* cs2, ECaseCompare cmp) const
{
   // Compare a string to char *cs2.

   const char* cs1 = Data();
   Ssiz_t len = Length();
   Ssiz_t i = 0;
   if (cmp == kExact) {
      for (; cs2[i]; ++i) {
         if (i == len) return -1;
         if (cs1[i] != cs2[i]) return ((cs1[i] > cs2[i]) ? 1 : -1);
      }
   } else {                  // ignore case
      for (; cs2[i]; ++i) {
         if (i == len) return -1;
         char c1 = tolower((unsigned char)cs1[i]);
         char c2 = tolower((unsigned char)cs2[i]);
         if (c1 != c2) return ((c1 > c2) ? 1 : -1);
      }
   }
   return (i < len) ? 1 : 0;
}

//______________________________________________________________________________
int TString::CompareTo(const TString& str, ECaseCompare cmp) const
{
   // Compare a string to another string.

   const char* s1 = Data();
   const char* s2 = str.Data();
   Ssiz_t len = str.Length();
   if (Length() < len) len = Length();
   if (cmp == kExact) {
      int result = memcmp(s1, s2, len);
      if (result != 0) return result;
   } else {
      Ssiz_t i = 0;
      for (; i < len; ++i) {
         char c1 = tolower((unsigned char)s1[i]);
         char c2 = tolower((unsigned char)s2[i]);
         if (c1 != c2) return ((c1 > c2) ? 1 : -1);
      }
   }
   // strings are equal up to the length of the shorter one.
   if (Length() == str.Length()) return 0;
   return (Length() > str.Length()) ? 1 : -1;
}

//______________________________________________________________________________
TString TString::Copy() const
{
   // Copy a string.

   TString temp(*this);          // Has increased reference count
   temp.Clone();                 // Distinct copy
   return temp;
}

//______________________________________________________________________________
unsigned TString::Hash(ECaseCompare cmp) const
{
   // Return hash value.

   return (cmp == kExact) ? Pref()->Hash() : Pref()->HashFoldCase();
}

//______________________________________________________________________________
static int MemIsEqual(const char* p, const char* q, Ssiz_t n)
{
   // Returns false if strings are not equal.

   while (n--)
   {
      if (tolower((unsigned char)*p) != tolower((unsigned char)*q))
         return kFALSE;
      p++; q++;
   }
   return kTRUE;
}

//______________________________________________________________________________
Ssiz_t TString::Index(const char* pattern, Ssiz_t plen, Ssiz_t startIndex,
                      ECaseCompare cmp) const
{
   // Search for a string in the TString. Plen is the length of pattern,
   // startIndex is the index from which to start and cmp selects the type
   // of case-comparison.

   Ssiz_t slen = Length();
   if (slen < startIndex + plen) return kNPOS;
   if (plen == 0) return startIndex;
   slen -= startIndex + plen;
   const char* sp = Data() + startIndex;
   if (cmp == kExact) {
      char first = *pattern;
      for (Ssiz_t i = 0; i <= slen; ++i)
         if (sp[i] == first && memcmp(sp+i+1, pattern+1, plen-1) == 0)
            return i + startIndex;
   } else {
      int first = tolower((unsigned char) *pattern);
      for (Ssiz_t i = 0; i <= slen; ++i)
         if (tolower((unsigned char) sp[i]) == first &&
             MemIsEqual(sp+i+1, pattern+1, plen-1))
            return i + startIndex;
   }
   return kNPOS;
}

//______________________________________________________________________________
TString& TString::Prepend(char c, Ssiz_t rep)
{
   // Prepend characters to self.

   Ssiz_t tot = Length() + rep;  // Final string length

   // Check for shared representation or insufficient capacity
   if ( Pref()->References() > 1 || Capacity() < tot ) {
      TStringRef* temp = TStringRef::GetRep(AdjustCapacity(tot), tot);
      memcpy(temp->Data()+rep, Data(), Length());
      Pref()->UnLink();
      fData = temp->Data();
   } else {
      memmove(fData + rep, Data(), Length());
      fData[Pref()->fNchars = tot] = '\0';
   }

   char* p = fData;
   while (rep--)
      *p++ = c;

   return *this;
}

//______________________________________________________________________________
TString& TString::Replace(Ssiz_t pos, Ssiz_t n1, const char* cs, Ssiz_t n2)
{
   // Remove at most n1 characters from self beginning at pos,
   // and replace them with the first n2 characters of cs.

   n1 = TMath::Min(n1, Length()-pos);
   if (!cs) n2 = 0;

   Ssiz_t tot = Length()-n1+n2;  // Final string length
   Ssiz_t rem = Length()-n1-pos; // Length of remnant at end of string

   // Check for shared representation, insufficient capacity,
   // excess waste, or overlapping copy
   if (Pref()->References() > 1 ||
       Capacity() < tot ||
       Capacity() - tot > GetMaxWaste() ||
       (cs && (cs >= Data() && cs < Data()+Length())))
   {
      TStringRef* temp = TStringRef::GetRep(AdjustCapacity(tot), tot);
      if (pos) memcpy(temp->Data(), Data(), pos);
      if (n2 ) memcpy(temp->Data()+pos, cs, n2);
      if (rem) memcpy(temp->Data()+pos+n2, Data()+pos+n1, rem);
      Pref()->UnLink();
      fData = temp->Data();
   } else {
      if (rem) memmove(fData+pos+n2, Data()+pos+n1, rem);
      if (n2 ) memmove(fData+pos   , cs, n2);
      fData[Pref()->fNchars = tot] = 0;   // Add terminating null
   }

   return *this;
}

//______________________________________________________________________________
TString& TString::ReplaceAll(const char *s1, Ssiz_t ls1, const char *s2, Ssiz_t ls2)
{
   // Find & Replace ls1 symbols of s1 with ls2 symbols of s2 if any.

   if (s1 && ls1>0) {
      Ssiz_t index = 0;
      while ((index = Index(s1,ls1,index, kExact)) != kNPOS) {
         Replace(index,ls1,s2,ls2);
         index += ls2;
      }
   }
   return *this;
}

//______________________________________________________________________________
void TString::Resize(Ssiz_t n)
{
   // Resize the string. Truncate or add blanks as necessary.

   if (n < Length())
      Remove(n);                  // Shrank; truncate the string
   else
      Append(' ', n-Length());    // Grew or staid the same
}

//______________________________________________________________________________
TSubString TString::Strip(EStripType st, char c)
{
   // Return a substring of self stripped at beginning and/or end.

   Ssiz_t start = 0;             // Index of first character
   Ssiz_t end = Length();        // One beyond last character
   const char* direct = Data();  // Avoid a dereference w dumb compiler

   if (st & kLeading)
      while (start < end && direct[start] == c)
         ++start;
   if (st & kTrailing)
      while (start < end && direct[end-1] == c)
         --end;
   if (end == start) start = end = kNPOS;  // make the null substring
   return TSubString(*this, start, end-start);
}

//______________________________________________________________________________
TSubString TString::Strip(EStripType st, char c) const
{
   // Just use the "non-const" version, adjusting the return type.

   return ((TString*)this)->Strip(st,c);
}

//______________________________________________________________________________
void TString::ToLower()
{
   // Change string to lower-case.

   Cow();
   register Ssiz_t n = Length();
   register char* p = fData;
   while (n--) {
      *p = tolower((unsigned char)*p);
      p++;
   }
}

//______________________________________________________________________________
void TString::ToUpper()
{
   // Change string to upper case.

   Cow();
   register Ssiz_t n = Length();
   register char* p = fData;
   while (n--) {
      *p = toupper((unsigned char)*p);
      p++;
   }
}

//______________________________________________________________________________
char& TString::operator[](Ssiz_t i)
{
   // Return charcater at location i.

   AssertElement(i);
   Cow();
   return fData[i];
}

//______________________________________________________________________________
void TString::AssertElement(Ssiz_t i) const
{
   // Check to make sure a string index is in range.

   if (i == kNPOS || i >= Length())
      Error("TString::AssertElement",
            "out of bounds: i = %d, Length = %d", i, Length()-1);
}

//______________________________________________________________________________
TString::TString(const char* a1, Ssiz_t n1, const char* a2, Ssiz_t n2)
{
   // Special constructor to initialize with the concatenation of a1 and a2.

   if (!a1) n1=0;
   if (!a2) n2=0;
   Ssiz_t tot = n1+n2;
   fData = TStringRef::GetRep(AdjustCapacity(tot), tot)->Data();
   memcpy(fData,    a1, n1);
   memcpy(fData+n1, a2, n2);
}

//______________________________________________________________________________
Ssiz_t TString::AdjustCapacity(Ssiz_t nc)
{
   // Calculate a nice capacity greater than or equal to nc.

   Ssiz_t ic = GetInitialCapacity();
   if (nc <= ic) return ic;
   Ssiz_t rs = GetResizeIncrement();
   return (nc - ic + rs - 1) / rs * rs + ic;
}

//______________________________________________________________________________
void TString::Clobber(Ssiz_t nc)
{
   // Clear string and make sure it has a capacity of nc.

   if (Pref()->References() > 1 || Capacity() < nc) {
      Pref()->UnLink();
      fData = TStringRef::GetRep(nc, 0)->Data();
   } else
      fData[Pref()->fNchars = 0] = 0;
}

//______________________________________________________________________________
void TString::Clone()
{
   // Make string a distinct copy; preserve previous contents.

   TStringRef* temp = TStringRef::GetRep(Length(), Length());
   memcpy(temp->Data(), Data(), Length());
   Pref()->UnLink();
   fData = temp->Data();
}

//______________________________________________________________________________
void TString::Clone(Ssiz_t nc)
{
   // Make self a distinct copy with capacity of at least nc.
   // Preserve previous contents.

   Ssiz_t len = Length();
   if (len > nc) len = nc;
   TStringRef* temp = TStringRef::GetRep(nc, len);
   memcpy(temp->Data(), Data(), len);
   Pref()->UnLink();
   fData = temp->Data();
}

// ------------------- ROOT I/O ------------------------------------

//______________________________________________________________________________
void TString::FillBuffer(char *&buffer)
{
   // Copy string into I/O buffer.

   UChar_t nwh;
   Int_t   nchars = Length();

   if (nchars > 254) {
      nwh = 255;
      tobuf(buffer, nwh);
      tobuf(buffer, nchars);
   } else {
      nwh = UChar_t(nchars);
      tobuf(buffer, nwh);
   }
   for (int i = 0; i < nchars; i++) buffer[i] = fData[i];
   buffer += nchars;
}

//______________________________________________________________________________
void TString::ReadBuffer(char *&buffer)
{
   // Read string from I/O buffer.

   Pref()->UnLink();

   UChar_t nwh;
   Int_t   nchars;

   frombuf(buffer, &nwh);
   if (nwh == 255)
      frombuf(buffer, &nchars);
   else
      nchars = nwh;

   fData = TStringRef::GetRep(nchars, nchars)->Data();

   for (int i = 0; i < nchars; i++) frombuf(buffer, &fData[i]);
}

//______________________________________________________________________________
Int_t TString::Sizeof() const
{
   // Returns size string will occupy on I/O buffer.

   if (Length() > 254)
      return Length()+sizeof(UChar_t)+sizeof(Int_t);
   else
      return Length()+sizeof(UChar_t);
}

//_______________________________________________________________________
void TString::Streamer(TBuffer &b)
{
   // Stream a string object

   Int_t   nbig;
   UChar_t nwh;
   if (b.IsReading()) {
      b >> nwh;
      if (nwh == 255)
         b >> nbig;
      else
         nbig = nwh;
      Pref()->UnLink();
      fData = TStringRef::GetRep(nbig,nbig)->Data();
      for (int i = 0; i < nbig; i++) b >> fData[i];
   } else {
      nbig = Length();
      if (nbig > 254) {
         nwh = 255;
         b << nwh;
         b << nbig;
      } else {
         nwh = UChar_t(nbig);
         b << nwh;
      }
      for (int i = 0; i < nbig; i++) b << fData[i];
   }
}

//_______________________________________________________________________
TBuffer &operator>>(TBuffer &buf, TString &s)
{
   // Read string from TBuffer.

   s.Streamer(buf);
   return buf;
}

//_______________________________________________________________________
TBuffer &operator<<(TBuffer &buf, const TString &s)
{
   // Write string to TBuffer.

   ((TString&)s).Streamer(buf);
   return buf;
}

// ------------------- Related global functions --------------------

//______________________________________________________________________________
Bool_t operator==(const TString& s1, const char* s2)
{
   // Compare TString with a char *.

   const char* data = s1.Data();
   Ssiz_t len = s1.Length();
   Ssiz_t i;
   for (i = 0; s2[i]; ++i)
      if (data[i] != s2[i] || i == len) return kFALSE;
   return (i == len);
}

#if defined(R__MWERKS) || defined(R__ALPHA)
//______________________________________________________________________________
Bool_t operator==(const TString& s1, const TString& s2)
{
   // Compare two TStrings.

   return ((s1.Length() == s2.Length()) && !memcmp(s1.Data(), s2.Data(), s1.Length()));
}
#endif

//______________________________________________________________________________
TString ToLower(const TString& str)
{
   // Return a lower-case version of str.

   register Ssiz_t n = str.Length();
   TString temp((char)0, n);
   register const char* uc = str.Data();
   register       char* lc = (char*)temp.Data();
   // Guard against tolower() being a macro
   while (n--) { *lc++ = tolower((unsigned char)*uc); uc++; }
   return temp;
}

//______________________________________________________________________________
TString ToUpper(const TString& str)
{
   // Return an upper-case version of str.

   register Ssiz_t n = str.Length();
   TString temp((char)0, n);
   register const char* uc = str.Data();
   register       char* lc = (char*)temp.Data();
   // Guard against toupper() being a macro
   while (n--) { *lc++ = toupper((unsigned char)*uc); uc++; }
   return temp;
}

//______________________________________________________________________________
TString operator+(const TString& s, const char* cs)
{
   // Use the special concatenation constructor.

   return TString(s.Data(), s.Length(), cs, strlen(cs));
}

//______________________________________________________________________________
TString operator+(const char* cs, const TString& s)
{
   // Use the special concatenation constructor.

   return TString(cs, strlen(cs), s.Data(), s.Length());
}

//______________________________________________________________________________
TString operator+(const TString& s1, const TString& s2)
{
   // Use the special concatenation constructor.

   return TString(s1.Data(), s1.Length(), s2.Data(), s2.Length());
}

//______________________________________________________________________________
TString operator+(const TString& s, char c)
{
   // Add char to string.

   return TString(s.Data(), s.Length(), &c, 1);
}

//______________________________________________________________________________
TString operator+(const TString& s, Long_t i)
{
   // Add integer to string.

   const char *si = Form("%ld", i);
   return TString(s.Data(), s.Length(), si, strlen(si));
}

//______________________________________________________________________________
TString operator+(const TString& s, ULong_t i)
{
   // Add integer to string.

   const char *si = Form("%lu", i);
   return TString(s.Data(), s.Length(), si, strlen(si));
}

//______________________________________________________________________________
TString operator+(char c, const TString& s)
{
   // Add string to integer.

   return TString(&c, 1, s.Data(), s.Length());
}

//______________________________________________________________________________
TString operator+(Long_t i, const TString& s)
{
   // Add string to integer.

   const char *si = Form("%ld", i);
   return TString(si, strlen(si), s.Data(), s.Length());
}

//______________________________________________________________________________
TString operator+(ULong_t i, const TString& s)
{
   // Add string to integer.

   const char *si = Form("%lu", i);
   return TString(si, strlen(si), s.Data(), s.Length());
}

// -------------------- Static Member Functions ----------------------

// Static member variable initialization:
Ssiz_t          TString::fgInitialCapac     = 15;
Ssiz_t          TString::fgResizeInc        = 16;
Ssiz_t          TString::fgFreeboard        = 15;

//______________________________________________________________________________
Ssiz_t TString::InitialCapacity(Ssiz_t ic)
{
   // Set default initial capacity for all TStrings. Default is 15.

   Ssiz_t ret = fgInitialCapac;
   fgInitialCapac = ic;
   return ret;
}

//______________________________________________________________________________
Ssiz_t TString::ResizeIncrement(Ssiz_t ri)
{
   // Set default resize increment for all TStrings. Default is 16.

   Ssiz_t ret = fgResizeInc;
   fgResizeInc = ri;
   return ret;
}

//______________________________________________________________________________
Ssiz_t TString::MaxWaste(Ssiz_t mw)
{
   // Set maximum space that may be wasted in a string before doing a resize.
   // Default is 15.

   Ssiz_t ret = fgFreeboard;
   fgFreeboard = mw;
   return ret;
}


//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TSubString                                                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

//
// A zero lengthed substring is legal. It can start
// at any character. It is considered to be "pointing"
// to just before the character.
//
// A "null" substring is a zero lengthed substring that
// starts with the nonsense index kNPOS. It can
// be detected with the member function IsNull().
//

//______________________________________________________________________________
TSubString::TSubString(const TString& str, Ssiz_t start, Ssiz_t nextent)
   : fStr((TString*)&str), fBegin(start), fExtent(nextent)
{
   // Private constructor.
}

//______________________________________________________________________________
TSubString TString::operator()(Ssiz_t start, Ssiz_t len)
{
   // Return sub-string of string starting at start with length len.

   if (start < Length()) {
      if (start+len > Length())
         len = Length() - start;
   } else {
      start = kNPOS;
      len   = 0;
   }
   return TSubString(*this, start, len);
}

//______________________________________________________________________________
TSubString TString::SubString(const char* pattern, Ssiz_t startIndex,
                              ECaseCompare cmp)
{
   // Returns a substring matching "pattern", or the null substring
   // if there is no such match.  It would be nice if this could be yet another
   // overloaded version of operator(), but this would result in a type
   // conversion ambiguity with operator(Ssiz_t, Ssiz_t).

   Ssiz_t len = strlen(pattern);
   Ssiz_t i = Index(pattern, len, startIndex, cmp);
   return TSubString(*this, i, i == kNPOS ? 0 : len);
}

//______________________________________________________________________________
char& TSubString::operator[](Ssiz_t i)
{
   // Return character at pos i from sub-string. Check validity of i.

   AssertElement(i);
   fStr->Cow();
   return fStr->fData[fBegin+i];
}

//______________________________________________________________________________
char& TSubString::operator()(Ssiz_t i)
{
   // Return character at pos i from sub-string. No check on i.

   fStr->Cow();
   return fStr->fData[fBegin+i];
}

//______________________________________________________________________________
TSubString TString::operator()(Ssiz_t start, Ssiz_t len) const
{
   // Return sub-string of string starting at start with length len.

   if (start < Length()) {
      if (start+len > Length())
         len = Length() - start;
   } else {
      start = kNPOS;
      len   = 0;
   }
   return TSubString(*this, start, len);
}

//______________________________________________________________________________
TSubString TString::SubString(const char* pattern, Ssiz_t startIndex,
                              ECaseCompare cmp) const
{
   // Return sub-string matching pattern, starting at index. Cmp selects
   // the type of case conversion.

   Ssiz_t len = strlen(pattern);
   Ssiz_t i = Index(pattern, len, startIndex, cmp);
   return TSubString(*this, i, i == kNPOS ? 0 : len);
}

//______________________________________________________________________________
TSubString& TSubString::operator=(const TString& str)
{
   // Assign string to sub-string.

   if (!IsNull())
      fStr->Replace(fBegin, fExtent, str.Data(), str.Length());

   return *this;
}

//______________________________________________________________________________
TSubString& TSubString::operator=(const char* cs)
{
   // Assign char* to sub-string.

   if (!IsNull())
      fStr->Replace(fBegin, fExtent, cs, strlen(cs));

   return *this;
}

//______________________________________________________________________________
Bool_t operator==(const TSubString& ss, const char* cs)
{
   // Compare sub-string to char *.

   if (ss.IsNull()) return *cs =='\0'; // Two null strings compare equal

   const char* data = ss.fStr->Data() + ss.fBegin;
   Ssiz_t i;
   for (i = 0; cs[i]; ++i)
      if (cs[i] != data[i] || i == ss.fExtent) return kFALSE;
   return (i == ss.fExtent);
}

//______________________________________________________________________________
Bool_t operator==(const TSubString& ss, const TString& s)
{
   // Compare sub-string to string.

   if (ss.IsNull()) return s.IsNull(); // Two null strings compare equal.
   if (ss.fExtent != s.Length()) return kFALSE;
   return !memcmp(ss.fStr->Data() + ss.fBegin, s.Data(), ss.fExtent);
}

//______________________________________________________________________________
Bool_t operator==(const TSubString& s1, const TSubString& s2)
{
   // Compare two sub-strings.

   if (s1.IsNull()) return s2.IsNull();
   if (s1.fExtent != s2.fExtent) return kFALSE;
   return !memcmp(s1.fStr->Data()+s1.fBegin, s2.fStr->Data()+s2.fBegin,
                  s1.fExtent);
}

//______________________________________________________________________________
void TSubString::ToLower()
{
   // Convert sub-string to lower-case.

   if (!IsNull()) {                             // Ignore null substrings
      fStr->Cow();
      register char* p = (char*)(fStr->Data() + fBegin); // Cast away constness
      Ssiz_t n = fExtent;
      while (n--) { *p = tolower((unsigned char)*p); p++;}
   }
}

//______________________________________________________________________________
void TSubString::ToUpper()
{
   // Convert sub-string to upper-case.
   if (!IsNull()) {                             // Ignore null substrings
      fStr->Cow();
      register char* p = (char*)(fStr->Data() + fBegin); // Cast away constness
      Ssiz_t n = fExtent;
      while (n--) { *p = toupper((unsigned char)*p); p++;}
   }
}

//______________________________________________________________________________
void TSubString::SubStringError(Ssiz_t sr, Ssiz_t start, Ssiz_t n) const
{
   Error("TSubString::SubStringError",
         "out of bounds: start = %d, n = %d, sr = %d", start, n, sr);
}

//______________________________________________________________________________
void TSubString::AssertElement(Ssiz_t i) const
{
   // Check to make sure a sub-string index is in range.

   if (i == kNPOS || i >= Length())
      Error("TSubString::AssertElement",
            "out of bounds: i = %d, Length = %d", i, Length());
}

//______________________________________________________________________________
Bool_t TString::IsAscii() const
{
   // Return true if all characters in string are ascii.

   const char* cp = Data();
   for (Ssiz_t i = 0; i < Length(); ++i)
      if (cp[i] & ~0x7F)
         return kFALSE;
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TString::EndsWith(const char* s, ECaseCompare cmp) const
{
   Ssiz_t l = strlen(s);
   Ssiz_t i = Index(s, l, (Ssiz_t)0, cmp);
   if (i == kNPOS) return kFALSE;
   return i == Length() - l;
}


//---- Global String Handling Functions ----------------------------------------

static const int cb_size  = 4096;
static const int fld_size = 2048;

// a circular formating buffer
static char formbuf[cb_size];       // some slob for form overflow
static char *bfree  = formbuf;
static char *endbuf = &formbuf[cb_size-1];

//______________________________________________________________________________
static char *Format(const char *format, va_list ap)
{
   // Format a string in a circular formatting buffer (using a printf style
   // format descriptor).

   char *buf = bfree;

   if (buf+fld_size > endbuf)
      buf = formbuf;

   int n = vsnprintf(buf, fld_size, format, ap);
   // old vsnprintf's return -1 if string is truncated new ones return
   // total number of characters that would have been written
   if (n == -1 || n >= fld_size) {
      Warning("Format", "string truncated: %.30s...", buf);
      n = fld_size - 1;
   }

   bfree = buf+n+1;
   return buf;
}

//______________________________________________________________________________
char *Form(const char* va_(fmt), ...)
{
   // Formats a string in a circular formatting buffer. Removes the need to
   // create and delete short lived strings. Espcially useful to embed
   // in arguments.

   va_list ap;
   va_start(ap,va_(fmt));
   char *b = Format(va_(fmt), ap);
   va_end(ap);
   return b;
}

//______________________________________________________________________________
void Printf(const char* va_(fmt), ...)
{
   // Formats a string in a circular formatting buffer and prints the string.
   // Appends a newline.

   va_list ap;
   va_start(ap,va_(fmt));
   char *b = Format(va_(fmt), ap);
   va_end(ap);
   printf("%s\n", b);
   fflush(stdout);
}

//______________________________________________________________________________
char *Strip(const char *s, char c)
{
   // Strip leading and trailing c (blanks by default) from a string.
   // The returned string has to be deleted by the user.

   int l = strlen(s);
   char *buf = new char[l+1];

   if (l == 0) {
      *buf = '\0';
      return buf;
   }

   // get rid of leading c's
   const char *t1 = s;
   while (*t1 == c)
      t1++;

   // get rid of trailing c's
   const char *t2 = s + l - 1;
   while (*t2 == c && t2 > s)
      t2--;

   if (t1 > t2) {
      *buf = '\0';
      return buf;
   }
   strncpy(buf, t1, (Ssiz_t) (t2-t1+1));
   *(buf+(t2-t1+1)) = '\0';

   return buf;
}

//______________________________________________________________________________
char *StrDup(const char *str)
{
   // Duplicate the string str. The returned string has to be deleted by
   // the user.

   if (!str) return 0;

   char *s = new char[strlen(str)+1];
   if (s) strcpy(s, str);

   return s;
}

//______________________________________________________________________________
char *Compress(const char *str)
{
   // Remove all blanks from the string str. The returned string has to be
   // deleted by the user.

   if (!str) return 0;

   const char *p = str;
   char *s, *s1 = new char[strlen(str)+1];
   s = s1;

   while (*p) {
      if (*p != ' ')
         *s++ = *p;
      p++;
   }
   *s = '\0';

   return s1;
}

//______________________________________________________________________________
int EscChar(const char *src, char *dst, int dstlen, char *specchars, char escchar)
{
   // Escape specchars in src with escchar and copy to dst.

   const char *p;
   char *q, *end = dst+dstlen-1;

   for (p = src, q = dst; *p && q < end; ) {
      if (strchr(specchars, *p)) {
         *q++ = escchar;
         if (q < end)
            *q++ = *p++;
      } else
         *q++ = *p++;
   }
   *q = '\0';

   if (*p != 0)
      return -1;
   return q-dst;
}

//______________________________________________________________________________
int UnEscChar(const char *src, char *dst, int dstlen, char *specchars, char)
{
   // Un-escape specchars in src from escchar and copy to dst.

   const char *p;
   char *q, *end = dst+dstlen-1;

   for (p = src, q = dst; *p && q < end; ) {
      if (strchr(specchars, *p))
         p++;
      else
         *q++ = *p++;
   }
   *q = '\0';

   if (*p != 0)
      return -1;
   return q-dst;
}

#ifdef NEED_STRCASECMP
//______________________________________________________________________________
int strcasecmp(const char *str1, const char *str2)
{
   // Case insensitive string compare.

   return strncasecmp(str1, str2, strlen(str2) + 1);
}

//______________________________________________________________________________
int strncasecmp(const char *str1, const char *str2, Ssiz_t n)
{
   // Case insensitive string compare of n characters.

   while (n > 0) {
      int c1 = *str1;
      int c2 = *str2;

      if (isupper(c1))
         c1 = tolower(c1);

      if (isupper(c2))
         c2 = tolower(c2);

      if (c1 != c2)
         return c1 - c2;

      str1++;
      str2++;
      n--;
   }
   return 0;
}
#endif
