// @(#)root/base:$Name:  $:$Id: TString.h,v 1.6 2000/12/19 15:27:44 rdm Exp $
// Author: Fons Rademakers   04/08/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TString
#define ROOT_TString


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TString                                                              //
//                                                                      //
// Basic string class.                                                  //
//                                                                      //
// Cannot be stored in a TCollection... use TObjString instead.         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef __CINT__
#include <string.h>
#endif

#ifndef ROOT_TMath
#include "TMath.h"
#endif

#ifndef ROOT_TRefCnt
#include "TRefCnt.h"
#endif

#ifdef R__MWERKS
#   ifdef Length
#      undef Length
#   endif
#endif

#if defined(R__ANSISTREAM)
#include <iosfwd>
using namespace std;
#elif R__MWERKS
template <class charT> class ios_traits;
template <class charT, class traits> class basic_istream;
template <class charT, class traits> class basic_ostream;
typedef basic_istream<char, ios_traits<char> > istream;
typedef basic_ostream<char, ios_traits<char> > ostream;
#else
class istream;
class ostream;
#endif

class TRegexp;


//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TStringRef                                                          //
//                                                                      //
//  This is the dynamically allocated part of a TString.                //
//  It maintains a reference count. It contains no public member        //
//  functions.                                                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
class TStringRef : public TRefCnt {

friend class TString;
friend class TStringLong;
friend class TSubString;

private:
   Ssiz_t       fCapacity;      // Max string length (excluding null)
   Ssiz_t       fNchars;        // String length (excluding null)

   void         UnLink(); // disconnect from a TStringRef, maybe delete it

   Ssiz_t       Length() const   { return fNchars; }
   Ssiz_t       Capacity() const { return fCapacity; }
   char        *Data() const     { return (char*)(this+1); }

   char&        operator[](Ssiz_t i)       { return ((char*)(this+1))[i]; }
   char         operator[](Ssiz_t i) const { return ((char*)(this+1))[i]; }

   Ssiz_t       First(char c) const;
   Ssiz_t       First(const char *s) const;
   unsigned     Hash() const;
   unsigned     HashFoldCase() const;
   Ssiz_t       Last(char) const;

   static TStringRef *GetRep(Ssiz_t capac, Ssiz_t nchar);
};


//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TSubString                                                          //
//                                                                      //
//  The TSubString class allows selected elements to be addressed.      //
//  There are no public constructors.                                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
class TSubString {

friend class TStringLong;
friend class TString;

private:
   TString      *fStr;           // Referenced string
   Ssiz_t        fBegin;         // Index of starting character
   Ssiz_t        fExtent;        // Length of TSubString

   // NB: the only constructor is private
   TSubString(const TString& s, Ssiz_t start, Ssiz_t len);

   friend Bool_t operator==(const TSubString& s1, const TSubString& s2);
   friend Bool_t operator==(const TSubString& s1, const TString& s2);
   friend Bool_t operator==(const TSubString& s1, const char *s2);

protected:
   void          SubStringError(Ssiz_t, Ssiz_t, Ssiz_t) const;
   void          AssertElement(Ssiz_t i) const;  // Verifies i is valid index

public:
   TSubString(const TSubString& s)
     : fStr(s.fStr), fBegin(s.fBegin), fExtent(s.fExtent) { }

   TSubString&   operator=(const char *s);       // Assignment to char*
   TSubString&   operator=(const TString& s);    // Assignment to TString
   char&         operator()(Ssiz_t i);           // Index with optional bounds checking
   char&         operator[](Ssiz_t i);           // Index with bounds checking
   char          operator()(Ssiz_t i) const;     // Index with optional bounds checking
   char          operator[](Ssiz_t i) const;     // Index with bounds checking

   const char   *Data() const;
   Ssiz_t        Length() const          { return fExtent; }
   Ssiz_t        Start() const           { return fBegin; }
   void          ToLower();              // Convert self to lower-case
   void          ToUpper();              // Convert self to upper-case

   // For detecting null substrings
   Bool_t        IsNull() const          { return fBegin == kNPOS; }
   int           operator!() const       { return fBegin == kNPOS; }
};


//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TString                                                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
class TString {

friend class TSubString;
friend class TStringRef;

friend TString operator+(const TString& s1, const TString& s2);
friend TString operator+(const TString& s,  const char *cs);
friend TString operator+(const char *cs, const TString& s);
friend TString operator+(const TString& s, char c);
friend TString operator+(const TString& s, Long_t i);
friend TString operator+(const TString& s, ULong_t i);
friend TString operator+(char c, const TString& s);
friend TString operator+(Long_t i, const TString& s);
friend TString operator+(ULong_t i, const TString& s);
friend Bool_t  operator==(const TString& s1, const TString& s2);
friend Bool_t  operator==(const TString& s1, const char *s2);

private:
   static Ssiz_t  fgInitialCapac;   // Initial allocation Capacity
   static Ssiz_t  fgResizeInc;      // Resizing increment
   static Ssiz_t  fgFreeboard;      // Max empty space before reclaim

   void           Clone();          // Make self a distinct copy
   void           Clone(Ssiz_t nc); // Make self a distinct copy w. capacity nc

protected:
   char          *fData;          // ref. counted data (TStringRef is in front)

   // Special concatenation constructor
   TString(const char *a1, Ssiz_t n1, const char *a2, Ssiz_t n2);
   TStringRef    *Pref() const { return (((TStringRef*) fData) - 1); }
   void           AssertElement(Ssiz_t nc) const; // Index in range
   void           Clobber(Ssiz_t nc);             // Remove old contents
   void           Cow();                          // Do copy on write as needed
   void           Cow(Ssiz_t nc);                 // Do copy on write as needed
   static Ssiz_t  AdjustCapacity(Ssiz_t nc);
   void           InitChar(char c);               // Initialize from char

public:
   enum EStripType   { kLeading = 0x1, kTrailing = 0x2, kBoth = 0x3 };
   enum ECaseCompare { kExact, kIgnoreCase };

   TString();                       // Null string
   TString(Ssiz_t ic);              // Suggested capacity
   TString(const TString& s)        // Copy constructor
      { fData = s.fData; Pref()->AddReference(); }

   TString(const char *s);              // Copy to embedded null
   TString(const char *s, Ssiz_t n);    // Copy past any embedded nulls
   TString(char c) { InitChar(c); }

   TString(char c, Ssiz_t s);

   TString(const TSubString& sub);

   virtual ~TString();

   // ROOT I/O interface
   virtual void     FillBuffer(char *&buffer);
   virtual void     ReadBuffer(char *&buffer);
   virtual Int_t    Sizeof() const;

   static TString  *ReadString(TBuffer &b, const TClass *clReq);
   static void      WriteString(TBuffer &b, const TString *a);

   friend TBuffer &operator<<(TBuffer &b, const TString *obj);


   // Type conversion
   operator const char*() const { return fData; }

   // Assignment
   TString&    operator=(char s);                // Replace string
   TString&    operator=(const char *s);
   TString&    operator=(const TString& s);
   TString&    operator=(const TSubString& s);
   TString&    operator+=(const char *s);        // Append string
   TString&    operator+=(const TString& s);
   TString&    operator+=(char c);
   TString&    operator+=(Short_t i);
   TString&    operator+=(UShort_t i);
   TString&    operator+=(Int_t i);
   TString&    operator+=(UInt_t i);
   TString&    operator+=(Long_t i);
   TString&    operator+=(ULong_t i);

   // Indexing operators
   char&         operator[](Ssiz_t i);         // Indexing with bounds checking
   char&         operator()(Ssiz_t i);         // Indexing with optional bounds checking
   TSubString    operator()(Ssiz_t start, Ssiz_t len);   // Sub-string operator
   TSubString    operator()(const TRegexp& re);          // Match the RE
   TSubString    operator()(const TRegexp& re, Ssiz_t start);
   TSubString    SubString(const char *pat, Ssiz_t start = 0,
                           ECaseCompare cmp = kExact);
   char          operator[](Ssiz_t i) const;
   char          operator()(Ssiz_t i) const;
   TSubString    operator()(Ssiz_t start, Ssiz_t len) const;
   TSubString    operator()(const TRegexp& re) const;   // Match the RE
   TSubString    operator()(const TRegexp& re, Ssiz_t start) const;
   TSubString    SubString(const char *pat, Ssiz_t start = 0,
                           ECaseCompare cmp = kExact) const;

   // Non-static member functions
   TString&     Append(const char *cs);
   TString&     Append(const char *cs, Ssiz_t n);
   TString&     Append(const TString& s);
   TString&     Append(const TString& s, Ssiz_t n);
   TString&     Append(char c, Ssiz_t rep = 1);   // Append c rep times
   Bool_t       BeginsWith(const char *s,      ECaseCompare cmp = kExact) const;
   Bool_t       BeginsWith(const TString& pat, ECaseCompare cmp = kExact) const;
   Ssiz_t       Capacity() const         { return Pref()->Capacity(); }
   Ssiz_t       Capacity(Ssiz_t n);
   TString&     Chop();
   int          CompareTo(const char *cs,    ECaseCompare cmp = kExact) const;
   int          CompareTo(const TString& st, ECaseCompare cmp = kExact) const;
   Bool_t       Contains(const char *pat,    ECaseCompare cmp = kExact) const;
   Bool_t       Contains(const TString& pat, ECaseCompare cmp = kExact) const;
   TString      Copy() const;
   const char  *Data() const                 { return fData; }
   Bool_t       EndsWith(const char *pat,    ECaseCompare cmp = kExact) const;
   Ssiz_t       First(char c) const          { return Pref()->First(c); }
   Ssiz_t       First(const char *cs) const  { return Pref()->First(cs); }
   unsigned     Hash(ECaseCompare cmp = kExact) const;
   Ssiz_t       Index(const char *pat, Ssiz_t i = 0,
                      ECaseCompare cmp = kExact) const;
   Ssiz_t       Index(const TString& s, Ssiz_t i = 0,
                      ECaseCompare cmp = kExact) const;
   Ssiz_t       Index(const char *pat, Ssiz_t patlen, Ssiz_t i,
                      ECaseCompare cmp) const;
   Ssiz_t       Index(const TString& s, Ssiz_t patlen, Ssiz_t i,
                      ECaseCompare cmp) const;
   Ssiz_t       Index(const TRegexp& pat, Ssiz_t i = 0) const;
   Ssiz_t       Index(const TRegexp& pat, Ssiz_t *ext, Ssiz_t i = 0) const;
   TString&     Insert(Ssiz_t pos, const char *s);
   TString&     Insert(Ssiz_t pos, const char *s, Ssiz_t extent);
   TString&     Insert(Ssiz_t pos, const TString& s);
   TString&     Insert(Ssiz_t pos, const TString& s, Ssiz_t extent);
   Bool_t       IsAscii() const;
   Bool_t       IsNull() const              { return Pref()->fNchars == 0; }
   Ssiz_t       Last(char c) const          { return Pref()->Last(c); }
   Ssiz_t       Length() const              { return Pref()->fNchars; }
   TString&     Prepend(const char *cs);     // Prepend a character string
   TString&     Prepend(const char *cs, Ssiz_t n);
   TString&     Prepend(const TString& s);
   TString&     Prepend(const TString& s, Ssiz_t n);
   TString&     Prepend(char c, Ssiz_t rep = 1);  // Prepend c rep times
   istream&     ReadFile(istream& str);      // Read to EOF or null character
   istream&     ReadLine(istream& str,
                         Bool_t skipWhite = kTRUE);   // Read to EOF or newline
   istream&     ReadString(istream& str);             // Read to EOF or null character
   istream&     ReadToDelim(istream& str, char delim = '\n'); // Read to EOF or delimitor
   istream&     ReadToken(istream& str);                // Read separated by white space
   TString&     Remove(Ssiz_t pos);                     // Remove pos to end of string
   TString&     Remove(Ssiz_t pos, Ssiz_t n);           // Remove n chars starting at pos
   TString&     Replace(Ssiz_t pos, Ssiz_t n, const char *s);
   TString&     Replace(Ssiz_t pos, Ssiz_t n, const char *s, Ssiz_t ns);
   TString&     Replace(Ssiz_t pos, Ssiz_t n, const TString& s);
   TString&     Replace(Ssiz_t pos, Ssiz_t n1, const TString& s, Ssiz_t n2);
   TString&     ReplaceAll(const TString& s1, const TString& s2); // Find&Replace all s1 with s2 if any
   TString&     ReplaceAll(const TString& s1, const char *s2);    // Find&Replace all s1 with s2 if any
   TString&     ReplaceAll(const    char *s1, const TString& s2); // Find&Replace all s1 with s2 if any
   TString&     ReplaceAll(const char *s1, const char *s2);       // Find&Replace all s1 with s2 if any
   TString&     ReplaceAll(const char *s1, Ssiz_t ls1, const char *s2, Ssiz_t ls2);  // Find&Replace all s1 with s2 if any
   void         Resize(Ssiz_t n);                       // Truncate or add blanks as necessary
   TSubString   Strip(EStripType s = kTrailing, char c = ' ');
   TSubString   Strip(EStripType s = kTrailing, char c = ' ') const;
   void         ToLower();                              // Change self to lower-case
   void         ToUpper();                              // Change self to upper-case

   // Static member functions
   static Ssiz_t  InitialCapacity(Ssiz_t ic = 15);      // Initial allocation capacity
   static Ssiz_t  MaxWaste(Ssiz_t mw = 15);             // Max empty space before reclaim
   static Ssiz_t  ResizeIncrement(Ssiz_t ri = 16);      // Resizing increment
   static Ssiz_t  GetInitialCapacity();
   static Ssiz_t  GetResizeIncrement();
   static Ssiz_t  GetMaxWaste();

   ClassDef(TString,1)  //Basic string class
};

// Related global functions
istream&  operator>>(istream& str,       TString& s);
ostream&  operator<<(ostream& str, const TString& s);
TBuffer&  operator>>(TBuffer& buf,       TString& s);
TBuffer&  operator<<(TBuffer& buf, const TString& s);

TString ToLower(const TString&);    // Return lower-case version of argument
TString ToUpper(const TString&);    // Return upper-case version of argument
inline  unsigned Hash(const TString& s) { return s.Hash(); }
inline  unsigned Hash(const TString *s) { return s->Hash(); }

extern char *Form(const char *fmt, ...);     // format in circular buffer
extern void  Printf(const char *fmt, ...);   // format and print
extern char *Strip(const char *str, char c = ' '); // strip c off str, free with delete []
extern char *StrDup(const char *str);        // duplicate str, free with delete []
extern char *Compress(const char *str);      // remove blanks from string, free with delele []
extern int   EscChar(const char *src, char *dst, int dstlen, char *specchars,
                     char escchar);          // copy from src to dst escaping specchars by escchar
extern int   UnEscChar(const char *src, char *dst, int dstlen, char *specchars,
                     char escchar);          // copy from src to dst removing escchar from specchars

#ifdef NEED_STRCASECMP
extern int strcasecmp(const char *str1, const char *str2);
extern int strncasecmp(const char *str1, const char *str2, Ssiz_t n);
#endif


//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  Inlines                                                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

inline void TStringRef::UnLink()
{ if (RemoveReference() == 0) delete [] (char*)this; }

inline void TString::Cow()
{ if (Pref()->References() > 1) Clone(); }

inline void TString::Cow(Ssiz_t nc)
{ if (Pref()->References() > 1  || Capacity() < nc) Clone(nc); }

inline TString& TString::Append(const char *cs)
{ return Replace(Length(), 0, cs, strlen(cs)); }

inline TString& TString::Append(const char* cs, Ssiz_t n)
{ return Replace(Length(), 0, cs, n); }

inline TString& TString::Append(const TString& s)
{ return Replace(Length(), 0, s.Data(), s.Length()); }

inline TString& TString::Append(const TString& s, Ssiz_t n)
{ return Replace(Length(), 0, s.Data(), TMath::Min(n, s.Length())); }

inline TString& TString::operator+=(const char* cs)
{ return Append(cs, strlen(cs)); }

inline TString& TString::operator+=(const TString& s)
{ return Append(s.Data(), s.Length()); }

inline TString& TString::operator+=(char c)
{ return Append(c); }

inline TString& TString::operator+=(Long_t i)
{ return operator+=(Form("%ld", i)); }

inline TString& TString::operator+=(ULong_t i)
{ return operator+=(Form("%lu", i)); }

inline TString& TString::operator+=(Short_t i)
{ return operator+=((Long_t) i); }

inline TString& TString::operator+=(UShort_t i)
{ return operator+=((ULong_t) i); }

inline TString& TString::operator+=(Int_t i)
{ return operator+=((Long_t) i); }

inline TString& TString::operator+=(UInt_t i)
{ return operator+=((ULong_t) i); }

inline Bool_t TString::BeginsWith(const char* s, ECaseCompare cmp) const
{ return Index(s, strlen(s), (Ssiz_t)0, cmp) == 0; }

inline Bool_t TString::BeginsWith(const TString& pat, ECaseCompare cmp) const
{ return Index(pat.Data(), pat.Length(), (Ssiz_t)0, cmp) == 0; }

inline Bool_t TString::Contains(const TString& pat, ECaseCompare cmp) const
{ return Index(pat.Data(), pat.Length(), (Ssiz_t)0, cmp) != kNPOS; }

inline Bool_t TString::Contains(const char* s, ECaseCompare cmp) const
{ return Index(s, strlen(s), (Ssiz_t)0, cmp) != kNPOS; }

inline Ssiz_t TString::Index(const char* s, Ssiz_t i, ECaseCompare cmp) const
{ return Index(s, strlen(s), i, cmp); }

inline Ssiz_t TString::Index(const TString& s, Ssiz_t i, ECaseCompare cmp) const
{ return Index(s.Data(), s.Length(), i, cmp); }

inline Ssiz_t TString::Index(const TString& pat, Ssiz_t patlen, Ssiz_t i,
                             ECaseCompare cmp) const
{ return Index(pat.Data(), patlen, i, cmp); }

inline TString& TString::Insert(Ssiz_t pos, const char* cs)
{ return Replace(pos, 0, cs, strlen(cs)); }

inline TString& TString::Insert(Ssiz_t pos, const char* cs, Ssiz_t n)
{ return Replace(pos, 0, cs, n); }

inline TString& TString::Insert(Ssiz_t pos, const TString& s)
{ return Replace(pos, 0, s.Data(), s.Length()); }

inline TString& TString::Insert(Ssiz_t pos, const TString& s, Ssiz_t n)
{ return Replace(pos, 0, s.Data(), TMath::Min(n, s.Length())); }

inline TString& TString::Prepend(const char* cs)
{ return Replace(0, 0, cs, strlen(cs)); }

inline TString& TString::Prepend(const char* cs, Ssiz_t n)
{ return Replace(0, 0, cs, n); }

inline TString& TString::Prepend(const TString& s)
{ return Replace(0, 0, s.Data(), s.Length()); }

inline TString& TString::Prepend(const TString& s, Ssiz_t n)
{ return Replace(0, 0, s.Data(), TMath::Min(n, s.Length())); }

inline TString& TString::Remove(Ssiz_t pos)
{ return Replace(pos, TMath::Max(0, Length()-pos), 0, 0); }

inline TString& TString::Remove(Ssiz_t pos, Ssiz_t n)
{ return Replace(pos, n, 0, 0); }

inline TString& TString::Chop()
{ return Remove(TMath::Max(0,Length()-1)); }

inline TString& TString::Replace(Ssiz_t pos, Ssiz_t n, const char* cs)
{ return Replace(pos, n, cs, strlen(cs)); }

inline TString& TString::Replace(Ssiz_t pos, Ssiz_t n, const TString& s)
{ return Replace(pos, n, s.Data(), s.Length()); }

inline TString& TString::Replace(Ssiz_t pos, Ssiz_t n1, const TString& s,
                                 Ssiz_t n2)
{ return Replace(pos, n1, s.Data(), TMath::Min(s.Length(), n2)); }

inline TString&  TString::ReplaceAll(const TString& s1,const TString& s2)
{ return ReplaceAll( s1.Data(), s1.Length(), s2.Data(), s2.Length()) ; }

inline TString&  TString::ReplaceAll(const TString& s1,const char *s2)
{ return ReplaceAll( s1.Data(), s1.Length(), s2, s2 ? strlen(s2):0) ; }

inline TString&  TString::ReplaceAll(const char *s1,const TString& s2)
{ return ReplaceAll( s1, s1 ? strlen(s1): 0, s2.Data(), s2.Length()) ; }

inline TString&  TString::ReplaceAll(const char *s1,const char *s2)
{ return ReplaceAll( s1, s1?strlen(s1):0, s2, s2?strlen(s2):0) ; }

inline char& TString::operator()(Ssiz_t i)
{ Cow(); return fData[i]; }

inline char TString::operator[](Ssiz_t i) const
{ AssertElement(i); return fData[i]; }

inline char TString::operator()(Ssiz_t i) const
{ return fData[i]; }

inline const char* TSubString::Data() const
{ return fStr->Data() + fBegin; }

// Access to elements of sub-string with bounds checking
inline char TSubString::operator[](Ssiz_t i) const
{ AssertElement(i); return fStr->fData[fBegin+i]; }

inline char TSubString::operator()(Ssiz_t i) const
{ return fStr->fData[fBegin+i]; }

// String Logical operators
#if !defined(R__MWERKS) && !defined(R__ALPHA)
inline Bool_t     operator==(const TString& s1, const TString& s2)
{
   return ((s1.Length() == s2.Length()) &&
            !memcmp(s1.Data(), s2.Data(), s1.Length()));
}
#endif

inline Bool_t     operator!=(const TString& s1, const TString& s2)
{ return !(s1 == s2); }

inline Bool_t     operator< (const TString& s1, const TString& s2)
{ return s1.CompareTo(s2)< 0; }

inline Bool_t     operator> (const TString& s1, const TString& s2)
{ return s1.CompareTo(s2)> 0; }

inline Bool_t     operator<=(const TString& s1, const TString& s2)
{ return s1.CompareTo(s2)<=0; }

inline Bool_t     operator>=(const TString& s1, const TString& s2)
{ return s1.CompareTo(s2)>=0; }

//     Bool_t     operator==(const TString& s1, const char* s2);
inline Bool_t     operator!=(const TString& s1, const char* s2)
{ return !(s1 == s2); }

inline Bool_t     operator< (const TString& s1, const char* s2)
{ return s1.CompareTo(s2)< 0; }

inline Bool_t     operator> (const TString& s1, const char* s2)
{ return s1.CompareTo(s2)> 0; }

inline Bool_t     operator<=(const TString& s1, const char* s2)
{ return s1.CompareTo(s2)<=0; }

inline Bool_t     operator>=(const TString& s1, const char* s2)
{ return s1.CompareTo(s2)>=0; }

inline Bool_t     operator==(const char* s1, const TString& s2)
{ return (s2 == s1); }

inline Bool_t     operator!=(const char* s1, const TString& s2)
{ return !(s2 == s1); }

inline Bool_t     operator< (const char* s1, const TString& s2)
{ return s2.CompareTo(s1)> 0; }

inline Bool_t     operator> (const char* s1, const TString& s2)
{ return s2.CompareTo(s1)< 0; }

inline Bool_t     operator<=(const char* s1, const TString& s2)
{ return s2.CompareTo(s1)>=0; }

inline Bool_t     operator>=(const char* s1, const TString& s2)
{ return s2.CompareTo(s1)<=0; }

// SubString Logical operators
//     Bool_t     operator==(const TSubString& s1, const TSubString& s2);
//     Bool_t     operator==(const TSubString& s1, const char* s2);
//     Bool_t     operator==(const TSubString& s1, const TString& s2);
inline Bool_t     operator==(const TString& s1,    const TSubString& s2)
{ return (s2 == s1); }

inline Bool_t     operator==(const char* s1, const TSubString& s2)
{ return (s2 == s1); }

inline Bool_t     operator!=(const TSubString& s1, const char* s2)
{ return !(s1 == s2); }

inline Bool_t     operator!=(const TSubString& s1, const TString& s2)
{ return !(s1 == s2); }

inline Bool_t     operator!=(const TSubString& s1, const TSubString& s2)
{ return !(s1 == s2); }

inline Bool_t     operator!=(const TString& s1,   const TSubString& s2)
{ return !(s2 == s1); }

inline Bool_t     operator!=(const char* s1,       const TSubString& s2)
{ return !(s2 == s1); }

#endif

