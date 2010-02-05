#ifndef __OUC_STRING_H__
#define __OUC_STRING_H__
/******************************************************************************/
/*                                                                            */
/*                     X r d O u c S t r i n g . h h                          */
/*                                                                            */
/* (c) 2005 F. Furano (INFN Padova), G. Ganis (CERN)                          */
/*     All Rights Reserved. See XrdInfo.cc for complete License Terms         */
/******************************************************************************/

//           $Id$

/******************************************************************************/
/*                                                                            */
/*  Light string manipulation class                                           */
/*                                                                            */
/*  This class has three private members: a buffer (char *) and two integers, */
/*  indicating the effective length of the null-terminated string (len), and  */
/*  the buffer capacity (siz), i.e. the allocated size. The capacity is set   */
/*  at construction either at the value needed by the initializing string or  */
/*  to the value requested by the user; by default the capacity is never      */
/*  decreased during manipulations (it is increased if required by the        */
/*  operation). The capacity can be changed at any time by calling resize().  */
/*  The user can choose a granularity other than 1 to increase the capacity   */
/*  by calling XrdOucString::setblksize(nbs) with nbs > 1: this will make     */
/*  new allocations to happen in blocks multiple of nbs bytes.                */
/*                                                                            */
/*  1. Constructors                                                           */
/*                                                                            */
/*     XrdOucString(int lmx = 0)                                              */
/*      - create an empty string; set capacity to lmx+1 if lmx > 0            */
/*     XrdOucString(const char *s, int lmx = 0)                               */
/*      - create a string containing s; capacity is set to strlen(s)+1 or     */
/*        to lmx+1 if lmx > 0; in the latter case truncation may occur.       */
/*     XrdOucString(const char c, int lmx = 0)                                */
/*      - create a string char c; capacity is set to 2 or to lmx+1 if lmx > 0.*/
/*     XrdOucString(const XrdOucString &s)                                    */
/*      - create string copying from XrdOucString s .                         */
/*     XrdOucString(const XrdOucString &s, int j, int k = -1, int lmx = 0)    */
/*      - create string copying a portion of XrdOucString s; portion is       */
/*        defined by j to k inclusive; if k == -1 the portion copied will be  */
/*        from j to end-of-string; if j or k are inconsistent they are taken  */
/*        as 0 or len-1, respectively; capacity is set to k-j bytes or to     */
/*        to lmx+1 if lmx > 0; in the latter case truncation of the portion   */
/*        may occur.                                                          */
/*                                                                            */
/*  2. Access to information                                                  */
/*                                                                            */
/*     const char   *c_str() const                                            */
/*      - return pointer to stored string                                     */
/*     int           length() const                                           */
/*      - return length stored string                                         */
/*     int           capacity() const                                         */
/*      - return capacity of the allocated buffer                             */
/*                                                                            */
/*     char         &operator[](int j)                                        */
/*      - array-like operator returning char at position j; abort is invoked  */
/*        if j is not in the correct range                                    */
/*                                                                            */
/*     int           find(const char c, int start = 0, bool forward = 1);     */
/*      - find first occurence of char c starting from position start in      */
/*        forward (forward == 1, default) or backward (forward == 0)          */
/*        direction; returns STR_NPOS if nothing is found                     */
/*     int           find(const char *s, int start = 0)                       */
/*      - find first occurence of string s starting from position start in    */
/*        forward direction; returns STR_NPOS if nothing is found             */
/*     int           find(XrdOucString s, int start = 0)                      */
/*      - find first occurence of XrdOucString s starting from position start */
/*        in forward direction; returns STR_NPOS if nothing is found          */
/*                                                                            */
/*     int           rfind(const char c, int start = STR_NPOS)                */
/*      - find first occurence of char c starting from position start in      */
/*        backward direction; returns STR_NPOS if nothing is found.           */
/*     int           rfind(const char *s, int start = STR_NPOS)               */
/*      - find first occurence of string s starting from position start in    */
/*        backward direction; returns STR_NPOS if nothing is found;           */
/*        if start == STR_NPOS search starts from position len-strlen(s)      */
/*     int           rfind(XrdOucString s, int start = STR_NPOS)              */
/*      - find first occurence of XrdOucString s starting from position start */
/*        in backward direction; returns STR_NPOS if nothing is found;        */
/*        if start == STR_NPOS search starts from position len-s.lenght()     */
/*                                                                            */
/*     bool          beginswith(char c)                                       */
/*      - returns 1 if the stored string starts with char c                   */
/*     bool          beginswith(const char *s)                                */
/*      - returns 1 if the stored string starts with string s                 */
/*     bool          beginswith(XrdOucString s)                               */
/*      - returns 1 if the stored string starts with XrdOucString s           */
/*                                                                            */
/*     bool          endswith(char c)                                         */
/*      - returns 1 if the stored string ends with char c                     */
/*     bool          endswith(const char *s)                                  */
/*      - returns 1 if the stored string ends with string s                   */
/*     bool          endswith(XrdOucString s)                                 */
/*      - returns 1 if the stored string ends with XrdOucString s             */
/*                                                                            */
/*     int           matches(const char *s, char wch = '*')                   */
/*      - check if stored string is compatible with s allowing for wild char  */
/*        wch (default: '*'); return the number of matching characters.       */
/*                                                                            */
/*  3. Modifiers                                                              */
/*                                                                            */
/*     void          resize(int lmx = 0)                                      */
/*      - resize buffer capacity to lmx+1 bytes; if lmx <= 0, free the buffer.*/
/*                                                                            */
/*     void          append(const int i)                                      */
/*      - append to stored string the string representation of integer i,     */
/*        e.g. if string is initially "x*", after append(5) it will be "x*5". */
/*     void          append(const char c)                                     */
/*      - append char c to stored string, e.g. if string is initially "pop",  */
/*        after append('_') it will be "pop_".                                */
/*     void          append(const char *s)                                    */
/*      - append string s to stored string, e.g. if string is initially "pop",*/
/*        after append("star") it will be "popstar".                          */
/*     void          append(const XrdOucString s)                             */
/*      - append s.c_str() to stored string, e.g. if string is initially      */
/*        "anti", after append("star") it will be "antistar".                 */
/*                                                                            */
/*     void          assign(const char *s, int j, int k = -1)                 */
/*      - copy to allocated buffer a portion of string s; portion is defined  */
/*        by j to k inclusive; if k == -1 the portion copied will be from j   */
/*        to end-of-string; if j or k are inconsistent they are taken as 0 or */
/*        len-1, respectively; if necessary, capacity is increased to k-j     */
/*        bytes.                                                              */
/*     void          assign(const XrdOucString s, int j, int k = -1)          */
/*      - copy to allocated buffer a portion of s.c_str(); portion is defined */
/*        by j to k inclusive; if k == -1 the portion copied will be from j   */
/*        to end-of-string; if j or k are inconsistent they are taken as 0 or */
/*        len-1, respectively; if necessary, capacity is increased to k-j     */
/*        bytes.                                                              */
/*                                                                            */
/*     int           keep(int start = 0, int size = 0)                        */
/*      - drop chars outside the range of size bytes starting at start        */
/*                                                                            */
/*     void          insert(const int i, int start = -1)                      */
/*      - insert the string representation of integer i at position start of  */
/*        the stored string, e.g. if string is initially "*x", after          */
/*        insert(5,0) it will be "5*x"; default action is append.             */
/*     void          insert(const char c, int start = -1)                     */
/*      - insert the char c at position start of the stored string, e.g.      */
/*        if string is initially "pok", after insert('_',0) it will be "_poc";*/
/*        default action is append.                                           */
/*     void          insert(const char *s, int start = -1, int lmx = 0)       */
/*      - insert string s at position start of the stored string, e.g.        */
/*        if string is initially "forth", after insert("backand",0) it will be*/
/*        "backandforth"; default action is append.                           */
/*     void          insert(const XrdOucString s, int start = -1)             */
/*      - insert string s.c_str() at position start of the stored string.     */
/*                                                                            */
/*     int           replace(const char *s1, const char *s2,                  */
/*                           int from = 0, int to = -1);                      */
/*      - replace all occurrencies of string s1 with string s2 in the region  */
/*        from position 'from' to position 'to' inclusive; the method is      */
/*        optimized to minimize the memory movements; with s2 == 0 or ""      */
/*        removes all instances of s1 in the specified region.                */
/*     int           replace(const XrdOucString s1, const char *s2,           */
/*                           int from = 0, int to = -1);                      */
/*     int           replace(const char *s1, const XrdOucString s2,           */
/*                           int from = 0, int to = -1);                      */
/*     int           replace(const XrdOucString s1, const XrdOucString s2,    */
/*                           int from = 0, int to = -1);                      */
/*      - interfaces to replace(const char *, const char *, int, int)         */
/*                                                                            */
/*     int           erase(int start = 0, int size = 0)                       */
/*      - erase size bytes starting at start                                  */
/*     int           erase(const char *s, int from = 0, int to = -1)          */
/*      - erase occurences of string s within position 'from' and position    */
/*        'to' (inclusive), e.g if stored string is "aabbccefccddgg", then    */
/*        erase("cc",0,9) will result in string "aabbefccddgg".               */
/*     int           erase(XrdOucString s, int from = 0, int to = -1)         */
/*      - erase occurences of s.c_str() within position 'from' and position   */
/*        'to' (inclusive).                                                   */
/*     int           erasefromstart(int sz = 0)                               */
/*      - erase sz bytes from the start.                                      */
/*     int           erasefromend(int sz = 0)                                 */
/*      - erase sz bytes from the end.                                        */
/*                                                                            */
/*     void          lower(int pos, int size = 0)                             */
/*      - set to lower case size bytes from position start.                   */
/*     void          upper(int pos, int size = 0)                             */
/*      - set to upper case size bytes from position start.                   */
/*                                                                            */
/*     void          hardreset()                                              */
/*      - memset to 0 the len meaningful bytes of the buffer.                 */
/*                                                                            */
/*     int           tokenize(XrdOucString &tok, int from, char del)          */
/*      - search for tokens delimited by 'del' (def ':') in string s; search  */
/*        starts from 'from' and the token is returned in 'tok'.              */
/*                                                                            */
/*  4. Assignement operators                                                  */
/*     XrdOucString &operator=(const int i)                                   */
/*     XrdOucString &operator=(const char c)                                  */
/*     XrdOucString &operator=(const char *s)                                 */
/*     XrdOucString &operator=(const XrdOucString s)                          */
/*                                                                            */
/*  5. Addition operators                                                     */
/*     XrdOucString &operator+(const int i)                                   */
/*     XrdOucString &operator+(const char c)                                  */
/*     XrdOucString &operator+(const char *s)                                 */
/*     XrdOucString &operator+(const XrdOucString s)                          */
/*     XrdOucString &operator+=(const int i)                                  */
/*     XrdOucString &operator+=(const char c)                                 */
/*     XrdOucString &operator+=(const char *s)                                */
/*     XrdOucString &operator+=(const XrdOucString s)                         */
/*     XrdOucString const operator+(const char *s1, const XrdOucString s2)    */
/*     XrdOucString const operator+(const char c, const XrdOucString s)       */
/*     XrdOucString const operator+(const int i, const XrdOucString s)        */
/*                                                                            */
/*  6. Equality operators                                                     */
/*     int operator==(const int i)                                            */
/*     int operator==(const char c)                                           */
/*     int operator==(const char *s)                                          */
/*     int operator==(const XrdOucString s)                                   */
/*                                                                            */
/*  7. Inequality operators                                                   */
/*     int operator!=(const int i)                                            */
/*     int operator!=(const char c)                                           */
/*     int operator!=(const char *s)                                          */
/*     int operator!=(const XrdOucString s)                                   */
/*                                                                            */
/*  8. Static methods to change / monitor the blksize                         */
/*     static int getblksize();                                               */
/*     static void setblksize(const int bs);                                  */
/*                                                                            */
/******************************************************************************/
#include "XrdSys/XrdSysHeaders.hh"

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

using namespace std;

#define STR_NPOS -1

class XrdOucString {

private:
   char *str;
   int   len;
   int   siz;

   // Mininal block size to be used in (re-)allocations
   // Option switched off by default; use XrdOucString::setblksize()
   // and XrdOucString::getblksize() to change / monitor 
   static int blksize;

   // Private methods
   int         adjust(int ls, int &j, int &k, int nmx = 0);
   char       *bufalloc(int nsz);
   inline void init() { str = 0; len = 0; siz = 0; }

public:
   XrdOucString(int lmx = 0) { init(); if (lmx > 0) str = bufalloc(lmx+1); }
   XrdOucString(const char *s, int lmx = 0);
   XrdOucString(const char c, int lmx = 0);
   XrdOucString(const XrdOucString &s);
   XrdOucString(const XrdOucString &s, int j, int k = -1, int lmx = 0);
   virtual ~XrdOucString();

   // Info access
   const char   *c_str() const { return (const char *)str; }
   int           length() const { return len; }
   int           capacity() const { return siz; }
   char         &operator[](int j);
   int           find(const char c, int start = 0, bool forward = 1);
   int           find(const char *s, int start = 0);
   int           find(XrdOucString s, int start = 0);
   int           rfind(const char c, int start = STR_NPOS)
                                             { return find(c, start, 0); }
   int           rfind(const char *s, int start = STR_NPOS);
   int           rfind(XrdOucString s, int start = STR_NPOS);
   bool          beginswith(char c) { return (find(c) == 0); }
   bool          beginswith(const char *s) { return (find(s) == 0); }
   bool          beginswith(XrdOucString s) { return (find(s) == 0); }
   bool          endswith(char c);
   bool          endswith(const char *s);
   bool          endswith(XrdOucString s) { return (endswith(s.c_str())); }
   int           matches(const char *s, char wch = '*');

   // Tokenizer
   int           tokenize(XrdOucString &tok, int from, char del = ':');

   // Modifiers
   void          resize(int lmx = 0) { int ns = (lmx > 0) ? lmx + 1 : 0;
                                       str = bufalloc(ns); }
   void          append(const int i);
   void          append(const char c);
   void          append(const char *s);
   void          append(const XrdOucString s);
   void          assign(const char *s, int j, int k = -1);
   void          assign(const XrdOucString s, int j, int k = -1);
#if !defined(WINDOWS)
   int           form(const char *fmt, ...);
#endif
   int           keep(int start = 0, int size = 0);
   void          insert(const int i, int start = -1);
   void          insert(const char c, int start = -1);
   void          insert(const char *s, int start = -1, int lmx = 0);
   void          insert(const XrdOucString s, int start = -1);
   int           replace(const char *s1, const char *s2,
                                         int from = 0, int to = -1);
   int           replace(const XrdOucString s1, const XrdOucString s2,
                                                int from = 0, int to = -1);
   int           replace(const XrdOucString s1, const char *s2,
                                                int from = 0, int to = -1);
   int           replace(const char *s1, const XrdOucString s2,
                                                int from = 0, int to = -1);
   int           erase(int start = 0, int size = 0);
   int           erase(const char *s, int from = 0, int to = -1);
   int           erase(XrdOucString s, int from = 0, int to = -1);
   int           erasefromstart(int sz = 0) { return erase(0,sz); }
   int           erasefromend(int sz = 0) { return erase(len-sz,sz); }
   void          lower(int pos, int size = 0);
   void          upper(int pos, int size = 0);
   void          reset(const char c, int j = 0, int k = -1);
   void          hardreset();
   void          setbuffer(char *buf);

   // Assignement operators
   XrdOucString &operator=(const int i);
   XrdOucString &operator=(const char c);
   XrdOucString &operator=(const char *s);
   XrdOucString &operator=(const XrdOucString s);

   // Add operators
   friend XrdOucString operator+(const XrdOucString &s1, const int i);
   friend XrdOucString operator+(const XrdOucString &s1, const char c);
   friend XrdOucString operator+(const XrdOucString &s1, const char *s);
   friend XrdOucString operator+(const XrdOucString &s1, const XrdOucString &s);
   XrdOucString &operator+=(const int i);
   XrdOucString &operator+=(const char c);
   XrdOucString &operator+=(const char *s);
   XrdOucString &operator+=(const XrdOucString s);   

   // Equality operators
   int operator==(const int i);
   int operator==(const char c);
   int operator==(const char *s);
   int operator==(const XrdOucString s);

   // Inequality operators
   int operator!=(const int i) { return !(*this == i); }
   int operator!=(const char c) { return !(*this == c); }
   int operator!=(const char *s) { return !(*this == s); }
   int operator!=(const XrdOucString s) { return !(*this == s); }

   // Miscellanea
   bool isdigit(int from = 0, int to = -1);
   long atoi(int from = 0, int to = -1);

   // Static methods to change / monitor the default blksize
   static int getblksize();
   static void setblksize(const int bs);

#if !defined(WINDOWS)
   // format a string
   static int form(XrdOucString &str, const char *fmt, ...);
#endif
};

// Operator << is useful to print a string into a stream
ostream &operator<< (ostream &, const XrdOucString s);

XrdOucString const operator+(const char *s1, const XrdOucString s2);
XrdOucString const operator+(const char c, const XrdOucString s);
XrdOucString const operator+(const int i, const XrdOucString s);

#endif

