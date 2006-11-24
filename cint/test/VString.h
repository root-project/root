/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/***********************************************************************
* VString.h , C++
*
************************************************************************
* Description:
*
***********************************************************************/

#ifndef VSTRING_H
#define VSTRING_H

#include "VType.h"

/***********************************************************************
***********************************************************************/
class VString {
 public:

  VString() { len = 0; str = (Char_t*)NULL; }
  VString(const Char_t* strIn);
  VString(const VString& kstrIn);
  VString& operator=(const VString& obj);
  VString& operator=(const Char_t* s);
  ~VString() { if(str) delete[] str; }
  Int_t operator==(const VString& x) { 
    if((0==len&&0==x.len) || (len==x.len && 0==strcmp(str,x.str))) 
         return MATCH; 
    else return UNMATCH;
  }

  Int_t Write(FILE* fp);
  Int_t Read(FILE* fp);

  void append(const VString& s);
  void append(const Char_t* s);

  Int_t Length() { return len; }
  Char_t* String() { if(str) return str; else return "";}

  friend int strcmp(const VString& a,const Char_t* b);
  friend int strcmp(const Char_t* b,const VString& a);
#if 0
  Char_t operator[](Int_t index);
  friend int strcmp(const VString& a,const VString& b);
  friend char* strcpy(VString& a,const VString& b);
#endif

 private: 
  Int_t len;   
  Char_t* str; 
};



#endif // VString_H

