/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/***********************************************************************
* VString.cxx , C++
*
************************************************************************
* Description:
***********************************************************************/

#include "VString.h"

///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
VString::VString(const Char_t* strIn )
{
  str = (Char_t*)NULL;
  len = 0;

  if(strIn && strIn[0]) {
    len = strlen(strIn);
    str = new Char_t[len+1];
    strcpy(str,strIn);
  }
}

///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
VString::VString(const VString& kstrIn )
{
  str = (Char_t*)NULL;
  len = 0;

  if(kstrIn.str) {
    len = kstrIn.len;
    str = new Char_t[len+1];
    strcpy(str,kstrIn.str);
  }
}

///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
VString& VString::operator=(const VString& obj)
{
  if(str) delete[] str;
  str = (Char_t*)NULL;
  len = 0;

  if(obj.str) {
    len = obj.len;
    str = new Char_t[len+1];
    strcpy(str,obj.str);
  }

  return(*this);
}
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
VString& VString::operator=(const Char_t* s)
{
  if(str) delete[] str;
  str = (Char_t*)NULL;
  len = 0;

  if(s && s[0]) {
    len = strlen(s);
    str = new Char_t[len+1];
    strcpy(str,s);
  }

  return(*this);
}

///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
void VString::append(const VString& s)
{
  if(0==s.len) return;
  append(s.str);
}

///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
void VString::append(const Char_t* s)
{
  if(!s) return;
  if(str) {
    len = len + strlen(s);
    Char_t* p = new Char_t[len+1];
    sprintf(p,"%s%s",str,s);
    delete[] str;
    str = p;
  }
  else {
    *this = s;
  }
}

///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
Int_t VString::Write(FILE* fp)
{
  fwrite((void*)(&len) ,sizeof(len) ,1,fp);

  if(len) fwrite((void*)(str) ,(size_t)(len+1) ,1,fp);

  return(SUCCESS);
}


///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
Int_t VString::Read(FILE* fp)
{
  if(str) delete[] str;
  str = (Char_t*)NULL;
  len = 0;

  fread((void*)(&len) ,sizeof(len) ,1,fp);

  if(len) {
    str = new Char_t[len+1];
    fread((void*)(str) ,(size_t)(len+1) ,1,fp);
  }

  return(SUCCESS);
}

Int_t Debug=0;

///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
int strcmp(const VString& a,const Char_t* b)
{
  if(a.len==0 && strlen(b)==0) {
    return 0;
  }
  else if(a.len>0 && strlen(b)>0) {
    return(strcmp(a.str,b));
  }
  else {
    return 1;
  }
}

///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
int strcmp(const Char_t* b,const VString& a)
{
  if(a.len==0 && strlen(b)==0) {
    return 0;
  }
  else if(a.len>0 && strlen(b)>0) {
    return(strcmp(a.str,b));
  }
  else {
    return 1;
  }
}

#if 0
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
Char_t VString::operator[](Int_t index)
{
  if(len && 0<=index && index<len) return str[index];
  else return 0;
}

///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
char* strcpy(VString& a,const VString& b)
{
  a=b;
}
#endif
