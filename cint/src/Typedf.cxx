/* /% C++ %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file Typedf.cxx
 ************************************************************************
 * Description:
 *  Extended Run Time Type Identification API
 ************************************************************************
 * Author                  Masaharu Goto 
 * Copyright(c) 1995~1999  Masaharu Goto 
 *
 * Permission to use, copy, modify and distribute this software and its 
 * documentation for any purpose is hereby granted without fee,
 * provided that the above copyright notice appear in all copies and
 * that both that copyright notice and this permission notice appear
 * in supporting documentation.  The author makes no
 * representations about the suitability of this software for any
 * purpose.  It is provided "as is" without express or implied warranty.
 ************************************************************************/

#include "Api.h"
#include "common.h"

/*********************************************************************
* class G__TypedefInfo
* 
*********************************************************************/
///////////////////////////////////////////////////////////////////////////
void G__TypedefInfo::Init() 
{
  type = 0;
  typenum = -1;
  tagnum = -1;
  isconst = 0;
}
///////////////////////////////////////////////////////////////////////////
void G__TypedefInfo::Init(const char *typenamein)
{
#ifndef G__OLDIMPLEMENTATION1944
  char store_var_type = G__var_type;
#endif
  typenum = G__defined_typename(typenamein);
  if(-1!=typenum&&typenum<G__newtype.alltype) {
    tagnum = G__newtype.tagnum[typenum];
    type = G__newtype.type[typenum];
    reftype = G__newtype.reftype[typenum];
    isconst = 0;
  }
  else {
    type=0;
    tagnum= -1;
    typenum= -1;
    isconst= 0;
  }
#ifndef G__OLDIMPLEMENTATION1944
  G__var_type = store_var_type;
#endif
}
///////////////////////////////////////////////////////////////////////////
void G__TypedefInfo::Init(int typenumin)
{
  typenum = typenumin;
  if(-1!=typenum&&typenum<G__newtype.alltype) {
    tagnum = G__newtype.tagnum[typenum];
    type = G__newtype.type[typenum];
    reftype = G__newtype.reftype[typenum];
    isconst = 0;
  }
  else {
    type=0;
    tagnum= -1;
    typenum= -1;
    isconst= 0;
  }
}
///////////////////////////////////////////////////////////////////////////
void G__TypedefInfo::SetGlobalcomp(int globalcomp)
{
  if(IsValid()) {
    G__newtype.globalcomp[typenum] = globalcomp;
  }
}
///////////////////////////////////////////////////////////////////////////
int G__TypedefInfo::IsValid()
{
  if(-1!=typenum&&typenum<G__newtype.alltype) {
    return(1);
  }
  else {
    return(0);
  }
}
///////////////////////////////////////////////////////////////////////////
int G__TypedefInfo::SetFilePos(const char *fname)
{
  struct G__dictposition* dict=G__get_dictpos((char*)fname);
  if(!dict) return(0);
  Init((int)dict->typenum-1);
  return(1);
}
///////////////////////////////////////////////////////////////////////////
int G__TypedefInfo::Next()
{
  Init((int)typenum+1);
  return(IsValid());
}
///////////////////////////////////////////////////////////////////////////
const char* G__TypedefInfo::Title()
{
  static char buf[G__INFO_TITLELEN];
  buf[0]='\0';
  if(IsValid()) {
    G__getcommenttypedef(buf,&G__newtype.comment[typenum],(int)typenum);
    return(buf);
  }
  else {
    return((char*)NULL);
  }
}
///////////////////////////////////////////////////////////////////////////
G__ClassInfo G__TypedefInfo::EnclosingClassOfTypedef()
{
  if(IsValid()) {
    G__ClassInfo enclosingclass(G__newtype.parent_tagnum[typenum]);
    return(enclosingclass);
  }
  else {
    G__ClassInfo enclosingclass;
    return(enclosingclass);
  }
}
///////////////////////////////////////////////////////////////////////////
const char* G__TypedefInfo::FileName() {
#ifdef G__TYPEDEFFPOS
  if(IsValid()) {
    return(G__srcfile[G__newtype.filenum[typenum]].filename);
  }
  else {
    return((char*)NULL);
  }
#else
  G__fprinterr("Warning: G__TypedefInfo::FIleName() not supported in this configuration. define G__TYPEDEFFPOS macro in platform dependency file and recompile cint");
  G__printlinenum();
  return((char*)NULL);
#endif
}
///////////////////////////////////////////////////////////////////////////
int G__TypedefInfo::LineNumber() {
#ifdef G__TYPEDEFFPOS
  if(IsValid()) {
    return(G__newtype.linenum[typenum]);
  }
  else {
    return(-1);
  }
#else
  G__fprinterr("Warning: G__TypedefInfo::LineNumber() not supported in this configuration. define G__TYPEDEFFPOS macro in platform dependency file and recompile cint");
  G__printlinenum();
  return(-1);
#endif
}
///////////////////////////////////////////////////////////////////////////
