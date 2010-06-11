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
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "Api.h"
#include "common.h"

/*********************************************************************
* class G__TypedefInfo
* 
*********************************************************************/
///////////////////////////////////////////////////////////////////////////
void Cint::G__TypedefInfo::Init() 
{
  type = 0;
  typenum = -1;
  tagnum = -1;
  isconst = 0;
}
///////////////////////////////////////////////////////////////////////////
void Cint::G__TypedefInfo::Init(const char *typenamein)
{
  char store_var_type = G__var_type;
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
  G__var_type = store_var_type;
}
///////////////////////////////////////////////////////////////////////////
void Cint::G__TypedefInfo::Init(int typenumin)
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
void Cint::G__TypedefInfo::SetGlobalcomp(int globalcomp)
{
  if(IsValid()) {
    G__newtype.globalcomp[typenum] = globalcomp;
  }
}
///////////////////////////////////////////////////////////////////////////
int Cint::G__TypedefInfo::IsValid()
{
  if(-1!=typenum&&typenum<G__newtype.alltype) {
    return(1);
  }
  else {
    return(0);
  }
}
///////////////////////////////////////////////////////////////////////////
int Cint::G__TypedefInfo::SetFilePos(const char *fname)
{
  struct G__dictposition* dict=G__get_dictpos((char*)fname);
  if(!dict) return(0);
  Init((int)dict->typenum-1);
  return(1);
}
///////////////////////////////////////////////////////////////////////////
int Cint::G__TypedefInfo::Next()
{
  Init((int)typenum+1);
  return(IsValid());
}
///////////////////////////////////////////////////////////////////////////
const char* Cint::G__TypedefInfo::Title()
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
G__ClassInfo Cint::G__TypedefInfo::EnclosingClassOfTypedef()
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
const char* Cint::G__TypedefInfo::FileName() {
#ifdef G__TYPEDEFFPOS
  if(IsValid()) {
    return(G__srcfile[G__newtype.filenum[typenum]].filename);
  }
  else {
    return((char*)NULL);
  }
#else
  G__fprinterr("Warning: Cint::G__TypedefInfo::FIleName() not supported in this configuration. define G__TYPEDEFFPOS macro in platform dependency file and recompile cint");
  G__printlinenum();
  return((char*)NULL);
#endif
}
///////////////////////////////////////////////////////////////////////////
int Cint::G__TypedefInfo::LineNumber() {
#ifdef G__TYPEDEFFPOS
  if(IsValid()) {
    return(G__newtype.linenum[typenum]);
  }
  else {
    return(-1);
  }
#else
  G__fprinterr("Warning: Cint::G__TypedefInfo::LineNumber() not supported in this configuration. define G__TYPEDEFFPOS macro in platform dependency file and recompile cint");
  G__printlinenum();
  return(-1);
#endif
}
///////////////////////////////////////////////////////////////////////////
int Cint::G__TypedefInfo::GetNumTypedefs()
{
   // Retrieve the number of typedef registered in the system.
   return G__newtype.alltype;
}
