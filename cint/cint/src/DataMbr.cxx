/* /% C++ %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file DataMbr.cxx
 ************************************************************************
 * Description:
 *  Extended Run Time Type Identification API
 ************************************************************************
 * Author                  Masaharu Goto
 * Copyright(c) 1995~2004  Masaharu Goto 
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/


#include "Api.h"
#include "common.h"
#include "FastAllocString.h"

/*********************************************************************
* class G__DataMemberInfo
*
*
*********************************************************************/
///////////////////////////////////////////////////////////////////////////
Cint::G__DataMemberInfo::G__DataMemberInfo():
   handle(0), index(0), belongingclass(NULL), type() 
{ Init(); }
///////////////////////////////////////////////////////////////////////////
Cint::G__DataMemberInfo::G__DataMemberInfo(const G__DataMemberInfo& dmi): 
   handle(dmi.handle), index(dmi.index), belongingclass(dmi.belongingclass), 
   type(dmi.type)
{}
///////////////////////////////////////////////////////////////////////////
Cint::G__DataMemberInfo::G__DataMemberInfo(class G__ClassInfo &a):
   handle(0), index(0), belongingclass(NULL), type()  
{ Init(a); }
///////////////////////////////////////////////////////////////////////////
Cint::G__DataMemberInfo& Cint::G__DataMemberInfo::operator=(const G__DataMemberInfo& dmi)
{
   if (&dmi != this) {
     handle=dmi.handle;
     index=dmi.index;
     belongingclass=dmi.belongingclass;
     type=dmi.type;
   }
   return *this;
}
///////////////////////////////////////////////////////////////////////////
void Cint::G__DataMemberInfo::Init()
{
  belongingclass = (G__ClassInfo*)NULL;
  handle = (long)(&G__global);
  index = -1;
}
///////////////////////////////////////////////////////////////////////////
void Cint::G__DataMemberInfo::Init(class G__ClassInfo &a)
{
  if(a.IsValid()) {
    belongingclass = &a;
    handle = (long)G__struct.memvar[a.Tagnum()];
    index = -1;
    G__incsetup_memvar((int)a.Tagnum());
  }
  else {
    belongingclass = (G__ClassInfo*)NULL;
    handle = 0;
    index = -1;
  }
}
///////////////////////////////////////////////////////////////////////////
void Cint::G__DataMemberInfo::Init(long handlein,long indexin
	,G__ClassInfo *belongingclassin)
{
  if(handlein) {
    handle = handlein;
    index = indexin;
    if(
       belongingclassin &&
       belongingclassin->IsValid()) {
      belongingclass = belongingclassin;
    }
    else {
      belongingclass=(G__ClassInfo*)NULL;
    }

    /* Set type */
    struct G__var_array *var;
    var = (struct G__var_array*)handle;
    type.type = var->type[index];
    type.tagnum=var->p_tagtable[index];
    type.typenum=var->p_typetable[index];
    type.reftype=var->reftype[index];
    type.class_property=0;
    type.isconst=var->constvar[index];
  }
  else {
    handle=handlein;
    index = -1;
  }
}
///////////////////////////////////////////////////////////////////////////
const char* Cint::G__DataMemberInfo::Name()
{
  if(IsValid()) {
    struct G__var_array *var;
    var = (struct G__var_array*)handle;
    return(var->varnamebuf[index]);
  }
  else {
    return((char*)NULL);
  }
}
///////////////////////////////////////////////////////////////////////////
const char* Cint::G__DataMemberInfo::Title()
{
  static char buf[G__INFO_TITLELEN];
  buf[0]='\0';
  if(IsValid()) {
    struct G__var_array *var;
    var = (struct G__var_array*)handle;
    G__getcomment(buf,&var->comment[index],var->tagnum);
    return(buf);
  }
  else {
    return((char*)NULL);
  }
}
///////////////////////////////////////////////////////////////////////////
Cint::G__TypeInfo* Cint::G__DataMemberInfo::Type()
{
   return &type;
}
///////////////////////////////////////////////////////////////////////////
Cint::G__ClassInfo* Cint::G__DataMemberInfo::MemberOf()
{
   return belongingclass;
}
///////////////////////////////////////////////////////////////////////////
long Cint::G__DataMemberInfo::Property()
{
  if(IsValid()) {
    long property=0;
    struct G__var_array *var;
    var = (struct G__var_array*)handle;
    switch(var->access[index]) {
    case G__PUBLIC: property|=G__BIT_ISPUBLIC; break;
    case G__PROTECTED: property|=G__BIT_ISPROTECTED; break;
    case G__PRIVATE: property|=G__BIT_ISPRIVATE; break;
    }
    if(G__LOCALSTATIC==var->statictype[index]) property|=G__BIT_ISSTATIC;
    if(G__USING_STATIC_VARIABLE==var->statictype[index]) property|=G__BIT_ISUSINGVARIABLE|G__BIT_ISSTATIC;
    if(G__USING_VARIABLE==var->statictype[index]) property|=G__BIT_ISUSINGVARIABLE;
    if(G__PARAREFERENCE==var->reftype[index]) property|=G__BIT_ISREFERENCE;
    if(isupper(var->type[index])) property|=G__BIT_ISPOINTER;
    if(var->constvar[index]&G__CONSTVAR) property|=G__BIT_ISCONSTANT;
    if(var->constvar[index]&G__PCONSTVAR) property|=G__BIT_ISPCONSTANT;
    if(var->paran[index]) property|=G__BIT_ISARRAY;
    if(-1!=var->p_typetable[index]) property|=G__BIT_ISTYPEDEF;
    if(-1==var->p_tagtable[index]) property|=G__BIT_ISFUNDAMENTAL;
    else {
      if(strcmp(G__struct.name[var->p_tagtable[index]],"G__longlong")==0 ||
	 strcmp(G__struct.name[var->p_tagtable[index]],"G__ulonglong")==0 ||
	 strcmp(G__struct.name[var->p_tagtable[index]],"G__longdouble")==0) {
	property |= G__BIT_ISFUNDAMENTAL;
	if(-1!=var->p_typetable[index] && 
	   (strcmp(G__newtype.name[var->p_typetable[index]],"long long")==0 ||
	    strcmp(G__newtype.name[var->p_typetable[index]],"unsigned long long")==0 ||
	    strcmp(G__newtype.name[var->p_typetable[index]],"long double")==0)) {
	  property &= (~G__BIT_ISTYPEDEF);
	}
      }
      else {
	switch(G__struct.type[var->p_tagtable[index]]) {
	case 'c': property|=G__BIT_ISCLASS; break;
	case 's': property|=G__BIT_ISSTRUCT; break;
	case 'u': property|=G__BIT_ISUNION; break;
	case 'e': property|=G__BIT_ISENUM; break;
	case 'n': property|=G__BIT_ISNAMESPACE; break;
	default:  break;
	}
      }
    }
    return(property);
  }
  else {
    return(0);
  }
}
///////////////////////////////////////////////////////////////////////////
long Cint::G__DataMemberInfo::Offset()
{
  if(IsValid()) {
    struct G__var_array *var;
    var = (struct G__var_array*)handle;
    return(var->p[index]);
  }
  else {
    return(-1);
  }
}
///////////////////////////////////////////////////////////////////////////
int Cint::G__DataMemberInfo::Bitfield() 
{
  if(IsValid()) {
    struct G__var_array *var;
    var = (struct G__var_array*)handle;
    return(var->bitfield[index]);
  }
  else {
    return(-1);
  }
}
///////////////////////////////////////////////////////////////////////////
int Cint::G__DataMemberInfo::ArrayDim()
{
  if(IsValid()) {
    struct G__var_array *var;
    var = (struct G__var_array*)handle;
    return(var->paran[index]);
  }
  else {
    return(-1);
  }
}
///////////////////////////////////////////////////////////////////////////
long Cint::G__DataMemberInfo::MaxIndex(int dim)
{
  if (IsValid()) {
    struct G__var_array* var = (struct G__var_array*) handle;
    if ((dim > -1) && (dim < var->paran[index])) {
      if (dim) {
        // -- Stored directly for second and greater dimensions.
	return var->varlabel[index][dim+1];
      }
      else {
        // -- For first dimension divide number of elements by stride.
        // Note: This may be zero, if this is not an array!
	return var->varlabel[index][1] /* num of elements*/ / var->varlabel[index][0] /* stride */;
      }
    }
  }
  return -1;
}
///////////////////////////////////////////////////////////////////////////
void Cint::G__DataMemberInfo::SetGlobalcomp(G__SIGNEDCHAR_T globalcomp)
{
  if(IsValid()) {
    struct G__var_array *var;
    var = (struct G__var_array*)handle;
    var->globalcomp[index] = globalcomp;
    if(G__NOLINK==globalcomp) var->access[index]=G__PRIVATE;
    else                      var->access[index]=G__PUBLIC;
  }
}
///////////////////////////////////////////////////////////////////////////
int Cint::G__DataMemberInfo::SerialNumber()
{
   return G__globals_serial;
}
///////////////////////////////////////////////////////////////////////////
int Cint::G__DataMemberInfo::IsValid()
{
  if(handle) {
    struct G__var_array *var;
    var = (struct G__var_array*)handle;
    if(0<=index&&index<var->allvar) return(1);
  }
  return(0);
}
///////////////////////////////////////////////////////////////////////////
int Cint::G__DataMemberInfo::SetFilePos(const char* fname)
{
  struct G__dictposition* dict=G__get_dictpos((char*)fname);
  if(!dict) return(0);
  handle = (long)dict->var;
  index = (long)(dict->ig15-1);
  belongingclass=(G__ClassInfo*)NULL;
  return(1);
}
///////////////////////////////////////////////////////////////////////////
int Cint::G__DataMemberInfo::Next()
{
  if(handle) {
    struct G__var_array *var;
    var = (struct G__var_array*)handle;
    ++index;
    if(var->allvar<=index) {
      int t = var->tagnum;
      var=var->next;
      if(var) {
	var->tagnum=t;
	index=0;
	handle=(long)var;
      }
      else {
	handle=0;
	index = -1;
      }
    }
    if(IsValid()) {
      type.type = var->type[index];
      type.tagnum=var->p_tagtable[index];
      type.typenum=var->p_typetable[index];
      type.reftype=var->reftype[index];
      type.class_property=0;
      type.isconst=var->constvar[index];
      return(1);
    }
    else {
      return(0);
    }
  }
  else {
    return(0);
  }
}
///////////////////////////////////////////////////////////////////////////
#include <vector>
namespace std { }
using namespace std;
int Cint::G__DataMemberInfo::Prev()
{
  struct G__var_array *var;
  static vector<void*> prevbuf;
  static size_t prevbufindex;
  if(handle) {
    if(-1==index) {
      var = (struct G__var_array*)handle;
      prevbuf.clear();
      while(var) {
	prevbuf.push_back((void*)var);
	var = var->next;
      } 
      prevbufindex = prevbuf.size()-1;
      handle = (long)prevbuf[prevbufindex];
      var = (struct G__var_array*)handle;
      index = var->allvar-1;
    }
    else {
      var = (struct G__var_array*)handle;
      --index;
      if(index<0) { 
	if(prevbufindex>0) {
	  int t = var->tagnum;
	  handle = (long)prevbuf[--prevbufindex];
	  var = (struct G__var_array*)handle;
	  index = var->allvar-1;
	  var->tagnum=t;
	}
	else {
	  handle=0;
	  index = -1;
	}
      }
    }
    if(IsValid()) {
      type.type = var->type[index];
      type.tagnum=var->p_tagtable[index];
      type.typenum=var->p_typetable[index];
      type.reftype=var->reftype[index];
      type.class_property=0;
      type.isconst=var->constvar[index];
      return(1);
    }
    else {
      return(0);
    }
  }
  else {
    return(0);
  }
}
///////////////////////////////////////////////////////////////////////////
const char* Cint::G__DataMemberInfo::FileName() {
#ifdef G__VARIABLEFPOS
  if(IsValid()) {
    struct G__var_array *var;
    var = (struct G__var_array*)handle;
    if(var->filenum[index]>=0) {
      return(G__srcfile[var->filenum[index]].filename);
    }
    else {
      return("(compiled)");
    }
  }
  else {
    return((char*)NULL);
  }
#else
  G__fprinterr("Warning: Cint::G__DataMemberInfo::Filename() not supported in this configuration. define G__VARIABLEFPOS macro in platform dependency file and recompile cint");
  G__printlinenum();
  return((char*)NULL);
#endif
}
///////////////////////////////////////////////////////////////////////////
int Cint::G__DataMemberInfo::LineNumber() {
#ifdef G__VARIABLEFPOS
  if(IsValid()) {
    struct G__var_array *var;
    var = (struct G__var_array*)handle;
    if(var->filenum[index]>=0) {
      return(var->linenum[index]);
    }
    else {
      return(-1);
    }
  }
  else {
    return(-1);
  }
#else
  G__fprinterr("Warning: Cint::G__DataMemberInfo::LineNumber() not supported in this configuration. define G__VARIABLEFPOS macro in platform dependency file and recompile cint");
  G__printlinenum();
  return(-1);
#endif
}

///////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////

// Find the data member whose name exactly match varname
// If this is 'augmented' to be able to interpret specified variable name
// (for example *fN)
// the function IsInt need to be changed to take that in consideration.
static G__DataMemberInfo GetDataMemberFromAll(G__ClassInfo & cl, const char* varname) {
  G__DataMemberInfo member(cl);
  while (member.Next()) {
    if (!strcmp(varname,member.Name())) {
      return member;
    }
  }
  // after the last Next member is now invalid.
  return member;
}

static int IsInt(G__DataMemberInfo &member) {
  int type = member.Type()->Type();
  if (member.Property() & G__BIT_ISARRAY) return 0;
  switch(type) {
    // A lower case indicated that it is NOT a pointer to that
    // type
    case 'b':  // unsigned char
    case 'c':  // char
    case 'r':  // unsigned short
    case 's':  // short
    case 'h':  // unsigned int
    case 'i':  // int
    case 'k':  // unsigned long
    case 'l': return 1; // long
    default: return 0;
  }
}

//
// Recurse through all the bases classes to find a 
// data member.
//
static G__DataMemberInfo GetDataMemberFromAllParents(G__ClassInfo & cl, const char* varname) {
  G__DataMemberInfo index;

  G__BaseClassInfo b( cl );
  while (b.Next()) {
    index = GetDataMemberFromAll(b, varname);
    if ( index.IsValid() ) {
      return index;
    }
    index = GetDataMemberFromAllParents( b, varname );
    if ( index.IsValid() ) {
      return index;
    }
  }
  return G__DataMemberInfo();
}

///////////////////////////////////////////////////////////////////////////

// ValidArrayIndex return a static string (so use it or copy it immediatly, do not
// call GrabIndex twice in the same expression) containing the size of the
// array data member.
// In case of error, or if the size is not specified, GrabIndex returns 0.
// If errnum is not null, *errnum updated with the error number:
//   Cint::G__DataMemberInfo::G__VALID     : valid array index
//   Cint::G__DataMemberInfo::G__NOT_INT   : array index is not an int
//   Cint::G__DataMemberInfo::G__NOT_DEF   : index not defined before array 
//                                          (this IS an error for streaming to disk)
//   Cint::G__DataMemberInfo::G__IS_PRIVATE: index exist in a parent class but is private
//   Cint::G__DataMemberInfo::G__UNKNOWN   : index is not known
// If errstr is not null, *errstr is updated with the address of a static
//   string containing the part of the index with is invalid.
const char* Cint::G__DataMemberInfo::ValidArrayIndex(int *errnum, char **errstr) {
  const char* title;
  //long dummy;

  // Let's see if the user provided us with some information
  // with the format: //[dimension] this is the dim of the array
  // dimension can be an arithmetical expression containing, literal integer,
  // the operator *,+ and - and data member of integral type.  In addition the
  // data members used for the size of the array need to be defined prior to
  // the array.

  if (errnum) *errnum = VALID;
  title = Title(); 

  if ((strncmp(title, "[", 1)!=0) ||
      (strstr(title,"]")     ==0)  ) return 0;

  G__FastAllocString working(G__INFO_TITLELEN);
  static char indexvar[G__INFO_TITLELEN];
  strncpy(indexvar, title + 1, sizeof(indexvar) - 1);
  strstr(indexvar,"]")[0] = '\0';
  
  // now we should have indexvar=dimension
  // Let's see if this is legal.
  // which means a combination of data member and digit separated by '*','+','-'

  // First we remove white spaces.
  unsigned int i,j;
  size_t indexvarlen = strlen(indexvar);
  for ( i=0,j=0; i<=indexvarlen; i++) {
    if (!isspace(indexvar[i])) {
       working.Set(j++, indexvar[i]);
    }
  };
 
  // Now we go through all indentifiers
  const char * tokenlist = "*+-";
  char *current = working;
  current = strtok(current,tokenlist);
  
  while (current!=0) {
    // Check the token
    if (isdigit(current[0])) {
      for(i=0;i<strlen(current);i++) {
	if (!isdigit(current[0])) {
	  // Error we only access integer.
	  //NOTE: *** Need to print an error;
	  //fprintf(stderr,"*** Datamember %s::%s: size of array (%s) is not an interger\n",
	  //	    member.MemberOf()->Name(), member.Name(), current);
	  if (errstr) *errstr = current;
	  if (errnum) *errnum = NOT_INT;
	  return 0;
	}
      }
    } else { // current token is not a digit
      // first let's see if it is a data member:
      int found = 0;
      G__DataMemberInfo index1 = GetDataMemberFromAll(*belongingclass, current );
      if ( index1.IsValid() ) {
	if ( IsInt(index1) ) {
	  found = 1;
	  // Let's see if it has already been wrote down in the
	  // Streamer.
	  // Let's see if we already wrote it down in the
	  // streamer.
	  G__DataMemberInfo m_local( *belongingclass);
	  while (m_local.Next()) {
	    if (!strcmp(m_local.Name(),Name())) {
	      // we reached the current data member before
	      // reaching the index so we have not wrote it yet!
	      //NOTE: *** Need to print an error;
	      //fprintf(stderr,"*** Datamember %s::%s: size of array (%s) has not been defined before the array \n",
	      //	member.MemberOf()->Name(), member.Name(), current);
	      if (errstr) *errstr = current;
	      if (errnum) *errnum = NOT_DEF;
	      return 0;
	    }
	    if (!strcmp(m_local.Name(),current) ) {
	      break;
	    }
	  } // end of while (m_local.Next())
	} else {
	  //NOTE: *** Need to print an error;
	  //fprintf(stderr,"*** Datamember %s::%s: size of array (%s) is not int \n",
	  //	    member.MemberOf()->Name(), member.Name(), current);
	  if (errstr) *errstr = current;
	  if (errnum) *errnum = NOT_INT;
	  return 0;
	}
      } else {
	// There is no variable by this name in this class, let see
	// the base classes!:
	index1 = GetDataMemberFromAllParents( *belongingclass, current );
	if ( index1.IsValid() ) {
	  if ( IsInt(index1) ) {
	    found = 1;
	  } else {
	    // We found a data member but it is the wrong type
	    //NOTE: *** Need to print an error;
	    //fprintf(stderr,"*** Datamember %s::%s: size of array (%s) is not int \n",
	    //	member.MemberOf()->Name(), member.Name(), current);
	    if (errnum) *errnum = NOT_INT;
	    if (errstr) *errstr = current;
	    return 0;
	  }
	  if ( found && (index1.Property() & G__BIT_ISPRIVATE) ) {
	    //NOTE: *** Need to print an error;
	    //fprintf(stderr,"*** Datamember %s::%s: size of array (%s) is a private member of %s \n",
	    if (errstr) *errstr = current;
	    if (errnum) *errnum = IS_PRIVATE;
	    return 0;
	  }
	}
	if (!found) {
	  //NOTE: *** Need to print an error;
	  //fprintf(stderr,"*** Datamember %s::%s: size of array (%s) is not known \n",
	  //	    member.MemberOf()->Name(), member.Name(), indexvar);
	  if (errstr) *errstr = indexvar;
	  if (errnum) *errnum = UNKNOWN;
	  return 0;
	} // end of if not found
      } // end of if is a data member of the class
    } // end of if isdigit

    current = strtok(0,tokenlist);
  } // end of while loop on tokens	

  return indexvar;

}

///////////////////////////////////////////////////////////////////////////

		
///////////////////////////////////////////////////////////////////////////


