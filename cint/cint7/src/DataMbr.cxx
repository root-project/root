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
#include "Dict.h"
#include "common.h"
#include "fproto.h"

#include <cstdlib>
#include <cstring>
#include <cctype>
#include <string>
#include <vector>

using namespace Cint::Internal;
using namespace std;

// File static functions.
static Cint::G__DataMemberInfo GetDataMemberFromAll(G__ClassInfo& cl, const char* varname);
static int IsInt(G__DataMemberInfo& member);
static Cint::G__DataMemberInfo GetDataMemberFromAllParents(G__ClassInfo& cl, const char* varname);

//______________________________________________________________________________
//
//  Static functions.
//

//______________________________________________________________________________
static Cint::G__DataMemberInfo GetDataMemberFromAll(G__ClassInfo& cl, const char* varname)
{
   // Find the data member whose name exactly match varname
   // If this is 'augmented' to be able to interpret specified variable name
   // (for example *fN)
   // the function IsInt need to be changed to take that in consideration.
   G__DataMemberInfo member(cl);
   while (member.Next()) {
      if (!strcmp(varname, member.Name())) {
         return member;
      }
   }
   // after the last Next member is now invalid.
   return member;
}

//______________________________________________________________________________
static int IsInt(G__DataMemberInfo& member)
{
   int type = member.Type()->Type();
   if (member.Property() & G__BIT_ISARRAY) {
      return 0;
   }
   switch (type) {
      // -- A lower case indicated that it is NOT a pointer to that type.
      case 'b': // unsigned char
      case 'c': // char
      case 'r': // unsigned short
      case 's': // short
      case 'h': // unsigned int
      case 'i': // int
      case 'k': // unsigned long
      case 'l': // long
         return 1;
         break;
   }
   return 0;
}

//______________________________________________________________________________
static Cint::G__DataMemberInfo GetDataMemberFromAllParents(G__ClassInfo& cl, const char* varname)
{
   //
   // Recurse through all the bases classes to find a
   // data member.
   //
   G__DataMemberInfo index;
   G__BaseClassInfo b(cl);
   while (b.Next()) {
      index = GetDataMemberFromAll(b, varname);
      if (index.IsValid()) {
         return index;
      }
      index = GetDataMemberFromAllParents(b, varname);
      if (index.IsValid()) {
         return index;
      }
   }
   return G__DataMemberInfo();
}

//______________________________________________________________________________
//
//  Class Member Functions.
//

//______________________________________________________________________________
Cint::G__DataMemberInfo::~G__DataMemberInfo()
{
   delete m_memberof;
   delete m_typeinfo;
}

//______________________________________________________________________________
Cint::G__DataMemberInfo::G__DataMemberInfo()
: m_memberiter(-1)
, m_typeinfo(0) // we own, cache object
, m_memberof(0) // we own, cache object
{
}

//______________________________________________________________________________
Cint::G__DataMemberInfo::G__DataMemberInfo(const G__DataMemberInfo& rhs)
: m_scope(rhs.m_scope)
, m_datambr(rhs.m_datambr)
, m_memberiter(rhs.m_memberiter)
, m_typeinfo(0) // we own, cache object
, m_memberof(0) // we own, cache object
{
}

//______________________________________________________________________________
Cint::G__DataMemberInfo::G__DataMemberInfo(class G__ClassInfo& class_info)
: m_memberiter(-1)
, m_typeinfo(0) // we own, cache object
, m_memberof(0) // we own, cache object
{
   Init(class_info);
}

#if 0
//______________________________________________________________________________
Cint::G__DataMemberInfo::G__DataMemberInfo(const ::Reflex::Member mbr)
: m_scope(mbr.DeclaringScope())
, m_datambr(mbr)
, m_memberiter(-1)
, m_typeinfo(0) // we own, cache object
, m_memberof(0) // we own, cache object
{
}
#endif // 0

//______________________________________________________________________________
G__DataMemberInfo& Cint::G__DataMemberInfo::operator=(const G__DataMemberInfo& rhs)
{
   if (this != &rhs) {
      m_scope = rhs.m_scope;
      m_datambr = rhs.m_datambr;
      m_memberiter = rhs.m_memberiter;
      delete m_typeinfo; // we own, cache object
      m_typeinfo = 0; // we own, cache object
      delete m_memberof; // we own, cache object
      m_memberof =  0; // we own, cache object
   }
   return *this;
}

//______________________________________________________________________________
void Cint::G__DataMemberInfo::Init()
{
   m_scope = ::Reflex::Scope();
   m_datambr = ::Reflex::Member();
   m_memberiter = -1;
   delete m_typeinfo;
   m_typeinfo = 0; // we own, cache object
   delete m_memberof;
   m_memberof = 0; // we own, cache object
}

//______________________________________________________________________________
void Cint::G__DataMemberInfo::Init(class G__ClassInfo& a)
{
   if (a.IsValid()) {
      m_scope = G__Dict::GetDict().GetScope(a.Tagnum());
      m_datambr = ::Reflex::Member();
      m_memberiter = -1;
      delete m_typeinfo; // we own, cache object
      m_typeinfo = 0; // we own, cache object
      delete m_memberof; // we own, cache object
      m_memberof = 0; // we own, cache object
      G__incsetup_memvar((int) a.Tagnum());
   }
   else {
      m_scope = ::Reflex::Scope();
      m_datambr = ::Reflex::Member();
      m_memberiter = -1;
      delete m_typeinfo; // we own, cache object
      m_typeinfo = 0; // we own, cache object
      delete m_memberof; // we own, cache object
      m_memberof = 0; // we own, cache object
   }
}

#if 0
//______________________________________________________________________________
void Cint::G__DataMemberInfo::Init(const ::Reflex::Scope a)
{
   m_memberiter = -1;
   delete m_typeinfo; // we own, cache object
   m_typeinfo = 0; // we own, cache object
   delete m_memberof; // we own, cache object
   m_memberof = 0; // we own, cache object
   if (!a) {
      m_scope = ::Reflex::Scope();
      m_datambr = ::Reflex::Member();
   }
   else {
      m_scope = a;
      m_datambr = ::Reflex::Member();
      G__incsetup_memvar(G__get_tagnum(a));
   }
}
#endif // 0

//______________________________________________________________________________
void Cint::G__DataMemberInfo::Init(long handlein, long indexin, G__ClassInfo* belongingclassin)
{
   // handlein: reflex scope containg data member
   // indexin: data member index in scope
   // belongingclassin: class info of containing class
   m_scope = ::Reflex::Scope();
   m_datambr = ::Reflex::Member();
   m_memberiter = -1;
   delete m_typeinfo; // we own, cache object
   m_typeinfo = 0; // we own, cache object
   delete m_memberof; // we own, cache object
   m_memberof =  0; // we own, cache object
   if (!handlein) {
      return;
   }
   if (belongingclassin && belongingclassin->IsValid()) {
      m_scope = G__Dict::GetDict().GetScope(belongingclassin->Tagnum());
      m_datambr = m_scope.DataMemberAt(indexin);
      m_memberiter = indexin;
      ::Reflex::Type ty = m_datambr.TypeOf();
      m_typeinfo = new G__TypeInfo;
      G__get_cint5_type_tuple_long(ty, &m_typeinfo->fType, &m_typeinfo->fTagnum, &m_typeinfo->fTypenum, &m_typeinfo->fReftype, &m_typeinfo->fIsconst);
      m_memberof = new G__ClassInfo(m_scope.Name(::Reflex::SCOPED).c_str());
   }
}

//______________________________________________________________________________
size_t Cint::G__DataMemberInfo::Handle()
{
   return (size_t) m_datambr.Id();
}

//______________________________________________________________________________
int Cint::G__DataMemberInfo::Index()
{
   return 0;
}

//______________________________________________________________________________
const char* Cint::G__DataMemberInfo::Name()
{
   // Note: We need a static buffer because the string must continue
   //       to exist after we exit and Reflex::Member::Name() returns
   //       a std::string by value.  We cannot use a G__StrBuf for this
   //       because there is no guarantee that its buffer reservoir has
   //       been initialized at static initialization time.
   static char mname[G__MAXNAME];
   if (!IsValid()) {
      return 0;
   }
   mname[0] = '\0';
   strcpy(mname, m_datambr.Name().c_str());
   return mname;
}

//______________________________________________________________________________
const char* Cint::G__DataMemberInfo::Title()
{
   static char buf[G__INFO_TITLELEN];
   buf[0] = '\0';
   if (IsValid()) {
      G__getcomment(buf, &G__get_properties(m_datambr)->comment, G__get_tagnum(m_scope));
      return buf;
   }
   return 0;
}

#if 0
//______________________________________________________________________________
::Reflex::Type Cint::G__DataMemberInfo::ReflexType()
{
   // Return the data member type as a Reflex Type object.
   return m_datambr.TypeOf();
}
#endif // 0

//______________________________________________________________________________
G__TypeInfo* Cint::G__DataMemberInfo::Type()
{
   // Return the data member type as a Cint TypeInfo object.
   // Note this is slow use ReflexType instead.
   if (m_typeinfo) {
      return m_typeinfo;
   }
   ::Reflex::Type ty = m_datambr.TypeOf();
   m_typeinfo = new G__TypeInfo;
   G__get_cint5_type_tuple_long(ty, &m_typeinfo->fType, &m_typeinfo->fTagnum, &m_typeinfo->fTypenum, &m_typeinfo->fReftype, &m_typeinfo->fIsconst);
   return m_typeinfo;
}

//______________________________________________________________________________
long Cint::G__DataMemberInfo::Property()
{
   if (!IsValid()) {
      return 0L;
   }
   char type = '\0';
   int tagnum = -1;
   int typenum = -1;
   int reftype = 0;
   int isconst = 0;
   G__get_cint5_type_tuple(m_datambr.TypeOf(), &type, &tagnum, &typenum, &reftype, &isconst);
   G__RflxVarProperties* prop = G__get_properties(m_datambr);
   long result = 0L;
   if (m_datambr.IsPublic()) {
      result |= G__BIT_ISPUBLIC;
   }
   else if (m_datambr.IsProtected()) {
      result |= G__BIT_ISPROTECTED;
   }
   else if (m_datambr.IsPrivate()) {
      result |= G__BIT_ISPRIVATE;
   }
   if (prop->statictype == G__LOCALSTATIC) {
      result |= G__BIT_ISSTATIC;
   }
   if (reftype == G__PARAREFERENCE) {
      result |= G__BIT_ISREFERENCE;
   }
   if (isupper(type)) {
      result |= G__BIT_ISPOINTER;
   }
   if (isconst & G__CONSTVAR) {
      result |= G__BIT_ISCONSTANT;
   }
   if (isconst & G__PCONSTVAR) {
      result |= G__BIT_ISPCONSTANT;
   }
   if (G__get_paran(m_datambr)) {
      result |= G__BIT_ISARRAY;
   }
   if (typenum != -1) {
      result |= G__BIT_ISTYPEDEF;
   }
   if (tagnum == -1) {
      result |= G__BIT_ISFUNDAMENTAL;
   }
   else {
      std::string cname(m_datambr.TypeOf().RawType().Name());
      if (
         (cname == "G__longlong") ||
         (cname == "G__ulonglong") ||
         (cname == "G__longdouble")
      ) {
         result |= G__BIT_ISFUNDAMENTAL;
         if (typenum != -1) {
            std::string tname = G__Dict::GetDict().GetTypedef(typenum).Name();
            if (
               (tname == "long long") ||
               (tname == "unsigned long long") ||
               (tname == "long double")
            ) {
               result &= (~G__BIT_ISTYPEDEF);
            }
         }
      }
      else {
         switch (G__get_tagtype(m_datambr.TypeOf().RawType())) {
            case 'c':
               result |= G__BIT_ISCLASS;
               break;
            case 'e':
               result |= G__BIT_ISENUM;
               break;
            case 'n':
               result |= G__BIT_ISNAMESPACE;
               break;
            case 's':
               result |= G__BIT_ISSTRUCT;
               break;
            case 'u':
               result |= G__BIT_ISUNION;
               break;
         }
      }
   }
   return result;
}

//______________________________________________________________________________
long Cint::G__DataMemberInfo::Offset()
{
   if (!IsValid()) {
      return -1;
   }
   return (long) G__get_offset(m_datambr);
}

//______________________________________________________________________________
int Cint::G__DataMemberInfo::Bitfield()
{
   if (!IsValid()) {
      return -1;
   }
   return G__get_bitfield_width(m_datambr);
}

//______________________________________________________________________________
int Cint::G__DataMemberInfo::ArrayDim()
{
   if (!IsValid()) {
      return -1;
   }
   return G__get_paran(m_datambr);
}

//______________________________________________________________________________
int Cint::G__DataMemberInfo::MaxIndex(int dim)
{
   if (!IsValid() || (dim < 0) || (dim >= G__get_paran(m_datambr))) {
      return -1;
   }
   if (dim) {
      // -- Stored directly for second and greater dimensions.
      return G__get_varlabel(m_datambr, dim + 1);
   }
   // -- For first dimension divide number of elements by stride.
   // Note: This may be zero, if this is not an array!
   return G__get_varlabel(m_datambr, 1) /* num of elements */ / G__get_varlabel(m_datambr, 0) /* stride */;
}

#if 0
//______________________________________________________________________________
::Reflex::Scope Cint::G__DataMemberInfo::DeclaringScope()
{
   // Return the scope to which the member belongs.
   return m_scope;
}
#endif // 0

//______________________________________________________________________________
G__ClassInfo* Cint::G__DataMemberInfo::MemberOf()
{
   // Return the scope to which the member belongs as a legacy Cint object.
   // Note: this is slow use DeclaringScope instead.
   delete m_memberof;
   m_memberof = new G__ClassInfo(m_scope.Name(::Reflex::SCOPED).c_str());
   return m_memberof;
}

//______________________________________________________________________________
void Cint::G__DataMemberInfo::SetGlobalcomp(int globalcomp)
{
   if (IsValid()) {
      G__get_properties(m_datambr)->globalcomp = globalcomp;
   }
}

//______________________________________________________________________________
int Cint::G__DataMemberInfo::IsValid()
{
   bool valid = m_datambr;
   return valid;
}

//______________________________________________________________________________
int Cint::G__DataMemberInfo::SetFilePos(const char* fname)
{
   struct G__dictposition* dict = G__get_dictpos((char*) fname);
   if (!dict) {
      return 0;
   }
   delete m_typeinfo;
   m_typeinfo = 0;
   delete m_memberof;
   m_memberof = 0;
   m_scope = dict->var;
   m_memberiter = dict->ig15 - 1;
   m_datambr = m_scope.DataMemberAt(m_memberiter);
   return 1;
}

//______________________________________________________________________________
int Cint::G__DataMemberInfo::Next()
{
   if (!m_scope) {
      return 0;
   }
   ++m_memberiter;
   delete m_typeinfo;
   m_typeinfo = 0;
   if (m_memberiter < (int) m_scope.DataMemberSize()) {
      m_datambr = m_scope.DataMemberAt(m_memberiter);
   } else {
      m_memberiter = -1;
      m_datambr = ::Reflex::Member();
   }
   return (bool) m_datambr;
}

//______________________________________________________________________________
int Cint::G__DataMemberInfo::Prev()
{
   if (!m_scope) {
      return 0;
   }
   delete m_typeinfo;
   m_typeinfo = 0;
   if (m_memberiter == -1) {
      m_memberiter = m_scope.DataMemberSize();
   }
   --m_memberiter;
   if (m_memberiter < 0) {
      m_datambr = ::Reflex::Member();
      return 0;
   }
   m_datambr = m_scope.DataMemberAt(m_memberiter);
   return 1;
}

//______________________________________________________________________________
const char* Cint::G__DataMemberInfo::ValidArrayIndex(int* errnum /*= 0*/, char** errstr /*= 0*/)
{
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
   const char* title = 0;
   //long dummy;

   // Let's see if the user provided us with some information
   // with the format: //[dimension] this is the dim of the array
   // dimension can be an arithmetical expression containing, literal integer,
   // the operator *,+ and - and data member of integral type.  In addition the
   // data members used for the size of the array need to be defined prior to
   // the array.

   if (errnum) {
      *errnum = VALID;
   }
   title = Title();
   if (strncmp(title, "[", 1) || !strstr(title, "]")) {
      return 0;
   }
   // FIXME: This is not thread safe!
   static char working[G__INFO_TITLELEN];
   static char indexvar[G__INFO_TITLELEN];
   strcpy(indexvar, title + 1);
   strstr(indexvar, "]")[0] = '\0';

   // now we should have indexvar=dimension
   // Let's see if this is legal.
   // which means a combination of data member and digit separated by '*','+','-'

   // First we remove white spaces.
   unsigned int i = 0;
   unsigned int j = 0;
   for (; i <= strlen(indexvar); ++i) {
      if (!isspace(indexvar[i])) {
         working[j++] = indexvar[i];
      }
   };

   // Now we go through all indentifiers
   const char* tokenlist = "*+-";
   char* current = working;
   current = strtok(current, tokenlist);
   G__ClassInfo belongingclass(G__get_tagnum(m_scope));
   while (current) {
      // Check the token
      if (isdigit(current[0])) {
         for(i = 0; i <strlen(current); ++i) {
            if (!isdigit(current[0])) {
               // Error we only access integer.
               //NOTE: *** Need to print an error;
               //fprintf(stderr,"*** Datamember %s::%s: size of array (%s) is not an interger\n",
               //	    member.MemberOf()->Name(), member.Name(), current);
               if (errstr) {
                  *errstr = current;
               }
               if (errnum) {
                  *errnum = NOT_INT;
               }
               return 0;
            }
         }
      } else {
         // current token is not a digit
         // first let's see if it is a data member:
         int found = 0;
         G__DataMemberInfo index1 = GetDataMemberFromAll(belongingclass, current );
         if (index1.IsValid()) {
            if (IsInt(index1)) {
               found = 1;
               // Let's see if it has already been wrote down in the
               // Streamer.
               // Let's see if we already wrote it down in the
               // streamer.
               G__DataMemberInfo m_local(belongingclass);
               while (m_local.Next()) {
                  if (m_local.m_datambr.Name() == m_datambr.Name()) {
                     // we reached the current data member before
                     // reaching the index so we have not wrote it yet!
                     //NOTE: *** Need to print an error;
                     //fprintf(stderr,"*** Datamember %s::%s: size of array (%s) has not been defined before the array \n",
                     //	member.MemberOf()->Name(), member.Name(), current);
                     if (errstr) {
                        *errstr = current;
                     }
                     if (errnum) {
                        *errnum = NOT_DEF;
                     }
                     return 0;
                  }
                  if (!strcmp(m_local.Name(), current)) {
                     break;
                  }
               } // end of while (m_local.Next())
            } else {
               //NOTE: *** Need to print an error;
               //fprintf(stderr,"*** Datamember %s::%s: size of array (%s) is not int \n",
               //	    member.MemberOf()->Name(), member.Name(), current);
               if (errstr) {
                  *errstr = current;
               }
               if (errnum) {
                  *errnum = NOT_INT;
               }
               return 0;
            }
         } else {
            // There is no variable by this name in this class, let see
            // the base classes!
            index1 = GetDataMemberFromAllParents(belongingclass, current);
            if (index1.IsValid()) {
               if (IsInt(index1)) {
                  found = 1;
               } else {
                  // We found a data member but it is the wrong type
                  //NOTE: *** Need to print an error;
                  //fprintf(stderr,"*** Datamember %s::%s: size of array (%s) is not int \n",
                  //	member.MemberOf()->Name(), member.Name(), current);
                  if (errnum) {
                     *errnum = NOT_INT;
                  }
                  if (errstr) {
                     *errstr = current;
                  }
                  return 0;
               }
               if (found && (index1.Property() & G__BIT_ISPRIVATE)) {
                  //NOTE: *** Need to print an error;
                  //fprintf(stderr,"*** Datamember %s::%s: size of array (%s) is a private member of %s \n",
                  if (errstr) {
                     *errstr = current;
                  }
                  if (errnum) {
                     *errnum = IS_PRIVATE;
                  }
                  return 0;
               }
            }
            if (!found) {
               //NOTE: *** Need to print an error;
               //fprintf(stderr,"*** Datamember %s::%s: size of array (%s) is not known \n",
               //	    member.MemberOf()->Name(), member.Name(), indexvar);
               if (errstr) {
                  *errstr = indexvar;
               }
               if (errnum) {
                  *errnum = UNKNOWN;
               }
               return 0;
            } // end of if not found
         } // end of if is a data member of the class
      } // end of if isdigit
      current = strtok(0, tokenlist);
   } // end of while loop on tokens	
   return indexvar;
}

//______________________________________________________________________________
const char* Cint::G__DataMemberInfo::FileName()
{
   // --
#ifdef G__VARIABLEFPOS
   if (!IsValid()) {
      return 0;
   }
   if (G__get_properties(m_datambr)->filenum >= 0) {
      return G__srcfile[G__get_properties(m_datambr)->filenum].filename;
   }
   return "(compiled)";
#else // G__VARIABLEFPOS
   G__fprinterr("Warning: Cint::G__DataMemberInfo::Filename() not supported in this configuration. define G__VARIABLEFPOS macro in platform dependency file and recompile cint");
   G__printlinenum();
   return 0;
#endif // G__VARIABLEFPOS
   // --
}

//______________________________________________________________________________
int Cint::G__DataMemberInfo::LineNumber()
{
   // --
#ifdef G__VARIABLEFPOS
   if (!IsValid()) {
      return -1;
   }
   if (G__get_properties(m_datambr)->filenum >= 0) { // interpreted code
      return G__get_properties(m_datambr)->linenum;
   }
   return -1; // compiled code
#else // G__VARIABLEFPOS
   G__fprinterr("Warning: Cint::G__DataMemberInfo::LineNumber() not supported in this configuration. define G__VARIABLEFPOS macro in platform dependency file and recompile cint");
   G__printlinenum();
   return -1;
#endif // G__VARIABLEFPOS
   // --
}

