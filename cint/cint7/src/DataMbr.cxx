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

namespace std {}
using namespace std;
using namespace Cint::Internal;

static Cint::G__DataMemberInfo GetDataMemberFromAll(G__ClassInfo& cl, const char* varname);
static int IsInt(G__DataMemberInfo& member);
static Cint::G__DataMemberInfo GetDataMemberFromAllParents(G__ClassInfo& cl, const char* varname);

//______________________________________________________________________________
Cint::G__DataMemberInfo::~G__DataMemberInfo()
{
   delete m_memberof;
   delete m_typeinfo;
}

//______________________________________________________________________________
void Cint::G__DataMemberInfo::Init()
{
   m_datambr = ::Reflex::Member();
   m_scope = ::Reflex::Scope();
   m_memberiter = -1;
}

//______________________________________________________________________________
void Cint::G__DataMemberInfo::Init(class G__ClassInfo& a)
{
   m_name = "";
   if (a.IsValid()) {
      m_scope = a.ReflexType();
      m_datambr = ::Reflex::Member();
      m_memberiter = -1;
      G__incsetup_memvar((int)a.Tagnum());
   }
   else {
      m_scope = ::Reflex::Scope();
      m_datambr = ::Reflex::Member();
      m_memberiter = -1;
   }
}

//______________________________________________________________________________
void Cint::G__DataMemberInfo::Init(const ::Reflex::Scope &a)
{
   m_name = "";
   if (a) {
      m_scope = a;
      m_datambr = ::Reflex::Member();
      m_memberiter = -1;
      G__incsetup_memvar(G__get_tagnum(a));
   }
   else {
      m_scope = ::Reflex::Scope();
      m_datambr = ::Reflex::Member();
      m_memberiter = -1;
   }
}

//______________________________________________________________________________
void Cint::G__DataMemberInfo::Init(long handlein, long indexin, G__ClassInfo* belongingclassin)
{
   m_name = "";
   if (handlein) {
      m_datambr = G__Dict::GetDict().GetDataMember(handlein);
      m_memberiter = -1;
      if (belongingclassin && belongingclassin->IsValid()) {
         m_scope = belongingclassin->ReflexType();
      }
      else {
         m_scope = ::Reflex::Scope();
      }
   }
   else {
      m_datambr = ::Reflex::Member();
      m_memberiter = -1;
   }
}

//______________________________________________________________________________
const char* Cint::G__DataMemberInfo::Name()
{
   if (IsValid()) {
      if (m_name.length()==0) {
         m_name = m_datambr.Name();
      }
      return m_name.c_str();
   }
   return 0;
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

//______________________________________________________________________________
long Cint::G__DataMemberInfo::Property()
{
   if (IsValid()) {
      long property = 0;
      switch (G__get_access(m_datambr)) {
         case G__PUBLIC: property |= G__BIT_ISPUBLIC; break;
         case G__PROTECTED: property |= G__BIT_ISPROTECTED; break;
         case G__PRIVATE: property |= G__BIT_ISPRIVATE; break;
      }
      if (G__test_static(m_datambr, G__LOCALSTATIC)) property |= G__BIT_ISSTATIC;
      if (m_datambr.TypeOf().FinalType().IsReference()) property |= G__BIT_ISREFERENCE;
      if (isupper(G__get_type(m_datambr.TypeOf()))) property |= G__BIT_ISPOINTER;
      if (G__test_const(m_datambr, G__CONSTVAR)) property |= G__BIT_ISCONSTANT;
      if (G__test_const(m_datambr, G__PCONSTVAR)) property |= G__BIT_ISPCONSTANT;
      if (G__get_paran(m_datambr)) property |= G__BIT_ISARRAY;
      if (m_datambr.TypeOf().IsTypedef()) property |= G__BIT_ISTYPEDEF;
      if (m_datambr.TypeOf().RawType().IsFundamental()) property |= G__BIT_ISFUNDAMENTAL;
      else {
         std::string cname(m_datambr.TypeOf().RawType().Name());
         if ((cname == "G__longlong") || (cname == "G__ulonglong") || (cname == "G__longdouble")) {
            property |= G__BIT_ISFUNDAMENTAL;
            cname = m_datambr.TypeOf().FinalType().Name(Reflex::FINAL);
            if (
               (property & G__BIT_ISTYPEDEF) &&
               ((cname == "long long") || (cname == "unsigned long long") || (cname == "long double"))
            ) {
               property &= (~G__BIT_ISTYPEDEF);
            }
         }
         else {
            switch (G__get_tagtype(m_datambr.TypeOf().RawType())) {
               case 'c': property |= G__BIT_ISCLASS; break;
               case 's': property |= G__BIT_ISSTRUCT; break;
               case 'u': property |= G__BIT_ISUNION; break;
               case 'e': property |= G__BIT_ISENUM; break;
               case 'n': property |= G__BIT_ISNAMESPACE; break;
               default:  break;
            }
         }
      }
      return property;
   }
   return 0;
}

//______________________________________________________________________________
long Cint::G__DataMemberInfo::Offset()
{
   if (IsValid()) {
      return (long) G__get_properties(m_datambr)->addressOffset;
   }
   return -1;
}

//______________________________________________________________________________
int Cint::G__DataMemberInfo::Bitfield()
{
   if (IsValid()) {
      return G__get_bitfield_width(m_datambr);
   }
   return -1;
}

//______________________________________________________________________________
int Cint::G__DataMemberInfo::ArrayDim()
{
   if (IsValid()) {
      return G__get_paran(m_datambr);
   }
   return -1;
}

//______________________________________________________________________________
int Cint::G__DataMemberInfo::MaxIndex(int dim)
{
   if (IsValid() && (dim > -1) && (dim < G__get_paran(m_datambr))) {
      if (dim) {
         // -- Stored directly for second and greater dimensions.
         return G__get_varlabel(m_datambr, dim + 1);
      }
      else {
         // -- For first dimension divide number of elements by stride.
         // Note: This may be zero, if this is not an array!
         return G__get_varlabel(m_datambr, 1) /* num of elements */ / G__get_varlabel(m_datambr, 0) /* stride */;
      }
   }
   return -1;
}

//______________________________________________________________________________
void Cint::G__DataMemberInfo::SetGlobalcomp(int globalcomp)
{
   if (IsValid()) {
      G__get_properties(m_datambr)->globalcomp = globalcomp;
      //NOTE: Why do the following (and how in Reflex?)
      //if (G__NOLINK==globalcomp) var->access[index]=G__PRIVATE;
      //else                      var->access[index]=G__PUBLIC;
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
   m_scope = dict->var;
   m_memberiter = dict->ig15 - 1;
   m_datambr = m_scope.DataMemberAt(m_memberiter);
   return 1;
}

//______________________________________________________________________________
int Cint::G__DataMemberInfo::Next()
{
   if (m_scope) {
      m_name = "";
      ++m_memberiter;
      if (m_memberiter < (int)m_scope.DataMemberSize()) {
         m_datambr = m_scope.DataMemberAt(m_memberiter);
      } else {
         m_datambr = ::Reflex::Member();
         m_memberiter = -1;
      }
      return (bool) m_datambr;
   }
   return 0;
}

//______________________________________________________________________________
int Cint::G__DataMemberInfo::Prev()
{
   // FIXME: This is not thread safe!
   static std::vector<void*> prevbuf;
   static int prevbufindex;
   struct G__var_array* var = 0;
   if (m_scope) {
      m_name = "";
      if (m_memberiter == -1) {
         m_memberiter = m_scope.DataMemberSize();
      }
      --m_memberiter;
      if (m_memberiter > -1) {
         m_datambr = m_scope.DataMemberAt(m_memberiter);
         return 1;
      } else {
         m_datambr = ::Reflex::Member();
         return 0;
      }
   }
   return 0;
}

//______________________________________________________________________________
const char* Cint::G__DataMemberInfo::FileName()
{
#ifdef G__VARIABLEFPOS
   if (IsValid()) {
      if (G__get_properties(m_datambr)->filenum >= 0) {
         return G__srcfile[G__get_properties(m_datambr)->filenum].filename;
      }
      else {
         return "(compiled)";
      }
   }
   return 0;
#else
   G__fprinterr("Warning: Cint::G__DataMemberInfo::Filename() not supported in this configuration. define G__VARIABLEFPOS macro in platform dependency file and recompile cint");
   G__printlinenum();
   return 0;
#endif
}

//______________________________________________________________________________
int Cint::G__DataMemberInfo::LineNumber()
{
#ifdef G__VARIABLEFPOS
   if (IsValid()&& (G__get_properties(m_datambr)->filenum >= 0)) {
      return G__get_properties(m_datambr)->linenum;
   }
   return -1;
#else
   G__fprinterr("Warning: Cint::G__DataMemberInfo::LineNumber() not supported in this configuration. define G__VARIABLEFPOS macro in platform dependency file and recompile cint");
   G__printlinenum();
   return -1;
#endif
}

//______________________________________________________________________________
::Reflex::Type Cint::G__DataMemberInfo::ReflexType()
{
   // Return the data member type as a Reflex Type object.

   return m_datambr.TypeOf();
}

//______________________________________________________________________________
G__TypeInfo *Cint::G__DataMemberInfo::Type()
{
   // Return the data member type as a Cint TypeInfo object.
   // Note this is slow use ReflexType instead.

   delete m_typeinfo;
   m_typeinfo = new G__TypeInfo( m_datambr.TypeOf() );
   return m_typeinfo;
}

//______________________________________________________________________________
::Reflex::Scope Cint::G__DataMemberInfo::DeclaringScope()
{
   // Return the scope to which the member belongs.
   return m_scope;
}

//______________________________________________________________________________
G__ClassInfo *Cint::G__DataMemberInfo::MemberOf()
{
   // Return the scope to which the member belongs as a legacy Cint object.
   // Note: this is slow use DeclaringScope instead.
   
   delete m_memberof;
   m_memberof = new G__ClassInfo( m_scope.Name( ::Reflex::SCOPED ).c_str() );
   return m_memberof;
}

//______________________________________________________________________________
const char* Cint::G__DataMemberInfo::ValidArrayIndex(int* errnum, char** errstr)
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
                  if (!strcmp(m_local.Name(), Name())) {
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
//
// File static functions.
//______________________________________________________________________________

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

