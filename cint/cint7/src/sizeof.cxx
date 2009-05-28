/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file sizeof.c
 ************************************************************************
 * Description:
 *  Getting object size
 ************************************************************************
 * Copyright(c) 1995~2002  Masaharu Goto
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "common.h"
#include "vararg.h"

#include "Dict.h"

using namespace Cint::Internal;

//______________________________________________________________________________
// array index for type_info. This must correspond to the class member
// layout of class type_info in the <typeinfo.h>

#define G__TYPEINFO_VIRTUALID 0
#define G__TYPEINFO_TYPE      1
#define G__TYPEINFO_TAGNUM    2
#define G__TYPEINFO_TYPENUM   3
#define G__TYPEINFO_REFTYPE   4
#define G__TYPEINFO_SIZE      5
#define G__TYPEINFO_ISCONST   6

//______________________________________________________________________________
int G__rootCcomment = 0;

//______________________________________________________________________________
extern "C" void G__loadlonglong(int* ptag, int* ptype, int which)
{
   // -- FIXME: Describe this function!
   int lltag = -1;
   ::Reflex::Type lltype;
   ::Reflex::Type ulltype;
   ::Reflex::Type ldtype;
   int ulltag = -1;
   int ldtag = -1;
   int store_decl = G__decl;
   int store_def_struct_member = G__def_struct_member;
   int flag = 0;
   ::Reflex::Scope store_tagdefining = G__tagdefining;
   ::Reflex::Scope store_def_tagnum = G__def_tagnum;
   G__tagdefining = ::Reflex::Scope();
   G__def_tagnum = ::Reflex::Scope();
   G__def_struct_member = 0;
   G__decl = 0;
   if (0 == G__defined_macro("G__LONGLONG_H")) {
      G__loadfile("long.dll"); // used to switch case between .dl and .dll
      flag = 1;
   }
   G__decl = 1;
   G__def_struct_member = store_def_struct_member;
   if (which == G__LONGLONG || flag) {
      lltag = G__defined_tagname("G__longlong", 2);
      lltype = Reflex::Type::ByName("long long");
   }
   if (which == G__ULONGLONG || flag) {
      ulltag = G__defined_tagname("G__ulonglong", 2);
      ulltype = Reflex::Type::ByName("unsigned long long");
   }
   if (which == G__LONGDOUBLE || flag) {
      ldtag = G__defined_tagname("G__longdouble", 2);
      ldtype = Reflex::Type::ByName("long double");
   }
   switch (which) {
      case G__LONGLONG:
         *ptag = lltag;
         *ptype = G__get_typenum(lltype);
         break;
      case G__ULONGLONG:
         *ptag = ulltag;
         *ptype = G__get_typenum(ulltype);
         break;
      case G__LONGDOUBLE:
         *ptag = ldtag;
         *ptype = G__get_typenum(ldtype);
         break;
   }
   G__def_tagnum = store_def_tagnum;
   G__tagdefining = store_tagdefining;
   G__decl = store_decl;
   return;
}

//______________________________________________________________________________
extern "C" int G__sizeof(G__value* object)
{
   ::Reflex::Type ty = G__value_typenum(*object);
   char type = G__get_type(ty);
   int reftype = G__get_reftype(ty);
   if (isupper(type) && (reftype != G__PARANORMAL)) {
      return G__LONGALLOC;
   }
   switch (toupper(type)) {
      case 'B': // unsigned char*
      case 'C': // char*
      case 'E': // FILE*
      case 'Y': // void
#ifndef G__OLDIMPLEMENTATION2191
      case '1': // pointer to function
#else // G__OLDIMPLEMENTATION2191
      case 'Q': // pointer to function
#endif // G__OLDIMPLEMENTATION2191
         return G__CHARALLOC;
      case 'R':
      case 'S':
         return G__SHORTALLOC;
      case 'H':
      case 'I':
         return G__INTALLOC;
      case 'K':
      case 'L':
         return G__LONGALLOC;
      case 'F':
         return G__FLOATALLOC;
      case 'D':
         return G__DOUBLEALLOC;
      case 'U':
         return ty.RawType().SizeOf();
      case 'A': // pointer to member function
         return G__P2MFALLOC;
      case 'G': // bool
         // --
#ifdef G__BOOL4BYTE
         return G__INTALLOC;
#else // G__BOOL4BYTE
         return G__CHARALLOC;
#endif // G__BOOL4BYTE
      case 'N':
      case 'M':
         return G__LONGLONGALLOC;
#ifndef G__OLDIMPLEMENTATION2191
      case 'Q':
         return G__LONGDOUBLEALLOC;
#endif // G__OLDIMPLEMENTATION2191
         // --
   }
   return 1;
}

//______________________________________________________________________________
int Cint::Internal::G__sizeof_deref(const G__value* object)
{
   ::Reflex::Type type(G__deref(G__value_typenum(*object)));
   int result = type.SizeOf();
   if (G__get_type(type) == 'g') {
      // --
#ifdef G__BOOL4BYTE
      assert(G__INTALLOC == result);
#else // G__BOOL4BYTE
      assert(G__CHARALLOC == result);
#endif // G__BOOL4BYTE
      // --
   }
   return result;
}

//______________________________________________________________________________
long Cint::Internal::G__Loffsetof(char* tagname, char* memname)
{
   int tagnum = G__defined_tagname(tagname, 0);
   if (tagnum == -1) {
      return -1;
   }
   int hash;
   int junk;
   G__hash(memname, hash, junk)
   G__incsetup_memvar(tagnum);
   ::Reflex::Type type = G__Dict::GetDict().GetType(tagnum);
   ::Reflex::Member m = type.MemberByName(memname);
   if (!m) {
      G__fprinterr(G__serr, "Error: member %s not found in %s ", memname, tagname);
      G__genericerror(0);
      return -1;
   }
   return (long) G__get_offset(m);
}

//______________________________________________________________________________
extern "C" int G__Lsizeof(const char* type_name_in)
{
   const char* type_name = type_name_in;
   //
   //  If the type name ends with '*', then
   //  return the size of a pointer to void.
   //
   if (type_name[strlen(type_name)-1] == '*') {
      return sizeof(void*);
   }
   //
   //  Remove some possible qualifiers from
   //  the type name.
   //
   if (
      !strncmp(type_name, "struct", 6) ||
      !strncmp(type_name, "signed", 6)
   ) {
      type_name = type_name + 6;
   }
   else if (!strncmp(type_name, "class", 5)) {
      type_name = type_name + 5;
   }
   else if (!strncmp(type_name, "union", 5)) {
      type_name = type_name + 5;
   }
   //
   //  Handle a class name now.
   //
   {
      int tagnum = G__defined_tagname(type_name, 1);
      if (tagnum != -1) {
         if (G__struct.type[tagnum] == 'e') {
            return G__INTALLOC;
         }
         return G__struct.size[tagnum];
      }
   }
   //
   //  Handle a typedef now.
   //
   {
      ::Reflex::Type typenum = G__find_typedef(type_name);
      if (typenum) {
         int result = -1;
         switch (G__get_type(typenum)) {
            case 'n':
            case 'm':
               result = sizeof(G__int64);
               break;
            case 'q':
               result = sizeof(long double);
               break;
            case 'g':
#ifdef G__BOOL4BYTE
               result = sizeof(int);
               break;
#endif // G__BOOL4BYTE
            case 'b':
            case 'c':
               result = sizeof(char);
               break;
            case 'h':
            case 'i':
               result = sizeof(int);
               break;
            case 'r':
            case 's':
               result = sizeof(short);
               break;
            case 'k':
            case 'l':
               result = sizeof(long);
               break;
            case 'f':
               result = sizeof(float);
               break;
            case 'd':
               result = sizeof(double);
               break;
            case 'v':
               return -1;
            default:
               // if struct or union
               if (isupper(G__get_type(typenum))) {
                  result = sizeof(void *);
               }
               else if (G__get_tagnum(typenum) != -1) {
                  result = G__struct.size[G__get_tagnum(typenum)];
               }
               return 0;
         }
         if (G__get_nindex(typenum)) {
            std::vector<int> ind = G__get_index(typenum);
            for (unsigned int ig15 = 0; ig15 < ind.size(); ++ig15) {
               result *= ind[ig15];
            }
         }
         return result;
      }
   }
   //
   //  Handle a fundamental type now.
   //
   if (
      !strcmp(type_name, "int") ||
      !strcmp(type_name, "unsignedint")
   ) {
      return sizeof(int);
   }
   else if (
      !strcmp(type_name, "long") ||
      !strcmp(type_name, "longint") ||
      !strcmp(type_name, "unsignedlong") ||
      !strcmp(type_name, "unsignedlongint")
   ) {
      return sizeof(long);
   }
   else if (
      !strcmp(type_name, "short") ||
      !strcmp(type_name, "shortint") ||
      !strcmp(type_name, "unsignedshort") ||
      !strcmp(type_name, "unsignedshortint")
   ) {
      return sizeof(short);
   }
   else if (
      !strcmp(type_name, "char") ||
      !strcmp(type_name, "unsignedchar")
   ) {
      return sizeof(char);
   }
   else if (
      !strcmp(type_name, "float") ||
      !strcmp(type_name, "float")
   ) {
      return sizeof(float);
   }
   else if (!strcmp(type_name, "double")) {
      return sizeof(double);
   }
   else if (!strcmp(type_name, "longdouble")) {
      return sizeof(long double);
   }
   else if (
      !strcmp(type_name, "longlong") ||
      !strcmp(type_name, "longlongint")
   ) {
      return sizeof(G__int64);
   }
   else if (
      !strcmp(type_name, "unsignedlonglong") ||
      !strcmp(type_name, "unsignedlonglongint")
   ) {
      return sizeof(G__uint64);
   }
   else if (!strcmp(type_name, "void")) {
      return sizeof(void*);
   }
   else if (!strcmp(type_name, "FILE")) {
      return sizeof(FILE);
   }
   else if (!strcmp(type_name, "bool")) {
      // --
#ifdef G__BOOL4BYTE
      return sizeof(int);
#else // G__BOOL4BYTE
      return sizeof(unsigned char);
#endif // G__BOOL4BYTE
      // --
   }
   //
   //  Handle a variable name now.
   //
   {
      int pointlevel = 0;
      while (type_name[pointlevel] == '*') {
         ++pointlevel;
      }
      G__StrBuf namebody_sb(G__MAXNAME + 20);
      char* namebody = namebody_sb;
      strcpy(namebody, type_name + pointlevel);
      {
         char* p = strrchr(namebody, '[');
         while (p) {
            *p = '\0';
            ++pointlevel;
            p = strrchr(namebody, '[');
         }
      }
      ::Reflex::Member var;
      {
         int hash = 0;
         int junk = 0;
         G__hash(namebody, hash, junk)
         var = G__getvarentry(namebody, hash, ::Reflex::Scope::GlobalScope(), G__p_local);
      }
      if (!var) {
         std::string temp;
         G__get_stack_varname(temp, namebody, G__func_now, G__get_tagnum(G__memberfunc_tagnum));
         int hash = 0;
         int junk = 0;
         G__hash(temp.c_str(), hash, junk);
         var = G__getvarentry(temp.c_str(), hash, ::Reflex::Scope::GlobalScope(), G__p_local);
      }
      if (var) {
         char type = '\0';
         int tagnum = -1;
         int typenum = -1;
         int reftype = 0;
         int isconst = 0;
         G__get_cint5_type_tuple(var.TypeOf(), &type, &tagnum, &typenum, &reftype, &isconst);
         int paran = G__get_nindex(var.TypeOf());
         if (G__get_varlabel(var.TypeOf(), 1) == INT_MAX /* unspecified size array flag */) {
            if (type == 'c') {
               return strlen((char*) G__get_offset(var)) + 1;
            }
            else {
               return sizeof(void *);
            }
         }
         char buf_type = type;
         int buf_tagnum = tagnum;
         int buf_typenum = typenum;
         int buf_reftype = 0;
         int buf_isconst = 0;
         if (isupper(type)) {
            buf_reftype = reftype;
         }
         int num_of_elements = 0;
         if (pointlevel > paran) {
            switch (pointlevel) {
               case 0:
                  break;
               case 1:
                  if (buf_reftype == G__PARANORMAL) {
                     buf_type = tolower(buf_type);
                  }
                  else if (buf_reftype == G__PARAP2P) {
                     buf_reftype = G__PARANORMAL;
                  }
                  else {
                     --buf_reftype;
                  }
                  break;
               case 2:
                  if (buf_reftype == G__PARANORMAL) {
                     buf_type = tolower(buf_type);
                  }
                  else if (buf_reftype == G__PARAP2P) {
                     buf_type = tolower(buf_type);
                     buf_reftype = G__PARANORMAL;
                  }
                  else if (buf_reftype == G__PARAP2P2P) {
                     buf_reftype = G__PARANORMAL;
                  }
                  else {
                     buf_reftype -= 2;
                  }
                  break;
            }
            G__value buf;
            G__value_typenum(buf) = G__cint5_tuple_to_type(buf_type, buf_tagnum, buf_typenum, buf_reftype, buf_isconst);
            return G__sizeof(&buf);
         }
         switch (pointlevel) {
            case 0:
               num_of_elements = G__get_varlabel(var.TypeOf(), 1) /* num of elements */;
               if (!num_of_elements) {
                  num_of_elements = 1;
               }
               break;
            case 1:
               num_of_elements = G__get_varlabel(var.TypeOf(), 0) /* stride */;
               break;
            default:
               num_of_elements = G__get_varlabel(var.TypeOf(), 0) /* stride */;
               for (int i = 1; i < pointlevel; ++i) {
                  num_of_elements /= G__get_varlabel(var.TypeOf(), i + 1);
               }
               break;
         }
         if (isupper(type)) {
            return num_of_elements * sizeof(void*);
         }
         G__value buf;
         G__value_typenum(buf) = G__cint5_tuple_to_type(buf_type, buf_tagnum, buf_typenum, buf_reftype, buf_isconst);
         return num_of_elements * G__sizeof(&buf);
      }
   }
   //
   //  Handle an expression now.
   //
   G__value buf = G__getexpr((char*) type_name);
   if (G__get_type(G__value_typenum(buf))) {
      if (
         (G__get_type(G__value_typenum(buf)) == 'C') &&
         (type_name[0] == '"')
      ) {
         return strlen((char*) buf.obj.i) + 1;
      }
      return G__sizeof(&buf);
   }
   //
   //  We have completely failed.
   //
   return -1;
}

//______________________________________________________________________________
#ifdef G__TYPEINFO
long* Cint::Internal::G__typeid(char* typenamein)
{
   G__value buf;
   int c;
   long *type_info;
   int type = 0;
   int tagnum;
   int reftype = G__PARANORMAL;
   int size = 0;
   ::Reflex::Type typenum;
   int len;
   int pointlevel = 0;
   int isref = 0;
   int tag_type_info;
   G__StrBuf typenamebuf_sb(G__MAXNAME*2);
   char* typenamebuf = typenamebuf_sb;
   char* type_name;
   int isconst = 0;

   /**********************************************************************
   * Get type_info tagname
   ***********************************************************************/
   tag_type_info = G__defined_tagname("type_info", 1);
   if (tag_type_info == -1) {
      G__genericerror("Error: class type_info not defined. <typeinfo.h> must be included");
      return 0;
   }
   /**********************************************************************
   * In case of typeid(X&) , typeid(X*) , strip & or *
   ***********************************************************************/
   strcpy(typenamebuf, typenamein);
   type_name = typenamebuf;
   len = strlen(type_name);

   while ('*' == (c = type_name[len-1]) || '&' == c) {
      switch (c) {
         case '*':
            ++pointlevel;
            break;
         case '&':
            isref = 1;
            break;
      }
      --len;
      type_name[len] = '\0';
   }
   /**********************************************************************
   * Search for typedef names
   **********************************************************************/
   typenum = G__find_typedef(type_name);
   if (typenum) {
      type    = G__get_type(typenum);
      tagnum  = G__get_tagnum(typenum);
      reftype = G__get_reftype(typenum);
      if (-1 != tagnum) {
         size = G__struct.size[tagnum];
      }
      else {
         switch (tolower(type)) {
            case 'n':
            case 'm':
               size = G__LONGLONGALLOC;
               break;
            case 'g':
#ifdef G__BOOL4BYTE
               size = G__INTALLOC;
               break;
#endif // G__BOOL4BYTE
            case 'b':
            case 'c':
               size = G__CHARALLOC;
               break;
            case 'r':
            case 's':
               size = G__SHORTALLOC;
               break;
            case 'h':
            case 'i':
               size = G__INTALLOC;
               break;
            case 'k':
            case 'l':
               size = G__LONGALLOC;
               break;
            case 'f':
               size = G__FLOATALLOC;
               break;
            case 'd':
               size = G__DOUBLEALLOC;
               break;
            case 'e':
            case 'y':
               size = -1;
               break;
            case 'a':
               size = G__sizep2memfunc;
               break;
            case 'q':
               break;
         }
      }
   }
   else {
      /*********************************************************************
       * Search for class/struct/union names
       *********************************************************************/
      if ((strncmp(type_name, "struct", 6) == 0)) {
         type_name = type_name + 6;
      }
      else if ((strncmp(type_name, "class", 5) == 0)) {
         type_name = type_name + 5;
      }
      else if ((strncmp(type_name, "union", 5) == 0)) {
         type_name = type_name + 5;
      }
      tagnum = G__defined_tagname(type_name, 1);
      if (-1 != tagnum) {
         reftype = G__PARANORMAL;
         switch (G__struct.type[tagnum]) {
            case 'u':
            case 's':
            case 'c':
               type = 'u';
               size = G__struct.size[tagnum];
               break;
            case 'e':
               type = 'i';
               size = G__INTALLOC;
               break;
            case 'n':
               size = G__struct.size[tagnum];
               G__genericerror("Error: can not get sizeof namespace");
               break;
         }
      }
      else {
         /********************************************************************
          * Search for intrinsic types
          *******************************************************************/
         reftype = G__PARANORMAL;
         if (strcmp(type_name, "int") == 0) {
            type = 'i';
            size = G__INTALLOC;
         }
         if (strcmp(type_name, "unsignedint") == 0) {
            type = 'h';
            size = G__INTALLOC;
         }
         if ((strcmp(type_name, "long") == 0) ||
               (strcmp(type_name, "longint") == 0)) {
            type = 'l';
            size = G__LONGALLOC;
         }
         if ((strcmp(type_name, "unsignedlong") == 0) ||
               (strcmp(type_name, "unsignedlongint") == 0)) {
            type = 'k';
            size = G__LONGALLOC;
         }
         if ((strcmp(type_name, "longlong") == 0)) {
            type = 'n';
            size = G__LONGLONGALLOC;
         }
         if ((strcmp(type_name, "unsignedlonglong") == 0)) {
            type = 'm';
            size = G__LONGLONGALLOC;
         }
         if ((strcmp(type_name, "short") == 0) ||
               (strcmp(type_name, "shortint") == 0)) {
            type = 's';
            size = G__SHORTALLOC;
         }
         if ((strcmp(type_name, "unsignedshort") == 0) ||
               (strcmp(type_name, "unsignedshortint") == 0)) {
            type = 'r';
            size = G__SHORTALLOC;
         }
         if ((strcmp(type_name, "char") == 0) ||
               (strcmp(type_name, "signedchar") == 0)) {
            type = 'c';
            size = G__CHARALLOC;
         }
         if (strcmp(type_name, "unsignedchar") == 0) {
            type = 'b';
            size = G__CHARALLOC;
         }
         if (strcmp(type_name, "float") == 0) {
            type = 's';
            size = G__FLOATALLOC;
         }
         if ((strcmp(type_name, "double") == 0)
            ) {
            type = 'd';
            size = G__DOUBLEALLOC;
         }
         if ((strcmp(type_name, "longdouble") == 0)
            ) {
            type = 'q';
            size = G__LONGDOUBLEALLOC;
         }
         if (strcmp(type_name, "void") == 0) {
            type = 'y';
            size = sizeof(void*);
         }
         if (strcmp(type_name, "FILE") == 0) {
            type = 'e';
            size = -1;
         }
      }
   }
   /**********************************************************************
    * If no type name matches, evaluate the expression and get the type
    * information of the object
    *********************************************************************/
   if (!type) {
      buf = G__getexpr(typenamein);
      typenum = G__value_typenum(buf);
      type = G__get_type(typenum);
      tagnum = G__get_tagnum(typenum);
      isref = 0;
      isconst = typenum.IsConst();
      if (-1 != tagnum && 'u' == tolower(type) && buf.ref && G__PVOID != G__struct.virtual_offset[tagnum]) {
         /* In case of polymorphic object, get the actual tagnum from the hidden
          * virtual identity field.  */
         tagnum = *(long*)(buf.obj.i + G__struct.virtual_offset[tagnum]);
      }
   }
   /*********************************************************************
    * Identify reference and pointer level
    *********************************************************************/
   if (isref) {
      reftype = G__PARAREFERENCE;
      if (pointlevel) type = toupper(type);
   }
   else {
      if (isupper(type)) {
         ++pointlevel;
         type = tolower(type);
      }
      switch (pointlevel) {
         case 0:
            reftype = G__PARANORMAL;
            break;
         case 1:
            type = toupper(type);
            reftype = G__PARANORMAL;
            break;
         case 2:
            type = toupper(type);
            reftype = G__PARAP2P;
            break;
         case 3:
            type = toupper(type);
            reftype = G__PARAP2P2P;
            break;
      }
   }
   if (isupper(type)) size = G__LONGALLOC;
   /**********************************************************************
    * Create temporary object for return value and copy the reslut
    **********************************************************************/
   G__alloc_tempobject(tag_type_info, -1);
   type_info = (long*)G__p_tempbuf->obj.obj.i;
   type_info[G__TYPEINFO_VIRTUALID] = tag_type_info;
   type_info[G__TYPEINFO_TYPE] = type;
   type_info[G__TYPEINFO_TAGNUM] = tagnum;
   type_info[G__TYPEINFO_TYPENUM] = G__get_typenum(typenum);
   type_info[G__TYPEINFO_REFTYPE] = reftype;
   type_info[G__TYPEINFO_SIZE] = size;
   type_info[G__TYPEINFO_ISCONST] = isconst;
   return type_info;
}
#endif

//______________________________________________________________________________
void Cint::Internal::G__getcomment(char* buf, int tagnum)
{
   G__RflxProperties* prop = G__get_properties(G__Dict::GetDict().GetScope(tagnum));
   G__getcomment(buf, &prop->comment, tagnum);
}

//______________________________________________________________________________
void Cint::Internal::G__getcomment(char* buf, Reflex::Scope scope)
{
   G__RflxProperties* prop = G__get_properties(scope);
   G__getcomment(buf, &prop->comment, G__get_tagnum(scope));
}

//______________________________________________________________________________
void Cint::Internal::G__getcomment(char* buf, G__comment_info* pcomment, int tagnum)
{
   buf[0] = '\0';
   if (pcomment->filenum == -1) { // No comment.
      return;
   }
   if (pcomment->filenum == -2) { // Compiled class, get comment from dictionary, not source file.
      strcpy(buf, pcomment->p.com);
      return;
   }
   if ( // We have an invalid filenum, an invalid class, or a compiled class, cannot do anything.
      (pcomment->filenum < 0) || // We have an invalid filenum, or
      (tagnum == -1) || // We have an invalid class, or
      (G__struct.iscpplink[tagnum] != G__NOLINK) // class is not interpreted, so we have no source file.
   ) {
      return;
   }
   FILE* fp = G__mfp;
   if (pcomment->filenum != G__MAXFILE) {
      fp = G__srcfile[pcomment->filenum].fp;
   }
   int flag = 0; // If 1, then we did not open a file.
   fpos_t store_pos;
   if (fp) {
      flag = 1;
      fgetpos(fp, &store_pos);
   }
   else {
      // Open the right file even in case where we use the preprocessor.
      if ((pcomment->filenum < G__MAXFILE) && G__srcfile[pcomment->filenum].prepname) {
         fp = fopen(G__srcfile[pcomment->filenum].prepname, "r");
      }
      else {
         fp = fopen(G__srcfile[pcomment->filenum].filename, "r");
      }
   }
   // Set file position to the comment string.
   fsetpos(fp, &pcomment->p.pos);
   // Read in one line.
   fgets(buf, G__ONELINE - 1, fp);
   //
   //  Remove end-of-line characters.
   //
   char* p = strchr(buf, '\n');
   if (p) {
      *p = '\0';
   }
   p = strchr(buf, '\r');
   if (p) {
      *p = '\0';
   }
   if (G__rootCcomment) { // If we are processing C-style comments.
      // Remove the comment termination sequence.
      p = G__strrstr(buf, "*/");
      if (p) {
         *p = '\0';
      }
   }
   if (flag) { // We did not open a file, restore original file position.
      fsetpos(fp, &store_pos);
   }
   else {
      fclose(fp); // otherwise, close the file we opened.
   }
   return;
}

//______________________________________________________________________________
void Cint::Internal::G__getcommenttypedef(char* buf, G__comment_info* pcomment, ::Reflex::Type typenum)
{
   buf[0] = '\0';
   if (!typenum || (pcomment->filenum == -1)) { // Invalid type, or no comment.
      return;
   }
   if (pcomment->filenum == -2) { // Compiled class, get comment from dictionary, not source file.
      strcpy(buf, pcomment->p.com);
      return;
   }
   if ( // We have an invalid filenum, no properties, or a compiled class, cannot do anything.
      (pcomment->filenum < 0) ||
      !G__get_properties(typenum) ||
      (G__get_properties(typenum)->iscpplink != G__NOLINK)
   ) {
      return;
   }
   FILE* fp = G__mfp;
   if (pcomment->filenum != G__MAXFILE) {
      fp = G__srcfile[pcomment->filenum].fp;
   }
   int flag = 0; // If 1, then we did not open a file.
   fpos_t store_pos;
   if (fp) {
      flag = 1;
      fgetpos(fp, &store_pos);
   }
   else {
      // Open the right file even in case where we use the preprocessor.
      if ((pcomment->filenum < G__MAXFILE) && G__srcfile[pcomment->filenum].prepname) {
         fp = fopen(G__srcfile[pcomment->filenum].prepname, "r");
      }
      else {
         fp = fopen(G__srcfile[pcomment->filenum].filename, "r");
      }
   }
   if (!fp) {
      G__fprinterr(G__serr, "G__getcommenttypedef: Could not open the file #%d (%s) to retrieve the comment!", pcomment->filenum, G__srcfile[pcomment->filenum].filename);
      G__genericerror(0);
      fprintf(stderr, "G__getcommenttypedef %p %d %d %p %p %s\n", pcomment, pcomment->filenum, G__nfile, pcomment->p.com, fp, G__srcfile[pcomment->filenum].filename);
      return;
   }
   // Set file position to the comment string.
   fsetpos(fp, &pcomment->p.pos);
   // Read in one line.
   fgets(buf, G__ONELINE - 1, fp);
   //
   //  Remove end-of-line characters.
   //
   char* p = strchr(buf, '\n');
   if (p) {
      *p = '\0';
   }
   p = strchr(buf, '\r');
   if (p) {
      *p = '\0';
   }
   //
   //  Terminate string after first semicolon.
   //
   p = strchr(buf, ';');
   if (p) {
      p[1] = '\0';
   }
   if (flag) { // We did not open a file, restore original file position.
      fsetpos(fp, &store_pos);
   }
   else {
      fclose(fp); // otherwise, close the file we opened.
   }
   return;
}

//______________________________________________________________________________
long Cint::Internal::G__get_classinfo(char* item, int tagnum)
{
   char *buf;
   int tag_string_buf;
   struct G__inheritance *baseclass;
   int p;
   size_t i;

   /**********************************************************************
    * get next class/struct
    **********************************************************************/
   if (strcmp("next", item) == 0) {
      while (1) {
         ++tagnum;
         if (tagnum < 0 || G__struct.alltag <= tagnum) return(-1);
         if (('s' == G__struct.type[tagnum] || 'c' == G__struct.type[tagnum]) &&
               -1 == G__struct.parent_tagnum[tagnum]) {
            return((long)tagnum);
         }
      }
   }

   /**********************************************************************
    * check validity
    **********************************************************************/
   if (tagnum < 0 || G__struct.alltag <= tagnum ||
         ('c' != G__struct.type[tagnum] && 's' != G__struct.type[tagnum]))
      return(0);

   /**********************************************************************
    * return type
    **********************************************************************/
   if (strcmp("type", item) == 0) {
      switch (G__struct.type[tagnum]) {
         case 'e':
            return((long)'i');
         default:
            return((long)'u');
      }
   }

   /**********************************************************************
    * size
    **********************************************************************/
   if (strcmp("size", item) == 0) {
      return(G__struct.size[tagnum]);
   }

   /**********************************************************************
    * baseclass
    **********************************************************************/
   if (strcmp("baseclass", item) == 0) {
      tag_string_buf = G__defined_tagname("G__string_buf", 0);
      G__alloc_tempobject(tag_string_buf, -1);
      buf = (char*)G__p_tempbuf->obj.obj.i;

      baseclass = G__struct.baseclass[tagnum];
      if (!baseclass) return((long)0);
      p = 0;
      buf[0] = '\0';
      for (i = 0;i < baseclass->vec.size();i++) {
         if (baseclass->vec[i].property&G__ISDIRECTINHERIT) {
            if (p) {
               sprintf(buf + p, ",");
               ++p;
            }
            sprintf(buf + p, "%s%s" , G__access2string(baseclass->vec[i].baseaccess)
                    , G__struct.name[baseclass->vec[i].basetagnum]);
            p = strlen(buf);
         }
      }

      return((long)buf);
   }

   /**********************************************************************
    * title
    **********************************************************************/
   if (strcmp("title", item) == 0) {
      tag_string_buf = G__defined_tagname("G__string_buf", 0);
      G__alloc_tempobject(tag_string_buf, -1);
      buf = (char*)G__p_tempbuf->obj.obj.i;

      G__getcomment(buf, tagnum);
      return((long)buf);
   }

   /**********************************************************************
    * isabstract
    **********************************************************************/
   if (strcmp("isabstract", item) == 0) {
      return(G__struct.isabstract[tagnum]);
   }
   return(0);
}

//______________________________________________________________________________
static const char* CopyInTempObject(const std::string& what)
{
   // Copy a string and put in the list of temporary object.

   int tag_string_buf = G__defined_tagname("G__string_buf", 0);
   G__alloc_tempobject(tag_string_buf, -1);
   char *buf = (char*)G__p_tempbuf->obj.obj.i;
   if (what.size() > 254) {
      strncpy(buf, what.c_str(), 252);
      strcpy(buf + 252, "...");
   }
   else {
      strcpy(buf, what.c_str());
   }
   return buf;
}

//______________________________________________________________________________
long Cint::Internal::G__get_variableinfo(char* item, long* phandle, long* pindex, int tagnum)
{
   char *buf;
   int tag_string_buf;
   ::Reflex::Scope var;
   unsigned int index;

   /*******************************************************************
    * new
    *******************************************************************/
   if (strcmp("new", item) == 0) {
      *pindex = 0;
      if (-1 == tagnum) {
         *phandle = (long)(::Reflex::Scope::GlobalScope().Id());
      }
      else if (G__Dict::GetDict().GetScope(tagnum)) {
         G__incsetup_memvar(tagnum);
         *phandle = (long)(G__Dict::GetDict().GetScope(tagnum).Id());
      }
      else {
         *phandle = 0;
      }
      return(0);
   }

   var = G__Dict::GetDict().GetScope(*phandle);
   index = (*pindex);

   if (!var || var.DataMemberSize() <= index) {
      *phandle = 0;
      *pindex = 0;
      return(0);
   }

   /*******************************************************************
    * next
    *******************************************************************/
   if (strcmp("next", item) == 0) {
      *pindex = index + 1;
      index = (*pindex);
      if (var && index < var.DataMemberSize()) return(1);
      else {
         *phandle = 0;
         return(0);
      }
   }

   /*******************************************************************
    * name
    *******************************************************************/
   if (strcmp("name", item) == 0) {
      return (long) CopyInTempObject(var.DataMemberAt(index).Name());
   }

   /*******************************************************************
    * type
    *******************************************************************/
   if (strcmp("type", item) == 0) {
      ::Reflex::Member m(var.DataMemberAt(index));
      return (long) CopyInTempObject(m.TypeOf().Name(Reflex::SCOPED));
   }

   /*******************************************************************
    * offset
    *******************************************************************/
   if (strcmp("offset", item) == 0) {
      ::Reflex::Member m(var.DataMemberAt(index));
      return (long)G__get_offset(m);
   }

   /*******************************************************************
    * title
    *******************************************************************/
   if (strcmp("title", item) == 0) {
      if (-1 != tagnum) {
         tag_string_buf = G__defined_tagname("G__string_buf", 0);
         G__alloc_tempobject(tag_string_buf, -1);
         buf = (char*)G__p_tempbuf->obj.obj.i;
         ::Reflex::Member m(var.DataMemberAt(index));
         G__getcomment(buf, &G__get_properties(m)->comment, tagnum);
         return((long)buf);
      }
      else {
         G__genericerror("Error: title only supported for class/struct member");
         return((long)0);
      }
   }
   return(0);
}

//______________________________________________________________________________
long Cint::Internal::G__get_functioninfo(char* item, long* phandle, long* pindex, int tagnum)
{
   char *buf;
   int tag_string_buf;
   /* char temp[G__MAXNAME]; */
   ::Reflex::Scope ifunc;
   unsigned int index;
   int p;

   /*******************************************************************
   * new
   *******************************************************************/
   if (strcmp("new", item) == 0) {
      *pindex = 0;
      if (-1 == tagnum) {
         *phandle = (long)(::Reflex::Scope::GlobalScope().Id());
      }
      else if ((G__Dict::GetDict().GetScope(tagnum))) {
         G__incsetup_memfunc(tagnum);
         *phandle = (long)(G__Dict::GetDict().GetScope(tagnum).Id());
      }
      else {
         *phandle = 0;
      }
      return(0);
   }

   ifunc = G__Dict::GetDict().GetScope(*phandle);
   index = (*pindex);

   if (ifunc || ifunc.FunctionMemberSize() <= index) {
      *phandle = 0;
      *pindex = 0;
      return(0);
   }


   /*******************************************************************
   * next
   *******************************************************************/
   if (strcmp("next", item) == 0) {
      *pindex = index + 1;
      if (ifunc && index < ifunc.FunctionMemberSize()) return(1);
      else {
         *phandle = 0;
         return(0);
      }
   }

   /*******************************************************************
   * name
   *******************************************************************/
   if (strcmp("name", item) == 0) {
      return (long) CopyInTempObject(ifunc.FunctionMemberAt(index).Name());
   }

   /*******************************************************************
   * type
   *******************************************************************/
   if (strcmp("type", item) == 0) {
      ::Reflex::Member func(ifunc.FunctionMemberAt(index));
      return (long) CopyInTempObject(func.Name(::Reflex::SCOPED));
   }

   /*******************************************************************
    * arglist
    *******************************************************************/
   if (strcmp("arglist", item) == 0) {
      tag_string_buf = G__defined_tagname("G__string_buf", 0);
      G__alloc_tempobject(tag_string_buf, -1);
      buf = (char*)G__p_tempbuf->obj.obj.i;

      buf[0] = '\0';
      p = 0;
      ::Reflex::Member func(G__Dict::GetDict().GetFunction((G__ifunc_table*)phandle, index));
      int i = 0;
      for (::Reflex::Type_Iterator iter(func.TypeOf().FunctionParameter_Begin());
            iter != func.TypeOf().FunctionParameter_End();
            ++iter, ++i) {

         if (p) {
            sprintf(buf + p, ",");
            ++p;
         }
         sprintf(buf + p, "%s", iter->Name(::Reflex::SCOPED).c_str());
         p = strlen(buf);
         if (func.FunctionParameterDefaultAt(i).length()) {
            sprintf(buf + p, "=%s", func.FunctionParameterDefaultAt(i).c_str());
         }
         p = strlen(buf);
      }
      return((long)buf);
   }

   /*******************************************************************
    * title
    *******************************************************************/
   if (strcmp("title", item) == 0) {
      if (-1 != tagnum) {
         tag_string_buf = G__defined_tagname("G__string_buf", 0);
         G__alloc_tempobject(tag_string_buf, -1);
         buf = (char*)G__p_tempbuf->obj.obj.i;

         ::Reflex::Member func(G__Dict::GetDict().GetFunction((G__ifunc_table*)phandle, index));
         G__getcomment(buf, &G__get_funcproperties(func)->comment, tagnum);
         return((long)buf);
      }
      else {
         G__genericerror("Error: title only supported for class/struct member");
         return((long)0);
      }
   }
   return(0);
}

//______________________________________________________________________________
#ifdef G__VAARG_INC_COPY_N
static int G__va_arg_align_size = G__VAARG_INC_COPY_N;
#else // G__VAARG_INC_COPY_N
static int G__va_arg_align_size = 0;
#endif // G__VAARG_INC_COPY_N

//______________________________________________________________________________
void G__va_arg_setalign(int n)
{
   G__va_arg_align_size = n;
}

//______________________________________________________________________________
extern "C" void G__va_arg_copyvalue(int t, void *p, G__value *pval, int objsize)
{
   // --
#ifdef G__VAARG_PASS_BY_REFERENCE
   if (objsize > G__VAARG_PASS_BY_REFERENCE) {
      // would need to use G__getproperties(var)->isBytecodeArena
      if (pval->ref > 0x1000) *(long*)(p) = pval->ref;
      else *(long*)(p) = (long)G__int(*pval);
      return;
   }

#endif
   switch (t) {
      case 'n':
      case 'm':
         *(G__int64*)(p) = (G__int64)G__Longlong(*pval);
         break;
      case 'g':
#ifdef G__BOOL4BYTE
         *(int*)(p) = (int)G__int(*pval);
         break;
#endif
      case 'c':
      case 'b':
#if defined(__GNUC__) || defined(G__WIN32)
         *(int*)(p) = (int)G__int(*pval);
#else
         *(char*)(p) = (char)G__int(*pval);
#endif
         break;
      case 'r':
      case 's':
#if defined(__GNUC__) || defined(G__WIN32)
         *(int*)(p) = (int)G__int(*pval);
#else
         *(short*)(p) = (short)G__int(*pval);
#endif
         break;
      case 'h':
      case 'i':
         *(int*)(p) = (int)G__int(*pval);
         break;
      case 'k':
      case 'l':
         *(long*)(p) = (long)G__int(*pval);
         break;
      case 'f':
#define G__OLDIMPLEMENTATION2235
#if defined(__GNUC__) || defined(G__WIN32)
         *(double*)(p) = (double)G__double(*pval);
#else
         *(float*)(p) = (float)G__double(*pval);
#endif
         break;
      case 'd':
         *(double*)(p) = (double)G__double(*pval);
         break;
      case 'u':
         memcpy((void*)(p), (void*)pval->obj.i, objsize);
         break;
      default:
         *(long*)(p) = (long)G__int(*pval);
         break;
   }
}

//______________________________________________________________________________
#if (defined(__PPC__)||defined(__ppc__))&&(defined(_AIX)||defined(__APPLE__))
#define G__alignof_ppc(objsize)  (objsize>4?16:4)
#define G__va_rounded_size_ppc(typesize) ((typesize + 3) & ~3)
#define G__va_align_ppc(AP, objsize)                                           \
   ((((unsigned long)(AP)) + ((G__alignof_ppc(objsize) == 16) ? 15 : 3)) \
    & ~((G__alignof_ppc(objsize) == 16) ? 15 : 3))

#elif (defined(__PPC__)||defined(__ppc__))&&(defined(__linux)||defined(__linux__))

#endif

//______________________________________________________________________________
extern "C" void G__va_arg_put(G__va_arg_buf* pbuf, G__param* libp, int n)
{
   int objsize;
   int type;
   int i;
#if defined(__hpux) || defined(__hppa__)
   int j2 = G__VAARG_SIZE;
#endif
   int j = 0;
   int mod;
#ifdef G__VAARG_NOSUPPORT
   G__genericerror("Limitation: Variable argument is not supported for this platform");
#endif
   for (i = n;i < libp->paran;i++) {
      type = G__get_type(G__value_typenum(libp->para[i]));
      if (isupper(type)) objsize = G__LONGALLOC;
      else              objsize = G__sizeof(&libp->para[i]);
#if defined(__GNUC__) || defined(G__WIN32)
      switch (type) {
         case 'c':
         case 'b':
         case 's':
         case 'r':
            objsize = sizeof(int);
            break;
         case 'f':
            objsize = sizeof(double);
            break;
      }
#endif

      /* Platform that decrements address */
#if (defined(__linux)&&defined(__i386))||defined(_WIN32)||defined(G__CYGWIN)
      /* nothing */
#elif defined(__hpux) || defined(__hppa__)
      if (objsize > G__VAARG_PASS_BY_REFERENCE) {
         j2 = j2 - sizeof(long);
         j = j2;
      }
      else {
         j2 = (j2 - objsize) & (objsize > 4 ? 0xfffffff8 : 0xfffffffc);
         j = j2 + ((8 - objsize) % 4);
      }
#elif defined(__sparc) || defined(__sparc__) || defined(__SUNPRO_C) || \
      defined(__SUNPRO_CC)
      /* nothing */
#elif (defined(__PPC__)||defined(__ppc__))&&(defined(_AIX)||defined(__APPLE__))
      /* nothing */
#elif (defined(__PPC__)||defined(__ppc__))&&(defined(__linux)||defined(__linux__))
      /* nothing */
#elif defined(__x86_64__) && defined(__linux)
      /* nothing */
#else
      /* nothing */
#endif

      G__va_arg_copyvalue(type, (void*)(&pbuf->x.d[j]), &libp->para[i], objsize);

      /* Platform that increments address */
#if (defined(__linux)&&defined(__i386))||defined(_WIN32)||defined(G__CYGWIN)
      j += objsize;
      mod = j % G__va_arg_align_size;
      if (mod) j = j - mod + G__va_arg_align_size;
#elif defined(__hpux) || defined(__hppa__)
      /* nothing */
#elif defined(__sparc) || defined(__sparc__) || defined(__SUNPRO_C) || \
      defined(__SUNPRO_CC)
      j += objsize;
      mod = j % G__va_arg_align_size;
      if (mod) j = j - mod + G__va_arg_align_size;
#elif (defined(__PPC__)||defined(__ppc__))&&(defined(_AIX)||defined(__APPLE__))
      //j =  G__va_align_ppc(j, objsize) + G__va_rounded_size_ppc(objsize);
#ifdef G__VAARG_PASS_BY_REFERENCE
      if (objsize > G__VAARG_PASS_BY_REFERENCE) objsize = G__VAARG_PASS_BY_REFERENCE;
#endif
      j += objsize;
      mod = j % G__va_arg_align_size;
      if (mod) j = j - mod + G__va_arg_align_size;
#elif (defined(__PPC__)||defined(__ppc__))&&(defined(__linux)||defined(__linux__))
#ifdef G__VAARG_PASS_BY_REFERENCE
      if (objsize > G__VAARG_PASS_BY_REFERENCE) objsize = G__VAARG_PASS_BY_REFERENCE;
#endif
      j += objsize;
      mod = j % G__va_arg_align_size;
      if (mod) j = j - mod + G__va_arg_align_size;
#elif defined(__x86_64__) && defined(__linux)
#ifdef G__VAARG_PASS_BY_REFERENCE
      if (objsize > G__VAARG_PASS_BY_REFERENCE) objsize = G__VAARG_PASS_BY_REFERENCE;
#endif
      j += objsize;
      mod = j % G__va_arg_align_size;
      if (mod) j = j - mod + G__va_arg_align_size;
#else
#ifdef G__VAARG_PASS_BY_REFERENCE
      if (objsize > G__VAARG_PASS_BY_REFERENCE) objsize = G__VAARG_PASS_BY_REFERENCE;
#endif
      j += objsize;
      mod = j % G__va_arg_align_size;
      if (mod) j = j - mod + G__va_arg_align_size;
#endif

   }
}

//______________________________________________________________________________
#ifdef G__VAARG_COPYFUNC
void Cint::Internal::G__va_arg_copyfunc(FILE* fp, G__ifunc_table* ifunc, int ifn)
{
   FILE *xfp;
   int n;
   int c;
   int nest = 0;
   int double_quote = 0;
   int single_quote = 0;
   int flag = 0;

   if (G__srcfile[ifunc->pentry[ifn]->filenum].fp)
      xfp = G__srcfile[ifunc->pentry[ifn]->filenum].fp;
   else {
      xfp = fopen(G__srcfile[ifunc->pentry[ifn]->filenum].filename, "r");
      flag = 1;
   }
   if (!xfp) return;
   fsetpos(xfp, &ifunc->pentry[ifn]->pos);

   fprintf(fp, "%s ", G__type2string(ifunc->type[ifn]
                                     , ifunc->p_tagtable[ifn]
                                     , ifunc->p_typetable[ifn]
                                     , ifunc->reftype[ifn]
                                     , ifunc->isconst[ifn]));
   fprintf(fp, "%s(", ifunc->funcname[ifn]);

   /* print out parameter types */
   for (n = 0;n < ifunc->para_nu[ifn];n++) {

      if (n != 0) {
         fprintf(fp, ",");
      }

      if ('u' == ifunc->para_type[ifn][n] &&
            0 == strcmp(G__struct.name[ifunc->para_p_tagtable[ifn][n]], "va_list")) {
         fprintf(fp, "struct G__param* G__VA_libp,int G__VA_n");
         break;
      }
      /* print out type of return value */
      fprintf(fp, "%s", G__type2string(ifunc->para_type[ifn][n]
                                       , ifunc->para_p_tagtable[ifn][n]
                                       , ifunc->para_p_typetable[ifn][n]
                                       , ifunc->para_reftype[ifn][n]
                                       , ifunc->para_isconst[ifn][n]));

      if (ifunc->para_name[ifn][n]) {
         fprintf(fp, " %s", ifunc->para_name[ifn][n]);
      }
      if (ifunc->para_def[ifn][n]) {
         fprintf(fp, "=%s", ifunc->para_def[ifn][n]);
      }
   }
   fprintf(fp, ")");
   if (ifunc->isconst[ifn]&G__CONSTFUNC) {
      fprintf(fp, " const");
   }

   c = 0;
   while (c != '{') c = fgetc(xfp);
   fprintf(fp, "{");

   nest = 1;
   while (c != '}' || nest) {
      c = fgetc(xfp);
      fputc(c, fp);
      switch (c) {
         case '"':
            if (!single_quote) double_quote ^= 1;
            break;
         case '\'':
            if (!double_quote) single_quote ^= 1;
            break;
         case '{':
            if (!single_quote && !double_quote) ++nest;
            break;
         case '}':
            if (!single_quote && !double_quote) --nest;
            break;
      }
   }
   fprintf(fp, "\n");
   if (flag && xfp) fclose(xfp);
}
#endif

//______________________________________________________________________________
static void G__typeconversion(const ::Reflex::Member& ifunc, G__param* libp)
{
   for (unsigned int i = 0;i < ((unsigned int)libp->paran) && i < ifunc.TypeOf().FunctionParameterSize();++i) {
      ::Reflex::Type formal_type(ifunc.TypeOf().FunctionParameterAt(i));
      ::Reflex::Type param_type(G__value_typenum(libp->para[i]));

      switch (G__get_type(formal_type)) {
         case 'd':
         case 'f':
            switch (G__get_type(param_type)) {
               case 'c':
               case 's':
               case 'i':
               case 'l':
               case 'b':
               case 'r':
               case 'h':
               case 'k':
                  libp->para[i].obj.d = libp->para[i].obj.i;
                  G__value_typenum(libp->para[i]) = formal_type;
                  libp->para[i].ref = (long)(&libp->para[i].obj.d);
                  break;
            }
            break;
         case 'c':
         case 's':
         case 'i':
         case 'l':
         case 'b':
         case 'r':
         case 'h':
         case 'k':
            switch (G__get_type(param_type)) {
               case 'd':
               case 'f':
                  libp->para[i].obj.i = (long)libp->para[i].obj.d;
                  G__value_typenum(libp->para[i]) = formal_type;
                  libp->para[i].ref = (long)(&libp->para[i].obj.i);
                  break;
            }
            break;
      }
   }
}

//______________________________________________________________________________
int Cint::Internal::G__DLL_direct_globalfunc(G__value* result7, G__CONST char* funcname, struct G__param* libp, int hash)
{
   ::Reflex::Member ifunc(G__Dict::GetDict().GetFunction((G__ifunc_table*)funcname, hash));
   int (*itp2f)(G__va_arg_buf);
   double(*dtp2f)(G__va_arg_buf);
   void (*vtp2f)(G__va_arg_buf);
   G__va_arg_buf(*utp2f)(G__va_arg_buf);
   G__va_arg_buf G__va_arg_return;
   G__va_arg_buf G__va_arg_bufobj;
   G__typeconversion(ifunc, libp);
   G__va_arg_put(&G__va_arg_bufobj, libp, 0);
   ::Reflex::Type returnType(ifunc.TypeOf().ReturnType());
   switch (G__get_type(returnType)) {
      case 'd':
      case 'f':
         dtp2f = (double(*)(G__va_arg_buf))G__get_funcproperties(ifunc)->entry.tp2f;
         G__letdouble(result7, G__get_type(returnType), dtp2f(G__va_arg_bufobj));
         G__value_typenum(*result7) = returnType;
         break;
      case 'u':
         utp2f = (G__va_arg_buf(*)(G__va_arg_buf))G__get_funcproperties(ifunc)->entry.tp2f;
         G__va_arg_return = utp2f(G__va_arg_bufobj);
         G__value_typenum(*result7) = returnType;
         result7->obj.i = (long)(&G__va_arg_return); /* incorrect! experimental */
         break;
      case 'y':
         vtp2f = (void (*)(G__va_arg_buf))G__get_funcproperties(ifunc)->entry.tp2f;
         vtp2f(G__va_arg_bufobj);
         G__setnull(result7);
         break;
      default:
         itp2f = (int(*)(G__va_arg_buf))G__get_funcproperties(ifunc)->entry.tp2f;
         G__letint(result7, G__get_type(returnType), itp2f(G__va_arg_bufobj));
         G__value_typenum(*result7) = returnType;
         break;
   }
   return 1;
}

/*
 * Local Variables:
 * c-tab-always-indent:nil
 * c-indent-level:3
 * c-continued-statement-offset:3
 * c-brace-offset:-3
 * c-brace-imaginary-offset:0
 * c-argdecl-indent:0
 * c-label-offset:-3
 * compile-command:"make -k"
 * End:
 */
