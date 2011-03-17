/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file typedef.c
 ************************************************************************
 * Description:
 *  typedef handling
 ************************************************************************
 * Copyright(c) 1995~2003  Masaharu Goto
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "common.h"
#include "DataMemberHandle.h"

extern "C" {

static int G__static_parent_tagnum = -1;
static int G__static_isconst = 0;

// Static functions.
static void G__shiftstring(char* s, int n);
static int G__defined_typename_exact(char* type_name);
static int G__make_uniqueP2Ftypedef(char* type_name);

// Internal functions.
void G__define_type();
int G__defined_typename_noerror(const char* type_name, int noerror);

// Functions in the C interface.
int G__defined_typename(const char* type_name);
int G__search_typename(const char* typenamein, int typein, int tagnum, int reftype);
int G__search_typename2(const char* type_name, int typein, int tagnum, int reftype, int parent_tagnum);

//______________________________________________________________________________
//
//  Static functions.
//

//______________________________________________________________________________
static void G__shiftstring(char* s, int n)
{
   int i = 0, j = n;
   while (s[j]) s[i++] = s[j++];
   s[i] = 0;
}

//______________________________________________________________________________
static int G__defined_typename_exact(char* type_name)
{
   // Search already defined typedef names, -1 is returned if not found
   int i, flag = 0;
   char ispointer = 0;
   int len = strlen(type_name) + 1;
   G__FastAllocString temp(len);
   char *p;
   G__FastAllocString temp2(type_name);
   int env_tagnum;
   char *par;

   /* find 'xxx::yyy' */
   p = (char*)G__find_last_scope_operator(temp2);

   /* abandon scope operator if 'zzz (xxx::yyy)www' */
   par = strchr(temp2, '(');
   if (par && p && par < p) p = (char*)NULL;

   if (p) {
      temp = p + 2;
      *p = '\0';
      if (temp2 == p) env_tagnum = -1; /* global scope */
#ifndef G__STD_NAMESPACE
      else if (strcmp(temp2, "std") == 0
               && G__ignore_stdnamespace
              ) env_tagnum = -1;
#endif
      else {
         // first try a typedef, so we don't trigger autoloading here:
         long env_typenum = G__defined_typename_noerror(temp2, 1);
         if (env_typenum != -1 && G__newtype.type[env_typenum] == 'u')
            env_tagnum = G__newtype.tagnum[env_typenum];
         else
            env_tagnum = G__defined_tagname(temp2, 0);
      }
   }
   else {
      temp = temp2;
      env_tagnum = G__get_envtagnum();
   }

   len = strlen(temp);

   if (temp[len-1] == '*') {
      temp[--len] = '\0';
      ispointer = 'A' - 'a';
   }

   NameMap::Range nameRange = G__newtype.namerange->Find(temp);
   if (nameRange) {
      for (i = nameRange.First();i <= nameRange.Last(); i++) {
         if (len == G__newtype.hash[i] && strcmp(G__newtype.name[i], temp) == 0 && (
                  env_tagnum == G__newtype.parent_tagnum[i]
               )) {
            flag = 1;
            /* This must be a bad manner. Somebody needs to reset G__var_type
             * especially when typein is 0. */
            G__var_type = G__newtype.type[i] + ispointer ;
            break;
         }
      }
   } else {
      i = G__newtype.alltype;
   }

   if (flag == 0) return(-1);
   return(i);
}

//______________________________________________________________________________
static int G__make_uniqueP2Ftypedef(char* type_name)
{
   //  input: 'void* (*)(int , void * , short )'
   // output: 'void* (*)(int,void*,short)'
   char *from;
   char *to;
   int spacecnt = 0;
   int isstart = 1;

   /*  input  'void* (*)(int , void * , short )'
    *         ^ start                         */
   from = strchr(type_name, '(');
   if (!from) return(1);
   from = strchr(from + 1, '(');
   if (!from) return(1);
   ++from;
   to = from;
   /*  input  'void* (*)(int , void * , short )'
    *                    ^ got this position  */

   while (*from) {
      if (isspace(*from)) {
         if (0 == spacecnt && 0 == isstart) {
            /*  input  'void* (*)(int   * , void  * , short )'
             *                       ^ here  */
            *(to++) = ' ';
         }
         else {
            /*  input  'void* (*)(int   * , void  * , short )'
             *                        ^^ here  */
            /* Ignore consequitive space */
         }
         if (0 == isstart) ++spacecnt;
         else           spacecnt = 0;
         isstart = 0;
      }
      else {
         isstart = 0;
         if (spacecnt) {
            switch (*from) {
               case ',':
                  isstart = 1;
                  *(to - 1) = *from;
                  break;
               case ')':
               case '*':
               case '&':
                  /*  input  'void* (*)(int   * , void  * , short )'
                   *                          ^ here
                   *  output 'void* (*)(int*
                   *                       ^ put here */
                  *(to - 1) = *from;
                  break;
               default:
                  /*  input  'void* (*)(unsigned  int   * , void  * , short )'
                   *                              ^ here
                   *  output 'void* (*)(unsigned i
                   *                             ^ put here */
                  *(to++) = *from;
                  break;
            }
         }
         else {
            /*  input  'void* (*)(unsigned  int   * , void  * , short )'
             *                      ^ here   */
            *(to++) = *from;
         }
         spacecnt = 0;
      }
      ++from;
   }

   *to = 0;

   /* int (*)(void) to int (*)() */
   from = strchr(type_name, '(');
   if (!from) return(1);
   from = strchr(from + 1, '(');
   if (!from) return(1);
   if (strcmp(from, "(void)") == 0) {
      *(++from) = ')';
      *(++from) = 0;
   }

   return(0);
}

//______________________________________________________________________________
//
//  Internal Functions.
//

//______________________________________________________________________________
void G__define_type()
{
   // typedef [struct|union|enum] tagname { member } newtype;
   //        ^
   // typedef fundamentaltype   newtype;
   //        ^
   fpos_t rewind_fpos;
   int c;
   G__FastAllocString type1(G__LONGLINE);
   G__FastAllocString tagname(G__LONGLINE);
   G__FastAllocString type_name(G__LONGLINE);
   G__FastAllocString temp(G__LONGLINE);
   int isnext;
   fpos_t next_fpos;
   int store_tagnum;
   int store_def_struct_member = 0;
   struct G__var_array* store_local;
   G__FastAllocString memname(G__MAXNAME);
   G__FastAllocString val(G__ONELINE);
   char type;
   char tagtype = 0;
   int unsigned_flag = 0;
   int mem_def = 0;
   int temp_line;
   int len;
   int taglen;
   G__value enumval;
   int store_tagdefining;
   int typedef2 = 0;
   int itemp;
   int nindex = 0;
   int index[G__MAXVARDIM];
   G__FastAllocString aryindex(G__MAXNAME);
   char* p;
   int store_var_type;
   int typenum;
   int isorgtypepointer = 0;
   int store_def_tagnum;
   int reftype = G__PARANORMAL;
   int rawunsigned = 0;
   int env_tagnum;
   int isconst = 0;
   fpos_t pos_p2fcomment;
   int line_p2fcomment;
   int flag_p2f = 0;
   tagname[0] = '\0';
   fgetpos(G__ifile.fp, &pos_p2fcomment);
   line_p2fcomment = G__ifile.line_number;
#ifdef G__ASM
#ifdef G__ASM_DBG
   if (G__asm_dbg && G__asm_noverflow) {
      G__genericerror(G__LOOPCOMPILEABORT);
   }
#endif // G__ASM_DBG
   G__abortbytecode();
#endif // G__ASM
   store_tagnum = G__tagnum;
   store_tagdefining = G__tagdefining;
   store_def_tagnum = G__def_tagnum;

   /*
    *  typedef  [struct|union|enum] tagname { member } newtype;
    *  typedef  fundamentaltype   newtype;
    *          ^
    * read type
    */

   c = G__fgetname_template(type1, 0, "*{");
   if (c == '*') {
      type1 += "*";
      c = ' ';
   }
   // Consume any const, volatile, mutable, or typename qualifier. // FIXME: mutable is illegal in a typedef.
   while (
      isspace(c) &&
      (
         !strcmp(type1, "const") ||
         !strcmp(type1, "volatile") ||
         !strcmp(type1, "mutable") ||
         !strcmp(type1, "typename")
      )
   ) {
      if (!strcmp(type1, "const")) {
         isconst |= G__CONSTVAR;
      }
      c = G__fgetname_template(type1, 0, "{");
   }
   if (!strcmp(type1, "::")) {  // FIXME: This makes no sense, there cannot be typedef ::{...};
      // skip a :: without a namespace in front of it (i.e. global namespace!)
      c = G__fgetspace(); // skip the next ':'
      c = G__fgetname_template(type1, 0, "{");
   }
   if (!strncmp(type1, "::", 2)) { // Strip a leading :: (global namespace operator)
      // A leading '::' causes other typename matching functions to fail so
      // we remove it. This is not the ideal solution (neither was the one
      // above since it does not allow for distinction between global
      // namespace and local namespace) ... but at least it is an improvement
      // over the current behavior.
      strcpy((char*)type1, type1 + 2);  // Okay since we reduce the size ...
   }
   while (isspace(c)) {
      len = strlen(type1);
      c = G__fgetspace();
      if (c == ':') {
         c = G__fgetspace(); // skip the next ':'
         type1 += "::";
         c = G__fgetname_template(temp, 0, "{");
         type1 += temp;
      }
      else if ((c == '<') || (c == ',') || (type1[len-1] == '<') || (type1[len-1] == ',')) {
         type1[len++] = c;
         do { // ignore white space inside template
            // humm .. thoes this translate correctly nested templates?
            c = G__fgetstream_template(type1, len, ">");
            len = strlen(type1);
         }
         while (isspace(c));
         type1[len++] = c;
         type1[len] = '\0';
      }
      else if (c == '>') {
         type1[len++] = c;
         type1[len] = '\0';
      }
      else {
         c = ' ';
         fseek(G__ifile.fp, -1, SEEK_CUR);
         if (G__dispsource) {
            G__disp_mask = 1;
         }
         break;
      }
   }

   /*
    *  typedef unsigned  int  newtype ;
    *                   ^
    * read type
    */

   if (!strcmp(type1, "unsigned")) {
      unsigned_flag = 1;
      c = G__fgetname(type1, 0, "");
   }
   else if (!strcmp(type1, "signed")) {
      unsigned_flag = 0;
      c = G__fgetname(type1, 0, "");
   }
   else if (!strcmp(type1, "unsigned*")) {
      unsigned_flag = 1;
      type1 = "int*";
   }
   else if (!strcmp(type1, "signed*")) {
      unsigned_flag = 0;
      type1 = "int*";
   }
   else if (!strcmp(type1, "unsigned&")) {
      unsigned_flag = 1;
      type1 = "int&";
   }
   else if (!strcmp(type1, "signed&")) {
      unsigned_flag = 0;
      type1 = "int&";
   }
   else if (!strcmp(type1, "unsigned*&")) {
      unsigned_flag = 1;
      type1 = "int*&";
   }
   else if (!strcmp(type1, "signed*&")) {
      unsigned_flag = 0;
      type1 = "int*&";
   }

   /*
    *  typedef  [struct|union|enum]  tagname { member } newtype;
    *                               ^
    *  typedef  fundamentaltype   newtype;
    *                           ^
    *  typedef unsigned  int  newtype ;
    *                        ^
    */

   if (type1[0] && (type1[strlen(type1)-1] == '&')) {
      reftype = G__PARAREFERENCE;
      type1[strlen(type1)-1] = '\0';
   }
   if (type1[0] && (type1[strlen(type1)-1] == '*')) {
      isorgtypepointer = 'A' -'a';
      type1[strlen(type1)-1] = '\0';
      while (type1[0] && (type1[strlen(type1)-1] == '*')) {
         if (G__PARANORMAL == reftype) {
            reftype = G__PARAP2P;
         }
         else if (reftype >= G__PARAP2P) {
            ++reftype;
         }
         type1[strlen(type1)-1] = '\0';
      }
   }

   if (strcmp(type1, "char") == 0) {
      if (unsigned_flag == 0) type = 'c';
      else                 type = 'b';
   }
   else if (strcmp(type1, "short") == 0) {
      if (unsigned_flag == 0) type = 's';
      else                 type = 'r';
   }
   else if (strcmp(type1, "int") == 0) {
      if (unsigned_flag == 0) type = 'i';
      else                 type = 'h';
   }
   else if (strcmp(type1, "long") == 0) {
      if (unsigned_flag == 0) type = 'l';
      else                 type = 'k';
   }
   else if (strcmp(type1, "bool") == 0) {
      type = 'g';
   }
   else if (strcmp(type1, "void") == 0) {
      type = 'y';
   }
   else if (strcmp(type1, "float") == 0) {
      type = 'f';
   }
   else if (strcmp(type1, "double") == 0) {
      type = 'd';
   }
   else if (strcmp(type1, "FILE") == 0) {
      type = 'e';
   }
   else if ((strcmp(type1, "struct") == 0) || (strcmp(type1, "union") == 0) ||
            (strcmp(type1, "enum") == 0) || (strcmp(type1, "class") == 0)) {
      type = 'u';
      if (strcmp(type1, "struct") == 0) tagtype = 's';
      if (strcmp(type1, "class") == 0) tagtype = 'c';
      if (strcmp(type1, "union") == 0) tagtype = 'u';
      if (strcmp(type1, "enum") == 0) tagtype = 'e';
      tagname[0] = 0;

      /*  typedef [struct|union|enum]{ member } newtype;
       *                              ^ */

      /*  typedef [struct|union|enum]  tagname { member } newtype;
       *  typedef [struct|union|enum]  tagname  newtype;
       *  typedef [struct|union|enum]  { member } newtype;
       *                              ^
       *  read tagname
       */
      if (c != '{') {
         c = G__fgetname(tagname, 0, "{");
      }


      /*
       *  typedef [struct|union|enum]{ member } newtype;
       *                              ^
       *  typedef [struct|union|enum] tagname  { member } newtype;
       *                                      ^
       *  typedef [struct|union|enum] tagname{ member } newtype;
       *                                      ^
       *  typedef [struct|union|enum]              { member } newtype;
       *                                     ^
       *  typedef [struct|union|enum] tagname  newtype;
       *                                      ^            */
      if (c != '{') {
         c = G__fgetspace();
         /* typedef [struct] tag   { member } newtype;
          *                         ^
          * typedef [struct|union|enum] tagname  newtype;
          *                                       ^     */
         if (c != '{') {
            fseek(G__ifile.fp, -1, SEEK_CUR);
            if (G__dispsource) G__disp_mask = 1;
         }
      }

      /*  typedef [struct|union|enum]{ member } newtype;
       *                              ^
       *  typedef [struct|union|enum] tagname  { member } newtype;
       *                                        ^
       *  typedef [struct|union|enum] tagname{ member } newtype;
       *                                      ^
       *  typedef [struct|union|enum]              { member } newtype;
       *                                     ^
       *  typedef [struct|union|enum] tagname  newtype;
       *                                       ^
       *  skip member declaration if exists */
      if (c == '{') {
         mem_def = 1;
         fseek(G__ifile.fp, -1, SEEK_CUR);
         fgetpos(G__ifile.fp, &rewind_fpos);
         if (G__dispsource) G__disp_mask = 1;
         G__fgetc();
         G__fignorestream("}");
      }
   }
   else if (unsigned_flag) {
      len = strlen(type1);
      if (';' == type1[len-1]) {
         c = ';';
         type1[len-1] = '\0';
      }
      type = 'h';
      rawunsigned = 1;
   }
   else {
      itemp = G__defined_typename(type1);
      if (itemp != -1) {
         type = G__newtype.type[itemp];
         switch (reftype) {
            case G__PARANORMAL:
               reftype = G__newtype.reftype[itemp];
               break;
            case G__PARAREFERENCE:
               switch (G__newtype.reftype[itemp]) {
                  case G__PARANORMAL:
                  case G__PARAREFERENCE:
                     break;
                  default:
                     if (G__newtype.reftype[itemp] < G__PARAREF) {
                        reftype = G__newtype.reftype[itemp] + G__PARAREF;
                     }
                     else {
                        reftype = G__newtype.reftype[itemp];
                     }
                     break;
               }
               break;
            default:
               switch (G__newtype.reftype[itemp]) {
                  case G__PARANORMAL:
                     break;
                  case G__PARAREFERENCE:
                     G__fprinterr(G__serr, "Limitation: reference or pointer type not handled properly (2)");
                     G__printlinenum();
                     break;
                  default:
                     break;
               }
               break;
         }
         itemp = G__newtype.tagnum[itemp];
      }
      else {
         type = 'u';
         itemp = G__defined_tagname(type1, 0);
      }
      if (itemp != -1) {
         tagtype = G__struct.type[itemp];
#ifndef G__OLDIMPLEMENTATION1503
         if (G__struct.parent_tagnum[itemp] != -1) {
            tagname = G__fulltagname(G__struct.parent_tagnum[itemp], 0);
            tagname += "::";
            tagname += G__struct.name[itemp];
         }
         else {
            tagname = G__struct.name[itemp];
         }
#else // G__OLDIMPLEMENTATION1503
         tagname = G__fulltagname(itemp, 0);
#endif // G__OLDIMPLEMENTATION1503
         ++G__struct.istypedefed[itemp];
      }
      else {
         tagtype = 0;
         tagname[0] = 0;
      }
      typedef2 = 1;
   }

   if (isorgtypepointer) {
      type = toupper(type);
   }

   /*
    *  typedef [struct|union|enum] tagname { member } newtype ;
    *                                                ^^
    * skip member declaration if exists
    */

   if (rawunsigned) {
      type_name = type1;
   }
   else {
      c = G__fgetname_template(type_name, 0, ";,[");
   }

   if (
      !strncmp(type_name, "long", 4) &&
      (
         (strlen(type_name) == 4) ||
         (
            (strlen(type_name) >= 5) &&
            (
               (type_name[4] == '&') ||
               (type_name[4] == '*')
            )
         )
      )
   ) {
      // int tmptypenum;
      if (strlen(type_name) >= 5) {
         // Rewind.
         size_t copylen = strlen(type_name) - strlen("long") + 1;
         strncpy(type_name,type_name + strlen("long"),copylen);
      } else {
         c = G__fgetname(type_name, 0, ";,[");
      }
      tagname = "";
      if (type == 'l') {
         type = 'n';
      }
      else if (type == 'k') {
         type = 'm';
      }
   }
   if (
      !strncmp(type_name, "double", strlen("double")) &&
      (
         (strlen(type_name) == strlen("double")) ||
         (
            (strlen(type_name) > strlen("double")) &&
            (
               (type_name[strlen("double")] == '&') ||
               (type_name[strlen("double")] == '*')
            )
         )
      )
   ) {
      static const size_t lendouble = strlen("double");
      if (strlen(type_name) > lendouble) {
         // Rewind.
         size_t copylen = strlen(type_name) - lendouble + 1;
         strncpy(type_name,type_name + lendouble,copylen);
      } else {
         c = G__fgetname(type_name, 0, ";,[");
      }
      if (type == 'l') {
         // int tmptypenum;
         type = 'q';
         tagname = "";
      }
   }

   /* in case of
    *  typedef unsigned long int  int32;
    *                           ^
    *  read type_name
    */
   if (
      !strncmp(type_name, "int", 3) && // we have at least int, and
      (
         (strlen(type_name) == 3) || // we have exactly int, or
         (
            (strlen(type_name) >= 4) && // we have more than int, and
            (
               (type_name[3] == '&') || // we have at least int&, or
               (type_name[3] == '*') // we have at least int*
            )
         )
      )
   ) {
      static const size_t lenint = strlen("int");
      if (strlen(type_name) > lenint) { // we have at least int& or int*
         // Rewind.
         size_t copylen = strlen(type_name) - lenint + 1;
         strncpy(type_name,type_name + lenint,copylen);
      } else {
         c = G__fgetstream(type_name, 0, ";,[");
      }
   }
   if (!strcmp(type_name, "*")) {
      fpos_t tmppos;
      fgetpos(G__ifile.fp, &tmppos);
      int tmpline = G__ifile.line_number;
      c = G__fgetname(type_name, 1, ";,[");
      if (isspace(c) && !strcmp(type_name, "*const")) {
         isconst |= G__PCONSTVAR;
         c = G__fgetstream(type_name, 1, ";,[");
      }
      else {
         G__disp_mask = strlen(type_name) - 1;
         G__ifile.line_number = tmpline;
         fsetpos(G__ifile.fp, &tmppos);
         c = G__fgetstream(type_name, 1, ";,[");
      }
   }
   else if (!strcmp(type_name, "**")) {
      fpos_t tmppos;
      fgetpos(G__ifile.fp, &tmppos);
      int tmpline = G__ifile.line_number;
      c = G__fgetname(type_name, 1, ";,[");
      if (isspace(c) && !strcmp(type_name, "*const")) {
         isconst |= G__PCONSTVAR;
         c = G__fgetstream(type_name, 1, ";,[");
      }
      else {
         G__disp_mask = strlen(type_name) - 1;
         G__ifile.line_number = tmpline;
         fsetpos(G__ifile.fp, &tmppos);
         c = G__fgetstream(type_name, 1, ";,[");
      }
      isorgtypepointer = 1;
      type = toupper(type);
   }
   else if (!strcmp(type_name, "&")) {
      reftype = G__PARAREFERENCE;
      c = G__fgetstream(type_name, 0, ";,[");
   }
   else if (!strcmp(type_name, "*&")) {
      reftype = G__PARAREFERENCE;
      type = toupper(type);
      c = G__fgetstream(type_name, 0, ";,[");
   }
   else if (!strcmp(type_name, "*const")) {
      isconst |= G__PCONSTVAR;
      c = G__fgetstream(type_name, 1, ";,[");
   }
#ifndef G__OLDIMPLEMENTATION1856
   else if (!strcmp(type_name, "const*")) {
      isconst |= G__CONSTVAR;
      type = toupper(type);
      c = G__fgetstream(type_name, 0, "*&;,[");
      if ((c == '*') && (type_name[0] != '*')) {
         if (!strcmp(type_name, "const")) {
            isconst |= G__CONSTVAR;
         }
         type_name[0] = '*';
         c = G__fgetstream(type_name, 1, ";,[");
      }
      if ((c == '&') && (type_name[0] != '&')) {
         reftype = G__PARAREFERENCE;
         if (!strcmp(type_name, "const")) {
            isconst |= G__CONSTVAR;
         }
         c = G__fgetstream(type_name, 0, ";,[");
      }
   }
   else if (!strcmp(type_name, "const**")) {
      isconst |= G__CONSTVAR;
      isorgtypepointer = 1;
      type = toupper(type);
      type_name[0] = '*';
      c = G__fgetstream(type_name, 1, "*;,[");
   }
   else if (!strcmp(type_name, "const*&")) {
      isconst |= G__CONSTVAR;
      reftype = G__PARAREFERENCE;
      type = toupper(type);
      c = G__fgetstream(type_name, 0, ";,[");
   }
#endif
   if (isspace(c)) {
      if ((type_name[0] == '(') && (c != ';') && (c != ',')) {
         do {
            c = G__fgetstream(type_name, strlen(type_name), ";,");
            size_t lentype_name = strlen(type_name);
            type_name.Set(lentype_name, c);
            type_name.Set(lentype_name + 1, 0);
         }
         while ((c != ';') && (c != ','));
         type_name[strlen(type_name)-1] = 0;
      }
      else if (!strcmp(type_name, "const")) {
         isconst |= G__PCONSTVAR;
         c = G__fgetstream(type_name, 0, ";,[");
         if (!strncmp(type_name, "*const*", 7)) {
            isconst |= G__CONSTVAR;
            isorgtypepointer = 1;
            type = toupper(type);
            G__shiftstring(type_name, 6);
         }
         else if (!strncmp(type_name, "*const&", 7)) {
            isconst |= G__CONSTVAR;
            reftype = G__PARAREFERENCE;
            type = toupper(type);
            G__shiftstring(type_name, 7);
         }
         else if (!strncmp(type_name, "const*", 6)) {
         }
         else if (!strncmp(type_name, "const&", 6)) {
         }
      }
      else if (!strcmp(type_name, "const*")) {
         isconst |= G__PCONSTVAR;
         type_name[0] = '*';
         c = G__fgetstream(type_name, 1, ";,[");
      }
      else {
         G__FastAllocString ltemp1(G__LONGLINE);
         c = G__fgetstream(ltemp1, 0, ";,[");
         if (ltemp1[0] == '(') {
            type = 'q';
         }
      }
   }
   //
   // in case of
   //   typedef <unsigned long int|struct A {}>  int32 , *pint32;
   //                                                   ^
   //
   nindex = 0;
   while (c == '[') {
      store_var_type = G__var_type;
      G__var_type = 'p';
      c = G__fgetstream(aryindex, 0, "]");
      index[nindex++] = G__int(G__getexpr(aryindex));
      c = G__fignorestream("[,;");
      G__var_type = store_var_type;
   }
   next_name:
   p = strchr(type_name, '(');
   if (p) {
      flag_p2f = 1;
      if (p == type_name) {
         // function to pointer 'typedef type (*newtype)();'
         // handle this as 'typedef void* newtype;'
         val = p + 1;
         p = strchr(val, ')');
         if (p) {
            *p = '\0';
            type_name = val;
            type = 'y';
            p = strstr(type_name, "::*");
         }
         if (p) {
            // pointer to member function 'typedef type (A::*p)();
            val = p + 3;
            type_name = val;
            type = 'a';
         }
      }
      else if ((p == (type_name + 1)) && (type_name[0] == '*')) {
         // function to pointer 'typedef type *(*newtype)();'
         // handle this as 'typedef void* newtype;'
         val = p + 1;
         p = strchr(val, ')');
         if (p) {
            *p = '\0';
            type_name = val;
            type = 'Q';
            p = strstr(type_name, "::*");
         }
         if (p) {
            // pointer to member function 'typedef type (A::*p)();
            val = p + 3;
            type_name = val;
            type = 'a';
         }
      }
      else {
         // function type 'typedef type newtype();'
         // handle this as 'typedef void newtype;'
         *p = '\0';
         type = 'y';
      }
   }
   isnext = 0;
   if (c == ',') {
      isnext = 1;
      fgetpos(G__ifile.fp, &next_fpos);
   }
   //
   //  typedef [struct|union|enum] tagname { member } newtype  ;
   //                                                           ^
   //  read over. Store line number. This will be restored after
   //  struct,union.enum member declaration
   //
   temp_line = G__ifile.line_number;
   // typedef oldtype* newtype
   // newtype is a pointer to oldtype
   if (type_name[0] == '*') {
      int ix = 1;
      if (
         isupper(type)
#ifndef G__OLDIMPLEMENTATION2191
            && (type != '1')
#else // G__OLDIMPLEMENTATION2191
            && (type != 'Q')
#endif // G__OLDIMPLEMENTATION2191
         // --
      ) {
         reftype = G__PARAP2P;
         while (type_name[ix] == '*') {
            if (G__PARANORMAL == reftype) {
               reftype = G__PARAP2P;
            }
            else if (reftype >= G__PARAP2P) {
               ++reftype;
            }
            ++ix;
         }
      }
      else {
         type = toupper(type);
         while (type_name[ix] == '*') {
            if (G__PARANORMAL == reftype) reftype = G__PARAP2P;
            else if (reftype >= G__PARAP2P) ++ reftype;
            ++ix;
         }
      }
      val = type_name;
      type_name = val + ix;
   }
   // typedef oldtype &newtype
   if (type_name[0] == '&') {
      if (reftype >= G__PARAP2P) {
         reftype += G__PARAREF;
      }
      else {
         reftype = G__PARAREFERENCE;
      }
      if (strlen(type_name) > 1) {
         val = type_name;
         type_name = val + 1;
      }
   }
   //
   //  Check if typedef is already defined.
   //
   typenum = G__defined_typename_exact(type_name);
   //
   //  If new typedef, add it to newtype table.
   //
   if (typenum != -1) { // return if the type is already defined
      if (c != ';') {
         G__fignorestream(";");
      }
      return;
   }
   else {
      if (G__newtype.alltype == G__MAXTYPEDEF) {
         G__fprinterr(G__serr, "Limitation: Number of typedef exceed %d FILE:%s LINE:%d\nFatal error, exit program. Increase G__MAXTYPEDEF in G__ci.h and recompile %s\n", G__MAXTYPEDEF, G__ifile.name, G__ifile.line_number, G__nam);
         G__eof = 1;
         return;
      }
      typenum = G__newtype.alltype;
      len = strlen(type_name);
      G__newtype.name[typenum] = (char*) malloc(len + 2);
      strcpy(G__newtype.name[typenum], type_name); // Okay, we allocated the right size
      G__newtype.namerange->Insert(G__newtype.name[typenum], typenum);
      G__newtype.iscpplink[typenum] = G__NOLINK;
      G__newtype.comment[typenum].p.com = 0;
      G__newtype.comment[typenum].filenum = -1;
      G__newtype.nindex[typenum] = nindex;
#ifdef G__TYPEDEFFPOS
      // 21/05/07
      // G__ifile.filenum at this moment might point to a temporary file
      // but later on a real file will be registered with this number
      // and will create confusion.  If this is a temporary file just
      // ignore its number.
      if ((strlen(G__ifile.name) > 3) && !strncmp("(tmp", G__ifile.name, 4)) {
         G__newtype.filenum[typenum] = -1;
         G__newtype.linenum[typenum] = -1;
      }
      else {
         G__newtype.filenum[typenum] = G__ifile.filenum;
         G__newtype.linenum[typenum] = G__ifile.line_number;
      }
#endif // G__TYPEDEFFPOS
      if (nindex) {
         G__newtype.index[typenum] = (int*) malloc(nindex * G__INTALLOC);
         memcpy((void*) G__newtype.index[typenum], (void*) index, nindex * G__INTALLOC);
      }
      G__newtype.hash[typenum] = len;
      if (!tagname[0]) {
         if (G__globalcomp == G__CPPLINK) {
            tagname = type_name;
         }
         else {
            tagname = "$";
            tagname += type_name;
         }
         taglen = strlen(tagname);
      }
      else {
         taglen = strlen(tagname);
         if (tagname[taglen-1] == '*') {
            type = toupper(type);
            tagname[taglen-1] = '\0';
         }
      }
      G__newtype.tagnum[typenum] = -1;
      G__newtype.type[typenum] = type;
      G__newtype.globalcomp[typenum] = G__default_link ? G__globalcomp : G__NOLINK;
      G__newtype.reftype[typenum] = reftype;
      G__newtype.isconst[typenum] = isconst;
      if (G__def_struct_member) {
         env_tagnum = G__tagnum;
      }
      else if (G__func_now != -1) {
         env_tagnum = -2;
         G__fprinterr(G__serr, "Limitation: In function typedef not allowed in cint");
         G__printlinenum();
      }
      else {
         env_tagnum = -1;
      }
      G__newtype.parent_tagnum[typenum] = env_tagnum;
      ++G__newtype.alltype;
   }
   if (tolower(type) == 'u') {
      G__tagnum = G__search_tagname(tagname, tagtype);
      if (G__tagnum < 0) {
         G__fignorestream(";");
         return;
      }
      G__newtype.tagnum[typenum] = G__tagnum;
      if (mem_def == 1) {
         if (!G__struct.size[G__tagnum]) {
            fsetpos(G__ifile.fp, &rewind_fpos);
            G__struct.line_number[G__tagnum] = G__ifile.line_number;
            G__struct.filenum[G__tagnum] = G__ifile.filenum;
            G__struct.parent_tagnum[G__tagnum] = env_tagnum;
            //
            // in case of enum
            //
            if (tagtype == 'e') {
               G__disp_mask = 10000;
               while ((c = G__fgetc()) != '{') {}
               enumval.obj.i = -1;
               enumval.type = 'i' ;
               enumval.tagnum = G__tagnum ;
               enumval.typenum = typenum ;
               G__constvar = G__CONSTVAR;
               G__enumdef = 1;
               do {
                  c = G__fgetstream(memname, 0, "=,}");
                  if (c == '=') {
                     int store_prerun = G__prerun;
                     char store_var_type = G__var_type;
                     G__var_type = 'p';
                     G__prerun = 0;
                     c = G__fgetstream(val, 0, ",}");
                     enumval = G__getexpr(val);
                     G__prerun = store_prerun;
                     G__var_type = store_var_type;
                  }
                  else {
                     ++enumval.obj.i;
                  }
                  G__var_type = 'i';
                  if (store_tagnum != -1) {
                     store_def_struct_member = G__def_struct_member;
                     G__def_struct_member = 0;
                     G__static_alloc = 1;
                  }
                  Cint::G__DataMemberHandle member;
                  G__letvariable(memname, enumval, &G__global , G__p_local, member);
                  if (store_tagnum != -1) {
                     G__def_struct_member = store_def_struct_member;
                     G__static_alloc = 0;
                  }
               }
               while (c != '}');
               G__constvar = 0;
               G__enumdef = 0;
               G__fignorestream(";");
               G__disp_mask = 0;
               G__ifile.line_number = temp_line;
            }
            //
            // in case of struct,union
            //
            else {
               if (tagtype != 's' && tagtype != 'c' && tagtype != 'u') {
                  /* enum already handled above */
                  G__fprinterr(G__serr, "Error: Illegal tagtype. struct,union,enum expected\n");
               }
               store_local = G__p_local;
               G__p_local = G__struct.memvar[G__tagnum];
               store_def_struct_member = G__def_struct_member;
               G__def_struct_member = 1;
               G__disp_mask = 10000;
               G__tagdefining = G__tagnum;
               G__def_tagnum = G__tagdefining;
               // Tell the parser to process the entire struct block.
               int brace_level = 0;
               // And call the parser.
               G__exec_statement(&brace_level);
               G__tagnum = G__tagdefining;
               G__def_tagnum = store_def_tagnum;
               //
               // Padding for PA-RISC, Spark, etc
               // If struct size can not be divided by G__DOUBLEALLOC the size is aligned.
               //
               if (G__struct.memvar[G__tagnum]->allvar == 1) {
                  // this is still questionable, inherit0.c
                  struct G__var_array* v = G__struct.memvar[G__tagnum];
                  if (v->type[0] == 'c') {
                     if (isupper(v->type[0])) {
                        int num_of_elements = v->varlabel[0][1] /* number of elements */;
                        if (!num_of_elements) {
                           num_of_elements = 1;
                        }
                        G__struct.size[G__tagnum] = num_of_elements * G__LONGALLOC;
                     }
                     else {
                        G__value buf;
                        buf.type = v->type[0];
                        buf.tagnum = v->p_tagtable[0];
                        buf.typenum = v->p_typetable[0];
                        int num_of_elements = v->varlabel[0][1] /* number of elements */;
                        if (!num_of_elements) {
                           num_of_elements = 1;
                        }
                        G__struct.size[G__tagnum] = num_of_elements * G__sizeof(&buf);
                     }
                  }
               }
               else if (G__struct.size[G__tagnum] % G__DOUBLEALLOC) {
                  G__struct.size[G__tagnum] += G__DOUBLEALLOC - G__struct.size[G__tagnum] % G__DOUBLEALLOC;
               }
               else if (!G__struct.size[G__tagnum]) {
                  G__struct.size[G__tagnum] = G__CHARALLOC;
               }
               G__tagdefining = store_tagdefining;
               G__def_struct_member = store_def_struct_member;
               G__p_local = store_local;
               G__fignorestream(";");
               G__disp_mask = 0;
               G__ifile.line_number = temp_line;
            }
         }
      }
      G__tagnum = store_tagnum;
   }
   if (isnext) {
      fsetpos(G__ifile.fp, &next_fpos);
      c = G__fgetstream(type_name, 0, ",;");
      goto next_name;
   }
   if (G__fons_comment) {
      G__fsetcomment(&G__newtype.comment[G__newtype.alltype-1]);
   }
   if (
      flag_p2f &&
      (G__newtype.comment[G__newtype.alltype-1].filenum < 0) &&
      !G__newtype.comment[G__newtype.alltype-1].p.com
   ) {
      fpos_t xpos;
      if (G__ifile.filenum > G__nfile) {
         G__fprinterr(G__serr, "Warning: pointer to function typedef incomplete in command line or G__exec_text(). Declare in source file or use G__load_text()\n");
         return;
      }
      ++G__macroORtemplateINfile;
      fgetpos(G__ifile.fp, &xpos);
      fsetpos(G__ifile.fp, &pos_p2fcomment);
      if (G__ifile.fp == G__mfp) {
         G__newtype.comment[G__newtype.alltype-1].filenum = G__MAXFILE;
      }
      else {
         G__newtype.comment[G__newtype.alltype-1].filenum = G__ifile.filenum;
      }
      fgetpos(G__ifile.fp, &G__newtype.comment[G__newtype.alltype-1].p.pos);
      fsetpos(G__ifile.fp, &xpos);
   }
}

//______________________________________________________________________________
int G__defined_typename_noerror(const char* type_name, int noerror)
{
   // Search already defined typedef names, -1 is returned if not found
   //
   // Note: This functions modifies G__var_type.
   //
   int i;
   int len;
   char ispointer = 0;
   char* p;
   int env_tagnum;
   int typenum = -1;
   unsigned long matchflag = 0;
   unsigned long thisflag = 0;
   char* par;

   G__FastAllocString temp2(type_name);

   // find 'xxx::yyy'
   // const ns::T - the const does not belong to the ns!
   char* skipconst = temp2;
   while (!strncmp(skipconst, "const ", 6)) {
      skipconst += 6;
   }
   p = (char*) G__find_last_scope_operator(skipconst);

   G__FastAllocString temp(strlen(skipconst));

   // abandon scope operator if 'zzz (xxx::yyy)www'
   par = strchr(skipconst, '(');
   if (par && p && par < p) {
      p = 0;
   }

   if (p) {
      temp = p + 2;
      *p = '\0';
      if (skipconst == p) {
         env_tagnum = -1; // global scope
      }
#ifndef G__STD_NAMESPACE /* ON745 */
      else if (!strcmp(skipconst, "std") && G__ignore_stdnamespace) {
         env_tagnum = -1;
      }
#endif
      else {
         // first try a typedef, so we don't trigger autoloading here:
         long env_typenum = G__defined_typename_noerror(skipconst, 1);
         if (env_typenum != -1 && G__newtype.type[env_typenum] == 'u')
            env_tagnum = G__newtype.tagnum[env_typenum];
         else
            env_tagnum = G__defined_tagname(skipconst, noerror);
      }
   }
   else {
      temp = skipconst;
      env_tagnum = G__get_envtagnum();
   }

   len = strlen(temp);

   if ((len > 0) && (temp[len-1] == '*')) {
      temp[--len] = '\0';
      ispointer = 'A' - 'a';
   }

   if (G__newtype.namerange) {
      NameMap::Range nameRange = G__newtype.namerange->Find(temp);
      if (nameRange) {
         for (i = nameRange.First(); i <= nameRange.Last(); ++i) {
            if ((len == G__newtype.hash[i]) && !strcmp(G__newtype.name[i], temp)) {
               thisflag = 0;
               // global scope
               if (
                   (G__newtype.parent_tagnum[i] == -1)
#if !defined(G__OLDIMPLEMTATION2100)
                   && (!p || ((skipconst == p) || !strcmp("std", skipconst)))
#elif !defined(G__OLDIMPLEMTATION1890)
                   && (!p || (skipconst == p))
#endif
                   ) {
                  thisflag = 0x1;
               }
               // enclosing tag scope
               if (G__isenclosingclass(G__newtype.parent_tagnum[i], env_tagnum)) {
                  thisflag = 0x2;
               }
               // template definition enclosing class scope
               if (G__isenclosingclass(G__newtype.parent_tagnum[i], G__tmplt_def_tagnum)) {
                  thisflag = 0x4;
               }
               // baseclass tag scope
               if (-1 != G__isanybase(G__newtype.parent_tagnum[i], env_tagnum, G__STATICRESOLUTION)) {
                  thisflag = 0x8;
               }
               // template definition base class scope
               if (-1 != G__isanybase(G__newtype.parent_tagnum[i], G__tmplt_def_tagnum, G__STATICRESOLUTION)) {
                  thisflag = 0x10;
               }
               if (!thisflag && G__isenclosingclassbase(G__newtype.parent_tagnum[i], env_tagnum)) {
                  thisflag = 0x02;
               }
               if (!thisflag && G__isenclosingclassbase(G__newtype.parent_tagnum[i], G__tmplt_def_tagnum)) {
                  thisflag = 0x04;
               }
               // exact template definition scope
               if ((G__tmplt_def_tagnum > -1) && (G__tmplt_def_tagnum == G__newtype.parent_tagnum[i])) {
                  thisflag = 0x20;
               }
               // exact tag scope
               if ((env_tagnum > -1) && (env_tagnum == G__newtype.parent_tagnum[i])) {
                  thisflag = 0x40;
               }
               
               if (thisflag && (thisflag >= matchflag)) {
                  matchflag = thisflag;
                  typenum = i;
                  G__var_type = G__newtype.type[i] + ispointer;
               }
            }
         }
      } else {
         i = G__newtype.alltype;
      }
   } else {
      i = G__newtype.alltype;
   }      
   return typenum;
}

//______________________________________________________________________________
//
//  Functions in the C interface.
//

//______________________________________________________________________________
int G__defined_typename(const char* type_name)
{
   // Search already defined typedef names, -1 is returned if not found
   //
   // Note: This functions modifies G__var_type.
   //
   return G__defined_typename_noerror(type_name, 0);
}

//______________________________________________________________________________
int G__search_typename(const char* typenamein, int typein, int tagnum, int reftype)
{
   // Search typedef name, and if not found, allocate new entry if typein is not zero.
   int i, flag = 0, len;
   char ispointer = 0;

   G__FastAllocString type_name(typenamein);
   /* keep uniqueness for pointer to function typedefs */
#ifndef G__OLDIMPLEMENTATION2191
   if ('1' == typein) G__make_uniqueP2Ftypedef(type_name);
#else
   if ('Q' == typein) G__make_uniqueP2Ftypedef(type_name);
#endif

   /* G__OLDIMPLEMENTATIONON620 should affect, but not implemented here */
   /* Doing exactly the same thing as G__defined_typename() */
   len = strlen(type_name);
   if (
      len &&
      type_name[len-1] == '*') {
      type_name[--len] = '\0';
      ispointer = 'A' - 'a';
   }
   const char *atom_tagname = type_name;
   if ( strstr(type_name,"(*)") == 0 && strstr(type_name,"::*)") == 0 ) {
      // Deal with potential scope in the name but only when not a function type
      char *p = (char*) G__find_last_scope_operator(type_name);
      if (p) {
         if (G__static_parent_tagnum != -1) {
            // humm something is wrong, we specify the parent in 2 different
            // ways.
         }
         atom_tagname = p+2;
         *p = '\0';
         int scope_tagnum = -1;
         if (p == type_name) {
            scope_tagnum = -1;  // global scope
         }
#ifndef G__STD_NAMESPACE
         else if (!strcmp(type_name, "std") && G__ignore_stdnamespace) {
            scope_tagnum = -1;
         }
#endif // G__STD_NAMESPACE
         else {
            // first try a typedef, so we don't trigger autoloading here:
            int scope_typenum = G__defined_typename_noerror(type_name, 1);
            if (scope_typenum != -1 && G__newtype.type[scope_typenum] == 'u')
               scope_tagnum = G__newtype.tagnum[scope_typenum];
            else
               scope_tagnum = G__defined_tagname(type_name, 0);
         }
         G__static_parent_tagnum = scope_tagnum;
         len = strlen(atom_tagname);
      }
   }
   
   NameMap::Range nameRange = G__newtype.namerange->Find(atom_tagname);
   if (nameRange) {
      for (i = nameRange.First();i <= nameRange.Last();i++) {
         if (len == G__newtype.hash[i] && strcmp(G__newtype.name[i], atom_tagname) == 0 &&
             (G__static_parent_tagnum == -1 ||
              G__newtype.parent_tagnum[i] == G__static_parent_tagnum)) {
            flag = 1;
            G__var_type = G__newtype.type[i] + ispointer ;
            break;
         }
      }
   } else {
      i = G__newtype.alltype;
   }
   /* Above is same as G__defined_typename() */

   /* allocate new type table entry */
   if (flag == 0 && typein) {
      if (G__newtype.alltype == G__MAXTYPEDEF) {
         G__fprinterr(G__serr,
                      "Limitation: Number of typedef exceed %d FILE:%s LINE:%d\nFatal error, exit program. Increase G__MAXTYPEDEF in G__ci.h and recompile %s\n"
                      , G__MAXTYPEDEF , G__ifile.name , G__ifile.line_number , G__nam);
         G__eof = 1;
         G__var_type = 'p';
         return(-1);
      }
      G__newtype.hash[G__newtype.alltype] = len;
      G__newtype.name[G__newtype.alltype] = (char*)malloc((size_t)(len + 1));
      strcpy(G__newtype.name[G__newtype.alltype], atom_tagname); // Okay, we allocated the right size
      G__newtype.namerange->Insert(G__newtype.name[G__newtype.alltype], G__newtype.alltype);
      G__newtype.nindex[G__newtype.alltype] = 0;
      G__newtype.parent_tagnum[G__newtype.alltype] = G__static_parent_tagnum;
      G__newtype.isconst[G__newtype.alltype] = G__static_isconst;
      G__newtype.type[G__newtype.alltype] = typein + ispointer;
      G__newtype.tagnum[G__newtype.alltype] = tagnum;
      G__newtype.globalcomp[G__newtype.alltype]
      = G__default_link ? G__globalcomp : G__NOLINK;
      G__newtype.reftype[G__newtype.alltype] = reftype;
      G__newtype.iscpplink[G__newtype.alltype] = G__NOLINK;
      G__newtype.comment[G__newtype.alltype].p.com = (char*)NULL;
      G__newtype.comment[G__newtype.alltype].filenum = -1;
#ifdef G__TYPEDEFFPOS
      G__newtype.filenum[G__newtype.alltype] = G__ifile.filenum;
      G__newtype.linenum[G__newtype.alltype] = G__ifile.line_number;
#endif
      ++G__newtype.alltype;
   }
   return(i);
}

//______________________________________________________________________________
int G__search_typename2(const char* type_name, int typein, int tagnum, int reftype, int parent_tagnum)
{
   int ret;
   G__static_parent_tagnum = parent_tagnum;
   if (-1 == G__static_parent_tagnum && G__def_struct_member &&
         'n' == G__struct.type[G__tagdefining]) {
      G__static_parent_tagnum = G__tagdefining;
   }
   G__static_isconst = reftype / 0x100;
   reftype = reftype % 0x100;
   ret = G__search_typename(type_name, typein, tagnum, reftype);
   G__static_parent_tagnum = -1;
   G__static_isconst = 0;
   G__setnewtype_settypeum(ret);
   return(ret);
}

} // extern "C"

