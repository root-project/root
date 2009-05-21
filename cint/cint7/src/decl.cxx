/* /% C++ %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file decl.c
 ************************************************************************
 * Description:
 *  Variable declaration
 ************************************************************************
 * Copyright(c) 1995~2003  Masaharu Goto
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "common.h"
#include "Dict.h"
#include "Reflex/Builder/TypeBuilder.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace Cint::Internal;

int Cint::Internal::G__initval_eval = 0;
int Cint::Internal::G__dynconst = 0;

// Static functions.
static int G__get_newname(char* new_name);
static int G__setvariablecomment(const char* new_name,Reflex::Member &var);
//--
static void G__removespacetemplate(char* name);
static int G__initstruct(char* new_name);
static int G__initary(char* new_name,Reflex::Member &var);
static void G__initstructary(char* new_name, int tagnum);
static int G__readpointer2function(char* new_name, char* pvar_type);

// Internal functions.
namespace Cint {
namespace Internal {
void G__define_var(int tagnum, ::Reflex::Type typenum);
} // namespace Internal
} // namespace Cint

// Functions in the C interface.
// None.

//______________________________________________________________________________
//
//  Static functions.
//

//______________________________________________________________________________
static int G__get_newname(char* new_name)
{
   // -- Parse a variable name from the input file.
   //
   //  Context is:
   //
   //      type int var1, var2;
   //              ^
   //      type int operator+(type param1, ...);
   //              ^
   //      type operator +(type param1, ...);
   //                   ^
   //
   int store_def_struct_member = 0;
   G__StrBuf temp_sb(G__ONELINE);
   char *temp = temp_sb;
   G__StrBuf temp1_sb(G__ONELINE);
   char *temp1 = temp1_sb;
   int cin = G__fgetvarname(new_name, "*&,;=():}");
   if (cin == '&') {
      if (!strcmp(new_name, "operator")) {
         new_name[8] = cin;
         cin = G__fgetvarname(new_name + 9, ",;=():}");
      }
      else {
         strcat(new_name, "&");
         cin = ' ';
      }
   }
   else if (cin == '*') {
      if (!strcmp(new_name, "operator")) {
         new_name[8] = cin;
         cin = G__fgetvarname(new_name + 9, ",;=():}");
      }
      else {
         strcat(new_name, "*");
         cin = ' ';
      }
   }
   if (isspace(cin)) {
      if (!strcmp(new_name, "const*")) {
         new_name[0] = '*';
         cin = G__fgetvarname(new_name + 1, ",;=():}");
         G__constvar |= G__CONSTVAR;
      }
      if (!strcmp(new_name, "friend")) {
         store_def_struct_member = G__def_struct_member;
         ::Reflex::Scope store_tagdefining = G__tagdefining;
         G__def_struct_member = 0;
         G__tagdefining = ::Reflex::Scope();
         G__define_var(G__get_tagnum(G__tagnum), G__typenum);
         G__def_struct_member = store_def_struct_member;
         G__tagdefining = store_tagdefining;
         new_name[0] = '\0';
         return(';');
      }
      else if (!strcmp(new_name, "&") || !strcmp(new_name, "*")) {
         cin = G__fgetvarname(new_name + 1, ",;=():");
      }
      if (!strcmp(new_name, "&*") || !strcmp(new_name, "*&")) {
         cin = G__fgetvarname(new_name + 2, ",;=():");
      }
      if (!strcmp(new_name, "double") && (G__var_type != 'l')) {
         cin = G__fgetvarname(new_name, ",;=():");
         G__var_type = 'd';
      }
      else if (!strcmp(new_name, "int")) {
         cin = G__fgetvarname(new_name, ",;=():");
      }
      else if (
         !strcmp(new_name, "long") ||
         !strcmp(new_name, "long*") ||
         !strcmp(new_name, "long**") ||
         !strcmp(new_name, "long&")
      ) {
         ::Reflex::Scope store_tagnum = G__tagnum;
         ::Reflex::Type store_typenum = G__typenum;
         int store_decl = G__decl;
         if (!strcmp(new_name, "long")) {
            G__var_type = 'n' + G__unsigned;
            G__tagnum = ::Reflex::Scope();
            G__typenum = ::Reflex::Type();
            G__reftype = G__PARANORMAL;
         }
         else if (!strcmp(new_name, "long*")) {
            G__var_type = 'N' + G__unsigned;
            G__tagnum = ::Reflex::Scope();
            G__typenum = ::Reflex::Type();
            G__reftype = G__PARANORMAL;
         }
         else if (!strcmp(new_name, "long**")) {
            G__var_type = 'N' + G__unsigned;
            G__tagnum = ::Reflex::Scope();
            G__typenum = ::Reflex::Type();
            G__reftype = G__PARAP2P;
         }
         else if (!strcmp(new_name, "long&")) {
            G__var_type = 'n' + G__unsigned;
            G__tagnum = ::Reflex::Scope();
            G__typenum = ::Reflex::Type();
            G__reftype = G__PARAREFERENCE;
         }
         G__define_var(G__get_tagnum(G__tagnum), G__typenum);
         G__var_type = 'p';
         G__tagnum = store_tagnum;
         G__typenum = store_typenum;
         G__decl = store_decl;
         return 0;
      }
      else if (
         (G__var_type == 'l') &&
         (
            !strcmp(new_name, "double") ||
            !strcmp(new_name, "double*") ||
            !strcmp(new_name, "double**") ||
            !strcmp(new_name, "double&")
         )
      ) {
         ::Reflex::Scope store_tagnum = G__tagnum;
         ::Reflex::Type store_typenum = G__typenum;
         int store_decl = G__decl;
         if (!strcmp(new_name, "double")) {
            G__var_type = 'q';
            G__tagnum = ::Reflex::Scope();
            G__typenum = ::Reflex::Type();
            G__reftype = G__PARANORMAL;
         }
         else if (!strcmp(new_name, "double*")) {
            G__var_type = 'Q';
            G__tagnum = ::Reflex::Scope();
            G__typenum = ::Reflex::Type();
            G__reftype = G__PARANORMAL;
         }
         else if (!strcmp(new_name, "double**")) {
            G__var_type = 'Q';
            G__tagnum = ::Reflex::Scope();
            G__typenum = ::Reflex::Type();
            G__reftype = G__PARAP2P;
         }
         else if (!strcmp(new_name, "double&")) {
            G__var_type = 'q';
            G__tagnum = ::Reflex::Scope();
            G__typenum = ::Reflex::Type();
            G__reftype = G__PARAREFERENCE;
         }
         G__define_var(G__get_tagnum(G__tagnum), G__typenum);
         G__var_type = 'p';
         G__tagnum = store_tagnum;
         G__typenum = store_typenum;
         G__decl = store_decl;
         return 0;
      }
      else if (!strcmp(new_name, "unsigned") || !strcmp(new_name, "signed")) {
         cin = G__fgetvarname(new_name, ",;=():");
         --G__var_type; // make it unsigned
         if (!strcmp(new_name, "int*")) {
            G__var_type = toupper(G__var_type);
            cin = G__fgetvarname(new_name, ",;=():");
         }
         else if (!strcmp(new_name, "int&")) {
            G__var_type = toupper(G__var_type);
            cin = G__fgetvarname(new_name, ",;=():");
            G__reftype = G__PARAREFERENCE;
         }
         else if (!strcmp(new_name, "int")) {
            cin = G__fgetvarname(new_name, ",;=():");
         }
      }
      else if (!strcmp(new_name, "int*")) {
         cin = G__fgetvarname(new_name, ",;=():");
         G__var_type = toupper(G__var_type);
      }
      else if (!strcmp(new_name, "double*")) {
         cin = G__fgetvarname(new_name, ",;=():");
         G__var_type = 'D';
      }
      else if (!strcmp(new_name, "int&")) {
         cin = G__fgetvarname(new_name, ",;=():");
         G__reftype = G__PARAREFERENCE;
      }
      else if (!strcmp(new_name, "double&")) {
         cin = G__fgetvarname(new_name, ",;=():");
         G__reftype = G__PARAREFERENCE;
      }
      if (isspace(cin)) {
         if (!strcmp(new_name, "static")) {
            cin = G__fgetvarname(new_name, ",;=():");
            G__static_alloc = 1;
         }
      }
      if (isspace(cin)) {
         if (strcmp(new_name, "*const") == 0 || strcmp(new_name, "const") == 0) {
            if (new_name[0]=='*') {
              cin = G__fgetvarname(new_name + 1, ",;=():");
              G__constvar |= G__PCONSTVAR;
            } else {
              cin = G__fgetvarname(new_name, ",;=():");
              if (isupper(G__var_type)) {
                 G__constvar |= G__PCONSTVAR;
              }   
              else {
                 G__constvar |= G__CONSTVAR;
              }
            }
            if (!strcmp(new_name, "&*") || !strcmp(new_name, "*&")) {
               G__reftype = G__PARAREFERENCE;
               new_name[0] = '*';
               cin = G__fgetvarname(new_name + 1, ",;=():");
            }
            else if (!strcmp(new_name, "&")) {
               G__reftype = G__PARAREFERENCE;
               cin = G__fgetvarname(new_name, ",;=():");
            }
            if (!strcmp(new_name, "*")) {
               cin = G__fgetvarname(new_name + 1, ",;=():");
               if (!strcmp(new_name, "*const")) {
                  G__constvar |= G__PCONSTVAR;
                  cin = G__fgetvarname(new_name + 1, ",;=():");
               }
            }
         }
         else if (!strcmp(new_name, "const&")) {
            cin = G__fgetvarname(new_name, ",;=():");
            G__reftype = G__PARAREFERENCE;
            G__constvar |= G__PCONSTVAR;
         }
         else if (!strcmp(new_name, "*const&")) {
            cin = G__fgetvarname(new_name + 1, ",;=():");
            G__constvar |= G__PCONSTVAR;
            G__reftype = G__PARAREFERENCE;
         }
#ifndef G__OLDIMPLEMENTATION1857
         else if (!strcmp(new_name, "const*&")) {
            new_name[0] = '*';
            cin = G__fgetvarname(new_name + 1, ",;=():");
            G__constvar |= G__CONSTVAR;
            G__reftype = G__PARAREFERENCE;
         }
         else if (!strcmp(new_name, "const**")) {
            new_name[0] = '*';
            cin = G__fgetvarname(new_name + 1, ",;=():");
            G__constvar |= G__CONSTVAR;
            G__var_type = 'U';
            G__reftype = G__PARAP2P;
         }
#endif // G__OLDIMPLEMENTATION1857
         else if (!strcmp(new_name, "volatile")) {
            cin = G__fgetvarname(new_name, ",;=():");
         }
         else if (!strcmp(new_name, "*volatile")) {
            cin = G__fgetvarname(new_name + 1, ",;=():");
         }
         else if (!strcmp(new_name, "**volatile")) {
            cin = G__fgetvarname(new_name + 2, ",;=():");
         }
         else if (!strcmp(new_name, "***volatile")) {
            cin = G__fgetvarname(new_name + 3, ",;=():");
         }
         else if (!strcmp(new_name, "inline")) {
            cin = G__fgetvarname(new_name, ",;=():");
         }
         else if (!strcmp(new_name, "*inline")) {
            cin = G__fgetvarname(new_name + 1, ",;=():");
         }
         else if (!strcmp(new_name, "**inline")) {
            cin = G__fgetvarname(new_name + 2, ",;=():");
         }
         else if (!strcmp(new_name, "***inline")) {
            cin = G__fgetvarname(new_name + 3, ",;=():");
         }
         else if (!strcmp(new_name, "virtual")) {
            G__virtual = 1;
            cin = G__fgetvarname(new_name, ",;=():");
         }
      }
      if (isspace(cin)) {
         if (
            !strcmp(new_name, "operator") ||
            !strcmp(new_name, "*operator") ||
            !strcmp(new_name, "*&operator") ||
            !strcmp(new_name, "&operator")
         ) {
            // Read real name.
            cin = G__fgetstream(temp1, "(");
            // came to
            // type  operator  +(var1 , var2);
            //                  ^
            // type  int   operator + (var1 , var2);
            //                       ^
            switch (temp1[0]) {
               case '+':
               case '-':
               case '*':
               case '/':
               case '%':
               case '^':
               case '<':
               case '>':
               case '@':
               case '&':
               case '|':
               case '=':
               case '!':
               case '[':
               case ',':
                  sprintf(temp, "%s%s", new_name, temp1);
                  strcpy(new_name, temp);
                  break;
               case '\0':
                  cin = G__fgetstream(temp1, ")");
                  if (strcmp(temp1, "") || (cin != ')')) {
                     G__fprinterr(G__serr, "Error: Syntax error '%s(%s%c' ", new_name, temp1, cin);
                     G__genericerror(0);
                  }
                  cin = G__fgetstream(temp1, "(");
                  if (strcmp(temp1, "") || (cin != '(')) {
                     G__fprinterr(G__serr, "Error: Syntax error '%s()%s%c' ", new_name, temp1, cin);
                     G__genericerror(0);
                  }
                  sprintf(temp, "%s()", new_name);
                  strcpy(new_name, temp);
                  break;
               default:
                  sprintf(temp, "%s %s", new_name, temp1);
                  strcpy(new_name, temp);
                  break;
            }
            return cin;
         }
         int store_len = strlen(new_name);
         do {
            cin = G__fgetstream(new_name + strlen(new_name), ",;=():");
            if (cin == ']') {
               strcpy(new_name + strlen(new_name), "]");
            }
         }
         while (cin == ']');
         if ((store_len > 1) && isalnum(new_name[store_len]) && isalnum(new_name[store_len-1])) {
            if (G__dispmsg >= G__DISPWARN) {
               G__fprinterr(G__serr, "Warning: %s  Syntax error??", new_name);
               G__printlinenum();
            }
         }
         return cin;
      }
   }
   else if ((cin == '(') && !new_name[0]) {
      // check which case
      //  1. f(type (*p)(int))  -> do nothing here
      //  2. f(type (*p)[4][4]) -> convert to f(type p[][4][4])
      fpos_t tmppos;
      int tmpline = G__ifile.line_number;
      fgetpos(G__ifile.fp, &tmppos);
      if (G__dispsource) {
         G__disp_mask = 1000;
      }
      cin = G__fgetvarname(new_name, ")");
      if ((new_name[0] != '*') || !new_name[1]) {
         goto escapehere;
      }
      strcpy(temp, new_name + 1);
      cin = G__fgetvarname(new_name, ",;=():}");
      if (new_name[0] != '[') {
         goto escapehere;
      }
      if (G__dispsource) {
         G__disp_mask = 0;
         G__fprinterr(G__serr, "*%s)%s", temp, new_name);
      }
      strcat(temp, "[]");
      strcat(temp, new_name);
      strcpy(new_name, temp);
      return cin;
      escapehere:
      if (G__dispsource) {
         G__disp_mask = 0;
      }
      fsetpos(G__ifile.fp, &tmppos);
      G__ifile.line_number = tmpline;
      new_name[0] = 0;
      cin = '(';
   }
   if (!strncmp(new_name, "operator", 8) && (!new_name[8] || G__isoperator(new_name[8]))) {
      if (cin == '=') {
         fseek(G__ifile.fp, -1, SEEK_CUR);
         if (G__dispsource) {
            G__disp_mask = 1;
         }
         cin = G__fgetstream(new_name + strlen(new_name), "(");
      }
      else if ((cin == '(') && !new_name[8]) {
         cin = G__fgetstream(new_name, ")");
         cin = G__fgetstream(new_name, "(");
         sprintf(new_name, "operator()");
      }
      else if ((cin == ',') && !new_name[8]) {
         cin = G__fgetstream(new_name, "(");
         sprintf(new_name, "operator,");
      }
      return cin;
   }
   else if (
      (!strncmp(new_name, "*operator", 9) || !strncmp(new_name, "&operator", 9)) &&
      (G__isoperator(new_name[9]) || !new_name[9])
   ) {
      if (cin == '=') {
         fseek(G__ifile.fp, -1, SEEK_CUR);
         if (G__dispsource) {
            G__disp_mask = 1;
         }
         cin = G__fgetstream(new_name + strlen(new_name), "(");
      }
      else if ((cin == '(') && !new_name[9]) {
         cin = G__fignorestream(")");
         cin = G__fignorestream("(");
         strcpy(new_name + 9, "()");
      }
      return cin;
   }
   else if (
      (!strncmp(new_name, "&*operator", 10) || !strncmp(new_name, "*&operator", 10)) &&
      (G__isoperator(new_name[10]) || !new_name[10])
   ) {
      if (cin == '=') {
         fseek(G__ifile.fp, -1, SEEK_CUR);
         if (G__dispsource) {
            G__disp_mask = 1;
         }
         cin = G__fgetstream(new_name + strlen(new_name), "(");
      }
      else if ((cin  == '(') && !new_name[10]) {
         cin = G__fignorestream(")");
         cin = G__fignorestream("(");
         strcpy(new_name + 10, "()");
      }
      return cin;
   }
   return cin;
}

//______________________________________________________________________________
static int G__setvariablecomment(const char* new_name, Reflex::Member &var)
{
   // Set the variable comment.
   
   if (var) {
      G__get_properties(var)->comment.filenum = -1;
      G__get_properties(var)->comment.p.com = 0;
      G__fsetcomment(&(G__get_properties(var)->comment));
      return 1;
   }
   else if (new_name && new_name[0] && 0==strchr(new_name,':')) {
      G__fprinterr(G__serr, "Internal warning: %s comment can not set", new_name);
      G__printlinenum();
   }
   return 0;
}

//______________________________________________________________________________
//-- 01
//-- 02
//-- 03
//-- 04
//-- 05
//-- 06
//-- 07
//-- 08
//-- 09
//-- 10
//-- 01
//-- 02

//______________________________________________________________________________
static void G__removespacetemplate(char* name)
{
   // -- FIXME: Describe this function!
   G__StrBuf buf_sb(G__LONGLINE);
   char *buf = buf_sb;
   int c = 0;
   int i = 0;
   int j = 0;
   while ((c = name[i])) {
      if (isspace(c) && (i > 0)) {
         switch (name[i-1]) {
            case ':':
            case '<':
            case ',':
               break;
            case '>':
               if (name[i+1] == '>') {
                  buf[j++] = c;
               }
               break;
            default:
               switch (name[i+1]) {
                  case ':':
                  case '<':
                  case '>':
                  case ',':
                     break;
                  default:
                     buf[j++] = c;
                     break;
               }
               break;
         }
      }
      else {
         buf[j++] = c;
      }
      ++i;
   }
   buf[j] = 0;
   strcpy(name, buf);
}

//______________________________________________________________________________
static int G__initstruct(char* new_name)
{
   // FIXME: We do not handle brace nesting properly,
   //        we need to default initialize members
   //        whose initializers were omitted.
   G__StrBuf expr_sb(G__ONELINE);
   char *expr = expr_sb;
#ifdef G__ASM
   G__abortbytecode();
#endif // G__ASM
   // Separate the variable name from any index specification.
   G__StrBuf name_sb(G__MAXNAME);
   char *name = name_sb;
   std::strcpy(name, new_name);
   {
      char* p = std::strchr(name, '[');
      if (p) {
         *p = '\0';
      }
   }
   if (G__static_alloc && !G__prerun) {
      // -- Ignore a local static structure initialization at runtime.
      int c = G__fignorestream("}");
      c = G__fignorestream(",;");
      return c;
   }
   if (G__static_alloc && G__func_now) {
      // -- Function-local static structure initialization at prerun, use a special global variable name.
      std::string temp;
      G__get_stack_varname(temp, name, G__func_now, G__get_tagnum(G__memberfunc_tagnum));
      //--
      //--
      //--
      //--
      std::strcpy(name, temp.c_str());
   }
   //
   // Lookup the variable.
   //
   ::Reflex::Member member;
   //--
   {
      char* p = std::strstr(name, "::");
      if (p) {
         // -- Qualified name, do the lookup in the specified context.
         *p = '\0';
         p += 2;
         ::Reflex::Scope tagnum = G__Dict::GetDict().GetScope(G__defined_tagname(name, 0));
         if (tagnum) {
            ::Reflex::Scope store_memberfunc_tagnum = G__memberfunc_tagnum;
            int store_def_struct_member = G__def_struct_member;
            int store_exec_memberfunc = G__exec_memberfunc;
            ::Reflex::Scope store_tagnum = G__tagnum;
            G__memberfunc_tagnum = tagnum;
            G__tagnum = tagnum;
            G__def_struct_member = 0;
            G__exec_memberfunc = 1;
            //--
            int hash = 0;
            int i = 0;
            G__hash(p, hash, i);
            member = G__getvarentry(p, hash, tagnum, tagnum);
            G__def_struct_member = store_def_struct_member;
            G__memberfunc_tagnum = store_memberfunc_tagnum;
            G__exec_memberfunc = store_exec_memberfunc;
            G__tagnum = store_tagnum;
         }
      }
      else {
         // -- Unqualified name, do a lookup.
         int hash = 0;
         int i = 0;
         G__hash(name, hash, i);
         member = G__getvarentry(name, hash, ::Reflex::Scope::GlobalScope(), G__p_local);
      }
   }
   if (!member) {
      G__fprinterr(G__serr, "Limitation: %s initialization ignored", name);
      G__printlinenum();
      int c = G__fignorestream("},;");
      if (c == '}') {
         c = G__fignorestream(",;");
      }
      return c;
   }
   // We must be an aggregate type, enforce that.
   if (!G__struct.baseclass[G__get_tagnum(member.TypeOf().RawType())]->vec.empty()) {
      // -- We have base classes, i.e., we are not an aggregate.
      // FIXME: This test should be stronger, the accessibility
      //        of the data members should be tested for example.
      G__fprinterr(G__serr, "Error: %s must be initialized by a constructor", name);
      G__genericerror(0);
      int c = G__fignorestream("}");
      //  type var1[N] = { 0, 1, 2.. }  , ... ;
      // came to                      ^
      c = G__fignorestream(",;");
      //  type var1[N] = { 0, 1, 2.. } , ... ;
      // came to                        ^  or ^
      return c;
   }
   int num_of_elements = G__get_varlabel(member.TypeOf(), 1);
   const int stride = G__get_varlabel(member.TypeOf(), 0);
   // Check for an unspecified length array.
   int isauto = 0;
   if (num_of_elements == INT_MAX /* unspecified length flag */) {
      // -- Set isauto flag and reset number of elements.
      if (G__asm_wholefunction) {
         // -- We cannot bytecompile an unspecified length array.
         G__abortbytecode();
         G__genericerror(0);
      }
      isauto = 1;
      num_of_elements = 0;
      member = G__update_array_dimension(member, 0);
   }
   // Initialize buf.
   G__value buf;
   G__value_typenum(buf) = ::Reflex::PointerBuilder(member.TypeOf());
   //--
   //--
   //--
   buf.ref = 0;
   // Get size.
   ::Reflex::Type element_type = member.TypeOf().FinalType();
   for (; element_type.IsArray(); element_type = element_type.ToType()) {
      // Intentionally empty.
   }
   int size = element_type.SizeOf();
   G__ASSERT((stride > 0) && (size > 0));
   // Get a pointer to the first data member.
   unsigned int memindex = 0;
   G__incsetup_memvar(G__get_tagnum(member.TypeOf().RawType()));
   //
   // Read and process the initializer specification.
   //
   int mparen = 1;
   int linear_index = -1;
   while (mparen) {
      // -- Read the next initializer value.
      int c = G__fgetstream(expr, ",{}");
      if (expr[0]) {
         // -- We have an initializer expression.
         // FIXME: Do we handle a string literal correctly here?  See similar code in G__initary().
         ++linear_index;
         // If we are an array, make sure we have not gone beyond the end.
         if ((num_of_elements || isauto) && (linear_index >= num_of_elements)) {
            // -- We have gone past the end of the array.
            if (isauto) {
               // -- Unspecified length array, make it bigger to fit.
               // Allocate another stride worth of elements.
               num_of_elements += stride;
               member = G__update_array_dimension(member, member.TypeOf().ArrayLength() + stride);
               long tmp = 0L;
               if (G__get_offset(member)) {
                  // -- We already had some elements, resize.
                  tmp = (long) std::realloc((void*) G__get_offset(member), size * num_of_elements);
               }
               else {
                  // -- No elements allocate yet, get some.
                  tmp = (long) std::malloc(size * num_of_elements);
               }
               if (tmp) {
                  G__get_offset(member) = (char*) tmp;
               }
               else {
                  G__malloc_error(new_name);
               }
            }
            else {
               // -- Fixed-size array, error, array index out of range.
               if (G__asm_wholefunction == G__ASM_FUNC_NOP) {
                  if (!G__const_noerror) {
                     G__fprinterr(G__serr, "Error: %s: %d: Array initialization out of range *(%s+%d), upto %d ", __FILE__, __LINE__, name, linear_index, num_of_elements);
                  }
               }
               G__genericerror(0);
               while (mparen-- && (c != ';')) {
                  c = G__fignorestream("};");
               }
               if (c != ';') {
                  c = G__fignorestream(";");
               }
               return c;
            }
         }
         // Loop over the data members and initialize them.
         do {
            ::Reflex::Member m = member.TypeOf().RawType().DataMemberAt(memindex);
            G__value_typenum(buf) = ::Reflex::PointerBuilder(m.TypeOf());
            buf.obj.i = (long) (G__get_offset(member) + (linear_index * size) + (long) G__get_offset(m));
            G__value reg = G__getexpr(expr);
            if (m.TypeOf().FinalType().IsPointer()) {
               // -- Data member is a pointer.
               *(long*)buf.obj.i = (long) G__int(reg);
            }
            else if (
               (G__get_type(m.TypeOf()) == 'c') && // character array
               (G__get_varlabel(m.TypeOf(), 1) /* number of elements */ > 0) &&
               (expr[0] == '"') // string literal
            ) {
               // -- Data member is a fixed-size character array.
               // FIXME: We do not handle a data member which is an unspecified length array.
               if (G__get_varlabel(m.TypeOf(), 1) /* number of elements */ > (int) std::strlen((char*) reg.obj.i)) {
                  std::strcpy((char*) buf.obj.i, (char*) reg.obj.i);
               }
               else {
                  std::strncpy((char*) buf.obj.i, (char*) reg.obj.i, G__get_varlabel(m.TypeOf(), 1) /* number of elements */);
               }
            }
            else {
               G__letvalue(&buf, reg);
            }
            // Move to next data member.
            ++memindex;
            if ((c == '}') || (memindex == member.TypeOf().RawType().DataMemberSize())) {
               // -- All done if no more data members or end of list.
               // FIXME: We are not handling nesting of braces properly.
               //        We need to default initialize the rest of the members.
               break;
            }
            // Get next initializer expression.
            c = G__fgetstream(expr, ",{}");
         }
         while (memindex < member.TypeOf().RawType().DataMemberSize());
         // Reset back to the beginning of the data member list.
         memindex = 0;
      }
      // Change parser state for next initializer expression.
      switch (c) {
         case '{':
            // -- Increment nesting level.
            ++mparen;
            break;
         case '}':
            // -- Decrement nesting level and move to next dimension.
            --mparen;
            break;
         case ',':
            // -- Normal end of an initializer expression.
            break;
      }
   }
   // Read and discard up to the next comma or semicolon.
   int c = G__fignorestream(",;");
   // MyClass var1[N] = { 0, 1, 2.. } , ... ;
   // came to                        ^  or ^
   //
   // Note: The return value c is either a comma or a semicolon.
   return c;
}

//______________________________________________________________________________
static int G__initary(char* new_name,Reflex::Member &in_var)
{
   // -- Parse and execute an array initialization.
   //
   //printf("Begin G__initary for '%s'...\n", new_name);
   static char expr[G__ONELINE];

   // Static array initialization at runtime is special.
   if (G__static_alloc && !G__prerun) {
      // -- A static array initialization at runtime.
      ::Reflex::Member var = in_var;
      assert( var == in_var );
      if (var && G__get_varlabel(var, 1) /* number of elements */ == INT_MAX /* unspecified length flag */) {

         // -- Variable exists and is an unspecified length array.
         // Look for a corresponding special name, both locally and globally.
         std::string namestatic;
         G__get_stack_varname(namestatic, var.Name_c_str(), G__func_now, G__get_tagnum(G__memberfunc_tagnum));

         int dummy;
         int hashstatic = 0;
         G__hash(namestatic.c_str(), hashstatic, dummy)
         //--
         ::Reflex::Member varstatic = G__getvarentry(namestatic.c_str(), hashstatic, ::Reflex::Scope::GlobalScope(), G__p_local);
         if (varstatic) {
            // -- We found the special name variable, copy its array bounds to this variable.
            // FIXME: Do we need to copy any properties here?
            G__RflxVarProperties prop = *G__get_properties(varstatic);
            prop.statictype = G__LOCALSTATIC; // new mbr will be static
            varstatic.InterpreterOffset(G__get_offset(var)); // new mbr has same storage as as old mbr
            ::Reflex::Scope varscope = var.DeclaringScope();
            std::string varname = var.Name();
            varscope.RemoveDataMember(var);
            Reflex::Member newmember = G__add_scopemember(varscope, varname.c_str(), varstatic.TypeOf(), 0, varstatic.Offset(), G__get_offset(var), G__PUBLIC, G__LOCALSTATIC);
            *G__get_properties(newmember) = prop;
         }
      }
      // Ignore initializer.
      // FIXME: This will not properly skip past all the braces in a brace-enclosed initializer list!
      int c = G__fignorestream("}");
      // FIXME: This will not properly skip past all the commas a brace-enclosed initializer list!
      c = G__fignorestream(",;");
      return c;
   }

#ifdef G__ASM
   G__abortbytecode();
#endif // G__ASM

   //
   //  Lookup the variable.
   //
   ::Reflex::Member var = in_var;

   //
   //  At this point we have found the variable.
   //
   //--
   assert( var == in_var );
   // Get number of dimensions.
   const int num_of_dimensions = G__get_paran(var);
   int num_of_elements = G__get_varlabel(var, 1);
   const int stride = G__get_varlabel(var, 0);
   // Check for an unspecified length array.
   int isauto = 0;
   if (num_of_elements == INT_MAX /* unspecified length flag */) {
      // -- Set isauto flag and reset number of elements.
      isauto = 1;
      num_of_elements = 0;
      var = G__update_array_dimension(var, 0);
   }
   std::vector<int> array_bounds(num_of_dimensions + 1);
   if (isauto) {
      array_bounds[1] = 1;
   }
   else {
      array_bounds[1] = num_of_elements / stride;
   }
   for (int i = 2; i <= num_of_dimensions; ++i) {
      array_bounds[i] = G__get_varlabel(var, i);
   }
   std::vector<int> strides(num_of_dimensions + 1);
   {
      int prev_stride = 1;
      for (int i = num_of_dimensions; i > 0; --i) {
         strides[i] = prev_stride * array_bounds[i];
         prev_stride = strides[i];
      }
   }
   //
   //  Get the type of an array element.
   //
   ::Reflex::Type element_type = var.TypeOf().FinalType();
   for (; element_type.IsArray(); element_type = element_type.ToType()) {
      // Intentionally empty
   }
   //
   //  Get the size of an array element.
   //
   int size = element_type.SizeOf();
   //
   //  Assert sanity checks.
   //
   G__ASSERT((stride > 0) && (size > 0));
   //
   //  Initialize buf, we will use it to write to individual
   //  array elements in the initializer handling code.
   //
   G__value buf;
   G__value_typenum(buf) = ::Reflex::PointerBuilder(element_type);
   buf.ref = 0;
   //
   //  Read and execute the intializer specification.
   //
   int brace_level = 1;
   int current_dimension = 1;
   std::vector<int> dimension_stack;
   // Account for the already scanned first brace,
   // we need to restore something after scanning
   // the closing brace.
   dimension_stack.push_back(-1);
   std::vector<int> initializer_count(num_of_dimensions + 1);
   int linear_index = 0;
   while (brace_level) {
      // -- Read the next initializer value.
      int c = G__fgetstream(expr, ",{}");
      if (expr[0]) {
         // -- Found one.
         //printf("%d: '%s', ", linear_index, expr);
         // Check for going past the end of the array.
         if (!isauto && (linear_index >= num_of_elements)) {
            // -- Semantic error, gone past the end of the array.
            if (G__asm_wholefunction == G__ASM_FUNC_NOP) {
               if (!G__const_noerror) {
                  G__fprinterr(G__serr, "Error: Too many initializers, exceeded length of array for '%s'", var.Name(Reflex::SCOPED).c_str());
               }
            }
            G__genericerror(0);
            // Skip the rest of the initializer.
            while (brace_level-- && (c != ';')) {
               c = G__fignorestream("};");
            }
            // Skip any following declarators until the end of the statement.
            if (c != ';') {
               c = G__fignorestream(";");
            }
            return c;
         }
         // Check for too many initializers for a subaggregate.
         if (!isauto && (initializer_count[current_dimension] == strides[current_dimension])) {
            // -- Semantic error, too many initializers.
            if (G__asm_wholefunction == G__ASM_FUNC_NOP) {
               if (!G__const_noerror) {
                  G__fprinterr(G__serr, "Error: Too many initializers for '%s'", var.Name(Reflex::SCOPED).c_str());
               }
            }
            G__genericerror(0);
            // Skip the rest of the initializer.
            while (brace_level-- && (c != ';')) {
               c = G__fignorestream("};");
            }
            // Skip any following declarators until the end of the statement.
            if (c != ';') {
               c = G__fignorestream(";");
            }
            return c;
         }
         //
         //  Evaluate the initializer expression,
         //  we need the value to determine how
         //  much to expand the size of an unspecified
         //  length array of characters.
         //
         G__value reg;
         {
            int store_prerun = G__prerun;
            G__prerun = 0;
            reg = G__getexpr(expr);
            G__prerun = store_prerun;
            //char valbuf[4096];
            //printf("%d: '%s'\n", linear_index, G__valuemonitor(reg, valbuf));
         }
         //
         //  Check for the special case of an array of
         //  characters initialized with a string constant.
         //
         int stringflag = 0;
         if ((G__get_type(var.TypeOf()) == 'c') && (expr[0] == '"')) {
            // -- We have a character array element initialized with a string literal.
            stringflag = 1;
         }
         //
         //  Assign the initializer value to the array element.
         //
         if (!stringflag) {
            // -- Normal case.
            // Auto expand an unspecified length array.
            if (isauto && (linear_index >= num_of_elements)) {
               // -- Unspecified length array is now too small, make it bigger to fit.
               // Allocate another stride worth of elements.
               var = G__update_array_dimension(var, var.TypeOf().ArrayLength() + stride);
               num_of_elements += stride;
               array_bounds[1] = num_of_elements / stride;
               strides[1] = num_of_elements;
               void* tmp = 0;
               if (G__get_offset(var)) {
                  // -- We already had some elements, resize.
                  tmp = std::realloc((void*) G__get_offset(var), size * num_of_elements);
               }
               else {
                  // -- No elements allocate yet, get some.
                  tmp = std::malloc(size * num_of_elements);
               }
               if (tmp) {
                  G__get_offset(var) = static_cast<char*>(tmp);
               }
               else {
                  G__malloc_error(new_name);
               }
            }
            buf.obj.i = ((long) G__get_offset(var)) + (linear_index * size);
            G__letvalue(&buf, reg);
            // Count initializers seen.
            ++initializer_count[current_dimension];
            // Next array element.
            ++linear_index;
         }
         else {
            // -- The initializer is a string constant, and we are an array of characters.
            // We must be on a subaggregate boundary (in this case a string boundary).
            if (linear_index % strides[num_of_dimensions]) {
               // -- Semantic error, attempt to initialize a char with a string array constant.
               if (G__asm_wholefunction == G__ASM_FUNC_NOP) {
                  if (!G__const_noerror) {
                     G__fprinterr(G__serr, "Error: Attempt to initialize a char with a string constant for '%s'", var.Name(Reflex::SCOPED).c_str());
                  }
               }
               G__genericerror(0);
               // Skip the rest of the initializer.
               while (brace_level-- && (c != ';')) {
                  c = G__fignorestream("};");
               }
               // Skip any following declarators until the end of the statement.
               if (c != ';') {
                  c = G__fignorestream(";");
               }
               return c;
            }
            // Get the length of the initializer.
            // Note: We need to count the zero byte at the end of the string constant.
            int len = std::strlen((char*) reg.obj.i) + 1;
            // Initializer must not be too big for the string subaggregate.
            if (
               (!isauto || (num_of_dimensions > 1)) && // We have a fixed-size to compare against, and
               (len > strides[num_of_dimensions]) // the initializer is too big
            ) {
               // -- Semantic error, attempt to initialize a char array with a string array constant of the wrong size.
               if (G__asm_wholefunction == G__ASM_FUNC_NOP) {
                  if (!G__const_noerror) {
                     G__fprinterr(G__serr, "Error: Initializer for a char array is too big while intializing '%s'", var.Name(Reflex::SCOPED).c_str());
                  }
               }
               G__genericerror(0);
               // Skip the rest of the initializer.
               while (brace_level-- && (c != ';')) {
                  c = G__fignorestream("};");
               }
               // Skip any following declarators until the end of the statement.
               if (c != ';') {
                  c = G__fignorestream(";");
               }
               return c;
            }
            // Auto expand an unspecified length array of characters.
            if (isauto && (linear_index >= num_of_elements)) {
               // -- Unspecified length array is now too small, make it bigger to fit.
               // Allocate another stride worth of elements.
               if (num_of_dimensions > 1) {
                  // -- Unspecified length array of fixed-size arrays of char.
                  // For example:
                  //
                  //      char ary[][4] = { "abc" };
                  //
                  // Note that the zero byte in the string constant is counted.
                  //
                  var = G__update_array_dimension(var, var.TypeOf().ArrayLength() + stride);
                  num_of_elements += stride;
               } else {
                  // -- Unspecified length array of char.
                  // For example:
                  //
                  //      char ary[] = { "abc" };
                  //
                  // Note that the zero byte in the string constant is counted.
                  //
                  var = G__update_array_dimension(var, var.TypeOf().ArrayLength() + len);
                  num_of_elements += len;
               }
               array_bounds[1] = num_of_elements / stride;
               strides[1] = num_of_elements;
               void* tmp = 0;
               if (G__get_offset(var)) {
                  // -- We already had some elements, resize.
                  tmp = std::realloc((void*) G__get_offset(var), size * num_of_elements);
               }
               else {
                  // -- No elements allocate yet, get some.
                  tmp = std::malloc(size * num_of_elements);
               }
               if (tmp) {
                  G__get_offset(var) = static_cast<char*>(tmp);
               }
               else {
                  G__malloc_error(new_name);
               }
            }
            buf.obj.i = (long) G__get_offset(var) + (linear_index * size);
            //printf("initializer %d: '%s' --> %08X\n", linear_index, (char*) reg.obj.i, (long) buf.obj.i);
            std::memcpy((void*) buf.obj.i, (void*) reg.obj.i, len);
            // Count initializers seen.
            initializer_count[current_dimension] += len;
            // Next array element.
            linear_index += len;
            // Default initialize any omitted elements.
            if (
               (!isauto || (num_of_dimensions > 1)) && // We have a fixed-size, and
               (len < strides[num_of_dimensions]) // the initializer was too small
            ) {
               // -- Default initialize the omitted array elements.
               int num_omitted = strides[num_of_dimensions] - len;
               buf.obj.i = ((long) G__get_offset(var)) + (linear_index * size);
               std::memset((void*) buf.obj.i, 0, num_omitted);
               // Count initializers seen.
               initializer_count[current_dimension] += num_omitted;
               // Next array element.
               linear_index += num_omitted;
            }
            // --
         }
      }
      switch (c) {
         case '{':
            {
               // -- Begin of a subaggregate initializer.
               //printf("\n{ ");
               // Increase brace nesting level.
               ++brace_level;
               if (current_dimension == num_of_dimensions) {
                  // -- Syntax error, too many open curly braces, exceeded dimensionality of array.
                  G__genericerror("Error: Nesting level too deep in initializer, exceeded dimensionality of array.");
                  // Skip the rest of the initializer.
                  while (brace_level-- && (c != ';')) {
                     c = G__fignorestream("};");
                  }
                  // Skip any following declarators until the end of the statement.
                  if (c != ';') {
                     c = G__fignorestream(";");
                  }
                  //  int var1[3] = { 0, 1, 2 }, i, j, k;
                  //                                     ^
                  return c;
               }
               if (linear_index % strides[num_of_dimensions]) {
                  // -- Syntax error, a brace must begin on a boundary of the lowest subaggregate.
                  G__genericerror("Error: Attempt to initialize an array element with a bracketed expression.");
                  // Skip the rest of the initializer.
                  while (brace_level-- && (c != ';')) {
                     c = G__fignorestream("};");
                  }
                  // Skip any following declarators until the end of the statement.
                  if (c != ';') {
                     c = G__fignorestream(";");
                  }
                  //  int var1[3] = { 0, 1, 2 }, i, j, k;
                  //                                     ^
                  return c;
               }
               // Remember previous dimension.
               //printf("stacking dimension: %d \n", current_dimension);
               dimension_stack.push_back(current_dimension);
               // Change to the nearest allowed subaggregate.
               for (int i = current_dimension + 1; i <= num_of_dimensions; ++i) {
                  if (!(linear_index % strides[i])) {
                     current_dimension = i;
                     //printf("new dimension: %d \n", current_dimension);
                     break;
                  }
               }
               // Zero initializer count for newly opened subaggregate.
               initializer_count[current_dimension] = 0;
            }
            break;
         case '}':
            {
               // -- End of a subaggregate initializer.
               // Default initialize omitted elements.
               // FIXME: This will not work for strings.
               //
               // Note: A series of close curly braces can count as many omitted elements, e.g.,
               //
               //      int ary[3][3][3] = { { {1, 2, 3} } };
               //
               //printf("\n} ");
               int local_stride = strides[current_dimension];
               //printf("local_stride: %d\n", local_stride);
               //printf("linear_index: %d\n", linear_index);
               int num_given = 0;
               if (initializer_count[current_dimension]) {
                  // -- There were initializers.
                  if (linear_index) {
                     num_given = linear_index - (((linear_index - 1) / local_stride) * local_stride);
                  }
               }
               //printf("num_given: %d\n", num_given);
               int num_omitted = local_stride - num_given;
               //printf("num_omitted: %d\n", num_omitted);
               for (int i = 0; i < num_omitted; ++i) {
                  buf.obj.i = ((long) G__get_offset(var)) + (linear_index * size);
                  G__letvalue(&buf, G__null);
                  //printf("%d: 0, ", linear_index);
                  ++linear_index;
               }
               G__ASSERT(!(linear_index % local_stride));
               // Decrease brace nesting level.
               --brace_level;
               // Change to the previous dimension.
               current_dimension = dimension_stack.back();
               dimension_stack.pop_back();
               //printf("restored dimension: %d\n", current_dimension);
               // Add the recently closed aggregate to the initializer count.
               if (current_dimension != -1) {
                  initializer_count[current_dimension] += local_stride;
               }
            }
            break;
         case ',':
            // -- Next initializer.
            break;
      }
   }
   G__ASSERT(linear_index == num_of_elements);
   // FIXME: This is bizzare!
   if (!G__asm_noverflow && G__no_exec_compile) {
      // FIXME: Why?
      G__no_exec = 1;
   }
   // Read and discard up to the next ',' or ';'.
   // FIXME: We should only allow spaces here!
   int c = G__fignorestream(",;");
   //  type var1[N] = { 0, 1, 2.. } , ... ;
   // came to                        ^  or ^
   return c;
}

//______________________________________________________________________________
static void G__initstructary(char* new_name, int tagnum)
{
   // -- Initialize an array of structures.
   //
   // A string[3] = { "abc", "def", "hij" };
   // A string[]  = { "abc", "def", "hij" };
   //                ^
   int cin = 0;
   char* store_struct_offset = G__store_struct_offset;
   char* store_globalvarpointer = G__globalvarpointer;
   G__StrBuf buf_sb(G__ONELINE);
   char *buf = buf_sb;
#ifdef G__ASM
   G__abortbytecode();
#endif // G__ASM
   // Count number of array elements if needed.
   int p_inc = 0;
   char* index = std::strchr(new_name, '[');
   if (*(index + 1) == ']') {
      // -- Unspecified length array.
      // Remember the beginning the of the initializer spec.
      int store_line = G__ifile.line_number;
      std::fpos_t store_pos;
      fgetpos(G__ifile.fp, &store_pos);
      // Now count initializers.
      // FIXME: This does not allow nested curly braces.
      p_inc = 0;
      do {
         cin = G__fgetstream(buf, ",}");
         ++p_inc;
      }
      while (cin != '}');
      // Now modify the name by adding the calculated dimensionality.
      // FIXME: We modify new_name, which may not be big enough!
      std::strcpy(buf, index + 1);
      std::sprintf(index + 1, "%d", p_inc);
      std::strcat(new_name, buf);
      // Rewind the file back to the beginning of the initializer spec.
      G__ifile.line_number = store_line;
      std::fsetpos(G__ifile.fp, &store_pos);
   }
   else {
      p_inc = G__getarrayindex(index);
   }
   // Allocate memory.
   G__value reg = G__null;
   G__decl_obj = 2;
   char* adr = (char*) G__int(G__letvariable(new_name, reg, ::Reflex::Scope::GlobalScope(), G__p_local));
   G__decl_obj = 0;
   // Read and initalize each element.
   std::strcpy(buf, G__struct.name[tagnum]);
   strcat(buf, "(");
   long len = strlen(buf);
   int i = 0;
   do {
      cin = G__fgetstream(buf + len, ",}");
      std::strcat(buf, ")");
      if (G__struct.iscpplink[tagnum] != G__CPPLINK) {
         G__store_struct_offset = adr + (i * G__struct.size[tagnum]);
      }
      else {
         G__globalvarpointer = adr + (i * G__struct.size[tagnum]);
      }
      int known = 0;
      G__getfunction(buf, &known, G__CALLCONSTRUCTOR);
      ++i;
   }
   while (cin != '}');
   G__store_struct_offset = store_struct_offset;
   G__globalvarpointer = store_globalvarpointer;
}

//______________________________________________________________________________
static int G__readpointer2function(char* new_name, char* pvar_type)
{
   // -- FIXME: Describe this function!
   //
   // 'type (*func[n])(type var1,...);'
   // 'type (*ary)[n];'
   //
   int isp2memfunc = G__POINTER2FUNC;
   // Flag that name started with '*'.
   int ispointer = 0;
   if (new_name[0] == '*') {
      ispointer = 1;
   }
   // Pointer to function.
   //
   //   Function call returning pointer to function:
   //
   //        type (*funcpointer(type arg,...))(type var1,...)
   //              ^
   //
   //   Array of pointers to function:
   //
   //        type (*funcpointer[n])(type var1,...)
   //              ^
   //
   //--
   // Read variable name of function pointer.
   fpos_t pos2;
   fgetpos(G__ifile.fp, &pos2);
   int line2 = G__ifile.line_number;
   int c = G__fgetstream(new_name, "()");
   if ((new_name[0] != '*') && !strstr(new_name, "::*")) {
      fsetpos(G__ifile.fp, &pos2);
      G__ifile.line_number = line2;
      return G__CONSTRUCTORFUNC;
   }
   if (c == '(') {
      // -- We have a function call returning a pointer to function.
      fgetpos(G__ifile.fp, &pos2);
      line2 = G__ifile.line_number;
      c = G__fignorestream(")");
      c = G__fignorestream(")");
   }
   else {
      line2 = 0;
   }
   G__StrBuf tagname_sb(G__ONELINE);
   char *tagname = tagname_sb;
   tagname[0] = '\0';
   {
      char* p = strstr(new_name, "::*");
      if (p) {
         isp2memfunc = G__POINTER2MEMFUNC;
         // (A::*p)(...)  => new_name="p" , tagname="A::"
         strcpy(tagname, new_name);
         p = strstr(tagname, "::*");
         strcpy(new_name, p + 3);
         *(p + 2) = '\0';
      }
   }
   // pointer to function
   //   type ( *funcpointer[n])( type var1,.....)
   //                          ^
   c = G__fignorestream("([");
   // pointer to function
   //   type ( *funcpointer[n])( type var1,.....)
   //                           ^
   if (c == '[') {
      // -- type (*pary)[n]; pointer to array
      //                ^
      G__StrBuf temp_sb(G__ONELINE);
      char *temp = temp_sb;
      int n = 0;
      while (c == '[') {
         c = G__fgetstream(temp, "]");
         G__p2arylabel[n++] = G__int(G__getexpr(temp));
         c = G__fgetstream(temp, "[;,)=");
      }
      G__p2arylabel[n] = 0;
      fseek(G__ifile.fp, -1, SEEK_CUR);
      if (G__dispsource) {
         G__disp_mask = 1;
      }
   }
   else {
      // -- type (*pfunc)(...); pointer to function
      //                 ^
      // Set newtype for pointer to function.
      G__StrBuf temp_sb(G__ONELINE);
      char *temp = temp_sb;
      fpos_t pos;
      fgetpos(G__ifile.fp, &pos);
      int line = G__ifile.line_number;
      if (G__dispsource) {
         G__disp_mask = 1000; // FIXME: Gross hack!
      }
      if (ispointer) {
         sprintf(temp, "%s *(%s*)(", G__type2string(G__var_type, G__get_tagnum(G__tagnum), G__get_typenum(G__typenum), G__reftype, G__constvar), tagname);
      }
      else {
         sprintf(temp, "%s (%s*)(", G__type2string(G__var_type, G__get_tagnum(G__tagnum), G__get_typenum(G__typenum), G__reftype, G__constvar) , tagname);
      }
      c = G__fdumpstream(temp + strlen(temp), ")");
      temp[strlen(temp)+1] = '\0';
      temp[strlen(temp)] = c;
      G__tagnum = ::Reflex::Scope();
      if (isp2memfunc == G__POINTER2MEMFUNC) {
         G__typenum = G__declare_typedef(temp, 'a', -1, 0, 0, G__NOLINK, -1, true);
         sprintf(temp, "G__p2mf%d", G__get_typenum(G__typenum));
         G__typenum = G__declare_typedef(temp, 'a', -1, 0, 0, G__NOLINK, -1, true);
         G__var_type = 'a';
         *pvar_type = 'a';
      }
      else {
         // --
#ifndef G__OLDIMPLEMENTATION2191
         G__typenum = G__declare_typedef(temp, '1', -1, 0, 0, G__NOLINK, -1, true);
         G__var_type = '1';
         *pvar_type = '1';
#else // G__OLDIMPLEMENTATION2191
         G__typenum = G__declare_typedef(temp, 'Q', -1, 0, 0, G__NOLINK, -1, true);
         G__var_type = 'Q';
         *pvar_type = 'Q';
#endif // G__OLDIMPLEMENTATION2191
         // --
      }
      G__ifile.line_number = line;
      fsetpos(G__ifile.fp, &pos);
      if (G__dispsource) {
         G__disp_mask = 0;
      }
      if (G__asm_dbg) {
         if (G__dispmsg >= G__DISPNOTE) {
            G__fprinterr(G__serr, "Note: pointer to function exists");
            G__printlinenum();
         }
      }
      if (line2) {
         // function returning pointer to function
         //   type (*funcpointer(type arg))(type var1,.....)
         //                      ^ <------- ^
         fsetpos(G__ifile.fp, &pos2);
         G__ifile.line_number = line2;
         return G__FUNCRETURNP2F;
      }
      G__fignorestream(")");
   }
   return isp2memfunc;
}

//______________________________________________________________________________
//
//  Internal functions.
//

//______________________________________________________________________________
void Cint::Internal::G__define_var(int tagnum, ::Reflex::Type typenum)
{
   // -- Declaration of variable, function or ANSI function header
   //
   // Note: This function is part of the parser proper.
   //
   // variable:   type  varname1, varname2=initval ;
   //                 ^
   // function:   type  funcname(param decl) { body }
   //                 ^
   // ANSI function header: funcname(  type para1, type para2,...)
   //                                ^     or     ^
   //
   // Note: overrides global variables
   //
   char var_type = '\0';
   int cin = '\0';
   int store_decl = 0;
   int largestep = 0;
   ::Reflex::Scope store_tagnum;
   ::Reflex::Type store_typenum;
   int store_def_struct_member = 0;
   ::Reflex::Scope store_def_tagnum;
   int i = 0;
   int p_inc = 0;
   char* index = 0;
   int initary = 0;
   int known = 0;
   char* store_struct_offset = 0;
   int store_prerun = 0;
   int store_debug = 0;
   int store_step = 0;
   int staticclassobject = 0;
   int store_var_type = 0;
   ::Reflex::Scope store_tagnum_default;
   int store_def_struct_member_default = 0;
   int store_exec_memberfunc = 0;
   ::Reflex::Scope store_memberfunc_tagnum;
   int store_constvar = 0;
   int store_static_alloc = 0;
   ::Reflex::Scope store_tagdefining;
   int store_line = 0;
   int store_static_alloc2 = 0;
   static int padn = 0;
   static int bitfieldwarn = 0;
   fpos_t store_fpos;
   G__value reg = G__null;
   G__StrBuf temp1_sb(G__LONGLINE);
   char *temp1 = temp1_sb;
   G__StrBuf new_name_sb(G__LONGLINE);
   char *new_name = new_name_sb;
   G__StrBuf temp_sb(G__LONGLINE);
   char *temp = temp_sb;
   store_static_alloc2 = G__static_alloc;
   new_name[0] = '\0';
   store_tagnum = G__tagnum;
   store_typenum = G__typenum;
   G__tagnum = G__Dict::GetDict().GetScope(tagnum);
   G__typenum = typenum;
   store_decl = G__decl;
   G__decl = 1;
   Reflex::Member newMember;

   //fprintf(stderr, "\nG__define_var: Begin.\n");
   //
   // We have:
   //
   // type var1, var2;
   //     ^
   //
   // or:
   //
   // type int var1, var2;
   //     ^
   //
   // Read variable name.
   //
   cin = G__get_newname(new_name);
   //fprintf(stderr, "G__define_var: G__getnewname returned: '%s'\n", new_name);
   G__unsigned = 0;
   if (!cin) {
      // -- long long handling, and return.
      G__decl = store_decl;
      G__constvar = 0;
      G__tagnum = store_tagnum;
      G__typenum = store_typenum;
      G__reftype = G__PARANORMAL;
      G__static_alloc = store_static_alloc2;
      G__dynconst = 0;
      G__globalvarpointer = G__PVOID;
      return;
   }
   var_type = G__var_type;
   if (new_name[0] == '&') {
      G__reftype = G__PARAREFERENCE;
      strcpy(temp, new_name + 1);
      strcpy(new_name, temp);
   }
   else if ((new_name[0] == '*') && (new_name[1] == '&')) {
      G__reftype = G__PARAREFERENCE;
      sprintf(temp, "*%s", new_name + 2);
      strcpy(new_name, temp);
   }
   //
   //  Now we have:
   //
   //  type var1, var2;
   //            ^
   //  or:
   //
   //  type var1 = initval, var2;
   //             ^
   //  or:
   //
   //  type var1 : N, var2;
   //             ^
   //  or:
   //
   //  type int var1, var2;
   //                ^
   //  or:
   //
   //  type var1(val1), var2(val2);
   //           ^
   //
   while (1) {
      // -- Loop over declarator list.
      if (G__ansiheader) {
         // -- We are parsing ANSI function parameters, handle one parameter and return.
         //
         //   funcname(type var1  , type var2,...)
         //                      ^    or         ^
         //   funcname(type var1= 5 , type var2,...)
         //                      ^    or         ^
         //  return one by one
         //
         //fprintf(stderr, "G__define_var: Parsing an ansi function parameter: '%s'\n", new_name);
         char *pxx = strstr(new_name, "...");
         if (pxx) {
            *pxx = 0;
         }
#ifdef G__ASM
         if (G__asm_noverflow && G__asm_wholefunction) {
            // -- We are generating bytecode for an entire function.
            char* p = strchr(new_name, '[');
            if (p) {
               // -- The parameter is an array.
               char* p2 = strchr(p + 1, '[');
               if (p2) {
                  // -- We have more than one dimension.
                  // For example: f(T a[][10])
                  // FIXME: We cannot handle this in bytecode yet, stop generating code.
                  G__abortbytecode();
               }
               else if (*(++p) != ']') {
                  // -- The array has bounds, change to an unspecified-length array.
                  // -- f(T a[10]) -> f(T a[])
                  // FIXME: We do this because array initializers are pointers, G__value needs to be fixed to support array types.
                  *(p++) = ']';
                  *p = 0;
               }
            }
         }
#endif // G__ASM
         if (cin == '(') {
            //fprintf(stderr, "G__define_var: func parm, I see a '(': '%s'\n", new_name);
            if ((new_name[0] == '\0') || !strcmp(new_name, "*")) {
               // pointer to function
               //   type (*funcpointer[n])(type var1, ...)
               //        ^
               //fprintf(stderr, "G__define_var: func parm, calling G__readpointer2function.   new_name: '%s'\n", new_name);
               G__readpointer2function(new_name, &var_type);
               //fprintf(stderr, "G__define_var: func parm, finished G__readpointer2function.  new_name: '%s' var_type: '%c'\n", new_name);
               //fprintf(stderr, "G__define_var: func parm, ignoring up to next ',)='\n");
               cin = G__fignorestream(",)=");
               //fprintf(stderr, "G__define_var: func parm, ignoring stopped at: '%c'\n", (char) cin);
            }
         }
         //
         //  If there is a default parameter, read it.
         //
         if (cin != '=') {
            temp[0] = '\0';
         }
         else {
            cin = G__fgetstream(temp, ",)");
            store_var_type = G__var_type;
            G__var_type = 'p';
            if (G__def_tagnum && !G__def_tagnum.IsTopScope()) {
               store_tagnum_default = G__tagnum;
               G__tagnum = G__def_tagnum;
               store_def_struct_member_default = G__def_struct_member;
               store_exec_memberfunc = G__exec_memberfunc;
               store_memberfunc_tagnum = G__memberfunc_tagnum;
               G__memberfunc_tagnum = G__tagnum;
               G__exec_memberfunc = 1;
               G__def_struct_member = 0;
            }
            else if (G__exec_memberfunc) {
               store_tagnum_default = G__tagnum;
               G__tagnum = store_tagnum;
               store_def_struct_member_default = G__def_struct_member;
               store_exec_memberfunc = G__exec_memberfunc;
               store_memberfunc_tagnum = G__memberfunc_tagnum;
               G__memberfunc_tagnum = G__tagnum;
               G__exec_memberfunc = 1;
               G__def_struct_member = 0;
            }
            else {
               store_exec_memberfunc = 0;
            }
            strcpy(G__def_parameter, temp);
            G__default_parameter = G__getexpr(temp);
            if (!G__value_typenum(G__default_parameter)) {
               G__default_parameter.ref = G__int(G__strip_quotation(temp));
            }
            if ((G__def_tagnum && !G__def_tagnum.IsTopScope()) || store_exec_memberfunc) {
               G__tagnum = store_tagnum_default;
               G__exec_memberfunc = store_exec_memberfunc;
               G__def_struct_member = store_def_struct_member_default;
               G__memberfunc_tagnum = store_memberfunc_tagnum;
            }
            G__var_type = store_var_type;
         }
         if (G__reftype == G__PARAREFERENCE) {
            G__globalvarpointer = (char*) G__ansipara.ref;
            reg = G__null;
            if (
               !G__globalvarpointer &&
               (G__get_type(G__value_typenum(G__ansipara)) == 'u') &&
               (!G__prerun && !G__no_exec_compile)
            ) {
               G__referencetypeerror(new_name);
            }
         }
         else {
            // -- Set default value if parameter is omitted.
            if (G__get_type(G__value_typenum(G__ansipara))) {
               reg = G__ansipara;
            }
            else {
               // -- This case is not needed after changing default parameter handling.
               store_var_type = G__var_type;
               G__var_type = 'p';
               if (G__def_tagnum && !G__def_tagnum.IsTopScope()) {
                  store_tagnum_default = G__tagnum;
                  G__tagnum = G__def_tagnum;
                  store_def_struct_member_default = G__def_struct_member;
                  store_exec_memberfunc = G__exec_memberfunc;
                  store_memberfunc_tagnum = G__memberfunc_tagnum;
                  G__memberfunc_tagnum = G__tagnum;
                  G__exec_memberfunc = 1;
                  G__def_struct_member = 0;
               }
               else if (G__exec_memberfunc) {
                  store_tagnum_default = G__tagnum;
                  G__tagnum = store_tagnum;
                  store_def_struct_member_default = G__def_struct_member;
                  store_exec_memberfunc = G__exec_memberfunc;
                  store_memberfunc_tagnum = G__memberfunc_tagnum;
                  G__memberfunc_tagnum = G__tagnum;
                  G__exec_memberfunc = 1;
                  G__def_struct_member = 0;
               }
               else {
                  store_exec_memberfunc = 0;
               }
               reg = G__getexpr(temp);
               if ((G__def_tagnum && !G__def_tagnum.IsTopScope()) || store_exec_memberfunc) {
                  G__tagnum = store_tagnum_default;
                  G__exec_memberfunc = store_exec_memberfunc;
                  G__def_struct_member = store_def_struct_member_default;
                  G__memberfunc_tagnum = store_memberfunc_tagnum;
               }
               G__var_type = store_var_type;
            }
         }
         G__var_type = var_type;
         //
         //  Initialization of formal parameter.
         //
         Reflex::Member paramvar;
         if (
            // -- Parameter is a struct, not a pointer nor an array.  (FIXME: Should check for reference?)
            (G__var_type == 'u') &&
            (G__reftype == G__PARANORMAL) &&
            (new_name[0] != '*') &&
            !strstr(new_name, "[]")
         ) {
            // -- Parameter is a struct, not a pointer nor an array.  (FIXME: Should check for reference?)
            G__ansiheader = 0;
            if (G__struct.iscpplink[tagnum] == G__CPPLINK) {
               // -- The struct is compiled code.
               G__StrBuf tttt_sb(G__ONELINE);
               char *tttt = tttt_sb;
               G__valuemonitor(reg, tttt);
               sprintf(temp1, "%s(%s)", G__struct.name[tagnum], tttt);
               if (G__struct.parent_tagnum[tagnum] != -1) {
                  int local_store_exec_memberfunc = G__exec_memberfunc;
                  ::Reflex::Scope local_store_memberfunc_tagnum = G__memberfunc_tagnum;
                  G__exec_memberfunc = 1;
                  G__memberfunc_tagnum = G__Dict::GetDict().GetScope(G__struct.parent_tagnum[tagnum]);
                  reg = G__getfunction(temp1, &known, G__CALLCONSTRUCTOR);
                  G__exec_memberfunc = local_store_exec_memberfunc;
                  G__memberfunc_tagnum = local_store_memberfunc_tagnum;
               }
               else {
                  reg = G__getfunction(temp1, &known, G__CALLCONSTRUCTOR);
               }
               G__globalvarpointer = (char*) G__int(reg);
               G__cppconstruct = 1;
               G__letvariable(new_name, G__null, ::Reflex::Scope::GlobalScope(), G__p_local, paramvar);
               G__cppconstruct = 0;
               G__globalvarpointer = G__PVOID;
            }
            else {
               // -- The struct is interpreted.
               // Create object.
               G__letvariable(new_name, G__null, ::Reflex::Scope::GlobalScope(), G__p_local, paramvar);
               // And initialize it.
               G__letvariable(new_name, reg, ::Reflex::Scope::GlobalScope(), G__p_local, paramvar);
            }
         }
         else {
            // -- The parameter is of fundamental type.
            G__letvariable(new_name, reg, ::Reflex::Scope::GlobalScope(), G__p_local, paramvar);
         }
         G__ansiheader = 1;
         G__globalvarpointer = G__PVOID;
 
#ifdef G__ASM
         if (!paramvar) { // Case of unnamed paramerter
            // --
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "%3x,%3x: POP  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
            }
#endif // G__ASM_DBG
            G__asm_inst[G__asm_cp] = G__POP;
            G__inc_cp_asm(1, 0);
#endif // G__ASM
            // --
         }
         if (cin == ')') {
            // -- End of ANSI parameter header.
            // funcname(type var1 , type var2,...)
            //                                   ^
            G__ansiheader = 0;
         }
         G__decl = store_decl;
         G__constvar = 0;
         G__tagnum = store_tagnum;
         G__typenum = store_typenum;
         G__reftype = G__PARANORMAL;
         G__static_alloc = store_static_alloc2;
         G__dynconst = 0;
         G__globalvarpointer = G__PVOID;
         return;
      }
      if (cin == '(') {
         // -- Function, pointer to function, or function style initializer.
         //
         //   type funcname( type var1,.....)
         //                 ^
         //            or
         //   type ( *funcpointer)(type var1,...)
         //         ^
         // This part should be called only at pre-run. (Used to be true)
         // C++:
         //   type obj(const,const);
         // is used to give constant parameter to constructor.
         //
         //fprintf(stderr, "G__define_var: I see a '(': '%s'\n", new_name);
         if (!new_name[0] || !strcmp(new_name, "*")) {
            // -- Pointer to function.
            //
            //   type ( *funcpointer[n])( type var1,.....)
            //         ^
            //fprintf(stderr, "G__define_var: Calling G__readpointer2function: '%s'\n", new_name);
            switch (G__readpointer2function(new_name, &var_type)) {
               case G__POINTER2FUNC:
                  break;
               case G__FUNCRETURNP2F:
                  G__isfuncreturnp2f = 1;
                  goto define_function;
               case G__POINTER2MEMFUNC:
                  break;
               case G__CONSTRUCTORFUNC:
                  if (G__tagnum && !G__tagnum.IsTopScope()) {
                     cin = '(';
                     strcpy(new_name, G__struct.name[G__get_tagnum(G__tagnum)]);
                     G__var_type = 'i';
                     goto define_function;
                  }
            }
            // Initialization of pointer to function.
            // CAUTION: Now, I don't do this.
            //   G__var_type = 'q';
            // Thus, type of function pointer is declared type
            cin = G__fignorestream("=,;}");
            G__constvar = 0;
            G__reftype = G__PARANORMAL;
         }
         else {
            // -- Function or function style initializer.
            //
            // We have either a function declaration:
            //
            //      type funcname(type param1, ...)
            //                   ^
            // or a function style initializer:
            //
            //      type varname(expr, ...);
            //                  ^
            // --
            //
            //  Figure out which case we have.
            //
            define_function:
            //
            //  Read next non-whitespace character,
            //  and backup one character.
            //
            cin = G__fgetspace();
            fseek(G__ifile.fp, -1, SEEK_CUR);
            if (cin == '\n') {
               --G__ifile.line_number;
            }
            if (G__dispsource) {
               G__disp_mask = 1;
            }
            //
            //  If we are declaring a class member, we must have a function declaration,
            //  because initializers are not allowed for class members.
            //  FIXME: Initializers are allowed for static class members of integral type.
            //  Otherwise, if the next character could be part of a type name, do further
            //  checking.
            //
            if (
               (
                  G__def_struct_member &&
                  (
                     !G__tagdefining ||
                     (G__tagdefining.IsTopScope() || !G__tagdefining.IsNamespace())
                  )
               ) ||
               !strchr("0123456789!\"%&'(*+-./<=>?[]^|~", cin)
            ) {
               // -- It is clear that above check is not sufficient to distinguish
               // class object instantiation and function header.  Following
               // code is added to make it fully compliant to ANSI C++.
               fgetpos(G__ifile.fp, &store_fpos);
               store_line = G__ifile.line_number;
               if (G__dispsource) {
                  G__disp_mask = 1000;
               }
               cin = G__fgetname(temp, ",)!\"%&'*+-./<=>?[]^|~"); // FIXME: No '(' here because then given "B f(A(3, 5), 12)" the "A" passes the typename test and we parse it as a function declaration.
               if (strlen(temp) && isspace(cin)) {
                  // -- There was an argument and the parsing was stopped by a white
                  // space rather than on of ",)*&<=", it is possible that
                  // we have a namespace followed by '::' in which case we have
                  // to grab more before stopping!
                  int namespace_tagnum;
                  G__StrBuf more_sb(G__LONGLINE);
                  char *more = more_sb;
                  namespace_tagnum = G__defined_tagname(temp, 2);
                  while (
                     isspace(cin) &&
                     (
                        ((namespace_tagnum != -1) && (G__struct.type[namespace_tagnum] == 'n')) ||
                        !strcmp("std", temp) ||
                        (temp[strlen(temp)-1] == ':')
                     )
                  ) {
                     cin = G__fgetname(temp, ",)!\"%&'*+-./<=>?[]^|~"); // FIXME: No '(' here because then given "B f(A(3, 5), 12)" the "A" passes the typename test and we parse it as a function declaration.
                     strcat(temp, more);
                     namespace_tagnum = G__defined_tagname(temp, 2);
                  }
               }
               fsetpos(G__ifile.fp, &store_fpos);
               if (G__dispsource) {
                  G__disp_mask = 1;
               }
               G__ifile.line_number = store_line;
               if (
                  !G__iscpp || // Not C++, cannot be function-style initializer.
                  !temp[0] || // Empty parameter, not an initializer.
                  G__istypename(temp) ||  // First identifier is a typename, must be declaration.
                  (!temp[0] && (cin == ')')) || // Empty parameter list, not an initializer.
                  !strncmp(new_name, "operator", 8) || // The syntax operator+() cannot be an initializer.
                  ((cin == '<') && G__defined_templateclass(temp)) // First identifier is a template id, must be decl.
               ) {
                  // -- Handle a function declaration and return.
                  //fprintf(stderr, "G__define_var: Handle a function declaration: '%s('\n", new_name);
                  G__var_type = var_type;
                  // function definition
                  //   type funcname( type var1,.....)
                  //                  ^
                  sprintf(temp, "%s(", new_name);
                  G__make_ifunctable(temp);
                  G__isfuncreturnp2f = 0; // this is set above in this function
                  // body of the function is skipped all
                  // the way
                  //   type funcname(type var1,..) {....}
                  //                                     ^
                  G__decl = store_decl;
                  G__constvar = 0;
                  G__tagnum = store_tagnum;
                  G__typenum = store_typenum;
                  G__reftype = G__PARANORMAL;
                  G__static_alloc = store_static_alloc2;
                  G__dynconst = 0;
                  G__globalvarpointer = G__PVOID;
                  return;
               }
               G__var_type = var_type;
            }
            // If didn't meet above conditions, this is a constructor call.
            //   type varname(const, const);
            //                ^
            // Read parameter list and build command string.
            cin = G__fgetstream_newtemplate(temp, ")");
            if ((new_name[0] == '*') && (var_type != 'c') && (temp[0] == '"')) {
               G__genericerror("Error: illegal pointer initialization");
            }
            if (G__static_alloc && !G__prerun) {
               // -- This is a static or const variable, and we are executing, not parsing.
               // Skip to the next comma or semicolon.
               if ((cin != ',') && (cin != ';')) {
                  cin = G__fignorestream(",;");
               }
               if (cin == '{') { // Don't know if this part is needed.
                  while (cin != '}') {
                     cin = G__fignorestream(",;");
                  }
               }
               // Perform the initialization.
               G__var_type = var_type;
               G__value temp_reg = G__null;
               G__letvariable(new_name, temp_reg, ::Reflex::Scope::GlobalScope(), G__p_local);
               // Continue scanning.
               goto readnext;
            }
            if (
               (!G__tagnum || G__tagnum.IsTopScope()) ||
               (var_type != 'u') ||
               (new_name[0] == '*')
            ) {
               if ((tolower(G__var_type) != 'c') && strchr(temp, ',')) {
                  reg = G__null;
                  G__genericerror("Error: G__define_var:2167 Syntax error");
               }
               else {
                  reg = G__getexpr(temp);
               }
               cin = G__fignorestream(",;");
               if ((G__reftype == G__PARAREFERENCE) && !G__asm_wholefunction) {
                  if (!reg.ref) {
                     G__fprinterr(G__serr, "Error: reference type %s with no initialization ", new_name);
                     G__genericerror(0);
                  }
                  G__globalvarpointer = (char*) reg.ref;
               }
               goto create_body;
            }
            sprintf(temp1, "%s(%s)", G__struct.name[G__get_tagnum(G__tagnum)], temp);
            // Store flags.
            store_prerun = G__prerun;
            G__prerun = 0;
            if (store_prerun) {
               store_debug = G__debug;
               store_step = G__step;
               G__debug = G__debugtrace;
               G__step = G__steptrace;
               G__setdebugcond();
            }
            else {
               if (G__breaksignal) {
                  G__break = 0;
                  G__setdebugcond();
                  if (G__pause() == 3) {
                     if (G__return == G__RETURN_NON) {
                        G__step = 0;
                        G__setdebugcond();
                        largestep = 1;
                     }
                  }
                  if (G__return > G__RETURN_NORMAL) {
                     G__decl = store_decl;
                     G__constvar = 0;
                     G__tagnum = store_tagnum;
                     G__typenum = store_typenum;
                     G__reftype = G__PARANORMAL;
                     G__static_alloc = store_static_alloc2;
                     G__dynconst = 0;
                     G__globalvarpointer = G__PVOID;
                     return;
                  }
               }
            }
            // skip until comma or semicolon
            cin = G__fgetspace(); // G__fignorestream(",;");
            if (cin != ',' && cin != ';') {
               if (!G__xrefflag) {
                  G__fprinterr(G__serr, "Error: expected ‘,’ or ‘;’ before ‘%c’ ", cin);
               }
               G__genericerror(0);
            }
            
            //   type varname( const,const) , ;
            //                               ^
            // allocate memory area
            G__var_type = var_type;
            store_struct_offset = G__store_struct_offset ;
            if (G__struct.iscpplink[tagnum] != G__CPPLINK) {
               G__prerun = store_prerun;
               G__value val = G__letvariable(new_name, G__null, ::Reflex::Scope::GlobalScope(), G__p_local);
               G__store_struct_offset = (char*) G__int(val);
               if (G__return > G__RETURN_NORMAL) {
                  G__decl = store_decl;
                  G__constvar = 0;
                  G__tagnum = store_tagnum;
                  G__typenum = store_typenum;
                  G__reftype = G__PARANORMAL;
                  G__static_alloc = store_static_alloc2;
                  G__dynconst = 0;
                  G__globalvarpointer = G__PVOID;
                  return;
               }
               G__prerun = 0;
#ifndef G__OLDIMPLEMENTATION1073
               if (!G__store_struct_offset && G__asm_wholefunction && G__asm_noverflow) {
                  G__store_struct_offset = G__PVOID;
               }
#endif // G__OLDIMPLEMENTATION1073
               // --
            }
            else {
               G__store_struct_offset = G__PVOID;
            }
            if (G__dispsource) {
               G__fprinterr(G__serr, "\n!!!Calling constructor 0x%lx.%s for declaration of %s  %s:%d", G__store_struct_offset, temp1, new_name, __FILE__, __LINE__);
            }
            // call constructor, error if no constructor.
            G__decl = 0;
            store_constvar = G__constvar;
            store_static_alloc = G__static_alloc;
            G__constvar = 0;
            G__static_alloc = 0;
            if (G__struct.iscpplink[tagnum] == G__CPPLINK) {
               // -- This is a precompiled class.
               // -- These has to be stored because G__getfunction can call the bytecode compiler.
               ::Reflex::Scope bc_tagnum = G__tagnum;
               ::Reflex::Type bc_typenum = G__typenum;
               reg = G__getfunction(temp1, &known, G__CALLCONSTRUCTOR);
               G__tagnum = bc_tagnum;
               G__typenum = bc_typenum;
               G__var_type = var_type;
               G__globalvarpointer = (char*) G__int(reg);
               G__static_alloc = store_static_alloc;
               G__prerun = store_prerun;
               G__cppconstruct = 1;
               if (G__globalvarpointer || G__no_exec_compile) {
                  int store_constvar2 = G__constvar;
                  G__constvar = store_constvar;
                  G__letvariable(new_name, G__null,::Reflex::Scope::GlobalScope(), G__p_local);
                  G__constvar = store_constvar2;
               }
               else if (G__asm_wholefunction) {
                  G__abortbytecode();
                  G__asm_wholefunc_default_cp = 0;
                  // FIXME: Should we be turning on code skipping here?
                  G__no_exec = 1;
                  G__return = G__RETURN_NORMAL;
               }
               G__cppconstruct = 0;
               G__globalvarpointer = G__PVOID;
#ifndef G__OLDIMPLEMENTATION1073
               if (G__asm_wholefunction && G__no_exec_compile) {
                  // --
#ifdef G__ASM_DBG
                  if (G__asm_dbg) {
                     G__fprinterr(G__serr, "%3x,%3x: SETGVP -1  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
                  }
#endif // G__ASM_DBG
                  G__asm_inst[G__asm_cp] = G__SETGVP;
                  G__asm_inst[G__asm_cp+1] = -1;
                  G__inc_cp_asm(2, 0);
               }
#endif // G__OLDIMPLEMENTATION1073
               // --
            }
            else {
               if (G__store_struct_offset) {
                  G__getfunction(temp1, &known, G__CALLCONSTRUCTOR);
#ifndef G__OLDIMPLEMENTATION1073
                  if (G__asm_wholefunction && G__asm_noverflow) {
                     // --
#ifdef G__ASM_DBG
                     if (G__asm_dbg) {
                        G__fprinterr(G__serr, "%3x,%3x: POPSTROS  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
                     }
#endif // G__ASM_DBG
                     G__asm_inst[G__asm_cp] = G__POPSTROS;
                     G__inc_cp_asm(1, 0);
                  }
#endif // G__OLDIMPLEMENTATION1073
                  // --
               }
               else {
                  // -- Temporary solution, later this must be deleted.
                  if ((G__asm_wholefunction == G__ASM_FUNC_NOP) || G__asm_noverflow) {
                     if (!G__xrefflag) {
                        G__fprinterr(G__serr, "Error: %s not allocated(1), maybe duplicate declaration ", new_name);
                     }
                     G__genericerror(0);
                  }
               }
            }
            G__constvar = store_constvar;
            G__static_alloc = store_static_alloc;
            G__decl = 1;
            if (G__return > G__RETURN_NORMAL) {
               G__decl = store_decl;
               G__constvar = 0;
               G__tagnum = store_tagnum;
               G__typenum = store_typenum;
               G__reftype = G__PARANORMAL;
               G__static_alloc = store_static_alloc2;
               G__dynconst = 0;
               G__globalvarpointer = G__PVOID;
               return;
            }
            if (largestep) {
               G__step = 1;
               G__setdebugcond();
               largestep = 0;
            }
            // restore flags
            if (store_prerun) {
               G__debug = store_debug;
               G__step = store_step;
               G__setdebugcond();
            }
            G__prerun = store_prerun;
            G__store_struct_offset = store_struct_offset;
            // To skip following condition.
            new_name[0] = '\0';
         }
      }
      if (cin == ':') {
         // -- Ignore bit-field declaration or we have a qualified name in function call.
         //
         //   unsigned int  var1  :  2  ;
         //                        ^
         // or
         //   returntype X::func()
         //                 ^
         //
         cin = G__fgetc();
         //
         // Memberfunction definition.
         //
         //   type X::func()
         //          ^
         if (cin == ':') {
            store_def_struct_member = G__def_struct_member;
            G__def_struct_member = 1;
            store_def_tagnum = G__def_tagnum;
            store_tagdefining = G__tagdefining;
            i = 0;
            while ('*' == new_name[i]) ++i;
            if (i) {
               var_type = toupper(var_type);
               /* if(i>1) G__reftype = i+1;  not needed */
            }
            if (strchr(new_name + i, '<')) {
               G__removespacetemplate(new_name + i);
            }
            do {
               G__def_tagnum = G__Dict::GetDict().GetScope(G__defined_tagname(new_name + i, 0));
               /* protect against a non defined tagname */
               if (!G__def_tagnum || G__def_tagnum.IsTopScope()) {
                  /* Hopefully restore all values! */
                  G__decl = store_decl;
                  G__constvar = 0;
                  G__tagnum = store_tagnum;
                  G__typenum = store_typenum;
                  G__reftype = G__PARANORMAL;
                  G__static_alloc = store_static_alloc2;
                  G__dynconst = 0;
                  G__globalvarpointer = G__PVOID;
                  G__def_struct_member = store_def_struct_member;
                  return;
               }
               G__tagdefining  = G__def_tagnum;
               cin = G__fgetstream(new_name + i, "(=;:");
            }
            while (':' == cin && EOF != (cin = G__fgetc()));
            temp[0] = '\0';
            switch (cin) {
               case '=':
                  if (strncmp(new_name + i, "operator", 8) == 0) {
                     cin = G__fgetstream(new_name + strlen(new_name) + 1, "(");
                     new_name[strlen(new_name)] = '=';
                     break;
                  }
               case ';':
                  /* PHILIPPE17: the following is fixed in 1306! */
                  /* static class object member must call constructor
                   * TO BE IMPLEMENTED */
                  sprintf(temp, "%s::%s", G__fulltagname(G__get_tagnum(G__def_tagnum), 1), new_name + i);
                  strcpy(new_name, temp);
                  if ('u' != var_type || G__reftype) var_type = 'p';
                  else staticclassobject = 1;
                  G__def_struct_member = store_def_struct_member;
                  G__tagnum = ::Reflex::Scope::GlobalScope(); /*do this to pass letvariable scopeoperator()*/
                  G__def_tagnum = store_def_tagnum;
                  G__tagdefining  = store_tagdefining;
                  continue; /* big while(1) loop */
                  /* If neither case, handle as member function definition
                   * It is possible that this is initialization of class object as
                   * static member, like 'type X::obj(1,2)' . This syntax is not
                   * handled correctly. */
            }
            if (strcmp(new_name + i, "operator") == 0) {
               sprintf(temp, "%s()(", new_name);
               cin = G__fignorestream(")");
               cin = G__fignorestream("(");
            }
            else {
               sprintf(temp, "%s(", new_name);
            }
            G__make_ifunctable(temp);
            G__def_struct_member = store_def_struct_member;
            G__def_tagnum = store_def_tagnum;
            G__decl = store_decl;
            G__constvar = 0;
            G__tagnum = store_tagnum;
            G__typenum = store_typenum;
            G__reftype = G__PARANORMAL;
            G__static_alloc = store_static_alloc2;
            G__tagdefining = store_tagdefining;
            G__dynconst = 0;
            G__globalvarpointer = G__PVOID;
            return;
         }
         else {
            fseek(G__ifile.fp, -1, SEEK_CUR);
            if (cin == '\n') {
               --G__ifile.line_number;
            }
            if (G__dispsource) {
               G__disp_mask = 1;
            }
         }
         if (G__globalcomp != G__NOLINK) {
            if (!bitfieldwarn) {
               if (G__dispmsg >= G__DISPNOTE) {
                  G__fprinterr(G__serr, "Note: Bit-field not accessible from interpreter");
                  G__printlinenum();
               }
               bitfieldwarn = 1;
            }
            cin = G__fgetstream(temp, ",;=}");
            sprintf(new_name, "%s : %s", new_name, temp);
            G__bitfield = 1;
         }
         else {
            cin = G__fgetstream(temp, ",;=}");
            G__bitfield = atoi(temp);
            if (!G__bitfield) {
               G__bitfield = -1;
            }
            if (!new_name[0]) {
               sprintf(new_name, "G__pad%x", padn++);
            }
         }
      }
      temp[0] = '\0';
      if (cin == '=') {
         // -- Read initializer.
         //
         //  type var1 = initval , ...
         //             ^
         //  set reg = G__getexpr("initval");
         //
         ::Reflex::Scope store_tagnumB = G__tagnum;
         G__tagnum = G__get_envtagnum();
         // Scan the initializer into temp.
         if ((var_type == 'u')) {
            cin = G__fgetstream_newtemplate(temp, ",;{}");
         }
         else {
            cin = G__fgetstream_new(temp, ",;{");
         }
         if (
            G__def_struct_member &&
            (G__tagdefining  && !G__tagdefining.IsTopScope()) &&
            (
               (G__struct.type[G__get_tagnum(G__tagdefining)] == 'c') ||
               (G__struct.type[G__get_tagnum(G__tagdefining)] == 's')
            ) &&
            G__static_alloc &&
            (G__constvar != G__CONSTVAR)
         ) {
            // -- Semantic error, in-class intialization of a non-const static member.
            if (G__dispmsg >= G__DISPWARN) {
               G__fprinterr(G__serr, "Warning: In-class initialization of non-const static member not allowed in C++ standard");
               G__printlinenum();
            }
         }
         //
         // ignore array and struct initialization
         //  type var1[N] = { 0, 1, 2.... }
         //                  ^
         //
         if (cin == '{') {
            initary = 1;
            // reg=G__getexpr(temp); is going to be G__null because temp is ""
         }
         G__var_type = 'p';
         if (G__reftype == G__PARAREFERENCE) {
            int local_store_reftype = G__reftype;
            int local_store_prerun = G__prerun;
            int local_store_decl = G__decl;
            int local_store_constvar = G__constvar;
            int local_store_static_alloc = G__static_alloc;
            if (G__globalcomp == G__NOLINK) {
               G__prerun = 0;
               G__decl = 0;
               if (G__constvar & G__CONSTVAR) {
                  G__initval_eval = 1;
               }
               G__constvar = 0;
               G__static_alloc = 0;
            }
            --G__templevel;
            G__reftype = G__PARANORMAL;
            if (local_store_prerun || !local_store_static_alloc || G__IsInMacro()) {
               reg = G__getexpr(temp);
            }
            else {
               reg = G__null;
            }
            ++G__templevel;
            G__prerun = local_store_prerun;
            G__decl = local_store_decl;
            G__constvar = local_store_constvar;
            G__static_alloc = local_store_static_alloc;
            G__initval_eval = 0;
            G__reftype = local_store_reftype;
            G__globalvarpointer = (char*) reg.ref;
            reg = G__null;
            if (
               !G__globalvarpointer &&
               (G__get_type(G__value_typenum(G__ansipara)) == 'u') &&
               (!G__prerun && !G__no_exec_compile)
            ) {
               G__referencetypeerror(new_name);
            }
         }
         else {
            if ((var_type == 'u') && !G__def_struct_member && (new_name[0] != '*')) {
               // -- If struct or class, handled later with constructor.
               reg = G__null;
               // Avoiding assignment, ignored in G__letvariable when reg==G__null
               if (staticclassobject) {
                  reg = G__one;
               }
#ifdef G__OLDIMPLEMENTATION1032_YET
               if (!strncmp(temp, "new ", 4)) {
                  G__assign_error(new_name, &G__null);
               }
#endif // G__OLDIMPLEMENTATION1032_YET
               // --
            }
            else if ((var_type == 'u') && (new_name[0] == '*') && !strncmp(temp, "new ", 4)) {
               // --
               int local_store_prerun = G__prerun;
               int local_store_decl = G__decl;
               int local_store_constvar = G__constvar;
               int local_store_static_alloc = G__static_alloc;
               if (G__globalcomp == G__NOLINK) {
                  G__prerun = 0;
                  G__decl = 0;
                  if (G__constvar & G__CONSTVAR) {
                     G__initval_eval = 1;
                  }
                  G__constvar = 0;
                  G__static_alloc = 0;
               }
               if (local_store_prerun || !local_store_static_alloc || G__IsInMacro()) {
                  reg = G__getexpr(temp);
               }
               else {
                  reg = G__null;
               }
               G__prerun = local_store_prerun;
               G__decl = local_store_decl;
               G__constvar = local_store_constvar;
               G__static_alloc = local_store_static_alloc;
               G__initval_eval = 0;
               if (
                  (G__get_type(G__value_typenum(reg)) != 'U') &&
                  (G__get_type(G__value_typenum(reg)) != 'Y') &&
                  reg.obj.i
               ) {
                  G__assign_error(new_name + 1, &reg);
                  reg = G__null;
               }
            }
            else {
               // --
               int local_store_prerun = G__prerun;
               int local_store_decl = G__decl;
               int local_store_constvar = G__constvar;
               int local_store_static_alloc = G__static_alloc;
               if (G__globalcomp == G__NOLINK) {
                  G__prerun = 0;
                  G__decl = 0;
                  if (G__constvar & G__CONSTVAR) {
                     G__initval_eval = 1;
                  }
                  G__constvar = 0;
                  G__static_alloc = 0;
               }
               if (local_store_prerun || !local_store_static_alloc || G__IsInMacro()) {
                  ::Reflex::Scope local_store_tagdefiningC = G__tagdefining;
                  int local_store_eval_localstatic = G__eval_localstatic;
                  G__eval_localstatic = 1;
                  // Evaluate the initializer expression.
                  reg = G__getexpr(temp);
                  G__eval_localstatic = local_store_eval_localstatic;
                  G__tagdefining = local_store_tagdefiningC;
               }
               else {
                  reg = G__null;
               }
               G__prerun = local_store_prerun;
               G__decl = local_store_decl;
               G__constvar = local_store_constvar;
               G__static_alloc = local_store_static_alloc;
               G__initval_eval = 0;
               if (
                  (var_type == 'u') &&
                  (new_name[0] == '*') &&
                  (G__get_type(G__value_typenum(reg)) != 'U') &&
                  reg.obj.i &&
                  (G__get_type(G__value_typenum(reg)) != 'Y')
               ) {
                  G__assign_error(new_name + 1, &reg);
                  reg = G__null;
               }
            }
         }
         G__tagnum = store_tagnumB;
      }
      else {
         // -- There is no initializer, check if this is an error.
         if (
            new_name[0] &&
            (G__globalcomp  == G__NOLINK) &&
            (G__reftype == G__PARAREFERENCE) &&
            !G__def_struct_member
         ) {
            G__fprinterr(G__serr, "Error: reference type %s with no initialization ", new_name);
            G__genericerror(0);
         }
         reg = G__null;
      }
      create_body:
      if (new_name[0]) {
         G__var_type = var_type;
         if (
            // -- Struct type, not ptr, not ref, (not a mbr, or is a mbr of a namespace).
            (var_type == 'u') && // class, enum, namespace, struct, or union, and
            (new_name[0] != '*') && // not a pointer, and
            (G__reftype == G__PARANORMAL) && // not a reference, and
            (
               !G__def_struct_member || // not a member, or
               (!G__def_tagnum || G__def_tagnum.IsTopScope()) || // FIXME: This is probably meant to protect the next check, it cannot happen.
               (!G__def_tagnum.IsTopScope() && G__struct.type[G__get_tagnum(G__def_tagnum)] == 'n') // is a member of a namespace
            )
         ) {
            // -- Declaration of struct object which is not a class member, and not a pointer, and not a reference.
            store_prerun = G__prerun;
            if (store_prerun) {
               store_debug = G__debug;
               store_step = G__step;
               G__debug = G__debugtrace;
               G__step = G__steptrace;
               G__prerun = 0;
               G__setdebugcond();
               G__prerun = store_prerun;
            }
            else {
               if (G__breaksignal) {
                  G__break = 0;
                  G__setdebugcond();
                  if (G__pause() == 3) {
                     if (G__return == G__RETURN_NON) {
                        G__step = 0;
                        G__setdebugcond();
                        largestep = 1;
                     }
                  }
                  if (G__return > G__RETURN_NORMAL) {
                     G__decl = store_decl;
                     G__constvar = 0;
                     G__tagnum = store_tagnum;
                     G__typenum = store_typenum;
                     G__reftype = G__PARANORMAL;
                     G__prerun = store_prerun;
                     G__static_alloc = store_static_alloc2;
                     G__dynconst = 0;
                     G__globalvarpointer = G__PVOID;
                     return;
                  }
               }
            }
            if (G__static_alloc && !G__prerun) {
               // -- Static or const variable, we are running not parsing.
               // Skip until the next comma or semicolon.
               // FIXME: Can this code block be right?
               if (cin == '{') {
                  while (cin != '}') {
                     cin = G__fignorestream(",;");
                  }
               }
               if ((cin != ',') && (cin != ';')) {
                  cin = G__fignorestream(",;");
               }
               // Perform the initialization.
               G__var_type = var_type;
               G__letvariable(new_name, reg, ::Reflex::Scope::GlobalScope(), G__p_local, newMember);
               // Continue scanning.
               goto readnext;
            }
            if (
               initary &&
               strchr(new_name, '[') &&
               (G__struct.funcs[G__get_tagnum(G__tagnum)] & G__HAS_CONSTRUCTOR)
            ) {
               store_prerun = G__prerun;
               if ((G__globalcomp == G__NOLINK) && !G__func_now) {
                  // -- We want to run the constructors.
                  G__prerun = 0;
               }
               G__initstructary(new_name, G__get_tagnum(G__tagnum));
               G__decl = store_decl;
               G__constvar = 0;
               G__tagnum = store_tagnum;
               G__typenum = store_typenum;
               G__reftype = G__PARANORMAL;
               G__static_alloc = store_static_alloc2;
               G__dynconst = 0;
               G__globalvarpointer = G__PVOID;
               G__prerun = store_prerun;
               return;
            }
            // Memory allocation and variable creation.
            store_struct_offset = G__store_struct_offset;
            if (G__struct.iscpplink[tagnum] != G__CPPLINK) {
               // -- Interpreted class, allocate memory now.
               G__var_type = var_type;
               G__decl_obj = 1;
               G__value val = G__letvariable(new_name, reg, ::Reflex::Scope::GlobalScope(), G__p_local, newMember);
               G__store_struct_offset = (char*) G__int(val);
               G__decl_obj = 0;
#ifndef G__OLDIMPLEMENTATION1073
               if (!G__store_struct_offset && G__asm_wholefunction && G__asm_noverflow) {
                  G__store_struct_offset = G__PVOID;
               }
#endif // G__OLDIMPLEMENTATION1073
               // --
            }
            else {
               // -- Precompiled class, memory will be allocated by new in constructor function below.
               G__store_struct_offset = G__PVOID;
            }
            if (G__return > G__RETURN_NORMAL) {
               G__decl = store_decl;
               G__constvar = 0;
               G__tagnum = store_tagnum;
               G__typenum = store_typenum;
               G__reftype = G__PARANORMAL;
               G__static_alloc = store_static_alloc2;
               G__dynconst = 0;
               G__globalvarpointer = G__PVOID;
               return;
            }
            // Flag that we want to actually run the constructor.
            G__prerun = 0;
            if (G__store_struct_offset) {
               // -- We have allocated memory for the object.
               if (
                  !temp[0] && // No initializer, and
                  (G__get_tagnum(G__tagnum) != -1) // var is of class type.
               ) {
                  // -- We need to call the default constructor.
                  //
                  // We have:
                  //
                  // type a;
                  //
                  sprintf(temp, "%s()", G__struct.name[G__get_tagnum(G__tagnum)]);
                  if (G__dispsource) {
                     G__fprinterr(G__serr, "\n!!!Calling default constructor for declaration of '%s'  addr: 0x%lx  funcname: '%s'  %s:%d", new_name, G__store_struct_offset, temp, __FILE__, __LINE__);
                  }
                  G__decl = 0;
                  if ((index = strchr(new_name, '['))) {
                     // -- Calling constructor for an array of objects.
                     p_inc = G__getarrayindex(index);
                     if (G__struct.iscpplink[tagnum] == G__CPPLINK) {
                        // -- Precompiled class. First, call constructor (new) function.
#ifdef G__ASM
                        if (G__asm_noverflow && (p_inc > 1)) {
                           // --
#ifdef G__ASM_DBG
                           if (G__asm_dbg) {
                              G__fprinterr(G__serr, "%3x,%3x: SETARYINDEX  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
                           }
#endif // G__ASM_DBG
                           G__asm_inst[G__asm_cp] = G__SETARYINDEX;
                           G__asm_inst[G__asm_cp+1] = 0;
                           G__inc_cp_asm(2, 0);
                        }
#endif // G__ASM
                        G__cpp_aryconstruct = p_inc;
                        reg = G__getfunction(temp, &known, G__CALLCONSTRUCTOR);
                        G__cpp_aryconstruct = 0;
                        // Register the pointer we get from new to member variable table.
                        G__globalvarpointer = (char*) G__int(reg);
                        G__cppconstruct = 1;
                        G__var_type = var_type;
                        G__letvariable(new_name, G__null,::Reflex::Scope::GlobalScope(), G__p_local, newMember);
                        G__cppconstruct = 0;
#ifdef G__ASM
                        if (G__asm_noverflow && (p_inc > 1)) {
                           // --
#ifdef G__ASM_DBG
                           if (G__asm_dbg) {
                              G__fprinterr(G__serr, "%3x,%3x: RESETARYINDEX  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
                           }
#endif // G__ASM_DBG
                           G__asm_inst[G__asm_cp] = G__RESETARYINDEX;
                           G__asm_inst[G__asm_cp+1] = 0;
                           G__inc_cp_asm(2, 0);
                        }
#endif // G__ASM
#ifndef G__OLDIMPLEMENTATION1073
                        if (G__asm_wholefunction && G__no_exec_compile) {
                           // --
#ifdef G__ASM_DBG
                           if (G__asm_dbg) {
                              G__fprinterr(G__serr, "%3x,%3x: SETGVP -1  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
                           }
#endif // G__ASM_DBG
                           G__asm_inst[G__asm_cp] = G__SETGVP;
                           G__asm_inst[G__asm_cp+1] = -1;
                           G__inc_cp_asm(2, 0);
                        }
#endif // G__OLDIMPLEMENTATION1073
                        G__globalvarpointer = G__PVOID;
                     }
                     else {
                        // -- Interpreted class, memory area was already allocated above.
                        for (i = 0; i < p_inc; ++i) {
                           if (G__struct.isctor[tagnum]) {
                              G__getfunction(temp, &known, G__CALLCONSTRUCTOR);
                           }
                           else {
                              G__getfunction(temp, &known, G__TRYCONSTRUCTOR);
                           }
                           if ((G__return > G__RETURN_NORMAL) || !known) {
                              break;
                           }
                           G__store_struct_offset += G__struct.size[G__get_tagnum(G__tagnum)];
                           if (G__asm_noverflow) {
                              // --
#ifdef G__ASM_DBG
                              if (G__asm_dbg) {
                                 G__fprinterr(G__serr, "%3x,%3x: ADDSTROS %d  %s:%d\n", G__asm_cp, G__asm_dt, G__struct.size[G__get_tagnum(G__tagnum)], __FILE__, __LINE__);
                              }
#endif // G__ASM_DBG
                              G__asm_inst[G__asm_cp] = G__ADDSTROS;
                              G__asm_inst[G__asm_cp+1] = G__struct.size[G__get_tagnum(G__tagnum)];
                              G__inc_cp_asm(2, 0);
                           }
#ifndef G__OLDIMPLEMENTATION1073
                           if (G__asm_wholefunction && G__asm_noverflow) {
                              // --
#ifdef G__ASM_DBG
                              if (G__asm_dbg) {
                                 G__fprinterr(G__serr, "%3x,%3x: ADDSTROS %d  %s:%d\n", G__asm_cp, G__asm_dt, G__struct.size[G__get_tagnum(G__tagnum)], __FILE__, __LINE__);
                              }
#endif // G__ASM_DBG
                              G__asm_inst[G__asm_cp] = G__POPSTROS; // FIXME: Should be ADDSTROS
                              G__asm_inst[G__asm_cp+1] = G__struct.size[G__get_tagnum(G__tagnum)];
                              G__inc_cp_asm(2, 0);
                           }
#endif // G__OLDIMPLEMENTATION1073
                           // --
                        }
#ifndef G__OLDIMPLEMENTATION1073
                        if (G__asm_wholefunction && G__asm_noverflow) {
                           // --
#ifdef G__ASM_DBG
                           if (G__asm_dbg) {
                              G__fprinterr(G__serr, "%3x,%3x: POPSTROS  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
                           }
#endif // G__ASM_DBG
                           G__asm_inst[G__asm_cp] = G__POPSTROS;
                           G__inc_cp_asm(1, 0);
                        }
#endif // G__OLDIMPLEMENTATION1073
                        // --
                     }
                  }
                  else {
                     // -- Calling constructor to normal object.
                     if (G__struct.iscpplink[tagnum] == G__CPPLINK) {
                        // Precompiled class. First, call constructor (new) function.
                        reg = G__getfunction(temp, &known, G__TRYCONSTRUCTOR);
                        //
                        // Now register the pointer we get from new to member variable table.
                        //
                        //--
                        // Set the pointer to the allocated object.
                        G__globalvarpointer = (char*) G__int(reg);
                        // Flag that we just called the constructor (and so we allocated
                        // and initialized the memory for the object).
                        G__cppconstruct = 1;
                        // Set what kind of object we are.
                        G__var_type = var_type;
                        // If everything was ok, then create the variable.
                        if (
                           (
                              known && // Constructor was found, and
                              (
                                 G__globalvarpointer || // constructor allocated memory, or
                                 G__asm_noverflow // we are generating bytecode,
                              )
                           ) || // or,
                           (G__globalcomp != G__NOLINK) || // we are making a dictionary, or
                           G__xrefflag // we are generating a variable cross-reference
                        ) {
                           // -- Create a variable in the local or global variable chain.
                           G__letvariable(new_name, G__null,::Reflex::Scope::GlobalScope(), G__p_local, newMember);
                        }
                        else if (!G__xrefflag) {
                           // -- Could not find the default constructor,
                           // and  not cross-referencing, we can generate an error message.
                           if (G__asm_wholefunction == G__ASM_FUNC_NOP) {
                              // -- We are not generating bytecode for a whole function,
                              // so we are allowed to print this error message.
                              G__fprinterr(G__serr, "Error: %s no default constructor", temp);
                           }
                           // Print a generic error message, and keep going.
                           G__genericerror(0);
                        }
                        // Reset the flag.
                        // FIXME: Should we save and restore instead?
                        G__cppconstruct = 0;
                        // Reset the object pointer.
                        // FIXME: Should we save and restore instead?
                        G__globalvarpointer = G__PVOID;
                        // Generate bytecode to reset the object pointer.
#ifndef G__OLDIMPLEMENTATION1073
                        if (G__asm_wholefunction && G__no_exec_compile) {
                           // --
#ifdef G__ASM_DBG
                           if (G__asm_dbg) {
                              G__fprinterr(G__serr, "%3x,%3x: SETGVP -1  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
                           }
#endif // G__ASM_DBG
                           G__asm_inst[G__asm_cp] = G__SETGVP;
                           G__asm_inst[G__asm_cp+1] = (long) G__PVOID;
                           G__inc_cp_asm(2, 0);
                        }
#endif // G__OLDIMPLEMENTATION1073
                        // --
                     }
                     else {
                        // -- Interpreted class, memory area was already allocated above.
                        if (G__struct.isctor[tagnum]) {
                           G__getfunction(temp, &known, G__CALLCONSTRUCTOR);
                        }
                        else {
                           G__getfunction(temp, &known, G__TRYCONSTRUCTOR);
                        }
#ifndef G__OLDIMPLEMENTATION1073
                        if (G__asm_wholefunction && G__asm_noverflow) {
                           // --
#ifdef G__ASM_DBG
                           if (G__asm_dbg) {
                              G__fprinterr(G__serr, "%3x,%3x: POPSTROS  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
                           }
#endif // G__ASM_DBG
                           G__asm_inst[G__asm_cp] = G__POPSTROS;
                           G__inc_cp_asm(1, 0);
                        }
#endif // G__OLDIMPLEMENTATION1073
                        // --
                     }
                  }
                  G__decl = 1;
                  if (G__return > G__RETURN_NORMAL) {
                     G__decl = store_decl;
                     G__constvar = 0;
                     G__tagnum = store_tagnum;
                     G__typenum = store_typenum;
                     G__reftype = G__PARANORMAL;
                     G__static_alloc = store_static_alloc2;
                     G__dynconst = 0;
                     G__globalvarpointer = G__PVOID;
                     return;
                  }
                  // struct class initialization = { x, y, z }
                  if (initary) {
                     if (
                        known &&
                        (G__struct.funcs[tagnum] & G__HAS_XCONSTRUCTOR)
                     ) {
                        G__fprinterr(G__serr, "Error: Illegal initialization of %s. Constructor exists ", new_name);
                        G__genericerror(0);
                        cin = G__fignorestream("}");
                        cin = G__fignorestream(",;");
                     }
                     else {
                        if (store_prerun) {
                           G__debug = store_debug;
                           G__step = store_step;
                           G__setdebugcond();
                           G__prerun = store_prerun;
                        }
                        cin = G__initstruct(new_name);
                     }
                     initary = 0;
                  }
               }
               else {
                  // --
                  // If temp == 'classname(arg)', this is OK,
                  // If temp == 'classobject', copy constructor
                  //
                  int flag = 0;
                  if (staticclassobject) {
                     // to pass G__getfunction()
                     G__tagnum = store_tagnum;
                  }
                  sprintf(temp1, "%s(", G__struct.name[G__get_tagnum(G__tagnum)]);
                  // FIXME: ifdef G__TEMPLATECLASS: Need to evaluate template argument list here.
                  if (temp == strstr(temp, temp1)) {
                     int c;
                     int isrc = 0;
                     G__StrBuf buf_sb(G__LONGLINE);
                     char *buf = buf_sb;
                     flag = 1;
                     c = G__getstream_template(temp, &isrc, buf, "(");
                     if (c == '(') {
                        c = G__getstream_template(temp, &isrc, buf, ")");
                        if (c == ')') {
                           if (temp[isrc]) {
                              flag = 0;
                           }
                        }
                     }
                  }
                  else if (G__struct.istypedefed[G__get_tagnum(G__tagnum)]) {
                     index = strchr(temp, '(');
                     if (index) {
                        *index = '\0';
                        ::Reflex::Type typetemp = G__find_typedef(temp);
                        if (typetemp.RawType() == (Reflex::Type) G__tagnum) {
                           sprintf(temp1, "%s(%s", G__tagnum.Name().c_str(), index + 1);
                           strcpy(temp, temp1);
                           flag = 1;
                        }
                        else {
                           flag = 0;
                        }
                        if (!flag) {
                           *index = '(';
                        }
                     }
                     else {
                        flag = 0;
                     }
                  }
                  else {
                     flag = 0;
                  }
                  if (flag) {
                     // -- Call explicit constructor, error if no constructor.
                     if (G__dispsource) {
                        G__fprinterr(G__serr, "\n!!!Calling constructor 0x%lx.%s for declaration of %s", G__store_struct_offset, temp, new_name);
                     }
                     G__decl = 0;
                     if (G__struct.iscpplink[tagnum] == G__CPPLINK) {
                        reg = G__getfunction(temp, &known, G__CALLCONSTRUCTOR);
                        G__var_type = var_type;
                        G__globalvarpointer = (char*) G__int(reg);
                        G__cppconstruct = 1;
                        if (G__globalvarpointer) {
                           G__letvariable(new_name, G__null, ::Reflex::Scope::GlobalScope(), G__p_local, newMember);
                        }
                        G__cppconstruct = 0;
                        G__globalvarpointer = G__PVOID;
                     }
                     else {
                        // -- There are similar cases above, but they are either
                        // default ctor or precompiled class which should be fine.
                        int store_static_alloc3 = G__static_alloc;
                        G__static_alloc = 0;
                        G__getfunction(temp, &known, G__CALLCONSTRUCTOR);
                        G__static_alloc = store_static_alloc3;
                     }
                     G__decl = 1;
                     if (G__return > G__RETURN_NORMAL) {
                        G__decl = store_decl;
                        G__constvar = 0;
                        G__tagnum = store_tagnum;
                        G__typenum = store_typenum;
                        G__reftype = G__PARANORMAL;
                        G__static_alloc = store_static_alloc2;
                        G__dynconst = 0;
                        G__globalvarpointer = G__PVOID;
                        return;
                     }
                  }
                  else {
                     char* store_struct_offsetB = G__store_struct_offset;
                     ::Reflex::Scope store_tagdefiningB = G__tagdefining;
                     //
                     // G__COPYCONSTRUCTOR
                     // default and user defined copy constructor
                     // is switched in G__letvariable()
                     //
                     // Call copy constructor with G__decl=1 argument reg.
                     char store_var_typeB = G__var_type;
                     ::Reflex::Scope store_tagnumB = G__tagnum;
                     ::Reflex::Type store_typenumB = G__typenum;
                     G__var_type = 'p';
                     G__tagnum = G__memberfunc_tagnum;
                     G__typenum = ::Reflex::Type();
                     G__store_struct_offset = G__memberfunc_struct_offset;
#ifndef G__OLDIMPLEMENTATION1073
                     if (G__asm_noverflow && G__asm_wholefunction) {
                        // --
#ifdef G__ASM_DBG
                        if (G__asm_dbg) {
                           G__fprinterr(G__serr, "%3x,%3x: SETMEMFUNCENV  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
                        }
#endif // G__ASM_DBG
                        G__asm_inst[G__asm_cp] = G__SETMEMFUNCENV;
                        G__inc_cp_asm(1, 0);
                     }
#endif // G__OLDIMPLEMENTATION1073
                     reg = G__getexpr(temp);
                     G__store_struct_offset = store_struct_offsetB;
#ifndef G__OLDIMPLEMENTATION1073
                     if (G__asm_noverflow && G__asm_wholefunction) {
                        // --
#ifdef G__ASM_DBG
                        if (G__asm_dbg) {
                           G__fprinterr(G__serr, "%3x,%3x: RECMEMFUNCENV  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
                        }
#endif // G__ASM_DBG
                        G__asm_inst[G__asm_cp] = G__RECMEMFUNCENV;
                        G__inc_cp_asm(1, 0);
                     }
#endif // G__OLDIMPLEMENTATION1073
                     G__var_type = store_var_typeB;
                     G__tagnum = store_tagnumB;
                     G__typenum = store_typenumB;
                     G__tagdefining = store_tagdefiningB;
                     if (G__struct.iscpplink[tagnum] == G__CPPLINK) {
                        if (
                           (G__get_tagnum(G__value_typenum(reg).RawType()) == tagnum) &&
                           (G__get_type(G__value_typenum(reg)) == 'u')
                        ) {
                           if (reg.obj.i < 0) {
                              sprintf(temp, "%s((%s)(%ld))", G__struct.name[tagnum], G__struct.name[tagnum], G__int(reg));
                           }
                           else {
                              sprintf(temp, "%s((%s)%ld)", G__struct.name[tagnum], G__struct.name[tagnum], G__int(reg));
                           }
                        }
                        else {
                           G__StrBuf tttt_sb(G__ONELINE);
                           char *tttt = tttt_sb;
                           G__valuemonitor(reg, tttt);
                           sprintf(temp, "%s(%s)", G__struct.name[tagnum], tttt);
                        }
#ifndef G__OLDIMPLEMENTATION1073
                        if (G__asm_wholefunction) {
                           G__oprovld = 1;
                        }
#endif // G__OLDIMPLEMENTATION1073
                        G__oprovld = 1;
                        G__decl = 0;
                        if (G__struct.parent_tagnum[tagnum] != -1) {
                           int local_store_exec_memberfunc = G__exec_memberfunc;
                           ::Reflex::Scope local_store_memberfunc_tagnum = G__memberfunc_tagnum;
                           G__exec_memberfunc = 1;
                           G__memberfunc_tagnum = G__Dict::GetDict().GetScope(G__struct.parent_tagnum[tagnum]);
                           reg = G__getfunction(temp, &known, G__CALLCONSTRUCTOR);
                           G__exec_memberfunc = local_store_exec_memberfunc;
                           G__memberfunc_tagnum = local_store_memberfunc_tagnum;
                        }
                        else {
                           reg = G__getfunction(temp, &known, G__CALLCONSTRUCTOR);
                        }
                        G__globalvarpointer = (char*) G__int(reg);
                        G__cppconstruct = 1;
                        G__letvariable(new_name, G__null, ::Reflex::Scope::GlobalScope(), G__p_local, newMember);
                        G__cppconstruct = 0;
                        G__globalvarpointer = G__PVOID;
                        G__oprovld = 0;
                        G__decl = 1;
#ifndef G__OLDIMPLEMENTATION1073
                        if (G__asm_wholefunction) {
                           G__oprovld = 0;
#ifdef G__ASM_DBG
                           if (G__asm_dbg) {
                              G__fprinterr(G__serr, "%3x,%3x: SETGVP -1  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
                           }
#endif // G__ASM_DBG
                           G__asm_inst[G__asm_cp] = G__SETGVP;
                           G__asm_inst[G__asm_cp+1] = -1;
                           G__inc_cp_asm(2, 0);
                        }
#endif // G__OLDIMPLEMENTATION1073
                        // --
                     }
                     else {
                        // --
#ifndef G__OLDIMPLEMENTATION1073
                        if (G__asm_wholefunction) {
                           G__oprovld = 1;
                        }
#endif // G__OLDIMPLEMENTATION1073
                        G__letvariable(new_name, reg, ::Reflex::Scope::GlobalScope(), G__p_local, newMember);
#ifndef G__OLDIMPLEMENTATION1073
                        if (G__asm_wholefunction) {
                           G__oprovld = 0;
#ifdef G__ASM_DBG
                           if (G__asm_dbg) {
                              G__fprinterr(G__serr, "%3x,%3x: POPSTROS  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
                           }
#endif // G__ASM_DBG
                           G__asm_inst[G__asm_cp] = G__POPSTROS;
                           G__inc_cp_asm(1, 0);
                        }
#endif // G__OLDIMPLEMENTATION1073
                        // --
                     }
                     if (G__return > G__RETURN_NORMAL) {
                        G__decl = store_decl;
                        G__constvar = 0;
                        G__tagnum = store_tagnum;
                        G__typenum = store_typenum;
                        G__reftype = G__PARANORMAL;
                        G__static_alloc = store_static_alloc2;
                        G__dynconst = 0;
                        G__globalvarpointer = G__PVOID;
                        G__prerun = store_prerun;
                        return;
                     }
                  }
               }
            }
            else {
               if (G__var_type == 'u') {
                  G__fprinterr(G__serr, "Error: %s not allocated(2), maybe duplicate declaration ", new_name);
                  G__genericerror(0);
               }
               // else OK because this is type name[];
               if (initary) {
                  if (store_prerun) {
                     G__debug = store_debug;
                     G__step = store_step;
                     G__setdebugcond();
                     G__prerun = store_prerun;
                  }
                  cin = G__initstruct(new_name);
               }
            }
            if (largestep) {
               largestep = 0;
               G__step = 1;
               G__setdebugcond();
            }
            if (store_prerun) {
               G__debug = store_debug;
               G__step = store_step;
               G__setdebugcond();
            }
            G__prerun = store_prerun;
            G__store_struct_offset = store_struct_offset;
#ifdef G__ASM
            if (G__asm_noverflow) {
               // -- We are generating bytecode.
#ifdef G__ASM_DBG
               if (G__asm_dbg) {
                  G__fprinterr(G__serr, "%3x,%3x: POPSTROS  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
               }
#endif // G__ASM_DBG
               G__asm_inst[G__asm_cp] = G__POPSTROS;
               G__inc_cp_asm(1, 0);
            }
#endif // G__ASM
            // --
         }
         else {
            // -- Declaration of scaler object, or a pointer or reference to an object of class type.
            if (
              (G__globalcomp != G__NOLINK) && // generating a dictionary, and
              (var_type == 'u') && // class, enum, namespace, struct, or union, and
              (new_name[0] != '*') && // not a pointer, and
              (G__reftype == G__PARANORMAL) && // not a reference, and
              G__def_struct_member && // data member, and
              G__static_alloc && // const or static data member, and
              G__prerun // in prerun
            ) {
              // -- Static data member of class type in prerun while generating a dictionary.
              // Disable memory allocation, just create variable.
              G__globalvarpointer = G__PINVALID;
            }
            // FIXME: Static data members of class type do not get their constructors run!
            G__letvariable(new_name, reg, ::Reflex::Scope::GlobalScope(), G__p_local, newMember);
            if (G__return > G__RETURN_NORMAL) {
               G__decl = store_decl;
               G__tagnum = store_tagnum;
               G__typenum = store_typenum;
               G__constvar = 0;
               G__dynconst = 0;
               G__reftype = G__PARANORMAL;
               G__static_alloc = store_static_alloc2;
               G__globalvarpointer = G__PVOID;
               return;
            }
            // Insert array initialization.
            if (initary) {
               cin = G__initary(new_name,newMember);
               initary = 0;
               if (cin == EOF) {
                  G__decl = store_decl;
                  G__constvar = 0;
                  G__tagnum = store_tagnum;
                  G__typenum = store_typenum;
                  G__reftype = G__PARANORMAL;
                  G__static_alloc = store_static_alloc2;
                  G__dynconst = 0;
                  G__globalvarpointer = G__PVOID;
                  return;
               }
            }
         }
         if (G__ansiheader == 2) {
            G__ansiheader = 0;
         }
      }
      G__globalvarpointer = G__PVOID;
      readnext:
      if (cin == ';') {
         // -- End of declaration, return.
         //
         // type  var1 , var2 ;
         //                   ^
         G__decl = store_decl;
         G__constvar = 0;
         G__tagnum = store_tagnum;
         G__typenum = store_typenum;
         G__reftype = G__PARANORMAL;
         if (G__fons_comment && G__def_struct_member) {
            G__setvariablecomment(new_name,newMember);
         }
#ifdef G__ASM
         if (G__asm_noverflow) {
            G__asm_clear();
         }
#endif // G__ASM
         G__static_alloc = store_static_alloc2;
         G__dynconst = 0;
         G__globalvarpointer = G__PVOID;
         return;
      }
      else if (cin == '}') {
         // -- Syntax error, missing semicolon, return.
         G__decl = store_decl;
         G__constvar = 0;
         G__tagnum = store_tagnum;
         G__typenum = store_typenum;
         G__reftype = G__PARANORMAL;
         fseek(G__ifile.fp, -1, SEEK_CUR);
         G__missingsemicolumn(new_name);
         G__static_alloc = store_static_alloc2;
         G__dynconst = 0;
         G__globalvarpointer = G__PVOID;
         return;
      }
      //
      // Read next declaration in a comma-separated list.
      //
      // type  var1, var2, var3;
      //             ^
      cin = G__fgetstream(new_name, ",;=():");
      if (cin == EOF) {
         // -- Reached end of input file, syntax error, missing semicolon, return.
         G__decl = store_decl;
         G__constvar = 0;
         G__tagnum = store_tagnum;
         G__typenum = store_typenum;
         G__reftype = G__PARANORMAL;
         fseek(G__ifile.fp, -1, SEEK_CUR);
         G__missingsemicolumn(new_name);
         G__static_alloc = store_static_alloc2;
         G__dynconst = 0;
         G__globalvarpointer = G__PVOID;
         return;
      }
      if (G__typepdecl) {
         var_type = tolower(var_type);
         G__var_type = var_type;
         if (G__asm_dbg) {
            if (G__dispmsg >= G__DISPNOTE) {
               G__fprinterr(G__serr, "Note: type* a,b,... declaration");
               G__printlinenum();
            }
         }
      }
      //
      // type  var1 , var2 , var3 ;
      // came to            ^  or  ^
      //
      if (new_name[0] == '&') {
         G__reftype = G__PARAREFERENCE;
         strcpy(temp, new_name + 1);
         strcpy(new_name, temp);
      }
      else if ((new_name[0] == '*') && (new_name[1] == '&')) {
         G__reftype = G__PARAREFERENCE;
         sprintf(temp, "*%s", new_name + 2);
         strcpy(new_name, temp);
      }
   }
}

//______________________________________________________________________________
//-- 01
//-- 02
//-- 03
//-- 04
//-- 05
//-- 06
//-- 07
//-- 08
//-- 09
//-- 10
//-- 01
//-- 02
//-- 03
//-- 04
//-- 05
//-- 06
//-- 07
//-- 08
//-- 09

//______________________________________________________________________________
//-- 01
//-- 02
//-- 03
//-- 04
//-- 05
//-- 06
//-- 07
//-- 08
//-- 09
//-- 10
//-- 01
//-- 02
//-- 03
//-- 04
//-- 05
//-- 06
//-- 07
//-- 08
//-- 09
//-- 10
//-- 01


//______________________________________________________________________________
//
//  Functions in the C interface.
//

// None.

//--

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
