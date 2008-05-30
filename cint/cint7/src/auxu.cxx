/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file aux.c
 ************************************************************************
 * Description:
 *  Auxuary function
 ************************************************************************
 * Copyright(c) 1995~2002  Masaharu Goto
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "common.h"
using namespace Cint::Internal;

#include "Reflex/Builder/TypeBuilder.h"

// Static functions.
// None.

// Functions in the C interface.
extern "C" int G__readline(FILE* fp, char* line, char* argbuf, int* argn, char* arg[]);
extern "C" int G__split(char* line, char* sstring, int* argc, char* argv[]);

//______________________________________________________________________________
//
//  Static functions.
//

// None.

//______________________________________________________________________________
//
//  External functions.
//

//______________________________________________________________________________
int Cint::Internal::G__readsimpleline(FILE* fp, char* line)
{
   // -- FIXME: Describe this function!
   char* null_fgets = fgets(line, 2 * G__LONGLINE, fp); // FIXME: Possible buffer overflow here!
   if (null_fgets) {
      char* p = strchr(line, '\n');
      if (p) {
         *p = '\0';
      }
      p = strchr(line, '\r');
      if (p) {
         *p = '\0';
      }
   }
   else {
      line[0] = '\0';
   }
   if (!null_fgets) {
      return 0;
   }
   return 1;
}

#ifndef G__SMALLOBJECT

//______________________________________________________________________________
int Cint::Internal::G__cmparray(short array1[], short array2[], int num, short mask)
{
   // -- FIXME: Describe this function!
   int i, fail = 0, firstfail = -1, fail1 = 0, fail2 = 0;
   for (i = 0;i < num;i++) {
      if ((array1[i]&mask) != (array2[i]&mask)) {
         if (firstfail == -1) {
            firstfail = i;
            fail1 = array1[i];
            fail2 = array2[i];
         }
         fail++;
      }
   }
   if (fail != 0) {
      G__fprinterr(G__serr, "G__cmparray() failcount=%d from [%d] , %d != %d\n",
                   fail, firstfail, fail1, fail2);
   }
   return(fail);
}

//______________________________________________________________________________
void Cint::Internal::G__setarray(short array[], int num, short mask, char* mode)
{
   // -- FIXME: Describe this function!
   int i;

   if (strcmp(mode, "rand") == 0) {
      for (i = 0;i < num;i++) {
         array[i] = rand() & mask;
      }
   }
   if (strcmp(mode, "inc") == 0) {
      for (i = 0;i < num;i++) {
         array[i] = i & mask;
      }
   }
   if (strcmp(mode, "dec") == 0) {
      for (i = 0;i < num;i++) {
         array[i] = (num - i) & mask;
      }
   }
   if (strcmp(mode, "check1") == 0)  {
      for (i = 0;i < num;i++) {
         array[i] = 0xaaaa & mask;
         array[++i] = 0x5555 & mask;
      }
   }
   if (strcmp(mode, "check2") == 0) {
      for (i = 0;i < num;i++) {
         array[i] = 0x5555 & mask;
         array[++i] = 0xaaaa & mask;
      }
   }
   if (strcmp(mode, "check3") == 0) {
      for (i = 0;i < num;i++) {
         array[i] = 0xaaaa & mask;
         array[++i] = 0xaaaa & mask;
         array[++i] = 0x5555 & mask;
         array[++i] = 0x5555 & mask;
      }
   }
   if (strcmp(mode, "check4") == 0) {
      for (i = 0;i < num;i++) {
         array[i] = 0x5555 & mask;
         array[++i] = 0x5555 & mask;
         array[++i] = 0xaaaa & mask;
         array[++i] = 0xaaaa & mask;
      }
   }
   if (strcmp(mode, "zero") == 0) {
      for (i = 0;i < num;i++) {
         array[i] = 0;
      }
   }
   if (strcmp(mode, "one") == 0) {
      for (i = 0;i < num;i++) {
         array[i] = 0xffff & mask;
      }
   }
}

//______________________________________________________________________________
int Cint::Internal::G__graph(double* xdata, double* ydata, int ndata, char* title, int mode)
{
   // -- FIXME: Describe this function!
   //
   //  xdata[i] : *double pointer of x data array
   //  ydata[i] : *double pointer of y data array
   //  ndata    : int number of data
   //  title    : *char title
   //  mode     : int mode 0:wait for close,
   //                      1:leave window and proceed
   //                      2:kill xgraph window
   FILE* fp = 0;
   if (mode == 2) {
      system("killproc xgraph");
      return 1;
   }
   switch (mode) {
      case 1:
      case 0:
         fp = fopen("G__graph", "w");
         fprintf(fp, "TitleText: %s\n", title);
         break;
      case 2:
         fp = fopen("G__graph", "w");
         fprintf(fp, "TitleText: %s\n", title);
         break;
      case 3:
         fp = fopen("G__graph", "a");
         fprintf(fp, "\n");
         fprintf(fp, "TitleText: %s\n", title);
         break;
      case 4:
      default:
         fp = fopen("G__graph", "a");
         fprintf(fp, "\n");
         fprintf(fp, "TitleText: %s\n", title);
         break;
   }
   fprintf(fp, "\"%s\"\n", title);
   for (int i = 0; i < ndata; ++i) {
      fprintf(fp, "%e %e\n", xdata[i], ydata[i]);
   }
   fclose(fp);
   switch (mode) {
      case 1:
      case 4:
         system("xgraph G__graph&");
         break;
      case 0:
         system("xgraph G__graph");
         break;
   }
   return 0;
}

#ifndef G__NSTOREOBJECT
//______________________________________________________________________________
int Cint::Internal::G__storeobject(G__value* buf1, G__value* buf2)
{
   // -- Copy object buf2 to buf1 if not a pointer.
   G__value lbuf1;
   G__value lbuf2;
   if (
      !G__value_typenum(*buf1).RawType().IsClass() ||
      !G__value_typenum(*buf1).FinalType().IsPointer() ||
      !G__value_typenum(*buf2).RawType().IsClass() ||
      !G__value_typenum(*buf2).FinalType().IsPointer() ||
      (G__value_typenum(*buf1).RawType() != G__value_typenum(*buf2).RawType())
   ) {
      G__genericerror("Error:G__storeobject buf1,buf2 different type or non struct");
      G__fprinterr(G__serr, "buf1 type is '%s', buf2->type is '%s'\n", G__value_typenum(*buf1).Name(::Reflex::SCOPED).c_str(), G__value_typenum(*buf2).Name(::Reflex::SCOPED).c_str());
      return 1;
   }
   G__incsetup_memvar((G__value_typenum(*buf1)));
   ::Reflex::Scope varscope = G__value_typenum(*buf1).RawType();
   {
      for (unsigned int i = 0; i < varscope.DataMemberSize(); ++i) {
         ::Reflex::Member m(varscope.DataMemberAt(i));
         char* offset = G__get_offset(m);
         void* p1 = (void*) (buf1->obj.i + offset);
         void* p2 = (void*) (buf2->obj.i + offset);
         int num_of_elements = G__get_varlabel(m, 1);
         if (!num_of_elements) {
            num_of_elements = 1;
         }
         switch (G__get_type(m.TypeOf())) {
            case 'u':
               lbuf1.obj.i = (long) p1;
               lbuf2.obj.i = (long) p2;
               G__value_typenum(lbuf1) = ::Reflex::PointerBuilder(m.TypeOf().RawType());
               G__value_typenum(lbuf2) = G__value_typenum(lbuf1);
               G__storeobject(&lbuf1, &lbuf2);
               break;
            case 'g':
#ifdef G__BOOL4BYTE
               memcpy(p1, p2, num_of_elements * G__INTALLOC);
               break;
#endif // G__BOOL4BYTE
            case 'b':
            case 'c':
               memcpy(p1, p2, num_of_elements * G__CHARALLOC);
               break;
            case 'r':
            case 's':
               memcpy(p1, p2, num_of_elements * G__SHORTALLOC);
               break;
            case 'h':
            case 'i':
               memcpy(p1, p2, num_of_elements * G__INTALLOC);
               break;
            case 'k':
            case 'l':
               memcpy(p1, p2, num_of_elements * G__LONGALLOC);
               break;
            case 'f':
               memcpy(p1, p2, num_of_elements * G__FLOATALLOC);
               break;
            case 'd':
            case 'w':
               memcpy(p1, p2, num_of_elements * G__DOUBLEALLOC);
               break;
         }
      }
   }
   return 0;
}
#endif // G__NSTOREOBJECT

#ifndef G__NSTOREOBJECT
//______________________________________________________________________________
int Cint::Internal::G__scanobject(G__value* buf)
{
   // -- Scan struct object and call G__do_scanobject(ptr, name, type, tagname, type_name).
   if (G__get_type(G__value_typenum(*buf)) != 'U') {
      G__genericerror("Error:G__scanobject buf not a struct");
      return 1;
   }
   G__incsetup_memvar(G__value_typenum(*buf));
   ::Reflex::Scope varscope = G__value_typenum(*buf).RawType();
   for (unsigned int i = 0; i < varscope.DataMemberSize(); ++i) {
      ::Reflex::Member m(varscope.DataMemberAt(i));
      std::string name( m.Name() );
      char type = G__get_type(m.TypeOf());
      long pointer = (long) (buf->obj.i + G__get_offset(m));
      Reflex::Type raw( m.TypeOf().RawType() );
      std::string tagname;
      if (raw.IsClass() || raw.IsUnion() || raw.IsEnum()) {
         tagname = raw.Name();
      }
      std::string type_name( m.TypeOf().Name() );
      G__StrBuf ifunc_sb(G__ONELINE);
      char *ifunc = ifunc_sb;
      sprintf(ifunc, "G__do_scanobject((%s *)%ld,%ld,%d,%ld,%ld)", tagname.size() ? tagname.c_str() : 0, pointer, (long) name.c_str(), type, (long) tagname.c_str(), (long) type_name.c_str());
      G__getexpr(ifunc);
   }
   return 0;
}
#endif // G__NSTOREOBJECT

#ifndef G__NSTOREOBJECT
//______________________________________________________________________________
int Cint::Internal::G__dumpobject(char* file, void* buf, int size)
{
   // -- Dump object into a file.
   FILE* fp = fopen(file, "wb");
   fwrite(buf, size, 1, fp);
   fflush(fp);
   fclose(fp);
   return 1;
}
#endif // G__NSTOREOBJECT

#ifndef G__NSTOREOBJECT
//______________________________________________________________________________
int Cint::Internal::G__loadobject(char* file, void* buf, int size)
{
   // -- Load object from a file.
   FILE* fp = fopen(file, "rb");
   fread(buf, size, 1, fp);
   fclose(fp);
   return 1;
}
#endif // G__NSTOREOBJECT

#ifndef G__NSEARCHMEMBER
//______________________________________________________________________________
long Cint::Internal::G__what_type(char* name, char* type, char* tagname, char* type_name)
{
   // -- FIXME: Describe this function!
   G__value buf = G__calc_internal(name);
   char buf_type = G__get_type(G__value_typenum(buf));
   char ispointer[3] = "";
   if (isupper(buf_type)) {
      sprintf(ispointer, " *");
   }
   static char vtype[80];
   switch (tolower(buf_type)) {
      case 'u':
         sprintf(vtype, "struct %s %s", G__value_typenum(buf).RawType().Name().c_str(), ispointer);
         break;
      case 'b':
         sprintf(vtype, "unsigned char %s", ispointer);
         break;
      case 'c':
         sprintf(vtype, "char %s", ispointer);
         break;
      case 'r':
         sprintf(vtype, "unsigned short %s", ispointer);
         break;
      case 's':
         sprintf(vtype, "short %s", ispointer);
         break;
      case 'h':
         sprintf(vtype, "unsigned int %s", ispointer);
         break;
      case 'i':
         sprintf(vtype, "int %s", ispointer);
         break;
      case 'k':
         sprintf(vtype, "unsigned long %s", ispointer);
         break;
      case 'l':
         sprintf(vtype, "long %s", ispointer);
         break;
      case 'f':
         sprintf(vtype, "float %s", ispointer);
         break;
      case 'd':
         sprintf(vtype, "double %s", ispointer);
         break;
      case 'e':
         sprintf(vtype, "FILE %s", ispointer);
         break;
      case 'y':
         sprintf(vtype, "void %s", ispointer);
         break;
      case 'w':
         sprintf(vtype, "logic %s", ispointer);
         break;
      case 0:
         sprintf(vtype, "NULL %s", ispointer);
         break;
      case 'p':
         sprintf(vtype, "macro");
         break;
      case 'o':
         sprintf(vtype, "automatic");
         break;
      case 'g':
         sprintf(vtype, "bool");
         break;
      default:
         sprintf(vtype, "unknown %s", ispointer);
         break;
   }
   if (type) {
      strcpy(type, vtype);
   }
   if (tagname) {
      ::Reflex::Type rtype(G__value_typenum(buf).RawType());
      if (rtype && (rtype.IsClass() || rtype.IsUnion() || rtype.IsEnum())) {
         strcpy(tagname, rtype.Name().c_str());
      }
   }
   if (type_name && G__value_typenum(buf)) {
      strcpy(type_name, G__value_typenum(buf).Name(::Reflex::SCOPED).c_str());
   }
#ifdef __GNUC__
#else
#pragma message(FIXME("do we really need to go via string here?! I.e. can't we change G__calc_internal?"))
#endif
   sprintf(vtype, "&%s", name);
   buf = G__calc_internal(vtype);
   return buf.obj.i;
}
#endif // G__NSEARCHMEMBER

#endif // G__SMALLOBJECT

//______________________________________________________________________________
int Cint::Internal::G__textprocessing(FILE* fp)
{
   // -- FIXME: Describe this function!
   return G__readline(fp, G__oline, G__argb, &G__argn, G__arg);
}

#ifdef G__REGEXP
//______________________________________________________________________________
int Cint::Internal::G__matchregex(char* pattern, char* string)
{
   // -- FIXME: Describe this function!
   regex_t re;
   int i = regcomp(&re, pattern, REG_EXTENDED | REG_NOSUB);
   if (i) {
      return 0;
   }
   i = regexec(&re, string, 0, 0, 0);
   regfree(&re);
   if (i) {
      return 0;
   }
   // match
   return 1;
}
#endif // G__REGEXP

#ifdef G__REGEXP1
//______________________________________________________________________________
int Cint::Internal::G__matchregex(char* pattern, char* string)
{
   // -- FIXME: Describe this function!
   char* re = regcmp(pattern, 0);
   if (!re) {
      return 0;
   }
   char* s = regex(re, string);
   free(re);
   if (!s) {
      return 0;
   }
   // match
   return 1;
}
#endif // G__REGEXP1

//______________________________________________________________________________
//
//  Functions in the C interface.
//

//______________________________________________________________________________
extern "C" int G__readline(FILE* fp, char* line, char* argbuf, int* argn, char* arg[])
{
   // -- FIXME: Describe this function!
   char* null_fgets = fgets(line, 2 * G__LONGLINE, fp);  // FIXME: Possible buffer overflow here!
   if (null_fgets) {
      strcpy(argbuf, line);
      G__split(line, argbuf, argn, arg);
   }
   else {
      line[0] = '\0';
      argbuf = '\0';
      *argn = 0;
      arg[0] = line;
   }
   if (!null_fgets) {
      return 0;
   }
   return 1;
}

//______________________________________________________________________________
extern "C" int G__split(char* line, char* sstring, int* argc, char* argv[])
{
   // -- Split arguments separated by space char.
   //
   //  CAUTION: input string will be modified. If you want to keep
   //           the original string, you should copy it to another string.
   //
   unsigned char* string = (unsigned char*) sstring;
   int i = 0;
   while ((string[i] != '\n') && (string[i] != '\r') && string[i]) {
      ++i;
   }
   string[i] = '\0';
   line[i] = '\0';
   int lenstring = i;
   argv[0] = line;
   *argc = 0;
   int single_quote = 0;
   int double_quote = 0;
   int back_slash = 0;
   int flag = 0;
   for (i = 0; i < lenstring; ++i) {
      switch (string[i]) {
         case '\\':
            // -- Backslash.
            back_slash ^= 1;
            break;
         case '\'':
            // -- Single quote.
            if (!double_quote && !back_slash) {
               single_quote ^= 1;
               string[i] = '\0';
               flag = 0;
            }
            break;
         case '"':
            // -- Double quote.
            if (!single_quote && !back_slash) {
               double_quote ^= 1;
               string[i] = '\0';
               flag = 0;
            }
            break;
         default:
            // -- Any other character.
            if ((isspace(string[i])) && !back_slash && !single_quote && !double_quote) {
               string[i] = '\0';
               flag = 0;
            }
            else {
               if (!flag) {
                  (*argc)++;
                  argv[*argc] = (char*) &string[i];
                  flag = 1;
               }
            }
            back_slash = 0;
            break;
      }
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
