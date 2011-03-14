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
#include "FastAllocString.h"

//______________________________________________________________________________
int G__readline_FastAlloc(FILE* fp, G__FastAllocString& line, G__FastAllocString& argbuf,
                          int* argn, char* arg[])
{
   // -- FIXME: Describe this function!
   char* null_fgets = fgets(line, (size_t) (line.Capacity() - 1), fp);
   if (null_fgets) {
      argbuf = line;
      G__split(line, argbuf, argn, arg);
   }
   else {
      line = "";
      argbuf = "";
      *argn = 0;
      arg[0] = line;
   }
   if (!null_fgets) {
      return 0;
   }
   return 1;
}

extern "C" {

// Static functions.
// None.

// External functions.
int G__readsimpleline(FILE* fp, char* line);
int G__readline(FILE* fp, char* line, char* argbuf, int* argn, char* arg[]);
#ifndef G__SMALLOBJECT
int G__cmparray(short array1[], short array2[], int num, short mask);
void G__setarray(short array[], int num, short mask, char* mode);
int G__graph(double* xdata, double* ydata, int ndata, char* title, int mode);
#ifndef G__NSTOREOBJECT
int G__storeobject(G__value *buf1, G__value *buf2);
int G__scanobject(G__value* buf);
int G__dumpobject(char* file, void* buf, int size);
int G__loadobject(char* file, void* buf, int size);
#endif // G__NSTOREOBJECT
#ifndef G__NSEARCHMEMBER
long G__what_type(char* name, char* type, char* tagname, char* type_name);
#endif // G__NSEARCHMEMBER
#endif // G__SMALLOBJECT
int G__textprocessing(FILE* fp);
#ifdef G__REGEXP
int G__matchregex(char* pattern, char* string);
#endif // G__REGEXP
#ifdef G__REGEXP1
int G__matchregex(char* pattern, char* string);
#endif // G__REGEXP1

// Functions in the C interface.
int G__split(char* line, char* sstring, int* argc, char* argv[]);

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
int G__readsimpleline(FILE* fp, char* line)
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

//______________________________________________________________________________
int G__readline(FILE* fp, char* line, char* argbuf, int* argn, char* arg[])
{
   // -- FIXME: Describe this function!
   char* null_fgets = fgets(line, 2 * G__LONGLINE, fp);  // FIXME: Possible buffer overflow here!
   if (null_fgets) {
      strcpy(argbuf, line); // Legacy code, we have no way of knowing the buffer size
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


#ifndef G__SMALLOBJECT

//______________________________________________________________________________
int G__cmparray(short array1[], short array2[], int num, short mask)
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
void G__setarray(short array[], int num, short mask, char* mode)
{
   // -- FIXME: Describe this function!
   int i;

   if (strcmp(mode, "rand") == 0) {
      for (i = 0;i < num;i++) {
         array[i] = rand() & mask; // This is a direct user request, can't be avoided
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
int G__graph(double* xdata, double* ydata, int ndata, char* title, int mode)
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
      if (system("killproc xgraph"))
         return 0;
      return 1;
   }
   switch (mode) {
      case 1:
      case 0:
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
         if (system("xgraph G__graph&")) return 1;
         break;
      case 0:
         if (system("xgraph G__graph")) return 1;
         break;
   }
   return 0;
}

#ifndef G__NSTOREOBJECT
//______________________________________________________________________________
int G__storeobject(G__value *buf1, G__value *buf2)
{
   // -- Copy object buf2 to buf1 if not a pointer.
   int i;
   struct G__var_array* var1;
   struct G__var_array* var2;
   G__value lbuf1;
   G__value lbuf2;
   if ((buf1->type != 'U') || (buf2->type != 'U') || (buf1->tagnum != buf2->tagnum)) {
      G__genericerror("Error:G__storeobject buf1,buf2 different type or non struct");
      G__fprinterr(G__serr, "buf1->type = %c , buf2->type = %c\n", buf1->type, buf2->type);
      G__fprinterr(G__serr, "buf1->tagnum = %d , buf2->tagnum = %d\n", buf1->tagnum, buf2->tagnum);
      return 1;
   }
   G__incsetup_memvar(buf1->tagnum);
   G__incsetup_memvar(buf2->tagnum);
   var1 = G__struct.memvar[buf1->tagnum];
   var2 = G__struct.memvar[buf2->tagnum];
   do {
      for (i = 0; i < var1->allvar; ++i) {
         void* p1 = (void*) (buf1->obj.i + var1->p[i]);
         void* p2 = (void*) (buf2->obj.i + var2->p[i]);
         int num_of_elements = var1->varlabel[i][1];
         if (!num_of_elements) {
            num_of_elements = 1;
         }
         switch (var1->type[i]) {
            case 'u':
               lbuf1.obj.i = buf1->obj.i + var1->p[i];
               lbuf2.obj.i = buf2->obj.i + var2->p[i];
               lbuf1.type = 'U';
               lbuf2.type = 'U';
               lbuf1.tagnum = var1->p_tagtable[i];
               lbuf2.tagnum = var2->p_tagtable[i];
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
      var1 = var1->next;
      var2 = var2->next;
   }
   while (var1);
   return 0;
}
#endif // G__NSTOREOBJECT

#ifndef G__NSTOREOBJECT
//______________________________________________________________________________
int G__scanobject(G__value* buf)
{
   // -- Scan struct object and call G__do_scanobject(ptr, name, type, tagname, type_name).
   if (buf->type != 'U') {
      G__genericerror("Error:G__scanobject buf not a struct");
      return 1;
   }
   G__incsetup_memvar(buf->tagnum);
   G__var_array* var = G__struct.memvar[buf->tagnum];
   do {
      for (int i = 0; i < var->allvar; ++i) {
         char* name = var->varnamebuf[i];
         char type = var->type[i] ;
         long pointer = buf->obj.i + var->p[i];
         char* tagname = 0;
         if (var->p_tagtable[i] > -1) {
            tagname = G__struct.name[var->p_tagtable[i]];
         }
         char* type_name = 0;
         if (var->p_typetable[i] > -1) {
            type_name = G__newtype.name[var->p_typetable[i]];
         }
         G__FastAllocString ifunc(G__ONELINE);
         ifunc.Format("G__do_scanobject((%s *)%ld,%ld,%d,%ld,%ld)", tagname, pointer, (long) name, type, (long) tagname, (long) type_name);
         G__getexpr(ifunc);
      }
      var = var->next;
   }
   while (var);
   return 0;
}
#endif // G__NSTOREOBJECT

#ifndef G__NSTOREOBJECT
//______________________________________________________________________________
int G__dumpobject(char* file, void* buf, int size)
{
   // -- Dump object into a file.
   FILE* fp;
   fp = fopen(file, "wb");
   fwrite(buf, size , 1, fp);
   fflush(fp);
   fclose(fp);
   return 1;
}
#endif // G__NSTOREOBJECT

#ifndef G__NSTOREOBJECT
//______________________________________________________________________________
int G__loadobject(char* file, void* buf, int size)
{
   // -- Load object from a file.
   FILE* fp = fopen(file, "rb");
   size_t read = fread(buf, size, 1, fp);
   if ( read != (size_t)size) {
      G__fprinterr(G__serr, "G__loadobject: cannot read full object (%d instead of %d bytes)", read, size);
   }
   fclose(fp);
   return (read == (size_t)size);
}
#endif // G__NSTOREOBJECT

#ifndef G__NSEARCHMEMBER
//______________________________________________________________________________
long G__what_type(char* name, char* type, char* tagname, char* type_name)
{
   // -- FIXME: Describe this function!
   G__value buf = G__calc_internal(name);
   const char* ispointer = "";
   if (isupper(buf.type)) {
      ispointer = " *";
   }
   G__FastAllocString vtype(80);
   switch (tolower(buf.type)) {
      case 'u':
         vtype.Format("struct %s %s", G__struct.name[buf.tagnum], ispointer);
         break;
      case 'b':
         vtype.Format("unsigned char %s", ispointer);
         break;
      case 'c':
         vtype.Format("char %s", ispointer);
         break;
      case 'r':
         vtype.Format("unsigned short %s", ispointer);
         break;
      case 's':
         vtype.Format("short %s", ispointer);
         break;
      case 'h':
         vtype.Format("unsigned int %s", ispointer);
         break;
      case 'i':
         vtype.Format("int %s", ispointer);
         break;
      case 'k':
         vtype.Format("unsigned long %s", ispointer);
         break;
      case 'l':
         vtype.Format("long %s", ispointer);
         break;
      case 'f':
         vtype.Format("float %s", ispointer);
         break;
      case 'd':
         vtype.Format("double %s", ispointer);
         break;
      case 'e':
         vtype.Format("FILE %s", ispointer);
         break;
      case 'y':
         vtype.Format("void %s", ispointer);
         break;
      case 'w':
         vtype.Format("logic %s", ispointer);
         break;
      case 0:
         vtype.Format("NULL %s", ispointer);
         break;
      case 'p':
         vtype = "macro";
         break;
      case 'o':
         vtype = "automatic";
         break;
      case 'g':
         vtype = "bool";
         break;
      default:
         vtype.Format("unknown %s", ispointer);
         break;
   }
   if (type) {
      strcpy(type, vtype); // Legacy interface, we have no way of knowing input size
   }
   if (tagname && (buf.tagnum > -1)) {
      strcpy(tagname, G__struct.name[buf.tagnum]); // Legacy interface, we have no way of knowing input size
   }
   if (type_name && (buf.typenum > -1)) {
      strcpy(type_name, G__newtype.name[buf.typenum]); // Legacy interface, we have no way of knowing input size
   }
   vtype.Format("&%s", name);
   buf = G__calc_internal(vtype);
   return buf.obj.i;
}
#endif // G__NSEARCHMEMBER

#endif // G__SMALLOBJECT

//______________________________________________________________________________
int G__textprocessing(FILE* fp)
{
   // -- FIXME: Describe this function!
   return G__readline(fp, G__oline, G__argb, &G__argn, G__arg);
}

#ifdef G__REGEXP
//______________________________________________________________________________
int G__matchregex(char* pattern, char* string)
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
int G__matchregex(char* pattern, char* string)
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
int G__split(char* line, char* sstring, int* argc, char* argv[])
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

} // extern "C"

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
