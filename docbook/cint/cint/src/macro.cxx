/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file macro.c
 ************************************************************************
 * Description:
 *  Define macro
 ************************************************************************
 * Copyright(c) 1995~2002  Masaharu Goto 
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "common.h"

extern "C" {

// Static Functions.
static int G__handle_as_typedef(char* oldtype, char* newtype);
static void G__createmacro(G__FastAllocString &new_name, char* initvalue);
static int G__createfuncmacro(char* new_name);
static int G__replacefuncmacro(const char* item, G__Callfuncmacro* callfuncmacro, G__Charlist* callpara, G__Charlist* defpara, FILE* def_fp, fpos_t def_pos, int nobraces, int nosemic);
static int G__transfuncmacro(const char* item, G__Deffuncmacro* deffuncmacro, G__Callfuncmacro* callfuncmacro, fpos_t call_pos, char* p, int nobraces, int nosemic);
static int G__argsubstitute(G__FastAllocString &symbol, G__Charlist* callpara, G__Charlist* defpara);
static int G__getparameterlist(char* paralist, G__Charlist* charlist);

// External Functions.
void G__define();
G__value G__execfuncmacro(const char* item, int* done);
int G__execfuncmacro_noexec(const char* macroname);
int G__maybe_finish_macro();
int G__freedeffuncmacro(G__Deffuncmacro* deffuncmacro);
int G__freecharlist(G__Charlist* charlist);

// Functions in the C interface.
// None.

//______________________________________________________________________________
struct G__funcmacro_stackelt
{
   fpos_t pos;
   struct G__input_file file;
   struct G__funcmacro_stackelt* next;
};

struct G__funcmacro_stackelt* G__funcmacro_stack = 0;

//______________________________________________________________________________
//
//  Static Functions.
//

//______________________________________________________________________________
static int G__handle_as_typedef(char* oldtype, char* newtype)
{
   // -- Handle #define MYMACRO int
   //
   // Note: This routine is part of the parser proper.
   //
   int type = '\0' , tagnum = -1 , ispointer = 0 , isunsigned;
   int typenum;
   char *p, *ptype;
   p = strchr(oldtype, '*');
   if (p) {
      ispointer = 'A' -'a';
      *p = '\0';
   }
   else {
      ispointer = 0;
   }
   if (strncmp(oldtype, "unsigned", 8) == 0) {
      ptype = oldtype + 8;
      isunsigned = -1; /* 0 */
   }
   else if (strncmp(oldtype, "signed", 6) == 0) {
      ptype = oldtype + 6;
      isunsigned = 0;
   }
   else {
      ptype = oldtype;
      isunsigned = 0; /* -1 */
   }
   if (strcmp(ptype, "int") == 0) {
      type = 'i' + ispointer + isunsigned;
   }
   else if (strcmp(ptype, "char") == 0) {
      type = 'c' + ispointer + isunsigned;
   }
   else if (strcmp(oldtype, "double") == 0) {
      type = 'd' + ispointer; /* bug fix */
   }
   else if (strcmp(oldtype, "long long") == 0) {
      type = 'n' + ispointer;
   }
   else if (strcmp(oldtype, "unsigned long long") == 0) {
      type = 'm' + ispointer;
   }
   else if (strcmp(oldtype, "long double") == 0) {
      type = 'q' + ispointer;
   }
   else if (strcmp(ptype, "short") == 0) {
      type = 's' + ispointer + isunsigned;
   }
   else if (strcmp(ptype, "long") == 0) {
      type = 'l' + ispointer + isunsigned;
   }
   else if (strcmp(oldtype, "float") == 0) {
      type = 'f' + ispointer;
   }
   else if (strcmp(oldtype, "bool") == 0) {
      type = 'g' + ispointer;
   }
   else if (strncmp(oldtype, "struct", 6) == 0) {
      ptype = oldtype + 6;
      type = 'u' + ispointer;
      tagnum = G__defined_tagname(ptype, 0);
   }
   else if (strncmp(oldtype, "class", 5) == 0) {
      ptype = oldtype + 5;
      type = 'u' + ispointer;
      tagnum = G__defined_tagname(ptype, 0);
   }
   else if (strncmp(oldtype, "enum", 4) == 0) {
      ptype = oldtype + 4;
      type = 'i' + ispointer;
      tagnum = G__defined_tagname(ptype, 0);
   }
   else {
      tagnum = G__defined_tagname(oldtype, 1);
      if (tagnum >= 0) {
         type = 'u' + ispointer;
      }
      else {
         typenum = G__defined_typename(oldtype);
         if (typenum >= 0) {
            type = G__newtype.type[typenum];
            tagnum = G__newtype.tagnum[typenum];
         }
      }
   }
   /* this is only workaround for STL Allocator */
   if (strcmp(newtype, "Allocator") == 0) {
      G__strlcpy(G__Allocator, oldtype, G__ONELINE);
   }
   else if (strcmp(newtype, "vector") == 0) {}
   else if (strcmp(newtype, "list") == 0) {}
   else if (strcmp(newtype, "deque") == 0) {}
   else if (strcmp(newtype, "rb_tree") == 0) {}
   else
      if (type) {
         if (strcmp(newtype, "bool") != 0) {
            if (G__dispmsg >= G__DISPNOTE) {
               G__fprinterr(G__serr, "Note: macro handled as typedef %s %s;"
                            , oldtype, newtype);
               G__printlinenum();
            }
         }
         G__search_typename(newtype, type, tagnum, 0);
      }
      else {
         G__add_replacesymbol(newtype, oldtype);
#if G__NEVER
         if (G__dispmsg >= G__DISPNOTE) {
            G__fprinterr(G__serr, "Note: #define %s %s", newtype, oldtype);
            G__printlinenum();
         }
#endif // G__NEVER
      }
   return 0;
}

//______________________________________________________________________________
static void G__createmacro(G__FastAllocString &new_name, char* initvalue)
{
   // -- Handle #define MYMACRO ...\<EOL>
   //                   ...\<EOL>
   //                   ...
   //
   // Note: This routine is part of the parser proper.
   //
   G__FastAllocString line(G__ONELINE);
   int c;
   char *p, *null_fgets;
   fpos_t pos;
   G__value evalval = G__null;
   /* Set flag that there is a macro or template in the source file,
    * so that this file won't be closed even with -cN option */
   ++G__macroORtemplateINfile;
   if (!G__mfp) {
#ifdef G__DEBUG
      G__fprinterr(G__serr, "Limitation: This form of macro may not be expanded. Use +P or -p option");
      G__printlinenum();
#endif // G__DEBUG
      G__openmfp();
      fgetpos(G__mfp, &G__nextmacro);
      G__mline = 1;
   }
   else {
      fsetpos(G__mfp, &G__nextmacro);
   }
   /* print out header */
   ++G__mline;
   fprintf(G__mfp, "// #define %s  FILE:%s LINE:%d\n"
           , new_name()
           , G__ifile.name, G__ifile.line_number);
   fgetpos(G__mfp, &pos);
   fprintf(G__mfp, "# %d\n", ++G__mline);
   ++G__mline;
   fprintf(G__mfp, "{\n");
   fprintf(G__mfp, "%s\n", initvalue);
   /* translate macro */
   int start_line = G__ifile.line_number;
   do {
      null_fgets = fgets(line, G__ONELINE, G__ifile.fp);
      if (null_fgets == NULL) {
         G__fprinterr(G__serr, "Error: Missing newline at or after line %d.\n", start_line);
         G__unexpectedEOF("G__createmacro()");
      }
      ++G__ifile.line_number;
      p = strchr(line, '\n');
      if (p) {
         *p = '\0';
      }
      p = strchr(line, '\r');
      if (p) {
         *p = '\0';
      }
      p = line + strlen(line);
      c = '\n';
      if (*(p - 1) == '\\') {
         *(p - 1) = '\0';
         c = '\\';
      }
      if (G__dispsource) {
         G__fprinterr(G__serr, "\\\n%-5d", G__ifile.line_number);
         G__fprinterr(G__serr, "%s", line());
      }
      ++G__mline;
      fprintf(G__mfp, "%s\n", line());
   }
   while (c != '\n' && c != '\r');
   p = strrchr(line, ';');
   ++G__mline;
   if (p == NULL) {
      fprintf(G__mfp, ";}\n");
   }
   else {
      fprintf(G__mfp, "}\n");
   }
   fgetpos(G__mfp, &G__nextmacro);
#ifndef G__OLDIMPLEMENTATION2191
   G__var_type = 'j';
#else // G__OLDIMPLEMENTATION2191
   G__var_type = 'm';
#endif // G__OLDIMPLEMENTATION2191
   G__typenum = -1;
   G__tagnum = -1;
   evalval.obj.i = (long)(&pos);
   {
      int save_def_struct_member = G__def_struct_member;
      G__def_struct_member = 0;
      G__letvariable(new_name, evalval, &G__global, G__p_local);
      G__var_type = 'p';
      G__def_struct_member = save_def_struct_member;
   }
}

//______________________________________________________________________________
static int G__createfuncmacro(char* new_name)
{
   // -- Handle #define MYMACRO(...,...,...) ...
   //
   // Note: This routine is part of the parser proper.
   //
   struct G__Deffuncmacro *deffuncmacro;
   int hash, i;
   G__FastAllocString paralist(G__ONELINE);
   int c;
   if (G__ifile.filenum > G__gettempfilenum()) {
      G__fprinterr(G__serr, "Limitation: Macro function can not be defined in a command line or a tempfile\n");
      G__genericerror("You need to write it in a source file");
      G__fprinterr(G__serr, "Besides, it is recommended to use function template instead\n");
      return (-1);
   }
   // Set flag that there is a macro or template in the source file,
   // so that this file won't be closed even with -cN option.
   ++G__macroORtemplateINfile;
   /* Search for the end of list */
   deffuncmacro = &G__deffuncmacro;
   while (deffuncmacro->next) deffuncmacro = deffuncmacro->next;
   /* store name */
   deffuncmacro->name = (char*)malloc(strlen(new_name) + 1);
   strcpy(deffuncmacro->name, new_name); // Okay we allocate enough space
   /* store hash */
   G__hash(new_name, hash, i)
   deffuncmacro->hash = hash;
   /* read parameter list */
   c = G__fgetstream(paralist, 0, ")");
   G__ASSERT(')' == c);
   G__getparameterlist(paralist, &deffuncmacro->def_para);
   /* store file pointer, line number and position */
   deffuncmacro->def_fp = G__ifile.fp;
   fgetpos(G__ifile.fp, &deffuncmacro->def_pos);
   deffuncmacro->line = G__ifile.line_number;
   /* allocate and initialize next list */
   deffuncmacro->next = (struct G__Deffuncmacro*) malloc(sizeof(struct G__Deffuncmacro));
   deffuncmacro->next->callfuncmacro.next = 0;
   deffuncmacro->next->callfuncmacro.call_fp = 0;
   deffuncmacro->next->callfuncmacro.call_filenum = -1;
   deffuncmacro->next->def_para.string = 0;
   deffuncmacro->next->def_para.next = 0;
   deffuncmacro->next->next = 0;
   deffuncmacro->next->name = 0;
   deffuncmacro->next->hash = 0;
   // after this, source file is read to end of line
   return 0;
}

//______________________________________________________________________________
static int G__replacefuncmacro(const char* item, G__Callfuncmacro* callfuncmacro, G__Charlist* callpara, G__Charlist* defpara, FILE* def_fp, fpos_t def_pos, int nobraces, int nosemic)
{
   // -- Replace function macro parameter at the first execution of func macro.
   fpos_t pos;
   int c;
   int semicolumn;
   G__FastAllocString symbol(G__ONELINE);
   static const char *punctuation = " \t\n;:=+-)(*&^%$#@!~'\"\\|][}{/?.>,<";
   int double_quote = 0, single_quote = 0;
   fpos_t backup_pos;
   if (!G__mfp) {
      // --
#ifdef G__DEBUG
      G__fprinterr(G__serr, "Limitation: This form of macro may not be expanded. Use +P or -p option");
      G__printlinenum();
#endif // G__DEBUG
      G__openmfp();
      fgetpos(G__mfp, &G__nextmacro);
      G__mline = 1;
   }
   else {
      fsetpos(G__mfp, &G__nextmacro);
   }
   /* print out header */
   ++G__mline;
   fprintf(G__mfp, "// #define %s  FILE:%s LINE:%d\n", item, G__ifile.name, G__ifile.line_number);
   fgetpos(G__mfp, &pos);
   callfuncmacro->mfp_pos = pos;
   fprintf(G__mfp, "# %d\n", ++G__mline);
   ++G__mline;
   fprintf(G__mfp, "%s\n", nobraces ? "" : "{");
   //
   // read macro definition and substitute symbol
   //
   // set file pointer and position
   G__ifile.fp = def_fp;
   fsetpos(def_fp, &def_pos);
   // read definition and substitute
   fgetpos(G__mfp, &backup_pos);
   semicolumn = 0;
   bool quote = false;
   while (1) {
      G__disp_mask = 10000; // FIXME: Crazy!
      c = G__fgetstream(symbol, 0,  punctuation);
      if ('\0' != symbol[0]) {
         if (!double_quote && !single_quote) {
            G__argsubstitute(symbol, callpara, defpara);
         }
         if (quote) {
            fprintf(G__mfp, "\"%s\"", symbol());
            quote = false;
         } else {
            fprintf(G__mfp, "%s", symbol());
         }
         fgetpos(G__mfp, &backup_pos);
         semicolumn = 0;
      }
      if (!single_quote && !double_quote) {
         if ('\n' == c || '\r' == c) {
            break;
         }
         if ('\\' == c) {
            c = G__fgetc();
            if ('\n' == c) {
               continue;
            }
            if ('\r' == c) {
               c = G__fgetc();
            }
         }
         if (';' == c) {
            semicolumn = 1;
         }
         else if (!isspace(c)) {
            semicolumn = 0;
         }
         if (c == '#') {
            c = G__fgetc();
            if (c == '#') {
               // -- Token paste operation.
               fsetpos(G__mfp, &backup_pos);
               G__fgetspace();
               fseek(G__ifile.fp, -1, SEEK_CUR);
               continue;
            }
            else {
               fseek(G__ifile.fp, -1, SEEK_CUR);
               quote = true;
               continue;
            }
         }
      }
      if ('\'' == c && !double_quote) {
         single_quote = single_quote ^ 1 ;
      }
      else if ('"' == c && !single_quote) {
         double_quote = double_quote ^ 1 ;
      }
      fputc(c, G__mfp);
      if (!isspace(c)) {
         fgetpos(G__mfp, &backup_pos);
      }
      if ('\n' == c) {
         ++G__mline;
      }
   }
   // finishing up
   G__disp_mask = 0;
   if (!nosemic && !semicolumn) {
      fprintf(G__mfp, " ;");
   }
   G__mline += 2;
   fprintf(G__mfp, "\n%s\n" , nobraces ? "" : "}");
   fputc('\0', G__mfp); // Mark the end of this expansion.
   fgetpos(G__mfp, &G__nextmacro);
   fflush(G__mfp);
   return 0;
}

//______________________________________________________________________________
static int G__transfuncmacro(const char* item, G__Deffuncmacro* deffuncmacro, G__Callfuncmacro* callfuncmacro, fpos_t call_pos, char* p, int nobraces, int nosemic)
{
   // -- Translate function macro parameter at the first execution of func macro.
   struct G__Charlist call_para;
   /* set file pointer and position */
   callfuncmacro->call_fp = G__ifile.fp;
   callfuncmacro->call_filenum = G__ifile.filenum;
   if (G__ifile.fp) callfuncmacro->call_pos = call_pos;
   callfuncmacro->line = G__ifile.line_number;
   /* allocate and initialize next list */
   callfuncmacro->next = (struct G__Callfuncmacro*) malloc(sizeof(struct G__Callfuncmacro));
   callfuncmacro->next->next = 0;
   callfuncmacro->next->call_fp = 0;
   callfuncmacro->next->call_filenum = -1;
   /* get parameter list */
   G__getparameterlist(p + 1, &call_para);
   /* translate macro function */
   G__replacefuncmacro(item, callfuncmacro, &call_para, &deffuncmacro->def_para, deffuncmacro->def_fp, deffuncmacro->def_pos, nobraces, nosemic);
   G__freecharlist(&call_para);
   return 1;
}

//______________________________________________________________________________
static int G__argsubstitute(G__FastAllocString &symbol, G__Charlist* callpara, G__Charlist* defpara)
{
   // -- Substitute macro argument.
   while (defpara->next) {
      if (strcmp(defpara->string, symbol) == 0) {
         if (callpara->string) symbol = callpara->string;
         else {
            /* Line number is not quite correct in following error messaging */
            G__genericerror("Error: insufficient number of macro arguments");
            symbol[0] = 0;
         }
         break;
      }
      defpara = defpara->next;
      callpara = callpara->next;
   }
   return(0);
}

//______________________________________________________________________________
static int G__getparameterlist(char* paralist, G__Charlist* charlist)
{
   // -- FIXME: Describe this function!
   int isrc;
   G__FastAllocString string(G__ONELINE);
   int c;
   charlist->string = 0;
   charlist->next = 0;
   c = ',';
   isrc = 0;
   while (',' == c || ' ' == c) {
      c = G__getstream_template(paralist, &isrc, string, 0, " \t,)\0");
      if (c == '\t') c = ' ';
      if (charlist->string)
         charlist->string = (char*) realloc(charlist->string, strlen(charlist->string) + strlen(string) + 2);
      else {
         charlist->string = (char*)malloc(strlen(string) + 2);
         charlist->string[0] = '\0';
      }
      strcat(charlist->string, string); // Okay we just allocated enough space
      if (c == ' ') {
         if (charlist->string[0] != '\0')
            strcat(charlist->string, " ");  // Okay we just allocated enough space
      }
      else {
         size_t i = strlen(charlist->string);
         while (i > 0 && charlist->string[i-1] == ' ') {
            --i;
         }
         charlist->next = (struct G__Charlist*) malloc(sizeof(struct G__Charlist));
         charlist->next->next = 0;
         charlist->next->string = 0;
         charlist = charlist->next;
      }
   }
   return 0;
}

//______________________________________________________________________________
//
//  External Functions.
//

//______________________________________________________________________________
void G__define()
{
   // -- Handle #define.
   //
   // Note: This routine is part of the parser proper.
   //
   //  #define [NAME] [VALUE] \n => G__letvariable("NAME","VALUE")
   //
   G__FastAllocString new_name(G__ONELINE);
   G__FastAllocString initvalue(G__ONELINE);
   G__value evalval;
   int c;
   fpos_t pos;
   //
   //  #define   macro   value
   //          ^
   // read macro name
   //
   c = G__fgetname(new_name, 0, "(\n\r\\");
   //
   //  #define   macro   value
   //                  ^
   //
   //
   // Function macro handled elsewhere.
   //
   if (c == '(') {
      G__createfuncmacro(new_name);
      G__fignoreline();
      return;
   }
   if (c == '\\') {
      fseek(G__ifile.fp, -1, SEEK_CUR);
   }
   //
   // if
   //  #define   macro\n
   //                   ^
   //  #define   macro    value  \n
   //                  ^
   // no value , don't read
   //
   if ((c != '\n') && (c != '\r')) {
      // -- We have a value.
      // Remember position in case it is too hard for us to handle.
      fgetpos(G__ifile.fp, &pos);
      // Grab first part.
      c = G__fgetstream(initvalue, 0, "\n\r\\/");
      // Read and remove comments until done.
      while (c == '/') {
         // -- Possible comment coming next.
         // Reach next character.
         c = G__fgetc();
         // Handle possible comment.
         switch (c) {
            case '/':
               // -- C++ style comment, ignore rest of line.
               G__fignoreline();
               // And pretend the line ended, so we exit.
               c = '\n';
               break;
            case '*':
               // -- C style comment, ignore.
               G__skip_comment();
               // Scan in next part.
               c = G__fgetstream(initvalue, strlen(initvalue), "\n\r\\/");
               break;
             default: {
               // -- Not a comment, take character.
               // Accumulate character.
               size_t ilen = strlen(initvalue);
               if ( (ilen+3) > initvalue.Capacity() ) {
                  initvalue.Resize(ilen+3);
               }
               sprintf(initvalue + strlen(initvalue), "/%c", c);  // Okay we resized if needed.
               // Scan in next part.
               c = G__fgetstream(initvalue, strlen(initvalue), "\n\r\\/");
               break;
             }
         }
      }
      //
      //  Check for continuation, if so handle elsewhere.
      //
      if (c == '\\') {
         // -- We have a continuation, handle elsewhere.
         // Rewind file back to begin of value.
         fsetpos(G__ifile.fp, &pos);
         // And handle this elsewhere.
         G__createmacro(new_name, initvalue);
         // done.
         return;
      }
   }
   else {
      // -- Empty value.
      initvalue[0] = '\0';
   }
   //
   //  #define   macro   value \n
   //                            ^
   //  macro over
   //
   if (
      // --
      initvalue[0] &&
      (
         (initvalue[strlen(initvalue)-1] == '*') ||
         !strcmp(initvalue, "int") ||
         !strcmp(initvalue, "short") ||
         !strcmp(initvalue, "char") ||
         !strcmp(initvalue, "long") ||
         !strcmp(initvalue, "unsigned int") ||
         !strcmp(initvalue, "unsigned short") ||
         !strcmp(initvalue, "unsignedchar") ||
         !strcmp(initvalue, "unsigned long") ||
         !strcmp(initvalue, "signedint") ||
         !strcmp(initvalue, "signedshort") ||
         !strcmp(initvalue, "signedchar") ||
         !strcmp(initvalue, "signedlong") ||
         !strcmp(initvalue, "double") ||
         !strcmp(initvalue, "float") ||
         !strcmp(initvalue, "long double") ||
         (G__defined_typename(initvalue) != -1) ||
         (G__defined_tagname(initvalue, 2) != -1) ||
         G__defined_templateclass(initvalue)
      )
   ) {
      // -- We have #define newtype type*, handle as typedef type* newtype;.
      evalval = G__null;
   }
   else {
      evalval = G__calc_internal(initvalue);
   }
   if ((evalval.type == G__null.type) && initvalue[0]) {
      // -- We have #define newtype oldtype, handle as typedef oldtype newtype;.
      G__handle_as_typedef(initvalue, new_name);
   }
   else {
      // -- Define as an automatic variable.
      // Save state.
      int save_def_struct_member = G__def_struct_member;
      //
      //  Do the variable assignment.
      //
      G__def_struct_member = 0;
      G__var_type = 'p';
      G__typenum = -1;
      G__tagnum = -1;
      G__macro_defining = 1;
      G__letvariable(new_name, evalval, &G__global, G__p_local);
      //
      //  Restore state.
      //
      G__macro_defining = 0;
      G__def_struct_member = save_def_struct_member;
   }
}

//______________________________________________________________________________
G__value G__execfuncmacro(const char* item, int* done)
{
   // -- Execute function macro, balanced braces are required.
   //
   // input  char *item :  macro(para,para)
   // output int *done  :  1 if macro function called, 0 if no macro found
   //
   //
   //  Separate macro func name.
   //
   G__FastAllocString funcmacro(item);
   char* p = strchr(funcmacro, '(');
   if (p) *p = '\0';

   //
   //  Hash the name.
   //
   int hash = 0;
   int i = 0;
   G__hash(funcmacro, hash, i)
   //
   //  Search for macro func name.
   //
   int found = 0;
   struct G__Deffuncmacro* deffuncmacro = &G__deffuncmacro;
   while (deffuncmacro->next) {
      if ((hash == deffuncmacro->hash) && !strcmp(funcmacro, deffuncmacro->name)) {
         found = 1;
         break;
      }
      deffuncmacro = deffuncmacro->next;
   }
   //
   //  If not found, then return.
   //
   if (!found) {
      *done = 0;
      return G__null;
   }
   //
   //  Store calling file pointer and position.
   //
   struct G__input_file store_ifile = G__ifile;
   fpos_t call_pos;
   if (G__ifile.fp) {
      fgetpos(G__ifile.fp, &call_pos);
   }
   //
   //  Search for translated macro function.
   //
   found = 0;
   struct G__Callfuncmacro* callfuncmacro = &deffuncmacro->callfuncmacro;
   while (callfuncmacro->next) {
      if (G__ifile.fp && (
         // --
#ifdef G__NONSCALARFPOS
         G__ifile.line_number == callfuncmacro->line && G__ifile.filenum == callfuncmacro->call_filenum
#elif defined(G__NONSCALARFPOS2)
         call_pos.__pos == callfuncmacro->call_pos.__pos && G__ifile.filenum == callfuncmacro->call_filenum
#elif defined(G__NONSCALARFPOS_QNX)
         call_pos._Off == callfuncmacro->call_pos._Off && G__ifile.filenum == callfuncmacro->call_filenum
#else // G__NONSCALARFPOSxxx
         call_pos == callfuncmacro->call_pos && G__ifile.filenum == callfuncmacro->call_filenum
#endif // G__NONSCALARFPOSxxx
         // --
      ) ) {
         found = 1;
         break;
      }
      callfuncmacro = callfuncmacro->next;
   }
   //
   //  If not found, do substitute macro.
   //
   if (!found) {
      G__transfuncmacro(item, deffuncmacro, callfuncmacro, call_pos, p, 0, 0);
   }
   //
   //  Set macro file.
   //
   G__ifile.fp = G__mfp;
   fsetpos(G__ifile.fp, &callfuncmacro->mfp_pos);
   G__strlcpy(G__ifile.name, G__macro, G__MAXFILENAME);
   //
   //  Execute macro function.
   //
   G__nobreak = 1;
   int brace_level = 0;
   G__value result = G__exec_statement(&brace_level);
   G__nobreak = 0;
   //
   //  Restore source file information.
   //
   G__ifile = store_ifile;
   if (G__ifile.filenum >= 0)
      G__security = G__srcfile[G__ifile.filenum].security;
   else
      G__security = G__SECURE_LEVEL0;
   if (G__ifile.fp) {
      fsetpos(G__ifile.fp, &call_pos);
   }
   //
   //  We are done.
   //
   *done = 1;
   return result;
}

//______________________________________________________________________________
int G__execfuncmacro_noexec(const char* macroname)
{
   // -- Execute function macro with possibly unbalanced braces.
   //
   // input  char *item :  `macro('
   // returns 1 if macro function called, 0 if no macro found
   //
   //
   //  Separate macro func name.
   //
   G__FastAllocString funcmacro(macroname);
   char* p = strchr(funcmacro, '(');
   if (p) {
      *p = '\0';
   }
   else {
      if (G__dispmsg >= G__DISPWARN) {
         G__fprinterr(G__serr, "Warning: %s  Syntax error???", macroname);
         G__printlinenum();
      }
   }
   //
   //  Hash the name.
   //
   int hash = 0;
   int hash_i = 0;
   G__hash(funcmacro, hash, hash_i)
   //
   //  Search for macro func name.
   //
   int found = 0;
   struct G__Deffuncmacro* deffuncmacro = &G__deffuncmacro;
   while (deffuncmacro->next) {
      if ((hash == deffuncmacro->hash) && !strcmp(funcmacro, deffuncmacro->name)) {
         found = 1;
         break;
      }
      deffuncmacro = deffuncmacro->next;
   }
   //
   //  If not found, then return.
   //
   if (!found) {
      // --
      return 0;
   }
   //
   //  Snarf the arg list.
   //
   *p = '(';
   char c = G__fgetstream_spaces(funcmacro, p - funcmacro.data() + 1, ")");
   size_t i = strlen(funcmacro);
   funcmacro.Resize(i + 2);
   funcmacro[i++] = c;
   funcmacro[i] = '\0';
   //
   //  Store calling file pointer and position.
   //
   struct G__input_file store_ifile = G__ifile;
   fpos_t call_pos;
   if (G__ifile.fp) {
      fgetpos(G__ifile.fp, &call_pos);
   }
   //
   //  Search for translated macro function.
   //
   found = 0;
   struct G__Callfuncmacro* callfuncmacro = &deffuncmacro->callfuncmacro;
   while (callfuncmacro->next) {
      // --
      if (G__ifile.fp && (
         // --
#if defined(G__NONSCALARFPOS)
         G__ifile.line_number == callfuncmacro->line && G__ifile.filenum == callfuncmacro->call_filenum
#elif defined(G__NONSCALARFPOS2)
         call_pos.__pos == callfuncmacro->call_pos.__pos && G__ifile.filenum == callfuncmacro->call_filenum
#elif defined(G__NONSCALARFPOS_QNX)
         call_pos._Off == callfuncmacro->call_pos._Off && G__ifile.filenum == callfuncmacro->call_filenum
#else // G__NONSCALARFPOSxxx
         call_pos == callfuncmacro->call_pos && G__ifile.filenum == callfuncmacro->call_filenum
#endif // G__NONSCALARFPOSxxx
         // --
      ) ) {
         // --
         found = 1;
         break;
      }
      callfuncmacro = callfuncmacro->next;
   }
   //
   //  If not found, do substitute macro.
   //
   if (!found || (G__ifile.filenum > G__gettempfilenum())) {
      G__transfuncmacro(macroname, deffuncmacro, callfuncmacro, call_pos, p, 1, 1);
   }
   //
   //  Push onto the macro stack.
   //
   struct G__funcmacro_stackelt* stackelt = (struct G__funcmacro_stackelt*) malloc(sizeof(struct G__funcmacro_stackelt));
   if (G__ifile.fp) {
      stackelt->pos = call_pos;
   }
   stackelt->file = store_ifile;
   stackelt->next = G__funcmacro_stack;
   G__funcmacro_stack = stackelt;
   //
   //  Set macro file.
   //
   G__ifile.fp = G__mfp;
   fsetpos(G__ifile.fp, &callfuncmacro->mfp_pos);
   G__strlcpy(G__ifile.name, G__macro, G__MAXFILENAME);
   // Why not just call G__exec_statement recursively, i hear you ask,
   // instead of introducing this grotty funcstack stuff?
   // Because i want to allow funcmacros containing unbalanced
   // expressions, such as
   //   #define BEGIN_NS(N) namespace N {
   //   #define END_NS(N)   }
   //
   return 1;
}

//______________________________________________________________________________
int G__maybe_finish_macro()
{
   // -- Pop the current macro, if we're executing one.  Called at EOF.  Matches execfuncmacro_noexec.
   //
   // Returns 1 if we were executing a macro, 0 otherwise.
   //
   if (G__funcmacro_stack && (G__ifile.fp == G__mfp)) {
      // -- Pop the macro stack.
      struct G__funcmacro_stackelt* stackelt = G__funcmacro_stack;
      G__ifile = stackelt->file;
      if (G__ifile.fp) {
         fsetpos(G__ifile.fp, &stackelt->pos);
      }
      G__funcmacro_stack = stackelt->next;
      free(stackelt);
      return 1;
   }
   return 0;
}

//______________________________________________________________________________
int G__freedeffuncmacro(G__Deffuncmacro* deffuncmacro)
{
   // -- Free a deffuncmacro list.
   G__Deffuncmacro* dfmp = deffuncmacro;
   if (dfmp->name) {
      free(dfmp->name);
      dfmp->name = 0;
   }
   dfmp->def_fp = 0;
   G__freecharlist(&dfmp->def_para);
   G__Callfuncmacro* cfmp = &dfmp->callfuncmacro;
   cfmp->call_fp = 0;
   {
      G__Callfuncmacro* next = cfmp->next;
      cfmp->next = 0;
      cfmp = next;
   }
   while (cfmp) {
      cfmp->call_fp = 0;
      G__Callfuncmacro* next = cfmp->next;
      cfmp->next = 0;
      free(cfmp);
      cfmp = next;
   }
   {
      G__Deffuncmacro* next = dfmp->next;
      dfmp->next = 0;
      dfmp = next;
   }
   while (dfmp) {
      if (dfmp->name) {
         free(dfmp->name);
         dfmp->name = 0;
      }
      dfmp->def_fp = 0;
      G__freecharlist(&dfmp->def_para);
      G__Callfuncmacro* cfmp = &dfmp->callfuncmacro;
      cfmp->call_fp = 0;
      {
         G__Callfuncmacro* next = cfmp->next;
         cfmp->next = 0;
         cfmp = next;
      }
      while (cfmp) {
         cfmp->call_fp = 0;
         G__Callfuncmacro* next = cfmp->next;
         cfmp->next = 0;
         free(cfmp);
         cfmp = next;
      }
      {
         G__Deffuncmacro* next = dfmp->next;
         dfmp->next = 0;
         free(dfmp);
         dfmp = next;
      }
   }
   return 0;
}

//______________________________________________________________________________
int G__freecharlist(G__Charlist* charlist)
{
   // -- Free a charlist list.
   G__Charlist* p = charlist;
   if (p->string) {
      free(p->string);
      p->string = 0;
   }
   G__Charlist* next = p->next;
   p->next = 0;
   p = next;
   while (p) {
      if (p->string) {
         free(p->string);
         p->string = 0;
      }
      G__Charlist* next = p->next;
      p->next = 0;
      free(p);
      p = next;
   }
   return 0;
}

//______________________________________________________________________________
//
//  Functions in the C interface.
//

} /* extern "C" */

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
