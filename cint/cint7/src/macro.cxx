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
using namespace Cint::Internal;

//______________________________________________________________________________
namespace Cint { 
   namespace Internal
{
   struct G__funcmacro_stackelt {
   private:
      G__funcmacro_stackelt();
   public:
      G__funcmacro_stackelt(fpos_t p,G__input_file&f,G__funcmacro_stackelt*n)
         : pos(p),file(f),next(n) {}
      fpos_t pos;   
      struct G__input_file file;
      struct G__funcmacro_stackelt* next;
   };

   G__funcmacro_stackelt* G__funcmacro_stack = NULL;
}
}

/***********************************************************************
* int G__handle_as_typedef(oldtype,newtype)
***********************************************************************/
int Cint::Internal::G__handle_as_typedef(char *oldtype,char *newtype)
{
   int type='\0' , tagnum = -1 ,ispointer=0 , isunsigned;
   ::Reflex::Type typenum;
   char *p, *ptype;

   p = strchr(oldtype,'*');
   if(p) {
      ispointer = 'A'-'a';
      *p='\0';
   }
   else {
      ispointer=0;
   }

   if(strncmp(oldtype,"unsigned",8)==0) {
      ptype=oldtype+8;
      isunsigned = -1; /* 0 */
   }
   else if(strncmp(oldtype,"signed",6)==0) {
      ptype=oldtype+6;
      isunsigned = 0;
   }
   else {
      ptype=oldtype;
      isunsigned = 0; /* -1 */
   }

   if(strcmp(ptype,"int")==0) {
      type='i'+ispointer+isunsigned;
   }
   else if(strcmp(ptype,"char")==0) {
      type='c'+ispointer+isunsigned;
   }
   else if(strcmp(oldtype,"double")==0) {
      type='d'+ispointer; /* bug fix */
   }
   else if(strcmp(oldtype,"longlong")==0) {
      type='n'+ispointer; 
   }
   else if(strcmp(oldtype,"unsignedlonglong")==0) {
      type='m'+ispointer; 
   }
   else if(strcmp(oldtype,"longdouble")==0) {
      type='q'+ispointer; 
   }
   else if(strcmp(ptype,"short")==0) {
      type='s'+ispointer+isunsigned;
   }
   else if(strcmp(ptype,"long")==0) {
      type='l'+ispointer+isunsigned;
   }
   else if(strcmp(oldtype,"float")==0) {
      type='f'+ispointer;
   }
   else if(strcmp(oldtype,"bool")==0) {
      type='g'+ispointer;
   }
   else if(strncmp(oldtype,"struct",6)==0) {
      ptype=oldtype+6;
      type='u'+ispointer;
      tagnum=G__defined_tagname(ptype,0);
   } 
   else if(strncmp(oldtype,"class",5)==0) {
      ptype=oldtype+5;
      type='u'+ispointer;
      tagnum=G__defined_tagname(ptype,0);
   }
   else if(strncmp(oldtype,"enum",4)==0) {
      ptype=oldtype+4;
      type='i'+ispointer;
      tagnum=G__defined_tagname(ptype,0);
   }
   else {
      tagnum=G__defined_tagname(oldtype,1);
      if(tagnum>=0) {
         type = 'u'+ispointer;
      }
      else {
         typenum=G__find_typedef(oldtype);
         if(typenum) {
            type = G__get_type(typenum);
            tagnum=G__get_tagnum(typenum);
         }
      }
   }

   /* this is only workaround for STL Allocator */
   if(strcmp(newtype,"Allocator")==0) {
      strcpy(G__Allocator,oldtype);
   }
   else if(strcmp(newtype,"vector")==0) {
   }
   else if(strcmp(newtype,"list")==0) {
   }
   else if(strcmp(newtype,"deque")==0) {
   }
   else if(strcmp(newtype,"rb_tree")==0) {
   }
   else
      if(type) {
         if(strcmp(newtype,"bool")!=0) {
            if(G__dispmsg>=G__DISPNOTE) {
               G__fprinterr(G__serr,"Note: macro handled as typedef %s %s;"
                  ,oldtype,newtype);
               G__printlinenum();
            }
         }
         G__search_typename(newtype,type,tagnum,0);
      }
      else {
         G__add_replacesymbol(newtype,oldtype);
#if G__NEVER
         if(G__dispmsg>=G__DISPNOTE) {
            G__fprinterr(G__serr,"Note: #define %s %s",newtype,oldtype);
            G__printlinenum();
         }
#endif
      }

      return(0);
}

/**************************************************************************
* G__createmacro()
**************************************************************************/
void Cint::Internal::G__createmacro(const char *new_name,char *initvalue)
{
   // -- Handle #define MYMACRO ...\<EOL>
   //                   ...\<EOL>
   //                   ...
   //
   // Note: This routine is part of the parser proper.
   //
   G__StrBuf line_sb(G__ONELINE);
   char *line = line_sb;
   int c;
   char *p,*null_fgets;
   fpos_t pos;
   G__value evalval = G__null;

   /* Set flag that there is a macro or template in the source file,
   * so that this file won't be closed even with -cN option */
   ++G__macroORtemplateINfile;

   if(G__mfp==NULL) {
#ifdef G__DEBUG
      G__fprinterr(G__serr,"Limitation: This form of macro may not be expanded. Use +P or -p option");
      G__printlinenum();
#endif
      G__openmfp();
      fgetpos(G__mfp,&G__nextmacro);
      G__mline=1;
   }
   else {
      fsetpos(G__mfp,&G__nextmacro);
   }

   /* print out header */
   ++G__mline;
   fprintf(G__mfp,"// #define %s  FILE:%s LINE:%d\n"
      ,new_name
      ,G__ifile.name,G__ifile.line_number);

   fgetpos(G__mfp,&pos);

   fprintf(G__mfp,"# %d\n",++G__mline);
   ++G__mline;
   fprintf(G__mfp,"{\n");

   fprintf(G__mfp,"%s\n",initvalue);

   /* translate macro */
   int start_line = G__ifile.line_number;
   do {
      null_fgets=fgets(line,G__ONELINE,G__ifile.fp);
      if(null_fgets==NULL) {
         G__fprinterr(G__serr, "Error: Missing newline at or after line %d.\n", start_line);
         G__unexpectedEOF("G__createmacro()");
      }
      ++G__ifile.line_number;
      p=strchr(line,'\n');
      if(p) {
         *p = '\0';
      }
      p=strchr(line,'\r');
      if(p) {
         *p = '\0';
      }
      p = line + strlen (line);
      c='\n';
      if(*(p-1)=='\\') {
         *(p-1) = '\0';
         c='\\';
      }
      if(G__dispsource) {
         G__fprinterr(G__serr,"\\\n%-5d",G__ifile.line_number);
         G__fprinterr(G__serr,"%s",line);
      }
      ++G__mline;
      fprintf(G__mfp,"%s\n",line);
   } while(c!='\n' && c!='\r' );

   p = strrchr(line,';');
   ++G__mline;
   if(p==NULL) {
      fprintf(G__mfp,";}\n");
   }
   else {
      fprintf(G__mfp,"}\n");
   }
   fgetpos(G__mfp,&G__nextmacro);

#ifndef G__OLDIMPLEMENTATION2191
   G__var_type = 'j';
#else
   G__var_type = 'm';
#endif
   G__typenum = ::Reflex::Type();
   G__tagnum = ::Reflex::Scope();
   evalval.obj.i = (long)(&pos);
   {
      int save_def_struct_member = G__def_struct_member;
      G__def_struct_member=0;
      G__letvariable(new_name,evalval,::Reflex::Scope::GlobalScope(),G__p_local);
      G__var_type = 'p';
      G__def_struct_member=save_def_struct_member;
   }
}

/**************************************************************************
* G__createfuncmacro()
*
* Create deffuncmacro list when prerun
*
**************************************************************************/
int Cint::Internal::G__createfuncmacro(const char *new_name)
{
   // -- Handle #define MYMACRO(...,...,...) ...
   //
   // Note: This routine is part of the parser proper.
   //
   struct G__Deffuncmacro *deffuncmacro;
   int hash,i;
   G__StrBuf paralist_sb(G__ONELINE);
   char *paralist = paralist_sb;
   int c;

   if(G__ifile.filenum>G__gettempfilenum()) {
      G__fprinterr(G__serr,"Limitation: Macro function can not be defined in a command line or a tempfile\n");
      G__genericerror("You need to write it in a source file");
      G__fprinterr(G__serr,"Besides, it is recommended to use function template instead\n");
      return (-1);
   }

   /* Set flag that there is a macro or template in the source file,
   * so that this file won't be closed even with -cN option */
   ++G__macroORtemplateINfile;

   /* Search for the end of list */
   deffuncmacro = &G__deffuncmacro;
   /*
   G__fprinterr(G__serr,"Limitation: Macro %s() may not work well" ,new_name);
   G__printlinenum();
   */
   while(deffuncmacro->next) deffuncmacro=deffuncmacro->next;

   /* store name */
   deffuncmacro->name = (char*)malloc(strlen(new_name)+1);
   strcpy(deffuncmacro->name,new_name);

   /* store hash */
   G__hash(new_name,hash,i)
      deffuncmacro->hash=hash;

   /* read parameter list */
   c = G__fgetstream(paralist,")");
   G__ASSERT(')'==c);
   G__getparameterlist(paralist,&deffuncmacro->def_para);

   /* store file pointer, line number and position */
   deffuncmacro->def_fp = G__ifile.fp;
   fgetpos(G__ifile.fp,&deffuncmacro->def_pos);
   deffuncmacro->line = G__ifile.line_number;


   /* allocate and initialize next list */
   deffuncmacro->next = new G__Deffuncmacro;

   /* after this, source file is read to end of line */

   return(0);
}

/**************************************************************************
* G__replacefuncmacro()
*
* Replace function macro parameter at the first execution of func macro
*
**************************************************************************/
int Cint::Internal::G__replacefuncmacro(const char *item,G__Callfuncmacro *callfuncmacro
                                        ,G__Charlist *callpara,G__Charlist *defpara
                                        ,FILE *def_fp,fpos_t def_pos
                                        ,int nobraces,int nosemic
                                        )
{
   // -- Replace function macro parameter at the first execution of func macro.
   fpos_t pos;
   int c;
   int semicolumn;
   G__StrBuf symbol_sb(G__ONELINE);
   char *symbol = symbol_sb;
   const char *punctuation = " \t\n;:=+-)(*&^%$#@!~'\"\\|][}{/?.>,<";
   int double_quote=0,single_quote=0;
   fpos_t backup_pos;

   if(G__mfp==NULL) {
#ifdef G__DEBUG
      G__fprinterr(G__serr,"Limitation: This form of macro may not be expanded. Use +P or -p option");
      G__printlinenum();
#endif
      G__openmfp();
      fgetpos(G__mfp,&G__nextmacro);
      G__mline=1;
   }
   else {
      fsetpos(G__mfp,&G__nextmacro);
   }

   /* print out header */
   ++G__mline;
   fprintf(G__mfp,"// #define %s  FILE:%s LINE:%d\n"
      ,item
      ,G__ifile.name,G__ifile.line_number);

   fgetpos(G__mfp,&pos);
   callfuncmacro->mfp_pos = pos;

   fprintf(G__mfp,"# %d\n",++G__mline);
   ++G__mline;
   fprintf(G__mfp,"%s\n", nobraces ? "" : "{" );

   /*******************************************************************
   * read macro definition and substitute symbol
   *******************************************************************/
   /* set file pointer and position */
   G__ifile.fp = def_fp;
   fsetpos(def_fp,&def_pos);

   /* read definition and substitute */
   fgetpos(G__mfp,&backup_pos);
   semicolumn = 0;
   while(1) {
      G__disp_mask = 10000;
      c = G__fgetstream(symbol,punctuation);
      if('\0' != symbol[0]) {
         if(0==double_quote && 0==single_quote)
            G__argsubstitute(symbol,callpara,defpara);
         fprintf(G__mfp,"%s",symbol);
         fgetpos(G__mfp,&backup_pos);
         semicolumn = 0;
      }

      if(0==single_quote && 0==double_quote) {
         if('\n'==c || '\r'==c) break;
         if('\\'==c) {
            c=G__fgetc();
            /* Things got very complicated here, No one is sure which of
            * 973, 969, 948 is good */
            if('\n'==c) continue;
            if('\r'==c) c=G__fgetc();
         }

         if(';'==c) {
            semicolumn = 1;
         }
         else if(!isspace(c)) {
            semicolumn = 0;
         }
         if (c == '#') {
            c = G__fgetc ();
            if (c == '#') {
               /* Token paste operation */
               fsetpos (G__mfp, &backup_pos);
               G__fgetspace ();
               fseek(G__ifile.fp,-1,SEEK_CUR);
               continue;
            }
            else {
               fseek(G__ifile.fp,-1,SEEK_CUR);
            }
         }
      }

      if('\''==c && 0==double_quote) 
         single_quote = single_quote ^ 1 ;

      else if('"'==c && 0==single_quote) 
         double_quote = double_quote ^ 1 ;

      fputc(c,G__mfp);
      if (!isspace (c)) fgetpos (G__mfp, &backup_pos);
      if('\n'==c) ++G__mline;
   } 

   /* finishing up */
   G__disp_mask=0;
   if(!nosemic && 0==semicolumn) fprintf(G__mfp," ;");
   G__mline+=2;
   fprintf(G__mfp,"\n%s\n" , nobraces ? "" : "}");
   fputc('\0',G__mfp); /* Mark the end of this expansion. */
   fgetpos(G__mfp,&G__nextmacro);

   fflush(G__mfp);

   return(0);  
}

/**************************************************************************
* G__transfuncmacro()
*
* translate function macro parameter at the first execution of func macro
*
**************************************************************************/
int Cint::Internal::G__transfuncmacro(const char *item,G__Deffuncmacro *deffuncmacro
                                      ,G__Callfuncmacro *callfuncmacro,fpos_t call_pos
                                      ,char *p
                                      ,int nobraces,int nosemic
                                      )
{
   // -- Translate function macro parameter at the first execution of func macro.
   struct G__Charlist call_para;

   /* set file pointer and position */
   callfuncmacro->call_fp = G__ifile.fp;
   callfuncmacro->call_filenum = G__ifile.filenum;
   callfuncmacro->call_pos = call_pos;
   callfuncmacro->line = G__ifile.line_number;

   /* allocate and initialize next list */
   callfuncmacro->next = new G__Callfuncmacro;
   
   /* get parameter list */
   G__getparameterlist(p+1,&call_para);

   /* translate macro function */
   G__replacefuncmacro(item
      ,callfuncmacro
      ,&call_para
      ,&deffuncmacro->def_para
      ,deffuncmacro->def_fp
      ,deffuncmacro->def_pos
      ,nobraces,nosemic
      );

   G__freecharlist(&call_para);

   return(1);
}

/**************************************************************************
* G__argsubstitute()
*
* Substitute macro argument
*
**************************************************************************/
int Cint::Internal::G__argsubstitute(char *symbol,G__Charlist *callpara,G__Charlist *defpara)
{
   // -- Substitute macro argument.
   while(defpara->next) {
      if(strcmp(defpara->string,symbol)==0) {
         if(callpara->string) strcpy(symbol,callpara->string);
         else {
            /* Line number is not quite correct in following error messaging */
            G__genericerror("Error: insufficient number of macro arguments");
            symbol[0] = 0;
         }
         break;
      }
      defpara = defpara->next;
      callpara=callpara->next;
   }
   return(0);
}


/**************************************************************************
* G__getparameterlist()
**************************************************************************/
int Cint::Internal::G__getparameterlist(char *paralist, G__Charlist *charlist)
{
   // -- FIXME: Describe this function!
   int isrc;
   G__StrBuf string_sb(G__ONELINE);
   char *string = string_sb;
   int c;

   charlist->string = (char*)NULL;
   charlist->next = (struct G__Charlist*)NULL;

   c=',';
   isrc=0;
   while(','==c || ' ' == c) {
      c = G__getstream_template(paralist,&isrc,string," \t,)\0");
      if (c == '\t') c = ' ';

      if (charlist->string)
         charlist->string = (char*) realloc (charlist->string,
                                             strlen (charlist->string) +
                                             strlen (string) + 2);
      else {
         charlist->string = (char*)malloc(strlen(string)+2);
         charlist->string[0] = '\0';
      }
      strcat (charlist->string, string);
      if (c == ' ') {
         if (charlist->string[0] != '\0')
            strcat (charlist->string, " ");
      }
      else {
         int i = strlen (charlist->string);
         while (i > 0 && charlist->string[i-1] == ' ')
            --i;
         charlist->next = new G__Charlist;
         charlist = charlist->next;
      }
   }

   return(0);
}


/******************************************************************
* void G__define()
*
* Called by
*   G__exec_statement(&brace_level);
*
******************************************************************/
void Cint::Internal::G__define()
{
   // -- Handle #define.
   //
   // Note: This routine is part of the parser proper.
   //
   //  #define [NAME] [VALUE] \n => G__letvariable("NAME","VALUE")
   //

   char new_name[G__ONELINE],initvalue[G__ONELINE];
   G__value evalval;
   int c;
   /* int flag; */
   fpos_t pos;
   /* int inew=0; */

   /*
   *  #define   macro   value
   *          ^
   * read macro name
   */

   c=G__fgetname(new_name,"(\n\r\\");

   /*
   *  #define   macro   value
   *                  ^
   */

   /*
   * function macro not supported
   */
   if(c=='(') {
      G__createfuncmacro(new_name);
      G__fignoreline();
      return;
   }

   if (c == '\\') {
      fseek(G__ifile.fp,-1,SEEK_CUR);
   }

   /* if
   *  #define   macro\n
   *                   ^
   *  #define   macro    value  \n
   *                  ^
   * no value , don't read
   */
   if(c!='\n' && c!='\r') {
      fgetpos(G__ifile.fp,&pos);
      c=G__fgetstream(initvalue,"\n\r\\/");
      while('/'==c) {
         c=G__fgetc();
         switch(c) {
      case '/':
         G__fignoreline();
         c='\n';
         break;
      case '*':
         G__skip_comment();
         c=G__fgetstream(initvalue+strlen(initvalue),"\n\r\\/");
         break;
      default:
         sprintf(initvalue+strlen(initvalue),"/%c",c);
         c=G__fgetstream(initvalue+strlen(initvalue),"\n\r\\/");
         break;
         }
      }
      if(c=='\\') {
         fsetpos(G__ifile.fp,&pos);
         G__createmacro(new_name,initvalue);
         return;
      }
   }
   else {
      initvalue[0]='\0';
   }


   /*
   *  #define   macro   value \n
   *                            ^
   *  macro over
   */

   if(initvalue[0] && 
      ('*'==initvalue[strlen(initvalue)-1] ||
      strcmp(initvalue,"int")==0 ||
      strcmp(initvalue,"short")==0 ||
      strcmp(initvalue,"char")==0 ||
      strcmp(initvalue,"long")==0 ||
      strcmp(initvalue,"unsignedint")==0 ||
      strcmp(initvalue,"unsignedshort")==0 ||
      strcmp(initvalue,"unsignedchar")==0 ||
      strcmp(initvalue,"unsignedlong")==0 ||
      strcmp(initvalue,"signedint")==0 ||
      strcmp(initvalue,"signedshort")==0 ||
      strcmp(initvalue,"signedchar")==0 ||
      strcmp(initvalue,"signedlong")==0 ||
      strcmp(initvalue,"double")==0 ||
      strcmp(initvalue,"float")==0 ||
      strcmp(initvalue,"longdouble")==0 ||
      G__find_typedef(initvalue)||
      -1!=G__defined_tagname(initvalue,2)||
      G__defined_templateclass(initvalue))) {
         /* '#define newtype type *'
         * handle as 'typedef type* newtype;'
         */
         evalval=G__null;
   }
   else {
      evalval=G__calc_internal(initvalue);
   }

   if(G__value_typenum(G__null)==G__value_typenum(evalval) && initvalue[0]) {
      /* #define newtype oldtype
      * handled as typedef oldtype newtype;
      */
      G__handle_as_typedef(initvalue,new_name);
   }
   else {
      /* define as an automatic variable */
      int save_def_struct_member = G__def_struct_member;
      G__def_struct_member = 0;
      G__var_type = 'p';
      G__typenum = ::Reflex::Type();
      G__tagnum = ::Reflex::Scope::GlobalScope();
      G__macro_defining=1;
      G__letvariable(new_name,evalval,::Reflex::Scope::GlobalScope(),G__p_local);
      G__macro_defining=0;
      G__def_struct_member = save_def_struct_member;
   }
}

/**************************************************************************
* G__execfuncmacro()
*
* Execute function macro 
*
*  input  char *item :  macro(para,para)
*  output int *done  :  1 if macro function called, 0 if no macro found
*
**************************************************************************/
G__value Cint::Internal::G__execfuncmacro(const char *item,int *done)
{
   G__value result;
   struct G__Deffuncmacro *deffuncmacro;
   struct G__Callfuncmacro *callfuncmacro;
   char *p;
#ifndef G__OLDIMPLEMENTATION1823
   G__StrBuf buf_sb(G__ONELINE);
   char *buf = buf_sb;
   char *funcmacro=buf;
#else
   G__StrBuf funcmacro_sb(G__ONELINE);
   char *funcmacro = funcmacro_sb;
#endif
   int hash,i;
   int found;

   fpos_t call_pos;
   struct G__input_file store_ifile;


   /* separate macro func name */
#ifndef G__OLDIMPLEMENTATION1823
   if(strlen(item)>G__ONELINE-10) {
      funcmacro = (char*)malloc(strlen(item)+10);
   }
#else
   G__ASSERT(strlen(item)<G__ONELINE);
#endif
   strcpy(funcmacro,item);
   p = strchr(funcmacro,'(');
   *p='\0';
   /* get hash */
   G__hash(funcmacro,hash,i)

      /* search for macro func name */
      found=0;
   deffuncmacro = &G__deffuncmacro;
   while(deffuncmacro->next) {
      if(hash==deffuncmacro->hash && strcmp(funcmacro,deffuncmacro->name)==0) {
         found=1;
         break;
      }
      deffuncmacro=deffuncmacro->next;
   }

   if(0==found) {
      *done = 0;
#ifndef G__OLDIMPLEMENTATION1823
      if(funcmacro!=buf) delete funcmacro;
#endif
      return (G__null);
   }


   /* store calling file pointer and position */
#ifndef G__OLDIMPLEMENTATION1917
   if(G__ifile.fp) fgetpos(G__ifile.fp,&call_pos);
#else
   fgetpos(G__ifile.fp,&call_pos);
#endif
   store_ifile = G__ifile;


   /* search for translated macro function */
   found=0;
   callfuncmacro = &deffuncmacro->callfuncmacro;
   while(callfuncmacro->next) {
#if defined(G__NONSCALARFPOS)
      if(G__ifile.line_number == callfuncmacro->line &&
         G__ifile.filenum == callfuncmacro->call_filenum) {
#elif defined(G__NONSCALARFPOS2)
      if(call_pos.__pos == callfuncmacro->call_pos.__pos &&
         G__ifile.filenum == callfuncmacro->call_filenum) {
#elif defined(G__NONSCALARFPOS_QNX)
      if(call_pos._Off == callfuncmacro->call_pos._Off &&
         G__ifile.filenum == callfuncmacro->call_filenum) {
#else
      if(call_pos == callfuncmacro->call_pos &&
         G__ifile.filenum == callfuncmacro->call_filenum) {
#endif
            found=1;
            break;
      }
      callfuncmacro = callfuncmacro->next;
   }

   /* substitute macro if not already being done so */
   if(0==found) {
      G__transfuncmacro(item,deffuncmacro,callfuncmacro,call_pos,p,0,0);
   }

   /* set macro file */
   G__ifile.fp = G__mfp;
   fsetpos(G__ifile.fp,&callfuncmacro->mfp_pos);
   strcpy(G__ifile.name,G__macro);

   /* execute macro function */
   G__nobreak=1;
   int brace_level = 0;
   result=G__exec_statement(&brace_level);
   G__nobreak=0;

   /* re-store source file information */
   G__ifile = store_ifile;
   if(G__ifile.filenum>=0) 
      G__security = G__srcfile[G__ifile.filenum].security;
   else
      G__security = G__SECURE_LEVEL0;
#ifndef G__OLDIMPLEMENTATION1917
   if(G__ifile.fp) fsetpos(G__ifile.fp,&call_pos);
#else
   fsetpos(G__ifile.fp,&call_pos);
#endif

   *done = 1;
#ifndef G__OLDIMPLEMENTATION1823
   if(funcmacro!=buf) delete funcmacro;
#endif
   return(result);
}

/**************************************************************************
* G__execfuncmacro_noexec()
*
* Execute function macro in noexec mode.
*
*  input  char *item :  `macro('
*  returns 1 if macro function called, 0 if no macro found
*
**************************************************************************/
int Cint::Internal::G__execfuncmacro_noexec (const char* macroname)
{
   //
   //  Separate macro func name.
   //
   struct G__Deffuncmacro *deffuncmacro;
   struct G__Callfuncmacro *callfuncmacro;
   char *p;
#ifndef G__OLDIMPLEMENTATION1823
   G__StrBuf buf_sb(G__ONELINE);
   char *buf = buf_sb;
   char *funcmacro=buf;
#else
   G__StrBuf funcmacro_sb(G__ONELINE);
   char *funcmacro = funcmacro_sb;
#endif
   int hash,i,c;
   int found;

   fpos_t call_pos;
   struct G__input_file store_ifile;
   G__funcmacro_stackelt* stackelt;


   /* separate macro func name */
#ifndef G__OLDIMPLEMENTATION1823
   if(strlen(macroname)>G__ONELINE-10) {
      funcmacro = (char*)malloc(strlen(macroname)+10);
   }
#else
   G__ASSERT(strlen(macroname)<G__ONELINE);
#endif
   strcpy(funcmacro,macroname);
   p = strchr(funcmacro,'(');
   if(p) *p='\0';
   else {
      if(G__dispmsg>=G__DISPWARN) {
         G__fprinterr(G__serr,"Warning: %s  Syntax error???",macroname);
         G__printlinenum();
      }
   }
   /* get hash */
   G__hash(funcmacro,hash,i)

      /* search for macro func name */
      found=0;
   deffuncmacro = &G__deffuncmacro;
   while(deffuncmacro->next) {
      if(hash==deffuncmacro->hash && strcmp(funcmacro,deffuncmacro->name)==0) {
         found=1;
         break;
      }
      deffuncmacro=deffuncmacro->next;
   }

   if(0==found) {
#ifndef G__OLDIMPLEMENTATION1823
      if(funcmacro!=buf) delete funcmacro;
#endif
      return 0;
   }

   /* Snarf the arg list. */
   *p = '(';
   /* #define G__OLDIMPLEMENTATION1061 */
   c=G__fgetstream_spaces (p+1 ,")");
   i = strlen (funcmacro);
   funcmacro[i++] = c;
   funcmacro[i] = '\0';
#ifdef G__OLDIMPLEMENTATION1823
   G__ASSERT(strlen(funcmacro)<G__ONELINE);
#endif

   /* store calling file pointer and position */
#ifndef G__OLDIMPLEMENTATION1917
   if(G__ifile.fp) fgetpos(G__ifile.fp,&call_pos);
#else
   fgetpos(G__ifile.fp,&call_pos);
#endif
   store_ifile = G__ifile;


   /* search for translated macro function */
   found=0;
   callfuncmacro = &deffuncmacro->callfuncmacro;
   while(callfuncmacro->next) {
#if defined(G__NONSCALARFPOS)
      if(G__ifile.line_number == callfuncmacro->line &&
         G__ifile.filenum == callfuncmacro->call_filenum)
#elif defined(G__NONSCALARFPOS2)
      if(call_pos.__pos == callfuncmacro->call_pos.__pos &&
         G__ifile.filenum == callfuncmacro->call_filenum)
#elif defined(G__NONSCALARFPOS_QNX)
      if(call_pos._Off == callfuncmacro->call_pos._Off &&
         G__ifile.filenum == callfuncmacro->call_filenum)       
#else
      if(call_pos == callfuncmacro->call_pos &&
         G__ifile.filenum == callfuncmacro->call_filenum)
#endif
      { 
         found=1;
         break;
      }
      callfuncmacro = callfuncmacro->next;
   }

   /* substitute macro if not already done so */
   if(0==found
      || G__ifile.filenum > G__gettempfilenum() 
      ) {
         G__transfuncmacro(macroname,deffuncmacro,callfuncmacro,call_pos,p,1,1);
   }

   /* Push onto the macro stack. */
   stackelt = new G__funcmacro_stackelt(call_pos,store_ifile,G__funcmacro_stack);
   G__funcmacro_stack = stackelt;

   /* Jump to the macro. */
   G__ifile.fp = G__mfp;
   fsetpos(G__ifile.fp,&callfuncmacro->mfp_pos);
   strcpy(G__ifile.name,G__macro);

   /* Why not just call G__exec_statement recursively, i hear you ask,
   instead of introducing this grotty funcstack stuff?
   Because i want to allow funcmacros containing unbalanced
   expressions, such as

   #define BEGIN_NS(N) namespace N {
   #define END_NS(N)   }
   */

#ifndef G__OLDIMPLEMENTATION1823
   if(funcmacro!=buf) delete funcmacro;
#endif
   return 1;
}


/**************************************************************************
* G__maybe_finish_macro()
*
* Called at EOF.
* Pop the current macro, if we're executing one.
*
*  returns 1 if we were executing a macro, 0 otherwise.
*
**************************************************************************/
int Cint::Internal::G__maybe_finish_macro ()
{
   if (G__funcmacro_stack != 0 && G__ifile.fp == G__mfp) {
      /* Pop the macro stack. */
      struct G__funcmacro_stackelt* stackelt = G__funcmacro_stack;
      G__ifile = stackelt->file;
#ifndef G__OLDIMPLEMENTATION1917
      if(G__ifile.fp) fsetpos (G__ifile.fp, &stackelt->pos);
#else
      fsetpos (G__ifile.fp, &stackelt->pos);
#endif
      G__funcmacro_stack = stackelt->next;
      free (stackelt);

      return 1;
   }

   return 0;
}


/**************************************************************************
* G__freedeffuncmacro()
**************************************************************************/
int Cint::Internal::G__freedeffuncmacro(G__Deffuncmacro *deffuncmacro)
{
   // -- Free a deffuncmacro list.
   G__Deffuncmacro* dfmp = deffuncmacro;
   if (dfmp->name) {
      free(dfmp->name);
      dfmp->name = 0;
   }
   dfmp->def_fp = 0;
   G__freecharlist(&dfmp->def_para);
   G__Callfuncmacro* outer_cfmp = &dfmp->callfuncmacro;
   outer_cfmp->call_fp = 0;
   {
      G__Callfuncmacro* next = outer_cfmp->next;
      outer_cfmp->next = 0;
      outer_cfmp = next;
   }
   while (outer_cfmp) {
      outer_cfmp->call_fp = 0;
      G__Callfuncmacro* next = outer_cfmp->next;
      outer_cfmp->next = 0;
      delete outer_cfmp;
      outer_cfmp = next;
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
      G__Callfuncmacro* inner_cfmp = &dfmp->callfuncmacro;
      inner_cfmp->call_fp = 0;
      {
         G__Callfuncmacro* next = inner_cfmp->next;
         inner_cfmp->next = 0;
         inner_cfmp = next;
      }
      while (inner_cfmp) {
         inner_cfmp->call_fp = 0;
         G__Callfuncmacro* next = inner_cfmp->next;
         inner_cfmp->next = 0;
         delete (inner_cfmp);
         inner_cfmp = next;
      }
      {
         G__Deffuncmacro* next = dfmp->next;
         dfmp->next = 0;
         delete (dfmp);
         dfmp = next;
      }
   }
   return 0;
}

/**************************************************************************
* G__freecharlist()
**************************************************************************/
int Cint::Internal::G__freecharlist(G__Charlist *charlist)
{
   // -- Free a charlist list.
   G__Charlist* p = charlist;
   if (p->string) {
      free(p->string);
      p->string = 0;
   }
   {
      G__Charlist* next = p->next;
      p->next = 0;
      p = next;
   }
   while (p) {
      if (p->string) {
         free(p->string);
         p->string = 0;
      }
      {
         G__Charlist* next = p->next;
         p->next = 0;
         delete p;
         p = next;
      }
   }
   return 0;
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
