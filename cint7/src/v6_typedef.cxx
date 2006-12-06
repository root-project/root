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
#include "Dict.h"
#include "Reflex/Builder/TypeBuilder.h"
#include <vector>

using namespace Cint::Internal;

/******************************************************************
* G__shiftstring
******************************************************************/
static void G__shiftstring(char *s,int n)
{
  int i=0, j=n;
  while(s[j]) s[i++]=s[j++];
  s[i]=0;
}

/******************************************************************
* G__defined_typename_exact(type_name)
*
* Search already defined typedef names, -1 is returned if not found
******************************************************************/
static ::ROOT::Reflex::Type G__defined_typename_exact(char *type_name)
{
   Reflex::Scope envScope = G__get_envtagnum();
   return envScope.SubTypeByName(type_name);
#ifdef BEFORE_REFLEX
  int i,flag=0,len;
  char ispointer=0;
  char temp[G__LONGLINE];
  char *p;
  char temp2[G__LONGLINE];
  int env_tagnum;
  char *par;

  strcpy(temp2,type_name);

  /* find 'xxx::yyy' */
  p = G__find_last_scope_operator (temp2);

  /* abandon scope operator if 'zzz (xxx::yyy)www' */
  par = strchr(temp2,'(');
  if(par && p && par<p) p=(char*)NULL;

  if(p) {
    strcpy(temp,p+2);
    *p='\0';
    if(temp2==p) env_tagnum = -1; /* global scope */
#ifndef G__STD_NAMESPACE
    else if (strcmp (temp2, "std") == 0
             && G__ignore_stdnamespace
             ) env_tagnum = -1;
#endif
    else         env_tagnum = G__defined_tagname(temp2,0);
  }
  else {
    strcpy(temp,temp2);
    env_tagnum = G__get_envtagnum();
  }

  len=strlen(temp);

  if(temp[len-1]=='*') {
    temp[--len]='\0';
    ispointer = 'A' - 'a';
  }

  for(i=0;i<G__newtype.alltype;i++) {
    if(len==G__newtype.hash[i] && strcmp(G__newtype.name[i],temp)==0 && (
        env_tagnum==G__newtype.parent_tagnum[i]
        )) {
      flag=1;
      /* This must be a bad manner. Somebody needs to reset G__var_type
       * especially when typein is 0. */
      G__var_type=G__newtype.type[i] + ispointer ;
      break;
    }
  }

  if(flag==0) return(-1);
  return(i);
#endif
}

/******************************************************************
* G__define_struct_in_typedef()
*
* Parse and define the struct (or enum) declaration which is 
* part of a typedef.
*
******************************************************************/
static void G__define_struct_in_typedef(fpos_t fpos
                                        ,int /*tagnum*/
                                        ,int env_tagnum
                                        ,char tagtype
                                        /* ,::ROOT::Reflex::Type typenum */
                                        ,int stored_tagnum)
{
   // CHECKME: For now we do not pass the typenum.
   // See the enum section below
   char category[10],c;
   G__value enumval;
   char memname[G__MAXNAME];
   char val[G__ONELINE];
   int mparen,store_def_struct_member=0;

   ::Reflex::Scope store_def_tagnum = G__def_tagnum;
   ::Reflex::Scope store_tagdefining = G__tagdefining;

   fsetpos(G__ifile.fp,&fpos);

   G__struct.line_number[G__get_tagnum(G__tagnum)] = G__ifile.line_number;
   G__struct.filenum[G__get_tagnum(G__tagnum)] = G__ifile.filenum;

   G__struct.parent_tagnum[G__get_tagnum(G__tagnum)]=env_tagnum;

   /*
   * in case of enum
   */
   if(tagtype=='e') {
      G__disp_mask=10000;
      while((c=G__fgetc())!='{') ;
      enumval.obj.i = -1;
      enumval.type = 'i' ;
      enumval.tagnum = G__get_tagnum(G__tagnum);
      //CHECKME: check wether this is needed or not.
      //G__value_typenum(enumval) = typenum ;
      G__constvar=G__CONSTVAR;
      G__enumdef=1;
      do {
         c=G__fgetstream(memname,"=,}");
         if(c=='=') {
            int store_prerun = G__prerun;
            char store_var_type = G__var_type;
            G__var_type = 'p';
            G__prerun = 0;
            c=G__fgetstream(val,",}");
            enumval=G__getexpr(val);
            G__prerun = store_prerun;
            G__var_type = store_var_type;
         }
         else {
            enumval.obj.i++;
         }
         G__var_type='i';
         if(-1!=stored_tagnum) {
            store_def_struct_member=G__def_struct_member;
            G__def_struct_member=0;
            G__static_alloc=1;
         }
         G__letvariable(memname,enumval,&G__global ,G__p_local);
         if(-1!=stored_tagnum) {
            G__def_struct_member=store_def_struct_member;
            G__static_alloc=0;
         }
      } while(c!='}') ;
      G__constvar=0;
      G__enumdef=0;

      G__fignorestream(";");
      G__disp_mask=0;
   } /* end of enum */

   /*
   * in case of struct,union
   */
   else {
      switch(tagtype) {
          case 's':
             sprintf(category,"struct");
             break;
          case 'c':
             sprintf(category,"class");
             break;
          case 'u':
             sprintf(category,"union");
             break;
          default:
             /* enum already handled above */
             G__fprinterr(G__serr,"Error: Illegal tagtype. struct,union,enum expected\n");
             break;
      }

      struct G__var_array * store_local = G__p_local;
      G__p_local=G__struct.memvar[G__get_tagnum(G__tagnum)];

      store_def_struct_member=G__def_struct_member;
      G__def_struct_member=1;
      /* G__prerun = 1; */ /* redundant */
      G__switch = 0; /* redundant */
      mparen = G__mparen;
      G__mparen=0;

      G__disp_mask=10000;
      G__tagdefining=G__tagnum;
      G__def_tagnum = G__tagdefining;
      G__exec_statement();
      G__tagnum=G__tagdefining;
      G__def_tagnum = store_def_tagnum;

      /********************************************
      * Padding for PA-RISC, Spark, etc
      * If struct size can not be divided by G__DOUBLEALLOC
      * the size is aligned.
      ********************************************/
      if(1==G__struct.memvar[G__get_tagnum(G__tagnum)]->allvar) {
         /* this is still questionable, inherit0.c */
         struct G__var_array *v=G__struct.memvar[G__get_tagnum(G__tagnum)];
         if('c'==v->type[0]) {
            if(isupper(v->type[0])) {
               G__struct.size[G__get_tagnum(G__tagnum)] = G__LONGALLOC*(v->varlabel[0][1]+1);
            }
            else {
               G__value buf;
               buf.type = v->type[0];
               buf.tagnum = v->p_tagtable[0];
               G__value_typenum(buf) = v->p_typetable[0];
               G__struct.size[G__get_tagnum(G__tagnum)]
               =G__sizeof(&buf)*(v->varlabel[0][1]+1);
            }
         }
      } else
         if(G__struct.size[G__get_tagnum(G__tagnum)]%G__DOUBLEALLOC) {
            G__struct.size[G__get_tagnum(G__tagnum)]
            += G__DOUBLEALLOC
               - G__struct.size[G__get_tagnum(G__tagnum)]%G__DOUBLEALLOC;
         }
         else if(0==G__struct.size[G__get_tagnum(G__tagnum)]) {
            G__struct.size[G__get_tagnum(G__tagnum)] = G__CHARALLOC;
         }

         G__tagdefining = store_tagdefining;

         G__def_struct_member=store_def_struct_member;
         G__mparen=mparen;
         G__p_local = store_local;

         G__fignorestream(";");

         G__disp_mask=0;
   } /* end of struct, class , union */
}


/******************************************************************
* G__define_type()
*
*  typedef [struct|union|enum] tagname { member } newtype;
*  typedef fundamentaltype   newtype;
*
******************************************************************/

#define G__OLDIMPLEMENTATION63

void Cint::Internal::G__define_type()
/* struct G__input_file *fin; */
{
   fpos_t rewind_fpos;
   int c;
   char type1[G__LONGLINE],tagname[G__LONGLINE],type_name[G__LONGLINE];
   char temp[G__LONGLINE];
   char val[G__ONELINE];
   int isnext;
   fpos_t next_fpos;
   //int /* itag=0,*/ mparen;
   ::Reflex::Scope store_tagnum;
   //struct G__var_array* store_local;
   /* char store_tagname[G__LONGLINE] */
   char type,tagtype=0;
   int unsigned_flag=0 /* ,flag=0 */ ,mem_def=0,temp_line;
   int len,taglen;
   ::Reflex::Scope store_tagdefining;
   int typedef2=0;
   int nindex=0;
   int index[G__MAXVARDIM];
   char aryindex[G__MAXNAME];
   /* int lenheader; */
   char *p;
   int store_var_type;
   ::ROOT::Reflex::Type typenum;
   int isorgtypepointer=0;
   int reftype=G__PARANORMAL;
   int rawunsigned=0;
   ::Reflex::Scope env_tagnum = ::Reflex::Scope::GlobalScope();
   int isconst = 0;
   fpos_t pos_p2fcomment;
   int    line_p2fcomment;
   int    flag_p2f=0;

   tagname[0] = '\0';                /* initialize it */

   fgetpos(G__ifile.fp,&pos_p2fcomment);
   line_p2fcomment = G__ifile.line_number;

#ifdef G__ASM
#ifdef G__ASM_DBG
   if(G__asm_dbg&&G__asm_noverflow)
      G__genericerror(G__LOOPCOMPILEABORT);
#endif
   G__abortbytecode();
#endif

   store_tagnum=G__tagnum;


   /*
   *  typedef  [struct|union|enum] tagname { member } newtype;
   *  typedef  fundamentaltype   newtype;
   *          ^
   * read type
   */

   c=G__fgetname_template(type1,"*{");
   if('*'==c) { 
      strcat(type1,"*");
      c=' ';
   }
   /* just ignore the following 4 keywords as long as they are
   followed by a space */
   while(isspace(c) &&
      (strcmp(type1,"const")==0 ||strcmp(type1,"volatile")==0 ||
      strcmp(type1,"mutable")==0 || strcmp(type1,"typename") == 0)) {
         if(strcmp(type1,"const")==0) isconst |= G__CONSTVAR;
         c=G__fgetname_template(type1,"{");
   }
   if (strcmp(type1,"::")==0) {
      /* skip a :: without a namespace in front of it (i.e. global namespace!) */
      c = G__fgetspace(); /* skip the next ':' */
      c=G__fgetname_template(type1,"{");
   }   
   if (strncmp(type1,"::",2)==0) {
      /* A leading '::' causes other typename matching function to fails so 
      we remove it. This is not the ideal solution (neither was the one 
      above since it does not allow for distinction between global 
      namespace and local namespace) ... but at least it is an improvement
      over the current behavior */
      strcpy(type1,type1+2);
   }
   while( isspace(c) ) {
      len=strlen(type1);
      c = G__fgetspace();
      if(':'==c) {
         c = G__fgetspace(); /* skip the next ':' */
         strcat(type1,"::");
         c=G__fgetname_template(temp,"{");
         strcat(type1,temp);
      } else if('<'==c||','==c||'<'==type1[len-1]||','==type1[len-1]) {
         type1[len++]=c;
         do {
            /* humm .. thoes this translate correctly nested templates? */
            c=G__fgetstream_template(type1+len,">");
            len=strlen(type1);
         } while (isspace(c)); /* ignore white space inside template */
         type1[len++] = c;
         type1[len] = '\0';
      }
      else if('>'==c) {
         type1[len++] = c;
         type1[len] = '\0';
      }
      else {
         c=' ';
         fseek(G__ifile.fp,-1,SEEK_CUR);
         if(G__dispsource) G__disp_mask=1;
         break;
      }
   } 

   /*
   *  typedef unsigned  int  newtype ;
   *                   ^
   * read type 
   */
   if(strcmp(type1,"unsigned")==0) {
      unsigned_flag=1;
      c=G__fgetname(type1,"");
   }
   else if(strcmp(type1,"signed")==0) {
      unsigned_flag=0;
      c=G__fgetname(type1,"");
   }
   else if(strcmp(type1,"unsigned*")==0) {
      unsigned_flag=1;
      strcpy(type1,"int*");
   }
   else if(strcmp(type1,"signed*")==0) {
      unsigned_flag=0;
      strcpy(type1,"int*");
   }
   else if(strcmp(type1,"unsigned&")==0) {
      unsigned_flag=1;
      strcpy(type1,"int&");
   }
   else if(strcmp(type1,"signed&")==0) {
      unsigned_flag=0;
      strcpy(type1,"int&");
   }
   else if(strcmp(type1,"unsigned*&")==0) {
      unsigned_flag=1;
      strcpy(type1,"int*&");
   }
   else if(strcmp(type1,"signed*&")==0) {
      unsigned_flag=0;
      strcpy(type1,"int*&");
   }

   /*
   *  typedef  [struct|union|enum]  tagname { member } newtype;
   *                               ^
   *  typedef  fundamentaltype   newtype;
   *                           ^
   *  typedef unsigned  int  newtype ;
   *                        ^
   */

   if('\0'!=type1[0] && '&'==type1[strlen(type1)-1]) {
      reftype=G__PARAREFERENCE;
      type1[strlen(type1)-1]='\0';
   }

   if('\0'!=type1[0] && '*'==type1[strlen(type1)-1]) {
      isorgtypepointer = 'A'-'a';
      type1[strlen(type1)-1]='\0';
      while('\0'!=type1[0] && '*'==type1[strlen(type1)-1]) {
         if(G__PARANORMAL==reftype) reftype=G__PARAP2P;
         else if(reftype>=G__PARAP2P) ++reftype;
         type1[strlen(type1)-1]='\0';
      }
   }

   if(strcmp(type1,"char")==0) {
      if(unsigned_flag==0) type='c';
      else                 type='b';
   }
   else if(strcmp(type1,"short")==0) {
      if(unsigned_flag==0) type='s';
      else                 type='r';
   }
   else if(strcmp(type1,"int")==0) {
      if(unsigned_flag==0) type='i';
      else                 type='h';
   }
   else if(strcmp(type1,"long")==0) {
      if(unsigned_flag==0) type='l';
      else                 type='k';
   }
   else if(strcmp(type1,"bool")==0) {
      type='g';
   }
   else if(strcmp(type1,"void")==0) {
      type='y';
   }
   else if(strcmp(type1,"float")==0) {
      type='f';
   }
   else if(strcmp(type1,"double")==0) {
      type='d';
   }
   else if(strcmp(type1,"FILE")==0) {
      type='e';
   }

   else if((strcmp(type1,"struct")==0)||(strcmp(type1,"union")==0)||
      (strcmp(type1,"enum")==0)||(strcmp(type1,"class")==0)) {
         type='u';
         if(strcmp(type1,"struct")==0) tagtype='s';
         if(strcmp(type1,"class")==0) tagtype='c';
         if(strcmp(type1,"union")==0) tagtype='u';
         if(strcmp(type1,"enum")==0) tagtype='e';
         tagname[0]='\0';

         /*  typedef [struct|union|enum]{ member } newtype;
         *                              ^ */

         /*  typedef [struct|union|enum]  tagname { member } newtype;
         *  typedef [struct|union|enum]  tagname  newtype;
         *  typedef [struct|union|enum]  { member } newtype;
         *                              ^
         *  read tagname
         */
         if(c!='{') c=G__fgetname(tagname,"{");


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
         if(c!='{') {
            c=G__fgetspace();
            /* typedef [struct] tag   { member } newtype;
            *                         ^
            * typedef [struct|union|enum] tagname  newtype;
            *                                       ^     */
            if(c!='{') {
               fseek(G__ifile.fp,-1,SEEK_CUR);
               if(G__dispsource) G__disp_mask=1;
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
         if(c=='{') {
            mem_def=1;
            fseek(G__ifile.fp,-1,SEEK_CUR);
            fgetpos(G__ifile.fp,&rewind_fpos);
            if(G__dispsource) G__disp_mask=1;
            G__fgetc();
            G__fignorestream("}");
         }
   }
   else if(unsigned_flag) {
      len = strlen(type1);
      if(';'==type1[len-1]) {
         c=';';
         type1[len-1]='\0';
      }
      type='h';
      rawunsigned=1;
   }
   else {
      int tagnum = -1;
      ::ROOT::Reflex::Type itemp = G__find_typedef(type1);
      if(itemp) {
         type=G__get_type(itemp);
         switch(reftype) {
            case G__PARANORMAL:
               reftype=G__get_reftype(itemp);
               break;
            case G__PARAREFERENCE:
               switch(G__get_reftype(itemp)) {
            case G__PARANORMAL:
            case G__PARAREFERENCE:
               break;
            default:
               if(G__get_reftype(itemp)<G__PARAREF) 
                  reftype=G__get_reftype(itemp)+G__PARAREF;
               else reftype=G__get_reftype(itemp);
               break;
               }
               break;
            default:
               switch(G__get_reftype(itemp)) {
            case G__PARANORMAL:
               break;
            case G__PARAREFERENCE:
               G__fprinterr(G__serr,
                  "Limitation: reference or pointer type not handled properly (2)");
               G__printlinenum();
               break;
            default:
               break;
               }
               break;
         }
         tagnum = G__get_tagnum(itemp);
      }
      else {
         type = 'u';
         tagnum=G__defined_tagname(type1,0);
      }
      if(-1!=tagnum) {
         tagtype=G__struct.type[tagnum];
#ifndef G__OLDIMPLEMENTATION1503
         if(-1!=G__struct.parent_tagnum[tagnum])
            sprintf(tagname,"%s::%s"
            ,G__fulltagname(G__struct.parent_tagnum[tagnum],0)
            ,G__struct.name[tagnum]);
         else
            strcpy(tagname,G__struct.name[tagnum]);
#else
         strcpy(tagname,G__fulltagname(tagnum,0));
#endif
         ++G__struct.istypedefed[tagnum];
      }
      else {
         tagtype=0;
         tagname[0]='\0';
      }
      typedef2=1;
   }

   if(isorgtypepointer) type=toupper(type);

   /*
   *  typedef [struct|union|enum] tagname { member } newtype ;
   *                                                ^^
   * skip member declaration if exists
   */

   if(rawunsigned) {
      strcpy(type_name,type1);
   }
   else {
      c=G__fgetname_template(type_name,";,[");
   }

   if( strncmp(type_name,"long",4)==0
      && ( strlen(type_name)==4
      || (strlen(type_name)>=5 && (type_name[4]=='&' || type_name[4]=='*')) )
      ) {
         /* int tmptypenum; */
         if (strlen(type_name)>=5) {
            /* Rewind. */
            long rewindlen = 1 + (strlen(type_name) - strlen("long"));
            fseek(G__ifile.fp, -rewindlen ,SEEK_CUR);
         }
         if('l'==type) {
            type = 'n';
         }
         else if('k'==type) {
            type = 'm';
         }
         strcpy(tagname,""); /* ??? */
         c=G__fgetname(type_name,";,[");
   }
   if(strncmp(type_name,"double",strlen("double"))==0
      && ( strlen(type_name)==strlen("double")
      || (strlen(type_name)>strlen("double") && (type_name[strlen("double")]=='&' || type_name[strlen("double")]=='*')) )
      ) {
         if (strlen(type_name)>strlen("double")) {
            /* Rewind. */
            long rewindlen = 1 + (strlen(type_name) - strlen("double"));
            fseek(G__ifile.fp, -rewindlen ,SEEK_CUR);
         }
         if('l'==type) {
            /* int tmptypenum; */
            type = 'q';
            strcpy(tagname,""); /* ??? */
         }
         c=G__fgetname(type_name,";,[");
   }

   /* in case of
   *  typedef unsigned long int  int32;
   *                           ^
   *  read type_name
   */
   if(strncmp(type_name,"int",3)==0 
      && ( strlen(type_name)==3
      || (strlen(type_name)>=4 && (type_name[3]=='&' || type_name[3]=='*')) )
      ) {
         if (strlen(type_name)>=4) {
            /* Rewind. */
            long rewindlen = 1 + (strlen(type_name) - strlen("int"));
            fseek(G__ifile.fp,-rewindlen ,SEEK_CUR);
         }
         c=G__fgetstream(type_name,";,[");
   }
   if(strcmp(type_name,"*")==0) {
      fpos_t tmppos;
      int tmpline = G__ifile.line_number;
      fgetpos(G__ifile.fp,&tmppos);
      c=G__fgetname(type_name+1,";,[");
      if(isspace(c) && strcmp(type_name,"*const")==0) {
         isconst |= G__PCONSTVAR;
         c=G__fgetstream(type_name+1,";,[");
      }
      else {
         G__disp_mask = strlen(type_name)-1;
         G__ifile.line_number = tmpline;
         fsetpos(G__ifile.fp,&tmppos);
         c=G__fgetstream(type_name+1,";,[");
      }
   }
   else if(strcmp(type_name,"**")==0) {
      fpos_t tmppos;
      int tmpline = G__ifile.line_number;
      fgetpos(G__ifile.fp,&tmppos);
      c=G__fgetname(type_name+1,";,[");
      if(isspace(c) && strcmp(type_name,"*const")==0) {
         isconst |= G__PCONSTVAR;
         c=G__fgetstream(type_name+1,";,[");
      }
      else {
         G__disp_mask = strlen(type_name)-1;
         G__ifile.line_number = tmpline;
         fsetpos(G__ifile.fp,&tmppos);
         c=G__fgetstream(type_name+1,";,[");
      }
      isorgtypepointer=1;
      type=toupper(type);
   }
   else if(strcmp(type_name,"&")==0) {
      reftype = G__PARAREFERENCE;
      c=G__fgetstream(type_name,";,[");
   }
   else if(strcmp(type_name,"*&")==0) {
      reftype = G__PARAREFERENCE;
      type=toupper(type);
      c=G__fgetstream(type_name,";,[");
   }
   else if(strcmp(type_name,"*const")==0) {
      isconst |= G__PCONSTVAR;
      c=G__fgetstream(type_name+1,";,[");
   }
#ifndef G__OLDIMPLEMENTATION1856
   else if(strcmp(type_name,"const*")==0) {
      isconst |= G__CONSTVAR;
      type=toupper(type);
      c=G__fgetstream(type_name,"*&;,[");
      if('*'==c && '*'!=type_name[0]) {
         if(strcmp(type_name,"const")==0) isconst |= G__CONSTVAR;
         type_name[0] = '*';
         c=G__fgetstream(type_name+1,";,[");
      }
      if('&'==c && '&'!=type_name[0]) {
         reftype = G__PARAREFERENCE;
         if(strcmp(type_name,"const")==0) isconst |= G__CONSTVAR;
         c=G__fgetstream(type_name,";,[");
      }
   }
   else if(strcmp(type_name,"const**")==0) {
      isconst |= G__CONSTVAR;
      isorgtypepointer=1;
      type=toupper(type);
      type_name[0] = '*';
      c=G__fgetstream(type_name+1,"*;,[");
   }
   else if(strcmp(type_name,"const*&")==0) {
      isconst |= G__CONSTVAR;
      reftype = G__PARAREFERENCE;
      type=toupper(type);
      c=G__fgetstream(type_name,";,[");
   }
#endif

   if(isspace(c)) {
      if('('==type_name[0] && ';'!=c && ','!=c) {
         do {
            c=G__fgetstream(type_name+strlen(type_name),";,");
            sprintf(type_name+strlen(type_name),"%c",c);
         } while(';'!=c && ','!=c);
         type_name[strlen(type_name)-1]='\0';
      }
      else if(strcmp(type_name,"const")==0) {
         isconst |= G__PCONSTVAR;
         c=G__fgetstream(type_name,";,[");
         if(strncmp(type_name,"*const*",7)==0) {
            isconst |= G__CONSTVAR;
            isorgtypepointer=1;
            type=toupper(type);
            G__shiftstring(type_name,6);
         }
         else if(strncmp(type_name,"*const&",7)==0) {
            isconst |= G__CONSTVAR;
            reftype = G__PARAREFERENCE;
            type=toupper(type);
            G__shiftstring(type_name,7);
         }
         else if(strncmp(type_name,"const*",6)==0) {
         }
         else if(strncmp(type_name,"const&",6)==0) {
         }
      }
      else if(strcmp(type_name,"const*")==0) {
         isconst |= G__PCONSTVAR;
         type_name[0] = '*';
         c=G__fgetstream(type_name+1,";,[");
      }
      else {
         char ltemp1[G__LONGLINE];
         c = G__fgetstream(ltemp1,";,[");
         if('('==ltemp1[0]) {
            type = 'q';
         }
      }
   }

   /* in case of
   *   typedef <unsigned long int|struct A {}>  int32 , *pint32;
   *                                                   ^
   */

   nindex=0;
   while('['==c) {
      store_var_type = G__var_type;
      G__var_type = 'p';
      c=G__fgetstream(aryindex,"]");
      index[nindex++]=G__int(G__getexpr(aryindex));
      c=G__fignorestream("[,;");
      G__var_type = store_var_type;
   }

next_name:

   p=strchr(type_name,'(');
   if(p) {

      flag_p2f = 1;
      if(p==type_name) {
         /* function to pointer 'typedef type (*newtype)();'
         * handle this as 'typedef void* newtype;'
         */
         strcpy(val,p+1);
         p=strchr(val,')');
         *p='\0';
         strcpy(type_name,val);
         type='y';
         p = strstr(type_name,"::*");
         if(p) {
            /* pointer to member function 'typedef type (A::*p)(); */
            strcpy(val,p+3);
            strcpy(type_name,val);
            type='a';
         }
      }
      else if(p==type_name+1 && '*'==type_name[0]) {
         /* function to pointer 'typedef type *(*newtype)();'
         * handle this as 'typedef void* newtype;'
         */
         strcpy(val,p+1);
         p=strchr(val,')');
         *p='\0';
         strcpy(type_name,val);
         type='Q';
         p = strstr(type_name,"::*");
         if(p) {
            /* pointer to member function 'typedef type (A::*p)(); */
            strcpy(val,p+3);
            strcpy(type_name,val);
            type='a';
         }
      }
      else {
         /* function type 'typedef type newtype();'
         * handle this as 'typedef void newtype;'
         */
         *p = '\0';
         type='y';
      }
   }

   isnext=0;
   if(','==c) {
      isnext=1;
      fgetpos(G__ifile.fp,&next_fpos);
   }

   /*  typedef [struct|union|enum] tagname { member } newtype  ;
   *                                                           ^
   *  read over. Store line number. This will be restored after
   *  struct,union.enum member declaration
   */
   temp_line=G__ifile.line_number;


   /* anothername: */

   /* typedef  oldtype     *newtype
   * newtype is a pointer of oldtype  */
   if(type_name[0]=='*') {
      int ix=1;
      if(isupper(type)
#ifndef G__OLDIMPLEMENTATION2191
         &&'1'!=type
#else
         &&'Q'!=type
#endif
         ) {
            reftype = G__PARAP2P;
            while(type_name[ix]=='*') {
               if(G__PARANORMAL==reftype) reftype = G__PARAP2P;
               else if(reftype>=G__PARAP2P) ++ reftype;
               ++ix;
            }
      }
      else {
         type=toupper(type);
         while(type_name[ix]=='*') {
            if(G__PARANORMAL==reftype) reftype = G__PARAP2P;
            else if(reftype>=G__PARAP2P) ++ reftype;
            ++ix;
         }
      }
      strcpy(val,type_name);
      strcpy(type_name,val+ix);
   }

   /* typedef oldtype &newtype */
   if(type_name[0]=='&') {
      if(reftype>=G__PARAP2P) reftype += G__PARAREF;
      else                    reftype = G__PARAREFERENCE;
      if(strlen(type_name)>1) {
         strcpy(val,type_name);
         strcpy(type_name,val+1);
      }
      else {
         /* to be determined */
      }
   }

   /*
   * check if typedef hasn't been defined
   */
   typenum = G__defined_typename_exact(type_name);

   if(typenum) {
      /*
      *  return if the type is already defined
      */
      if(';'!=c) G__fignorestream(";");
      return;
   }

   // Fix the name and the pointer type.
   if(tagname[0]=='\0') {
#ifdef __GNUC__
#else
#pragma message (FIXME("What do we do with link off?"))
#endif
      if(G__CPPLINK==G__globalcomp) sprintf(tagname,"%s",type_name);
      else                          sprintf(tagname,"$%s",type_name);
      sprintf(tagname,"%s",type_name);
      taglen=strlen(tagname);
   }
   else {
      taglen=strlen(tagname);
      if(tagname[taglen-1]=='*') {
         type=toupper(type);
         tagname[taglen-1]='\0';
      }
   }

   // This section used to be __after__ the typenum creation
   // because the enum section in G__defined_struct_in_typedef
   // was using this information.
   ::ROOT::Reflex::Type baseType;
   if(G__def_struct_member) env_tagnum = G__tagnum;
   else if(G__func_now) {
#ifdef __GNUC__
#else
#pragma message (FIXME("What is tagnum==-2?!"))
#endif
      //env_tagnum = -2;
      env_tagnum = ::Reflex::Scope::GlobalScope();
   }
   else env_tagnum = ::Reflex::Scope::GlobalScope();


   if(tolower(type)=='u') {

#ifdef __GNUC__
#else
#pragma message(FIXME("Shouldn't we warn the user that we cannot find the underlying type? Or can we register a typedef to uninitialized type?"))
#endif
      // note that the tagname in next line is the _new_ name of the typedef, not the underlying type name!
      G__tagnum = G__Dict::GetDict().GetScope(G__search_tagname(tagname,tagtype));
      if(!G__tagnum || G__tagnum.IsTopScope()) {
         G__fignorestream(";");
         return;
      }

      baseType = G__tagnum;

      if(mem_def==1) {

         if(G__struct.size[G__get_tagnum(G__tagnum)]==0) {
            G__define_struct_in_typedef(rewind_fpos,G__get_tagnum(G__tagnum),
                                        G__get_tagnum(env_tagnum),tagtype/*,typenum*/,G__get_tagnum(store_tagnum));
            G__ifile.line_number=temp_line;
         } /* end of G__struct.size[G__tagnum]==0 */
      } /* end of mem_def==1 */

      else { /* mem_def!=1 */
         /* oldlink eliminated */
      }  /* of mem_def */

      G__tagnum=store_tagnum;


   } /* end of struct,class,union,enum */
   else { /* scalar type */
      baseType = G__get_from_type(type,false);
   } /* end of scalar type */


   /*
   * if new typedef, add it to newtype table
   */

    /*
    if( some memory threadhold is reached) {
      G__fprinterr(G__serr,
              "Limitation: Number of typedef exceed %d FILE:%s LINE:%d\nFatal error, exit program. Increase G__MAXTYPEDEF in G__ci.h and recompile %s\n"
              ,G__MAXTYPEDEF
              ,G__ifile.name
              ,G__ifile.line_number
              ,G__nam);
      G__eof=1;
      return;
    }
    */
   ::ROOT::Reflex::Type newType = 
      G__modify_type(baseType,isupper(type),reftype,isconst,nindex,index);

   std::string fullname;
   if (G__def_struct_member && env_tagnum) {
      fullname = G__fulltagname(G__get_tagnum(env_tagnum),0); // parentScope.Name(SCOPED);
      fullname += "::";
   } else if(-1!=G__func_now) {
      //env_tagnum = -2;
#ifdef __GNUC__
#else
#pragma message (FIXME("Did not inform Reflex that this typedef is invalid"))
#endif
      G__fprinterr(G__serr,"Limitation: In function typedef not allowed in cint");
      G__printlinenum();
   }
   fullname += type_name;
   if (!newType) {
      printf("Creating typedef %s to uninitialized type %s %s!\n", fullname.c_str(), 
         newType.TypeTypeAsString().c_str(), newType.Name(ROOT::Reflex::SCOPED).c_str());
      //G__dumpreflex();
   }
   ::ROOT::Reflex::Type result = 
      ::ROOT::Reflex::TypedefTypeBuilder(fullname.c_str(),newType);
   G__RflxProperties *prop = G__get_properties(result);
   if (prop) {
      prop->globalcomp = G__default_link?G__globalcomp:G__NOLINK;
      //fprintf(stderr, "Registering typedef '%s'\n", result.Name().c_str());
      prop->typenum = G__Dict::GetDict().Register(result);
#ifdef G__TYPEDEFFPOS
      prop->filenum = G__ifile.filenum;
      prop->linenum = G__ifile.line_number;
#endif
      prop->tagnum = G__get_tagnum(newType.RawType());
   }

   if(isnext) {
      fsetpos(G__ifile.fp,&next_fpos);
      c=G__fgetstream(type_name,",;");
      goto next_name;
   }

   if (prop) {
      if(G__fons_comment) {
         G__fsetcomment(&(prop->comment));
      }

      if(flag_p2f 
         && prop->comment.filenum<0 
         && !prop->comment.p.com) {
            fpos_t xpos;
            if(G__ifile.filenum > G__nfile) {
               G__fprinterr(G__serr
                  ,"Warning: pointer to function typedef incomplete in command line or G__exec_text(). Declare in source file or use G__load_text()\n");
               return;
            }
            ++G__macroORtemplateINfile;
            fgetpos(G__ifile.fp,&xpos);
            fsetpos(G__ifile.fp,&pos_p2fcomment);

            if(G__ifile.fp==G__mfp) 
               prop->comment.filenum = G__MAXFILE;
            else
               prop->comment.filenum = G__ifile.filenum;
            fgetpos(G__ifile.fp,&prop->comment.p.pos);

            fsetpos(G__ifile.fp,&xpos);
      }
   }
}


/******************************************************************
* G__defined_typename(type_name)
*
* Search already defined typedef names, -1 is returned if not found
* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
* Note that this modify G__var_type, you may need to reset it after
* calling this function
* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
******************************************************************/
int G__defined_typename(const char *type_name) 
{
   // FIXME: Neither G__defined_typename nor 
   // G__find_typedef should have to modify G__var_type.
   ::ROOT::Reflex::Type result = G__find_typedef(type_name);
   if (!result) return -1;
   return G__get_typenum(result);
}

::ROOT::Reflex::Type Cint::Internal::G__find_typedef(const char *type_name)
{
  int ispointer = 0;
  std::string temp = type_name; 
  int len = temp.size();
  if(temp.size()>0 && temp[len-1]=='*') 
  {
    temp[--len]='\0';
    ispointer = 'A' - 'a';
  }

  ::Reflex::Scope scope = G__get_envtagnum();
  if (G__tmplt_def_tagnum && !G__tmplt_def_tagnum.IsTopScope()) {
     scope = G__tmplt_def_tagnum;
  }
  if (!scope) {
     printf("Trying to look up typedef %s in an invalid enclosing scope!\n", type_name);
     while (scope.Id() && !scope && !scope.IsTopScope()) 
        scope = scope.DeclaringScope();
  }

  ::ROOT::Reflex::Type result = scope.LookupType(type_name);

  if (!result && strstr(type_name,"<")!=0) {
     // This may be a template that need instantiating so let's try a
     // different way.
     const char *p = G__find_last_scope_operator ((char*)type_name);
     if (p && p!=type_name) {
        std::string leftside(type_name,p-type_name);
        // Induce the instantiation of template if any
        if ( !(leftside=="std" && G__ignore_stdnamespace)
            && G__defined_tagname(leftside.c_str(),1)>=0) {
           result = scope.LookupType(type_name);
        }
     }
  }
  if (!result || !result.IsTypedef()) return ::ROOT::Reflex::Type();

  /* This must be a bad manner. Somebody needs to reset G__var_type
   * especially when typein is 0. */
  G__var_type= G__get_type(result) + ispointer ;

  return result;
}

/******************************************************************
* G__make_uniqueP2Ftypedef()
*
*  input  'void* (*)(int , void * , short )'
*  output 'void* (*)(int,void*,short)'
*
******************************************************************/
static int G__make_uniqueP2Ftypedef(char *type_name)
{
  char *from;
  char *to;
  int spacecnt=0;
  int isstart=1;

  /*  input  'void* (*)(int , void * , short )'
   *         ^ start                         */
  from = strchr(type_name,'(');
  if(!from) return(1);
  from = strchr(from+1,'(');
  if(!from) return(1);
  ++from;
  to = from;
  /*  input  'void* (*)(int , void * , short )'
   *                    ^ got this position  */

  while(*from) {
    if(isspace(*from)) {
      if(0==spacecnt && 0==isstart) {
        /*  input  'void* (*)(int   * , void  * , short )'
         *                       ^ here  */
        *(to++) = ' ';
      }
      else {
        /*  input  'void* (*)(int   * , void  * , short )'
         *                        ^^ here  */
        /* Ignore consequitive space */
      }
      if(0==isstart) ++spacecnt;
      else           spacecnt=0;
      isstart=0;
    }
    else {
      isstart=0;
      if(spacecnt) {
        switch(*from) {
        case ',':
          isstart = 1;
        case ')':
        case '*':
        case '&':
          /*  input  'void* (*)(int   * , void  * , short )'
           *                          ^ here
           *  output 'void* (*)(int*
           *                       ^ put here */
          *(to-1) = *from;
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
      spacecnt=0;
    }
    ++from;
  }

  *to = 0;

  /* int (*)(void) to int (*)() */
  from = strchr(type_name,'(');
  if(!from) return(1);
  from = strchr(from+1,'(');
  if(!from) return(1);
  if(strcmp(from,"(void)")==0) {
    *(++from) = ')';
    *(++from) = 0;
  }

  return(0);
}

/******************************************************************
* G__search_typename(type_name,type,tagnum,reftype)
*
* Used in G__cpplink.C
* Search typedef name. If not found, allocate new entry if typein
* isn't 0.
* This is superseeded by a call to:
*    G__declared_typedef(typenamein,typepin,tagnum,reftype,
                         0, G__NOLINK, -1,
*                        true);
* Do not forget the 'true' as the last argument.
******************************************************************/
int G__search_typename(const char *typenamein,int typein
                       ,int tagnum,int reftype)
{
   ::ROOT::Reflex::Type result = 
      G__declare_typedef(typenamein,typein,tagnum,reftype
                         ,0
                         ,G__default_link?G__globalcomp:G__NOLINK
                         ,-1,true);
   if (!result) return -1;
   return G__get_typenum(result);
}
                                     
// FIXME: This might be part of the new interface.
::ROOT::Reflex::Type Cint::Internal::G__declare_typedef(
   const char *typenamein,int typein,int tagnum,int reftype,
   int isconst, int globalcomp,
   int parent_tagnum,
   bool pointer_fix)
{
   // PointerFix is for G__search_typename replacement.

  int len;
  char ispointer=0;

  char type_name[G__LONGLINE];
  strcpy(type_name,typenamein);
  /* keep uniqueness for pointer to function typedefs */
#ifndef G__OLDIMPLEMENTATION2191
  if('1'==typein) G__make_uniqueP2Ftypedef(type_name);
#else
  if('Q'==typein) G__make_uniqueP2Ftypedef(type_name);
#endif
  
  // FIXME: similar to G__find_typedef except for the
  // source of the scope (here its G__static_parent_tagnum)
  if (pointer_fix){
     len=strlen(type_name);
     if(
        len &&
        type_name[len-1]=='*') {
           type_name[--len]='\0';
           ispointer = 'A' - 'a';
     }
  }

  ::ROOT::Reflex::Scope scope = G__Dict::GetDict().GetScope(parent_tagnum);
  ::ROOT::Reflex::Type typedf = scope.LookupType(type_name);

  if (pointer_fix && typedf && typedf.IsTypedef()) {
     G__var_type=G__get_type(typedf) + ispointer ;
  }

  /* allocate new type table entry */
  if(!typedf && typein) {
    /*
    if( some memory threshold is reached ) {
      G__fprinterr(G__serr,
              "Limitation: Number of typedef exceed %d FILE:%s LINE:%d\nFatal error, exit program. Increase G__MAXTYPEDEF in G__ci.h and recompile %s\n"
              ,G__MAXTYPEDEF ,G__ifile.name ,G__ifile.line_number ,G__nam);
      G__eof=1;
      G__var_type = 'p';
      return(-1);
    }
    */
    
     ::ROOT::Reflex::Type baseType;
     if (tagnum > -1) {
        baseType = G__Dict::GetDict().GetType(tagnum);
     } else {
        baseType = G__get_from_type(typein,false);
     }
     if (!baseType.Id()) {
        Cint::Internal::G__dumpreflex();
        G__fprinterr(G__serr,"Internal error: G__declare_typedef could not find the type for tagnum==%d (to define %s as %s)",
           tagnum,type_name,G__fulltagname(tagnum,0));
        G__genericerror((char*)NULL);
        return typedf;
     }
     ::ROOT::Reflex::Type newType = 
        G__modify_type(baseType,ispointer||isupper(typein),reftype,isconst,0,0);
     
     std::string fullname;
     if (parent_tagnum!=-1) {
        fullname = G__fulltagname(parent_tagnum,0); // parentScope.Name(SCOPED);
        if (fullname.length())
           fullname += "::";
     }
     fullname += type_name;
     //if (newType.Name(::ROOT::Reflex::SCOPED) != fullname) {
        ::ROOT::Reflex::Type result = 
           ::ROOT::Reflex::TypedefTypeBuilder(fullname.c_str(),newType);
        G__RflxProperties *prop = G__get_properties(result);
        if (prop) {
           prop->globalcomp = globalcomp;

#ifdef G__TYPEDEFFPOS
           prop->filenum = G__ifile.filenum;
           prop->linenum = G__ifile.line_number;
#endif
           //fprintf(stderr, "Registering typedef '%s'\n", result.Name().c_str());
           prop->typenum = G__Dict::GetDict().Register(result);
           if (tagnum!=-1)
              prop->tagnum = tagnum;
           else
              prop->tagnum = G__get_tagnum(newType.RawType());
        }
        typedf = result;
     //}
  }
  return(typedf);
}

#ifndef __CINT__
// FIXME: Used for global var setting!
namespace Cint { namespace Internal { 
   void G__setnewtype_settypenum(::ROOT::Reflex::Type typenum);
} }
#endif
/******************************************************************
* G__search_typename2()
******************************************************************/
extern "C" int G__search_typename2(const char *type_name,int typein
                                   ,int tagnum,int reftype
                                   ,int parent_tagnum)
{
   int ret;
   int real_parent_tagnum = parent_tagnum;
   if(-1==real_parent_tagnum && G__def_struct_member &&
      G__tagdefining.IsNamespace()) {
      real_parent_tagnum = G__get_tagnum(G__tagdefining);
   }
   int isconst = reftype/0x100;
   reftype = reftype%0x100;

   ::ROOT::Reflex::Type result = 
        G__declare_typedef(type_name,typein,tagnum,reftype
                         ,isconst
                         ,G__default_link?G__globalcomp:G__NOLINK
                         ,real_parent_tagnum,true);
   if (result) {
      G__setnewtype_settypenum(result);
      ret = G__get_typenum(result);
   } else {
      G__setnewtype_settypenum(::ROOT::Reflex::Type());
      ret = -1;
   }
 
   return(ret);
}

/******************************************************************
* G__defined_type(type_name,len)
*
* Search already defined type_name and tagname
* and allocate automatic variables.
******************************************************************/
int Cint::Internal::G__defined_type(char *type_name,int len)
/* struct G__input_file *fin; */
{
   ::Reflex::Scope store_tagnum;
   ::ROOT::Reflex::Type store_typenum;
   /* char type; */
   int cin;
   int refrewind = -2;
   fpos_t pos;
   int line;
   char store_typename[G__LONGLINE];  

   if(G__prerun&&'~'==type_name[0]) {
      G__var_type = 'y';
      cin = G__fignorestream("(");
      type_name[len++]=cin;
      type_name[len]='\0';
      G__make_ifunctable(type_name);
      return(1);
   }

   if(!isprint(type_name[0]) && len==1) {
      return(1);
   }

   fgetpos(G__ifile.fp,&pos);
   line=G__ifile.line_number;
   /* this is not the fastest to insure proper unwinding in case of 
   error, but it is the simpliest :( */
   strcpy(store_typename,type_name);

   /*************************************************************
   * check if this is a declaration or not
   * declaration:
   *     type varname... ; type *varname
   *          ^ must be alphabet '_' , '*' or '('
   * else
   *   if not alphabet, return
   *     type (param);   function name
   *     type = expr ;   variable assignment
   *************************************************************/
   cin = G__fgetspace();
   /* This change is risky. Need more evaluation */
   switch(cin) {
  case '*':
  case '&':
     cin=G__fgetc();
     fseek(G__ifile.fp,-2,SEEK_CUR);
     if(G__dispsource) G__disp_mask=2;
     if('='==cin) return(0);
     break;
  case '(':
  case '_':
     fseek(G__ifile.fp,-1,SEEK_CUR);
     if(G__dispsource) G__disp_mask=1;
     break;
  default:
     fseek(G__ifile.fp,-1,SEEK_CUR);
     if(G__dispsource) G__disp_mask=1;
     if(!isalpha(cin)) return(0);
     break;
   }

   if(type_name[len-1]=='&') {
      G__reftype=G__PARAREFERENCE;
      type_name[--len]='\0';
      --refrewind;
   }

   store_tagnum = G__tagnum;
   store_typenum=G__typenum;

   /* search for typedef names */
   if(len>2 && '*'==type_name[len-1] && '*'==type_name[len-2]) {
      /* pointer to pointer */
      len -=2;
      type_name[len]='\0';
      /* type** a;
      *     ^<<^      */
      fsetpos(G__ifile.fp,&pos);
      G__ifile.line_number=line;
      /* the following fseek is now potentialy wrong (because of fake_space!) */
      fseek(G__ifile.fp,-1,SEEK_CUR);
      cin = G__fgetc();
      if (cin=='*') {
         /* we have a fake space */
         fseek(G__ifile.fp,refrewind,SEEK_CUR);
      } else {
         fseek(G__ifile.fp,refrewind-1,SEEK_CUR);
      }
      if(G__dispsource) G__disp_mask=2;
   }
   else if(len>1 && '*'==type_name[len-1]) {
      int cin2;
      len -=1;
      type_name[len]='\0';
      fsetpos(G__ifile.fp,&pos);
      G__ifile.line_number=line;
      /* To know how much to rewind we need to know if there is a fakespace */
      fseek(G__ifile.fp,-1,SEEK_CUR);
      cin = G__fgetc();
      if (cin=='*') {
         /* we have a fake space */
         fseek(G__ifile.fp,refrewind+1,SEEK_CUR);
      } else {
         fseek(G__ifile.fp,refrewind,SEEK_CUR);
      }
      if(G__dispsource) G__disp_mask=1;
      cin2 = G__fgetc();
      if(!isalnum(cin2)
         && '>'!=cin2
         ) {
            fseek(G__ifile.fp,-1,SEEK_CUR);
            if(G__dispsource) G__disp_mask=1;
      }
   }

   G__typenum = G__find_typedef(type_name);

   if(!G__typenum) {
      /* search for class/struct/enum tagnames */
      int tagnum = G__defined_tagname(type_name,1);
      if (tagnum == -1) {
         /* This change is risky. Need more evaluation */
         if(G__fpundeftype && '('!=cin &&
            (-1==G__func_now || G__def_tagnum)) {
               G__tagnum = G__Dict::GetDict().GetScope(G__search_tagname(type_name,'c'));
               fprintf(G__fpundeftype,"class %s; /* %s %d */\n",type_name
                  ,G__ifile.name,G__ifile.line_number);
               fprintf(G__fpundeftype,"#pragma link off class %s;\n\n",type_name);
               G__struct.globalcomp[G__get_tagnum(G__tagnum)] = G__NOLINK;
         }
         else {
            /* if not found, return */
            /* Restore properly the previous state! */
            fsetpos(G__ifile.fp,&pos);
            G__ifile.line_number = line;
            strcpy(type_name,store_typename);
            G__tagnum = store_tagnum;
            G__typenum = store_typenum;
            G__reftype=G__PARANORMAL;
            return(0);
         }
      }
      else {
         G__tagnum = G__Dict::GetDict().GetScope(tagnum);
         G__typenum = G__find_typedef(type_name);
         if(G__typenum) {
            G__reftype += G__get_reftype(G__typenum);
            G__typedefindex = G__get_index(G__typenum);
            G__typedefnindex = G__typedefindex.size();
         }
      }
      G__var_type = 'u';
   }
   else {
      G__tagnum=G__typenum.RawType();
      /* Questionable adding of G__reftype, maybe |= instead */
      G__reftype += G__get_reftype(G__typenum);
      G__typedefindex = G__get_index(G__typenum);
      G__typedefnindex = G__typedefindex.size();
   }

   if(G__tagnum.IsEnum()) {
      /* in case of enum */
      G__var_type='i';
   }


   /* allocate variable */
   G__define_var(G__get_tagnum(G__tagnum),G__typenum);

   G__typedefnindex = 0;
   G__typedefindex.clear();

   G__tagnum=store_tagnum;
   G__typenum=store_typenum;

   G__reftype=G__PARANORMAL;

   return(1);
}

/*
 * Local Variables:
 * c-tab-always-indent:nil
 * c-indent-level:2
 * c-continued-statement-offset:2
 * c-brace-offset:-2
 * c-brace-imaginary-offset:0
 * c-argdecl-indent:0
 * c-label-offset:-2
 * compile-command:"make -k"
 * End:
 */
