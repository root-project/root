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
 * Permission to use, copy, modify and distribute this software and its
 * documentation for any purpose is hereby granted without fee,
 * provided that the above copyright notice appear in all copies and
 * that both that copyright notice and this permission notice appear
 * in supporting documentation.  The author makes no
 * representations about the suitability of this software for any
 * purpose.  It is provided "as is" without express or implied warranty.
 ************************************************************************/

#include "common.h"



#ifndef G__OLDIMPLEMENTATION776
static int G__static_parent_tagnum = -1;
#endif
#ifndef G__OLDIMPLEMENTATION1394
static int G__static_isconst = 0;
#endif

/******************************************************************
* G__shiftstring
******************************************************************/
void G__shiftstring(s,n)
char* s;
int n;
{
  int i=0, j=n;
  while(s[j]) s[i++]=s[j++];
  s[i]=0;
}

#ifndef G__OLDIMPLEMENTATION559
/******************************************************************
* G__defined_typename_exact(typename)
*
* Search already defined typedef names, -1 is returned if not found
******************************************************************/
static int G__defined_typename_exact(typename)
char *typename;
{
  int i,flag=0,len;
  char ispointer=0;
  char temp[G__LONGLINE];
  char *p;
  char temp2[G__LONGLINE];
  int env_tagnum;
#ifndef G__OLDIMPLEMENTATION745
  char *par;
#endif

  strcpy(temp2,typename);

  /* find 'xxx::yyy' */
  p = G__find_last_scope_operator (temp2);

#ifndef G__OLDIMPLEMENTATION745
  /* abandon scope operator if 'zzz (xxx::yyy)www' */
  par = strchr(temp2,'(');
  if(par && p && par<p) p=(char*)NULL;
#endif

  if(p) {
    strcpy(temp,p+2);
    *p='\0';
    if(temp2==p) env_tagnum = -1; /* global scope */
#ifndef G__STD_NAMESPACE
    else if (strcmp (temp2, "std") == 0
#ifndef G__OLDIMPLEMENTATION1285
	     && G__ignore_stdnamespace
#endif
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
#ifndef G__OLDIMPLEMENTATION620
	env_tagnum==G__newtype.parent_tagnum[i]
#else
       -1==G__newtype.parent_tagnum[i]||
	env_tagnum==G__newtype.parent_tagnum[i]
#endif
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
}
#endif

/******************************************************************
* G__define_type()
*
*  typedef [struct|union|enum] tagname { member } newtype;
*  typedef fundamentaltype   newtype;
*
******************************************************************/

#define G__OLDIMPLEMENTATION63

void G__define_type()
/* struct G__input_file *fin; */
{
  fpos_t rewind_fpos;
  int c;
  char type1[G__LONGLINE],tagname[G__LONGLINE],typename[G__LONGLINE];
#ifndef G__PHILIPPE8
  char temp[G__LONGLINE];
#endif
  int isnext;
  fpos_t next_fpos;
  int /* itag=0,*/ mparen,store_tagnum,store_def_struct_member=0;
  struct G__var_array *store_local;
  /* char store_tagname[G__LONGLINE] */
  char category[10],memname[G__MAXNAME],val[G__ONELINE];
  char type,tagtype=0;
  int unsigned_flag=0 /* ,flag=0 */ ,mem_def=0,temp_line;
  int len,taglen;
  G__value enumval;
  int store_tagdefining;
  int typedef2=0;
  int itemp;
  int nindex=0;
  int index[G__MAXVARDIM];
  char aryindex[G__MAXNAME];
  /* int lenheader; */
  char *p;
  int store_var_type;
  int typenum;
  int isorgtypepointer=0;
  int store_def_tagnum;
#ifndef G__OLDIMPLEMENTATION188
  int reftype=G__PARANORMAL;
  int rawunsigned=0;
#endif
  int env_tagnum;
#ifndef G__OLDIMPLEMENTATION1394
  int isconst = 0;
#endif
#ifndef G__OLDIMPLEMENTATION1763
  fpos_t pos_p2fcomment;
  int    line_p2fcomment;
  int    flag_p2f=0;
#endif

  tagname[0] = '\0';		/* initialize it */

#ifndef G__OLDIMPLEMENTATION1763
  fgetpos(G__ifile.fp,&pos_p2fcomment);
  line_p2fcomment = G__ifile.line_number;
#endif

#ifdef G__ASM
#ifdef G__ASM_DBG
  if(G__asm_dbg&&G__asm_noverflow)
    G__genericerror(G__LOOPCOMPILEABORT);
#endif
  G__abortbytecode();
#endif

  store_tagnum=G__tagnum;
  store_tagdefining = G__tagdefining;
  store_def_tagnum = G__def_tagnum;


  /*
   *  typedef  [struct|union|enum] tagname { member } newtype;
   *  typedef  fundamentaltype   newtype;
   *          ^
   * read type
   */

#ifndef G__PHILIPPE8
#ifndef G__OLDIMPLEMENTATION1685
  c=G__fgetname_template(type1,"*{");
  if('*'==c) { 
    strcat(type1,"*");
    c=' ';
  }
#else
  c=G__fgetname_template(type1,"{");
#endif
  /* just ignore the following 4 keywords as long as they are
     followed by a space */
  while(isspace(c) &&
	(strcmp(type1,"const")==0 ||strcmp(type1,"volatile")==0 ||
	 strcmp(type1,"mutable")==0 || strcmp(type1,"typename") == 0)) {
#ifndef G__OLDIMPLEMENTATION1394
    if(strcmp(type1,"const")==0) isconst |= G__CONSTVAR;
#endif
    c=G__fgetname_template(type1,"{");
  }
  if (strcmp(type1,"::")==0) {
    /* skip a :: without a namespace in front of it (i.e. global namespace!) */
    c = G__fgetspace(); /* skip the next ':' */
    c=G__fgetname_template(type1,"{");
  }   
#ifndef G__OLDIMPLEMENTATION1693
  if (strncmp(type1,"::",2)==0) {
    /* A leading '::' causes other typename matching function to fails so 
       we remove it. This is not the ideal solution (neither was the one 
       above since it does not allow for distinction between global 
       namespace and local namespace) ... but at least it is an improvement
       over the current behavior */
    strcpy(type1,type1+2);
  }
#endif
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

#else 
  c=G__fgetname_template(type1,"{");
  if(isspace(c)) {
    len=strlen(type1);
    c = G__fgetspace();
    if('<'==c||','==c||'<'==type1[len-1]||','==type1[len-1]) {
      type1[len++]=c;
      c=G__fgetstream_template(type1+len,">");
      len=strlen(type1);
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
    }
  }

#ifndef G__OLDIMPLEMENTATION678
  while(isspace(c) &&
	(strcmp(type1,"const")==0 ||strcmp(type1,"volatile")==0 ||
	 strcmp(type1,"mutable")==0 || strcmp(type1,"typename") == 0)) {
#ifndef G__OLDIMPLEMENTATION1394
    if(strcmp(type1,"const")==0) isconst |= G__CONSTVAR;
#endif
    c=G__fgetname_template(type1,"{");
  }
#else
#ifndef G__OLDIMPLEMENTATION666
  if(strcmp(type1,"const")==0 ||strcmp(type1,"volatile")==0 ||
     strcmp(type1,"mutable")==0 || strcmp(type1,"typename") == 0) {
#else
  if(strcmp(type1,"const")==0 ||strcmp(type1,"volatile")==0) {
#endif
    c=G__fgetname_template(type1,"{");
  }
#endif

#endif
  /*
   *  typedef unsigned  int  newtype ;
   *                   ^
   * read type 
   */
  if(strcmp(type1,"unsigned")==0) {
    unsigned_flag=1;
    c=G__fgetname(type1,"");
  }
#ifndef G__OLDIMPLEMENTATION527
  else if(strcmp(type1,"signed")==0) {
    unsigned_flag=0;
    c=G__fgetname(type1,"");
  }
#endif
#ifndef G__OLDIMPLEMENTATION1548
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
#endif

  /*
   *  typedef  [struct|union|enum]  tagname { member } newtype;
   *                               ^
   *  typedef  fundamentaltype   newtype;
   *                           ^
   *  typedef unsigned  int  newtype ;
   *                        ^
   */

#ifndef G__OLDIMPLEMENTATION188
  if('\0'!=type1[0] && '&'==type1[strlen(type1)-1]) {
    reftype=G__PARAREFERENCE;
    type1[strlen(type1)-1]='\0';
  }
#endif

  if('\0'!=type1[0] && '*'==type1[strlen(type1)-1]) {
    isorgtypepointer = 'A'-'a';
    type1[strlen(type1)-1]='\0';
#ifndef G__OLDIMPLEMENTATION919
    while('\0'!=type1[0] && '*'==type1[strlen(type1)-1]) {
      if(G__PARANORMAL==reftype) reftype=G__PARAP2P;
      else if(reftype>=G__PARAP2P) ++reftype;
      type1[strlen(type1)-1]='\0';
    }
#else
    if('\0'!=type1[0] && '*'==type1[strlen(type1)-1]) {
      reftype=G__PARAP2P;
      type1[strlen(type1)-1]='\0';
    }
#endif /* ON673 */
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
#ifndef G__OLDIMPLEMENTATION1604
  else if(strcmp(type1,"bool")==0) {
    type='g';
  }
#endif
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
     *  typedef [struct|union|enum] 	     { member } newtype;
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
     *  typedef [struct|union|enum] 	     { member } newtype;
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
    itemp = G__defined_typename(type1);
    if(-1!=itemp) {
      type=G__newtype.type[itemp];
#ifndef G__OLDIMPLEMENTATION1112
      switch(reftype) {
      case G__PARANORMAL:
	reftype=G__newtype.reftype[itemp];
	break;
      case G__PARAREFERENCE:
#ifndef G__OLDIMPLEMENTATION2033
	switch(G__newtype.reftype[itemp]) {
	case G__PARANORMAL:
	case G__PARAREFERENCE:
	  break;
	default:
	  if(G__newtype.reftype[itemp]<G__PARAREF) 
	    reftype=G__newtype.reftype[itemp]+G__PARAREF;
	  else reftype=G__newtype.reftype[itemp];
	  break;
	}
#else
	if(G__PARANORMAL!=G__newtype.reftype[itemp]) {
	  G__fprinterr(G__serr,
	 "Limitation: reference or pointer type not handled properly");
	  G__printlinenum();
	}
#endif
	break;
      default:
	switch(G__newtype.reftype[itemp]) {
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
#else
      reftype=G__newtype.reftype[itemp];
#endif
      itemp = G__newtype.tagnum[itemp];
    }
    else {
      type = 'u';
      itemp=G__defined_tagname(type1,0);
    }
    if(-1!=itemp) {
      tagtype=G__struct.type[itemp];
#ifndef G__OLDIMPLEMENTATION710
#ifndef G__OLDIMPLEMENTATION1503
      if(-1!=G__struct.parent_tagnum[itemp])
	sprintf(tagname,"%s::%s"
		,G__fulltagname(G__struct.parent_tagnum[itemp],0)
		,G__struct.name[itemp]);
      else
	strcpy(tagname,G__struct.name[itemp]);
#else
      strcpy(tagname,G__fulltagname(itemp,0));
#endif
#else
      strcpy(tagname,G__struct.name[itemp]);
#endif
      ++G__struct.istypedefed[itemp];
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
    strcpy(typename,type1);
  }
  else {
    c=G__fgetname_template(typename,";,[");
  }

#ifndef G__OLDIMPLEMENTATION1141
#ifndef G__PHILIPPE34
  if( strncmp(typename,"long",4)==0
      && ( strlen(typename)==4
           || (strlen(typename)>=5 && (typename[4]=='&' || typename[4]=='*')) )
     ) {
#else 
  if(strcmp(typename,"long")==0) {
#endif /* G__PHILIPPE34 */
#if !defined(G__OLDIMPLEMENTATION2189)
    /* int tmptypenum; */
#ifndef G__PHILIPPE34
    if (strlen(typename)>=5) {
      /* Rewind. */
      fseek(G__ifile.fp,-1-(strlen(typename) - strlen("long")) ,SEEK_CUR);
    }
#endif /* G__PHILIPPE34 */
    if('l'==type) {
      type = 'n';
    }
    else if('k'==type) {
      type = 'm';
    }
    strcpy(tagname,""); /* ??? */
#elif !defined(G__OLDIMPLEMENTATION1836)
    int tmptypenum;
#ifndef G__PHILIPPE34
    if (strlen(typename)>=5) {
      /* Rewind. */
      fseek(G__ifile.fp,-1-(strlen(typename) - strlen("long")) ,SEEK_CUR);
    }
#endif /* G__PHILIPPE34 */
    if('l'==type) {
      G__loadlonglong(&itemp,&tmptypenum,G__LONGLONG);
      type = 'u';
#ifndef G__OLDIMPLEMENTATION1850
      strcpy(tagname,"G__longlong");
#endif
    }
    else if('k'==type) {
      G__loadlonglong(&itemp,&tmptypenum,G__ULONGLONG);
      type = 'u';
#ifndef G__OLDIMPLEMENTATION1850
      strcpy(tagname,"G__ulonglong");
#endif
    }
#else /* 1836 */
    if('l'==type || 'k'==type) {
      if(0==G__defined_macro("G__LONGLONG_H")) {
#ifndef G__OLDIMPLEMENTATION1153
	int store_def_struct_member = G__def_struct_member;
	G__def_struct_member = 0;
#endif
	G__loadfile("long.dll"); /* used to switch case between .dl and .dll */
#ifndef G__OLDIMPLEMENTATION1153
	G__def_struct_member = store_def_struct_member;
#endif
      }
#ifndef G__OLDIMPLEMENTATION1686
      strcpy(tagname,"G__ulonglong");
      itemp=G__defined_tagname(tagname,2);
      G__search_typename("unsigned long long",'u',itemp,G__PARANORMAL);
#endif
#ifndef G__OLDIMPLEMENTATION1533
      strcpy(tagname,"G__longdouble");
      itemp=G__defined_tagname(tagname,2);
      G__search_typename("long double",'u',itemp,G__PARANORMAL);
#endif
      strcpy(tagname,"G__longlong");
      itemp=G__defined_tagname(tagname,2);
      if(-1==itemp) {
	G__genericerror("Error: 'long long' not ready. Go to $CINTSYSDIR/lib/longlong and run setup");
      }
      G__search_typename("long long",'u',itemp,G__PARANORMAL);
      type='u';
    }
#endif /* 1836 */
    c=G__fgetname(typename,";,[");
  }
#endif
#ifndef G__OLDIMPLEMENTATION1533
#ifndef G__PHILIPPE34
  if(strncmp(typename,"double",strlen("double"))==0
     && ( strlen(typename)==strlen("double")
          || (strlen(typename)>strlen("double") && (typename[strlen("double")]=='&' || typename[strlen("double")]=='*')) )
     ) {
     if (strlen(typename)>strlen("double")) {
       /* Rewind. */
        fseek(G__ifile.fp,-1-(strlen(typename) - strlen("double")) ,SEEK_CUR);
     }
#else
  if(strcmp(typename,"double")==0) {
#endif /* G__PHILIPPE34 */
    if('l'==type) {
#if !defined(G__OLDIMPLEMENTATION2189)
      /* int tmptypenum; */
      type = 'q';
      strcpy(tagname,""); /* ??? */
#elif !defined(G__OLDIMPLEMENTATION1836)
      int tmptypenum;
      G__loadlonglong(&itemp,&tmptypenum,G__LONGDOUBLE);
      type = 'u';
#ifndef G__PHILIPPE34
      strcpy(tagname,"G__longdouble");
#endif /* G__PHILIPPE34 */
#else /* 1836 */
      if(0==G__defined_macro("G__LONGLONG_H")) {
#ifndef G__OLDIMPLEMENTATION1153
	int store_def_struct_member = G__def_struct_member;
	G__def_struct_member = 0;
#endif
	G__loadfile("long.dll"); /* used to switch case between .dl and .dll */
#ifndef G__OLDIMPLEMENTATION1153
	G__def_struct_member = store_def_struct_member;
#endif
      }

#ifndef G__OLDIMPLEMENTATION1686
      strcpy(tagname,"G__ulonglong");
      itemp=G__defined_tagname(tagname,2);
      G__search_typename("unsigned long long",'u',itemp,G__PARANORMAL);
#endif
      strcpy(tagname,"G__longlong");
      itemp=G__defined_tagname(tagname,2);
      G__search_typename("long long",'u',itemp,G__PARANORMAL);

      strcpy(tagname,"G__longdouble");
      itemp=G__defined_tagname(tagname,2);
      if(-1==itemp) {
	G__genericerror("Error: 'long double' not ready. Go to $CINTSYSDIR/lib/longlong and run setup");
      }
      G__search_typename("long double",'u',itemp,G__PARANORMAL);
      type='u';
#endif /* 1836 */
    }
    c=G__fgetname(typename,";,[");
  }
#endif

  /* in case of
   *  typedef unsigned long int  int32;
   *                           ^
   *  read typename
   */
#ifndef G__PHILIPPE34
  if(strncmp(typename,"int",3)==0 
     && ( strlen(typename)==3
          || (strlen(typename)>=4 && (typename[3]=='&' || typename[3]=='*')) )
     ) {
     if (strlen(typename)>=4) {
       /* Rewind. */
       fseek(G__ifile.fp,-1-(strlen(typename) - strlen("int")) ,SEEK_CUR);
     }
#else
  if(strcmp(typename,"int")==0) {
#endif /* G__PHILIPPE34 */
    c=G__fgetstream(typename,";,[");
  }
#ifndef G__PHILIPPE34
  if(strcmp(typename,"*")==0) {
#else
  else if(strcmp(typename,"*")==0) {
#endif /* G__PHILIPPE34 */   
#ifndef G__OLDIMPLEMENTATION1396
    fpos_t tmppos;
    int tmpline = G__ifile.line_number;
    fgetpos(G__ifile.fp,&tmppos);
    c=G__fgetname(typename+1,";,[");
    if(isspace(c) && strcmp(typename,"*const")==0) {
      isconst |= G__PCONSTVAR;
      c=G__fgetstream(typename+1,";,[");
    }
    else {
      G__disp_mask = strlen(typename)-1;
      G__ifile.line_number = tmpline;
      fsetpos(G__ifile.fp,&tmppos);
      c=G__fgetstream(typename+1,";,[");
    }
#else
    c=G__fgetstream(typename+1,";,[");
#endif
  }
  else if(strcmp(typename,"**")==0) {
#ifndef G__OLDIMPLEMENTATION1396
    fpos_t tmppos;
    int tmpline = G__ifile.line_number;
    fgetpos(G__ifile.fp,&tmppos);
    c=G__fgetname(typename+1,";,[");
    if(isspace(c) && strcmp(typename,"*const")==0) {
      isconst |= G__PCONSTVAR;
      c=G__fgetstream(typename+1,";,[");
    }
    else {
      G__disp_mask = strlen(typename)-1;
      G__ifile.line_number = tmpline;
      fsetpos(G__ifile.fp,&tmppos);
      c=G__fgetstream(typename+1,";,[");
    }
#else
    c=G__fgetstream(typename+1,";,[");
#endif
    isorgtypepointer=1;
    type=toupper(type);
  }
#ifndef G__OLDIMPLEMENTATION673
  else if(strcmp(typename,"&")==0) {
    reftype = G__PARAREFERENCE;
    c=G__fgetstream(typename,";,[");
  }
  else if(strcmp(typename,"*&")==0) {
    reftype = G__PARAREFERENCE;
    type=toupper(type);
    c=G__fgetstream(typename,";,[");
  }
#endif
#ifndef G__OLDIMPLEMENTATION1396
  else if(strcmp(typename,"*const")==0) {
    isconst |= G__PCONSTVAR;
    c=G__fgetstream(typename+1,";,[");
  }
#endif
#ifndef G__OLDIMPLEMENTATION1856
  else if(strcmp(typename,"const*")==0) {
    isconst |= G__CONSTVAR;
    type=toupper(type);
    c=G__fgetstream(typename,"*&;,[");
    if('*'==c && '*'!=typename[0]) {
      if(strcmp(typename,"const")==0) isconst |= G__CONSTVAR;
      typename[0] = '*';
      c=G__fgetstream(typename+1,";,[");
    }
    if('&'==c && '&'!=typename[0]) {
      reftype = G__PARAREFERENCE;
      if(strcmp(typename,"const")==0) isconst |= G__CONSTVAR;
      c=G__fgetstream(typename,";,[");
    }
  }
  else if(strcmp(typename,"const**")==0) {
    isconst |= G__CONSTVAR;
    isorgtypepointer=1;
    type=toupper(type);
    typename[0] = '*';
    c=G__fgetstream(typename+1,"*;,[");
  }
  else if(strcmp(typename,"const*&")==0) {
    isconst |= G__CONSTVAR;
    reftype = G__PARAREFERENCE;
    type=toupper(type);
    c=G__fgetstream(typename,";,[");
  }
#endif

  if(isspace(c)) {
    if('('==typename[0] && ';'!=c && ','!=c) {
      do {
	c=G__fgetstream(typename+strlen(typename),";,");
	sprintf(typename+strlen(typename),"%c",c);
      } while(';'!=c && ','!=c);
      typename[strlen(typename)-1]='\0';
    }
    else if(strcmp(typename,"const")==0) {
#ifndef G__OLDIMPLEMENTATION1394
      isconst |= G__PCONSTVAR;
#endif
      c=G__fgetstream(typename,";,[");
#ifndef G__OLDIMPLEMENTATION1868
      if(strncmp(typename,"*const*",7)==0) {
	isconst |= G__CONSTVAR;
	isorgtypepointer=1;
	type=toupper(type);
	G__shiftstring(typename,6);
      }
      else if(strncmp(typename,"*const&",7)==0) {
	isconst |= G__CONSTVAR;
	reftype = G__PARAREFERENCE;
	type=toupper(type);
	G__shiftstring(typename,7);
      }
      else if(strncmp(typename,"const*",6)==0) {
      }
      else if(strncmp(typename,"const&",6)==0) {
      }
#endif
    }
#ifndef G__OLDIMPLEMENTATION1799
    else if(strcmp(typename,"const*")==0) {
      isconst |= G__PCONSTVAR;
      typename[0] = '*';
      c=G__fgetstream(typename+1,";,[");
    }
#endif
    else {
#ifndef G__OLDIMPLEMENTATION1347
      char ltemp1[G__LONGLINE];
      c = G__fgetstream(ltemp1,";,[");
      if('('==ltemp1[0]) {
	type = 'q';
      }
#else
      c = G__fignorestream(";,[");
#endif
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

  p=strchr(typename,'(');
  if(p) {

#ifndef G__OLDIMPLEMENTATION1763
    flag_p2f = 1;
#endif
    if(p==typename) {
      /* function to pointer 'typedef type (*newtype)();'
       * handle this as 'typedef void* newtype;'
       */
      strcpy(val,p+1);
      p=strchr(val,')');
      *p='\0';
      strcpy(typename,val);
      type='y';
      p = strstr(typename,"::*");
      if(p) {
	/* pointer to member function 'typedef type (A::*p)(); */
	strcpy(val,p+3);
	strcpy(typename,val);
	type='a';
      }
    }
    else if(p==typename+1 && '*'==typename[0]) {
      /* function to pointer 'typedef type *(*newtype)();'
       * handle this as 'typedef void* newtype;'
       */
      strcpy(val,p+1);
      p=strchr(val,')');
      *p='\0';
      strcpy(typename,val);
#ifndef G__OLDIMPLEMENTATION729
      type='Q';
#else
      type='Y';
#endif
      p = strstr(typename,"::*");
      if(p) {
	/* pointer to member function 'typedef type (A::*p)(); */
	strcpy(val,p+3);
	strcpy(typename,val);
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
  if(typename[0]=='*') {
    int ix=1;
    if(isupper(type)
#ifndef G__OLDIMPLEMENTATION933
#ifndef G__OLDIMPLEMENTATION2191
       &&'1'!=type
#else
       &&'Q'!=type
#endif
#endif
       ) {
      reftype = G__PARAP2P;
#ifndef G__OLDIMPLEMENTATION2094  /* 919 */
      while(typename[ix]=='*') {
        if(G__PARANORMAL==reftype) reftype = G__PARAP2P;
        else if(reftype>=G__PARAP2P) ++ reftype;
        ++ix;
      }
#endif
    }
    else {
      type=toupper(type);
#ifndef G__OLDIMPLEMENTATION919
      while(typename[ix]=='*') {
        if(G__PARANORMAL==reftype) reftype = G__PARAP2P;
        else if(reftype>=G__PARAP2P) ++ reftype;
        ++ix;
      }
#else
      if(typename[1]=='*') {
        reftype = G__PARAP2P;
        ++ix;
      }
#endif
    }
    strcpy(val,typename);
    strcpy(typename,val+ix);
  }

#ifndef G__OLDIMPLEMENTATION673
  /* typedef oldtype &newtype */
  if(typename[0]=='&') {
#ifndef G__OLDIMPLEMENTATION2033
    if(reftype>=G__PARAP2P) reftype += G__PARAREF;
    else                    reftype = G__PARAREFERENCE;
#else
    if(G__PARAP2P==reftype) G__fprinterr(stderr,"cint internal limitation in %s %d\n",__FILE__,__LINE__);
    reftype = G__PARAREFERENCE;
#endif
    if(strlen(typename)>1) {
      strcpy(val,typename);
      strcpy(typename,val+1);
    }
    else {
      /* to be determined */
    }
  }
#endif

  /*
   * check if typedef hasn't been defined
   */
#ifndef G__OLDIMPLEMENTATION559
  typenum = G__defined_typename_exact(typename);
#else
  typenum = G__defined_typename(typename);
#endif

  /*
   * if new typedef, add it to newtype table
   */
  if(-1==typenum) {

    if(G__newtype.alltype==G__MAXTYPEDEF) {
      G__fprinterr(G__serr,
	      "Limitation: Number of typedef exceed %d FILE:%s LINE:%d\nFatal error, exit program. Increase G__MAXTYPEDEF in G__ci.h and recompile %s\n"
	      ,G__MAXTYPEDEF
	      ,G__ifile.name
	      ,G__ifile.line_number
	      ,G__nam);
      G__eof=1;
      return;
    }
    typenum = G__newtype.alltype;

    len=strlen(typename);
    G__newtype.name[typenum] = malloc((size_t)(len+2));
    strcpy(G__newtype.name[typenum],typename);
    G__newtype.iscpplink[typenum] = G__NOLINK;
#ifdef G__FONS_COMMENT
#ifndef G__OLDIMPLEMENTATION469
    G__newtype.comment[typenum].p.com = (char*)NULL;
#else
    G__newtype.comment[typenum].p.pos = (fpos_t)0;
#endif
    G__newtype.comment[typenum].filenum = -1;
#endif
    G__newtype.nindex[typenum]=nindex;
#ifdef G__TYPEDEFFPOS
    G__newtype.filenum[typenum] = G__ifile.filenum;
    G__newtype.linenum[typenum] = G__ifile.line_number;
#endif
    if(nindex) {
      G__newtype.index[typenum]
	=(int*)malloc((size_t)(G__INTALLOC*nindex));
      memcpy((void*)G__newtype.index[typenum],(void*)index
	     ,G__INTALLOC*nindex);
    }
    G__newtype.hash[typenum] = len;
    if(tagname[0]=='\0') {
#ifndef G__OLDIMPLEMENTATION825
      if(G__CPPLINK==G__globalcomp) sprintf(tagname,"%s",typename);
      else                          sprintf(tagname,"$%s",typename);
#else
      sprintf(tagname,"$%s",typename);
#endif
      taglen=strlen(tagname);
    }
    else {
      taglen=strlen(tagname);
      if(tagname[taglen-1]=='*') {
	type=toupper(type);
	tagname[taglen-1]='\0';
      }
    }
    G__newtype.tagnum[typenum] = -1;

    /*
     * maybe better to change G__defined_type
     */
    G__newtype.type[typenum]=type;
#ifndef G__OLDIMPLEMENTATION1700
    G__newtype.globalcomp[typenum]=G__default_link?G__globalcomp:G__NOLINK;
#else
    G__newtype.globalcomp[typenum]=G__globalcomp;
#endif
    G__newtype.reftype[typenum]=reftype;
#ifndef G__OLDIMPLEMENTATION1394
    G__newtype.isconst[typenum] = isconst;
#endif

    if(G__def_struct_member) env_tagnum = G__tagnum;
    else if(-1!=G__func_now) {
      env_tagnum = -2;
#ifndef G__OLDIMPLEMENTATION1145
      G__fprinterr(G__serr,"Limitation: In function typedef not allowed in cint");
      G__printlinenum();
#endif
    }
    else                     env_tagnum = -1;
    G__newtype.parent_tagnum[typenum]=env_tagnum;
    ++G__newtype.alltype;
  }

  /*
   *  return if the type is already defined
   */
  else {
    if(';'!=c) G__fignorestream(";");
    return;
  }

  if(tolower(type)=='u') {


    G__tagnum=G__search_tagname(tagname,tagtype);
    if(G__tagnum<0) {
      G__fignorestream(";");
      return;
    }
    G__newtype.tagnum[typenum]=G__tagnum;

    if(mem_def==1) {

      if(G__struct.size[G__tagnum]==0) {
	fsetpos(G__ifile.fp,&rewind_fpos);

	G__struct.line_number[G__tagnum] = G__ifile.line_number;
	G__struct.filenum[G__tagnum] = G__ifile.filenum;

	G__struct.parent_tagnum[G__tagnum]=env_tagnum;
	
	/*
	 * in case of enum
	 */
	if(tagtype=='e') {
	  G__disp_mask=10000;
	  while((c=G__fgetc())!='{') ;
	  enumval.obj.i = -1;
	  enumval.type = 'i' ;
	  enumval.tagnum = G__tagnum ;
	  enumval.typenum = typenum ;
	  G__constvar=G__CONSTVAR;
	  G__enumdef=1;
	  do {
	    c=G__fgetstream(memname,"=,}");
	    if(c=='=') {
#ifndef G__OLDIMPLEMENTATION1676
	      int store_prerun = G__prerun;
#endif
#ifndef G__OLDIMPLEMENTATION1337
	      char store_var_type = G__var_type;
	      G__var_type = 'p';
#endif
#ifndef G__OLDIMPLEMENTATION1676
	      G__prerun = 0;
#endif
	      c=G__fgetstream(val,",}");
	      enumval=G__getexpr(val);
#ifndef G__OLDIMPLEMENTATION1676
	      G__prerun = store_prerun;
#endif
#ifndef G__OLDIMPLEMENTATION1337
	      G__var_type = store_var_type;
#endif
	    }
	    else {
	      enumval.obj.i++;
	    }
	    G__var_type='i';
	    if(-1!=store_tagnum) {
	      store_def_struct_member=G__def_struct_member;
	      G__def_struct_member=0;
	      G__static_alloc=1;
	    }
	    G__letvariable(memname,enumval,&G__global ,G__p_local);
	    if(-1!=store_tagnum) {
	      G__def_struct_member=store_def_struct_member;
	      G__static_alloc=0;
	    }
	  } while(c!='}') ;
	  G__constvar=0;
	  G__enumdef=0;
	
	  G__fignorestream(";");
	  G__disp_mask=0;
	  G__ifile.line_number=temp_line;
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
	
	  store_local = G__p_local;
	  G__p_local=G__struct.memvar[G__tagnum];
	
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
#ifndef G__OLDIMPLEMENTATION1777
	  if(1==G__struct.memvar[G__tagnum]->allvar) {
	    /* this is still questionable, inherit0.c */
	    struct G__var_array *v=G__struct.memvar[G__tagnum];
	    if('c'==v->type[0]) {
	      if(isupper(v->type[0])) {
		G__struct.size[G__tagnum] = G__LONGALLOC*(v->varlabel[0][1]+1);
	      }
	      else {
		G__value buf;
		buf.type = v->type[0];
		buf.tagnum = v->p_tagtable[0];
		buf.typenum = v->p_typetable[0];
		G__struct.size[G__tagnum]
		  =G__sizeof(&buf)*(v->varlabel[0][1]+1);
	      }
	    }
	  } else
#endif
	  if(G__struct.size[G__tagnum]%G__DOUBLEALLOC) {
	    G__struct.size[G__tagnum]
	      += G__DOUBLEALLOC
		- G__struct.size[G__tagnum]%G__DOUBLEALLOC;
	  }
#ifndef G__OLDIMPLEMENTATION591
	  else if(0==G__struct.size[G__tagnum]) {
	    G__struct.size[G__tagnum] = G__CHARALLOC;
	  }
#endif
	
	  G__tagdefining = store_tagdefining;
	
	  G__def_struct_member=store_def_struct_member;
	  G__mparen=mparen;
	  G__p_local = store_local;
	
	  G__fignorestream(";");
	
	  G__disp_mask=0;
	  G__ifile.line_number=temp_line;
	} /* end of struct, class , union */
      } /* end of G__struct.size[G__tagnum]==0 */
    } /* end of mem_def==1 */

    else { /* mem_def!=1 */
      /* oldlink eliminated */
    }  /* of mem_def */

    G__tagnum=store_tagnum;

  } /* end of struct,class,union,enum */
  else { /* scalar type */
    /* oldlink eliminated */
  } /* end of scalar type */

  if(isnext) {
    fsetpos(G__ifile.fp,&next_fpos);
    c=G__fgetstream(typename,",;");
    goto next_name;
  }

#ifdef G__FONS_COMMENT
  if(G__fons_comment) {
    G__fsetcomment(&G__newtype.comment[G__newtype.alltype-1]);
  }
#endif

#ifndef G__OLDIMPLEMENTATION1763
  if(flag_p2f 
     && G__newtype.comment[G__newtype.alltype-1].filenum<0 
     && !G__newtype.comment[G__newtype.alltype-1].p.com) {
    fpos_t xpos;
#ifndef G__OLDIMPLEMENTATION1920
    if(G__ifile.filenum > G__nfile) {
      G__fprinterr(G__serr
	   ,"Warning: pointer to function typedef incomplete in command line or G__exec_text(). Declare in source file or use G__load_text()\n");
      return;
    }
    ++G__macroORtemplateINfile;
#endif
    fgetpos(G__ifile.fp,&xpos);
    fsetpos(G__ifile.fp,&pos_p2fcomment);

    if(G__ifile.fp==G__mfp) 
      G__newtype.comment[G__newtype.alltype-1].filenum = G__MAXFILE;
    else
      G__newtype.comment[G__newtype.alltype-1].filenum = G__ifile.filenum;
    fgetpos(G__ifile.fp,&G__newtype.comment[G__newtype.alltype-1].p.pos);

    fsetpos(G__ifile.fp,&xpos);
  }
#endif

}


/******************************************************************
* G__defined_typename(typename)
*
* Search already defined typedef names, -1 is returned if not found
* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
* Note that this modify G__var_type, you may need to reset it after
* calling this function
* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
******************************************************************/
int G__defined_typename(typename)
char *typename;
{
  int i;
  int len;
  char ispointer=0;
#ifndef G__OLDIMPLEMENTATION1823
  char buf[G__BUFLEN];
  char buf2[G__BUFLEN];
  char *temp=buf;
  char *temp2=buf2;
#else
  char temp[G__LONGLINE];
  char temp2[G__LONGLINE];
#endif
  char *p;
  int env_tagnum;
  int typenum = -1;
  unsigned long matchflag=0;
  unsigned long thisflag=0;
#ifndef G__OLDIMPLEMENTATION745
  char *par;
#endif

#ifndef G__OLDIMPLEMENTATION1823
  if(strlen(typename)>G__BUFLEN-10) {
    temp2=(char*)malloc(strlen(typename)+10);
    temp=(char*)malloc(strlen(typename)+10);
  }
#endif
  strcpy(temp2,typename);

  /* find 'xxx::yyy' */
  p = G__find_last_scope_operator (temp2);


#ifndef G__OLDIMPLEMENTATION745
  /* abandon scope operator if 'zzz (xxx::yyy)www' */
  par = strchr(temp2,'(');
  if(par && p && par<p) p=(char*)NULL;
#endif

  if(p) {
    strcpy(temp,p+2);
    *p='\0';
    if(temp2==p) env_tagnum = -1; /* global scope */
#ifndef G__STD_NAMESPACE /* ON745 */
    else if (strcmp (temp2, "std") == 0
#ifndef G__OLDIMPLEMENTATION1285
	     && G__ignore_stdnamespace
#endif
	     ) env_tagnum = -1;
#endif
    else         env_tagnum = G__defined_tagname(temp2,0);
  }
  else {
    strcpy(temp,temp2);
    env_tagnum = G__get_envtagnum();
  }

  len=strlen(temp);

  if(
#ifndef G__OLDIMPLEMENTATION1863
     len>0 && 
#endif
     temp[len-1]=='*') {
    temp[--len]='\0';
    ispointer = 'A' - 'a';
  }

#ifndef G__OLDIMPLEMENTATION620
  for(i=0;i<G__newtype.alltype;i++) {
    if(len==G__newtype.hash[i] && strcmp(G__newtype.name[i],temp)==0) {
      thisflag=0;
      /* global scope */
      if(-1==G__newtype.parent_tagnum[i]
#if !defined(G__OLDIMPLEMTATION2100)
	 && (!p || (temp2==p || strcmp("std",temp2)==0))
#elif !defined(G__OLDIMPLEMTATION1890)
	 && (!p || temp2==p)
#endif
	 )
	thisflag=0x1;
      /* enclosing tag scope */
      if(G__isenclosingclass(G__newtype.parent_tagnum[i],env_tagnum))
	thisflag=0x2;
      /* template definition enclosing class scope */
      if(G__isenclosingclass(G__newtype.parent_tagnum[i],G__tmplt_def_tagnum))
	thisflag=0x4;
      /* baseclass tag scope */
      if(-1!=G__isanybase(G__newtype.parent_tagnum[i],env_tagnum
			  ,G__STATICRESOLUTION))
	thisflag=0x8;
      /* template definition base class scope */
      if(-1!=G__isanybase(G__newtype.parent_tagnum[i],G__tmplt_def_tagnum
			  ,G__STATICRESOLUTION))
	thisflag=0x10;
#ifndef G__OLDIMPLEMENTATION949
      if(thisflag == 0 &&
	 G__isenclosingclassbase(G__newtype.parent_tagnum[i],env_tagnum))
        thisflag = 0x02;
      if(thisflag == 0 &&
	 G__isenclosingclassbase(G__newtype.parent_tagnum[i],
				 G__tmplt_def_tagnum))
        thisflag = 0x04;
#endif
      /* exact template definition scope */
      if(0<=G__tmplt_def_tagnum &&
	 G__tmplt_def_tagnum==G__newtype.parent_tagnum[i])
	thisflag=0x20;
      /* exact tag scope */
      if(0<=env_tagnum && env_tagnum==G__newtype.parent_tagnum[i])
	thisflag=0x40;

      if(thisflag && thisflag>=matchflag) {
	matchflag = thisflag;
	typenum = i;
	/* This must be a bad manner. Somebody needs to reset G__var_type
	 * especially when typein is 0. */
	G__var_type=G__newtype.type[i] + ispointer ;
      }
    }
  }
#ifndef G__OLDIMPLEMENTATION1823
  if(temp!=buf) free((void*)temp);
  if(temp2!=buf2) free((void*)temp2);
#endif
  return(typenum);

#else
  for(i=0;i<G__newtype.alltype;i++) {
    if(len==G__newtype.hash[i] && strcmp(G__newtype.name[i],temp)==0 &&
       (-1==G__newtype.parent_tagnum[i]||
	env_tagnum==G__newtype.parent_tagnum[i]||
	-1!=G__isanybase(G__newtype.parent_tagnum[i],env_tagnum
			 ,G__STATICRESOLUTION)||
	G__isenclosingclass(G__newtype.parent_tagnum[i],env_tagnum)
	||G__tmplt_def_tagnum==G__newtype.parent_tagnum[i]
	||-1!=G__isanybase(G__newtype.parent_tagnum[i],G__tmplt_def_tagnum
			   ,G__STATICRESOLUTION)
	||G__isenclosingclass(G__newtype.parent_tagnum[i],G__tmplt_def_tagnum)
	)) {
      flag=1;
      /* This must be a bad manner. Somebody needs to reset G__var_type
       * especially when typein is 0. */
      G__var_type=G__newtype.type[i] + ispointer ;
      break;
    }
  }

#ifndef G__OLDIMPLEMENTATION1823
  if(temp!=buf) free((void*)temp);
  if(temp2!=buf2) free((void*)temp2);
#endif
  if(flag==0) return(-1);
  return(i);
#endif
}

#ifndef G__OLDIMPLEMENTATION708
/******************************************************************
* G__make_uniqueP2Ftypedef()
*
*  input  'void* (*)(int , void * , short )'
*  output 'void* (*)(int,void*,short)'
*
******************************************************************/
static int G__make_uniqueP2Ftypedef(typename)
char *typename;
{
  char *from;
  char *to;
  int spacecnt=0;
  int isstart=1;

  /*  input  'void* (*)(int , void * , short )'
   *         ^ start                         */
  from = strchr(typename,'(');
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
  from = strchr(typename,'(');
  if(!from) return(1);
  from = strchr(from+1,'(');
  if(!from) return(1);
  if(strcmp(from,"(void)")==0) {
    *(++from) = ')';
    *(++from) = 0;
  }

  return(0);
}
#endif

/******************************************************************
* G__search_typename(typename,type,tagnum,reftype)
*
* Used in G__cpplink.C
* Search typedef name. If not found, allocate new entry if typein
* isn't 0.
******************************************************************/
int G__search_typename(typenamein,typein,tagnum,reftype)
char *typenamein;
int typein;
int tagnum;
int reftype;
{
  int i,flag=0,len;
  char ispointer=0;

#ifndef G__OLDIMPLEMENTATION708
  char typename[G__LONGLINE];
  strcpy(typename,typenamein);
  /* keep uniqueness for pointer to function typedefs */
#ifndef G__OLDIMPLEMENTATION2191
  if('1'==typein) G__make_uniqueP2Ftypedef(typename);
#else
  if('Q'==typein) G__make_uniqueP2Ftypedef(typename);
#endif
#endif
  
/* G__OLDIMPLEMENTATIONON620 should affect, but not implemented here */
  /* Doing exactly the same thing as G__defined_typename() */
  len=strlen(typename);
  if(
#ifndef G__OLDIMPLEMENTATION1370
     len &&
#endif
     typename[len-1]=='*') {
    typename[--len]='\0';
    ispointer = 'A' - 'a';
  }
  for(i=0;i<G__newtype.alltype;i++) {
#ifndef G__OLDIMPLEMENTATION777
    if(len==G__newtype.hash[i]&&strcmp(G__newtype.name[i],typename)==0 &&
       (G__static_parent_tagnum == -1 ||
	G__newtype.parent_tagnum[i]==G__static_parent_tagnum)) {
#else
    if(len==G__newtype.hash[i]&&strcmp(G__newtype.name[i],typename)==0) {
#endif
      flag=1;
      G__var_type=G__newtype.type[i] + ispointer ;
      break;
    }
  }
  /* Above is same as G__defined_typename() */
  
  /* allocate new type table entry */
  if(flag==0 && typein) {
    if(G__newtype.alltype==G__MAXTYPEDEF) {
      G__fprinterr(G__serr,
	      "Limitation: Number of typedef exceed %d FILE:%s LINE:%d\nFatal error, exit program. Increase G__MAXTYPEDEF in G__ci.h and recompile %s\n"
	      ,G__MAXTYPEDEF ,G__ifile.name ,G__ifile.line_number ,G__nam);
      G__eof=1;
      G__var_type = 'p';
      return(-1);
    }
    G__newtype.hash[G__newtype.alltype]=len;
    G__newtype.name[G__newtype.alltype]=malloc((size_t)(len+1));
    strcpy(G__newtype.name[G__newtype.alltype],typename);
    G__newtype.nindex[G__newtype.alltype] = 0;
#ifndef G__OLDIMPLEMENTATION777
    G__newtype.parent_tagnum[G__newtype.alltype] = G__static_parent_tagnum;
#else
    G__newtype.parent_tagnum[G__newtype.alltype] = -1;
#endif
#ifndef G__OLDIMPLEMENTATION1394
    G__newtype.isconst[G__newtype.alltype] = G__static_isconst;
#endif
    G__newtype.type[G__newtype.alltype]=typein+ispointer;
    G__newtype.tagnum[G__newtype.alltype]=tagnum;
#ifndef G__OLDIMPLEMENTATION1700
    G__newtype.globalcomp[G__newtype.alltype]
      =G__default_link?G__globalcomp:G__NOLINK;
#else
    G__newtype.globalcomp[G__newtype.alltype]=G__globalcomp;
#endif
    G__newtype.reftype[G__newtype.alltype]=reftype;
    G__newtype.iscpplink[G__newtype.alltype] = G__NOLINK;
#ifdef G__FONS_COMMENT
#ifndef G__OLDIMPLEMENTATION469
    G__newtype.comment[G__newtype.alltype].p.com = (char*)NULL;
#else
    G__newtype.comment[G__newtype.alltype].p.pos = (fpos_t)0;
#endif
    G__newtype.comment[G__newtype.alltype].filenum = -1;
#endif
#ifdef G__TYPEDEFFPOS
    G__newtype.filenum[G__newtype.alltype] = G__ifile.filenum;
    G__newtype.linenum[G__newtype.alltype] = G__ifile.line_number;
#endif
    ++G__newtype.alltype;
  }
  return(i);
}

#ifndef G__OLDIMPLEMENTATION1743
#ifndef __CINT__
void G__setnewtype_settypeum G__P((int typenum));
#endif
#endif
/******************************************************************
* G__search_typename2()
******************************************************************/
int G__search_typename2(typename,typein,tagnum,reftype,parent_tagnum)
char *typename;
int typein;
int tagnum;
int reftype;
int parent_tagnum;
{
  int ret;
  G__static_parent_tagnum = parent_tagnum;
#ifndef G__OLDIMPLEMENTATION1284
  if(-1==G__static_parent_tagnum && G__def_struct_member &&
     'n'==G__struct.type[G__tagdefining]) {
    G__static_parent_tagnum = G__tagdefining;
  }
#endif
#ifndef G__OLDIMPLEMENTATION1394
  G__static_isconst = reftype/0x100;
  reftype = reftype%0x100;
#endif
  ret = G__search_typename(typename,typein,tagnum,reftype);
  G__static_parent_tagnum = -1;
#ifndef G__OLDIMPLEMENTATION1394
  G__static_isconst = 0;
#endif
#ifndef G__OLDIMPLEMENTATION1743
  G__setnewtype_settypeum(ret);
#endif
  return(ret);
}

/******************************************************************
* G__defined_type(typename,len)
*
* Search already defined typename and tagname
* and allocate automatic variables.
******************************************************************/
int G__defined_type(typename,len)
/* struct G__input_file *fin; */
char *typename;
int len;
{
  int store_tagnum,store_typenum;
  /* char type; */
  int cin;
  int refrewind = -2;
  fpos_t pos;
  int line;
#ifndef G__PHILIPPE14
  char store_typename[G__LONGLINE];  
#endif

  if(G__prerun&&'~'==typename[0]) {
    G__var_type = 'y';
    cin = G__fignorestream("(");
    typename[len++]=cin;
    typename[len]='\0';
    G__make_ifunctable(typename);
    return(1);
  }

#ifndef G__OLDIMPLEMENTATION632
  if(!isprint(typename[0]) && len==1) {
    return(1);
  }
#endif

  fgetpos(G__ifile.fp,&pos);
  line=G__ifile.line_number;
#ifndef G__PHILIPPE14
  /* this is not the fastest to insure proper unwinding in case of 
     error, but it is the simpliest :( */
  strcpy(store_typename,typename);
#endif

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
#ifndef G__OLDIMPLEMENTATION411
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
#else
  fseek(G__ifile.fp,-1,SEEK_CUR);
  if(G__dispsource) G__disp_mask=1;
  if(!isalpha(cin) && cin!='*' && cin !='_' && cin !='(' && cin!='&') {
    return(0);
  }
#endif
  
  if(typename[len-1]=='&') {
    G__reftype=G__PARAREFERENCE;
    typename[--len]='\0';
    --refrewind;
  }
  
  store_tagnum = G__tagnum;
  store_typenum=G__typenum;
  
  /* search for typedef names */
  if(len>2 && '*'==typename[len-1] && '*'==typename[len-2]) {
    /* pointer to pointer */
    len -=2;
    typename[len]='\0';
    /* type** a;
     *     ^<<^      */
    fsetpos(G__ifile.fp,&pos);
    G__ifile.line_number=line;
#ifndef G__PHILIPPE14
    /* the following fseek is now potentialy wrong (because of fake_space!) */
    fseek(G__ifile.fp,-1,SEEK_CUR);
    cin = G__fgetc();
    if (cin=='*') {
      /* we have a fake space */
      fseek(G__ifile.fp,refrewind,SEEK_CUR);
    } else {
      fseek(G__ifile.fp,refrewind-1,SEEK_CUR);
    }
#else
    fseek(G__ifile.fp,refrewind-1,SEEK_CUR);
#endif
    if(G__dispsource) G__disp_mask=2;
  }
  else if(len>1 && '*'==typename[len-1]) {
#ifndef G__OLDIMPLEMENTATION1262
    int cin2;
#endif
    len -=1;
    typename[len]='\0';
    fsetpos(G__ifile.fp,&pos);
    G__ifile.line_number=line;
#ifndef G__PHILIPPE24
    /* To know how much to rewind we need to know if there is a fakespace */
    fseek(G__ifile.fp,-1,SEEK_CUR);
    cin = G__fgetc();
    if (cin=='*') {
      /* we have a fake space */
      fseek(G__ifile.fp,refrewind+1,SEEK_CUR);
    } else {
      fseek(G__ifile.fp,refrewind,SEEK_CUR);
    }
#else
    fseek(G__ifile.fp,refrewind,SEEK_CUR);
#endif
    if(G__dispsource) G__disp_mask=1;
#ifndef G__OLDIMPLEMENTATION1262
    cin2 = G__fgetc();
    if(!isalnum(cin2)
#ifndef G__OLDIMPLEMENTATION1358
       && '>'!=cin2
#endif
       ) {
      fseek(G__ifile.fp,-1,SEEK_CUR);
      if(G__dispsource) G__disp_mask=1;
    }
#endif
  }

  G__typenum = G__defined_typename(typename);
  
  if(-1 == G__typenum) {
    /* search for class/struct/enum tagnames */
    G__tagnum = G__defined_tagname(typename,1);
    if(-1 == G__tagnum) {
#ifndef G__OLDIMPLEMENTATION411
      /* This change is risky. Need more evaluation */
      if(G__fpundeftype && '('!=cin &&
	 (-1==G__func_now || -1!=G__def_tagnum)) {
	G__tagnum=G__search_tagname(typename,'c');
	fprintf(G__fpundeftype,"class %s; /* %s %d */\n",typename
		,G__ifile.name,G__ifile.line_number);
#ifndef G__OLDIMPLEMENTATION1133
	fprintf(G__fpundeftype,"#pragma link off class %s;\n\n",typename);
	G__struct.globalcomp[G__tagnum] = G__NOLINK;
#endif
      }
      else {
	/* if not found, return */
#ifndef G__PHILIPPE14
        /* Restore properly the previous state! */
        fsetpos(G__ifile.fp,&pos);
        G__ifile.line_number = line;
        strcpy(typename,store_typename);
#endif  
	G__tagnum = store_tagnum;
	G__typenum = store_typenum;
	G__reftype=G__PARANORMAL;
	return(0);
      }
#else
      /* if not found, return */
      G__tagnum = store_tagnum;
      G__typenum = store_typenum;
      G__reftype=G__PARANORMAL;
      return(0);
#endif
    }
#ifndef G__OLDIMPLEMENTATION1244
    else {
      G__typenum = G__defined_typename(typename);
      if(-1!=G__typenum) {
	G__reftype += G__newtype.reftype[G__typenum];
	G__typedefnindex = G__newtype.nindex[G__typenum];
	G__typedefindex = G__newtype.index[G__typenum];
      }
    }
#endif
    G__var_type = 'u';
  }
  else {
    G__tagnum=G__newtype.tagnum[G__typenum];
    /* Questionable adding of G__reftype, maybe |= instead */
    G__reftype += G__newtype.reftype[G__typenum];
    G__typedefnindex = G__newtype.nindex[G__typenum];
    G__typedefindex = G__newtype.index[G__typenum];
  }

  if(-1 != G__tagnum && G__struct.type[G__tagnum]=='e') {
    /* in case of enum */
    G__var_type='i';
  }
  
  
  /* allocate variable */
  G__define_var(G__tagnum,G__typenum);
  
  G__typedefnindex = 0;
  G__typedefindex = (int*)NULL;
  
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
