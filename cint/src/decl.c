/* /% C %/ */
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
 * Permission to use, copy, modify and distribute this software and its 
 * documentation for any purpose is hereby granted without fee,
 * provided that the above copyright notice appear in all copies and
 * that both that copyright notice and this permission notice appear
 * in supporting documentation.  The author makes no
 * representations about the suitability of this software for any
 * purpose.  It is provided "as is" without express or implied warranty.
 ************************************************************************/

#include "common.h"

extern char G__declctor[];

#ifndef G__OLDIMPLEMENTATION1103
extern int G__const_noerror;
#endif
#ifndef G__OLDIMPLEMENTATION1119
int G__initval_eval=0;
int G__dynconst=0;
#endif


#ifndef G__OLDIMPLEMENTATION1836
/**************************************************************************
* G__loadlonglong()
**************************************************************************/
void G__loadlonglong(ptag,ptype,which)
int* ptag;
int* ptype;
int which;
{
  int lltag= -1,lltype= -1;
  int ulltag= -1,ulltype= -1;
  int ldtag= -1,ldtype= -1;
  int store_decl = G__decl;
  int store_def_struct_member = G__def_struct_member;
  int flag=0;
  int store_tagdefining=G__tagdefining;
  int store_def_tagnum=G__def_tagnum;

  G__tagdefining = -1;
  G__def_tagnum = -1;
  G__def_struct_member = 0;
  G__decl = 0;
  if(0==G__defined_macro("G__LONGLONG_H")) {
    G__loadfile("long.dll"); /* used to switch case between .dl and .dll */
    flag=1;
  }

  G__decl = 1;
  G__def_struct_member = store_def_struct_member;

  if(which==G__LONGLONG || flag) {
    lltag=G__defined_tagname("G__longlong",2);
    lltype=G__search_typename("long long",'u',G__tagnum,G__PARANORMAL);
    G__struct.defaulttypenum[lltag] = lltype;
    G__newtype.tagnum[lltype] = lltag;
  }

  if(which==G__ULONGLONG || flag) {
    ulltag=G__defined_tagname("G__ulonglong",2);
    ulltype
      = G__search_typename("unsigned long long",'u',G__tagnum,G__PARANORMAL);
    G__struct.defaulttypenum[ulltag] = ulltype;
    G__newtype.tagnum[ulltype] = ulltag;
  }

  if(which==G__LONGDOUBLE || flag) {
    ldtag=G__defined_tagname("G__longdouble",2);
    ldtype=G__search_typename("long double",'u',G__tagnum,G__PARANORMAL);
    G__struct.defaulttypenum[ldtag] = ldtype;
    G__newtype.tagnum[ldtype] = ldtag;
  }

  switch(which) {
  case G__LONGLONG:
    *ptag = lltag;
    *ptype = lltype;
    break;
  case G__ULONGLONG:
    *ptag = ulltag;
    *ptype = ulltype;
    break;
  case G__LONGDOUBLE:
    *ptag = ldtag;
    *ptype = ldtype;
    break;
  }

  G__def_tagnum = store_def_tagnum;
  G__tagdefining = store_tagdefining;
  G__decl = store_decl;
  return ;
}
#endif


/***********************************************************************
* G__get_newname()
*
***********************************************************************/
int G__get_newname(new_name)
char *new_name;
{
  char temp[G__ONELINE],temp1[G__ONELINE];
  /* char *endmark=",;=():+-*%/^<>&|=![~@"; */
  int cin;
  int store_def_struct_member,store_tagdefining;

#ifndef G__PHILIPPE12
#ifndef G__OLDIMPLEMENTATION2061
  cin=G__fgetvarname(new_name,"*&,;=():}");
#else
  cin=G__fgetvarname(new_name,"&,;=():}");
#endif
  if (cin=='&') {
#ifndef G__OLDIMPLEMENTATION1353
    if(0==strcmp(new_name,"operator")) {
      new_name[8] = cin;
      cin=G__fgetvarname(new_name+9,",;=():}");
    }
    else {
      strcat(new_name,"&");
      cin = ' ';
    }
#else
    strcat(new_name,"&");
    cin = ' ';
#endif
  }
#ifndef G__OLDIMPLEMENTATION2061
  else if (cin=='*') {
    if(0==strcmp(new_name,"operator")) {
      new_name[8] = cin;
      cin=G__fgetvarname(new_name+9,",;=():}");
    }
    else {
      strcat(new_name,"*");
      cin = ' ';
    }
  }
#endif
#else
  cin=G__fgetvarname(new_name,",;=():}");
#endif


  /*********************************************************
   * for overloading of operator
   * and operator function
   *********************************************************/
  /*********************************************************
   * C++ 
   * Definition of operator function
   * type operator [/+-*%^&|] (type para1 , type para2)
   *********************************************************/
  /* In case of
   * type  operator  +(var1 , var2);
   *                ^
   * type  int   operator +(var1 , var2);
   *           ^
   * type  int  var1 , var2;
   *           ^
   * read variable name
   */
  if(isspace(cin)) {

#ifndef G__OLDIMPLEMENTATION761 
    if(strcmp(new_name,"const*")==0) {
      new_name[0]='*';
      cin=G__fgetvarname(new_name+1,",;=():}");
      G__constvar |= G__CONSTVAR;
    }
#endif

    if(strcmp(new_name,"friend")==0) {
      store_def_struct_member=G__def_struct_member;
      store_tagdefining=G__tagdefining;
      G__def_struct_member = 0;
      G__tagdefining = -1;
      G__define_var(G__tagnum,G__typenum);
      G__def_struct_member = store_def_struct_member;
      G__tagdefining = store_tagdefining;
      new_name[0]='\0';
      return(';');
    }
    else if(strcmp(new_name,"&")==0 || strcmp(new_name,"*")==0) {
      cin=G__fgetvarname(new_name+1,",;=():");
    }
#ifndef G__PHILIPPE11
    else if(strcmp(new_name,"&*")==0 || strcmp(new_name,"*&")==0) {
      cin=G__fgetvarname(new_name+2,",;=():");
    } 
#endif

    if(strcmp(new_name,"double")==0
#ifndef G__OLDIMPLEMENTATION1533
       && 'l'!=G__var_type
#endif
       ) {
      cin=G__fgetvarname(new_name,",;=():");
      G__var_type='d';
    }
    else if(strcmp(new_name,"int")==0) {
      cin=G__fgetvarname(new_name,",;=():");
    }
#ifndef G__OLDIMPLEMENTATION883
    else if(strcmp(new_name,"long")==0 ||
	    strcmp(new_name,"long*")==0 ||
	    strcmp(new_name,"long**")==0 ||
	    strcmp(new_name,"long&")==0) {
      int store_tagnum = G__tagnum;
      int store_typenum = G__typenum;
      int store_decl = G__decl;
#if !defined(G__OLDIMPLEMENTATION2189)
#elif !defined(G__OLDIMPLEMENTATION1836)
      if(-1==G__unsigned) {
	G__loadlonglong(&G__tagnum,&G__typenum,G__ULONGLONG);
      }
      else {
	G__loadlonglong(&G__tagnum,&G__typenum,G__LONGLONG);
      }
#else /* 1836 */
      if(0==G__defined_macro("G__LONGLONG_H")) {
#ifndef G__OLDIMPLEMENTATION1153
	int store_def_struct_member = G__def_struct_member;
	G__def_struct_member = 0;
#endif
	G__decl=0;
	G__loadfile("long.dll"); /* used to switch case between .dl and .dll */
	G__decl=1;
#ifndef G__OLDIMPLEMENTATION1153
	G__def_struct_member = store_def_struct_member;
#endif
      }
#ifndef G__OLDIMPLEMENTATION1686
      G__tagnum=G__defined_tagname("G__ulonglong",2);
      G__typenum=G__search_typename("unsigned long long",'u',G__tagnum,G__PARANORMAL);
#endif
#ifndef G__OLDIMPLEMENTATION1533
      G__tagnum=G__defined_tagname("G__longdouble",2);
      G__typenum=G__search_typename("long double",'u',G__tagnum,G__PARANORMAL);
#endif
      G__tagnum=G__defined_tagname("G__longlong",2);
      if(-1==G__tagnum) {
	G__genericerror("Error: 'long long' not ready. Go to $CINTSYSDIR/lib/longlong and run setup");
      }
#ifndef G__OLDIMPLEMENTATION1688
      if(-1==G__unsigned) 
	G__typenum=G__search_typename("unsigned long long",'u',G__tagnum,G__PARANORMAL);
      else
	G__typenum=G__search_typename("long long",'u',G__tagnum,G__PARANORMAL);
#else
      G__typenum=G__search_typename("long long",'u',G__tagnum,G__PARANORMAL);
#endif
#endif /* 1836 */
      if(strcmp(new_name,"long")==0) {
#ifndef G__OLDIMPLEMENTATION2189
	G__var_type='n' + G__unsigned;
	G__tagnum = -1;
	G__typenum = -1;
	G__reftype = G__PARANORMAL;
#else
	fpos_t pos;
	int xlinenum = G__ifile.line_number;
	fgetpos(G__ifile.fp,&pos); 
	cin=G__fgetvarname(new_name,",;=():");
	if(strcmp(new_name,"int")!=0) {
	  fsetpos(G__ifile.fp,&pos);
	  G__ifile.line_number=xlinenum;
	}
	G__var_type='u';
	G__reftype = G__PARANORMAL;
#endif
      }
      else if(strcmp(new_name,"long*")==0) {
#ifndef G__OLDIMPLEMENTATION2189
	G__var_type='N' + G__unsigned;
	G__tagnum = -1;
	G__typenum = -1;
	G__reftype = G__PARANORMAL;
#else
	G__var_type='U';
	G__reftype = G__PARANORMAL;
#endif
      }
      else if(strcmp(new_name,"long**")==0) {
#ifndef G__OLDIMPLEMENTATION2189
	G__var_type='N' + G__unsigned;
	G__tagnum = -1;
	G__typenum = -1;
	G__reftype = G__PARAP2P;
#else
	G__var_type='U';
	G__reftype = G__PARAP2P;
#endif
      }
      else if(strcmp(new_name,"long&")==0) {
#ifndef G__OLDIMPLEMENTATION2189
	G__var_type='n' + G__unsigned;
	G__tagnum = -1;
	G__typenum = -1;
	G__reftype = G__PARAREFERENCE;
#else
	G__var_type='u';
	G__reftype = G__PARAREFERENCE;
#endif
      }
      G__define_var(G__tagnum,G__typenum);
      G__var_type='p';
      G__tagnum=store_tagnum;
      G__typenum=store_typenum;
      G__decl=store_decl;
      return(0);
    }
#endif
#ifndef G__OLDIMPLEMENTATION1533
    else if(
	    'l'==G__var_type &&
	    (strcmp(new_name,"double")==0 ||
	     strcmp(new_name,"double*")==0 ||
	     strcmp(new_name,"double**")==0 ||
	     strcmp(new_name,"double&")==0)) {
      int store_tagnum = G__tagnum;
      int store_typenum = G__typenum;
      int store_decl = G__decl;
#if !defined(G__OLDIMPLEMENTATION2189)
#elif !defined(G__OLDIMPLEMENTATION1836)
      G__loadlonglong(&G__tagnum,&G__typenum,G__LONGDOUBLE);
#else /* 1836 */
      if(0==G__defined_macro("G__LONGLONG_H")) {
#ifndef G__OLDIMPLEMENTATION1153
	int store_def_struct_member = G__def_struct_member;
	G__def_struct_member = 0;
#endif
	G__decl=0;
	G__loadfile("long.dll"); /* used to switch case between .dl and .dll */
	G__decl=1;
#ifndef G__OLDIMPLEMENTATION1153
	G__def_struct_member = store_def_struct_member;
#endif
      }
#ifndef G__OLDIMPLEMENTATION1686
      G__tagnum=G__defined_tagname("G__ulonglong",2);
      G__typenum=G__search_typename("unsigned long long",'u',G__tagnum,G__PARANORMAL);
#endif
      G__tagnum=G__defined_tagname("G__longlong",2);
      G__typenum=G__search_typename("long long",'u',G__tagnum,G__PARANORMAL);
      G__tagnum=G__defined_tagname("G__longdouble",2);
      if(-1==G__tagnum) {
	G__genericerror("Error: 'long double' not ready. Go to $CINTSYSDIR/lib/longlong and run setup");
      }
      G__typenum=G__search_typename("long double",'u',G__tagnum,G__PARANORMAL);
#endif /* 1836 */
      if(strcmp(new_name,"double")==0) {
#ifndef G__OLDIMPLEMENTATION2189
	G__var_type='q';
	G__tagnum = -1;
	G__typenum = -1;
	G__reftype = G__PARANORMAL;
#else
	G__var_type='u';
	G__reftype = G__PARANORMAL;
#endif
      }
      else if(strcmp(new_name,"double*")==0) {
#ifndef G__OLDIMPLEMENTATION2189
	G__var_type='Q';
	G__tagnum = -1;
	G__typenum = -1;
	G__reftype = G__PARANORMAL;
#else
	G__var_type='U';
	G__reftype = G__PARANORMAL;
#endif
      }
      else if(strcmp(new_name,"double**")==0) {
#ifndef G__OLDIMPLEMENTATION2189
	G__var_type='Q';
	G__tagnum = -1;
	G__typenum = -1;
	G__reftype = G__PARAP2P;
#else
	G__var_type='U';
	G__reftype = G__PARAP2P;
#endif
      }
      else if(strcmp(new_name,"double&")==0) {
#ifndef G__OLDIMPLEMENTATION2189
	G__var_type='q';
	G__tagnum = -1;
	G__typenum = -1;
	G__reftype = G__PARAREFERENCE;
#else
	G__var_type='u';
	G__reftype = G__PARAREFERENCE;
#endif
      }
      G__define_var(G__tagnum,G__typenum);
      G__var_type='p';
      G__tagnum=store_tagnum;
      G__typenum=store_typenum;
      G__decl=store_decl;
      return(0);
    }
#endif
    else if(strcmp(new_name,"unsigned")==0||strcmp(new_name,"signed")==0) {
      cin=G__fgetvarname(new_name,",;=():");
      --G__var_type; /* make it unsigned */
      if(strcmp(new_name,"int*")==0) {
	G__var_type = toupper(G__var_type);
	cin=G__fgetvarname(new_name,",;=():");
      }
      else if(strcmp(new_name,"int&")==0) {
	G__var_type = toupper(G__var_type);
	cin=G__fgetvarname(new_name,",;=():");
	G__reftype=G__PARAREFERENCE;
      }
      else if(strcmp(new_name,"int")==0) {
	cin=G__fgetvarname(new_name,",;=():");
      }
    }
    else if(strcmp(new_name,"int*")==0) {
      cin=G__fgetvarname(new_name,",;=():");
      G__var_type = toupper(G__var_type);
    }
    else if(strcmp(new_name,"double*")==0) {
      cin=G__fgetvarname(new_name,",;=():");
      G__var_type='D';
    }
    else if(strcmp(new_name,"int&")==0) {
      cin=G__fgetvarname(new_name,",;=():");
#ifdef G__OLDIMPLEMENTATION1526
      G__var_type = toupper(G__var_type);
#endif
      G__reftype=G__PARAREFERENCE;
    }
    else if(strcmp(new_name,"double&")==0) {
      cin=G__fgetvarname(new_name,",;=():");
#ifdef G__OLDIMPLEMENTATION1526
      G__var_type='D';
#endif
      G__reftype=G__PARAREFERENCE;
    }

    if(isspace(cin)) {
      if(strcmp(new_name,"static")==0) {
	cin=G__fgetvarname(new_name,",;=():");
	G__static_alloc=1;
      }
    }

    if(isspace(cin)) {
      if(strcmp(new_name,"*const")==0) {
	cin=G__fgetvarname(new_name+1,",;=():");
	G__constvar |= G__PCONSTVAR;
      }
      else if(strcmp(new_name,"const")==0) {
	cin=G__fgetvarname(new_name,",;=():");
#ifndef G__OLDIMPLEMENTATION1182
	if(strcmp(new_name,"&*")==0 || strcmp(new_name,"*&")==0) {
	  G__reftype=G__PARAREFERENCE;
	  new_name[0]='*';
	  cin=G__fgetvarname(new_name+1,",;=():");
	}
	else if(strcmp(new_name,"&")==0) {
	  G__reftype=G__PARAREFERENCE;
	  cin=G__fgetvarname(new_name,",;=():");
        }
	if(strcmp(new_name,"*")==0) {
	  cin=G__fgetvarname(new_name+1,",;=():");
#ifndef G__OLDIMPLEMENTATION1846
	  if(strcmp(new_name,"*const")==0) {
	    G__constvar |= G__PCONSTVAR;
	    cin=G__fgetvarname(new_name+1,",;=():");
	  }
#endif
	}
#endif
#ifndef G__OLDIMPLEMENTATION1134
	if(isupper(G__var_type)) G__constvar |= G__PCONSTVAR;
	else                     G__constvar |= G__CONSTVAR;
#else
	G__constvar |= G__PCONSTVAR;
#endif
      }
      else if(strcmp(new_name,"const&")==0) {
	cin=G__fgetvarname(new_name,",;=():");
	G__reftype=G__PARAREFERENCE;
	G__constvar |= G__PCONSTVAR;
      }
#ifndef G__PHILIPPE22
      else if(strcmp(new_name,"*const&")==0) {
	cin=G__fgetvarname(new_name+1,",;=():");
	G__constvar |= G__PCONSTVAR;
	G__reftype=G__PARAREFERENCE;
      }
#endif
#ifndef G__OLDIMPLEMENTATION1857
      else if(strcmp(new_name,"const*&")==0) {
	new_name[0] = '*';
	cin=G__fgetvarname(new_name+1,",;=():");
	G__constvar |= G__CONSTVAR;
	G__reftype=G__PARAREFERENCE;
      }
      else if(strcmp(new_name,"const**")==0) {
	new_name[0] = '*';
	cin=G__fgetvarname(new_name+1,",;=():");
	G__constvar |= G__CONSTVAR;
	G__var_type='U';
	G__reftype = G__PARAP2P;
      }
#endif
#ifndef G__OLDIMPLEMENTATION1216
      else if(strcmp(new_name,"volatile")==0) {
	cin=G__fgetvarname(new_name,",;=():");
      }
      else if(strcmp(new_name,"*volatile")==0) {
	cin=G__fgetvarname(new_name+1,",;=():");
      }
      else if(strcmp(new_name,"**volatile")==0) {
	cin=G__fgetvarname(new_name+2,",;=():");
      }
      else if(strcmp(new_name,"***volatile")==0) {
	cin=G__fgetvarname(new_name+3,",;=():");
      }
#endif
#ifndef G__OLDIMPLEMENTATION1428
      else if(strcmp(new_name,"inline")==0) {
	cin=G__fgetvarname(new_name,",;=():");
      }
      else if(strcmp(new_name,"*inline")==0) {
	cin=G__fgetvarname(new_name+1,",;=():");
      }
      else if(strcmp(new_name,"**inline")==0) {
	cin=G__fgetvarname(new_name+2,",;=():");
      }
      else if(strcmp(new_name,"***inline")==0) {
	cin=G__fgetvarname(new_name+3,",;=():");
      }
#endif
#ifndef G__OLDIMPLEMENTATION1630
      else if(strcmp(new_name,"virtual")==0) {
	G__virtual = 1;
	cin=G__fgetvarname(new_name,",;=():");
      }
#endif
    }

    if(isspace(cin)) {
#ifndef G__OLDIMPLEMENTATION1149
      int store_len;
#endif
      if(strcmp(new_name,"operator")==0 ||
	 strcmp(new_name,"*operator")==0||
#ifndef G__PHILIPPE11
	 strcmp(new_name,"*&operator")==0||
#endif
	 strcmp(new_name,"&operator")==0) {
	/* read real name */
	cin=G__fgetstream(temp1,"(");
	/* came to
	 * type  operator  +(var1 , var2);
	 *                  ^
	 * type  int   operator + (var1 , var2);
	 *                       ^
	 */
	switch(temp1[0]) {
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
#ifndef G__OLDIMPLEMENTATION1224
	case ',':
#endif
	  sprintf(temp,"%s%s",new_name,temp1);
	  strcpy(new_name,temp);
	  break;
	case '\0':
	  cin=G__fgetstream(temp1,")");
	  if(strcmp(temp1,"")!=0 || cin!=')') {
	    G__fprinterr(G__serr,"Error: Syntax error '%s(%s%c' "
		    ,new_name,temp1,cin);
	    G__genericerror((char*)NULL);
	  }
	  cin=G__fgetstream(temp1,"(");
	  if(strcmp(temp1,"")!=0 || cin!='(') {
	    G__fprinterr(G__serr,"Error: Syntax error '%s()%s%c' "
		    ,new_name,temp1,cin);
	    G__genericerror((char*)NULL);
	  }
	  sprintf(temp,"%s()",new_name);
	  strcpy(new_name,temp);
	  break;
	default:
	  sprintf(temp,"%s %s",new_name,temp1);
	  strcpy(new_name,temp);
	  /*
	     G__genericerror(
	     "Warning: name 'operator' will be a keyword for C++"
	     ); 
	     */
	  break;
	}
	return(cin);
      } /* if(strcmp(new_name,"operator")==0) */

#ifndef G__OLDIMPLEMENTATION1149
      store_len = strlen(new_name);
#endif
      
      do {
	cin = G__fgetstream(new_name+strlen(new_name),",;=():");
	if(']'==cin) strcpy(new_name+strlen(new_name),"]");
      } while(']'==cin); 

#ifndef G__OLDIMPLEMENTATION1149
      if(store_len>1&&isalnum(new_name[store_len])&&
	 isalnum(new_name[store_len-1])) {
	if(G__dispmsg>=G__DISPWARN) {
	  G__fprinterr(G__serr,"Warning: %s  Syntax error??",new_name);
	  G__printlinenum();
	}
      }
#endif

      return(cin);
      
    } /* of isspace(cin) */
  } /* of isspace(cin) */
#ifndef G__OLDIMPLEMENTATION1957
  else if('('==cin && 0==new_name[0]) {
    /* check which case
     *  1. f(type (*p)(int))  -> do nothing here 
     *  2. f(type (*p)[4][4]) -> convert to f(type p[][4][4])  */
    fpos_t tmppos;
    int tmpline = G__ifile.line_number;;
    fgetpos(G__ifile.fp,&tmppos);
    if(G__dispsource) G__disp_mask=1;

    cin=G__fgetvarname(new_name,")");
    if('*'!=new_name[0] || 0==new_name[1]) goto escapehere;
    strcpy(temp,new_name+1);

    cin=G__fgetvarname(new_name,",;=():}");
    if('['!=new_name[0]) goto escapehere;
    if(G__dispsource) {
      G__disp_mask=0;
      G__fprinterr(G__serr,"*%s)%s",temp,new_name);
    }

    strcat(temp,"[]");
    strcat(temp,new_name);
    strcpy(new_name,temp);

    return(cin);
    
  escapehere:
    if(G__dispsource) G__disp_mask=0;
    fsetpos(G__ifile.fp,&tmppos);
    G__ifile.line_number = tmpline;
    new_name[0] = 0;
    cin = '(';
  }
#endif

  if(strncmp(new_name,"operator",8)==0 && 
     (G__isoperator(new_name[8]) || '\0'==new_name[8])) {
    if('='==cin) {
      fseek(G__ifile.fp,-1,SEEK_CUR);
      if(G__dispsource) G__disp_mask=1;
      cin=G__fgetstream(new_name+strlen(new_name),"(");
    }
    else if('('==cin && '\0'==new_name[8]) {
      cin=G__fgetstream(new_name,")");
      cin=G__fgetstream(new_name,"(");
      sprintf(new_name,"operator()");
    }
#ifndef G__OLDIMPLEMENTATION1224
    else if(','==cin && '\0'==new_name[8]) {
      cin=G__fgetstream(new_name,"(");
      sprintf(new_name,"operator,");
    }
#endif
    return(cin);
  }
  else if((strncmp(new_name,"*operator",9)==0 ||
	   strncmp(new_name,"&operator",9)==0) &&
	  (G__isoperator(new_name[9]) || '\0'==new_name[9])) {
    if('='==cin) {
      fseek(G__ifile.fp,-1,SEEK_CUR);
      if(G__dispsource) G__disp_mask=1;
      cin=G__fgetstream(new_name+strlen(new_name),"(");
    }
    else if('('==cin && '\0'==new_name[9]) {
      cin=G__fignorestream(")");
      cin=G__fignorestream("(");
      strcpy(new_name+9,"()");
    }
    return(cin);
  }
#ifndef G__PHILIPPE11
  else if((strncmp(new_name,"&*operator",10)==0 ||
	   strncmp(new_name,"*&operator",10)==0) &&
	  (G__isoperator(new_name[10]) || '\0'==new_name[10])) {
    if('='==cin) {
      fseek(G__ifile.fp,-1,SEEK_CUR);
      if(G__dispsource) G__disp_mask=1;
      cin=G__fgetstream(new_name+strlen(new_name),"(");
    }
    else if('('==cin && '\0'==new_name[10]) {
      cin=G__fignorestream(")");
      cin=G__fignorestream("(");
      strcpy(new_name+10,"()");
    }
    return(cin);
  }  
#endif

  return(cin);

}

/******************************************************************
* G__unsignedintegral()
******************************************************************/
int G__unsignedintegral(pspaceflag,piout,mparen)
int *pspaceflag;
int *piout;
int mparen;
{
  char name[G__MAXNAME];
  fpos_t pos;

  G__unsigned = -1;
  fgetpos(G__ifile.fp,&pos);

  G__fgetname(name,"");

  if(strcmp(name,"int")==0)         G__var_type='i'-1;
  else if(strcmp(name,"char")==0)   G__var_type='c'-1;
  else if(strcmp(name,"short")==0)  G__var_type='s'-1;
  else if(strcmp(name,"long")==0)   G__var_type='l'-1;
  else if(strcmp(name,"int*")==0)   G__var_type='I'-1;
  else if(strcmp(name,"char*")==0)  G__var_type='C'-1;
  else if(strcmp(name,"short*")==0) G__var_type='S'-1;
  else if(strcmp(name,"long*")==0)  G__var_type='L'-1;
  else if(strcmp(name,"int&")==0) { 
    G__var_type='i'-1;
    G__reftype=G__PARAREFERENCE;
  }
  else if(strcmp(name,"char&")==0) {
    G__var_type='c'-1;
    G__reftype=G__PARAREFERENCE;
  }
  else if(strcmp(name,"short&")==0) {
    G__var_type='s'-1;
    G__reftype=G__PARAREFERENCE;
  }
  else if(strcmp(name,"long&")==0) {
    G__var_type='l'-1;
    G__reftype=G__PARAREFERENCE;
  }
#ifndef G__OLDIMPLEMENTATION1407
  else if(strchr(name,'*')) {
    if(strncmp(name,"int*",4)==0)        G__var_type='I'-1;
    else if(strncmp(name,"char*",5)==0)  G__var_type='C'-1;
    else if(strncmp(name,"short*",6)==0) G__var_type='S'-1;
    else if(strncmp(name,"long*",5)==0)  G__var_type='L'-1;
    if(strstr(name,"******")) G__reftype = G__PARAP2P+4;
    else if(strstr(name,"*****")) G__reftype = G__PARAP2P+3;
    else if(strstr(name,"****")) G__reftype = G__PARAP2P+2;
    else if(strstr(name,"***")) G__reftype = G__PARAP2P+1;
    else if(strstr(name,"**")) G__reftype = G__PARAP2P;
  }
#endif
  else {
    G__var_type='i'-1;
    fsetpos(G__ifile.fp,&pos);
  }

  G__define_var(-1,-1);

  G__reftype=G__PARANORMAL;
  G__unsigned = 0;
  *pspaceflag = -1;
  *piout = 0;

  if(mparen==0) return(1);
  else          return(0);
}

/**************************************************************************
* G__rawvarentry()
**************************************************************************/
static struct G__var_array *G__rawvarentry(name,hash,pig15,var)
char *name;
int hash;
int *pig15;
struct G__var_array *var;
{
  int ig15=0;
  while(var) {
    for(ig15=0;ig15<var->allvar;ig15++) {
      if(hash == var->hash[ig15] && strcmp(name,var->varnamebuf[ig15])==0) {
	*pig15 = ig15;
	return(var);
      }
    }
    var = var->next;
  }
  return(var);
}


/**************************************************************************
* G__setvariablecomment()
**************************************************************************/
static int G__setvariablecomment(new_name)
char *new_name;
{
  struct G__var_array *var;
  int ig15;
  int i;
  int hash;
  char name[G__MAXNAME];
  char *p;
#ifndef G__OLDIMPLEMENTATION2192
   unsigned int j,nest,scope;
#endif

#ifdef G__FONS_COMMENT
  if('\0'==new_name[0]) return(0);

  strcpy(name,new_name);
  p=strchr(name,'[');
  if(p) *p='\0';

#ifndef G__OLDIMPLEMENTATION2192
   /* Check to see if we were passed a qualified name or name */
   for(j=0,nest=0,scope=0;j<strlen(name);++j) {
      switch(name[j]) {
        case '<': ++nest; break;
        case '>': --nest; break;
        case ':': 
           if (nest==0 && name[j+1]==':') {
              scope = j;
           }; break;
      };
   }
   
   if (scope==0) {
     /* If scope is not null, this means that we are not really inside the
	the class declaration.  This might actually be an instantiation inside
        a namespace */
#endif
  
     G__hash(name,hash,i)
     /* only interpretation. no need to check for cpplink memvar setup */
     var = G__rawvarentry(name,hash,&ig15,G__struct.memvar[G__tagdefining]);
     if(var) {
       var->comment[ig15].filenum = -1;
       var->comment[ig15].p.com = (char*)NULL;
       G__fsetcomment(&var->comment[ig15]);
     }
     else {
       G__fprinterr(G__serr,"Internal warning: %s comment can not set",new_name);
       G__printlinenum();
     }
#ifndef G__OLDIMPLEMENTATION2192
   }
#endif
#endif
   return(0);
}

#ifndef G__OLDIMPLEMENTATION1037
/******************************************************************
* G__removespacetemplate()
******************************************************************/
void G__removespacetemplate(name) 
char *name;
{
  char buf[G__LONGLINE];
  int c;
  int i=0,j=0;
  while((c=name[i])) {
    if(isspace(c)&&i>0) {
      switch(name[i-1]) {
      case ':':
      case '<':
      case ',':
	break;
      case '>':
	if('>'==name[i+1]) buf[j++] = c;
	break;
      default:
	switch(name[i+1]) {
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
  strcpy(name,buf);
}
#endif

#ifndef G__OLDIMPLEMENTATION1552
/******************************************************************
* G__initstructary(p_inc,new_name)
*
*  A string[3] = { "abc", "def", "hij" };
*  A string[]  = { "abc", "def", "hij" };
*                 ^
******************************************************************/
void G__initstructary(new_name,tagnum)
char* new_name;
int tagnum;
{
  char *index;
  int p_inc;
  int cin;
  char buf[G__ONELINE];
  G__value reg;
  long store_struct_offset = G__store_struct_offset;
  long store_globalvarpointer = G__globalvarpointer;
  long adr;
  long len;
  int known;
  int i;

#ifdef G__ASM
  G__abortbytecode();
#endif

  /* count number of array elements if needed */
  index = strchr(new_name,'[');
  if(*(index+1)==']') {
    fpos_t store_pos;
    int store_line = G__ifile.line_number; 
    fgetpos(G__ifile.fp,&store_pos);

    p_inc=0;
    do {
      cin = G__fgetstream(buf,",}");
      ++p_inc;
    } while(cin!='}'); 

    strcpy(buf,index+1);
    sprintf(index+1,"%d",p_inc);
    strcat(new_name,buf);

    G__ifile.line_number = store_line; 
    fsetpos(G__ifile.fp,&store_pos);
  }
  else {
    p_inc=G__getarrayindex(index);
  }

  /* allocate memory */
  reg = G__null;
  G__decl_obj=2;
  adr=G__int(G__letvariable(new_name,reg,&G__global,G__p_local));
  G__decl_obj=0;

  /* read and initalize each element */
  strcpy(buf,G__struct.name[tagnum]);
  strcat(buf,"(");
  len = strlen(buf);
  i=0;
  do {
    cin = G__fgetstream(buf+len,",}");
    strcat(buf,")");
    if(G__CPPLINK!=G__struct.iscpplink[tagnum]) {
      G__store_struct_offset = adr + i*G__struct.size[tagnum];
    }
    else {
      G__globalvarpointer = adr + i*G__struct.size[tagnum];
    }
    reg=G__getfunction(buf,&known,G__CALLCONSTRUCTOR);
    ++i;
  } while(cin!='}'); 

  /* post processing */
  G__store_struct_offset = store_struct_offset;
  G__globalvarpointer = store_globalvarpointer;

}
#endif

/******************************************************************
* void G__define_var(tagnum,typenum)
*
* Called by
*   G__DEFVAR       macro
*   G__DEFREFVAR    macro
*   G__define_struct()
*   G__define_type()
*   G__define_type()
*
*  Declaration of variable, function or ANSI function header
*
*  variable:   type  varname1, varname2=initval ;
*                  ^
*  function:   type  funcname(param decl) { body }
*                  ^
*  ANSI function header: funcname(  type para1, type para2,...)
*                                 ^     or     ^
******************************************************************/
void G__define_var(tagnum,typenum)
int tagnum,typenum;      /* overrides global variables */
{
  G__value reg;
  char var_type;
  int cin='\0';
  int store_decl;

  int largestep=0;
  int store_tagnum,store_typenum;

  int store_def_struct_member;
  int store_def_tagnum;

  int i,p_inc;
  char *index;

  int initary=0;

  int flag;
  int  known;
  long  store_struct_offset; /* used to be int */
  int  store_prerun;
#ifndef G__OLDIMPLEMENTATION713
  int  store_debug=0,store_step=0;
#else
  int  store_debug,store_step;
#endif
  char temp1[G__LONGLINE];

  char new_name[G__LONGLINE],temp[G__LONGLINE];
  int staticclassobject=0;

  int store_var_type;
#ifndef G__OLDIMPLEMENTATION713
  int store_tagnum_default=0;
  int store_def_struct_member_default=0;
  int store_exec_memberfunc=0;
  int store_memberfunc_tagnum=0;
#else
  int store_tagnum_default;
  int store_def_struct_member_default;
  int store_exec_memberfunc;
  int store_memberfunc_tagnum;
#endif
  int store_constvar;
  int store_static_alloc;
  int store_tagdefining;
  fpos_t store_fpos;
  int store_line;
  int store_static_alloc2;
  static int padn=0;
  static int bitfieldwarn=0;

  store_static_alloc2=G__static_alloc;

  /* new_name is initialized in G__get_newname(). So following line is not
   * necessary. Just to avoid purify error message. */
  new_name[0] = '\0';

  /**********************************************************
  * handling of tagnum and typenum may be able to refine.
  **********************************************************/
  store_tagnum = G__tagnum;
  store_typenum = G__typenum;
  G__tagnum = tagnum;
  G__typenum = typenum;

  store_decl=G__decl;
  G__decl=1;
#ifdef G__OLDIMPLEMENTATION1688
  G__unsigned=0; /* this is now reset in the G__exec_statement() */
#endif


  /*
   * type  var1 , var2 ;
   *      ^
   *   or
   * type  int var1 , var2;
   *      ^
   * read variable name or 'int' identifier
   */
  cin=G__get_newname(new_name);
#ifndef G__OLDIMPLEMENTATION1688
  G__unsigned=0; /* this is now reset in the G__exec_statement() */
#endif
#ifndef G__OLDIMPLEMENTATION883
  if(0==cin) {
    G__decl=store_decl;
    G__constvar=0;
    G__tagnum = store_tagnum;
    G__typenum = store_typenum;
    G__reftype=G__PARANORMAL;
    G__static_alloc=store_static_alloc2;
#ifndef G__OLDIMPLEMENTATION1119
    G__dynconst=0;
#endif
#ifndef G__OLDIMPLEMENTATION1322
    G__globalvarpointer = G__PVOID;
#endif
    return; /* long long handling */
  }
#endif
  var_type = G__var_type;

  /* Now came to
   * type  var1  , var2 ;
   *              ^
   *   or
   * type  var1  = initval , var2 ;
   *              ^
   *   or
   * type  var1   : N , var2 ;
   *               ^
   *   or
   * type  int var1  , var2;
   *                  ^
   */

  while(1) {

    if('&'==new_name[0]) {
      G__reftype=G__PARAREFERENCE;
      strcpy(temp,new_name+1);
      strcpy(new_name,temp);
    }
    else if('*'==new_name[0] && '&'==new_name[1]) {
      G__reftype=G__PARAREFERENCE;
      sprintf(temp,"*%s",new_name+2);
      strcpy(new_name,temp);
    }

    /************************************************************
     * if ANSI function parameter 
     *   funcname(type var1  , type var2,...)
     *                      ^    or         ^
     *   funcname(type var1= 5 , type var2,...)
     *                      ^    or         ^
     *  return one by one
     ***********************************************************/
    if(G__ansiheader) {

#ifndef G__OLDIMPLEMENTATION1472
      char *pxx = strstr(new_name,"...");
      if(pxx) *pxx=0;
#endif

#ifndef G__OLDIMPLEMENTATION880
      if(G__asm_wholefunction&&G__asm_noverflow) {
	char *pwf=strchr(new_name,'[');
	if(pwf) {
	  char *pwf2=strchr(pwf+1,'[');
	  if(pwf2) G__abortbytecode(); /* f(T a[][10]) */
	  else if(']' != *(++pwf)) {   /* f(T a[10]) -> f(T a[]) */
	    *(pwf++) = ']';
	    *pwf = 0;
	  }
	}
      }
#endif

      if(cin=='(') {
	if(new_name[0]=='\0' || strcmp(new_name,"*")==0) {
	  /* pointer of function
	   *   type ( *funcpointer[n])( type var1,.....)
	   *         ^
	   */
	  G__readpointer2function(new_name,&var_type);
	  /* read to ,) */
	  cin=G__fignorestream(",)=");
	}
      }

      /**********************************************
       * If there is a default parameter, read it
       **********************************************/
      if(cin=='=') {
	cin=G__fgetstream(temp,",)");
	store_var_type = G__var_type;
	G__var_type = 'p';
	if(-1!=G__def_tagnum) {
	  store_tagnum_default = G__tagnum;
	  G__tagnum = G__def_tagnum;
	  store_def_struct_member_default=G__def_struct_member;
	  store_exec_memberfunc=G__exec_memberfunc;
	  store_memberfunc_tagnum = G__memberfunc_tagnum;
	  G__memberfunc_tagnum = G__tagnum;
	  G__exec_memberfunc=1;
	  G__def_struct_member=0;
	}
	else if(G__exec_memberfunc) {
	  store_tagnum_default = G__tagnum;
	  G__tagnum = store_tagnum;
	  store_def_struct_member_default=G__def_struct_member;
	  store_exec_memberfunc=G__exec_memberfunc;
	  store_memberfunc_tagnum = G__memberfunc_tagnum;
	  G__memberfunc_tagnum = G__tagnum;
	  G__exec_memberfunc=1;
	  G__def_struct_member=0;
	}
	else store_exec_memberfunc=0;
	strcpy(G__def_parameter,temp);
	G__default_parameter = G__getexpr(temp);
	if(G__default_parameter.type==G__DEFAULT_FUNCCALL) {
	  /* f(type a=f2()); experimental */
	  G__default_parameter.ref=G__int(G__strip_quotation(temp));
	}
	if(-1!=G__def_tagnum || store_exec_memberfunc) {
	  G__tagnum = store_tagnum_default;
	  G__exec_memberfunc=store_exec_memberfunc;
	  G__def_struct_member=store_def_struct_member_default;
	  G__memberfunc_tagnum = store_memberfunc_tagnum;
	}
	G__var_type = store_var_type;
#ifdef G__OLDOMPLEMENTATION183
	if(G__reftype /* == G__PARAREFERENCE */) {
	  G__fprinterr(G__serr,
		"Error: Can't use default parameter for reference type %s FILE:%s LINE:%d\n"
		,new_name,G__ifile.name,G__ifile.line_number);
	}
#endif
      }
      else {
	temp[0]='\0';
      }
 
      if(G__reftype 
#ifndef G__OLDIMPLEMENTATION921
	 == G__PARAREFERENCE 
#endif
         ) {
	G__globalvarpointer = G__ansipara.ref;
	reg=G__null;
	if(G__globalvarpointer==(long)NULL && 'u'==G__ansipara.type &&
	   (G__prerun==0 && 0==G__no_exec_compile)) {
	  G__referencetypeerror(new_name);
	}
      }
      else {
	/**********************************************
	 * set default value if parameter is omitted
	 **********************************************/
	if(G__ansipara.type=='\0') {
	  /* this case is not needed after changing default parameter
	   * handling */
	  store_var_type = G__var_type;
	  G__var_type = 'p';
	  if(-1!=G__def_tagnum) {
	    store_tagnum_default = G__tagnum;
	    G__tagnum = G__def_tagnum;
	    store_def_struct_member_default=G__def_struct_member;
	    store_exec_memberfunc=G__exec_memberfunc;
	    store_memberfunc_tagnum=G__memberfunc_tagnum;
	    G__memberfunc_tagnum = G__tagnum;
	    G__exec_memberfunc=1;
	    G__def_struct_member=0;
	  }
	  else if(G__exec_memberfunc) {
	    store_tagnum_default = G__tagnum;
	    G__tagnum = store_tagnum;
	    store_def_struct_member_default=G__def_struct_member;
	    store_exec_memberfunc=G__exec_memberfunc;
	    store_memberfunc_tagnum=G__memberfunc_tagnum;
	    G__memberfunc_tagnum = G__tagnum;
	    G__exec_memberfunc=1;
	    G__def_struct_member=0;
	  }
	  else {
	    store_exec_memberfunc=0;
	  }
	  reg = G__getexpr(temp);
	  if(-1!=G__def_tagnum || store_exec_memberfunc) {
	    G__tagnum = store_tagnum_default;
	    G__exec_memberfunc=store_exec_memberfunc;
	    G__def_struct_member=store_def_struct_member_default;
	    G__memberfunc_tagnum = store_memberfunc_tagnum;
	  }
	  G__var_type = store_var_type;
	}
	else {
	  reg = G__ansipara;
	}
      }

      G__var_type = var_type ;
      
      /**************************************************
      * initialization of formal parameter 
      *C++: G__COPYCONSTRUCTOR
      * Default and user specified copy constructor is
      * switched in G__letvariable()
      **************************************************/
      if('u'==G__var_type && G__PARANORMAL==G__reftype &&
	 '*'!=new_name[0] && !strstr(new_name,"[]") ) {
	G__ansiheader=0;
	if(G__CPPLINK==G__struct.iscpplink[tagnum]) {
#ifndef G__OLDIMPLEMENTATION1110
	  char tttt[G__ONELINE];
	  G__valuemonitor(reg,tttt);
	  sprintf(temp1,"%s(%s)",G__struct.name[tagnum],tttt);
#else
	  if(reg.obj.i<0)
	    sprintf(temp1,"%s((%s)(%ld))",G__struct.name[tagnum]
		    ,G__struct.name[tagnum],reg.obj.i);
	  else
	    sprintf(temp1,"%s((%s)%ld)",G__struct.name[tagnum]
		    ,G__struct.name[tagnum],reg.obj.i);
#endif
#ifndef G__OLDIMPLEMENTATION979
	  if(-1!=G__struct.parent_tagnum[tagnum]) {
            int store_exec_memberfunc=G__exec_memberfunc;
            int store_memberfunc_tagnum=G__memberfunc_tagnum;
            G__exec_memberfunc=1;
            G__memberfunc_tagnum=G__struct.parent_tagnum[tagnum];
	    reg = G__getfunction(temp1,&known,G__CALLCONSTRUCTOR);
            G__exec_memberfunc=store_exec_memberfunc;
            G__memberfunc_tagnum=store_memberfunc_tagnum;
	  }
          else {
	    reg = G__getfunction(temp1,&known,G__CALLCONSTRUCTOR);
          }
#else
	  reg = G__getfunction(temp1,&known,G__CALLCONSTRUCTOR);
#endif
	  G__globalvarpointer = G__int(reg);
	  G__cppconstruct = 1;
	  G__letvariable(new_name,G__null,&G__global,G__p_local);
	  G__cppconstruct = 0;
	  G__globalvarpointer = G__PVOID;
	}
	else {
	  /* create object */
	  G__letvariable(new_name,G__null,&G__global,G__p_local);
	  /* call copy constructor G__decl=1 with argment reg */
	  G__letvariable(new_name,reg,&G__global,G__p_local);
	}
      }
      else {
	G__letvariable(new_name,reg,&G__global,G__p_local);
      }

      G__ansiheader=1;

      G__globalvarpointer = G__PVOID;

#ifndef G__OLDIMPLEMENTATION989
#ifdef G__ASM
      if(0==new_name[0]) {
#ifdef G__ASM_DBG
        if(G__asm_dbg) G__fprinterr(G__serr,"%3x: POP\n",G__asm_cp);
#endif
	G__asm_inst[G__asm_cp] = G__POP;
	G__inc_cp_asm(1,0);
#endif
#endif
      }

      /* end of ANSI parameter header if cin==')'
       *   funcname(type var1 , type var2,...)
       *                                      ^
       */
      if(cin==')') G__ansiheader=0;
      G__decl=store_decl;
      G__constvar=0;
      G__tagnum = store_tagnum;
      G__typenum = store_typenum;
      G__reftype=G__PARANORMAL;
      G__static_alloc=store_static_alloc2;
#ifndef G__OLDIMPLEMENTATION1119
      G__dynconst=0;
#endif
#ifndef G__OLDIMPLEMENTATION1322
      G__globalvarpointer = G__PVOID;
#endif
      return;
    }


    /*************************************************************
     * function  if cin=='('
     *   type funcname( type var1,.....)
     *                 ^
     *            or
     *   type ( *funcpointer)(type var1,...)
     *         ^
     * This part should be called only at pre-run. (Used to be true)
     * C++:
     *   type obj(const,const);
     * is used to give constant parameter to constructor.
     ************************************************************/
    if(cin=='(') {

      if(new_name[0]=='\0' || strcmp(new_name,"*")==0) {
	/* pointer of function
	 *   type ( *funcpointer[n])( type var1,.....)
	 *         ^
	 */
	switch(G__readpointer2function(new_name,&var_type)) {
	case G__POINTER2FUNC:
	  break;
	case G__FUNCRETURNP2F:
	  G__isfuncreturnp2f=1;
	  goto define_function;
	case G__POINTER2MEMFUNC:
	  break;
	case G__CONSTRUCTORFUNC:
	  if(-1!=G__tagnum) {
	    cin='(';
	    strcpy(new_name,G__struct.name[G__tagnum]);
	    G__var_type = 'i';
	    /*
	    G__tagnum = -1;
	    G__typenum = -1;
	    */
	    goto define_function;
	  }
	}
	
	/* initialization of pointer to function
	 * CAUTION: Now, I don't do this.
	 *   G__var_type = 'q';
	 * Thus, type of function pointer is declared type
	 */
	/* G__letvariable(new_name,reg,&G__global,G__p_local); */
	
	/* read to =,; */
#ifndef G__OLDIMPLEMENTATION796
	cin=G__fignorestream("=,;}");
#else
	cin=G__fignorestream(",;}");
#endif
	G__constvar=0;
	G__reftype=G__PARANORMAL;
      }
      else {
	/* function definition
	 *   type funcname( type var1,.....)
	 *                 ^
	 * or C++ constructor
	 *   type varname( const,const);
	 *                ^                       */
	/***************************************************
	 * distinguish constructor or function definition
	 ***************************************************/
	
      define_function:
	
	/* read next non space char, and rewind */
	cin=G__fgetspace();
	fseek(G__ifile.fp,-1,SEEK_CUR);
	if(cin=='\n' /* ||cin=='\r' */) --G__ifile.line_number;
	if(G__dispsource) G__disp_mask=1;
	
	/* if defining class member, it must not be  constructor call 
	 * and if cin is not digit, not quotation and not '.'  this is 
	 * a funciton definition */
	if(G__def_struct_member!=0  || 	
	   ((!isdigit(cin))&&cin!='"'&&cin!='\''&&cin!='.'&&cin!='-'&&
#ifndef G__OLDIMPLEMENTATION990
	    cin!='+'&&
#endif
	    cin!='*'&&cin!='&')) {
	  
	  /* It is clear that above check is not sufficient to distinguish
	   * class object instantiation and function header. Following
	   * code is added to make it fully compliant to ANSI C++ */
	  fgetpos(G__ifile.fp,&store_fpos);
	  store_line = G__ifile.line_number;
	  if(G__dispsource) G__disp_mask=1000;
	  cin = G__fgetname(temp,",)*&<=");
#ifndef G__PHILIPPE8
          if (strlen(temp) && isspace(cin)) {
            /* There was an argument and the parsing was stopped by a white
             * space rather than on of ",)*&<=", it is possible that 
             * we have a namespace followed by '::' in which case we have
             * to grab more before stopping! */
            int namespace_tagnum;
            char more[G__LONGLINE];
   
            namespace_tagnum = G__defined_tagname(temp,2);
            while ( ( ( (namespace_tagnum!=-1)
                        && (G__struct.type[namespace_tagnum]=='n') )
                      || (strcmp("std",temp)==0)
                      || (temp[strlen(temp)-1]==':') )
                    && isspace(cin) ) {
              cin = G__fgetname(more,",)*&<=");
              strcat(temp,more);
              namespace_tagnum = G__defined_tagname(temp,2);
            }
          }
#endif         
	  fsetpos(G__ifile.fp,&store_fpos);
	  if(G__dispsource) G__disp_mask=1;
	  G__ifile.line_number = store_line;

	  if((!G__iscpp)||'\0'==temp[0]|| 
	     -1==tagnum || /* this is a problem for 'int f(A* b);' */
	     G__istypename(temp)||('\0'==temp[0]&&')'==cin)
#ifndef G__OLDIMPLEMENTATION690
	     || 0==strncmp(new_name,"operator",8)
	     || ('<'==cin&&G__defined_templateclass(temp))
#endif
	     ) {
	    

	    G__var_type = var_type;
	    /* function definition
	     *   type funcname( type var1,.....)
	     *                  ^                */
	    sprintf(temp,"%s(",new_name);
	    G__make_ifunctable(temp);
	    G__isfuncreturnp2f=0; /* this is set above in this function */
	    
	    /* body of the function is skipped all 
	     * the way
	     *   type funcname(type var1,..) {....}
	     *                                     ^     */
	    G__decl=store_decl;
	    G__constvar=0;
	    G__tagnum = store_tagnum;
	    G__typenum = store_typenum;
	    G__reftype=G__PARANORMAL;
	    G__static_alloc=store_static_alloc2;
#ifndef G__OLDIMPLEMENTATION1119
	    G__dynconst=0;
#endif
#ifndef G__OLDIMPLEMENTATION1322
	    G__globalvarpointer = G__PVOID;
#endif
	    return;
	  }
	  G__var_type = var_type;
	}
	
	
	/* If didn't meet above conditions, this is a 
	 * constructor call */
	
	/* C++ constructor
	 *   type varname( const,const);
	 *                 ^            */
	
	/* read parameter list and build command string */
#ifndef G__OLDIMPLEMENTATION1127
	cin = G__fgetstream_newtemplate(temp,")");
#else
	cin = G__fgetstream(temp,")");
#endif

#ifndef G__OLDIMPLEMENTATION1049
	if('*'==new_name[0]&&var_type!='c'&&'"'==temp[0]) {
	  G__genericerror("Error: illegal pointer initialization");
	}
#endif

#ifndef G__OLDIMPLEMENTATION927
	if(G__static_alloc&&0==G__prerun) {
	  if(';'!=cin&&','!=cin) cin = G__fignorestream(",;");
#ifndef G__OLDIMPLEMENTATION1624
	  if('{'==cin) { /* don't know if this part is needed */
	    while('}'!=cin) cin = G__fignorestream(";,");
	  }
#endif
	  G__var_type = var_type;
	  G__letvariable(new_name,reg,&G__global,G__p_local);
	  goto readnext;
	}
#endif

#ifndef G__OLDIMPLEMENTATION992
	if(-1==G__tagnum||'u'!=var_type||'*'==new_name[0]) {
#else
	if(-1==G__tagnum) {
#endif
#ifndef G__OLDIMPLEMENTATION838
	  if(tolower(G__var_type)!='c' && strchr(temp,',')) {
	    reg = G__null;
	    G__genericerror("Error: Syntax error");
	  }
	  else {
	    reg = G__getexpr(temp);
	  }
#else
	  reg = G__getexpr(temp);
#endif
	  cin = G__fignorestream(",;");
#ifndef G__OLDIMPLEMENTATION1623
	  if(G__PARAREFERENCE==G__reftype && 0==G__asm_wholefunction) {
	    if(0==reg.ref) {
	      G__fprinterr(G__serr
			   ,"Error: reference type %s with no initialization "
			   ,new_name);
	      G__genericerror((char*)NULL);
	    }
	    G__globalvarpointer = reg.ref;
	  }	
#endif
	  goto create_body;
	}
	sprintf(temp1,"%s(%s)",G__struct.name[G__tagnum],temp);
	
	/* store flags */
	store_prerun = G__prerun;
	G__prerun = 0;
	if(store_prerun) {
	  store_debug = G__debug;
	  store_step = G__step;
	  G__debug = G__debugtrace;
	  G__step = G__steptrace;
	  G__setdebugcond();
	}
	else {
	  if(G__breaksignal) {
	    G__break=0;
	    G__setdebugcond();
	    if(G__pause()==3) {
	      if(G__return==G__RETURN_NON) {
		G__step=0;
		G__setdebugcond();
		largestep=1;
	      }
	    }
	    if(G__return>G__RETURN_NORMAL) {
	      G__decl=store_decl;
	      G__constvar=0;
	      G__tagnum = store_tagnum;
	      G__typenum = store_typenum;
	      G__reftype=G__PARANORMAL;
	      G__static_alloc=store_static_alloc2;
#ifndef G__OLDIMPLEMENTATION1119
	      G__dynconst=0;
#endif
#ifndef G__OLDIMPLEMENTATION1322
	      G__globalvarpointer = G__PVOID;
#endif
	      return;
	    }
	  }
	}
	
	
	/* skip until , or ; */
	cin = G__fignorestream(",;");
	/*   type varname( const,const) , ;
	 *                               ^
	 */
	

	/* allocate memory area */
	G__var_type = var_type;
	
	store_struct_offset = G__store_struct_offset ;
	if(G__CPPLINK!=G__struct.iscpplink[tagnum]) {
	  G__prerun=store_prerun;
	  G__store_struct_offset=G__int(G__letvariable(new_name
						       ,G__null
						       ,&G__global
						       ,G__p_local));
	  if(G__return>G__RETURN_NORMAL) {
	    G__decl=store_decl;
	    G__constvar=0;
	    G__tagnum = store_tagnum;
	    G__typenum = store_typenum;
	    G__reftype=G__PARANORMAL;
	    G__static_alloc=store_static_alloc2;
#ifndef G__OLDIMPLEMENTATION1119
	    G__dynconst=0;
#endif
#ifndef G__OLDIMPLEMENTATION1322
	    G__globalvarpointer = G__PVOID;
#endif
	    return;
	  }
	  G__prerun=0;
#ifndef G__OLDIMPLEMENTATION1073
	  if(0==G__store_struct_offset &&
	     G__asm_wholefunction && G__asm_noverflow) {
	    G__store_struct_offset = G__PVOID;
	  }
#endif
	}
	else {
	  G__store_struct_offset = G__PVOID;
	}
	
	if(G__dispsource) {
	  G__fprinterr(G__serr,
		  "\n!!!Calling constructor 0x%lx.%s for declaration of %s"
		  ,G__store_struct_offset,temp1,new_name);
	}
	
	
#define G__OLDIMPLEMENTATION1306
#ifndef G__OLDIMPLEMENTATION1306
	call_constructor:
#endif
	/* call constructor, error if no constructor */
	G__decl = 0;
	store_constvar=G__constvar;
	store_static_alloc=G__static_alloc;
	G__constvar=0;
	G__static_alloc=0;
	if(G__CPPLINK==G__struct.iscpplink[tagnum]) {
	  /* These has to be stored because G__getfunction can call bytecode
           * compiler */
          int bc_tagnum=G__tagnum; 
          int bc_typenum=G__typenum;
	  reg = G__getfunction(temp1,&known,G__CALLCONSTRUCTOR);
          G__tagnum=bc_tagnum;
          G__typenum=bc_typenum;
          G__var_type=var_type;
	  G__globalvarpointer=G__int(reg);
#ifndef G__OLDIMPLEMENTATION927
	  G__static_alloc = store_static_alloc;
	  G__prerun = store_prerun;
#endif
	  G__cppconstruct = 1;
#ifndef G__OLDIMPLEMENTATION991
	  if(G__globalvarpointer||G__no_exec_compile) 
#else
	  if(G__globalvarpointer) 
#endif
	  {
	    int store_constvar2 = G__constvar;
	    G__constvar=store_constvar;
	    G__letvariable(new_name,G__null,&G__global,G__p_local);
	    G__constvar=store_constvar2;
	  }
#ifndef G__OLDIMPLEMENTATION831
	  else if(G__asm_wholefunction) {
	    G__abortbytecode();
	    G__asm_wholefunc_default_cp=0;
	    G__no_exec=1;
	    G__return=G__RETURN_NORMAL;
	  }
#endif
	  G__cppconstruct = 0;
	  G__globalvarpointer=G__PVOID;
#ifndef G__OLDIMPLEMENTATION1073
	  if(G__asm_wholefunction&&G__no_exec_compile) {
#ifdef G__ASM_DBG
	    if(G__asm_dbg) G__fprinterr(G__serr,"%3x: SETGVP -1\n",G__asm_cp);
#endif
	    G__asm_inst[G__asm_cp]=G__SETGVP;
	    G__asm_inst[G__asm_cp+1] = -1;
	    G__inc_cp_asm(2,0);
	  }
#endif
	}
	else {
	  if(G__store_struct_offset) {
	    G__getfunction(temp1,&known,G__CALLCONSTRUCTOR);
#ifndef G__OLDIMPLEMENTATION1073
	    if(G__asm_wholefunction && G__asm_noverflow) {
#ifdef G__ASM_DBG
	      if(G__asm_dbg) G__fprinterr(G__serr,"%3x: POPSTROS\n",G__asm_cp);
#endif
	      G__asm_inst[G__asm_cp]=G__POPSTROS;
	      G__inc_cp_asm(1,0);
	    }
#endif
	  }
	  else 
#ifndef G__OLDIMPLEMENTATION510_TEMP
	    /* tempolary solution, later this must be deleted */
	    if(G__ASM_FUNC_NOP==G__asm_wholefunction||G__asm_noverflow)
#endif
	  {
#ifndef G__OLDIMPLEMENTATION1164
	    if(0==G__xrefflag) {
	      G__fprinterr(G__serr,
		      "Error: %s not allocated(1), maybe duplicate declaration "
		      ,new_name);
	    }
#else
	    G__fprinterr(G__serr,
		    "Error: %s not allocated(1), maybe duplicate declaration "
		    ,new_name);
#endif
	    G__genericerror((char*)NULL);
	  }
	}
	G__constvar=store_constvar;
	G__static_alloc=store_static_alloc;
	G__decl = 1;
	if(G__return>G__RETURN_NORMAL) {
	  G__decl=store_decl;
	  G__constvar=0;
	  G__tagnum = store_tagnum;
	  G__typenum = store_typenum;
	  G__reftype=G__PARANORMAL;
	  G__static_alloc=store_static_alloc2;
#ifndef G__OLDIMPLEMENTATION1119
	  G__dynconst=0;
#endif
#ifndef G__OLDIMPLEMENTATION1322
	  G__globalvarpointer = G__PVOID;
#endif
	  return;
	}
	
	if(largestep) {
	  G__step=1;
	  G__setdebugcond();
	  largestep=0;
	}
	
	/* restore flags */
	if(store_prerun) {
	  G__debug = store_debug;
	  G__step = store_step;
	  G__setdebugcond();
	}
	G__prerun = store_prerun;
	G__store_struct_offset = store_struct_offset;
	
	/* to skip following condition */
	new_name[0] = '\0';
	
	
      }
    }


    /**********************************************************
     * if cin==':'  ignore bit-field declaration
     *   unsigned int  var1  :  2  ;
     *                        ^
     * or
     *   returntype X::func()
     *                 ^
     *********************************************************/
    if(cin==':') {

      cin = G__fgetc();
      /* memberfunction definition
       *   type X::func()
       *          ^
       */
      if(cin==':') {
	store_def_struct_member = G__def_struct_member;
	G__def_struct_member = 1;
	store_def_tagnum = G__def_tagnum;
	store_tagdefining = G__tagdefining;
	i=0;
	while('*'==new_name[i]) ++i;
#ifndef G__OLDIMPLEMENTATION1166
	if(i) {
	  var_type = toupper(var_type);
	  /* if(i>1) G__reftype = i+1;  not needed */
	}
#endif
#ifndef G__OLDIMPLEMENTATION1037
	if(strchr(new_name+i,'<')) G__removespacetemplate(new_name+i);
#endif
	do {
	  G__def_tagnum = G__defined_tagname(new_name+i,0) ;
#ifndef G__PHILIPPE9
          /* protect against a non defined tagname */
          if (G__def_tagnum<0) {
            /* Hopefully restore all values! */
            G__decl=store_decl;
            G__constvar=0;
            G__tagnum = store_tagnum;
            G__typenum = store_typenum;
            G__reftype=G__PARANORMAL;
            G__static_alloc=store_static_alloc2;
#ifndef G__OLDIMPLEMENTATION1119
            G__dynconst=0;
#endif
#ifndef G__OLDIMPLEMENTATION1322
            G__globalvarpointer = G__PVOID;
#endif
#ifndef G__PHILIPPE23
	    G__def_struct_member = store_def_struct_member;
#endif
            return;
          }
#endif /* G__PHILIPPE9 */
	  G__tagdefining  = G__def_tagnum;
	  cin = G__fgetstream(new_name+i,"(=;:");
	} while(':'==cin && EOF!=(cin=G__fgetc())) ;
	temp[0]='\0';
	switch(cin) {
#ifndef G__OLDIMPLEMENTATION1306
	case ';':
	  {
	    int store_var_typexxx = G__var_type;
	    sprintf(temp,"%s::%s"
		    ,G__fulltagname(G__tagdefining,1),new_name+i);
	    G__var_type = 'p';
	    G__store_struct_offset = G__int(G__getitem(temp));
	    G__def_struct_member = 0;
	    G__var_type = store_var_typexxx;
	    G__prerun = 0;
	    sprintf(temp1,"%s()",G__fulltagname(G__tagnum,1));
	    goto call_constructor;
	  }
#endif
	case '=':
	  if(strncmp(new_name+i,"operator",8)==0) {
	    cin=G__fgetstream(new_name+strlen(new_name)+1,"(");
	    new_name[strlen(new_name)] = '=';
	    break;
	  }
#ifdef G__OLDIMPLEMENTATION1306
	case ';':
#endif
          /* PHILIPPE17: the following is fixed in 1306! */
	  /* static class object member must call constructor 
	   * TO BE IMPLEMENTED */
#ifndef G__OLDIMPLEMENTATION1296
	  sprintf(temp,"%s::%s",G__fulltagname(G__def_tagnum,1),new_name+i);
#else
	  sprintf(temp,"%s::%s",G__struct.name[G__def_tagnum],new_name+i);
#endif
	  strcpy(new_name,temp);
	  if('u'!=var_type||G__reftype) var_type='p';
	  else staticclassobject=1;
	  G__def_struct_member = store_def_struct_member;
	  G__tagnum= -1; /*do this to pass letvariable scopeoperator()*/
	  G__def_tagnum = store_def_tagnum;
	  G__tagdefining  = store_tagdefining;
	  continue; /* big while(1) loop */
	  /* If neither case, handle as member function definition 
	   * It is possible that this is initialization of class object as 
	   * static member, like 'type X::obj(1,2)' . This syntax is not 
	   * handled correctly. */
#ifndef G__OLDIMPLEMENTATION1306
	case '(':
	  {
	    fpos_t xxpos;
	    int xxlinenum = G__ifile.line_number;
	    fgetpos(G__ifile.fp,&xxpos);
	    /* chck if argument is type or value */
	    G__disp_mask = 1;
	    G__fgetname(temp,",)*&");
	    G__disp_mask = 0;
	    G__ifile.line_number = xxlinenum;
	    fsetpos(G__ifile.fp,&xxpos);
	    if(!G__istypename(temp) && -1!=G__tagdefining && -1!=G__tagnum) {
	      int store_var_typexxx = G__var_type;
	      sprintf(temp,"%s::%s"
		      ,G__fulltagname(G__tagdefining,1),new_name+i);
	      G__var_type = 'p';
	      G__store_struct_offset = G__int(G__getitem(temp));
	      G__def_struct_member = 0;
	      G__var_type = store_var_typexxx;
	      G__prerun = 0;
	      cin = G__fgetstream_newtemplate(temp,")");
	      cin = G__fignorestream(",;");
	      sprintf(temp1,"%s(%s)",G__fulltagname(G__tagnum,1),temp);
	      goto call_constructor;
	    }
	  }
	  break;
#endif
	}
	if(strcmp(new_name+i,"operator")==0) {
	  sprintf(temp,"%s()(",new_name);
	  cin=G__fignorestream(")");
	  cin=G__fignorestream("(");
	}
	else {
	  sprintf(temp,"%s(",new_name);
	}
	G__make_ifunctable(temp);

	G__def_struct_member = store_def_struct_member;
	G__def_tagnum = store_def_tagnum;
	G__decl=store_decl;
	G__constvar=0;
	G__tagnum = store_tagnum;
	G__typenum = store_typenum;
	G__reftype=G__PARANORMAL;
	G__static_alloc=store_static_alloc2;
#ifndef G__OLDIMPLEMENTATION1106
	G__tagdefining = store_tagdefining; /* FIX */
#endif
#ifndef G__OLDIMPLEMENTATION1119
	G__dynconst=0;
#endif
#ifndef G__OLDIMPLEMENTATION1322
	G__globalvarpointer = G__PVOID;
#endif
	return;
      }
      else {
	fseek(G__ifile.fp,-1,SEEK_CUR);
	if(cin=='\n' /* ||cin=='\r' */ ) --G__ifile.line_number;
	if(G__dispsource) G__disp_mask=1;
      }


      if(G__globalcomp!=G__NOLINK) {
#ifndef G__OLDIMPLEMENTATION894
	if(0==bitfieldwarn) {
#endif
	  if(G__dispmsg>=G__DISPNOTE) {
	    G__fprinterr(G__serr,"Note: Bit-field not accessible from interpreter");
	    G__printlinenum();
	  }
#ifndef G__OLDIMPLEMENTATION894
	  bitfieldwarn=1;
	}
#endif
	cin=G__fgetstream(temp,",;=}");
	sprintf(new_name,"%s : %s",new_name,temp);
	G__bitfield=1;
      }
      else {
	cin=G__fgetstream(temp,",;=}");
	G__bitfield=atoi(temp);
	if(0==G__bitfield) G__bitfield = -1;
	if('\0'==new_name[0]) {
	  sprintf(new_name,"G__pad%x",padn++);
	}
      }
    }


    /***************************************************************
     * if cin=='=' read initial value
     *  type var1 = initval , ...
     *             ^
     *  set reg = G__getexpr("initval");
     ***************************************************************/
    temp[0] = '\0';

    if(cin=='=') {
#ifndef G__OLDIMPLEMENTATION737
      int store_tagnumB=G__tagnum;
      G__tagnum = G__get_envtagnum();
#endif
      if('u'==var_type)
	cin=G__fgetstream_newtemplate(temp,",;{}"); /* TEMPLATECLASS case12 */
      else 
	cin=G__fgetstream_new(temp,",;{"); 

#ifndef G__OLDIMPLEMENTATION1084
      if(G__def_struct_member && G__CONSTVAR!=G__constvar && G__static_alloc &&
	 -1!=G__tagdefining && 
	 ('c'==G__struct.type[G__tagdefining]||
	  's'==G__struct.type[G__tagdefining])) {
	if(G__dispmsg>=G__DISPWARN) {
	  G__fprinterr(G__serr,"Warning: In-class initialization of non-const static member not allowed in C++ standard");
	  G__printlinenum();
	}
      }
#endif

      /*************************************************************
      * ignore array and struct initialization
      *  type var1[N] = { 0, 1, 2.... }
      *                  ^
      *************************************************************/
      if(cin=='{') {
	initary=1;
	/* reg=G__getexpr(temp); is going to be G__null because temp is "" */
      }

      /*************************************************************
      * FIX due to G__NSPEEDUP0
      *  double pi=3.14;
      *  double a=pi;  <= pi has been searched as 'd' which was bad.
      *************************************************************/
      G__var_type = 'p';

      /* ON199 */
      if(G__reftype 
#ifndef G__OLDIMPLEMENTATION921
         == G__PARAREFERENCE 
#endif
        ) {
	int store_reftype = G__reftype;
	/*#define G__OLDIMPLEMENTATION1093*/
#ifndef G__OLDIMPLEMENTATION1093
	int store_prerun=G__prerun;
	int store_decl=G__decl;
#ifndef G__OLDIMPLEMENTATION1119
	int store_constvar=G__constvar;
	int store_static_alloc=G__static_alloc;
#endif
	if(G__NOLINK==G__globalcomp) {
	  G__prerun=0;
	  G__decl=0;
#ifndef G__OLDIMPLEMENTATION1119
	  if(G__CONSTVAR&G__constvar) G__initval_eval=1;
	  G__constvar=0;
	  G__static_alloc=0;
#endif
	}
#endif
#ifndef G__OLDIMPLEMENTATION1254
	--G__templevel;
#endif
	G__reftype=G__PARANORMAL;
#ifndef G__OLDIMPLEMENTATION1093
#ifndef G__OLDIMPLEMENTATION1549
	if(store_prerun||0==store_static_alloc||G__IsInMacro()) {
	  reg=G__getexpr(temp);
	}
#else
	if(store_prerun||0==G__static_alloc||G__IsInMacro())
	  reg=G__getexpr(temp);
#endif
	else reg=G__null;
#else
	reg = G__getexpr(temp);
#endif
#ifndef G__OLDIMPLEMENTATION1254
	++G__templevel;
#endif
#ifndef G__OLDIMPLEMENTATION1093
	G__prerun=store_prerun;
	G__decl=store_decl;
#ifndef G__OLDIMPLEMENTATION1119
	G__constvar=store_constvar;
	G__static_alloc=store_static_alloc;
	G__initval_eval=0;
#endif
#endif
	G__reftype=store_reftype;
	G__globalvarpointer = reg.ref;
	reg=G__null;
	if(G__globalvarpointer==(long)NULL && 'u'==G__ansipara.type &&
	   (G__prerun==0 && 0==G__no_exec_compile)) {
	  G__referencetypeerror(new_name);
	}
      }
      else {
	if(var_type=='u'&&G__def_struct_member==0&&new_name[0]!='*') {
	  /* if struct or class, handled later with constructor */
	  reg = G__null;
	  /* avoiding assignment ignore in G__letvariable when reg==G__null */
	  if(staticclassobject) reg=G__one;
#ifdef G__OLDIMPLEMENTATION1032_YET
	  if(0==strncmp(temp,"new ",4)) G__assign_error(new_name,&G__null);
#endif
	}
	else if('u'==var_type&&'*'==new_name[0]&&0==strncmp(temp,"new ",4)){
#ifndef G__OLDIMPLEMENTATION1093
	  int store_prerun=G__prerun;
	  int store_decl=G__decl;
#ifndef G__OLDIMPLEMENTATION1119
	  int store_constvar=G__constvar;
	  int store_static_alloc=G__static_alloc;
#endif
	  if(G__NOLINK==G__globalcomp) {
	    G__prerun=0;
	    G__decl=0;
#ifndef G__OLDIMPLEMENTATION1119
	    if(G__CONSTVAR&G__constvar) G__initval_eval=1;
	    G__constvar=0;
	    G__static_alloc=0;
#endif
	  }
#ifndef G__OLDIMPLEMENTATION1549
	  if(store_prerun||0==store_static_alloc||G__IsInMacro()) {
	    reg=G__getexpr(temp);
	  }
#else
	  if(store_prerun||0==G__static_alloc||G__IsInMacro())
	    reg = G__getpower(temp);
#endif
	  else reg=G__null;
	  G__prerun=store_prerun;
	  G__decl=store_decl;
#ifndef G__OLDIMPLEMENTATION1119
	  G__constvar=store_constvar;
	  G__static_alloc=store_static_alloc;
	  G__initval_eval=0;
#endif
#else
	  reg = G__getpower(temp);
#endif
#ifndef G__OLDIMPLEMENTATION1006
	  if('U'!=reg.type && 'Y'!=reg.type && 0!=reg.obj.i) {
	    G__assign_error(new_name+1,&reg);
	    reg = G__null;
	  }
#endif
	}
	else {
#ifndef G__OLDIMPLEMENTATION1093
	  int store_prerun=G__prerun;
	  int store_decl=G__decl;
#ifndef G__OLDIMPLEMENTATION1119
	  int store_constvar=G__constvar;
	  int store_static_alloc=G__static_alloc;
#endif
	  if(G__NOLINK==G__globalcomp) {
	    G__prerun=0;
	    G__decl=0;
#ifndef G__OLDIMPLEMENTATION1119
	    if(G__CONSTVAR&G__constvar) G__initval_eval=1;
	    G__constvar=0;
	    G__static_alloc=0;
#endif
	  }
#ifndef G__OLDIMPLEMENTATION1549
	  if(store_prerun||0==store_static_alloc||G__IsInMacro()) {
#ifndef G__OLDIMPLEMENTATION1927
	    /* int store_tagnumC = G__tagnum; */
	    /* int store_def_tagnumC = G__def_tagnum; */
	    int store_tagdefiningC = G__tagdefining;
#endif
#ifndef G__OLDIMPLEMENTATION1551
	    int store_eval_localstatic = G__eval_localstatic;
	    G__eval_localstatic=1;
#endif
	    reg=G__getexpr(temp);
#ifndef G__OLDIMPLEMENTATION1551
	    G__eval_localstatic=store_eval_localstatic;
#endif
#ifndef G__OLDIMPLEMENTATION1927
	    /* G__tagnum = store_tagnumC; shouldn't do this */
	    /* G__def_tagnum = store_def_tagnumC; shouldn't do this */
	    G__tagdefining = store_tagdefiningC;
#endif
	  }
#else
	  if(store_prerun||0==G__static_alloc||G__IsInMacro()) {
	    reg=G__getexpr(temp);
	  }
#endif
	  else reg=G__null;
	  G__prerun=store_prerun;
	  G__decl=store_decl;
#ifndef G__OLDIMPLEMENTATION1119
	  G__constvar=store_constvar;
	  G__static_alloc=store_static_alloc;
	  G__initval_eval=0;
#endif
#else
	  reg = G__getexpr(temp);
#endif
#ifndef G__OLDIMPLEMENTATION1006
	  if('u'==var_type&&'*'==new_name[0]&&'U'!=reg.type&&0!=reg.obj.i
	     && 'Y'!=reg.type) {
	    G__assign_error(new_name+1,&reg);
	    reg = G__null;
	  }
#endif
	}	
      }
#ifndef G__OLDIMPLEMENTATION737
      G__tagnum = store_tagnumB;
#endif
    }
    else {
      if(
#ifndef G__OLDIMPLEMENTATION779
	 '\0'!=new_name[0] && 
#endif
#ifndef G__OLDIMPLEMENTATION807
	 G__NOLINK==G__globalcomp &&
#endif
	 G__reftype== G__PARAREFERENCE && 0==G__def_struct_member) {
	G__fprinterr(G__serr,"Error: reference type %s with no initialization "
		,new_name);
	G__genericerror((char*)NULL);
      }
      reg = G__null;
    }


    /***************************************************************
     * Create body of variable 
     *
     ***************************************************************/
    create_body:
    if(new_name[0]!='\0') {
      G__var_type = var_type ;

      /**************************************************************
      * declaration of struct object, no pointer, no reference type
      **************************************************************/
      if(var_type=='u'&&
#ifndef G__OLDIMPLEMENTATION871
	 (G__def_struct_member==0||-1==G__def_tagnum||
	  'n'==G__struct.type[G__def_tagnum])
#else
	 G__def_struct_member==0
#endif
	 &&new_name[0]!='*'&&
	 G__reftype==G__PARANORMAL) {

	store_prerun = G__prerun;
	if(store_prerun) {
	  store_debug = G__debug;
	  store_step = G__step;
	  G__debug=G__debugtrace;
	  G__step = G__steptrace;
	  G__prerun = 0;
	  G__setdebugcond();
	  G__prerun=store_prerun;
	}
	else {
	  if(G__breaksignal) {
	    G__break=0;
	    G__setdebugcond();
	    if(G__pause()==3) {
	      if(G__return==G__RETURN_NON) {
		G__step=0;
		G__setdebugcond();
		largestep=1;
	      }
	    }
	    if(G__return>G__RETURN_NORMAL) {
	      G__decl=store_decl;
	      G__constvar=0;
	      G__tagnum = store_tagnum;
	      G__typenum = store_typenum;
	      G__reftype=G__PARANORMAL;
	      G__prerun=store_prerun;
	      G__static_alloc=store_static_alloc2;
#ifndef G__OLDIMPLEMENTATION1119
	      G__dynconst=0;
#endif
#ifndef G__OLDIMPLEMENTATION1322
	      G__globalvarpointer = G__PVOID;
#endif
	      return;
	    }
	  }
	}

#ifndef G__OLDIMPLEMENTATION927
	if(G__static_alloc&&0==G__prerun) {
#ifndef G__OLDIMPLEMENTATION1624
	  if('{'==cin) {
	    while('}'!=cin) cin = G__fignorestream(";,");
	  }
#endif
	  if(';'!=cin&&','!=cin) cin = G__fignorestream(";,");
	  G__var_type = var_type;
	  G__letvariable(new_name,reg,&G__global,G__p_local);
	  goto readnext;
	}
#endif

#ifndef G__OLDIMPLEMENTATION1552
	if(initary && strchr(new_name,'[') &&
	   (G__struct.funcs[G__tagnum]&G__HAS_CONSTRUCTOR)) {
	  G__initstructary(new_name,G__tagnum);
	  G__decl=store_decl;
	  G__constvar=0;
	  G__tagnum = store_tagnum;
	  G__typenum = store_typenum;
	  G__reftype=G__PARANORMAL;
	  G__static_alloc=store_static_alloc2;
	  G__dynconst=0;
	  G__globalvarpointer = G__PVOID;
	  return;
	}
#endif

	/************************************************************
	* memory allocation and table entry generation
	************************************************************/
	store_struct_offset = G__store_struct_offset;
	if(G__CPPLINK!=G__struct.iscpplink[tagnum]) {
	  /* allocate memory area for constructed object by interpreter */
	  G__var_type = var_type;
#ifndef G__OLDIMPLEMENTATION1349
	  G__decl_obj=1;
#endif
	  G__store_struct_offset=G__int(G__letvariable(new_name,reg,&G__global
						       ,G__p_local));
#ifndef G__OLDIMPLEMENTATION1349
	  G__decl_obj=0;
#endif
#ifndef G__OLDIMPLEMENTATION1073
	  if(0==G__store_struct_offset &&
	     G__asm_wholefunction && G__asm_noverflow) {
	    G__store_struct_offset = G__PVOID;
	  }
#endif
	}
	else {
	  /* precompiled class, 
	   * memory will be allocated by new in constructor function below */
	  G__store_struct_offset = G__PVOID;
	}


	if(G__return>G__RETURN_NORMAL) {
	  G__decl=store_decl;
	  G__constvar=0;
	  G__tagnum = store_tagnum;
	  G__typenum = store_typenum;
	  G__reftype=G__PARANORMAL;
	  G__static_alloc=store_static_alloc2;
#ifndef G__OLDIMPLEMENTATION1119
	  G__dynconst=0;
#endif
#ifndef G__OLDIMPLEMENTATION1322
	  G__globalvarpointer = G__PVOID;
#endif
	  return;
	}
	G__prerun = 0; /* FOR RUNNING CONSTRUCTOR */

	if(G__store_struct_offset) {
	  if(temp[0] == '\0'
#ifndef G__OLDIMPLEMENTATION1304
	     && -1!=G__tagnum 
#endif
	     ) {
	    /********************************************
	     * type a; 
	     * call default constructor
	     ********************************************/
	    sprintf(temp,"%s()",G__struct.name[G__tagnum]);
	    if(G__dispsource){
		G__fprinterr(G__serr,
	    "\n!!!Calling default constructor 0x%lx.%s for declaration of %s"
			,G__store_struct_offset
			,temp,new_name);
	    }
	    /******************************************************
	    * Calling constructor to array of object
	    ******************************************************/
	    G__decl=0; 
	    if((index=strchr(new_name,'['))) {
	      p_inc=G__getarrayindex(index);
	      if(G__CPPLINK==G__struct.iscpplink[tagnum]) {
		/* precompiled class. First, call constructor (new) function */
#ifndef G__OLDIMPLEMENTATION1437
#ifdef G__ASM
		if(G__asm_noverflow && p_inc>1) {
#ifdef G__ASM_DBG
		  if(G__asm_dbg) G__fprinterr(G__serr,"%3x: SETARYINDEX\n" ,G__asm_cp);
#endif
		  G__asm_inst[G__asm_cp]=G__SETARYINDEX;
		  G__asm_inst[G__asm_cp+1]= 0;
		  G__inc_cp_asm(2,0);
		}
#endif
#endif
		G__cpp_aryconstruct=p_inc;
		reg=G__getfunction(temp,&known,G__CALLCONSTRUCTOR);
		G__cpp_aryconstruct=0;
		/* Register the pointer we get from new to member variable table */
		G__globalvarpointer=G__int(reg);
		G__cppconstruct = 1;
		G__var_type = var_type;
		G__letvariable(new_name,G__null,&G__global,G__p_local);
		G__cppconstruct = 0;
#ifndef G__OLDIMPLEMENTATION1701
#ifdef G__ASM
		if(G__asm_noverflow && p_inc>1) {
#ifdef G__ASM_DBG
		  if(G__asm_dbg) G__fprinterr(G__serr,"%3x: RESETARYINDEX\n" ,G__asm_cp);
#endif
		  G__asm_inst[G__asm_cp]=G__RESETARYINDEX;
		  G__asm_inst[G__asm_cp+1]= 0;
		  G__inc_cp_asm(2,0);
		}
#endif
#endif /* 1701 */
		G__globalvarpointer=G__PVOID;
#ifndef G__OLDIMPLEMENTATION1073
		if(G__asm_wholefunction&&G__no_exec_compile) {
#ifdef G__ASM_DBG
		  if(G__asm_dbg) 
		    G__fprinterr(G__serr,"%3x: SETGVP -1\n",G__asm_cp);
#endif
		  G__asm_inst[G__asm_cp]=G__SETGVP;
		  G__asm_inst[G__asm_cp+1] = -1;
		  G__inc_cp_asm(2,0);
		}
#endif
	      }
	      else {
		/* interpreterd class, memory area is alread allocated above */
		for(i=0;i<p_inc;i++) {
#ifndef G__OLDIMPLEMENTATION1238
		  if(G__struct.isctor[tagnum]) 
		    G__getfunction(temp,&known,G__CALLCONSTRUCTOR);
		  else
		    G__getfunction(temp,&known,G__TRYCONSTRUCTOR);
#else
		  G__getfunction(temp,&known,G__TRYCONSTRUCTOR);
#endif
		  if(G__return>G__RETURN_NORMAL||0==known) break;
		  G__store_struct_offset+=G__struct.size[G__tagnum];
#ifndef G__OLDIMPLEMENTATION1444
		  if(G__asm_noverflow) {
#ifdef G__ASM_DBG
		    if(G__asm_dbg) G__fprinterr(G__serr,"%3x: ADDSTROS %d\n"
					   ,G__asm_cp
					   ,G__struct.size[G__tagnum]);
#endif
		    G__asm_inst[G__asm_cp]=G__ADDSTROS;
		    G__asm_inst[G__asm_cp+1]=G__struct.size[G__tagnum];
		    G__inc_cp_asm(2,0);
		  }
#endif
#ifndef G__OLDIMPLEMENTATION1073
		  if(G__asm_wholefunction && G__asm_noverflow) {
#ifdef G__ASM_DBG
		    if(G__asm_dbg) G__fprinterr(G__serr,"%3x: ADDSTROS %d\n"
			              ,G__asm_cp,G__struct.size[G__tagnum]);
#endif
		    G__asm_inst[G__asm_cp]=G__POPSTROS; /* ??? ADDSTROS */
		    G__asm_inst[G__asm_cp+1]=G__struct.size[G__tagnum];
		    G__inc_cp_asm(2,0);
		  }
#endif
		}
#ifndef G__OLDIMPLEMENTATION1073
		if(G__asm_wholefunction && G__asm_noverflow) {
#ifdef G__ASM_DBG
		  if(G__asm_dbg) G__fprinterr(G__serr,"%3x: POPSTROS\n",G__asm_cp);
#endif
		  G__asm_inst[G__asm_cp]=G__POPSTROS;
		  G__inc_cp_asm(1,0);
		}
#endif
	      }
	    }
	    /******************************************************
	    * Calling constructor to normal object
	    ******************************************************/
	    else {
	      if(G__CPPLINK==G__struct.iscpplink[tagnum]) {
		/* precompiled class. First, call constructor (new) function */
		reg=G__getfunction(temp,&known,G__TRYCONSTRUCTOR);
		/* Register the pointer we get from new to member variable table */
		G__globalvarpointer=G__int(reg);
		G__cppconstruct = 1;
		G__var_type = var_type;
#ifndef G__OLDIMPLEMENTATION1137
#ifndef G__OLDIMPLEMENTATION1251
		if((known && (G__globalvarpointer||G__asm_noverflow))
#ifndef G__OLDIMPLEMENTATION1325
		   || G__NOLINK != G__globalcomp 
#endif
		   ) {
#else
		if(G__globalvarpointer) {
#endif
		  G__letvariable(new_name,G__null,&G__global,G__p_local);
		}
		else {
#ifndef G__OLDIMPLEMENTATION1164
		  if(0==G__xrefflag) {
#ifndef G__OLDIMPLEMENTATION1180
		    if(G__ASM_FUNC_NOP==G__asm_wholefunction)
		      G__fprinterr(G__serr,"Error: %s no default constructor",temp);
#else
		    G__fprinterr(G__serr,"Error: %s no default constructor",temp);
#endif
		    G__genericerror((char*)NULL);
		  }
		  else {
		    G__letvariable(new_name,G__null,&G__global,G__p_local);
		  }
#else
		  G__fprinterr(G__serr,"Error: %s no default constructor",temp);
		  G__genericerror((char*)NULL);
#endif
		}
#else
		G__letvariable(new_name,G__null,&G__global,G__p_local);
#endif
		G__cppconstruct = 0;
		G__globalvarpointer=G__PVOID;
#ifndef G__OLDIMPLEMENTATION1073
		if(G__asm_wholefunction&&G__no_exec_compile) {
#ifdef G__ASM_DBG
		  if(G__asm_dbg) 
		    G__fprinterr(G__serr,"%3x: SETGVP -1\n",G__asm_cp);
#endif
		  G__asm_inst[G__asm_cp]=G__SETGVP;
		  G__asm_inst[G__asm_cp+1] = -1;
		  G__inc_cp_asm(2,0);
		}
#endif
	      }
	      else {
		/* interpreterd class, memory area is alread allocated above */
#ifndef G__OLDIMPLEMENTATION1238
		if(G__struct.isctor[tagnum]) 
		  G__getfunction(temp,&known,G__CALLCONSTRUCTOR);
		else
		  G__getfunction(temp,&known,G__TRYCONSTRUCTOR);
#else
		G__getfunction(temp,&known,G__TRYCONSTRUCTOR);
#endif
#ifndef G__OLDIMPLEMENTATION1073
		if(G__asm_wholefunction && G__asm_noverflow) {
#ifdef G__ASM_DBG
		  if(G__asm_dbg) G__fprinterr(G__serr,"%3x: POPSTROS\n",G__asm_cp);
#endif
		  G__asm_inst[G__asm_cp]=G__POPSTROS;
		  G__inc_cp_asm(1,0);
		}
#endif
	      }
	    }
	    G__decl=1;
	    if(G__return>G__RETURN_NORMAL) {
	      G__decl=store_decl;
	      G__constvar=0;
	      G__tagnum = store_tagnum;
	      G__typenum = store_typenum;
	      G__reftype=G__PARANORMAL;
	      G__static_alloc=store_static_alloc2;
#ifndef G__OLDIMPLEMENTATION1119
	      G__dynconst=0;
#endif
#ifndef G__OLDIMPLEMENTATION1322
	      G__globalvarpointer = G__PVOID;
#endif
	      return;
	    }
	    /* struct class initialization ={x,y,z} */
	    if(initary) {
	      if(known
#ifndef G__OLDIMPLEMENTATION2130
		 && (G__struct.funcs[tagnum]& G__HAS_XCONSTRUCTOR)
		 /* && (G__struct.funcs[tagnum]& G__HAS_DEFAULTCONSTRUCTOR) */
#endif
                ) {
		G__fprinterr(G__serr,
		"Error: Illegal initialization of %s. Constructor exists "
			,new_name);
		G__genericerror((char*)NULL);
		cin=G__fignorestream("}");
		cin=G__fignorestream(",;");
		
	      }
	      else {
		if(store_prerun) {
		  G__debug=store_debug;
		  G__step=store_step;
		  G__setdebugcond();
#ifndef G__OLDIMPLEMENTATION1624
		  G__prerun = store_prerun;
#endif
		}
		cin=G__initstruct(new_name);
	      }
	      initary=0;
	    }
          }
	  else {
	    /********************************************
	     * If temp == 'classname(arg)', this is OK,
	     * If temp == 'classobject', copy constructor
	     ********************************************/
	    if(staticclassobject) G__tagnum=store_tagnum; /* to pass G__getfunction() */
	    sprintf(temp1,"%s(",G__struct.name[G__tagnum]);
#ifdef G__TEMPLATECLASS
	    /* G__TEMPLATECLASS Need to evaluate template argument list here */
#endif
	    if( temp == strstr(temp,temp1)) {
#ifndef G__OLDIMPLEMENTATION1704
	      int c,isrc=0;
	      char buf[G__LONGLINE];
	      flag=1;
	      c=G__getstream_template(temp,&isrc,buf,"(");
	      if('('==c) {
		c=G__getstream_template(temp,&isrc,buf,")");
		if(')'==c) {
		  if(temp[isrc]) flag=0;
		}
	      }
#else
	      flag=1;
#endif
	    }
	    else if(G__struct.istypedefed[G__tagnum]) {
	      index=strchr(temp,'(');
	      if(index) {
		*index='\0';
		flag=G__defined_typename(temp);
		if(-1!=flag&&G__newtype.tagnum[flag]==G__tagnum) {
		  sprintf(temp1,"%s(%s",G__struct.name[G__tagnum],index+1);
		  strcpy(temp,temp1);
		  flag=1;
		}
		else flag=0;
#ifndef G__OLDIMPLEMENTATION732
	      if(!flag) *index='(';
#endif
	      }
	      else flag=0;
#ifdef G__OLDIMPLEMENTATION732
	      if(!flag) *index='(';
#endif
	    }
	    else flag=0;

	    if( flag ) {
	      /* call explicit constructor, error if no constructor */
	      if(G__dispsource){
		G__fprinterr(G__serr,
		   "\n!!!Calling constructor 0x%lx.%s for declaration of %s"
			,G__store_struct_offset
			,temp,new_name);
	      }
	      G__decl=0; 
	      if(G__CPPLINK==G__struct.iscpplink[tagnum]) {
		reg=G__getfunction(temp,&known,G__CALLCONSTRUCTOR);
		G__var_type=var_type;
		G__globalvarpointer=G__int(reg);
		G__cppconstruct=1;
		if(G__globalvarpointer)
		  G__letvariable(new_name,G__null,&G__global,G__p_local);
		G__cppconstruct=0;
		G__globalvarpointer=G__PVOID;
	      }
	      else {
#ifndef G__OLDIMPLEMENTATION986
		/* There are similar cases above, but they are either
                 * default ctor or precompiled class which should be fine */
                int store_static_alloc3=G__static_alloc;
		G__static_alloc=0;
#endif
		G__getfunction(temp,&known,G__CALLCONSTRUCTOR);
#ifndef G__OLDIMPLEMENTATION986
                G__static_alloc=store_static_alloc3;
#endif
	      }
	      G__decl=1;
	      if(G__return>G__RETURN_NORMAL) {
		G__decl=store_decl;
		G__constvar=0;
		G__tagnum = store_tagnum;
		G__typenum = store_typenum;
		G__reftype=G__PARANORMAL;
		G__static_alloc=store_static_alloc2;
#ifndef G__OLDIMPLEMENTATION1119
		G__dynconst=0;
#endif
#ifndef G__OLDIMPLEMENTATION1322
		G__globalvarpointer = G__PVOID;
#endif
		return;
	      }
	    }
	    else {
	      int store_var_typeB,store_tagnumB,store_typenumB;
#ifndef G__OLDIMPLEMENTATION931
	      long store_struct_offsetB=G__store_struct_offset;
#endif
#ifndef G__OLDIMPLEMENTATION1927
	      /* int store_def_tagnumB = G__def_tagnum; shouldn't do this */
	      int store_tagdefiningB = G__tagdefining;
#endif
	      /*********************************************
	       * G__COPYCONSTRUCTOR
	       * default and user defined copy constructor
	       * is switched in G__letvariable()
	       *********************************************/
	      /* Call copy constructor with G__decl=1 argument reg */
	      store_var_typeB=G__var_type;
	      store_tagnumB=G__tagnum;
	      store_typenumB=G__typenum;
	      G__var_type='p';
	      G__tagnum = G__memberfunc_tagnum;
	      G__typenum = -1;
#ifndef G__OLDIMPLEMENTATION931
	      G__store_struct_offset=G__memberfunc_struct_offset;
#ifndef G__OLDIMPLEMENTATION1073
	      if(G__asm_noverflow&&G__asm_wholefunction) {
#ifdef G__ASM_DBG
		if(G__asm_dbg) 
		  G__fprinterr(G__serr,"%3x: SETMEMFUNCENV\n",G__asm_cp);
#endif
		G__asm_inst[G__asm_cp]=G__SETMEMFUNCENV;
		G__inc_cp_asm(1,0);
	      }
#endif
#endif
	      reg=G__getexpr(temp);
#ifndef G__OLDIMPLEMENTATION931
	      G__store_struct_offset=store_struct_offsetB;
#ifndef G__OLDIMPLEMENTATION1073
	      if(G__asm_noverflow&&G__asm_wholefunction) {
#ifdef G__ASM_DBG
		if(G__asm_dbg) 
		  G__fprinterr(G__serr,"%3x: RECMEMFUNCENV\n",G__asm_cp);
#endif
		G__asm_inst[G__asm_cp]=G__RECMEMFUNCENV;
		G__inc_cp_asm(1,0);
	      }
#endif
#endif
	      G__var_type=store_var_typeB;
	      G__tagnum = store_tagnumB;
	      G__typenum = store_typenumB;
#ifndef G__OLDIMPLEMENTATION1927
	      /* G__def_tagnum = store_def_tagnumB; shouldn't do this */
	      G__tagdefining = store_tagdefiningB;
#endif
	      if(G__CPPLINK==G__struct.iscpplink[tagnum]) {
#ifndef G__OLDIMPLEMENTATION983
		if(reg.tagnum==tagnum && 'u'==reg.type) {
#else
		if(reg.tagnum>=0) {
#endif
		  if(reg.obj.i<0) 
		    sprintf(temp,"%s((%s)(%ld))" ,G__struct.name[tagnum]
			    ,G__struct.name[tagnum] ,G__int(reg));
		  else
		    sprintf(temp,"%s((%s)%ld)" ,G__struct.name[tagnum]
			    ,G__struct.name[tagnum] ,G__int(reg));
		}
		else {
		  char tttt[G__ONELINE];
#define G__OLDIMPLEMENTATION1780
#ifndef G__OLDIMPLEMENTATION1780
		  if(reg.type) {
		    G__valuemonitor(reg,tttt);
		    sprintf(temp,"%s(%s)",G__struct.name[tagnum],tttt);
		  }
		  else {
		    strcpy(tttt,temp);
		    sprintf(temp,"%s(%s)",G__struct.name[tagnum],tttt);
		  }
#else
		  G__valuemonitor(reg,tttt);
		  sprintf(temp,"%s(%s)",G__struct.name[tagnum],tttt);
#endif
		}
#ifndef G__OLDIMPLEMENTATION1073
		if(G__asm_wholefunction) {
		  G__oprovld=1;
		}
#endif
#ifndef G__OLDIMPLEMENTATION1096
		G__oprovld=1;
		G__decl=0;
#endif
#ifndef G__OLDIMPLEMENTATION979
		if(-1!=G__struct.parent_tagnum[tagnum]) {
                  int store_exec_memberfunc=G__exec_memberfunc;
                  int store_memberfunc_tagnum=G__memberfunc_tagnum;
                  G__exec_memberfunc=1;
                  G__memberfunc_tagnum=G__struct.parent_tagnum[tagnum];
		  reg=G__getfunction(temp,&known,G__CALLCONSTRUCTOR);
                  G__exec_memberfunc=store_exec_memberfunc;
                  G__memberfunc_tagnum=store_memberfunc_tagnum;
		}
                else {
		  reg=G__getfunction(temp,&known,G__CALLCONSTRUCTOR);
                }
#else
		reg=G__getfunction(temp,&known,G__CALLCONSTRUCTOR);
#endif
		G__globalvarpointer=G__int(reg);
		G__cppconstruct=1;
		G__letvariable(new_name,G__null,&G__global,G__p_local);
		G__cppconstruct=0;
		G__globalvarpointer=G__PVOID;
#ifndef G__OLDIMPLEMENTATION1096
		G__oprovld=0;
		G__decl=1;
#endif
#ifndef G__OLDIMPLEMENTATION1073
		if(G__asm_wholefunction) {
		  G__oprovld=0;
#ifdef G__ASM_DBG
		  if(G__asm_dbg) 
		    G__fprinterr(G__serr,"%3x: SETGVP -1\n",G__asm_cp);
#endif
		  G__asm_inst[G__asm_cp]=G__SETGVP;
		  G__asm_inst[G__asm_cp+1] = -1;
		  G__inc_cp_asm(2,0);
		}
#endif
	      }
	      else {
#ifndef G__OLDIMPLEMENTATION1073
		if(G__asm_wholefunction) {
		  G__oprovld=1;
		}
#endif
		G__letvariable(new_name ,reg ,&G__global ,G__p_local);
#ifndef G__OLDIMPLEMENTATION1073
		if(G__asm_wholefunction) {
		  G__oprovld=0;
#ifdef G__ASM_DBG
		  if(G__asm_dbg) G__fprinterr(G__serr,"%3x: POPSTROS\n",G__asm_cp);
#endif
		  G__asm_inst[G__asm_cp]=G__POPSTROS;
		  G__inc_cp_asm(1,0);
		}
#endif
	      }
	      if(G__return>G__RETURN_NORMAL) {
		G__decl=store_decl;
		G__constvar=0;
		G__tagnum = store_tagnum;
		G__typenum = store_typenum;
		G__reftype=G__PARANORMAL;
		G__static_alloc=store_static_alloc2;
#ifndef G__OLDIMPLEMENTATION1119
		G__dynconst=0;
#endif
#ifndef G__OLDIMPLEMENTATION1322
		G__globalvarpointer = G__PVOID;
#endif
#ifndef G__PHILIPPE21
		G__prerun = store_prerun;
#endif
		return;
	      }
	    }
	  }
	}
	else { /* of if(G__store_struct_offset */
	  if(G__var_type=='u') {
	    G__fprinterr(G__serr,
		    "Error: %s not allocated(2), maybe duplicate declaration "
		    ,new_name );
	    G__genericerror((char*)NULL);
	  }
	  /* else OK because this is type name[]; */
	  if(initary) {
	    if(store_prerun) {
	      G__debug=store_debug;
	      G__step=store_step;
	      G__setdebugcond();
#ifndef G__OLDIMPLEMENTATION1875
	      G__prerun = store_prerun;
#endif
	    }
	    cin=G__initstruct(new_name);
	  }
	} /* of if(G__store_struct_offset) else */

	if(largestep) {
		largestep=0;
		G__step=1;
		G__setdebugcond();
	}

	if(store_prerun) {
	  G__debug=store_debug;
	  G__step=store_step;
	  G__setdebugcond();
	}
	G__prerun = store_prerun;
	G__store_struct_offset = store_struct_offset;
#ifndef G__OLDIMPLEMENTATION1482
#ifdef G__ASM
	if(G__asm_noverflow) {
#ifdef G__ASM_DBG
	  if(G__asm_dbg) G__fprinterr(G__serr,"%3x: POPSTROS\n",G__asm_cp);
#endif
	  G__asm_inst[G__asm_cp] = G__POPSTROS;
	  G__inc_cp_asm(1,0);
	}
#endif
#endif
      } /* of if(var_type=='u'&&G__def_struct_member.... */

      /**************************************************************
      * declaration of scaler object, pointer or reference type.
      **************************************************************/
      else {
	G__letvariable(new_name,reg,&G__global,G__p_local);
	if(G__return>G__RETURN_NORMAL) {
	  G__decl=store_decl;
	  G__constvar=0;
	  G__tagnum = store_tagnum;
	  G__typenum = store_typenum;
	  G__reftype=G__PARANORMAL;
	  G__static_alloc=store_static_alloc2;
#ifndef G__OLDIMPLEMENTATION1119
	  G__dynconst=0;
#endif
#ifndef G__OLDIMPLEMENTATION1322
	  G__globalvarpointer = G__PVOID;
#endif
	  return;
	}
	/* insert array initialization */
	if(initary) {
	  cin=G__initary(new_name);
	  initary=0;
#ifndef G__OLDIMPLEMENTATION1118
	  if(EOF==cin) {
	    G__decl=store_decl;
	    G__constvar=0;
	    G__tagnum = store_tagnum;
	    G__typenum = store_typenum;
	    G__reftype=G__PARANORMAL;
	    G__static_alloc=store_static_alloc2;
#ifndef G__OLDIMPLEMENTATION1119
	    G__dynconst=0;
#endif
#ifndef G__OLDIMPLEMENTATION1322
	    G__globalvarpointer = G__PVOID;
#endif
	    return;
	  }
#endif
	}
      }
      /**************************************************************
      * end of if(var_type=='u'&&G__def_struct_member==0&&new_name[0]!='*'&&
      *           G__reftype==G__PARANORMAL) 
      *        else
      **************************************************************/

      if(G__ansiheader==2) G__ansiheader=0;
    }
    /***************************************************************
     * end of if(new_name[0]!='\0')
     ***************************************************************/
      
      
      G__globalvarpointer=G__PVOID;
      
      
    /***************************************************************
     * end of declaration or read next variable name 
     *
     ***************************************************************/
  readnext:
    if(cin==';') {
      /* type  var1 , var2 ;
       *                    ^
       *  end of declaration, return
       */
      G__decl=store_decl;
      G__constvar=0;
      G__tagnum = store_tagnum;
      G__typenum = store_typenum;
      G__reftype=G__PARANORMAL;
      
#ifdef G__FONS_COMMENT
      if(G__fons_comment && G__def_struct_member) {
	G__setvariablecomment(new_name);
      }
#endif

#ifdef G__ASM
	if(G__asm_noverflow) G__asm_clear();
#endif
      
      G__static_alloc=store_static_alloc2;
#ifndef G__OLDIMPLEMENTATION1119
      G__dynconst=0;
#endif
#ifndef G__OLDIMPLEMENTATION1322
      G__globalvarpointer = G__PVOID;
#endif
      return;
    }
    else if('}'==cin) {
      G__decl=store_decl;
      G__constvar=0;
      G__tagnum = store_tagnum;
      G__typenum = store_typenum;
      G__reftype=G__PARANORMAL;
      fseek(G__ifile.fp,-1,SEEK_CUR);
      G__missingsemicolumn(new_name);
      G__static_alloc=store_static_alloc2;
#ifndef G__OLDIMPLEMENTATION1119
      G__dynconst=0;
#endif
#ifndef G__OLDIMPLEMENTATION1322
      G__globalvarpointer = G__PVOID;
#endif
      return;
    }
    else {
      /* type  var1 , var2 , var3 ;
       *             ^  or  ^
       *  read variable name
       */
      cin=G__fgetstream(new_name,",;=():");
#ifndef G__OLDIMPLEMENTATION1118
      if(EOF==cin) {
	G__decl=store_decl;
	G__constvar=0;
	G__tagnum = store_tagnum;
	G__typenum = store_typenum;
	G__reftype=G__PARANORMAL;
	fseek(G__ifile.fp,-1,SEEK_CUR);
	G__missingsemicolumn(new_name);
	G__static_alloc=store_static_alloc2;
#ifndef G__OLDIMPLEMENTATION1119
	G__dynconst=0;
#endif
#ifndef G__OLDIMPLEMENTATION1322
	G__globalvarpointer = G__PVOID;
#endif
	return;
      }
#endif
      if(G__typepdecl) {
	var_type = tolower(var_type);
	G__var_type = var_type;
	if(G__asm_dbg) {
	  if(G__dispmsg>=G__DISPNOTE) {
	    G__fprinterr(G__serr,"Note: type* a,b,... declaration");
	    G__printlinenum();
	  }
	}
      }
      /* type  var1 , var2 , var3 ;
       * came to            ^  or  ^
       */
    }

  }  

}


/**************************************************************************
* G__initary()
**************************************************************************/
int G__initary(new_name)
char *new_name;
{
  struct G__var_array *var;
  char name[G__MAXNAME];
  char expr[G__ONELINE];
  G__value buf;
  int c;
  char *p;
  int hash,i;
  int ig15;
  int pinc,pindex,pi,inc;
  int mparen;
  G__value reg;
  /* int ispointer=0; */
  int isauto=0;
  int size;
  int prev;
  long tmp;
#ifndef G__OLDIMPLEMENTATION1607
  int stringflag=0;
#endif
#ifndef G__OLDIMPLEMENTATION1632
  int typedary=0; 
#endif
  
  /* G__ASSERT(0==G__store_struct_offset); */

#ifdef G__OLDIMPLEMENTATION987
#ifdef G__ASM
  G__abortbytecode();
#endif
#endif
  
  /* separate variable name header */
  strcpy(name,new_name);
  p=strchr(name,'[');
  if(p) *p='\0';
  
  /* handling static declaration */
  if(G__static_alloc==1) {
    if(G__prerun==0) {
#ifndef G__OLDIMPLEMENTATION1950
      /* calculate hash */
      G__hash(name,hash,i)
      /* get variable table entry */
      var = G__getvarentry(name,hash,&ig15,&G__global,G__p_local);
      if(var && INT_MAX==var->varlabel[ig15][1]) {
	struct G__var_array *varstatic;
	char namestatic[G__ONELINE];
	int hashstatic,ig15static;
	if(-1!=G__memberfunc_tagnum) /* questionable */
	  sprintf(namestatic,"%s\\%x\\%x\\%x",name,G__func_page,G__func_now
		  ,G__memberfunc_tagnum);
	else
	  sprintf(namestatic,"%s\\%x\\%x" ,name,G__func_page,G__func_now);
	
	G__hash(namestatic,hashstatic,i)
	  varstatic = G__getvarentry(namestatic,hashstatic,&ig15static
				     ,&G__global,G__p_local);
	if(varstatic) {
          for(i=0;i<G__MAXVARDIM;i++) 
	    var->varlabel[ig15][i] = varstatic->varlabel[ig15static][i];
	}
      }
#endif
      /* ignore local static at run time */
      c=G__fignorestream("}");
      c=G__fignorestream(",;");
      return(c);
    }
    else if(G__func_now != -1) {
      /* local static at prerun, get global name */
      if(-1!=G__memberfunc_tagnum) /* questionable */
	sprintf(expr,"%s\\%x\\%x\\%x" ,name,G__func_page,G__func_now
		,G__memberfunc_tagnum);
      else
	sprintf(expr,"%s\\%x\\%x" ,name,G__func_page,G__func_now);
      strcpy(name,expr);
    }
  }

#ifndef G__OLDIMPLEMENTATION987
#ifdef G__ASM
  G__abortbytecode();
#endif
#endif

#ifndef G__OLDIMPLEMENTATION1667
  { 
    char *pp = G__strrstr(name,"::");
    if(pp && G__prerun && -1==G__func_now) {
      /* Handle static data member initialization */
      int tagnum;
      *pp=0;
      tagnum = G__defined_tagname(name,0);
      strcpy(expr,pp+2);
      G__hash(expr,hash,i)
      var = G__getvarentry(expr,hash,&ig15,G__struct.memvar[tagnum]
			   ,G__struct.memvar[tagnum]);
    }
    else {
      /* calculate hash */
      G__hash(name,hash,i)
      /* get variable table entry */
      var = G__getvarentry(name,hash,&ig15,&G__global,G__p_local);
    }
  }
#else
  /* calculate hash */
  G__hash(name,hash,i)
  /* get variable table entry */
  var = G__getvarentry(name,hash,&ig15,&G__global,G__p_local);
#endif

#ifndef G__OLDIMPLEMENTATION1119
  if(!var) {
    char temp[G__ONELINE];
    int itmpx,varhash;
    char *px=strchr(name,'\\');
    if(px) *px=0;
    if(-1!=G__tagdefining)
      sprintf(temp,"%s\\%x\\%x\\%x",name,G__func_page,G__func_now
	      ,G__tagdefining);
    else
      sprintf(temp,"%s\\%x\\%x" ,name,G__func_page,G__func_now);
    G__hash(temp,varhash,itmpx);
    var = G__getvarentry(temp,varhash,&ig15,&G__global,G__p_local);
#ifndef G__OLDIMPLEMENTATION2196
    if(!var && -1!=G__tagdefining) {
      sprintf(temp,"%s" ,name);
      G__hash(temp,varhash,itmpx);
      var = G__getvarentry(temp,varhash,&ig15
			   ,G__struct.memvar[G__tagdefining]
			   ,G__struct.memvar[G__tagdefining]);
    }
#endif
    if(!var) {
      c=G__fignorestream(",;");
      G__genericerror("Error: array initialization");
      return(c);
    }
  }
#endif
  
  /*******************************************************
   * multidimensional array number of dimension
   *******************************************************/
  pindex=var->paran[ig15];
  
  /*******************************************************
   * check if  a[], a[][B][C] isauto sized array 
   *******************************************************/
  if(INT_MAX==var->varlabel[ig15][1]) {
    /* set isauto flag and reset varlabel[ig15][1] */
    isauto=1;
    var->varlabel[ig15][1] = -1;
#ifndef G__OLDIMPLEMENTATION2009
    if(-1!=var->tagnum && G__LOCALSTATIC==var->statictype[ig15]) {
      G__ASSERT(G__PINVALID==var->p[ig15]&&G__prerun&&-1==G__func_now);
    }
    else {
      G__ASSERT(G__PINVALID==var->p[ig15] &&
		G__COMPILEDGLOBAL==var->statictype[ig15]);
      if(G__static_alloc==1) {
	if(-1 != G__func_now) {
	  var->statictype[ig15]=G__LOCALSTATICBODY;
	}
	else {
	  var->statictype[ig15]=G__ifile.filenum;
	}
      }
      else {
	var->statictype[ig15]=G__AUTO;
      }
    }
#else
    G__ASSERT(G__PINVALID==var->p[ig15] &&
	      G__COMPILEDGLOBAL==var->statictype[ig15]);
    if(G__static_alloc==1) {
      if(-1 != G__func_now) {
	var->statictype[ig15]=G__LOCALSTATICBODY;
      }
      else {
	var->statictype[ig15]=G__ifile.filenum;
      }
    }
    else {
      var->statictype[ig15]=G__AUTO;
    }
#endif
  }

  G__ASSERT(G__COMPILEDGLOBAL!=var->statictype[ig15]);
  
  /* initialize buf */
  buf.type=toupper(var->type[ig15]);
  buf.tagnum=var->p_tagtable[ig15];
  buf.typenum=var->p_typetable[ig15];
  buf.ref=0;
  buf.obj.reftype.reftype=var->reftype[ig15];
  
  /* getting size */
  if(islower(var->type[ig15])) {
#ifndef G__OLDIMPLEMENTATION1329
    if(-1!=buf.typenum && G__newtype.nindex[buf.typenum]) {
      char store_var_type = G__var_type;
      size=G__Lsizeof(G__newtype.name[buf.typenum]);
      G__var_type = store_var_type;
#ifndef G__OLDIMPLEMENTATION1632
      typedary=1; 
#endif
    }
    else {
      size=G__sizeof(&buf);
    }
#else
    size=G__sizeof(&buf);
#endif
  }
  else {
    buf.type='L'; /* pointer assignement handled as long */
    size=G__LONGALLOC;
  }
  G__ASSERT(0<var->varlabel[ig15][0]&&0<size);
  
  
  /*******************************************************
   * read initialization list 
   *******************************************************/
  mparen=1;
  inc=0;
  pi=pindex;
  pinc=0;
  while(mparen) {
    c=G__fgetstream(expr,",{}");
    if(expr[0]) {
      /********************************************
       * increment the pointer
       ********************************************/
#ifndef G__OLDIMPLEMENTATION1607
      if('c'==var->type[ig15] && '"'==expr[0]) {
#ifndef G__OLDIMPLEMENTATION1632
	if(0==typedary) size = var->varlabel[ig15][var->paran[ig15]];
#else
	size = var->varlabel[ig15][var->paran[ig15]];
#endif
	stringflag=1;
#ifndef G__OLDIMPLEMENTATION1621
	if(0>size && -1==var->varlabel[ig15][1]) {
	  isauto=0;
	  size = 1;
	  stringflag=2;
	}
#endif
      }
#endif
      prev=pinc;
      if(inc) pinc = pinc - pinc%inc + inc;
      if(pinc>var->varlabel[ig15][1]) {
	if(isauto) {
	  var->varlabel[ig15][1] += var->varlabel[ig15][0];
	  if(G__PINVALID!=var->p[ig15]) {
	    tmp=(long)realloc((void*)var->p[ig15]
			      ,(size_t)(size*(var->varlabel[ig15][1]+1)));
	  }
	  else {
	    tmp=(long)malloc((size_t)(size*(var->varlabel[ig15][1]+1)));
	  }
	  if(tmp) var->p[ig15] = tmp;
	  else    G__malloc_error(new_name);
	}
#ifndef G__OLDIMPLEMENTATION1621
	else if(2==stringflag) {
	}
#endif
	else {
	  /*************************************
	   * error , array index out of range
	   ************************************/
#ifndef G__OLDIMPLEMENTATION877
	  if(G__ASM_FUNC_NOP==G__asm_wholefunction) {
#endif
#ifndef G__OLDIMPLEMENTATION1103
	    if(0==G__const_noerror) {
#endif
	      G__fprinterr(G__serr,
		 "Error: Array initialization out of range *(%s+%d), upto %d "
		 ,name,pinc ,var->varlabel[ig15][1]);
#ifndef G__OLDIMPLEMENTATION1103
	    }
#endif
#ifndef G__OLDIMPLEMENTATION877
	  }
#endif
	  G__genericerror((char*)NULL);
#ifndef G__OLDIMPLEMENTATION835
	  while(mparen--&&';'!=c) c=G__fignorestream("};"); 
	  if(';'!=c) c=G__fignorestream(";");
	  return(c);
#endif
	}
      }
      /*******************************************
       * initialized omitted objects to 0
       *******************************************/
      for(i=prev+1;i<pinc;i++) {
	buf.obj.i=var->p[ig15]+size*i;
	G__letvalue(&buf,G__null);
      }
      /*******************************************
       * initiazlize this element
       *******************************************/
      buf.obj.i=var->p[ig15]+size*pinc;
#ifndef G__OLDIMPLEMENTATION2125
      {
        int store_prerun=G__prerun;
        G__prerun=0;
        reg=G__getexpr(expr);
        G__prerun=store_prerun;
      }
#else
      reg=G__getexpr(expr);
#endif
#ifndef G__OLDIMPLEMENTATION1607
      if(
#ifndef G__OLDIMPLEMENTATION1621
	 1==
#endif
	 stringflag) {
	strcpy((char*)buf.obj.i,(char*)reg.obj.i);
      }
#ifndef G__OLDIMPLEMENTATION1621
      else if(2==stringflag && 0==var->p[ig15]) {
	var->varlabel[ig15][1]=strlen((char*)reg.obj.i);
	tmp=(long)malloc((size_t)(size*(var->varlabel[ig15][1]+1)));
	if(tmp) {
	  var->p[ig15] = tmp;
	  buf.obj.i = var->p[ig15];
	  strcpy((char*)buf.obj.i,(char*)reg.obj.i);
	}
	else    G__malloc_error(new_name);
      }
#endif
      else {
	G__letvalue(&buf,reg);
      }
#else
      G__letvalue(&buf,reg);
#endif
    }
    switch(c) {
    case '{':
      ++mparen;
#ifndef G__OLDIMPLEMENTATION1958
      if(stringflag && var->paran[ig15]>2) {
	inc *= var->varlabel[ig15][--pi]; /* not 100% sure,but.. */
      }
      else {
	inc *= var->varlabel[ig15][pi--];
      }
#else
      inc *= var->varlabel[ig15][pi--];
#endif
      break;
    case '}':
      ++pi;
      --mparen;
      break;
    case ',':
      inc=1;
      pi=pindex;
      break;
    }
  }
  
  /**********************************************************
   * initialize remaining object to 0
   **********************************************************/
#ifndef G__OLDIMPLEMENTATION1621
  if(0==stringflag)
#endif
#ifndef G__OLDIMPLEMENTATION1329
  {
    int initnum = var->varlabel[ig15][1];
    if(-1!=buf.typenum && G__newtype.nindex[buf.typenum]) {
      initnum /= size;
    }
    for(i=pinc+1;i<=initnum;i++) {
      buf.obj.i=var->p[ig15]+size*i;
      G__letvalue(&buf,G__null);
    }
  }
#else
  for(i=pinc+1;i<=var->varlabel[ig15][1];i++) {
    buf.obj.i=var->p[ig15]+size*i;
    G__letvalue(&buf,G__null);
  }
#endif
  
#ifndef G__OLDIMPLEMENTATION1535
  if(0==G__asm_noverflow && 1==G__no_exec_compile) {
    G__no_exec = 1;
  }
#endif
  /**********************************************************
   * read upto next , or ;
   **********************************************************/
  c=G__fignorestream(",;");
  /*  type var1[N] = { 0, 1, 2.. } , ... ;
   * came to                        ^  or ^
   */
  return(c);
  
}

/**************************************************************************
* G__ignoreinit()
**************************************************************************/
int G__ignoreinit(new_name)
char *new_name;
{
  int c;
  if(G__NOLINK==G__globalcomp) {
    G__fprinterr(G__serr,
    "Limitation: Initialization of class,struct %s ignored FILE:%s LINE:%d\n"
	    ,new_name,G__ifile.name,G__ifile.line_number);
  }

  c=G__fignorestream("}");
  /*  type var1[N] = { 0, 1, 2.. }  , ... ;
   * came to                      ^ */
  c=G__fignorestream(",;");
  /*  type var1[N] = { 0, 1, 2.. } , ... ;
   * came to                        ^  or ^ */
  return(c);
}

#ifndef G__OLDIMPLEMENTATION532
/**************************************************************************
* G__initmemvar()
**************************************************************************/
struct G__var_array* G__initmemvar(tagnum,pindex,pbuf)
int tagnum;
int* pindex;
G__value *pbuf;
{
  struct G__var_array* memvar;
  *pindex=0;
  if(-1!=tagnum) {
#ifndef G__OLDIMPLEMENTATION2131
    G__incsetup_memvar(tagnum);
#endif
    memvar=G__struct.memvar[tagnum];
    pbuf->tagnum=memvar->p_tagtable[*pindex];
    pbuf->typenum=memvar->p_typetable[*pindex];
    pbuf->type=toupper(memvar->type[*pindex]);
    pbuf->obj.reftype.reftype=memvar->reftype[*pindex];
    return(memvar);
  }
  else {
    return((struct G__var_array*)NULL);
  }
}

/**************************************************************************
* G__incmemvar()
**************************************************************************/
struct G__var_array* G__incmemvar(memvar,pindex,pbuf)
struct G__var_array* memvar;
int* pindex;
G__value *pbuf;
{
  /* increment memvar and index */
#ifndef G__OLDIMPLEMENTATION1468
  if(*pindex<memvar->allvar-1) {
#else
  if(*pindex<memvar->allvar) {
#endif
    ++(*pindex);
  }
  else {
    *pindex=0;
    if(memvar->next) {
      memvar=memvar->next;
    }
    else {
      memvar=(struct G__var_array*)NULL;
    }
  }
  if(memvar) {
    /* set assignment buffer */
    pbuf->tagnum=memvar->p_tagtable[*pindex];
    pbuf->typenum=memvar->p_typetable[*pindex];
    pbuf->type=toupper(memvar->type[*pindex]);
    pbuf->obj.reftype.reftype=memvar->reftype[*pindex];
  }
  return(memvar);
}

/**************************************************************************
* G__initstruct()
**************************************************************************/
int G__initstruct(new_name)
char *new_name;
{
  struct G__var_array *var;
  char name[G__MAXNAME];
  char expr[G__ONELINE];
  G__value buf;
  int c;
  char *p;
  int hash,i;
  int ig15;
  int pinc,pindex,pi,inc;
  int mparen;
  G__value reg;
  /* int ispointer=0; */
  int isauto=0;
  int size;
  int prev;
  long tmp;

  struct G__var_array *memvar;
  int memindex;
  /* int offset; */
  
  /* G__ASSERT(0==G__store_struct_offset); */

#ifdef G__ASM
  G__abortbytecode();
#endif
  
  /* separate variable name header */
  strcpy(name,new_name);
  p=strchr(name,'[');
  if(p) *p='\0';
  
  /* handling static declaration */
  if(G__static_alloc==1) {
    if(G__prerun==0) {
      /* ignore local static at run time */
      c=G__fignorestream("}");
      c=G__fignorestream(",;");
      return(c);
    }
    else if(G__func_now != -1) {
      /* local static at prerun, get global name */
      if(-1!=G__memberfunc_tagnum) /* questionable */
	sprintf(expr,"%s\\%x\\%x\\%x" ,name,G__func_page,G__func_now
		,G__memberfunc_tagnum);
      else
	sprintf(expr,"%s\\%x\\%x" ,name,G__func_page,G__func_now);
      strcpy(name,expr);
    }
  }

#ifndef G__OLDIMPLEMENTATION871
  p=strstr(name,"::");
  if(p) {
    int tagnum;
    struct G__var_array *memvar;
    *p='\0';
    p+=2;
    tagnum = G__defined_tagname(name,0);
    if(-1!=tagnum) {
      int store_memberfunc_tagnum=G__memberfunc_tagnum;
      int store_def_struct_member=G__def_struct_member;
      int store_exec_memberfunc=G__exec_memberfunc;
      int store_tagnum=G__tagnum;
      G__memberfunc_tagnum=tagnum;
      G__tagnum=tagnum;
      G__def_struct_member=0;
      G__exec_memberfunc=1;
      memvar=G__struct.memvar[tagnum];
      G__hash(p,hash,i)
      var = G__getvarentry(p,hash,&ig15,memvar,memvar);
      G__def_struct_member=store_def_struct_member;
      G__memberfunc_tagnum=store_memberfunc_tagnum;
      G__exec_memberfunc=store_exec_memberfunc;
      G__tagnum=store_tagnum;
    }
    else {
      var=(struct G__var_array*)NULL;
    }
  }
  else {
    /* calculate hash */
    G__hash(name,hash,i)
    /* get variable table entry */
    var = G__getvarentry(name,hash,&ig15,&G__global,G__p_local);
  }
#else
  /* calculate hash */
  G__hash(name,hash,i)
  /* get variable table entry */
  var = G__getvarentry(name,hash,&ig15,&G__global,G__p_local);
#endif
  

#ifndef G__OLDIMPLEMENTATION871
  if(!var) {
    G__fprinterr(G__serr,"Limitation: %s initialization ignored",name);
    G__printlinenum();
    c=G__fignorestream("},;");
    if('}'==c) c=G__fignorestream(",;");
    return(c);
  }
#endif

  if(G__struct.baseclass[var->p_tagtable[ig15]]->basen) {
    G__fprinterr(G__serr,"Error: %s must be initialized by a constructor",name);
    G__genericerror(NULL);
    c=G__fignorestream("}");
    /*  type var1[N] = { 0, 1, 2.. }  , ... ;
     * came to                      ^ */
    c=G__fignorestream(",;");
    /*  type var1[N] = { 0, 1, 2.. } , ... ;
     * came to                        ^  or ^ */
    return(c);
  }
  
  /*******************************************************
   * multidimensional array number of dimension
   *******************************************************/
  pindex=var->paran[ig15];
  
  /*******************************************************
   * check if  a[], a[][B][C] isauto sized array 
   *******************************************************/
  if(INT_MAX==var->varlabel[ig15][1]) {
    /* set isauto flag and reset varlabel[ig15][1] */
#ifndef G__OLDIMPLEMENTATION877
    if(G__asm_wholefunction) {
      G__abortbytecode();
      G__genericerror((char*)NULL);
    }
#endif
    isauto=1;
    var->varlabel[ig15][1] = -1;
    G__ASSERT(G__PINVALID==var->p[ig15] &&
	      G__COMPILEDGLOBAL==var->statictype[ig15]);
    if(G__static_alloc==1) {
      if(-1 != G__func_now) {
	var->statictype[ig15]=G__LOCALSTATICBODY;
      }
      else {
	var->statictype[ig15]=G__ifile.filenum;
      }
    }
    else {
      var->statictype[ig15]=G__AUTO;
    }
  }

  G__ASSERT(G__COMPILEDGLOBAL!=var->statictype[ig15]);
  
  /* initialize buf */
  buf.type=toupper(var->type[ig15]);
  buf.tagnum=var->p_tagtable[ig15];
  buf.typenum=var->p_typetable[ig15];
  buf.ref=0;
  buf.obj.reftype.reftype=var->reftype[ig15];
  
  /* getting size */
  if(islower(var->type[ig15])) {
    size=G__sizeof(&buf);
  }
  else {
    buf.type='L'; /* pointer assignement handled as long */
    size=G__LONGALLOC;
  }
  G__ASSERT(0<var->varlabel[ig15][0]&&0<size);
  
  
  /* initialize data member pointer */
  memvar=G__initmemvar(var->p_tagtable[ig15],&memindex,&buf);
  /*******************************************************
   * read initialization list 
   *******************************************************/
  mparen=1;
  inc=0;
  pi=pindex;
  pinc=0;
  while(mparen) {
    c=G__fgetstream(expr,",{}");
    if(expr[0]) {
      /********************************************
       * increment the pointer
       ********************************************/
      prev=pinc;
      if(inc) pinc = pinc - pinc%inc + inc;
      if(pinc>var->varlabel[ig15][1]) {
	if(isauto) {
	  var->varlabel[ig15][1] += var->varlabel[ig15][0];
	  if(G__PINVALID!=var->p[ig15]) {
	    tmp=(long)realloc((void*)var->p[ig15]
			      ,(size_t)(size*(var->varlabel[ig15][1]+1)));
	  }
	  else {
	    tmp=(long)malloc((size_t)(size*(var->varlabel[ig15][1]+1)));
	  }
	  if(tmp) var->p[ig15] = tmp;
	  else    G__malloc_error(new_name);
	}
	else {
	  /*************************************
	   * error , array index out of range
	   ************************************/
#ifndef G__OLDIMPLEMENTATION877
	  if(G__ASM_FUNC_NOP==G__asm_wholefunction) {
#endif
#ifndef G__OLDIMPLEMENTATION1103
	    if(0==G__const_noerror) {
#endif
	      G__fprinterr(G__serr,
		 "Error: Array initialization out of range *(%s+%d), upto %d "
		    ,name,pinc ,var->varlabel[ig15][1]);
#ifndef G__OLDIMPLEMENTATION1103
	    }
#endif
#ifndef G__OLDIMPLEMENTATION877
	  }
#endif
	  G__genericerror((char*)NULL);
	}
      }
      /*******************************************
       * initiazlize this element
       *******************************************/
      do {
        buf.obj.i=var->p[ig15]+size*pinc+memvar->p[memindex];
        reg=G__getexpr(expr);
        if(isupper(memvar->type[memindex])) {
          *(long *)(buf.obj.i)=(long)G__int(reg);
        }
#ifndef G__OLDIMPLEMENTATION1603
	else if('c'==memvar->type[memindex] && 
		0<memvar->varlabel[memindex][1] && '"'==expr[0]) {
	  if(memvar->varlabel[memindex][1]+1>(int)strlen((char*)reg.obj.i)) 
	    strcpy((char*)buf.obj.i,(char*)reg.obj.i);
	  else
	    strncpy((char*)buf.obj.i,(char*)reg.obj.i
		    ,memvar->varlabel[memindex][1]+1);
	}
#endif
        else {
          G__letvalue(&buf,reg);
        }
	memvar=G__incmemvar(memvar,&memindex,&buf);
        if('}'==c||!memvar) break;
        c=G__fgetstream(expr,",{}");
      } while(memvar);
      memvar=G__initmemvar(var->p_tagtable[ig15],&memindex,&buf);
    }
    switch(c) {
    case '{':
      ++mparen;
      /* inc *= var->varlabel[ig15][pi--]; */
      break;
    case '}':
      ++pi;
      --mparen;
      break;
    case ',':
      inc=1;
      pi=pindex;
      break;
    }
  }
  
  /**********************************************************
   * read upto next , or ;
   **********************************************************/
  c=G__fignorestream(",;");
  /*  type var1[N] = { 0, 1, 2.. } , ... ;
   * came to                        ^  or ^
   */
  return(c);
}
#endif /* ON532 */



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
