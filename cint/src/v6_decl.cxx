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
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "common.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" {

extern char G__declctor[];
extern int G__const_noerror;

int G__initval_eval = 0;
int G__dynconst = 0;

//______________________________________________________________________________
void G__loadlonglong(int* ptag, int* ptype, int which)
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

//______________________________________________________________________________
int G__get_newname(char* new_name)
{
  char temp[G__ONELINE],temp1[G__ONELINE];
  /* char *endmark=",;=():+-*%/^<>&|=![~@"; */
  int cin;
  int store_def_struct_member,store_tagdefining;

  cin=G__fgetvarname(new_name,"*&,;=():}");
  if (cin=='&') {
    if(0==strcmp(new_name,"operator")) {
      new_name[8] = cin;
      cin=G__fgetvarname(new_name+9,",;=():}");
    }
    else {
      strcat(new_name,"&");
      cin = ' ';
    }
  }
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
  if (isspace(cin)) {
    if(strcmp(new_name,"const*")==0) {
      new_name[0]='*';
      cin=G__fgetvarname(new_name+1,",;=():}");
      G__constvar |= G__CONSTVAR;
    }

    if(strcmp(new_name,"friend")==0) {
      store_def_struct_member=G__def_struct_member;
      store_tagdefining=G__tagdefining;
      G__def_struct_member = 0;
      G__tagdefining = -1;
      G__define_var(G__tagnum, G__typenum);
      G__def_struct_member = store_def_struct_member;
      G__tagdefining = store_tagdefining;
      new_name[0]='\0';
      return(';');
    }
    else if(strcmp(new_name,"&")==0 || strcmp(new_name,"*")==0) {
      cin=G__fgetvarname(new_name+1,",;=():");
    }
    else if(strcmp(new_name,"&*")==0 || strcmp(new_name,"*&")==0) {
      cin=G__fgetvarname(new_name+2,",;=():");
    }

    if(strcmp(new_name,"double")==0
       && 'l'!=G__var_type
       ) {
      cin=G__fgetvarname(new_name,",;=():");
      G__var_type='d';
    }
    else if(strcmp(new_name,"int")==0) {
      cin=G__fgetvarname(new_name,",;=():");
    }
    else if(strcmp(new_name,"long")==0 ||
            strcmp(new_name,"long*")==0 ||
            strcmp(new_name,"long**")==0 ||
            strcmp(new_name,"long&")==0) {
      int store_tagnum = G__tagnum;
      int store_typenum = G__typenum;
      int store_decl = G__decl;
      if(strcmp(new_name,"long")==0) {
        G__var_type='n' + G__unsigned;
        G__tagnum = -1;
        G__typenum = -1;
        G__reftype = G__PARANORMAL;
      }
      else if(strcmp(new_name,"long*")==0) {
        G__var_type='N' + G__unsigned;
        G__tagnum = -1;
        G__typenum = -1;
        G__reftype = G__PARANORMAL;
      }
      else if(strcmp(new_name,"long**")==0) {
        G__var_type='N' + G__unsigned;
        G__tagnum = -1;
        G__typenum = -1;
        G__reftype = G__PARAP2P;
      }
      else if(strcmp(new_name,"long&")==0) {
        G__var_type='n' + G__unsigned;
        G__tagnum = -1;
        G__typenum = -1;
        G__reftype = G__PARAREFERENCE;
      }
      G__define_var(G__tagnum, G__typenum);
      G__var_type='p';
      G__tagnum=store_tagnum;
      G__typenum=store_typenum;
      G__decl=store_decl;
      return(0);
    }
    else if(
            'l'==G__var_type &&
            (strcmp(new_name,"double")==0 ||
             strcmp(new_name,"double*")==0 ||
             strcmp(new_name,"double**")==0 ||
             strcmp(new_name,"double&")==0)) {
      int store_tagnum = G__tagnum;
      int store_typenum = G__typenum;
      int store_decl = G__decl;
      if(strcmp(new_name,"double")==0) {
        G__var_type='q';
        G__tagnum = -1;
        G__typenum = -1;
        G__reftype = G__PARANORMAL;
      }
      else if(strcmp(new_name,"double*")==0) {
        G__var_type='Q';
        G__tagnum = -1;
        G__typenum = -1;
        G__reftype = G__PARANORMAL;
      }
      else if(strcmp(new_name,"double**")==0) {
        G__var_type='Q';
        G__tagnum = -1;
        G__typenum = -1;
        G__reftype = G__PARAP2P;
      }
      else if(strcmp(new_name,"double&")==0) {
        G__var_type='q';
        G__tagnum = -1;
        G__typenum = -1;
        G__reftype = G__PARAREFERENCE;
      }
      G__define_var(G__tagnum, G__typenum);
      G__var_type='p';
      G__tagnum=store_tagnum;
      G__typenum=store_typenum;
      G__decl=store_decl;
      return(0);
    }
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
      G__reftype=G__PARAREFERENCE;
    }
    else if(strcmp(new_name,"double&")==0) {
      cin=G__fgetvarname(new_name,",;=():");
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
          if(strcmp(new_name,"*const")==0) {
            G__constvar |= G__PCONSTVAR;
            cin=G__fgetvarname(new_name+1,",;=():");
          }
        }
        if(isupper(G__var_type)) G__constvar |= G__PCONSTVAR;
        else                     G__constvar |= G__CONSTVAR;
      }
      else if(strcmp(new_name,"const&")==0) {
        cin=G__fgetvarname(new_name,",;=():");
        G__reftype=G__PARAREFERENCE;
        G__constvar |= G__PCONSTVAR;
      }
      else if(strcmp(new_name,"*const&")==0) {
        cin=G__fgetvarname(new_name+1,",;=():");
        G__constvar |= G__PCONSTVAR;
        G__reftype=G__PARAREFERENCE;
      }
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
      else if(strcmp(new_name,"virtual")==0) {
        G__virtual = 1;
        cin=G__fgetvarname(new_name,",;=():");
      }
    }

    if(isspace(cin)) {
      int store_len;
      if(strcmp(new_name,"operator")==0 ||
         strcmp(new_name,"*operator")==0||
         strcmp(new_name,"*&operator")==0||
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
        case ',':
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

      store_len = strlen(new_name);

      do {
        cin = G__fgetstream(new_name+strlen(new_name),",;=():");
        if(']'==cin) strcpy(new_name+strlen(new_name),"]");
      } while(']'==cin);

      if(store_len>1&&isalnum(new_name[store_len])&&
         isalnum(new_name[store_len-1])) {
        if(G__dispmsg>=G__DISPWARN) {
          G__fprinterr(G__serr,"Warning: %s  Syntax error??",new_name);
          G__printlinenum();
        }
      }

      return(cin);

    } /* of isspace(cin) */
  } /* of isspace(cin) */
  else if('('==cin && 0==new_name[0]) {
    // check which case
    //  1. f(type (*p)(int))  -> do nothing here
    //  2. f(type (*p)[4][4]) -> convert to f(type p[][4][4])
    fpos_t tmppos;
    int tmpline = G__ifile.line_number;;
    fgetpos(G__ifile.fp,&tmppos);
    if (G__dispsource) G__disp_mask=1;
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
  if (strncmp(new_name,"operator",8)==0 &&
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
    else if(','==cin && '\0'==new_name[8]) {
      cin=G__fgetstream(new_name,"(");
      sprintf(new_name,"operator,");
    }
    return(cin);
  }
  else if ((strncmp(new_name,"*operator",9)==0 ||
           strncmp(new_name,"&operator",9)==0) &&
          (G__isoperator(new_name[9]) || '\0'==new_name[9])) {
    if ('='==cin) {
      fseek(G__ifile.fp,-1,SEEK_CUR);
      if(G__dispsource) G__disp_mask=1;
      cin=G__fgetstream(new_name+strlen(new_name),"(");
    }
    else if ('('==cin && '\0'==new_name[9]) {
      cin=G__fignorestream(")");
      cin=G__fignorestream("(");
      strcpy(new_name+9,"()");
    }
    return cin;
  }
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
  return cin;
}

//______________________________________________________________________________
int G__unsignedintegral(int* pspaceflag, int* piout, int mparen)
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
  else {
    G__var_type='i'-1;
    fsetpos(G__ifile.fp,&pos);
  }

  G__define_var(-1, -1);

  G__reftype=G__PARANORMAL;
  G__unsigned = 0;
  *pspaceflag = -1;
  *piout = 0;

  if(mparen==0) return(1);
  else          return(0);
}

//______________________________________________________________________________
static struct G__var_array* G__rawvarentry(char* name, int hash, int* pig15, G__var_array* var)
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

//______________________________________________________________________________
static int G__setvariablecomment(char* new_name)
{
  struct G__var_array *var;
  int ig15;
  int i;
  int hash;
  char name[G__MAXNAME];
  char *p;
   unsigned int j,nest,scope;

  if('\0'==new_name[0]) return(0);

  strcpy(name,new_name);
  p=strchr(name,'[');
  if(p) *p='\0';

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
   }
   return(0);
}

//______________________________________________________________________________
void G__removespacetemplate(char* name)
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

//______________________________________________________________________________
void G__initstructary(char* new_name, int tagnum)
{
  // -- Initialize an array of structures.
  // 
  // A string[3] = { "abc", "def", "hij" };
  // A string[]  = { "abc", "def", "hij" };
  //                ^
  int cin = 0;
  long store_struct_offset = G__store_struct_offset;
  long store_globalvarpointer = G__globalvarpointer;
  char buf[G__ONELINE];
#ifdef G__ASM
  G__abortbytecode();
#endif
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
    } while (cin != '}');
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
  long adr = G__int(G__letvariable(new_name, reg, &G__global, G__p_local));
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
    reg = G__getfunction(buf, &known, G__CALLCONSTRUCTOR);
    ++i;
  } while (cin != '}');
  G__store_struct_offset = store_struct_offset;
  G__globalvarpointer = store_globalvarpointer;
}

//______________________________________________________________________________
void G__define_var(int tagnum, int typenum)
{
  // -- Declaration of variable, function or ANSI function header
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

  int  known;
  long  store_struct_offset; /* used to be int */
  int  store_prerun;
  int  store_debug=0,store_step=0;
  char temp1[G__LONGLINE];

  char new_name[G__LONGLINE],temp[G__LONGLINE];
  int staticclassobject=0;

  int store_var_type;
  int store_tagnum_default=0;
  int store_def_struct_member_default=0;
  int store_exec_memberfunc=0;
  int store_memberfunc_tagnum=0;
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


  /*
   * type  var1 , var2 ;
   *      ^
   *   or
   * type  int var1 , var2;
   *      ^
   * read variable name or 'int' identifier
   */
  cin=G__get_newname(new_name);
  G__unsigned=0; /* this is now reset in the G__exec_statement() */
  if(0==cin) {
    G__decl=store_decl;
    G__constvar=0;
    G__tagnum = store_tagnum;
    G__typenum = store_typenum;
    G__reftype=G__PARANORMAL;
    G__static_alloc=store_static_alloc2;
    G__dynconst=0;
    G__globalvarpointer = G__PVOID;
    return; /* long long handling */
  }
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
    if ('&'==new_name[0]) {
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
      char *pxx = strstr(new_name,"...");
      if(pxx) *pxx=0;

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
         == G__PARAREFERENCE
         ) {
        G__globalvarpointer = G__ansipara.ref;
        reg=G__null;
        if(G__globalvarpointer==(long)0 && 'u'==G__ansipara.type &&
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
          char tttt[G__ONELINE];
          G__valuemonitor(reg,tttt);
          sprintf(temp1,"%s(%s)",G__struct.name[tagnum],tttt);
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

#ifdef G__ASM
      if(0==new_name[0]) {
#ifdef G__ASM_DBG
        if(G__asm_dbg) G__fprinterr(G__serr,"%3x: POP\n",G__asm_cp);
#endif
        G__asm_inst[G__asm_cp] = G__POP;
        G__inc_cp_asm(1,0);
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
      G__dynconst=0;
      G__globalvarpointer = G__PVOID;
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
    if (cin=='(') {
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
        cin=G__fignorestream("=,;}");
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
        if(G__def_struct_member!=0
           && ( G__tagdefining == -1
           || G__struct.type[G__tagdefining] != 'n')
           ||
           ((!isdigit(cin))&&cin!='"'&&cin!='\''&&cin!='.'&&cin!='-'&&
            cin!='+'&&
            cin!='*'&&cin!='&')) {

          /* It is clear that above check is not sufficient to distinguish
           * class object instantiation and function header. Following
           * code is added to make it fully compliant to ANSI C++ */
          fgetpos(G__ifile.fp,&store_fpos);
          store_line = G__ifile.line_number;
          if(G__dispsource) G__disp_mask=1000;
          cin = G__fgetname(temp,",)*&<=");
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
          fsetpos(G__ifile.fp,&store_fpos);
          if(G__dispsource) G__disp_mask=1;
          G__ifile.line_number = store_line;

          if((!G__iscpp)||'\0'==temp[0]||
             -1==tagnum || /* this is a problem for 'int f(A* b);' */
             G__istypename(temp)||('\0'==temp[0]&&')'==cin)
             || 0==strncmp(new_name,"operator",8)
             || ('<'==cin&&G__defined_templateclass(temp))
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
            G__dynconst=0;
            G__globalvarpointer = G__PVOID;
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
        cin = G__fgetstream_newtemplate(temp,")");

        if('*'==new_name[0]&&var_type!='c'&&'"'==temp[0]) {
          G__genericerror("Error: illegal pointer initialization");
        }

        if(G__static_alloc&&0==G__prerun) {
          if(';'!=cin&&','!=cin) cin = G__fignorestream(",;");
          if('{'==cin) { /* don't know if this part is needed */
            while('}'!=cin) cin = G__fignorestream(";,");
          }
          G__var_type = var_type;
          G__letvariable(new_name,reg,&G__global,G__p_local);
          goto readnext;
        }

        if(-1==G__tagnum||'u'!=var_type||'*'==new_name[0]) {
          if(tolower(G__var_type)!='c' && strchr(temp,',')) {
            reg = G__null;
            G__genericerror("Error: Syntax error");
          }
          else {
            reg = G__getexpr(temp);
          }
          cin = G__fignorestream(",;");
          if(G__PARAREFERENCE==G__reftype && 0==G__asm_wholefunction) {
            if(0==reg.ref) {
              G__fprinterr(G__serr
                           ,"Error: reference type %s with no initialization "
                           ,new_name);
              G__genericerror((char*)NULL);
            }
            G__globalvarpointer = reg.ref;
          }
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
              G__dynconst=0;
              G__globalvarpointer = G__PVOID;
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
            G__dynconst=0;
            G__globalvarpointer = G__PVOID;
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
          G__static_alloc = store_static_alloc;
          G__prerun = store_prerun;
          G__cppconstruct = 1;
          if(G__globalvarpointer||G__no_exec_compile)
          {
            int store_constvar2 = G__constvar;
            G__constvar=store_constvar;
            G__letvariable(new_name,G__null,&G__global,G__p_local);
            G__constvar=store_constvar2;
          }
          else if(G__asm_wholefunction) {
            G__abortbytecode();
            G__asm_wholefunc_default_cp=0;
            G__no_exec=1;
            G__return=G__RETURN_NORMAL;
          }
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
            if(0==G__xrefflag) {
              G__fprinterr(G__serr,
                      "Error: %s not allocated(1), maybe duplicate declaration "
                      ,new_name);
            }
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
          G__dynconst=0;
          G__globalvarpointer = G__PVOID;
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
        if(i) {
          var_type = toupper(var_type);
          /* if(i>1) G__reftype = i+1;  not needed */
        }
        if(strchr(new_name+i,'<')) G__removespacetemplate(new_name+i);
        do {
          G__def_tagnum = G__defined_tagname(new_name+i,0) ;
          /* protect against a non defined tagname */
          if (G__def_tagnum<0) {
            /* Hopefully restore all values! */
            G__decl=store_decl;
            G__constvar=0;
            G__tagnum = store_tagnum;
            G__typenum = store_typenum;
            G__reftype=G__PARANORMAL;
            G__static_alloc=store_static_alloc2;
            G__dynconst=0;
            G__globalvarpointer = G__PVOID;
            G__def_struct_member = store_def_struct_member;
            return;
          }
          G__tagdefining  = G__def_tagnum;
          cin = G__fgetstream(new_name+i,"(=;:");
        } while(':'==cin && EOF!=(cin=G__fgetc())) ;
        temp[0]='\0';
        switch(cin) {
        case '=':
          if(strncmp(new_name+i,"operator",8)==0) {
            cin=G__fgetstream(new_name+strlen(new_name)+1,"(");
            new_name[strlen(new_name)] = '=';
            break;
          }
        case ';':
          /* PHILIPPE17: the following is fixed in 1306! */
          /* static class object member must call constructor
           * TO BE IMPLEMENTED */
          sprintf(temp,"%s::%s",G__fulltagname(G__def_tagnum,1),new_name+i);
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
        G__tagdefining = store_tagdefining; /* FIX */
        G__dynconst=0;
        G__globalvarpointer = G__PVOID;
        return;
      }
      else {
        fseek(G__ifile.fp,-1,SEEK_CUR);
        if(cin=='\n' /* ||cin=='\r' */ ) --G__ifile.line_number;
        if(G__dispsource) G__disp_mask=1;
      }


      if(G__globalcomp!=G__NOLINK) {
        if(0==bitfieldwarn) {
          if(G__dispmsg>=G__DISPNOTE) {
            G__fprinterr(G__serr,"Note: Bit-field not accessible from interpreter");
            G__printlinenum();
          }
          bitfieldwarn=1;
        }
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
      int store_tagnumB=G__tagnum;
      G__tagnum = G__get_envtagnum();
      if('u'==var_type)
        cin=G__fgetstream_newtemplate(temp,",;{}"); /* TEMPLATECLASS case12 */
      else
        cin=G__fgetstream_new(temp,",;{");

      if(G__def_struct_member && G__CONSTVAR!=G__constvar && G__static_alloc &&
         -1!=G__tagdefining &&
         ('c'==G__struct.type[G__tagdefining]||
          's'==G__struct.type[G__tagdefining])) {
        if(G__dispmsg>=G__DISPWARN) {
          G__fprinterr(G__serr,"Warning: In-class initialization of non-const static member not allowed in C++ standard");
          G__printlinenum();
        }
      }

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
         == G__PARAREFERENCE
        ) {
        int store_reftype = G__reftype;
        /*#define G__OLDIMPLEMENTATION1093*/
        int store_prerun=G__prerun;
        int store_decl=G__decl;
        int store_constvar=G__constvar;
        int store_static_alloc=G__static_alloc;
        if(G__NOLINK==G__globalcomp) {
          G__prerun=0;
          G__decl=0;
          if(G__CONSTVAR&G__constvar) G__initval_eval=1;
          G__constvar=0;
          G__static_alloc=0;
        }
        --G__templevel;
        G__reftype=G__PARANORMAL;
        if(store_prerun||0==store_static_alloc||G__IsInMacro()) {
          reg=G__getexpr(temp);
        }
        else reg=G__null;
        ++G__templevel;
        G__prerun=store_prerun;
        G__decl=store_decl;
        G__constvar=store_constvar;
        G__static_alloc=store_static_alloc;
        G__initval_eval=0;
        G__reftype=store_reftype;
        G__globalvarpointer = reg.ref;
        reg=G__null;
        if(G__globalvarpointer==(long)0 && 'u'==G__ansipara.type &&
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
          int store_prerun=G__prerun;
          int store_decl=G__decl;
          int store_constvar=G__constvar;
          int store_static_alloc=G__static_alloc;
          if(G__NOLINK==G__globalcomp) {
            G__prerun=0;
            G__decl=0;
            if(G__CONSTVAR&G__constvar) G__initval_eval=1;
            G__constvar=0;
            G__static_alloc=0;
          }
          if(store_prerun||0==store_static_alloc||G__IsInMacro()) {
            reg=G__getexpr(temp);
          }
          else reg=G__null;
          G__prerun=store_prerun;
          G__decl=store_decl;
          G__constvar=store_constvar;
          G__static_alloc=store_static_alloc;
          G__initval_eval=0;
          if('U'!=reg.type && 'Y'!=reg.type && 0!=reg.obj.i) {
            G__assign_error(new_name+1,&reg);
            reg = G__null;
          }
        }
        else {
          int store_prerun=G__prerun;
          int store_decl=G__decl;
          int store_constvar=G__constvar;
          int store_static_alloc=G__static_alloc;
          if(G__NOLINK==G__globalcomp) {
            G__prerun=0;
            G__decl=0;
            if(G__CONSTVAR&G__constvar) G__initval_eval=1;
            G__constvar=0;
            G__static_alloc=0;
          }
          if(store_prerun||0==store_static_alloc||G__IsInMacro()) {
            /* int store_tagnumC = G__tagnum; */
            /* int store_def_tagnumC = G__def_tagnum; */
            int store_tagdefiningC = G__tagdefining;
            int store_eval_localstatic = G__eval_localstatic;
            G__eval_localstatic=1;
            reg=G__getexpr(temp);
            G__eval_localstatic=store_eval_localstatic;
            /* G__tagnum = store_tagnumC; shouldn't do this */
            /* G__def_tagnum = store_def_tagnumC; shouldn't do this */
            G__tagdefining = store_tagdefiningC;
          }
          else reg=G__null;
          G__prerun=store_prerun;
          G__decl=store_decl;
          G__constvar=store_constvar;
          G__static_alloc=store_static_alloc;
          G__initval_eval=0;
          if('u'==var_type&&'*'==new_name[0]&&'U'!=reg.type&&0!=reg.obj.i
             && 'Y'!=reg.type) {
            G__assign_error(new_name+1,&reg);
            reg = G__null;
          }
        }
      }
      G__tagnum = store_tagnumB;
    }
    else {
      if(
         '\0'!=new_name[0] &&
         G__NOLINK==G__globalcomp &&
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
      if (
        (var_type == 'u') && // class, enum, namespace, struct, or union, and
        (new_name[0] != '*') && // not a pointer, and
        (G__reftype == G__PARANORMAL) && // not a reference, and
        (
          !G__def_struct_member || // not a member, or
          (G__def_tagnum == -1) || // FIXME: This is probably meant to protect the next check, it cannot happen, or can it?
          (G__struct.type[G__def_tagnum] == 'n') // is a member of a namespace
        )
      ) {

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
              G__dynconst=0;
              G__globalvarpointer = G__PVOID;
              return;
            }
          }
        }

        if(G__static_alloc&&0==G__prerun) {
          if('{'==cin) {
            while('}'!=cin) cin = G__fignorestream(";,");
          }
          if(';'!=cin&&','!=cin) cin = G__fignorestream(";,");
          G__var_type = var_type;
          G__letvariable(new_name,reg,&G__global,G__p_local);
          goto readnext;
        }

        if(initary && strchr(new_name,'[') &&
           (G__struct.funcs[G__tagnum]&G__HAS_CONSTRUCTOR)) {

          store_prerun = G__prerun;
          if(G__NOLINK==G__globalcomp && G__func_now==-1) G__prerun = 0; // Do run constructors
          G__initstructary(new_name,G__tagnum);
          G__decl=store_decl;
          G__constvar=0;
          G__tagnum = store_tagnum;
          G__typenum = store_typenum;
          G__reftype=G__PARANORMAL;
          G__static_alloc=store_static_alloc2;
          G__dynconst=0;
          G__globalvarpointer = G__PVOID;
          G__prerun = store_prerun;
          return;
        }

        /************************************************************
        * memory allocation and table entry generation
        ************************************************************/
        store_struct_offset = G__store_struct_offset;
        if(G__CPPLINK!=G__struct.iscpplink[tagnum]) {
          /* allocate memory area for constructed object by interpreter */
          G__var_type = var_type;
          G__decl_obj=1;
          G__store_struct_offset=G__int(G__letvariable(new_name,reg,&G__global
                                                       ,G__p_local));
          G__decl_obj=0;
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
          G__dynconst=0;
          G__globalvarpointer = G__PVOID;
          return;
        }
        G__prerun = 0; /* FOR RUNNING CONSTRUCTOR */

        if (G__store_struct_offset) {
          if ((temp[0] == '\0') && (G__tagnum != -1)) {
            /********************************************
             * type a;
             * call default constructor
             ********************************************/
            sprintf(temp, "%s()", G__struct.name[G__tagnum]);
            if (G__dispsource) {
                G__fprinterr(G__serr, "\n!!!Calling default constructor 0x%lx.%s for declaration of %s", G__store_struct_offset, temp, new_name);
            }
            /******************************************************
            * Calling constructor to array of object
            ******************************************************/
            G__decl = 0;
            if ((index = strchr(new_name, '['))) {
              p_inc = G__getarrayindex(index);
              if (G__CPPLINK == G__struct.iscpplink[tagnum]) {
                // -- Precompiled class. First, call constructor (new) function.
#ifdef G__ASM
                if (G__asm_noverflow && (p_inc > 1)) {
#ifdef G__ASM_DBG
                  if (G__asm_dbg) G__fprinterr(G__serr, "%3x: SETARYINDEX\n", G__asm_cp);
#endif
                  G__asm_inst[G__asm_cp] = G__SETARYINDEX;
                  G__asm_inst[G__asm_cp+1] = 0;
                  G__inc_cp_asm(2, 0);
                }
#endif
                G__cpp_aryconstruct = p_inc;
                reg = G__getfunction(temp, &known, G__CALLCONSTRUCTOR);
                G__cpp_aryconstruct = 0;
                // Register the pointer we get from new to member variable table.
                G__globalvarpointer = G__int(reg);
                G__cppconstruct = 1;
                G__var_type = var_type;
                G__letvariable(new_name, G__null, &G__global, G__p_local);
                G__cppconstruct = 0;
#ifdef G__ASM
                if (G__asm_noverflow && (p_inc > 1)) {
#ifdef G__ASM_DBG
                  if (G__asm_dbg) {
                    G__fprinterr(G__serr, "%3x: RESETARYINDEX\n", G__asm_cp);
                  }
#endif
                  G__asm_inst[G__asm_cp] = G__RESETARYINDEX;
                  G__asm_inst[G__asm_cp+1] = 0;
                  G__inc_cp_asm(2, 0);
                }
#endif
                G__globalvarpointer = G__PVOID;
#ifndef G__OLDIMPLEMENTATION1073
                if (G__asm_wholefunction && G__no_exec_compile) {
#ifdef G__ASM_DBG
                  if (G__asm_dbg) {
                    G__fprinterr(G__serr, "%3x: SETGVP -1\n", G__asm_cp);
                  }
#endif
                  G__asm_inst[G__asm_cp] = G__SETGVP;
                  G__asm_inst[G__asm_cp+1] = -1;
                  G__inc_cp_asm(2, 0);
                }
#endif
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
                  if ((G__return > G__RETURN_NORMAL) || (known == 0)) {
                    break;
                  }
                  G__store_struct_offset += G__struct.size[G__tagnum];
                  if (G__asm_noverflow) {
#ifdef G__ASM_DBG
                    if (G__asm_dbg) G__fprinterr(G__serr, "%3x: ADDSTROS %d\n", G__asm_cp, G__struct.size[G__tagnum]);
#endif
                    G__asm_inst[G__asm_cp] = G__ADDSTROS;
                    G__asm_inst[G__asm_cp+1] = G__struct.size[G__tagnum];
                    G__inc_cp_asm(2, 0);
                  }
#ifndef G__OLDIMPLEMENTATION1073
                  if (G__asm_wholefunction && G__asm_noverflow) {
#ifdef G__ASM_DBG
                    if (G__asm_dbg) G__fprinterr(G__serr,"%3x: ADDSTROS %d\n", G__asm_cp, G__struct.size[G__tagnum]);
#endif
                    G__asm_inst[G__asm_cp] = G__POPSTROS; // ??? ADDSTROS
                    G__asm_inst[G__asm_cp+1] = G__struct.size[G__tagnum];
                    G__inc_cp_asm(2, 0);
                  }
#endif
                }
#ifndef G__OLDIMPLEMENTATION1073
                if (G__asm_wholefunction && G__asm_noverflow) {
#ifdef G__ASM_DBG
                  if (G__asm_dbg) {
                    G__fprinterr(G__serr, "%3x: POPSTROS\n", G__asm_cp);
                  }
#endif
                  G__asm_inst[G__asm_cp] = G__POPSTROS;
                  G__inc_cp_asm(1, 0);
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
                if((known && (G__globalvarpointer||G__asm_noverflow))
                   || G__NOLINK != G__globalcomp
                   ) {
                  G__letvariable(new_name,G__null,&G__global,G__p_local);
                }
                else {
                  if(0==G__xrefflag) {
                    if(G__ASM_FUNC_NOP==G__asm_wholefunction)
                      G__fprinterr(G__serr,"Error: %s no default constructor",temp);
                    G__genericerror((char*)NULL);
                  }
                  else {
                    G__letvariable(new_name,G__null,&G__global,G__p_local);
                  }
                }
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
                if(G__struct.isctor[tagnum])
                  G__getfunction(temp,&known,G__CALLCONSTRUCTOR);
                else
                  G__getfunction(temp,&known,G__TRYCONSTRUCTOR);
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
              G__dynconst=0;
              G__globalvarpointer = G__PVOID;
              return;
            }
            /* struct class initialization ={x,y,z} */
            if(initary) {
              if(known
                 && (G__struct.funcs[tagnum]& G__HAS_XCONSTRUCTOR)
                 /* && (G__struct.funcs[tagnum]& G__HAS_DEFAULTCONSTRUCTOR) */
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
                  G__prerun = store_prerun;
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
            int flag = 0;
            if (staticclassobject) {
              // to pass G__getfunction()
              G__tagnum = store_tagnum;
            }
            sprintf(temp1, "%s(", G__struct.name[G__tagnum]);
            // FIXME: ifdef G__TEMPLATECLASS: Need to evaluate template argument list here.
            if (temp == strstr(temp, temp1)) {
              int c;
              int isrc = 0;
              char buf[G__LONGLINE];
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
            else if (G__struct.istypedefed[G__tagnum]) {
              index = strchr(temp, '(');
              if (index) {
                *index = '\0';
                flag = G__defined_typename(temp);
                if ((flag != -1) && (G__newtype.tagnum[flag] == G__tagnum)) {
                  sprintf(temp1, "%s(%s", G__struct.name[G__tagnum], index + 1);
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
              // Call explicit constructor, error if no constructor.
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
                /* There are similar cases above, but they are either
                 * default ctor or precompiled class which should be fine */
                int store_static_alloc3=G__static_alloc;
                G__static_alloc=0;
                G__getfunction(temp,&known,G__CALLCONSTRUCTOR);
                G__static_alloc=store_static_alloc3;
              }
              G__decl=1;
              if(G__return>G__RETURN_NORMAL) {
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
            }
            else {
              int store_var_typeB,store_tagnumB,store_typenumB;
              long store_struct_offsetB=G__store_struct_offset;
              /* int store_def_tagnumB = G__def_tagnum; shouldn't do this */
              int store_tagdefiningB = G__tagdefining;
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
              reg=G__getexpr(temp);
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
              G__var_type=store_var_typeB;
              G__tagnum = store_tagnumB;
              G__typenum = store_typenumB;
              /* G__def_tagnum = store_def_tagnumB; shouldn't do this */
              G__tagdefining = store_tagdefiningB;
              if(G__CPPLINK==G__struct.iscpplink[tagnum]) {
                if(reg.tagnum==tagnum && 'u'==reg.type) {
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
                  G__valuemonitor(reg,tttt);
                  sprintf(temp,"%s(%s)",G__struct.name[tagnum],tttt);
                }
#ifndef G__OLDIMPLEMENTATION1073
                if(G__asm_wholefunction) {
                  G__oprovld=1;
                }
#endif
                G__oprovld=1;
                G__decl=0;
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
                G__globalvarpointer=G__int(reg);
                G__cppconstruct=1;
                G__letvariable(new_name,G__null,&G__global,G__p_local);
                G__cppconstruct=0;
                G__globalvarpointer=G__PVOID;
                G__oprovld=0;
                G__decl=1;
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
                G__dynconst=0;
                G__globalvarpointer = G__PVOID;
                G__prerun = store_prerun;
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
              G__prerun = store_prerun;
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
#ifdef G__ASM
        if(G__asm_noverflow) {
#ifdef G__ASM_DBG
          if(G__asm_dbg) G__fprinterr(G__serr,"%3x: POPSTROS\n",G__asm_cp);
#endif
          G__asm_inst[G__asm_cp] = G__POPSTROS;
          G__inc_cp_asm(1,0);
        }
#endif
      } /* of if(var_type=='u'&&G__def_struct_member.... */

      /**************************************************************
      * declaration of scaler object, pointer or reference type.
      **************************************************************/
      else {
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
        G__letvariable(new_name,reg,&G__global,G__p_local);
        if(G__return>G__RETURN_NORMAL) {
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
        /* insert array initialization */
        if(initary) {
          cin=G__initary(new_name);
          initary=0;
          if(EOF==cin) {
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
    G__globalvarpointer = G__PVOID;
    /***************************************************************
     * end of declaration or read next variable name
     *
     ***************************************************************/
    readnext:
    if (cin==';') {
      /* type  var1 , var2 ;
       *                    ^
       *  end of declaration, return
       */
      G__decl=store_decl;
      G__constvar=0;
      G__tagnum = store_tagnum;
      G__typenum = store_typenum;
      G__reftype=G__PARANORMAL;

      if(G__fons_comment && G__def_struct_member) {
        G__setvariablecomment(new_name);
      }

#ifdef G__ASM
        if(G__asm_noverflow) G__asm_clear();
#endif

      G__static_alloc=store_static_alloc2;
      G__dynconst=0;
      G__globalvarpointer = G__PVOID;
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
      G__dynconst=0;
      G__globalvarpointer = G__PVOID;
      return;
    }
    else {
      /* type  var1 , var2 , var3 ;
       *             ^  or  ^
       *  read variable name
       */
      cin=G__fgetstream(new_name,",;=():");
      if(EOF==cin) {
        G__decl=store_decl;
        G__constvar=0;
        G__tagnum = store_tagnum;
        G__typenum = store_typenum;
        G__reftype=G__PARANORMAL;
        fseek(G__ifile.fp,-1,SEEK_CUR);
        G__missingsemicolumn(new_name);
        G__static_alloc=store_static_alloc2;
        G__dynconst=0;
        G__globalvarpointer = G__PVOID;
        return;
      }
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

//______________________________________________________________________________
int G__initary(char* new_name)
{
  // -- Parse and execute an array initialization.
  //
  static char expr[G__ONELINE];
  // Separate the array name from the index specification.
  char name[G__MAXNAME];
  std::strcpy(name, new_name);
  {
    char* p = std::strchr(name, '[');
    if (p) {
      *p = '\0';
    }
  }
  // Handle a static array initialization.
  if ((G__static_alloc == 1) && !G__prerun) {
      // -- A local static array initialization at runtime.
      // Get variable table entry.
      int hash = 0;
      int i = 0;
      G__hash(name, hash, i)
      int varid = 0;
      struct G__var_array* var = G__getvarentry(name, hash, &varid, &G__global, G__p_local);
      if (var && (var->varlabel[varid][1] /* number of elements */ == INT_MAX /* unspecified length flag */)) {
        // -- Variable exists and is an unspecified length array.
        char namestatic[G__ONELINE];
        if (G__memberfunc_tagnum != -1) { // questionable
          sprintf(namestatic, "%s\\%x\\%x\\%x", name, G__func_page, G__func_now, G__memberfunc_tagnum);
        }
        else {
          sprintf(namestatic, "%s\\%x\\%x", name, G__func_page, G__func_now);
        }
        int hashstatic = 0;
        G__hash(namestatic, hashstatic, i)
        int ig15static = 0;
        struct G__var_array* varstatic = G__getvarentry(namestatic, hashstatic, &ig15static, &G__global, G__p_local);
        if (varstatic) {
          for (int i = 0; i < G__MAXVARDIM; ++i) {
            var->varlabel[varid][i] = varstatic->varlabel[ig15static][i];
          }
        }
      }
      // Ignore a local static array initialization at runtime.
      int c = G__fignorestream("}");
      c = G__fignorestream(",;");
      return c;
  }
  if ((G__static_alloc == 1) && (G__func_now != -1)) {
    // -- Function-local static array initialization at prerun, use a special global variable name.
    if (G__memberfunc_tagnum != -1) { // questionable
      std::sprintf(expr, "%s\\%x\\%x\\%x", name, G__func_page, G__func_now, G__memberfunc_tagnum);
    }
    else {
      std::sprintf(expr, "%s\\%x\\%x", name, G__func_page, G__func_now);
    }
    std::strcpy(name, expr);
  }
#ifdef G__ASM
  G__abortbytecode();
#endif
  //
  // Lookup the variable.
  //
  struct G__var_array* var = 0;
  int varid = 0;
  {
    char* p = G__strrstr(name, "::");
    if (p && G__prerun && (G__func_now == -1)) {
      // -- Qualified name, do the lookup in the specified context.
      *p = '\0';
      p += 2;
      int tagnum = G__defined_tagname(name, 0);
      if (tagnum != -1) {
        struct G__var_array* memvar = G__struct.memvar[tagnum];
        int hash = 0;
        int i = 0;
        G__hash(p, hash, i)
        var = G__getvarentry(p, hash, &varid, memvar, memvar);
      }
    }
    else {
      // -- Unqualified name, do a lookup.
      int hash = 0;
      int i = 0;
      G__hash(name, hash, i)
      var = G__getvarentry(name, hash, &varid, &G__global, G__p_local);
    }
  }
  if (!var) {
    char* px = std::strchr(name, '\\');
    if (px) {
      *px = 0;
    }
    char temp[G__ONELINE];
    if (G__tagdefining != -1) {
      sprintf(temp, "%s\\%x\\%x\\%x", name, G__func_page, G__func_now, G__tagdefining);
    }
    else {
      sprintf(temp, "%s\\%x\\%x", name, G__func_page, G__func_now);
    }
    int varhash = 0;
    int itmpx = 0;
    G__hash(temp, varhash, itmpx);
    var = G__getvarentry(temp, varhash, &varid, &G__global, G__p_local);
    if (!var && (G__tagdefining != -1)) {
      std::sprintf(temp, "%s", name);
      G__hash(temp, varhash, itmpx);
      var = G__getvarentry(temp, varhash, &varid, G__struct.memvar[G__tagdefining], G__struct.memvar[G__tagdefining]);
    }
    if (!var) {
      int c = G__fignorestream(",;");
      G__genericerror("Error: array initialization");
      return c;
    }
  }
  // Get number of dimensions.
  const short num_of_dimensions = var->paran[varid];
  int& num_of_elements = var->varlabel[varid][1];
  const int stride = var->varlabel[varid][0];
  // Check for an unspecified length array.
  int isauto = 0;
  if (num_of_elements == INT_MAX /* unspecified length flag */) {
    // -- Set isauto flag and reset number of elements.
    isauto = 1;
    num_of_elements = 0;
    if ((var->tagnum != -1) && (var->statictype[varid] == G__LOCALSTATIC)) {
      G__ASSERT(!var->p[varid] && G__prerun && (G__func_now == -1));
    }
    else {
      G__ASSERT(!var->p[varid] && (var->statictype[varid] == G__COMPILEDGLOBAL));
      if (G__static_alloc == 1) {
        if (G__func_now != -1) {
          var->statictype[varid] = G__LOCALSTATICBODY;
        }
        else {
          var->statictype[varid] = G__ifile.filenum;
        }
      }
      else {
        var->statictype[varid] = G__AUTO;
      }
    }
  }
  G__ASSERT(var->statictype[varid] != G__COMPILEDGLOBAL);
  // Initialize buf.
  G__value buf;
  buf.type = std::toupper(var->type[varid]);
  buf.tagnum = var->p_tagtable[varid];
  buf.typenum = var->p_typetable[varid];
  buf.ref = 0;
  buf.obj.reftype.reftype = var->reftype[varid];
  // Get size.
  int size = 0;
  int typedefary = 0;
  if (std::islower(var->type[varid])) {
    // -- We are *not* a pointer.
    if ((buf.typenum != -1) && G__newtype.nindex[buf.typenum]) {
      // -- We are a typedef, get the size of the actual type.
      char store_var_type = G__var_type;
      size = G__Lsizeof(G__newtype.name[buf.typenum]);
      G__var_type = store_var_type;
      typedefary = 1;
    }
    else {
      // -- We are *not* a typedef, get the size.
      size = G__sizeof(&buf);
    }
  }
  else {
    // -- We are a pointer, handle as a long.
    buf.type = 'L';
    size = G__LONGALLOC;
  }
  G__ASSERT((stride > 0) && (size > 0));
  //
  // Read and execute the intializer specification.
  //
  int mparen = 1;
  int inc = 0;
  int pi = num_of_dimensions;
  int linear_index = 0;
  int stringflag = 0;
  while (mparen) {
    // -- Read the next initializer value.
    int c = G__fgetstream(expr, ",{}");
    if (expr[0]) {
      // -- Found one.
      // increment the pointer
      if ((var->type[varid] == 'c') && (expr[0] == '"')) {
        if (!typedefary) {
          size = var->varlabel[varid][var->paran[varid]];
        }
        stringflag = 1;
        if ((size < 0) && !num_of_elements) {
          isauto = 0;
          size = 1;
          stringflag = 2;
        }
      }
      int prev = linear_index;
      if (inc) {
        linear_index = (linear_index - (linear_index % inc)) + inc;
      }
      // Make sure we have not gone beyond the end of the array.
      if (linear_index >= num_of_elements) {
        // -- We have gone past the end of the array.
        if (isauto) {
          // -- Unspecified length array, make it bigger to fit.
          // Allocate another stride worth of elements.
          num_of_elements += stride;
          long tmp = 0L;
          if (var->p[varid]) {
            // -- We already had some elements, resize.
            tmp = (long) std::realloc((void*) var->p[varid], size * num_of_elements);
          }
          else {
            // -- No elements allocate yet, get some.
            tmp = (long) std::malloc(size * num_of_elements);
          }
          if (tmp) {
            var->p[varid] = tmp;
          }
          else {
            G__malloc_error(new_name);
          }
        }
        else if (stringflag == 2) {
        }
        else {
          // -- Fixed-size array, error, array index out of range.
          if (G__asm_wholefunction == G__ASM_FUNC_NOP) {
            if (!G__const_noerror) {
              G__fprinterr(G__serr, "Error: v6_decl.cxx(G__initary, 2734): Array initialization out of range *(%s+%d), upto %d ", name, linear_index, num_of_elements);
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
      // Initialize omitted elements to 0.
      for (int i = prev + 1; i < linear_index; ++i) {
        buf.obj.i = var->p[varid] + i * size;
        G__letvalue(&buf, G__null);
      }
      // Initialize this element.
      G__value reg;
      buf.obj.i = var->p[varid] + linear_index * size;
      {
        int store_prerun = G__prerun;
        G__prerun = 0;
        reg = G__getexpr(expr);
        G__prerun = store_prerun;
      }
      if (stringflag == 1) {
        std::strcpy((char*) buf.obj.i, (char*) reg.obj.i);
      }
      else if ((stringflag == 2) && !var->p[varid]) {
        var->varlabel[varid][1] = std::strlen((char*) reg.obj.i);
        long tmp = (long) std::malloc((size_t) (size * var->varlabel[varid][1]));
        if (tmp) {
          var->p[varid] = tmp;
          buf.obj.i = var->p[varid];
          std::strcpy((char*) buf.obj.i, (char*) reg.obj.i);
        }
        else {
          G__malloc_error(new_name);
        }
      }
      else {
        G__letvalue(&buf, reg);
      }
    }
    switch (c) {
    case '{':
      ++mparen;
      if (stringflag && (var->paran[varid] > 2)) {
        // not 100% sure, but ...
        inc *= var->varlabel[varid][--pi];
      }
      else {
        inc *= var->varlabel[varid][pi--];
      }
      break;
    case '}':
      ++pi;
      --mparen;
      break;
    case ',':
      inc = 1;
      pi = num_of_dimensions;
      break;
    }
  }
  // Initialize remaining elements to 0.
  if (!stringflag)
  {
    int initnum = num_of_elements;
    if ((buf.typenum != -1) && G__newtype.nindex[buf.typenum]) {
      // -- We are a typedef.
      // FIXME: This is wrong! We don't need to scale it!
      initnum /= size;
    }
    for (int i = linear_index + 1; i < initnum; ++i) {
      buf.obj.i = var->p[varid] + (i * size);
      G__letvalue(&buf, G__null);
    }
  }
  if (!G__asm_noverflow && (G__no_exec_compile == 1)) {
    // FIXME: Why?
    G__no_exec = 1;
  }
  // Read and discard up to the next ',' or ';'.
  int c = G__fignorestream(",;");
  //  type var1[N] = { 0, 1, 2.. } , ... ;
  // came to                        ^  or ^
  return c;
}

//______________________________________________________________________________
int G__ignoreinit(char* new_name)
{
  if (G__globalcomp == G__NOLINK) {
    G__fprinterr(G__serr, "Limitation: Initialization of class,struct %s ignored FILE:%s LINE:%d\n", new_name, G__ifile.name, G__ifile.line_number);
  }
  int c = G__fignorestream("}");
  //  type var1[N] = { 0, 1, 2.. }  , ... ;
  // came to                      ^
  c = G__fignorestream(",;");
  //  type var1[N] = { 0, 1, 2.. } , ... ;
  // came to                        ^  or ^
  return c;
}

//______________________________________________________________________________
struct G__var_array* G__initmemvar(int tagnum, int* pindex, G__value* pbuf)
{
  // -- Get pointer to first member variable of a structure.
  // Reset variable page index.
  *pindex = 0;
  if (tagnum != -1) {
    // -- We were given a valid tag, do work.
    // Do dictionary setup if needed.
    // Note: This is a delayed-initialization optimization.
    G__incsetup_memvar(tagnum);
    struct G__var_array* memvar = G__struct.memvar[tagnum];
    pbuf->tagnum = memvar->p_tagtable[*pindex];
    pbuf->typenum = memvar->p_typetable[*pindex];
    pbuf->type = std::toupper(memvar->type[*pindex]);
    pbuf->obj.reftype.reftype = memvar->reftype[*pindex];
    return memvar;
  }
  return 0;
}

//______________________________________________________________________________
G__var_array* G__incmemvar(G__var_array* memvar, int* pindex, G__value* pbuf)
{
  // -- Get pointer to next member variable in a structure.
  if (*pindex < (memvar->allvar - 1)) {
    // -- Increment index in page.
    ++(*pindex);
  }
  else {
    // -- Next page of variables.
    *pindex = 0;
    memvar = memvar->next;
  }
  if (memvar) {
    // -- We have another member variable, copy info into value buffer.
    pbuf->tagnum = memvar->p_tagtable[*pindex];
    pbuf->typenum = memvar->p_typetable[*pindex];
    pbuf->type = toupper(memvar->type[*pindex]);
    pbuf->obj.reftype.reftype = memvar->reftype[*pindex];
  }
  return memvar;
}

//______________________________________________________________________________
int G__initstruct(char* new_name)
{
  // FIXME: We do not handle brace nesting properly,
  //        we need to default initialize members
  //        whose initializers were omitted.
  char expr[G__ONELINE];
#ifdef G__ASM
  G__abortbytecode();
#endif
  // Separate the variable name from any index specification.
  char name[G__MAXNAME];
  std::strcpy(name, new_name);
  {
    char* p = std::strchr(name, '[');
    if (p) {
      *p = '\0';
    }
  }
  if ((G__static_alloc == 1) && !G__prerun) {
    // -- Ignore a local static structure initialization at runtime.
    int c = G__fignorestream("}");
    c = G__fignorestream(",;");
    return c;
  }
  if ((G__static_alloc == 1) && (G__func_now != -1)) {
    // -- Function-local static structure initialization at prerun, use a special global variable name.
    if (G__memberfunc_tagnum != -1) { // questionable
      std::sprintf(expr, "%s\\%x\\%x\\%x", name, G__func_page, G__func_now, G__memberfunc_tagnum);
    }
    else {
      std::sprintf(expr, "%s\\%x\\%x", name, G__func_page, G__func_now);
    }
    std::strcpy(name, expr);
  }
  //
  // Lookup the variable.
  //
  struct G__var_array* var = 0;
  int varid = 0;
  {
    char* p = std::strstr(name, "::");
    if (p) {
      // -- Qualified name, do the lookup in the specified context.
      *p = '\0';
      p += 2;
      int tagnum = G__defined_tagname(name, 0);
      if (tagnum != -1) {
        int store_memberfunc_tagnum = G__memberfunc_tagnum;
        int store_def_struct_member = G__def_struct_member;
        int store_exec_memberfunc = G__exec_memberfunc;
        int store_tagnum = G__tagnum;
        G__memberfunc_tagnum = tagnum;
        G__tagnum = tagnum;
        G__def_struct_member = 0;
        G__exec_memberfunc = 1;
        struct G__var_array* memvar = G__struct.memvar[tagnum];
        int hash = 0;
        int i = 0;
        G__hash(p, hash, i)
        var = G__getvarentry(p, hash, &varid, memvar, memvar);
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
      G__hash(name, hash, i)
      var = G__getvarentry(name, hash, &varid, &G__global, G__p_local);
    }
  }
  if (!var) {
    G__fprinterr(G__serr, "Limitation: %s initialization ignored", name);
    G__printlinenum();
    int c = G__fignorestream("},;");
    if (c == '}') {
      c = G__fignorestream(",;");
    }
    return c;
  }
  // We must be an aggregate type, enforce that.
  if (G__struct.baseclass[var->p_tagtable[varid]]->basen) {
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
  int& num_of_elements = var->varlabel[varid][1];
  const int stride = var->varlabel[varid][0];
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
    G__ASSERT(!var->p[varid] && (var->statictype[varid] == G__COMPILEDGLOBAL));
    if (G__static_alloc == 1) {
      if (G__func_now != -1) {
        var->statictype[varid] = G__LOCALSTATICBODY;
      }
      else {
        var->statictype[varid] = G__ifile.filenum;
      }
    }
    else {
      var->statictype[varid] = G__AUTO;
    }
  }
  G__ASSERT(var->statictype[varid] != G__COMPILEDGLOBAL);
  // Initialize buf.
  G__value buf;
  buf.type = std::toupper(var->type[varid]);
  buf.tagnum = var->p_tagtable[varid];
  buf.typenum = var->p_typetable[varid];
  buf.ref = 0;
  buf.obj.reftype.reftype = var->reftype[varid];
  // Get size.
  int size = 0;
  if (std::islower(var->type[varid])) {
    // -- We are *not* a pointer.
    // FIXME: Do we handle a typedef correctly here?  See similar code in G__initary().
    size = G__sizeof(&buf);
  }
  else {
    // -- We are a pointer, handle as a long.
    buf.type = 'L';
    size = G__LONGALLOC;
  }
  G__ASSERT((stride > 0) && (size > 0));
  // Get a pointer to the first data member.
  int memindex = 0;
  struct G__var_array* memvar = G__initmemvar(var->p_tagtable[varid], &memindex, &buf);
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
          long tmp = 0L;
          if (var->p[varid]) {
            // -- We already had some elements, resize.
            tmp = (long) std::realloc((void*) var->p[varid], size * num_of_elements);
          }
          else {
            // -- No elements allocate yet, get some.
            tmp = (long) std::malloc(size * num_of_elements);
          }
          if (tmp) {
            var->p[varid] = tmp;
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
        buf.obj.i = (var->p[varid] + (linear_index * size)) + memvar->p[memindex];
        G__value reg = G__getexpr(expr);
        if (std::isupper(memvar->type[memindex])) {
          // -- Data member is a pointer.
          *((long *) (buf.obj.i)) = (long) G__int(reg);
        }
        else if (
          (memvar->type[memindex] == 'c') && // character array
          (memvar->varlabel[memindex][1] /* number of elements */) > 0 &&
          (expr[0] == '"') // string literal
        ) {
          // -- Data member is a fixed-size character array.
          // FIXME: We do not handle a data member which is an unspecified length array.
          if (memvar->varlabel[memindex][1] /* number of elements */ > (int) std::strlen((char*)reg.obj.i)) {
            std::strcpy((char*) buf.obj.i, (char*) reg.obj.i);
          }
          else {
            std::strncpy((char*) buf.obj.i, (char*) reg.obj.i, memvar->varlabel[memindex][1] /* num of elements */);
          }
        }
        else {
          G__letvalue(&buf, reg);
        }
        // Move to next data member.
        memvar = G__incmemvar(memvar, &memindex, &buf);
        if ((c == '}') || !memvar) {
          // -- All done if no more data members or end of list.
          // FIXME: We are not handling nesting of braces properly.
          //        We need to default initialize the rest of the members.
          break;
        }
        // Get next initializer expression.
        c = G__fgetstream(expr, ",{}");
      } while (memvar);
      // Reset back to the beginning of the data member list.
      memvar = G__initmemvar(var->p_tagtable[varid], &memindex, &buf);
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

} // extern "C"

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
