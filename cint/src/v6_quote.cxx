/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file quote.c
 ************************************************************************
 * Description:
 *  Strip and add quotation
 ************************************************************************
 * Copyright(c) 1995~2002  Masaharu Goto 
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


/**************************************************************************
* G__value G__asm_gen_strip_quotation(string)
*
*  remove " and ' from string
**************************************************************************/
void G__asm_gen_strip_quotation(pval)
G__value *pval;
{
  /**************************************
   * G__LD instruction
   * 0 LD
   * 1 address in data stack
   * put defined
   **************************************/
#ifdef G__ASM_DBG
  if(G__asm_dbg) G__fprinterr(G__serr,"%3x: LD 0x%lx(%s) from %x\n"
			 ,G__asm_cp ,G__int(*pval)
			 ,(char *)G__int(*pval) ,G__asm_dt);
#endif
  G__asm_inst[G__asm_cp]=G__LD;
  G__asm_inst[G__asm_cp+1]=G__asm_dt;
  G__asm_stack[G__asm_dt] = *pval;
  G__inc_cp_asm(2,1);
}

#ifndef G__OLDIMPLEMENTATION1636
#ifdef G__CPPCONSTSTRING
char* G__saveconststring G__P((char *string));
#else
/******************************************************************
* char* G__savestring()
******************************************************************/
char* G__saveconststring(string)
char* string;
{
  int itemp,hash;
  struct G__ConstStringList *pconststring;
  /* Search existing const string list */
  G__hash(string,hash,itemp);
  pconststring = G__plastconststring;
  while(pconststring) {
    if(hash==pconststring->hash && strcmp(string,pconststring->string)==0) {
      return(pconststring->string);
    }
    pconststring = pconststring->prev;
  }

  /* Create new conststring entry */
  pconststring 
    = (struct G__ConstStringList*)malloc(sizeof(struct G__ConstStringList));
  pconststring->prev = G__plastconststring;
  G__plastconststring = pconststring;
  pconststring = G__plastconststring;

  pconststring->string=(char*)malloc(strlen(string)+2);
  pconststring->string[strlen(string)+1]='\0';
  strcpy(pconststring->string,string);
  pconststring->hash = hash;

  return(pconststring->string);
}
#endif
#endif

/******************************************************************
* G__value G__strip_quotation(string)
*
* Allocate memory and store const string expression. Then return 
* the string type value.
*
******************************************************************/
G__value G__strip_quotation(string)
char *string;
{
  int itemp,itemp2=0,hash;
#ifndef G__OLDIMPLEMENTATION1943
  int templen = G__LONGLINE;
  char *temp = (char*)malloc(G__LONGLINE);
#else
  char temp[G__LONGLINE];
#endif
  G__value result;
#ifdef G__OLDIMPLEMENTATION1636
  struct G__ConstStringList *pconststring;
#endif
#ifndef G__OLDIMPLEMENTATION1631
  int lenm1 = strlen(string)-1;
#endif

  result.tagnum = -1;
  result.typenum = -1;
  result.ref = 0;
#ifndef G__OLDIMPLEMENTATION2068
  result.isconst = G__CONSTVAR;
#endif
  if((string[0]=='"')||(string[0]=='\'')) {
    for(itemp=1;
#ifndef G__OLDIMPLEMENTATION1631
	itemp<lenm1;
#else
	itemp<strlen(string)-1;
#endif
	itemp++ ) {
      /*
      temp[itemp2++] = string[itemp];
      */
#ifndef G__OLDIMPLEMENTATION1943
      if(itemp2+1>templen) {
        temp = (char*)realloc(temp,2*templen);
        templen = 2*templen;
      }
#endif
      switch(string[itemp]) {
      case '\\' :
	switch(string[++itemp]) {
	/*
	case 'a':
	  temp[itemp2++] = '\a' ;
	  break;
	*/
	case 'b':
	  temp[itemp2++] = '\b' ;
	  break;
	case 'f':
	  temp[itemp2++] = '\f' ;
	  break;
	case 'n':
	  temp[itemp2++] = '\n' ;
	  break;
	case 'r':
	  temp[itemp2++] = '\r' ;
	  break;
	case 't':
	  temp[itemp2++] = '\t' ;
	  break;
	case 'v':
	  temp[itemp2++] = '\v' ;
	  break;
	case 'x':
	case 'X':
	  temp[itemp2]='0';
	  temp[itemp2+1]='x';
	  hash=1;
	  while(hash) {
	    switch(string[itemp+hash]) {
	    case 'a':
	    case 'A':
	    case 'b':
	    case 'B':
	    case 'c':
	    case 'C':
	    case 'd':
	    case 'D':
	    case 'e':
	    case 'E':
	    case 'f':
	    case 'F':
	    case '0':
	    case '1':
	    case '2':
	    case '3':
	    case '4':
	    case '5':
	    case '6':
	    case '7':
	    case '8':
	    case '9':
	      temp[itemp2+hash+1]=
		string[itemp+hash];
	      ++hash;
	      break;
	    default:
	      itemp += (hash-1);
	      temp[itemp2+hash+1]='\0';
	      hash=0;
	    }
	  }
	  temp[itemp2] = (char)G__int(G__checkBase(temp+itemp2 ,&hash));
	  ++itemp2;
	  break;
	case '0':
	case '1':
	case '2':
	case '3':
	case '4':
	case '5':
	case '6':
	case '7':
	  temp[itemp2]='0';
	  temp[itemp2+1]='o';
	  hash=0;
	  while(isdigit(string[itemp+hash])&&
		hash<3) {
	    temp[itemp2+hash+2]=
	      string[itemp+hash];
	    ++hash;
	  }
	  itemp += (hash-1);
	  temp[itemp2+hash+2]='\0';
	  hash=0;
	  temp[itemp2] = (char)G__int(G__checkBase(temp+itemp2 ,&hash));
	  ++itemp2;
	  break;
#ifndef G__OLDIMPLEMENTATION1245
	case '\n':
	  break;
#endif
	default:
	  temp[itemp2++] = string[itemp];
	  break;
	}
	break;
      case '"':
	if('"'==string[itemp+1]) {
#ifndef G__OLDIMPLEMENTATION1416
	  ++itemp;
#else
	  itemp+=2;
#endif
	}
#ifndef G__OLDIMPLEMENTATION998
	else if(G__NOLINK==G__globalcomp) 
	  G__genericerror("Error: String literal syntax error");
#endif
#ifndef G__OLDIMPLEMENTATION1416
	continue;
#endif
      default:
	temp[itemp2++] = string[itemp];
#ifdef G__MULTIBYTE
	if(G__IsDBCSLeadByte(string[itemp])) {
	  temp[itemp2++] = string[++itemp];
	  G__CheckDBCS2ndByte(string[itemp]);
	}
#endif
	break;
      }
    }
    temp[itemp2]='\0';
  }
  else {
    if(G__isvalue(string)) {
      /* string is a pointer */
      G__letint(&result,'C',atol(string));
#ifndef G__OLDIMPLEMENTATION1943
      free((void*)temp);
#endif
      return(result);
    }
    else {
      /* return string */
      strcpy(temp,string);
    }
  }


#ifndef G__OLDIMPLEMENTATION1636
  G__letint(&result,'C',(long)G__saveconststring(temp));
#else
  /* Search existing const string list */
  G__hash(temp,hash,itemp);
  pconststring = G__plastconststring;
  while(pconststring) {
    if(hash==pconststring->hash && strcmp(temp,pconststring->string)==0) {
      G__letint(&result,'C',(long)pconststring->string);
#ifndef G__OLDIMPLEMENTATION1943
      free((void*)temp);
#endif
      return(result);
    }
    pconststring = pconststring->prev;
  }

  /* Create new conststring entry */
  pconststring 
    = (struct G__ConstStringList*)malloc(sizeof(struct G__ConstStringList));
  pconststring->prev = G__plastconststring;
  G__plastconststring = pconststring;
  pconststring = G__plastconststring;

  pconststring->string=(char*)malloc(strlen(temp)+2);
  pconststring->string[strlen(temp)+1]='\0';
  strcpy(pconststring->string,temp);
  pconststring->hash = hash;

  G__letint(&result,'C',(long)pconststring->string);
#endif

#ifndef G__OLDIMPLEMENTATION1943
  free((void*)temp);
#endif
  return(result);
}


/******************************************************************
* char *G__charaddquote(c)
*
* Called by
*   G__tocharexpr()
*   G__valuemonitor()
******************************************************************/
char *G__charaddquote(string,c)
char *string,c;
{
  switch(c) {
  case '\\':
    sprintf(string,"'\\\\'");
    break;
  case '\'':
    sprintf(string,"'\\''");
    break;
  case '\0':
    sprintf(string,"'\\0'");
    break;
  case '\"':
    sprintf(string,"'\\\"'");
    break;
    /*
      case '\?':
      sprintf(string,"'\\?'");
      break;
      */
    /*
      case '\a':
      sprintf(string,"'\\a'");
      break;
      */
  case '\b':
    sprintf(string,"'\\b'");
    break;
  case '\f':
    sprintf(string,"'\\f'");
    break;
  case '\n':
    sprintf(string,"'\\n'");
    break;
  case '\r':
    sprintf(string,"'\\r'");
    break;
  case '\t':
    sprintf(string,"'\\t'");
    break;
  case '\v':
    sprintf(string,"'\\v'");
    break;
  default:
#ifdef G__MULTIBYTE
    if(G__IsDBCSLeadByte(c)) {
      G__genericerror("Limitation: Multi-byte char in single quote not handled property");
    }
#endif
    sprintf(string,"'%c'",c);
    break;
  }
  return(string);
}

/******************************************************************
* G__strip_singlequotation
*
* Called by
*   G__getitem()
*
******************************************************************/
G__value G__strip_singlequotation(string)
char *string;
{
  G__value result;
  int i;
  result.type='c';
  result.tagnum = -1;
  result.typenum = -1;
  result.ref = 0;
  if(string[0]=='\'') {
    switch(string[1]) {
    case '\\':
      switch(string[2]) {
	/*
	  case 'a' :
	  result.obj.i='\a';
	  break;
	  */
      case 'b' :
	result.obj.i='\b';
	break;
      case 'f' :
	result.obj.i='\f';
	break;
      case 'n' :
	result.obj.i='\n';
	break;
      case 'r' :
	result.obj.i='\r';
	break;
      case 't' :
	result.obj.i='\t';
	break;
      case 'v' :
	result.obj.i='\v';
	break;
	/*
	  case '0' :
	  result.obj.i='\0';
	  break;
	  */
      case 'x' :
      case 'X' :
	string[1]='0';
	string[strlen(string)-1]='\0';
	result.obj.i=G__int(G__checkBase(string+1,&i));
	break;
      case '0' :
      case '1' :
      case '2' :
      case '3' :
      case '4' :
      case '5' :
      case '6' :
      case '7' :
	string[0]='0';
	string[1]='o';
	string[strlen(string)-1]='\0';
	result.obj.i=G__int(G__checkBase(string,&i));
	break;
	default :
	  result.obj.i=string[2];
	break;
      }
      break;
    default:
      result.obj.i=string[1];
#ifdef G__MULTIBYTE
      if(G__IsDBCSLeadByte(string[1])) {
	G__CheckDBCS2ndByte(string[2]);
	result.obj.i=result.obj.i*0x100+string[2];
	result.typenum = G__defined_typename("wchar_t");
	result.tagnum = G__newtype.tagnum[result.typenum];
	result.type = G__newtype.type[result.typenum];
      }
#endif      
      break;
    }
  }
  else {
    result.obj.i=string[0];
  }
  return(result);
}

/******************************************************************
* char *G__add_quotation(string)
*
* Called by
*    
******************************************************************/
char *G__add_quotation(string,temp)
char *string;
char *temp;
{
  int c;
  short i=0,l=0;
  temp[i++]='"';
  while((c=string[l++])!='\0') {
    switch(c) {
    case '\n': 
      temp[i++]='\\';
      temp[i++]='n';
      break;
    case '\r': 
      temp[i++]='\\';
      temp[i++]='r';
      break;
    case '\\': 
      temp[i++]='\\';
      temp[i++]='\\';
      break;
    case '"': 
      temp[i++]='\\';
      temp[i++]='"';
      break;
    default: 
      temp[i++]=c;
      break;
    }
  }
  temp[i++]='"';
  temp[i]='\0';
  return(temp);
}


/******************************************************************
* char *G__tocharexpr(result7)
******************************************************************/
char *G__tocharexpr(result7)
char *result7;
{
  if((result7[0]=='\\')&&(result7[1]=='\'')) {
    G__charaddquote(result7,result7[2]);
  }
  return(NULL);
}



/****************************************************************
* char *G__string()
* 
****************************************************************/
char *G__string(buf,temp)
G__value buf;
char *temp;
{
  char temp1[G__MAXNAME];
  switch(buf.type) {
  case '\0':
    temp[0]='\0'; /* sprintf(temp,""); */
    break;
  case 'C': /* string */
    if(buf.obj.i) {
      G__add_quotation((char *)G__int(buf),temp);
    }
    else {
      temp[0]='\0';
    }
    break;
  case 'd':
  case 'f':
    sprintf(temp,"%.17e",buf.obj.d);
    break;
  case 'w':
    G__logicstring(buf,1,temp1);
    sprintf(temp,"0b%s",temp1);
    break;
  default:
#ifndef G__FONS31
    sprintf(temp,"%ld",buf.obj.i);
#else
    sprintf(temp,"%d",buf.obj.i);
#endif
    break;
  }
  return(temp);
}

/****************************************************************
* char *G__quotedstring()
* 
****************************************************************/
char *G__quotedstring(buf,result)
char *buf;
char *result;
{
	int i=0,r=0;
	int c;
	while((c=buf[i])) {
		switch(c) {
	        case '\\':
		case '"':
			result[r++] = '\\';
			result[r++] = c;
			break;
		default:
			result[r++] = c;
			break;
		}
		++i;
	}
	result[r]='\0';
	return(result);
}

/****************************************************************
* char *G__logicstring()
* 
****************************************************************/
char *G__logicstring(buf,dig,result)
G__value buf;
int dig;
char *result;
{
	char tristate[G__MAXNAME];
	unsigned int hilo,hiz,i,ii,flag;
	switch(buf.type) {
	case 'd': /* double */
	case 'f': /* float */
	case 'w': /* logic */
		hilo = buf.obj.i;
		hiz = *(&buf.obj.i+1);
		G__getbase(hilo,2,32,result);
		G__getbase(hiz,2,32,tristate);
		break;
	default:
		hilo = buf.obj.i;
		hiz = 0;
		G__getbase(hilo,2,32,result);
		G__getbase(hiz,2,32,tristate);
		break;
	}
	flag=0;
	ii=0;
	for(i=0;i<32;i++) {
		if((int)(32-i)<=dig) flag=1;
		switch(result[i]){
		case '0':
			if(tristate[i]=='0') {
				if(flag!=0) {
					result[ii++]='0';
				}
			}
			else {
				flag=1;
				result[ii++]='x';
			}
			break;
		case '1':
			flag=1;
			if(tristate[i]=='0') {
				result[ii++]='1';
			}
			else {
				result[ii++]='z';
			}
			break;
		}
	}
	if(ii!=0) result[ii]='\0';
	else      result[1]='\0';
	return(result);
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
