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
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "common.h"

extern "C" {

/**************************************************************************
* G__value G__asm_gen_strip_quotation(string)
*
*  remove " and ' from string
**************************************************************************/
void G__asm_gen_strip_quotation(G__value *pval)
{
  /**************************************
   * G__LD instruction
   * 0 LD
   * 1 address in data stack
   * put defined
   **************************************/
#ifdef G__ASM_DBG
  if (G__asm_dbg) {
    G__fprinterr(G__serr, "%3x,%3x: LD %ld  %s:%d\n", G__asm_cp, G__asm_dt, G__int(*pval), __FILE__, __LINE__);
  }
#endif
  G__asm_inst[G__asm_cp]=G__LD;
  G__asm_inst[G__asm_cp+1]=G__asm_dt;
  G__asm_stack[G__asm_dt] = *pval;
  G__inc_cp_asm(2,1);
}

#ifdef G__CPPCONSTSTRING
char* G__saveconststring G__P((char *string));
#else
/******************************************************************
* char* G__savestring()
******************************************************************/
char* G__saveconststring(char* string)
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
  strcpy(pconststring->string,string); // Okay, we allocated enough space
  pconststring->hash = hash;

  return(pconststring->string);
}
#endif

/******************************************************************
* G__value G__strip_quotation(string)
*
* Allocate memory and store const string expression. Then return 
* the string type value.
*
******************************************************************/
G__value G__strip_quotation(const char *string)
{
  int itemp,itemp2=0,hash;
  int templen = G__LONGLINE;
  char *temp = (char*)malloc(G__LONGLINE);
  G__value result;
  int lenm1 = strlen(string)-1;

  result.tagnum = -1;
  result.typenum = -1;
  result.ref = 0;
  result.isconst = G__CONSTVAR;
  if((string[0]=='"')||(string[0]=='\'')) {
    for(itemp=1;
        itemp<lenm1;
        itemp++ ) {
      /*
      temp[itemp2++] = string[itemp];
      */
      if(itemp2+1>templen) {
        temp = (char*)realloc(temp,2*templen);
        templen = 2*templen;
      }
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
        case '\n':
          break;
        default:
          temp[itemp2++] = string[itemp];
          break;
        }
        break;
      case '"':
        if('"'==string[itemp+1]) {
          ++itemp;
        }
        else if(G__NOLINK==G__globalcomp) 
          G__genericerror("Error: String literal syntax error");
        continue;
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
      free((void*)temp);
      return(result);
    }
    else {
      /* return string */
       G__strlcpy(temp,string,templen);
    }
  }


  G__letint(&result,'C',(long)G__saveconststring(temp));

  free((void*)temp);
  return(result);
}

} /* extern "C" */

/******************************************************************
* char *G__charaddquote(c)
*
* Called by
*   G__valuemonitor()
******************************************************************/
G__FastAllocString &G__charaddquote(G__FastAllocString &string,char c)
{
  switch(c) {
  case '\\':
    string.Format("'\\\\'");
    break;
  case '\'':
    string.Format("'\\''");
    break;
  case '\0':
    string.Format("'\\0'");
    break;
  case '\"':
    string.Format("'\\\"'");
    break;
    /*
      case '\?':
      string.Format("'\\?'");
      break;
      */
    /*
      case '\a':
      string.Format("'\\a'");
      break;
      */
  case '\b':
    string.Format("'\\b'");
    break;
  case '\f':
    string.Format("'\\f'");
    break;
  case '\n':
    string.Format("'\\n'");
    break;
  case '\r':
    string.Format("'\\r'");
    break;
  case '\t':
    string.Format("'\\t'");
    break;
  case '\v':
    string.Format("'\\v'");
    break;
  default:
#ifdef G__MULTIBYTE
    if(G__IsDBCSLeadByte(c)) {
      G__genericerror("Limitation: Multi-byte char in single quote not handled property");
    }
#endif
    string.Format("'%c'",c);
    break;
  }
  return(string);
}

extern "C" {

/******************************************************************
* G__strip_singlequotation
*
* Called by
*   G__getitem()
*
******************************************************************/
G__value G__strip_singlequotation(char *string)
{
  G__value result = G__null;
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
        result.obj.ch='\b';
        break;
      case 'f' :
        result.obj.ch='\f';
        break;
      case 'n' :
        result.obj.ch='\n';
        break;
      case 'r' :
        result.obj.ch='\r';
        break;
      case 't' :
        result.obj.ch='\t';
        break;
      case 'v' :
        result.obj.ch='\v';
        break;
        /*
          case '0' :
          result.obj.ch='\0';
          break;
          */
      case 'x' :
      case 'X' :
        string[1]='0';
        string[strlen(string)-1]='\0';
        result.obj.ch=G__int(G__checkBase(string+1,&i));
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
        result.obj.ch=G__int(G__checkBase(string,&i));
        break;
        default :
          result.obj.ch=string[2];
        break;
      }
      break;
    default:
      result.obj.ch=string[1];
#ifdef G__MULTIBYTE
      if(G__IsDBCSLeadByte(string[1])) {
        G__CheckDBCS2ndByte(string[2]);
        result.obj.i=result.obj.i*0x100+string[2];
        result.typenum = G__defined_typename("wchar_t");
        if (result.typenum>=0) {
          result.tagnum = G__newtype.tagnum[result.typenum];
          result.type = G__newtype.type[result.typenum];
        }
      }
#endif      
      break;
    }
  }
  else {
    result.obj.ch=string[0];
  }
  return(result);
}

} // extern "C"

/******************************************************************
* char *G__add_quotation(string)
*
* Called by
*    
******************************************************************/
char *G__add_quotation(const char* string,G__FastAllocString& temp)
{
  int c;
  short i=0,l=0;
  temp.Set(i++, '"');
  while((c=string[l++])!='\0') {
    switch(c) {
    case '\n': 
       temp.Set(i++, '\\');
       temp.Set(i++, 'n');
       break;
    case '\r': 
       temp.Set(i++, '\\');
       temp.Set(i++, 'r');
       break;
    case '\\': 
       temp.Set(i++, '\\');
       temp.Set(i++, '\\');
       break;
    case '"': 
       temp.Set(i++, '\\');
       temp.Set(i++, '"');
       break;
    default: 
       temp.Set(i++, c);
       break;
    }
  }
  temp.Set(i++, '"');
  temp.Set(i, 0);
  return temp;
}

/****************************************************************
* char *G__string()
* 
****************************************************************/
char *G__string(G__value buf, G__FastAllocString& temp)
{
  G__FastAllocString temp1(G__MAXNAME);
  switch(buf.type) {
  case '\0':
    temp[0]='\0';
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
     temp.Format("%.17e",buf.obj.d);
    break;
  case 'w':
    G__logicstring(buf,1,temp1);
    temp.Format("0b%s",temp1());
    break;
  default:
     temp.Format("%ld",buf.obj.i);
    break;
  }
  return(temp);
}

extern "C" {

/****************************************************************
* char *G__quotedstring()
* 
****************************************************************/
char *G__quotedstring(char *buf,char *result)
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
char *G__logicstring(G__value buf,int dig,char *result)
{
        G__FastAllocString tristate(G__MAXNAME);
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

} /* extern "C" */

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
