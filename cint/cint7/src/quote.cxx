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
#include "Api.h"

#include <set>
#include <string>

using namespace Cint::Internal;

/**************************************************************************
* G__value G__asm_gen_strip_quotation(string)
*
*  remove " and ' from string
**************************************************************************/
void Cint::Internal::G__asm_gen_strip_quotation(G__value *pval)
{
  /**************************************
   * G__LD instruction
   * 0 LD
   * 1 address in data stack
   * put defined
   **************************************/
#ifdef G__ASM_DBG
   if(G__asm_dbg)
      G__fprinterr(G__serr, "%3x,%3x: LD %ld  %s:%d\n", G__asm_cp, G__asm_dt, G__int(*pval), __FILE__, __LINE__);
#endif
  G__asm_inst[G__asm_cp]=G__LD;
  G__asm_inst[G__asm_cp+1]=G__asm_dt;
  G__asm_stack[G__asm_dt] = *pval;
  G__inc_cp_asm(2,1);
}

#include <set>
#include <string>
#if (!defined(__hpux) && !defined(_MSC_VER)) || __HP_aCC >= 53000
using namespace std;
#endif
/******************************************************************
* char* G__savestring()
******************************************************************/
static const char* G__saveconststring(const char* s)
{
   static std::set<std::string> static_conststring;
   std::string str(s);
   return static_conststring.insert(str).first->c_str();
}

/******************************************************************
* G__value G__strip_quotation(string)
*
* Allocate memory and store const string expression. Then return 
* the string type value.
*
******************************************************************/
G__value Cint::Internal::G__strip_quotation(const char *string)
{
  int itemp,itemp2=0,hash;
  int templen = G__LONGLINE;
  char *temp = (char*)malloc(G__LONGLINE);
  G__value result = G__null;
  int lenm1 = strlen(string)-1;

  G__value_typenum(result) = ::Reflex::Type::ByName("const char*");  // result.isconst = G__CONSTVAR;
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
       Cint::G__letpointer(&result,(long)string,G__value_typenum(result));
      free((void*)temp);
      return(result);
    }
    else {
      /* return string */
      strcpy(temp,string);
    }
  }


  G__letpointer(&result,(long)G__saveconststring(temp),G__value_typenum(result));

  free((void*)temp);
  return(result);
}


/******************************************************************
* char *G__charaddquote(c)
*
* Called by
*   G__tocharexpr()
*   G__valuemonitor()
******************************************************************/
char *Cint::Internal::G__charaddquote(char *string,char c)
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
G__value Cint::Internal::G__strip_singlequotation(const char *in_string)
{
  static Reflex::Type CharType( ::Reflex::Type::ByName("char") );
  G__value result = G__null;
  int i;
  G__value_typenum(result) = CharType;
   
  G__StrBuf string_sb(strlen(in_string));
  char *string = string_sb;
  strcpy(string,in_string);
   
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
        G__value_typenum(result) = G__find_typedef("wchar_t");
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

/******************************************************************
* char *G__add_quotation(string)
*
* Called by
*    
******************************************************************/
char *Cint::Internal::G__add_quotation(char *string,char *temp)
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
char *Cint::Internal::G__tocharexpr(char *result7)
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
char *Cint::Internal::G__string(G__value buf,char *temp)
{
  G__StrBuf temp1_sb(G__MAXNAME);
  char *temp1 = temp1_sb;
  switch(G__get_type(G__value_typenum(buf))) {
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
    sprintf(temp,"%ld",buf.obj.i);
    break;
  }
  return(temp);
}

/****************************************************************
* char *G__quotedstring()
* 
****************************************************************/
char *Cint::Internal::G__quotedstring(char *buf,char *result)
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
char *Cint::Internal::G__logicstring(G__value buf,int dig,char *result)
{
        G__StrBuf tristate_sb(G__MAXNAME);
        char *tristate = tristate_sb;
        unsigned int hilo,hiz,i,ii,flag;
        switch(G__get_type(G__value_typenum(buf))) {
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
