/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file val2a.c
 ************************************************************************
 * Description:
 *  G__value to ASCII expression
 ************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto 
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

#define G__OLDIMPLEMENTATION328

/****************************************************************
* char *G__valuemonitor()
* 
****************************************************************/
char *G__valuemonitor(buf,temp)
G__value buf;
char *temp;
{
  char temp2[G__ONELINE];

  if(buf.typenum != -1) {
    switch(buf.type) {
    case 'd':
    case 'f':
      /* typedef can be local to a class */
#ifndef G__OLDIMPLEMENTATION719
      if(buf.obj.d<0.0)
	sprintf(temp,"(%s)(%.17e)"
		,G__type2string(buf.type ,buf.tagnum ,buf.typenum
#ifndef G__OLDIMPLEMENTATION1990
				,0
#else
				,buf.obj.reftype.reftype
#endif
				,0)
				,buf.obj.d);
      else
	sprintf(temp,"(%s)%.17e"
		,G__type2string(buf.type ,buf.tagnum ,buf.typenum
#ifndef G__OLDIMPLEMENTATION1990
				,0
#else
				,buf.obj.reftype.reftype
#endif
				,0)
				,buf.obj.d);
      break;
#else /* ON719 */
      if(buf.obj.d<0.0)
	sprintf(temp,"(%s)(%.17e)" ,G__newtype.name[buf.typenum] ,buf.obj.d);
      else
	sprintf(temp,"(%s)%.17e" ,G__newtype.name[buf.typenum] ,buf.obj.d);
      break;
#endif /* ON719 */
#ifndef G__OLDIMPLEMENTATION1255
  case 'b':
    if(G__in_pause)
      sprintf(temp,"(unsigned char)%u",(unsigned char)buf.obj.i);
    else 
      sprintf(temp,"(unsignedchar)%u",(unsigned char)buf.obj.i);
    break;
  case 'r':
    if(G__in_pause)
      sprintf(temp,"(unsigned short)%u",(unsigned short)buf.obj.i);
    else
      sprintf(temp,"(unsignedshort)%u",(unsigned short)buf.obj.i);
    break;
  case 'h':
    if(G__in_pause)
      sprintf(temp,"(unsigned int)%u",(unsigned int)buf.obj.i);
    else
      sprintf(temp,"(unsignedint)%u",(unsigned int)buf.obj.i);
    break;
  case 'k':
    if(G__in_pause)
      sprintf(temp,"(unsigned long)%lu",(unsigned long)buf.obj.i);
    else
      sprintf(temp,"(unsignedlong)%lu",(unsigned long)buf.obj.i);
    break;
#endif
    default:
      if(islower(buf.type)) {
#ifndef G__OLDIMPLEMENTATION1912
	if('u'==buf.type && -1!=buf.tagnum &&
	   (strcmp(G__struct.name[buf.tagnum],"G__longlong")==0 ||
	    strcmp(G__struct.name[buf.tagnum],"G__ulonglong")==0 ||
	    strcmp(G__struct.name[buf.tagnum],"G__longdouble")==0)) {
	  if(G__in_pause) {
	    char llbuf[100];
	    sprintf(temp,"(%s)" 
		    ,G__type2string(buf.type ,buf.tagnum ,buf.typenum
				    ,buf.obj.reftype.reftype,0));
	    if(strcmp(G__struct.name[buf.tagnum],"G__longlong")==0) {
	      sprintf(llbuf
	  ,"G__printformatll((char*)(%ld),\"%%lld\",(void*)(%ld))"
		      ,(long)llbuf,buf.obj.i);
	      G__getitem(llbuf);
	      strcat(temp,llbuf);
	    }
	    else if(strcmp(G__struct.name[buf.tagnum],"G__ulonglong")==0) {
	      sprintf(llbuf
	  ,"G__printformatull((char*)(%ld),\"%%llu\",(void*)(%ld))"
		      ,(long)llbuf,buf.obj.i);
	      G__getitem(llbuf);
	      strcat(temp,llbuf);
	    }
	    else if(strcmp(G__struct.name[buf.tagnum],"G__longdouble")==0) {
	      sprintf(llbuf
	  ,"G__printformatld((char*)(%ld),\"%%LG\",(void*)(%ld))"
		      ,(long)llbuf,buf.obj.i);
	      G__getitem(llbuf);
	      strcat(temp,llbuf);
	    }
	  }
	  else
	    G__setiparseobject(&buf,temp);
	} else
#endif
#ifndef G__OLDIMPLEMENTATION2192
	  if (buf.type=='n' && buf.obj.ll<0) 
	    sprintf(temp,"(%s)(%lld)" 
		    ,G__type2string(buf.type ,buf.tagnum ,buf.typenum
				    ,0,0)
		    ,buf.obj.ll);
	  else if (buf.type=='m' || buf.type=='n') 
	    sprintf(temp,"(%s)%llu" 
		    ,G__type2string(buf.type ,buf.tagnum ,buf.typenum
				    ,0,0)
		    ,buf.obj.ull);
	
	  else 
#endif
#ifndef G__OLDIMPLEMENTATION719
	if(buf.obj.i<0)
	  sprintf(temp,"(%s)(%ld)" 
		  ,G__type2string(buf.type ,buf.tagnum ,buf.typenum
				  ,buf.obj.reftype.reftype,0)
		  ,buf.obj.i);
	else
	  sprintf(temp,"(%s)%ld" 
		  ,G__type2string(buf.type ,buf.tagnum ,buf.typenum
				  ,buf.obj.reftype.reftype,0)
		  ,buf.obj.i);
#else /* ON719 */
	if(buf.obj.i<0)
	  sprintf(temp,"(%s)(%ld)" ,G__newtype.name[buf.typenum] ,buf.obj.i);
	else
	  sprintf(temp,"(%s)%ld" ,G__newtype.name[buf.typenum] ,buf.obj.i);
#endif /* ON719 */
      }
      else {
#ifndef G__OLDIMPLEMENTATION719
	if('C'==buf.type&&G__in_pause && buf.obj.i>0x10000 &&
	   G__PARANORMAL==buf.obj.reftype.reftype) 
	  sprintf(temp,"(%s 0x%lx)\"%s\""
		  ,G__type2string(buf.type ,buf.tagnum ,buf.typenum
				  ,buf.obj.reftype.reftype,0)
		  ,buf.obj.i,(char*)buf.obj.i);
	else
	  sprintf(temp,"(%s)0x%lx" 
		  ,G__type2string(buf.type ,buf.tagnum ,buf.typenum
				  ,buf.obj.reftype.reftype,0)
		  ,buf.obj.i);
#else /* ON719 */
	switch(buf.obj.reftype.reftype) {
	case G__PARANORMAL:
#ifdef G__OLDIMPLEMENTATION626_YET
	  if('C'==buf.type && G__in_pause && buf.obj.i>0x10000) {
	    sprintf(temp,"(%s* 0x%lx)\"%s\"",G__newtype.name[buf.typenum]
		    ,buf.obj.i,(char*)buf.obj.i);
	  }
	  else {
	    sprintf(temp,"(%s*)0x%lx" ,G__newtype.name[buf.typenum] ,buf.obj.i);
	  }
#else
	  sprintf(temp,"(%s*)0x%lx" ,G__newtype.name[buf.typenum] ,buf.obj.i);
#endif
	  break;
	case G__PARAP2P:
	  sprintf(temp,"(%s**)0x%lx" ,G__newtype.name[buf.typenum] ,buf.obj.i);
	  break;
	case G__PARAP2P2P:
	  sprintf(temp,"(%s***)0x%lx",G__newtype.name[buf.typenum],buf.obj.i);
	  break;
#ifndef G__OLDIMPLEMENTATION707
	case (G__PARAP2P2P+1):
	  sprintf(temp,"(%s****)0x%lx",G__newtype.name[buf.typenum],buf.obj.i);
	  break;
	case (G__PARAP2P2P+2):
	  sprintf(temp,"(%s*****)0x%lx",G__newtype.name[buf.typenum],buf.obj.i);
	  break;
#endif
	}
#endif /* ON719 */
      }
    }
    return(temp);
  }
  
  switch(buf.type) {
  case '\0':
    sprintf(temp,"NULL");
    break;
  case 'b':
    if(G__in_pause)
      sprintf(temp,"(unsigned char)%u",(unsigned char)buf.obj.i);
    else 
      sprintf(temp,"(unsignedchar)%u",(unsigned char)buf.obj.i);
    break;
  case 'B':
    if(G__in_pause)
      sprintf(temp,"(unsigned char*)0x%lx",buf.obj.i);
    else 
      sprintf(temp,"(unsignedchar*)0x%lx",buf.obj.i);
    break;
#ifndef G__OLDIMPLEMENTATION1001
  case 'T':
#endif
  case 'C':
    if(buf.obj.i!=0) {
      if(G__in_pause && G__PARANORMAL==buf.obj.reftype.reftype) {
	if(strlen((char*)buf.obj.i)>G__ONELINE-25) {
	  strncpy(temp2,(char*)buf.obj.i,G__ONELINE-25);
	  temp2[G__ONELINE-25] = 0;
	  sprintf(temp,"(char* 0x%lx)\"%s\"...",buf.obj.i,temp2);
	}
	else {
	  G__add_quotation((char*)buf.obj.i,temp2);
	  sprintf(temp,"(char* 0x%lx)%s",buf.obj.i,temp2);
	}
      }
      else {
	sprintf(temp,"(char*)0x%lx",buf.obj.i);
      }
    }
    else {
      if(G__in_pause) 
	sprintf(temp,"(char* 0x0)\"\"");
      else 
	sprintf(temp,"(char*)0x0");
    }
    break;
  case 'c':
    G__charaddquote(temp2,(char)buf.obj.i);
    if(G__in_pause)
      sprintf(temp,"(char %ld)%s",buf.obj.i,temp2);
    else
      sprintf(temp,"(char)%ld",buf.obj.i);
    break;
  case 'r':
    if(G__in_pause)
      sprintf(temp,"(unsigned short)%u",(unsigned short)buf.obj.i);
    else
      sprintf(temp,"(unsignedshort)%u",(unsigned short)buf.obj.i);
    break;
  case 'R':
    if(G__in_pause)
      sprintf(temp,"(unsigned short*)0x%lx",buf.obj.i);
    else
      sprintf(temp,"(unsignedshort*)0x%lx",buf.obj.i);
    break;
  case 's':
    if(buf.obj.i<0)
      sprintf(temp,"(short)(%d)",(short)buf.obj.i);
    else 
      sprintf(temp,"(short)%d",(short)buf.obj.i);
    break;
  case 'S':
    sprintf(temp,"(short*)0x%lx",buf.obj.i);
    break;
  case 'h':
    if(G__in_pause)
      sprintf(temp,"(unsigned int)%u",(unsigned int)buf.obj.i);
    else
      sprintf(temp,"(unsignedint)%u",(unsigned int)buf.obj.i);
    break;
  case 'H':
    if(G__in_pause)
      sprintf(temp,"(unsigned int*)0x%lx",buf.obj.i);
    else
      sprintf(temp,"(unsignedint*)0x%lx",buf.obj.i);
    break;
  case 'i':
    if(buf.tagnum != -1) {
      if(G__struct.type[buf.tagnum]=='e') {
	if(buf.obj.i<0)
	  sprintf(temp,"(enum %s)(%d)",G__fulltagname(buf.tagnum,1),(int)buf.obj.i);
	else
	  sprintf(temp,"(enum %s)%d",G__fulltagname(buf.tagnum,1),(int)buf.obj.i);
      }
      else {
	if(buf.obj.i<0)
	  sprintf(temp,"(int)(%d)",(int)buf.obj.i);
	else 
	  sprintf(temp,"(int)%d",(int)buf.obj.i);
      }
    }
    else {
      if(buf.obj.i<0)
	sprintf(temp,"(int)(%d)",(int)buf.obj.i);
      else 
	sprintf(temp,"(int)%d",(int)buf.obj.i);
    }
    break;
  case 'I':
    if(buf.tagnum != -1) {
      if(G__struct.type[buf.tagnum]=='e') {
	sprintf(temp,"(enum %s*)0x%lx",G__fulltagname(buf.tagnum,1),buf.obj.i);
      }
      else {
	sprintf(temp,"(int*)0x%lx",buf.obj.i);
      }
    }
    else {
      sprintf(temp,"(int*)0x%lx",buf.obj.i);
    }
    break;
#ifndef G__OLDIMPLEMENTATION2189
  case 'n':
    if(buf.obj.ll<0)
      sprintf(temp,"(long long)(%lld)",buf.obj.ll);
    else 
      sprintf(temp,"(long long)%lld",buf.obj.ll);
    break;
  case 'm':
    sprintf(temp,"(unsigned long long)%llu",buf.obj.ull);
    break;
  case 'q':
#ifndef G__OLDIMPLEMENTATION2192
    if(buf.obj.ld<0)
      sprintf(temp,"(long double)(%Lg)",buf.obj.ld);
    else 
      sprintf(temp,"(long double)%Lg",buf.obj.ld);
#else
    if(buf.obj.ld<0)
      sprintf(temp,"(long double)(%g)",buf.obj.ld);
    else 
      sprintf(temp,"(long double)%g",buf.obj.ld);
#endif
    break;
#endif
#ifndef G__OLDIMPLEMENTATION1604
  case 'g':
#ifdef G__BOOL4BYTE
    sprintf(temp,"(bool)%d",(int)buf.obj.i?1:0);
#else
    sprintf(temp,"(bool)%d",(unsigned char)buf.obj.i?1:0);
#endif
    break;
#endif
  case 'k':
    if(G__in_pause)
      sprintf(temp,"(unsigned long)%lu",(unsigned long)buf.obj.i);
    else
      sprintf(temp,"(unsignedlong)%lu",(unsigned long)buf.obj.i);
    break;
  case 'K':
    if(G__in_pause)
      sprintf(temp,"(unsigned long*)0x%lx",buf.obj.i);
    else
      sprintf(temp,"(unsignedlong*)0x%lx",buf.obj.i);
    break;
  case 'l':
    if(buf.obj.i<0)
      sprintf(temp,"(long)(%ld)",buf.obj.i);
    else
      sprintf(temp,"(long)%ld",buf.obj.i);
    break;
  case 'L':
    sprintf(temp,"(long*)0x%lx",buf.obj.i);
    break;
  case 'y':
    if(buf.obj.i<0)
      sprintf(temp,"(void)(%ld)",buf.obj.i);
    else
      sprintf(temp,"(void)%ld",buf.obj.i);
    break;
#ifndef G__OLDIMPLEMENTATION2191
  case '1':
#else
  case 'Q':
#endif
  case 'Y':
    sprintf(temp,"(void*)0x%lx",buf.obj.i);
    break;
  case 'E':
    sprintf(temp,"(FILE*)0x%lx",buf.obj.i);
    break;
  case 'd':
    if(buf.obj.d<0.0)
      sprintf(temp,"(double)(%.17e)",buf.obj.d);
    else
      sprintf(temp,"(double)%.17e",buf.obj.d);
    break;
  case 'D':
    sprintf(temp,"(double*)0x%lx",buf.obj.i);
    break;
  case 'f':
    if(buf.obj.d<0.0)
      sprintf(temp,"(float)(%.17e)",buf.obj.d);
    else
      sprintf(temp,"(float)%.17e",buf.obj.d);
    break;
  case 'F':
    sprintf(temp,"(float*)0x%lx",buf.obj.i);
    break;
  case 'u':
    switch(G__struct.type[buf.tagnum]) {
    case 's':
      if(buf.obj.i<0)
	sprintf(temp,"(struct %s)(%ld)" 
		,G__fulltagname(buf.tagnum,1),buf.obj.i);
      else
	sprintf(temp,"(struct %s)%ld" ,G__fulltagname(buf.tagnum,1),buf.obj.i);
      break;
    case 'c':
#ifndef G__OLDIMPLEMENTATION1912
      if(-1!=buf.tagnum &&
	 (strcmp(G__struct.name[buf.tagnum],"G__longlong")==0 ||
	  strcmp(G__struct.name[buf.tagnum],"G__ulonglong")==0 ||
	  strcmp(G__struct.name[buf.tagnum],"G__longdouble")==0)) {
	if(G__in_pause) {
	  char llbuf[100];
	  sprintf(temp,"(%s)" 
		  ,G__type2string(buf.type ,buf.tagnum ,buf.typenum
				  ,buf.obj.reftype.reftype,0));
	  if(strcmp(G__struct.name[buf.tagnum],"G__longlong")==0) {
	    sprintf(llbuf
		    ,"G__printformatll((char*)(%ld),\"%%lld\",(void*)(%ld))"
		    ,(long)llbuf,buf.obj.i);
	    G__getitem(llbuf);
	    strcat(temp,llbuf);
	  }
	  else if(strcmp(G__struct.name[buf.tagnum],"G__ulonglong")==0) {
	    sprintf(llbuf
		    ,"G__printformatull((char*)(%ld),\"%%llu\",(void*)(%ld))"
		    ,(long)llbuf,buf.obj.i);
	    G__getitem(llbuf);
	    strcat(temp,llbuf);
	  }
	  else if(strcmp(G__struct.name[buf.tagnum],"G__longdouble")==0) {
	    sprintf(llbuf
		    ,"G__printformatld((char*)(%ld),\"%%LG\",(void*)(%ld))"
		    ,(long)llbuf,buf.obj.i);
	    G__getitem(llbuf);
	    strcat(temp,llbuf);
	  }
	}
	else
	  G__setiparseobject(&buf,temp);
      } else
#endif
      if(buf.obj.i<0) 
	sprintf(temp,"(class %s)(%ld)" 
		,G__fulltagname(buf.tagnum,1) ,buf.obj.i);
      else
	sprintf(temp,"(class %s)%ld" ,G__fulltagname(buf.tagnum,1) ,buf.obj.i);
      break;
    case 'u':
      if(buf.obj.i<0)
	sprintf(temp,"(union %s)(%ld)"
		,G__fulltagname(buf.tagnum,1) ,buf.obj.i);
      else
	sprintf(temp,"(union %s)%ld" ,G__fulltagname(buf.tagnum,1) ,buf.obj.i);
      break;
    case 'e':
      sprintf(temp,"(enum %s)%d",G__fulltagname(buf.tagnum,1),(int)buf.obj.i);
      break;
    default:
      if(buf.obj.i<0)
	sprintf(temp,"(unknown %s)(%ld)" 
		,G__struct.name[buf.tagnum] ,buf.obj.i);
      else 
	sprintf(temp,"(unknown %s)%ld" ,G__struct.name[buf.tagnum] ,buf.obj.i);
      break;
    }
    break;
  case 'U':
    switch(G__struct.type[buf.tagnum]) {
    case 's':
      sprintf(temp,"(struct %s*)0x%lx" ,G__fulltagname(buf.tagnum,1),buf.obj.i);
      break;
    case 'c':
      sprintf(temp,"(class %s*)0x%lx" ,G__fulltagname(buf.tagnum,1),buf.obj.i);
      break;
    case 'u':
      sprintf(temp,"(union %s*)0x%lx" ,G__fulltagname(buf.tagnum,1) ,buf.obj.i);
      break;
    case 'e':
      sprintf(temp,"(enum %s*)0x%lx" ,G__fulltagname(buf.tagnum,1) ,buf.obj.i);
      break;
    default:
      sprintf(temp,"(unknown %s*)0x%lx",G__fulltagname(buf.tagnum,1),buf.obj.i);
      break;
    }
    break;
  case 'w':
    G__logicstring(buf,1,temp2);
    sprintf(temp,"(logic)0b%s",temp2);
    break;
  default:
    if(buf.obj.i<0)
      sprintf(temp,"(unknown)(%ld)",buf.obj.i);
    else
      sprintf(temp,"(unknown)%ld",buf.obj.i);
    break;
  }

  if(isupper(buf.type)) {
    int i;
    char *p;
    char sbuf[G__ONELINE];
    p = strchr(temp,'*');
    switch(buf.obj.reftype.reftype) {
    case G__PARAP2P:
      strcpy(sbuf,p);
      strcpy(p+1,sbuf);
      break;
    case G__PARAP2P2P:
      strcpy(sbuf,p);
      *(p+1) = '*';
      strcpy(p+2,sbuf);
      break;
    case G__PARANORMAL:
      break;
    default:
#ifndef G__OLDIMPLEMENTATION707
      strcpy(sbuf,p);
      for(i=G__PARAP2P-1;i<buf.obj.reftype.reftype;i++) *(p+i)='*';
      strcpy(p+buf.obj.reftype.reftype-G__PARAP2P+1,sbuf);
#endif
      break;
    }
  }

  return(temp);
}


/****************************************************************
* G__access2string()
*
****************************************************************/
char *G__access2string(caccess)
int caccess;
{
  switch(caccess) {
  case G__PRIVATE: return("private:");
  case G__PROTECTED: return("protected:");
  case G__PUBLIC: return("public:");
  }
  return("");
}

/****************************************************************
* G__tagtype2string()
*
****************************************************************/
char *G__tagtype2string(tagtype)
int tagtype;
{
  switch(tagtype) {
  case 'c': return("class");
  case 's': return("struct");
  case 'e': return("enum");
  case 'u': return("union");
#ifndef G__OLDIMPLEMENTATION612
  case 'n': return("namespace");
#endif
  case  0 : return("(unknown)");  
  }
  G__genericerror("Internal error: Unexpected tagtype G__tagtype2string()");
  return("");
}

/****************************************************************
* G__fulltagname()
*
* return full tagname, if mask_dollar=1, $ for the typedef class
* is omitted
****************************************************************/
char *G__fulltagname(tagnum,mask_dollar)
int tagnum;
int mask_dollar;
{
#if !defined(G__OLDIMPLEMENTATION1823)
  static char string[G__LONGLINE];
#elif !defined(G__OLDIMPLEMENTATION711)
  static char string[G__ONELINE];
#else
  static char string[G__MAXNAME*2];
#endif
  int p_tagnum[G__MAXBASE];
  int pt;
  int len=0;
  int os;

  /* enclosed class scope , need to go backwards */
  p_tagnum[pt=0] = G__struct.parent_tagnum[tagnum];
  while(0<=p_tagnum[pt]) {
    p_tagnum[pt+1] = G__struct.parent_tagnum[p_tagnum[pt]];
    ++pt;
  }
  while(pt) {
    --pt;
    if('$'==G__struct.name[p_tagnum[pt]][0]) os = 1*mask_dollar;
    else                                     os = 0;
#define G__OLDIMPLEMENTATION1503
#ifndef G__OLDIMPLEMENTATION1503
    if(-1!=G__struct.defaulttypenum[p_tagnum[pt]]) 
      sprintf(string+len,"%s::"
	      ,G__newtype.name[G__struct.defaulttypenum[p_tagnum[pt]]]);
    else
      sprintf(string+len,"%s::",G__struct.name[p_tagnum[pt]]+os);
#else
    sprintf(string+len,"%s::",G__struct.name[p_tagnum[pt]]+os);
#endif
    len=strlen(string);
  }
  if('$'==G__struct.name[tagnum][0]) os = 1*mask_dollar;
  else                               os = 0;
#ifndef G__OLDIMPLEMENTATION1503
  if(-1!=G__struct.defaulttypenum[tagnum]) 
    sprintf(string+len,"%s",G__newtype.name[G__struct.defaulttypenum[tagnum]]);
  else
    sprintf(string+len,"%s",G__struct.name[tagnum]+os);
#else
  sprintf(string+len,"%s",G__struct.name[tagnum]+os);
#endif

  return(string);
}


/**************************************************************************
* G__type2string()
*
**************************************************************************/
char *G__type2string(type,tagnum,typenum,reftype,isconst)
int type,tagnum,typenum,reftype,isconst;
{
#ifndef G__OLDIMPLEMENTATION711
  static char stringbuf[G__LONGLINE];
#else
  static char stringbuf[G__MAXNAME*2];
#endif
  char *string;
  int len;
  /* int p_tagnum[G__MAXBASE]; */
  /* int pt; */
  int i;
#ifndef G__OLDIMPLEMENTATION1967
  int ref = G__REF(reftype);
  reftype = G__PLVL(reftype);
#endif

  string=stringbuf;
  if(isconst&G__CONSTVAR
#ifndef G__OLDIMPLEMENTATION1988
     && (-1==typenum || !(isconst&G__newtype.isconst[typenum]))
#endif
     ) {
    strcpy(string,"const ");
    string+=6;
  }
#ifndef G__OLDIMPLEMENTATION1836
  if(-1==typenum && tagnum>=0) { 
    char *ss = G__struct.name[tagnum];
    if(strcmp(ss,"G__longlong")==0 && !G__defined_macro("G__LONGLONGTMP")) {
      strcpy(stringbuf,"long long");
      return(stringbuf);
    }
    if(strcmp(ss,"G__ulonglong")==0 && !G__defined_macro("G__LONGLONGTMP")) {
      strcpy(stringbuf,"unsigned long long");
      return(stringbuf);
    }
    if(strcmp(ss,"G__longdouble")==0 && !G__defined_macro("G__LONGLONGTMP")) {
      strcpy(stringbuf,"long double");
      return(stringbuf);
    }
  }
#endif
#ifndef G__OLDIMPLEMENTATION1503
  if(-1==typenum && tagnum>=0 && -1!=G__struct.defaulttypenum[tagnum] &&
     'u'==G__newtype.type[G__struct.defaulttypenum[tagnum]]) {
    typenum = G__struct.defaulttypenum[tagnum];
  }
#endif
  if(-1!=typenum) {
    if(0<=G__newtype.parent_tagnum[typenum]) 
      sprintf(string,"%s::%s"
	      ,G__fulltagname(G__newtype.parent_tagnum[typenum],1)
	      ,G__newtype.name[typenum]);
    else
      sprintf(string,"%s",G__newtype.name[typenum]);
#ifndef G__OLDIMPLEMENTATION1329
    if(G__newtype.nindex[typenum]) {
      int pointlevel = 0;
      if(isupper(type)) pointlevel = 1;
      switch(reftype) {
      case G__PARANORMAL:
      case G__PARAREFERENCE:
	break;
      default:
	pointlevel = reftype;
	break;
      }
      pointlevel -= G__newtype.nindex[typenum];
      switch(pointlevel) {
      case 0:
	type = tolower(type);
	if(G__PARAREFERENCE!=reftype) reftype = G__PARANORMAL;
	break;
      case 1:
	type = toupper(type);
	if(G__PARAREFERENCE!=reftype) reftype = G__PARANORMAL;
	break;
      default:
#ifndef G__OLDIMPLEMENTATION1819
	if(pointlevel>0) {
	  type = toupper(type);
	  reftype = pointlevel;
	}
	else {
	}
#else
	type = toupper(type);
	reftype = pointlevel;
#endif
	break;
      }
    }
#endif
#ifndef G__OLDIMPLEMENTATION720
    if(isupper(G__newtype.type[typenum])) {
      switch(G__newtype.reftype[typenum]) {
      case G__PARANORMAL:
      case G__PARAREFERENCE:
	if(isupper(type)) {
	  switch(reftype) {
	  case G__PARAREFERENCE:
	  case G__PARANORMAL: type=tolower(type); break;
	  case G__PARAP2P:    reftype=G__PARANORMAL; break;
	  default:            --reftype; break;
	  }
	}
	else {
#ifndef G__OLDIMPLEMENTATION1967
	  G__type2string(type,tagnum,-1,reftype,isconst);
	  /* return buffer is static. So, stringbuf is set above */
	  goto endof_type2string;
#else
	  return(G__type2string(type,tagnum,-1,reftype,isconst));
#endif
	}
	break;
	
	break;
      default:
#if !defined(G__OLDIMPLEMENTATION2191)
	if('1'==type) {
	  switch(reftype) {
	  case G__PARAREFERENCE:
	  case G__PARANORMAL: type=tolower(type); break;
	  case G__PARAP2P:    reftype=G__PARANORMAL; break;
	  default: --reftype; break;
	  }
	} else 
#elif !defined(G__OLDIMPLEMENTATION729)
	if('Q'==type) {
	  if(isupper(type)) {
	    switch(reftype) {
	    case G__PARAREFERENCE:
	    case G__PARANORMAL: type=tolower(type); break;
	    case G__PARAP2P:    reftype=G__PARANORMAL; break;
	    default: --reftype; break;
	    }
	  }
	} else 
#endif
	if(islower(type) || G__newtype.reftype[typenum]>reftype) {
#ifndef G__OLDIMPLEMENTATION1967
	  G__type2string(type,tagnum,-1,reftype,isconst);
	  goto endof_type2string;
#else
	  return(G__type2string(type,tagnum,-1,reftype,isconst));
#endif
	}
	else if(G__newtype.reftype[typenum]==reftype) {
	  reftype = G__PARANORMAL;
	  type=tolower(type);
	}
	else if(G__newtype.reftype[typenum]+1==reftype) {
	  reftype = G__PARANORMAL;
	}
	else {
	  reftype = G__PARAP2P + reftype-G__newtype.reftype[typenum]-2;
	}
	break;
      }
    }
#else
    /* Bug fix, if defined type is already pointer */
    if(isupper(G__newtype.type[typenum])) type=tolower(type);
#endif
  }
  else if(-1!=tagnum) {
#ifndef G__OLDIMPLEMENTATION1139
    if(tagnum>=G__struct.alltag || !G__struct.name[tagnum]) {
      strcpy(stringbuf,"(invalid_class)");
      return(stringbuf);
    }
#endif
    if('$'==G__struct.name[tagnum][0]) {
      if('\0'==G__struct.name[tagnum][1]) {
	G__ASSERT('e'==G__struct.type[tagnum]);
	sprintf(string,"enum "); 
	len=5;
      }
      else {
	len=0;
      }
    }
    else {
      if(G__CPPLINK==G__globalcomp
#ifndef G__OLDIMPLEMENTATION772
	 || G__iscpp
#endif
	 ) {
	len=0;
      }
      else {
	switch(G__struct.type[tagnum]) {
	case 'e': sprintf(string,"enum "); break;
	case 'c': sprintf(string,"class "); break;
	case 's': sprintf(string,"struct "); break;
	case 'u': sprintf(string,"union "); break;
#ifndef G__OLDIMPLEMENTATION612
	case 'n': sprintf(string,"namespace "); break;
#endif
#ifndef G__OLDIMPLEMENTATION612
	case 'a': strcpy(string,""); break;
#endif
	case 0: sprintf(string,"(unknown) "); break;
	}
	len=strlen(string);
      }
    }
    sprintf(string+len,"%s",G__fulltagname(tagnum,1));
  }
  else {
    switch(tolower(type)) {
    case 'b': strcpy(string,"unsigned char"); break;
    case 'c': strcpy(string,"char"); break;
    case 'r': strcpy(string,"unsigned short"); break;
    case 's': strcpy(string,"short"); break;
    case 'h': strcpy(string,"unsigned int"); break;
    case 'i': strcpy(string,"int"); break;
    case 'k': strcpy(string,"unsigned long"); break;
    case 'l': strcpy(string,"long"); break;
#ifndef G__OLDIMPLEMENTATION1604
    case 'g': strcpy(string,"bool"); break;
#endif
#ifndef G__OLDIMPLEMENTATION2189
    case 'n': strcpy(string,"long long"); break;
    case 'm': strcpy(string,"unsigned long long"); break;
    case 'q': strcpy(string,"long double"); break; 
#endif
    case 'f': strcpy(string,"float"); break;
    case 'd': strcpy(string,"double"); break;
#ifndef G__OLDIMPLEMENTATION2191
    case '1': 
#else
    case 'q': 
#endif
    case 'y': strcpy(string,"void"); break;
    case 'e': strcpy(string,"FILE"); break;
    case 'u': strcpy(string,"enum");
#ifdef G__OLDIMPLEMENTATION1055
      G__fprinterr(G__serr,"%c %d %d %d\n",type,tagnum,typenum,reftype);
      /* Must be redundant */
      G__genericerror("Internal error: G__type2string()");
#endif
      break;
#ifndef G__OLDIMPLEMENTATION904
    case 't':
#endif
#ifndef G__OLDIMPLEMENTATION2191
    case 'j':
#else
    case 'm':
#endif
    case 'p': sprintf(string,"#define"); return(string);
    case 'o': string[0]='\0'; /* sprintf(string,""); */ return(string);
    case 'a': 
      /* G__ASSERT(isupper(type)); */
      strcpy(string,"G__p2memfunc");
      type=tolower(type);
      break;
    default:  strcpy(string,"(unknown)"); break;
    }
  }

  switch(type) { /* before tolower(type) */
/* #define G__OLDIMPLEMENTATION785 */
#ifdef G__OLDIMPLEMENTATION785 /* Not sure why this doesn't work, typeinfo.C */
  case 'f':
  case 'd':
#endif
  case 'q':
  case 'a':
    break;
  default:
    if(isupper(type)) {
      if((isconst&G__PCONSTVAR)&&G__PARANORMAL==reftype) 
	strcpy(string+strlen(string)," *const");
#ifdef G__OLDIMPLEMENTATION1859_YET
      else if((isconst&G__PCONSTVAR)&&G__PARAREFERENCE==reftype) {
	if((isconst&G__CONSTVAR)==0) strcpy(string+strlen(string)," const*&");
	else strcpy(string+strlen(string),"*&");
	reftype=G__PARANORMAL;
      }
#endif
      else 
	strcpy(string+strlen(string),"*");
    }
    switch(reftype) {
#ifndef G__OLDIMPLEMENTATION707
    case G__PARANORMAL: break;
#endif
    case G__PARAREFERENCE:
#ifndef G__OLDIMPLEMENTATION1112
      if(-1==typenum||G__PARAREFERENCE!=G__newtype.reftype[typenum]) {
	if(isconst&G__PCONSTVAR
#ifndef G__OLDIMPLEMENTATION1574
	   && 0==(isconst&G__CONSTVAR)
#endif
	   ) strcpy(string+strlen(string)," const&");
	else strcpy(string+strlen(string),"&");
      }
#else
      if(isconst&G__PCONSTVAR) strcpy(string+strlen(string)," const&");
      else strcpy(string+strlen(string),"&");
#endif
      break;
    case G__PARAP2P:
      if(isconst&G__PCONSTVAR) strcpy(string+strlen(string)," *const");
      else strcpy(string+strlen(string),"*");
      break;
    case G__PARAP2P2P:
      if(isconst&G__PCONSTVAR) strcpy(string+strlen(string)," **const");
      else strcpy(string+strlen(string),"**");
      break;
#ifndef G__OLDIMPLEMENTATION707
    default:
#ifndef G__OLDIMPLEMENTATION785
      if(reftype>10||reftype<0) break; /* workaround */
#endif
      if(isconst&G__PCONSTVAR) strcpy(string+strlen(string)," ");
      for(i=G__PARAP2P;i<=reftype;i++) strcpy(string+strlen(string),"*");
      if(isconst&G__PCONSTVAR) strcpy(string+strlen(string)," const");
      break;
#endif
    }
    break;
  }

#ifndef G__OLDIMPLEMENTATION1967
 endof_type2string:
  if(ref) {
    strcat(stringbuf,"&");
  }
#endif

#ifndef G__OLDIMPLEMENTATION2222
  if(strlen(stringbuf)>=sizeof(stringbuf)) {
    G__fprinterr(G__serr,
 "Error in G__type2sting: string length (%d) greater than buffer length (%d)!",
		 strlen(stringbuf),
		 sizeof(stringbuf));
    G__genericerror((char*)NULL);  
  }
#endif
  return(stringbuf);
}

/**************************************************************************
* G__val2pointer()
*
**************************************************************************/
int G__val2pointer(result7)
G__value *result7;
{
  if(0==result7->ref) {
    G__genericerror("Error: incorrect use of referencing operator '&'");
    return(1);
  }

  result7->type=toupper(result7->type);
  result7->obj.i=result7->ref;
  result7->ref=0;

#ifdef G__ASM
  if(G__asm_noverflow) {
#ifdef G__ASM_DBG
    G__fprinterr(G__serr,"%3x: TOPNTR\n",G__asm_cp);
#endif
    G__asm_inst[G__asm_cp]=G__TOPNTR;
    G__inc_cp_asm(1,0);
  }
#endif

  return(0);
}




#ifdef G__NEVER
/******************************************************************
* double G__atodouble(string)
*
* Called by
*   nothing    G__getitem() used to call, but commented out now
*
******************************************************************/
double G__atodouble(string)
char *string;
{
  int lenstring;
  int ig16=0,ig26=0;
  char exponent[G__ONELINE];
  double expo,polarity=1;
  double result5=0.0;
  double ratio=0.1;
  
  lenstring=strlen(string);
  if(string[ig16]=='-') {
    polarity = -1;
    ig16++;
  }
  while((isdigit(string[ig16]))&&(ig16<lenstring)) {
    result5 = result5*10 + (double)(string[ig16++]-'0');
  }
  if(string[ig16]=='.') {
    ig16++;
    while((isdigit(string[ig16]))&&(ig16<lenstring)) {
      result5 += ratio * (double)(string[ig16++]-'0');
      ratio /= 10;
    }
  }
  if(ig16<lenstring) {
    switch(string[ig16]) {
    case 'e':
    case 'E':
      ig16++;
      while(ig16<lenstring) exponent[ig26++]=string[ig16++];
      exponent[ig26]='\0';
      expo = (double)(atoi(exponent));
      expo = exp(expo*log(10.0));
      result5 *= expo;
      break;
    case 't':
    case 'T':
      result5 *= 1e12;
      break;
    case 'g':
    case 'G':
      result5 *= 1e9;
      break;
    case 'M':
      result5 *= 1e6;
      break;
    case 'k':
    case 'K':
      result5 *= 1e3;
      break;
    case 'm':
      result5 *= 1e-3;
      break;
    case 'u':
    case 'U':
      result5 *= 1e-6;
      break;
    case 'n':
    case 'N':
      result5 *= 1e-9;
      break;
    case 'p':
    case 'P':
      result5 *= 1e-12;
      break;
    case 'f':
    case 'F':
      result5 *= 1e-15;
      break;
    case 'a':
    case 'A':
      result5 *= 1e-18;
      break;
    }
  }
  return(polarity*result5);
}
#endif



/******************************************************************
* char *G__getbase(value,base,digit)
*
* Called by
*   G__logicstring()
*   G__logicstring()
*   G__logicstring()
*   G__logicstring()
*
******************************************************************/
char *G__getbase(expression,base,digit,result1)
unsigned int expression;
int base,digit;
char *result1;
{
  char result[G__MAXNAME];
  int ig18=0,ig28=0;
  unsigned int onedig,value;  /* bug fix  3 mar 1993 */

  value=expression;

  while((ig28<digit)||((digit==0)&&(value!=0))) {
    onedig = value % base ;
    result[ig28] = G__getdigit(onedig);
    value = (value - onedig)/base;
    ig28++ ;
  }
  ig28-- ;

  /*
  result1[ig18++]='0' ;
  switch(base) {
  case 2:
	  result1[ig18++]='b' ;
	  break;
  case 4:
	  result1[ig18++]='q' ;
	  break;
  case 8:
	  result1[ig18++]='o' ;
	  break;
  case 10:
	  result1[ig18++]='d' ;
	  break;
  case 16:
	  result1[ig18++]='x' ;
	  break;
  default:
	  result1[ig18++]= base + '0' ;
  }
  */

  while(0<=ig28) {
	  result1[ig18++] = result[ig28--] ;
  }
  if(ig18==0) {
	  result1[ig18++]='0';
  }
  result1[ig18]='\0';

  return(result1);
}



/******************************************************************
* G__getdigit(number)
*
* Called by
*   G__getbase
*
******************************************************************/
int G__getdigit(number)
unsigned int number;
{
  switch(number) {
  case 0:
  case 1:
  case 2:
  case 3:
  case 4:
  case 5:
  case 6:
  case 7:
  case 8:
  case 9:
	  return( number + '0' );
  case 10:
	  return('a');
  case 11:
	  return('b');
  case 12:
	  return('c');
  case 13:
	  return('d');
  case 14:
	  return('e');
  case 15:
	  return('f');
  default:
	  return('x');
  }
  /* return('x'); */
}



/******************************************************************
* G__value G__checkBase(string,known4)
*
* Called by
*   G__getitem()
*
******************************************************************/
G__value G__checkBase(string,known4)
char *string;
int *known4;
{
  G__value result4;
  int n=0,nchar,base=0;
  long int value=0,tristate=0;
  char type;
  int unsign = 0;

  /********************************************
  * default type int
  ********************************************/
  type = 'i';

  /********************************************
  * count number of characters
  ********************************************/
  nchar = strlen(string);


  /********************************************
  * scan string
  ********************************************/
  while(n<nchar) {
    
    /********************************************
     * error check, never happen because same
     * condition is already checked in G__getitem()
     * Thus, this condition can be removed
     ********************************************/
    if(string[0]!='0') {
#ifndef G__FONS31
      G__fprinterr(G__serr,"Error: G__checkBase(%s) " ,string);
      G__genericerror((char*)NULL);
#else
      G__fprinterr(G__serr,"Error: G__checkBase(%s) FILE:%s LINE:%d\n"
	      ,string,G__ifile.name,G__ifile.line_number);
#endif
      return(G__null);
    }
    else { 
      /**********************************************
       * check base specification
       **********************************************/
      switch(string[++n]) {
      case 'b':
      case 'B':
	/*******************************
	 * binary base doesn't exist in
	 * ANSI. Original enhancement
	 *******************************/
	base=2;
	break;
      case 'q':
      case 'Q':
	/******************************
	 * quad specifier may not exist
	 * in ANSI
	 ******************************/
	base=4;
	break;
      case 'o':
      case 'O':
	/******************************
	 * octal specifier may not exist
	 * in ANSI
	 ******************************/
	base=8;
	break;
      case 'h':
      case 'H':
	/*******************************
	 * 0h and 0H doesn't exist in
	 * ANSI. Original enhancement
	 *******************************/
      case 'x':
      case 'X':
	base=16;
	break;
      default:
	/*******************************
	 * Octal if other specifiers,
	 * namely a digit
	 *******************************/
#ifndef G__OLDIMPLEMENTATION426
	base=8;
	--n;
#else
	base=10;
#endif
	break;
      }
      
      /********************************************
       * initialize value
       ********************************************/
      value=0; tristate=0;
      
      /********************************************
       * increment scanning pointer
       ********************************************/
      n++;
      
      /********************************************
       * scan constant expression body
       ********************************************/
      while((string[n]!=' ')&&(string[n]!='\t')&&(n<nchar)) {
	  switch(string[n]) {
	  case '0':
	    /*
	      case 'L':
	      case 'l':
	      */
	    value=value*base;
	    tristate *= base;
	    break;
	  case '1':
	    /*
	      case 'H':
	      case 'h':
	      */
	    value=value*base+1;
	    tristate *= base;
	    break;
	  case '2':
	    value=value*base+2;
	    tristate *= base;
	    break;
	  case '3':
	    value=value*base+3;
	    tristate *= base;
	    break;
	  case '4':
	    value=value*base+4;
	    tristate *= base;
	    break;
	  case '5':
	    value=value*base+5;
	    tristate *= base;
	    break;
	  case '6':
	    value=value*base+6;
	    tristate *= base;
	    break;
	  case '7':
	    value=value*base+7;
	    tristate *= base;
	    break;
	  case '8':
	    value=value*base+8;
	    tristate *= base;
	    break;
	  case '9':
	    value=value*base+9;
	    tristate *= base;
	    break;
	  case 'a':
	  case 'A':
	    value=value*base+10;
	    tristate *= base;
	    break;
	  case 'b':
	  case 'B':
	    value=value*base+11;
	    tristate *= base;
	    break;
	  case 'c':
	  case 'C':
	    value=value*base+12;
	    tristate *= base;
	    break;
	  case 'd':
	  case 'D':
	    value=value*base+13;
	    tristate *= base;
	    break;
	  case 'e':
	  case 'E':
	    value=value*base+14;
	    tristate *= base;
	    break;
	  case 'f':
	  case 'F':
	    value=value*base+15;
	    tristate *= base;
	    break;
	  case 'l':
	  case 'L':
	    /********************************************
	     * long
	     ********************************************/
	    type = 'l';
	    break;
	  case 'u':
	  case 'U':
	    /********************************************
	     * unsigned
	     ********************************************/
	    unsign = 1;
	    break;
#ifdef G__NEVER
	  case 's':
	  case 'S':
	    /********************************************
	     * short, may not exist in ANSI
	     * and doesn't work fine. 
	     * shoud be done as casting
	     ********************************************/
	    type = 's';
	    break;
#endif
	  case 'x':
	  case 'X':
	    /***************************************
	     * Not ANSI
	     * enhancement for tristate logic expression
	     ***************************************/
	    value *= base;
	    tristate=tristate*base+(base-1);
	    break;
	  case 'z':
	  case 'Z':
	    /***************************************
	     * Not ANSI
	     * enhancement for tristate logic expression
	     ***************************************/
	    value=value*base+(base-1);
	    tristate=tristate*base+(base-1);
	    break;
	  default:
	    value=value*base;
#ifndef G__FONS31
	    G__fprinterr(G__serr, "Error: unexpected character in expression %s "
		    ,string);
	    G__genericerror((char*)NULL);
#else
	    G__fprinterr(G__serr,
	    "Error: unexpected character in expression %s FILE:%s LINE:%d\n"
		    ,string ,G__ifile.name,G__ifile.line_number);
#endif
	    break;
	  }
	  n++;
	}
      }
    }
  if(value>-1) {
    *known4=1;
  }

  /*******************************************************
  * store constant value and type to result4
  *******************************************************/
  type = type - unsign ;
  G__letint(&result4,type,(long)value);
  result4.tagnum = -1;
  result4.typenum = -1;

  /*******************************************************
  * Not ANSI
  * if tristate logic , specify it
  *******************************************************/
  if((base==2)||(tristate!=0)) {
    *(&result4.obj.i+1) = tristate;
    result4.type = 'w';
  }

  return(result4);
}


/******************************************************************
* int G__isfloat(string)
*
* Called by
*   G__getitem()
*
******************************************************************/
int G__isfloat(string,type)
char *string;
int *type;
{
  int ig17=0;
  int c;
  int result=0, unsing = 0 ;
  
  /*************************************************************
   * default type is int
   *************************************************************/
  *type = 'i';
  
  /**************************************************************
   * check type of constant expression
   **************************************************************/
  while((c=string[ig17++])!='\0') {
    switch(c) {
    case '.':
    case 'e':
    case 'E':
      /******************************
       * double
       ******************************/
      result = 1;
      *type = 'd';
      break;
    case 'f':
    case 'F':
      /******************************
       * float
       ******************************/
      result = 1;
      *type = 'f';
      break;
    case 'l':
    case 'L':
      /******************************
       * long
       ******************************/
#ifndef G__OLDIMPLEMENTATION1874
      if('l'==*type) *type = 'u';
      else           *type = 'l';
#else
      *type = 'l';
#endif
      break;
#ifdef G__NEVER
    case 's':
    case 'S':
      /******************************
       * short, This may not be
       * included in ANSI
       * and does't work fine.
       * should be done by casting
       ******************************/
      *type = 's';
      break;
#endif
    case 'u':
    case 'U':
      /******************************
       * unsigned
       ******************************/
      unsing = 1;
      break;
#ifndef G__OLDIMPLEMENTATION1014
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
    case '+':
    case '-':
      break;
    default:
      G__fprinterr(G__serr,"Warning: Illegal numerical expression %s",string);
      G__printlinenum();
      break;
#endif
    }
  }
  
  /**************************************************************
   * check illegal type specification
   **************************************************************/
  if(unsing) {
    switch(*type) {
    case 'd':
    case 'f':
#ifndef G__FONS31
      G__fprinterr(G__serr,
	      "Error: unsigned can not be specified for float or double %s "
	      ,string );
      G__genericerror((char*)NULL);
#else
      G__fprinterr(G__serr,"Error: unsigned can not be specified for float or double %s FILE:%s LINE:%d\n"
	      ,string ,G__ifile.name ,G__ifile.line_number);
#endif
      break;
    default:
      *type = *type - unsing ;
      break;
    }
  }
  
  /**************************************************************
   * return 1 if float or double
   **************************************************************/
  return(result);
}

/* #ifdef G__NEVER */
/******************************************************************
* int G__isoperator(c)
*
* Called by
*   nothing
******************************************************************/
int G__isoperator(c)
int c;
{
  switch(c) {
  case '+':
  case '-':
  case '*':
  case '/':
  case '@':
  case '&':
  case '%':
  case '|':
  case '^':
  case '>':
  case '<':
  case '=':
  case '~':
  case '!':
#ifdef G__NEVER
  case '.':
  case '(':
  case ')':
  case '[':
  case ']':
#endif
    return(1);
  default:
    return(0);
  }
}
/* #endif */


/******************************************************************
* int G__isexponent(expression4,lenexpr)
*
* Called by
*   G__getexpr()    to identify power and operator
*
******************************************************************/
int G__isexponent(expression4,lenexpr)
char *expression4;
int lenexpr;
{


  int c;
  int flag=0;

  G__ASSERT(lenexpr>1); /* must be guaranteed in G__getexpr() */

  if(toupper(expression4[--lenexpr])=='E') {
    while( isdigit(c=expression4[--lenexpr]) || '.'==c  ) {
      if(lenexpr<1) return(1);
      if('.'!=c) flag=1;
    }
    if( flag && (G__isoperator(c) || c==')' ) ) {
      return(1);
    }
    else {
      return(0);
    }
  }
  else {
    switch(expression4[lenexpr]) {
    case '*':
    case '/':
    case '%':
    case '@':
      return(1);
    }
    return(0);
  }
  
}


/******************************************************************
* int G__isvalue(temp)
*
* Called by
*   G__strip_quotation()   To identify char pointer and string
*
******************************************************************/
int G__isvalue(temp)
char *temp;
{
  if  ( (isdigit(temp[0])) ||((temp[0]=='-')&&(isdigit(temp[1])))) {
    return(1);
  }
  else {
    return(0);
  }
}

/******************************************************************
* G__string2type_body
*
******************************************************************/
G__value G__string2type_body(typenamin,noerror)
char *typenamin;
int noerror;
{
  char typenam[G__MAXNAME*2];
  char temp[G__MAXNAME*2];
  int len;
  int plevel=0;
  int rlevel=0;
  int flag;
  G__value result;
#ifndef G__OLDIMPLEMENTATION707
  int isconst=0;
#endif

  result = G__null;

#ifndef G__OLDIMPLEMENTATION758
  strcpy(typenam,typenamin);
#else
  if(strchr(typenamin,' ')) {
    int i=0,j=0;
    while(typenamin[i]) {
      if(' '!=typenamin[i]) typenam[j++]=typenamin[i];
      ++i;
    }
    typenam[j] = '\0';
  }
  else {
    strcpy(typenam,typenamin);
  }
#endif

#ifndef G__OLDIMPLEMENTATION758
  if(strncmp(typenam,"volatile ",9)==0) {
    strcpy(temp,typenam+9);
    strcpy(typenam,temp);
  } else
#endif
  if(strncmp(typenam,"volatile",8)==0) {
    strcpy(temp,typenam+8);
    strcpy(typenam,temp);
  }

#ifndef G__OLDIMPLEMENTATION758
  if(strncmp(typenam,"const ",6)==0) {
    strcpy(temp,typenam+6);
    strcpy(typenam,temp);
    isconst=1;
  } else
#endif
#ifndef G__OLDIMPLEMENTATION717
  if(strncmp(typenam,"const",5)==0 &&
     -1==G__defined_tagname(typenam,2) && -1==G__defined_typename(typenam)) {
#else
  if(strncmp(typenam,"const",5)==0) {
#endif
    strcpy(temp,typenam+5);
    strcpy(typenam,temp);
#ifndef G__OLDIMPLEMENTATION707
    isconst=1;
#endif
  }

  len = strlen(typenam);
  do {
    switch(typenam[len-1]) {
    case '*':
      ++plevel;
      typenam[--len]='\0';
      flag=1;
      break;
    case '&':
      ++rlevel;
      typenam[--len]='\0';
      flag=1;
      break;
#ifndef G__OLDIMPLEMENTATION1557
    case ' ':
    case '\t':
    case '\n':
    case '\r':
    case '\f':
      typenam[--len]='\0';
      flag=1;
      break;
#endif
    default:
      flag=0;
      break;
    }
  } while(flag);


  switch(len) {
  case 3:
    if(strcmp(typenam,"int")==0) {
      result.type = 'i';
    }
    break;
  case 4:
    if(strcmp(typenam,"char")==0) {
      result.type='c';
      break;
    }
    if(strcmp(typenam,"long")==0) {
      result.type='l';
      break;
    }
    if(strcmp(typenam,"FILE")==0) {
      result.type='e';
      break;
    }
    if(strcmp(typenam,"void")==0) {
      result.type='y';
      break;
    }
#ifndef G__OLDIMPLEMENTATION1604
    if(strcmp(typenam,"bool")==0) {
      result.type='g';
      break;
    }
#endif
    break;
  case 5:
    if(strcmp(typenam,"short")==0) {
      result.type='s';
      break;
    }
    if(strcmp(typenam,"float")==0) {
      result.type='f';
      break;
    }
    break;
  case 6:
    if(strcmp(typenam,"double")==0) {
      result.type='d';
      break;
    }
    break;
  case 8:
    if(strcmp(typenam,"unsigned")==0) {
      result.type='h';
      break;
    }
#ifndef G__OLDIMPLEMENTATION2189
    if(strcmp(typenam,"longlong")==0) {
      result.type='n';
      break;
    }
#endif
    break;
#ifndef G__OLDIMPLEMENTATION2189
  case 9:
    if(strcmp(typenam,"long long")==0) {
      result.type='n';
      break;
    }
    break;
#endif
  case 10:
#ifndef G__OLDIMPLEMENTATION2189
    if(strcmp(typenam,"longdouble")==0) {
      result.type='q';
      break;
    }
    break;
#else
    if(strcmp(typenam,"longdouble")==0) {
      result.type='d';
      break;
    }
    break;
#endif
  case 11:
    if(strcmp(typenam,"unsignedint")==0) {
      result.type='h';
      break;
    }
#ifdef G__OLDIMPLEMENTATION1836
#ifndef G__OLDIMPLEMENTATION707
    if(strcmp(typenam,"long double")==0) {
      result.type='d';
      break;
    }
#endif
#endif
    break;
  case 12:
    if(strcmp(typenam,"unsignedchar")==0) {
      result.type='b';
      break;
    }
    if(strcmp(typenam,"unsignedlong")==0) {
      result.type='k';
      break;
    }
#ifndef G__OLDIMPLEMENTATION707
    if(strcmp(typenam,"unsigned int")==0) {
      result.type='h';
      break;
    }
#endif
    break;
  case 13:
    if(strcmp(typenam,"unsignedshort")==0) {
      result.type='r';
      break;
    }
#ifndef G__OLDIMPLEMENTATION707
    if(strcmp(typenam,"unsigned char")==0) {
      result.type='b';
      break;
    }
    if(strcmp(typenam,"unsigned long")==0) {
      result.type='k';
      break;
    }
#endif
    break;
#ifndef G__OLDIMPLEMENTATION707
  case 14:
    if(strcmp(typenam,"unsigned short")==0) {
      result.type='r';
      break;
    }
    break;
#endif
#ifndef G__OLDIMPLEMENTATION2189
  case 16:
    if(strcmp(typenam,"unsignedlonglong")==0) {
      result.type='m';
      break;
    }
    break;
  case 18:
    if(strcmp(typenam,"unsigned long long")==0) {
      result.type='m';
      break;
    }
    break;
#endif
  }

  if(0==result.type) {
    if(strncmp(typenam,"struct",6)==0) {
      result.type='u';
      result.tagnum=G__defined_tagname(typenam+6,0);
    }
    else if(strncmp(typenam,"class",5)==0) {
      result.type='u';
      result.tagnum=G__defined_tagname(typenam+5,0);
    }
    else if(strncmp(typenam,"union",5)==0) {
      result.type='u';
      result.tagnum=G__defined_tagname(typenam+5,0);
    }
    else if(strncmp(typenam,"enum",4)==0) {
      result.type='i';
      result.tagnum=G__defined_tagname(typenam+4,0);
    }
  }

  if(0==result.type) {
    result.typenum=G__defined_typename(typenam);
    if(result.typenum != -1) {
      result.tagnum=G__newtype.tagnum[result.typenum];
      result.type=G__newtype.type[result.typenum];
      if(result.tagnum != -1 && G__struct.type[result.tagnum]=='e') {
	result.type='i';
      }
    }
    else {
      if(0==noerror) {
	result.tagnum=G__defined_tagname(typenam,0);
	if(result.tagnum== -1) result.type='Y'; /* checked */
	else                   result.type='u';
      }
      else {
	result.tagnum=G__defined_tagname(typenam,noerror);
	if(result.tagnum== -1) result.type=0; /* checked */
	else                   result.type='u';
      }
    }
  }

  if(result.type) {
#ifndef G__OLDIMPLEMENTATION707
    if(rlevel) result.obj.reftype.reftype = G__PARAREFERENCE;
    switch(plevel) {
    case 0:
      break;
    case 1:
      result.type = toupper(result.type);
      break;
    default:
      result.type = toupper(result.type);
#ifndef G__OLDIMPLEMENTATION2049
      if(rlevel) {
	result.obj.reftype.reftype = G__PARAREFP2P + plevel-2;
      }
      else {
	result.obj.reftype.reftype = G__PARAP2P + plevel-2;
      }
#else
      result.obj.reftype.reftype = G__PARAP2P + plevel-2;
#endif
      break;
    }
    result.obj.i = isconst; /* borrowing space of the value */
#else
    if(plevel) result.type = toupper(result.type);
    if(rlevel) result.ref = G__PVOID;
#endif
  }

  return(result);
}

/******************************************************************
* G__string2type
*
******************************************************************/
G__value G__string2type(typenamin)
char *typenamin;
{
#ifndef G__OLDIMPLEMENTATION1042
  int store_var_type = G__var_type;
  G__value buf = G__string2type_body(typenamin,0);
  G__var_type = store_var_type;
  return(buf);
#else
  return(G__string2type_body(typenamin,0));
#endif
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
