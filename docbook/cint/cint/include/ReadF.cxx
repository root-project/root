/*****************************************************************************
* ReadFile.C
*****************************************************************************/

#ifdef __CINT__
#pragma security level0
#endif

#include "ReadF.h"

/*****************************************************************************
* Constructors and destructors
*****************************************************************************/
ReadFile::ReadFile(const char *filename)
{
  fp = fopen(filename,"r");
  initialize();
  if(fp) {
    openflag=1;
  }
  else {
    fprintf(stderr,"ReadFile: can not open %s\n",filename);
  }
}

ReadFile::ReadFile(FILE *fpin)
{
  fp = fpin;
  openflag=0;
  initialize();
  if((FILE*)NULL==fpin) {
    fprintf(stderr,"ReadFile: NULL pointer is given\n");
  }
}

ReadFile::~ReadFile()
{
  if(openflag && fp) fclose(fp);
  fp = (FILE*)NULL;
}


/*****************************************************************************
* set routines
*****************************************************************************/
void ReadFile::initialize()
{
  line = 0;
  argc = 0;
  setseparator(" \t\v");
  setendofline("");
#ifndef G__OLDIMPLEMENTATION1960
  setdelimitor("");
#endif
#ifndef G__OLDIMPLEMENTATION3000
  setquotation("");
#endif
}

void ReadFile::setseparator(const char *separatorin)
{
  strcpy(separator,separatorin); 
  lenseparator = strlen(separator);
}

#ifndef G__OLDIMPLEMENTATION1960
void ReadFile::setdelimitor(const char *delimitorin)
{
  strcpy(delimitor,delimitorin); 
  lendelimitor = strlen(delimitor);
}
#endif

void ReadFile::setendofline(const char *endoflinein)
{
  strcpy(endofline,endoflinein); 
  lenendofline = strlen(endofline);
}

#ifndef G__OLDIMPLEMENTATION3000
void ReadFile::setquotation(const char *quotationin)
{
  strcpy(quotation,quotationin); 
  lenquotation = strlen(quotation);
}
#endif

/*****************************************************************************
* Reading one line
*****************************************************************************/
int ReadFile::read()
{
  char *null_fgets;

  if(!fp) return(0);

  null_fgets=fgets(buf,MAX_LINE_LENGTH,fp);
  if(null_fgets!=NULL) {
    ++line;
    separatearg();
  }
  else {
    buf[0]='\0';;
    argbuf[0]='\0';
    argc=0;
    argv[0]=buf;
  }
  if(null_fgets==NULL) return(0);
  else                 return(1);
}

/*****************************************************************************
* Reading one word
*****************************************************************************/
int ReadFile::readword()
{
  int c;
  int i=0;
  int flag;

  if(!fp) return(0);

  flag = 1;
  while(flag) {
    c = fgetc(fp);
    if(EOF==c) return(c);
    if(isendofline(c)||isseparator(c)) {
      flag=1;
    }
    else {
      buf[i++]=c;
      flag=0;
    }
  }

  flag = 1;
  while(flag) {
    c = fgetc(fp);
    if(isendofline(c)||isseparator(c)) {
      buf[i]='\0';
      flag=0;
    }
    else {
      buf[i]=c;
    }
    ++i;
  }
  argv[0]=buf;
  argc=i;
  return(c);
}


#ifdef G__NEVER
/*****************************************************************************
* Separate argument
*****************************************************************************/
int ReadFile::regex(char *pattern,char *string)
{
  int i;
  regex_t re;
  if((char*)NULL==string) string = argv[0];
  i=regcomp(&re,pattern,REG_EXTENDED|REG_NOSUB);
  if(i!=0) return(0); 
  i=regexec(&re,string,(size_t)0,(regmatch_t*)NULL,0);
  regfree(&re);
  if(i!=0) return(0); 
  return(1); /* match */
}
#endif

/*****************************************************************************
* Separate argument
*****************************************************************************/
void ReadFile::separatearg(void)
{
  char *p;
  int i;
  int c;

  // change \n to \0 
  p=strchr(buf,'\n');
  if(p) *p='\0';

  // if user defined end of line
  for(i=0;i<lenendofline;i++) {
    p=strchr(buf,endofline[i]);
    if(p) *p='\0';
  }

  argv[0] = buf;
  strcpy(argbuf,buf);

  // separate argument
  argc = 0;
  p = argbuf;
  do {
    while(isseparator((c = *p)) && c) ++p;
#ifndef G__OLDIMPLEMENTATION3000
    if(isquotation(c)) {
      argv[++argc] = ++p;
      while(!isquotation((c = *p)) && c) {
        if(c=='\\') ++p;
        ++p;
      }
      *p = '\0';
      ++p;
    }
    else 
#endif
    if(c) {
      argv[++argc] = p;
#ifndef G__OLDIMPLEMENTATION1960
      while(!isseparator((c = *p)) && !isdelimitor(c) && c) ++p;
#else
      while(!isseparator((c = *p)) && c) ++p;
#endif
      *p = '\0';
      ++p;
    }
  } while(c) ;
}

/*****************************************************************************
* isendofline()
*****************************************************************************/
int ReadFile::isendofline(int c)
{
  int i;
  if('\n'==c || EOF==c) return(1);
  for(i=0;i<lenendofline;i++) {
    if(c==endofline[i]) return(1);
  }
  return(0);
}

/*****************************************************************************
* isseparator()
*****************************************************************************/
int ReadFile::isseparator(int c)
{
  int i;
  for(i=0;i<lenseparator;i++) {
    if(c==separator[i]) return(1);
  }
  return(0);
}

#ifndef G__OLDIMPLEMENTATION1960
/*****************************************************************************
* isdelimitor()
*****************************************************************************/
int ReadFile::isdelimitor(int c)
{
  int i;
  for(i=0;i<lendelimitor;i++) {
    if(c==delimitor[i]) return(1);
  }
  return(0);
}

#endif

#ifndef G__OLDIMPLEMENTATION3000
/*****************************************************************************
* isquotation()
*****************************************************************************/
int ReadFile::isquotation(int c)
{
  int i;
  for(i=0;i<lenquotation;i++) {
    if(c==quotation[i]) return(1);
  }
  return(0);
}

#endif

/*****************************************************************************
* disp()
*****************************************************************************/
void ReadFile::disp()
{
  int i;
  printf("%4d %3d ",line,argc);
  for(i=1;i<=argc;i++) {
    printf("| %s ",argv[i]);
  }
  printf("\n");
}


