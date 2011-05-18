/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file lib/posix/mktypes.c
 ************************************************************************
 * Description:
 *  Create POSIX related types in include/systypes.h
 ************************************************************************
 * Copyright(c) 1995~1999  Masaharu Goto (MXJ02154@niftyserve.or.jp)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <time.h>
#include <sys/stat.h>
#include <dirent.h>
#include <sys/utsname.h>
#include <sys/types.h>

const char *header="../../include/systypes.h";

/*******************************************************************
* testdup()
*******************************************************************/
int testdup(FILE *fp,char *iden)
{
  fprintf(fp,"#ifndef %s\n",iden);
  fprintf(fp,"#define %s\n",iden);
  return(0);
}

/*******************************************************************
* 
*******************************************************************/
#define UINT_TYPEDEF_PREFER_INT(fp,type,ctype)                          \
  if(sizeof(type)==sizeof(unsigned char))                               \
    fprintf(fp,"typedef unsigned char %s;\n",ctype);                    \
  else if(sizeof(type)==sizeof(unsigned short))                         \
    fprintf(fp,"typedef unsigned short %s;\n",ctype);                   \
  else if(sizeof(type)==sizeof(unsigned int))                           \
    fprintf(fp,"typedef unsigned int %s;\n",ctype);                     \
  else if(sizeof(type)==sizeof(unsigned long))                          \
    fprintf(fp,"typedef unsigned long %s;\n",ctype);                    \
  else if(sizeof(type)==sizeof(unsigned long)*2) {                      \
    fprintf(fp,"typedef struct %s {\n",ctype);                          \
    fprintf(fp,"  unsigned long l,u;\n");                               \
    fprintf(fp,"  %s(unsigned long i){l=i;u=0;}\n",ctype);              \
    fprintf(fp,"  void operator=(unsigned long i){l=i;u=0;}\n");        \
    fprintf(fp,"} %s;\n",ctype);                                        \
    fprintf(fp,"#pragma link off class %s;\n",ctype);                   \
    fprintf(fp,"#pragma link off typedef %s;\n",ctype);                 \
  }                                                                     \
  else {                                                                \
    fprintf(fp,"typedef struct %s {\n",ctype);                          \
    fprintf(fp,"  char dmy[%d];\n",(int)sizeof(type));                     \
    fprintf(fp,"} %s;\n",ctype);                                        \
    fprintf(fp,"#pragma link off class %s;\n",ctype);                   \
    fprintf(fp,"#pragma link off typedef %s;\n",ctype);                 \
  }

#define INT_TYPEDEF_PREFER_INT(fp,type,ctype)                           \
  if(sizeof(type)==sizeof(char))                                        \
    fprintf(fp,"typedef char %s;\n",ctype);                             \
  else if(sizeof(type)==sizeof(short))                                  \
    fprintf(fp,"typedef short %s;\n",ctype);                            \
  else if(sizeof(type)==sizeof(int))                                    \
    fprintf(fp,"typedef int %s;\n",ctype);                              \
  else if(sizeof(type)==sizeof(long))                                   \
    fprintf(fp,"typedef long %s;\n",ctype);                             \
  else if(sizeof(type)==sizeof(long)*2) {                               \
    fprintf(fp,"typedef struct %s {\n",ctype);                          \
    fprintf(fp,"  long l,u;\n");                                        \
    fprintf(fp,"  %s(long i){l=i;u=0;}\n",ctype);                       \
    fprintf(fp,"  void operator=(long i){l=i;u=0;}\n");                 \
    fprintf(fp,"} %s;\n",ctype);                                        \
    fprintf(fp,"#pragma link off class %s;\n",ctype);                   \
    fprintf(fp,"#pragma link off typedef %s;\n",ctype);                 \
  }                                                                     \
  else {                                                                \
    fprintf(fp,"typedef struct %s {\n",ctype);                          \
    fprintf(fp,"  char dmy[%d];\n",(int)sizeof(type));                  \
    fprintf(fp,"} %s;\n",ctype);                                        \
    fprintf(fp,"#pragma link off class %s;\n",ctype);                   \
    fprintf(fp,"#pragma link off typedef %s;\n",ctype);                 \
  }

#define UINT_TYPEDEF_PREFER_LONG(fp,type,ctype)                         \
  if(sizeof(type)==sizeof(unsigned char))                               \
    fprintf(fp,"typedef unsigned char %s;\n",ctype);                    \
  else if(sizeof(type)==sizeof(unsigned short))                         \
    fprintf(fp,"typedef unsigned short %s;\n",ctype);                   \
  else if(sizeof(type)==sizeof(unsigned long))                          \
    fprintf(fp,"typedef unsigned long %s;\n",ctype);                    \
  else if(sizeof(type)==sizeof(unsigned int))                           \
    fprintf(fp,"typedef unsigned int %s;\n",ctype);                     \
  else if(sizeof(type)==sizeof(unsigned long)*2) {                      \
    fprintf(fp,"typedef struct %s {\n",ctype);                          \
    fprintf(fp,"  unsigned long l,u;\n");                               \
    fprintf(fp,"  %s(unsigned long i){l=i;u=0;}\n",ctype);              \
    fprintf(fp,"  void operator=(unsigned long i){l=i;u=0;}\n");        \
    fprintf(fp,"} %s;\n",ctype);                                        \
    fprintf(fp,"#pragma link off class %s;\n",ctype);                   \
    fprintf(fp,"#pragma link off typedef %s;\n",ctype);                 \
  }                                                                     \
  else {                                                                \
    fprintf(fp,"typedef struct %s {\n",ctype);                          \
    fprintf(fp,"  char dmy[%d];\n",(int)sizeof(type));                  \
    fprintf(fp,"} %s;\n",ctype);                                        \
    fprintf(fp,"#pragma link off class %s;\n",ctype);                   \
    fprintf(fp,"#pragma link off typedef %s;\n",ctype);                 \
  }

#define INT_TYPEDEF_PREFER_LONG(fp,type,ctype)                          \
  if(sizeof(type)==sizeof(char))                                        \
    fprintf(fp,"typedef char %s;\n",ctype);                             \
  else if(sizeof(type)==sizeof(short))                                  \
    fprintf(fp,"typedef short %s;\n",ctype);                            \
  else if(sizeof(type)==sizeof(long))                                   \
    fprintf(fp,"typedef long %s;\n",ctype);                             \
  else if(sizeof(type)==sizeof(int))                                    \
    fprintf(fp,"typedef int %s;\n",ctype);                              \
  else if(sizeof(type)==sizeof(long)*2) {                               \
    fprintf(fp,"typedef struct %s {\n",ctype);                          \
    fprintf(fp,"  long l,u;\n");                                        \
    fprintf(fp,"  %s(long i){l=i;u=0;}\n",ctype);                       \
    fprintf(fp,"  void operator=(long i){l=i;u=0;}\n");                 \
    fprintf(fp,"} %s;\n",ctype);                                        \
    fprintf(fp,"#pragma link off class %s;\n",ctype);                   \
    fprintf(fp,"#pragma link off typedef %s;\n",ctype);                 \
  }                                                                     \
  else {                                                                \
    fprintf(fp,"typedef struct %s {\n",ctype);                          \
    fprintf(fp,"  char dmy[%d];\n",(int)sizeof(type));                  \
    fprintf(fp,"} %s;\n",ctype);                                        \
    fprintf(fp,"#pragma link off class %s;\n",ctype);                   \
    fprintf(fp,"#pragma link off typedef %s;\n",ctype);                 \
  }

/*******************************************************************
*******************************************************************/
int gen_systypes()
{
  FILE *fp;
  fp=fopen(header,"w");
  testdup(fp,"G__SYSTYPES_H");

/* see v6_init.cxx, G__platformMacro
  INT_TYPEDEF_PREFER_INT(fp,ssize_t,"ssize_t"); */
  INT_TYPEDEF_PREFER_INT(fp,pid_t,"pid_t");
  UINT_TYPEDEF_PREFER_INT(fp,pid_t,"pid_t");
  fprintf(fp,"typedef void* ptr_t;\n");
  UINT_TYPEDEF_PREFER_LONG(fp,dev_t,"dev_t");
  UINT_TYPEDEF_PREFER_LONG(fp,gid_t,"gid_t");
  UINT_TYPEDEF_PREFER_LONG(fp,uid_t,"uid_t");
  UINT_TYPEDEF_PREFER_LONG(fp,mode_t,"mode_t");
  /* UINT_TYPEDEF_PREFER_LONG(fp,umode_t,"umode_t"); */
  INT_TYPEDEF_PREFER_LONG(fp,off_t,"off_t");
  UINT_TYPEDEF_PREFER_LONG(fp,ino_t,"ino_t");
  UINT_TYPEDEF_PREFER_LONG(fp,nlink_t,"nlink_t");
  fprintf(fp,"typedef unsigned short ushort;\n");
  INT_TYPEDEF_PREFER_INT(fp,key_t,"key_t");

  fprintf(fp,"typedef long long int64_t;\n");
  fprintf(fp,"typedef unsigned long long uint64_t;\n");

  fprintf(fp,"#endif\n");
  fclose(fp);
  return(0);
}

/*******************************************************************
* main()
*******************************************************************/
int main()
{
  gen_systypes();
  exit(EXIT_SUCCESS);
}
