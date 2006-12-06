/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
// cint/chmod.cxx
// Change status
//   *.exe      : 777
//   *.dll      : 777
//   directory  : 777

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <fcntl.h>
#include <time.h>
#include <string.h>


void todir(const char* fullfname) {
  char com[1000];
  sprintf(com,"chmod 1777 %s",fullfname);
  system(com);
  printf("%s\n",com);
  fprintf(stderr,"%s\n",com);
}

void toexe(const char* fullfname) {
  char com[1000];
  sprintf(com,"chmod 777 %s",fullfname);
  system(com);
  printf("%s\n",com);
  fprintf(stderr,"%s\n",com);
}

void totext(const char* fullfname) {
  char com[1000];
  sprintf(com,"chmod 666 %s",fullfname);
  system(com);
  printf("%s\n",com);
  fprintf(stderr,"%s\n",com);
}

void chstatus(const char* fullfname,const char* fname) {
  char *p = strrchr(fullfname,'.');
  if(p) {
    if(strcmp(p,".exe")==0 || 
       strcmp(p,".dll")==0 || 
       strcmp(p,".bat")==0 || 
       strcmp(p,".sh")==0) {
      toexe(fullfname);
    }
    else {
      totext(fullfname);
    }
  }
  else {
    totext(fullfname);
  }
}

int scandir(const char* base,const char* dname) {
  char fulldname[1000];
  char fullfname[1000];
  if(base) sprintf(fulldname,"%s/%s",base,dname);
  else     strcpy(fulldname,dname);
  struct DIR *dir = opendir(fulldname);
  struct dirent *d;
  int s;
  struct stat buf;
  int flag=0;

  while((d=readdir(dir))) {
    ++flag;
    s=stat(d->d_name,&buf);
    sprintf(fullfname,"%s/%s",fulldname,d->d_name);
#if 0
    printf("%d %16b %25s %s\n" ,S_ISDIR(buf.st_mode) ,buf.st_mode
	   ,fullfname ,d->d_name);
#endif
    if(1||S_ISDIR(buf.st_mode)) {
      if(d->d_name[0]!='.') {
	if(scandir(fulldname,d->d_name)) {
	  todir(fullfname);
	}
	else {
	  chstatus(fullfname,d->d_name);
	}
      }
    }
    else {
      chstatus(fullfname,d->d_name);
    }
  }

  closedir(dir);
  return(flag);
}

int main() {
  scandir(0,getenv("CINTSYSDIR"));
  return 0;
}
