/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include <stdio.h>
#include <unistd.h>

int main() {
  struct DIR *dir = opendir(getenv("CINTSYSDIR"));
  struct dirent *d;
  int s;
  struct stat buf;

  while((d=readdir(dir))) {
    s=stat(d->d_name,&buf);
    printf("%-25s %10d %10d\n",d->d_name,s,buf.st_size);
  }

  closedir(dir);
}
