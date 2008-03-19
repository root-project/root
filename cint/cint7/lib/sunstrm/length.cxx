/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
//#! cint 
// length.cxx
//
//  Limit C++ source length to certain number
//  Usage:  cint length.cxx [length] file1  file2  ...
//

const int MAXLINE = 10000;
int length=255;

///////////////////////////////////////////////////////////
//
///////////////////////////////////////////////////////////
int separateline(char* buf,FILE* ofp)
{
  int i,j=0,pos=0,prev=0;
  int single_quote=0, double_quote=0;
  int len = strlen(buf);
  for(i=0;i<len;++i) {
    switch(buf[i]) {
    case '\'':
      if(!double_quote) single_quote  ^= 1;
      break;
    case '"':
      if(!single_quote) double_quote  ^= 1;
      break;
    case ',':
      if(!single_quote && !double_quote) {
	pos = i;
      }
    }
    if(j>=length && pos) {
      buf[pos] = 0;
      fprintf(ofp,"%s\n",buf+prev);
      buf[pos] = ',';
      prev = pos;
      j=0;
    }
    else {
      ++j;
    }
  }
  fprintf(ofp,"%s\n",buf+prev);
}

///////////////////////////////////////////////////////////
// CheckLength
///////////////////////////////////////////////////////////
int CheckLength(char* fname)
{
  int flag=0;
  FILE* fp = fopen(fname,"rb");
  if(!fp) return(0);

  FILE* ofp = fopen("G__temp","wb");
  if(!ofp) {
    fclose(fp);
    fprintf(stderr,"Error: output flie can not open\n");
    return(0);
  }
  
  char buf[MAXLINE];
  char* null_fgets;
  for(;;) {
    null_fgets=fgets(buf,MAXLINE,fp);
    if(null_fgets) {
      char *p = strchr(buf,'\n');
      if(p) *p=0;
      if(strlen(buf)>length) { separateline(buf,ofp); ++flag; }
      else fprintf(ofp,"%s\n",buf);
    }
    else break;
  } 

  fclose(fp);
  fclose(ofp);
  if(flag) rename("G__temp",fname);
  else remove("G__temp");
}

///////////////////////////////////////////////////////////
// main
///////////////////////////////////////////////////////////
int main(int argc,char** argv) {
  if(argc<3) {
    fprintf(stderr,"Usage: length.cxx [length] [file1] <file2 <file3...>>\n");
    exit(1);
  }
  length = atoi(argv[1]);
  for(int i=2;i<argc;i++) {
    CheckLength(argv[i]);
  }
  return(0);
}
