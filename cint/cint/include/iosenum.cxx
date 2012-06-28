/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/************************************************************************
* include/iosenum.cxx   (processed by CINT)
*
*  This program runs once at installation creating include/iosenum.h 
* for platform dependent ios enum values.
************************************************************************/
#include "ReadF.C"
#include "configcint.h"

char CPPSRCPOST[20];
char CPP[2000];
char FNAME[400];
char COMMAND[3000];

void readmakeinfo(void) {
  CPP[0]=0;
  strcat(CPP, G__CFG_CXX);
  strcat(CPP, " ");
  strcat(CPP, G__CFG_CXXFLAGS);
  strcat(CPP, " ");
  strcat(CPP, G__CFG_CXXMACROS);
  // should be .cxx, but that means iosenum.cxx would be overwritten
  strcpy(CPPSRCPOST,".cpp");
}

/************************************************************************
* create source file, compile and run it to get platform dependent enum
* value
************************************************************************/
void iosenumdump(char *item) {
  FILE *fp;
  fp=fopen(FNAME,"w");
  if(!fp) {
    fprintf(stderr,"Error: Cannot open %s for writing\n",FNAME);
    return;
  }
  fprintf(fp,"#include <iostream>\n");
  fprintf(fp,"namespace std {}; using namespace std;\n");
  fprintf(fp,"int main() {\n");
  fprintf(fp,"  cout<<\"static int ios::%s=\"<<ios::%s<<\";\"<<endl;\n"
	,item,item);
  fprintf(fp,"  return(0);\n");
  fprintf(fp,"}\n");
  fclose(fp);

  if(0==system(COMMAND)) {
    printf("ios::%s exists\n",item);
#ifdef G__CYGWIN
    system("./a.exe >> include/iosenum.h");
#elif defined(G__MSC_VER)
    system("iosenum.exe >> include/iosenum.h");
#else
    system("./a.out >> include/iosenum.h");
#endif
  }
  else printf("ios::%s does not exist\n",item);
#ifdef G__CYGWIN
  remove("a.exe");
#elif defined(G__MSC_VER)
  remove("iosenum.exe");
#else
  remove("a.out");
#endif
  remove(FNAME);
}

void iosbaseenumdump(char *item) {
  FILE *fp;
  fp=fopen(FNAME,"w");
  if(!fp) {hard
    fprintf(stderr,"Error: Cannot open %s for writing\n",FNAME);
    return;
  }
  fprintf(fp,"#include <iostream>\n");
  fprintf(fp,"#ifndef __hpux\n");
  fprintf(fp,"using namespace std;\n");
  fprintf(fp,"#endif\n");
  fprintf(fp,"int main() {\n");
  fprintf(fp,"  cout<<\"static ios_base::fmtflags ios_base::%s=\"<<ios_base::%s<<\";\"<<endl;\n"
	,item,item);
  fprintf(fp,"  return(0);\n");
  fprintf(fp,"}\n");
  fclose(fp);

  if(0==system(COMMAND)) {
    printf("ios_base::%s exists\n",item);
#ifdef G__CYGWIN
    system("./a.exe >> include/iosenum.h");
#elif defined(G__MSC_VER)
    system("iosenum.exe >> include/iosenum.h");
#else
    system("./a.out >> include/iosenum.h");
#endif
  }
  else printf("ios_base::%s does not exist\n",item);
#ifdef G__CYGWIN
  remove("a.exe");
#elif defined(G__MSC_VER)
  remove("iosenum.exe");
#else
  remove("a.out");
#endif
  remove(FNAME);
}

/************************************************************************
* checkcompilerversion_core
************************************************************************/
void ccv(FILE* fp,const char* name) {
  fprintf(fp,"#if !defined(%s) || (%s!=%d)\n",name,name,G__calc(name));
  fprintf(fp,"#error $CINTSYSDIR/include/iosenum.h compiler version mismatch. Do'cd $CINTSYSDIR/include; cint iosenum.cxx' to restore\n");
  fprintf(fp,"#endif\n");
}

/************************************************************************
* checkcompilerversion
************************************************************************/
void checkcompilerversion(FILE* fp) {
  // check compiler dependent flags
#if defined(G__CYGWIN)
  ccv(fp,"G__CYGWIN");
#endif
#if defined(G__GNUC)
  ccv(fp,"G__GNUC");
#endif
#if defined(G__HP_aCC)
  ccv(fp,"G__HP_aCC");
#endif
#if defined(G__SUNPRO_CC)
  ccv(fp,"G__SUNPRO_CC");
#endif
#if defined(G__SUNPRO_C)
  ccv(fp,"G__SUNPRO_C");
#endif
#if defined(G__MSC_VER)
  ccv(fp,"G__MSC_VER");
#endif
#if defined(G__SYMANTEC)
  ccv(fp,"G__SYMANTEC");
#endif
#if defined(G__BORLAND)
  ccv(fp,"G__BORLAND");
#endif
#if defined(G__BCPLUSPLUS)
  ccv(fp,"G__BCPLUSPLUS");
#endif
#if defined(G__KCC)
  ccv(fp,"G__KCC");
#endif
#if defined(G__INTEL_COMPILER)
  ccv(fp,"G__INTEL_COMPILER");
#endif

  // check OS dependent flags
#if defined(G__HPUX)
  ccv(fp,"G__HPUX");
#endif
#if defined(G__SUN)
  ccv(fp,"G__SUN");
#endif
#if defined(G__WIN32)
  ccv(fp,"G__WIN32");
#endif
#if defined(G__AIX)
  ccv(fp,"G__AIX");
#endif
#if defined(G__SGI)
  ccv(fp,"G__SGI");
#endif
}

/************************************************************************
* main() function
************************************************************************/
int main() {
  printf("Creating include/iosenum.h for implementation dependent enum value\n");

  FILE* fp;
  fp = fopen("include/iosenum.h","w");
  if(!fp) {
    fprintf(stderr,"Error: Cannot open %s for writing\n","include/iosenum.h");
    exit(1);
  }
  fprintf(fp,"/* include/iosenum.h\n");
  fprintf(fp," *  This file contains platform dependent ios enum value.\n");
  fprintf(fp," *  Run 'cint include/iosenum.cxx' to create this file. It is done\n");
  fprintf(fp," *  only once at installation. */\n");
  checkcompilerversion(fp);

  fclose(fp);

  readmakeinfo();
  FNAME[0]=0;
  strcat(FNAME, "include/iosenum");
  strcat(FNAME, CPPSRCPOST);
  COMMAND[0]=0;
  strcat(COMMAND, CPP); strcat(COMMAND, " ");
  strcat(COMMAND, FNAME);
#ifdef G__MSC_VER
  strcat(COMMAND, " > NUL");
#else
  strcat(COMMAND, " 2> /dev/null");
#endif

#ifndef G__BORLANDCC5
#ifndef G__MSC_VER
  system("echo '#pragma ifndef G__TMPLTIOS' >> include/iosenum.h");
#else
  system("echo #pragma ifndef G__TMPLTIOS >> include/iosenum.h");
#endif
  iosenumdump("goodbit");
  iosenumdump("eofbit");
  iosenumdump("failbit");
  iosenumdump("badbit");
  iosenumdump("hardfail");
  iosenumdump("in");
  iosenumdump("out");
  iosenumdump("ate");
  iosenumdump("app");
  iosenumdump("trunc");
  iosenumdump("nocreate");
  iosenumdump("noreplace");
  iosenumdump("binary");
  //iosenumdump("bin");
  iosenumdump("beg");
  iosenumdump("cur");
  iosenumdump("end");
  iosenumdump("boolalpha");
  iosenumdump("adjustfield");
  iosenumdump("basefield");
  iosenumdump("floatfield");
  iosenumdump("skipws");
  iosenumdump("left");
  iosenumdump("right");
  iosenumdump("internal");
  iosenumdump("dec");
  iosenumdump("oct");
  iosenumdump("hex");
  iosenumdump("showbase");
  iosenumdump("showpoint");
  iosenumdump("uppercase");
  iosenumdump("showpos");
  iosenumdump("scientific");
  iosenumdump("fixed");
  iosenumdump("unitbuf");
  iosenumdump("stdio");
#ifndef G__MSC_VER
  system("echo '#pragma else' >> include/iosenum.h");
#else
  system("echo #pragma else >> include/iosenum.h");
#endif

  // added for g++3.0
  iosbaseenumdump("boolalpha");
  iosbaseenumdump("dec");
  iosbaseenumdump("fixed");
  iosbaseenumdump("hex");
  iosbaseenumdump("internal");
  iosbaseenumdump("left");
  iosbaseenumdump("oct");
  iosbaseenumdump("right");
  iosbaseenumdump("scientific");
  iosbaseenumdump("showbase");
  iosbaseenumdump("showpoint");
  iosbaseenumdump("showpos");
  iosbaseenumdump("skipws");
  iosbaseenumdump("unitbuf");
  iosbaseenumdump("uppercase");
  iosbaseenumdump("adjustfield");
  iosbaseenumdump("basefield");
  iosbaseenumdump("floatfield");
  iosbaseenumdump("badbit");
  iosbaseenumdump("eofbit");
  iosbaseenumdump("failbit");
  iosbaseenumdump("goodbit");
  iosbaseenumdump("openmode");
  iosbaseenumdump("app");
  iosbaseenumdump("ate");
  iosbaseenumdump("binary");
  iosbaseenumdump("in");
  iosbaseenumdump("out");
  iosbaseenumdump("trunc");
  iosbaseenumdump("beg");
  iosbaseenumdump("cur");
  iosbaseenumdump("end");
#ifndef G__MSC_VER
  system("echo '#pragma endif' >> include/iosenum.h");
#else
  system("echo #pragma endif >> include/iosenum.h");
#endif
#endif
  
  exit(0);
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
