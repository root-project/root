/************************************************************************
* include/iosenum.cxx   (processed by CINT)
*
*  This program runs once at installation creating include/iosenum.h 
* for platform dependent ios enum values.
************************************************************************/
#include <ReadF.C>
#include "configcint.h"

char FNAME[400];
char COMMAND[5000];

/************************************************************************
* create source file, compile and run it to get platform dependent enum
* value
************************************************************************/
void iosenumdump(char *item) {
  FILE *fp;
  fp=fopen(FNAME,"w");
  if(!fp) {
    fprintf(stderr,"Error: Can not open %s for writing\n",FNAME);
    return;
  }
  fprintf(fp,"#include <iostream.h>\n");
  fprintf(fp,"int main() {\n");
  fprintf(fp,"  cout<<\"static int ios::%s=\"<<ios::%s<<\";\"<<endl;\n"
	,item,item);
  fprintf(fp,"  return(0);\n");
  fprintf(fp,"}\n");
  fclose(fp);

  if(0==system(COMMAND)) {
    printf("ios::%s exists\n",item);
#ifdef G__CYGWIN
    system("./a.exe >> iosenum.h");
#else
    system("./a.out >> iosenum.h");
#endif
  }
  else printf("ios::%s does not exist\n",item);
#ifdef G__CYGWIN
  remove("a.exe");
#else
  remove("a.out");
#endif
  remove(FNAME);
}

void iosbaseenumdump(char *item) {
  FILE *fp;
  fp=fopen(FNAME,"w");
  if(!fp) {
    fprintf(stderr,"Error: Can not open %s for writing\n",FNAME);
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
    system("./a.exe >> iosenum.h");
#else
    system("./a.out >> iosenum.h");
#endif
  }
  else printf("ios_base::%s does not exist\n",item);
#ifdef G__CYGWIN
  remove("a.exe");
#else
  remove("a.out");
#endif
  remove(FNAME);
}


/************************************************************************
* main() function
************************************************************************/
int main() {
  printf("Creating include/iosenum.h for implementation dependent enum value\n");

  FILE* fp;
  fp = fopen("iosenum.h","w");
  if(!fp) {
    fprintf(stderr,"Error: Can not open %s for writing\n","iosenum.h");
    exit(1);
  }
  fprintf(fp,"/* include/platform/iosenum.h\n");
  fprintf(fp," *  This file contains platform dependent ios enum value.\n");
  fprintf(fp," *  Run 'cint iosenum.cxx' to create this file. It is done\n");
  fprintf(fp," *  only once at installation. */\n");
  fclose(fp);

  const char* flags_common[] = {
     "boolalpha",
     "dec",
     "fixed",
     "hex",
     "internal",
     "left",
     "oct",
     "right",
     "scientific",
     "showbase",
     "showpoint",
     "showpos",
     "skipws",
     "unitbuf",
     "uppercase",
     "adjustfield",
     "basefield",
     "floatfield",
     "badbit",
     "eofbit",
     "failbit",
     "goodbit",
     "app",
     "ate",
     "binary",
     "in",
     "out",
     "trunc",
     "beg",
     "cur",
     "end",
     0
  };

  const char* flags_old[] = {
     "hardfail",
     "nocreate",
     "noreplace",
     //   "bin",
     "stdio",
      0
  };

  const char* flags_tmpltios[] = {
     "openmode",
     0
  };

  strcpy(FNAME,"iosenum.cxx");
  sprintf(COMMAND,"%s %s 2> /dev/null", G__CFG_CXX, FNAME);

  system("echo '#pragma ifndef G__TMPLTIOS' >> iosenum.h");

  for (int i = 0; flags_common[i]; ++i)
     iosenumdump(flags_common[i]);
  for (int i = 0; flags_old[i]; ++i)
     iosenumdump(flags_old[i]);

  system("echo '#pragma else' >> iosenum.h");

  // added for g++3.0
  for (int i = 0; flags_common[i]; ++i)
     iosbaseenumdump(flags_common[i]);
  for (int i = 0; flags_tmpltios[i]; ++i)
     iosbaseenumdump(flags_tmpltios[i]);

  system("echo '#pragma endif' >> iosenum.h");
  
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
