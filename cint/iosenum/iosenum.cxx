/************************************************************************
* include/iosenum.cxx   (processed by CINT)
*
*  This program runs once at installation creating include/iosenum.h 
* for platform dependent ios enum values.
************************************************************************/
#include <ReadF.C>

char CPPSRCPOST[20];
char CPP[400];
char FNAME[400];
char COMMAND[500];

/************************************************************************
* read MAKEINFO to get CPP and CPPSRCPOST
* exit program if MAKEINFO can not open or CPP is not set.
************************************************************************/
void readmakeinfo(void) {
  ReadFile f("cint/MAKEINFO"); 
  char *p;
  if(!f.isvalid()) {
    fprintf(stderr,"MAKEINFO not found. CINT may not be installed properly\n");
    exit(0);
  }
  while(f.read()) {
    if(f.argc>=3 && strcmp(f.argv[1],"CPP")==0) {
      p=strchr(f.argv[0],'=');
      strcpy(CPP,p+1);
    }
    else if(f.argc>=3 && strcmp(f.argv[1],"CPPSRCPOST")==0) {
      p=strchr(f.argv[0],'.');
      strcpy(CPPSRCPOST,p);
    }
  }
  int i=0;
  while(isspace(CPP[i])) ++i;
  if('\0'==CPP[i]) {
    printf("CINT installed without C++ compiler. No need to create iosenum.h contents\n");
    exit(0);
  }
}

/************************************************************************
* create source file, compile and run it to get platform dependent enum
* value
************************************************************************/
void iosenumdump(char *item) {
  FILE *fp;
  fp=fopen(FNAME,"w");
  fprintf(fp,"#include <iostream.h>\n");
  fprintf(fp,"main() {\n");
  fprintf(fp,"  cout<<\"static int ios::%s=\"<<ios::%s<<\";\"<<endl;\n"
	,item,item);
  fprintf(fp,"}\n");
  fclose(fp);

  if(0==system(COMMAND)) {
    printf("ios::%s exists\n",item);
    system("./a.out >> iosenum.h");
  }
  else printf("ios::%s does not exist\n",item);
  remove("a.out");
  remove(FNAME);
}


/************************************************************************
* main() function
************************************************************************/
int main() {
  printf("Creating include/iosenum.h for implementation dependent enum value\n");

  FILE* fp;
  fp = fopen("iosenum.h","w");
  fprintf(fp,"/* include/platform/iosenum.h\n");
  fprintf(fp," *  This file contains platform dependent ios enum value.\n");
  fprintf(fp," *  Run 'cint iosenum.cxx' to create this file. It is done\n");
  fprintf(fp," *  only once at installation. */\n");
  fclose(fp);

  readmakeinfo();
  sprintf(FNAME,"iosenum%s",CPPSRCPOST);
  sprintf(COMMAND,"%s %s 2> /dev/null",CPP,FNAME);

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
