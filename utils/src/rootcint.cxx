// @(#)root/utils:$Name:  $:$Id: rootcint.cxx,v 1.243 2006/04/25 15:36:07 pcanal Exp $
// Author: Fons Rademakers   13/07/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/rootcint.            *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// CREDITS                                                              //
//                                                                      //
// This program generates the CINT dictionaries needed in order to      //
// get access to your classes via the interpreter.                      //
// In addition rootcint can generate the Streamer(),                    //
// TBuffer &operator>>() and ShowMembers() methods for ROOT classes,    //
// i.e. classes using the ClassDef and ClassImp macros.                 //
//                                                                      //
// Rootcint can be used like:                                           //
//                                                                      //
//  rootcint TAttAxis.h[{+,-}][!] ... [LinkDef.h] > AxisDict.cxx        //
//                                                                      //
// or                                                                   //
//                                                                      //
//  rootcint [-v[0-4]][-l][-f] dict.C [-c] [-p]                         //
//           file.h[{+,-}][!] ... [LinkDef.h]                           //
//                                                                      //
// The difference between the two is that in the first case only the    //
// Streamer() and ShowMembers() methods are generated while in the      //
// latter case a complete compileable file is generated (including      //
// the include statements). The first method also allows the            //
// output to be appended to an already existing file (using >>).        //
// The optional - behind the header file name tells rootcint to not     //
// generate the Streamer() method. A custom method must be provided     //
// by the user in that case. For the + and ! options see below.         //
// When using option -c also the interpreter method interface stubs     //
// will be written to the output file (AxisDict.cxx in the above case). //
// By default the output file will not be overwritten if it exists.     //
// Use the -f (force) option to overwite the output file. The output    //
// file must have one of the following extensions: .cxx, .C, .cpp,      //
// .cc, .cp.                                                            //
// Use the -p option to request the use of the compiler's preprocessor  //
// instead of CINT's preprocessor.  This is useful to handle header\n"  //
// files with macro construct not handled by CINT.\n\n"                 //
// Use the -l (long) option to prepend the pathname of the              //
// dictionary source file to the include of the dictionary header.      //
// This might be needed in case the dictionary file needs to be         //
// compiled with the -I- option that inhibits the use of the directory  //
// of the source file as the first search directory for                 //
// "#include "file"".                                                   //
// The flag --lib-list-prefix=xxx can be used to produce a list of      //
// libraries needed by the header files being parsed. Rootcint will     //
// read the content of xxx.in for a list of rootmap files (see          //
// rlibmap). Rootcint will read these files and use them to deduce a    //
// list of libraries that are needed to properly link and load this     //
// dictionary. This list of libraries is saved in the file xxx.out.     //
// This feature is used by ACliC (the automatic library generator).     //
// The verbose flags have the following meaning:                        //
//      -v   Display all messages                                       //
//      -v0  Display no messages at all.                                //
//      -v1  Display only error messages (default).                     //
//      -v2  Display error and warning messages.                        //
//      -v3  Display error, warning and note messages.                  //
//      -v4  Display all messages                                       //
// rootcint also support the other CINT options (see 'cint -h)          //
//                                                                      //
// Before specifying the first header file one can also add include     //
// file directories to be searched and preprocessor defines, like:      //
//   -I$MYPROJECT/include -DDebug=1                                     //
//                                                                      //
// The (optional) file LinkDef.h looks like:                            //
//                                                                      //
// #ifdef __CINT__                                                      //
//                                                                      //
// #pragma link off all globals;                                        //
// #pragma link off all classes;                                        //
// #pragma link off all functions;                                      //
//                                                                      //
// #pragma link C++ class TAxis;                                        //
// #pragma link C++ class TAttAxis-;                                    //
// #pragma link C++ class TArrayC-!;                                    //
// #pragma link C++ class AliEvent+;                                    //
//                                                                      //
// #pragma link C++ function StrDup;                                    //
// #pragma link C++ function operator+(const TString&,const TString&);  //
//                                                                      //
// #pragma link C++ global gROOT;                                       //
// #pragma link C++ global gEnv;                                        //
//                                                                      //
// #pragma link C++ enum EMessageTypes;                                 //
//                                                                      //
// #endif                                                               //
//                                                                      //
// This file tells rootcint for which classes the method interface      //
// stubs should be generated. A trailing - in the class name tells      //
// rootcint to not generate the Streamer() method. This is necessary    //
// for those classes that need a customized Streamer() method.          //
// A trailing ! in the class name tells rootcint to not generate the    //
// operator>>(TBuffer &b, MyClass *&obj) function. This is necessary to //
// be able to write pointers to objects of classes not inheriting from  //
// TObject. See for an example the source of the TArrayF class.         //
// If the class contains a ClassDef macro, a trailing + in the class    //
// name tells rootcint to generate an automatic Streamer(), i.e. a      //
// streamer that let ROOT do automatic schema evolution. Otherwise, a   //
// trailing + in the class name tells rootcint to generate a ShowMember //
// function and a Shadow Class. The + option is mutually exclusive with //
// the - option. For new classes the + option is the                    //
// preferred option. For legacy reasons it is not yet the default.      //
// When the linkdef file is not specified a default version exporting   //
// the classes with the names equal to the include files minus the .h   //
// is generated.                                                        //
//                                                                      //
// *** IMPORTANT ***                                                    //
// 1) LinkDef.h must be the last argument on the rootcint command line. //
// 2) Note that the LinkDef file name MUST contain the string:          //
//    LinkDef.h, Linkdef.h or linkdef.h, i.e. NA49_LinkDef.h is fine    //
//    just like, linkdef1.h. Linkdef.h is case sensitive.               //
//                                                                      //
// The default constructor used by the ROOT I/O can be customized by    //
// using the rootcint pragma:                                           //
//    #pragma link C++ ioctortype UserClass;                            //
// For example, with this pragma and a class named MyClass,             //
// this method will called the first of the following 3                 //
// constructors which exists and is public:                             //
//    MyClass(UserClass*);                                              //
//    MyClass(TRootIOCtor*);                                            //
//    MyClass(); // Or a constructor with all its arguments defaulted.  //
//                                                                      //
// When more than one pragma ioctortype is used, the first seen has     //
// priority.  For example with:                                         //
//    #pragma link C++ ioctortype UserClass1;                           //
//    #pragma link C++ ioctortype UserClass2;                           //
// We look in the following order:                                      //
//    MyClass(UserClass1*);                                             //
//    MyClass(UserClass2*);                                             //
//    MyClass(TRootIOCtor*);                                            //
//    MyClass(); // Or a constructor with all its arguments defaulted.  //
//                                                                      //
// ----------- historical ---------                                     //
//                                                                      //
// Note that the file rootcint.C is constructed in such a way that it   //
// can also be interpreted by CINT. The above two statements become in  //
// that case:                                                           //
//                                                                      //
// cint -I$ROOTSYS/include +V TAttAxis.h TAxis.h LinkDef.h rootcint.C \ //
//                            TAttAxis.h TAxis.h > AxisGen.C            //
//                                                                      //
// or                                                                   //
//                                                                      //
// cint -I$ROOTSYS/include +V TAttAxis.h TAxis.h LinkDef.h rootcint.C \ //
//                            AxisGen.C TAttAxis.h TAxis.h              //
//                                                                      //
// The +V and -I$ROOTSYS/include options are added to the list of       //
// arguments in the compiled version of rootcint.                       //
//////////////////////////////////////////////////////////////////////////

#ifndef __CINT__

#ifdef HAVE_CONFIG
#include "config.h"
#endif

#include "RConfig.h"
#include "Api.h"
#include <iostream>

#ifdef fgets // in G__ci.h
#  undef fgets
#  undef printf
#  undef fprintf
#  undef fputc
#  undef putc
#  undef putchar
#  undef fputs
#  undef puts
#  undef fgets
#  undef gets
#  undef system
#endif

extern "C" {
   void  G__setothermain(int othermain);
   void  G__setglobalcomp(int globalcomp);
   int   G__main(int argc, char **argv);
   void  G__exit(int rtn);
   struct G__includepath *G__getipathentry();
}
const char *ShortTypeName(const char *typeDesc);

const char *help =
"\n"
"This program generates the CINT dictionaries needed in order to\n"
"get access to your classes via the interpreter.\n"
"In addition rootcint can generate the Streamer(), TBuffer &operator>>()\n"
"and ShowMembers() methods for ROOT classes, i.e. classes using the\n"
"ClassDef and ClassImp macros.\n"
"\n"
"Rootcint can be used like:\n"
"\n"
"  rootcint TAttAxis.h[{+,-}][!] ... [LinkDef.h] > AxisDict.cxx\n"
"\n"
"or\n"
"\n"
"  rootcint [-v[0-4]] [-l] [-f] dict.C [-c] [-p] TAxis.h[{+,-}][!] ... [LinkDef.h] \n"
"\n"
"The difference between the two is that in the first case only the\n"
"Streamer() and ShowMembers() methods are generated while in the\n"
"latter case a complete compileable file is generated (including\n"
"the include statements). The first method also allows the\n"
"output to be appended to an already existing file (using >>).\n"
"The optional - behind the header file name tells rootcint\n"
"to not generate the Streamer() method. A custom method must be\n"
"provided by the user in that case. For the + and ! options see below.\n\n"
"When using option -c also the interpreter method interface stubs\n"
"will be written to the output file (AxisDict.cxx in the above case).\n"
"By default the output file will not be overwritten if it exists.\n"
"Use the -f (force) option to overwite the output file. The output\n"
"file must have one of the following extensions: .cxx, .C, .cpp,\n"
".cc, .cp.\n\n"
"Use the -p option to request the use of the compiler's preprocessor\n"
"instead of CINT's preprocessor.  This is useful to handle header\n"
"files with macro construct not handled by CINT.\n\n"
"Use the -l (long) option to prepend the pathname of the\n"
"dictionary source file to the include of the dictionary header.\n"
"This might be needed in case the dictionary file needs to be\n"
"compiled with the -I- option that inhibits the use of the directory\n"
"of the source file as the first search directory for\n"
"\"#include \"file\"\".\n"
"The flag --lib-list-prefix=xxx can be used to produce a list of\n"
"libraries needed by the header files being parsed. Rootcint will\n"
"read the content of xxx.in for a list of rootmap files (see\n"
"rlibmap). Rootcint will read these files and use them to deduce a\n"
"list of libraries that are needed to properly link and load this\n"
"dictionary. This list of libraries is saved in the file xxx.out.\n"
"This feature is used by ACliC (the automatic library generator).\n"
"The verbose flags have the following meaning:\n"
"      -v   Display all messages\n"
"      -v0  Display no messages at all.\n"
"      -v1  Display only error messages (default).\n"
"      -v2  Display error and warning messages.\n"
"      -v3  Display error, warning and note messages.\n"
"      -v4  Display all messages\n"
"rootcint also support the other CINT options (see 'cint -h)\n"
"\n"
"Before specifying the first header file one can also add include\n"
"file directories to be searched and preprocessor defines, like:\n"
"   -I../include -DDebug\n"
"\n"
"The (optional) file LinkDef.h looks like:\n"
"\n"
"#ifdef __CINT__\n"
"\n"
"#pragma link off all globals;\n"
"#pragma link off all classes;\n"
"#pragma link off all functions;\n"
"\n"
"#pragma link C++ class TAxis;\n"
"#pragma link C++ class TAttAxis-;\n"
"#pragma link C++ class TArrayC-!;\n"
"#pragma link C++ class AliEvent+;\n"
"\n"
"#pragma link C++ function StrDup;\n"
"#pragma link C++ function operator+(const TString&,const TString&);\n"
"\n"
"#pragma link C++ global gROOT;\n"
"#pragma link C++ global gEnv;\n"
"\n"
"#pragma link C++ enum EMessageTypes;\n"
"\n"
"#endif\n"
"\n"
"This file tells rootcint for which classes the method interface\n"
"stubs should be generated. A trailing - in the class name tells\n"
"rootcint to not generate the Streamer() method. This is necessary\n"
"for those classes that need a customized Streamer() method.\n"
"A trailing ! in the class name tells rootcint to not generate the\n"
"operator>>(TBuffer &b, MyClass *&obj) method. This is necessary to\n"
"be able to write pointers to objects of classes not inheriting from\n"
"TObject. See for an example the source of the TArrayF class.\n"
"If the class contains a ClassDef macro, a trailing + in the class\n"
"name tells rootcint to generate an automatic Streamer(), i.e. a\n"
"streamer that let ROOT do automatic schema evolution. Otherwise, a\n"
"trailing + in the class name tells rootcint to generate a ShowMember\n"
"function and a Shadow Class. The + option is mutually exclusive with\n"
"the - option. For new classes the + option is the\n"
"preferred option. For legacy reasons it is not yet the default.\n"
"When this linkdef file is not specified a default version exporting\n"
"the classes with the names equal to the include files minus the .h\n"
"is generated.\n"
"\n"
"*** IMPORTANT ***\n"
"1) LinkDef.h must be the last argument on the rootcint command line.\n"
"2) Note that the LinkDef file name MUST contain the string:\n"
"   LinkDef.h, Linkdef.h or linkdef.h, i.e. NA49_LinkDef.h is fine,\n"
"   just like linkdef1.h. Linkdef.h is case sensitive.\n";

#else
#include <ertti.h>
#endif

#ifdef _WIN32
#ifdef system
#undef system
#endif
#include <process.h>
#include <errno.h>
#endif

#ifdef __MWERKS__
#include <console.h>
#endif

#include <time.h>
#include <string>
#include <vector>
#include <map>
#include <fstream>

namespace std {}
using namespace std;

//#include <fstream>
//#include <strstream>

#include "TClassEdit.h"
using namespace TClassEdit;

#include "RStl.h"
using namespace ROOT;

const char *autoldtmpl = "G__auto%dLinkDef.h";
char autold[64];

std::ostream* dictSrcOut=&std::cout;
G__ShadowMaker *shadowMaker=0;

enum EDictType {
   kDictTypeCint,
   kDictTypeReflex,
   kDictTypeGCCXML
} dict_type = kDictTypeCint;

char *StrDup(const char *str);

typedef map<string,bool> Funcmap_t;
Funcmap_t gFunMap;

vector<string> gIoConstructorTypes;
void AddConstructorType(const char *arg)
{
   if (arg) gIoConstructorTypes.push_back(string(arg));
}

//const char* root_style()  {
//  static const char* s = ::getenv("MY_ROOT");
//  return s;
//}

// static int check = 0;
//______________________________________________________________________________
void SetFun (const string &fname)
{
   gFunMap[fname] = true;
}

//______________________________________________________________________________
bool GetFun(const string &fname)
{
   return gFunMap[fname];
}

//______________________________________________________________________________

#ifndef ROOT_Varargs
#include "Varargs.h"
#endif

const int kInfo     =      0;
const int kNote     =    500;
const int kWarning  =   1000;
const int kError    =   2000;
const int kSysError =   3000;
const int kFatal    =   4000;
const int kMaxLen   =   1024;

static int gErrorIgnoreLevel = kError;

//______________________________________________________________________________
void LevelPrint(bool prefix, int level, const char *location,
                const char *fmt, va_list ap)
{
   if (level < gErrorIgnoreLevel)
      return;

   const char *type = 0;

   if (level >= kInfo)
      type = "Info";
   if (level >= kNote)
      type = "Note";
   if (level >= kWarning)
      type = "Warning";
   if (level >= kError)
      type = "Error";
   if (level >= kSysError)
      type = "SysError";
   if (level >= kFatal)
      type = "Fatal";

   if (!location || strlen(location) == 0) {
      if (prefix) fprintf(stderr, "%s: ", type);
      vfprintf(stderr, (char*)va_(fmt), ap);
   } else {
      if (prefix) fprintf(stderr, "%s in <%s>: ", type, location);
      else fprintf(stderr, "In <%s>: ", location);
      vfprintf(stderr, (char*)va_(fmt), ap);
   }

   fflush(stderr);
}

//______________________________________________________________________________
void Error(const char *location, const char *va_(fmt), ...)
{
   // Use this function in case an error occured.

   va_list ap;
   va_start(ap,va_(fmt));
   LevelPrint(true, kError, location, va_(fmt), ap);
   va_end(ap);
}

//______________________________________________________________________________
void SysError(const char *location, const char *va_(fmt), ...)
{
   // Use this function in case a system (OS or GUI) related error occured.

   va_list ap;
   va_start(ap, va_(fmt));
   LevelPrint(true, kSysError, location, va_(fmt), ap);
   va_end(ap);
}

//______________________________________________________________________________
void Info(const char *location, const char *va_(fmt), ...)
{
   // Use this function for informational messages.

   va_list ap;
   va_start(ap,va_(fmt));
   LevelPrint(true, kInfo, location, va_(fmt), ap);
   va_end(ap);
}

//______________________________________________________________________________
void Warning(const char *location, const char *va_(fmt), ...)
{
   // Use this function in warning situations.

   va_list ap;
   va_start(ap,va_(fmt));
   LevelPrint(true, kWarning, location, va_(fmt), ap);
   va_end(ap);
}

//______________________________________________________________________________
void Fatal(const char *location, const char *va_(fmt), ...)
{
   // Use this function in case of a fatal error. It will abort the program.

   va_list ap;
   va_start(ap,va_(fmt));
   LevelPrint(true, kFatal, location, va_(fmt), ap);
   va_end(ap);
}

//______________________________________________________________________________
string R__tmpnam()
{
   // return a unique temporary file name as defined by tmpnam

   static char filename[L_tmpnam+2];
   static string tmpdir;

   if (tmpdir.length() == 0 && strlen(P_tmpdir) <= 2) {
      // P_tmpdir will be prepended to the result of tmpnam
      // if it is less that 2 character it is likely to
      // just be '/' or '\\'.
      // Let's add the temp directory.
      char *tmp;
      if ((tmp = getenv("CINTTMPDIR"))) tmpdir = tmp;
      else if ((tmp=getenv("TEMP")))    tmpdir = tmp;
      else if ((tmp=getenv("TMP")))     tmpdir = tmp;
      else tmpdir = ".";
      tmpdir += '/';
   }
#if 0 && defined(R__USE_MKSTEMP)
   else {
      tmpdir  = P_tmpdir;
      tmpdir += '/';
   }

   static char pattern[L_tmpnam+2];
   const char *radix = "XXXXXX";
   const char *appendix = "_rootcint";
   if (tmpdir.length() + strlen(radix) + strlen(appendix) + 2) {
      // too long

   }
   sprintf(pattern,"%s%s",tmpdir.c_str(),radix);
   strcpy(filename,pattern);
   close(mkstemp(filename));/*mkstemp not only generate file name but also opens the file*/
   remove(filename);
   fprintf(stderr,"pattern is %s filename is %s\n",pattern,filename);
   return filename;

#else
   tmpnam(filename);

   string result(tmpdir);
   result += filename;
   result += "_rootcint";

   return result;
#endif
}

#ifdef WIN32
#include "windows.h"
//______________________________________________________________________________
// defined in newlink.c
extern "C" FILE *FOpenAndSleep(const char *filename, const char *mode);

# ifdef fopen
#  undef fopen
# endif
# define fopen(A,B) FOpenAndSleep((A),(B))
#endif


//______________________________________________________________________________
typedef map<string,string> Recmap_t;
Recmap_t gAutoloads;
string gLiblistPrefix;
string gLibsNeeded;

int AutoLoadCallbackImpl(char *c, char *) {
   string need( gAutoloads[c] );
   if (need.length() && gLibsNeeded.find(need)==string::npos) {
      gLibsNeeded += " " + need;
   }
   return 1;
}

extern "C" int AutoLoadCallback(char *c, char *l) {
   return AutoLoadCallbackImpl(c,l);
}

void LoadLibraryMap() {

   string filelistname = gLiblistPrefix + ".in";
   ifstream filelist(filelistname.c_str());

   string filename;
   static char *buffer = 0;
   static unsigned int sbuffer = 0;

   while ( filelist >> filename ) {
      ifstream file(filename.c_str());

      string line;
      string classname;

      while ( file >> line ) {

         if (line.substr(0,8)=="Library.") {

            int pos = line.find(":",8);
            classname = line.substr(8,pos-8);

            pos = 0;
            while ( (pos=classname.find("@@",pos)) >= 0 ) {
               classname.replace(pos,2,"::");
            }
            while ( (pos=classname.find("-",pos)) >= 0 ) {
               classname.replace(pos,1," ");
            }

            getline(file,line,'\n');

            while( line[0]==' ' ) line.replace(0,1,"");

            if ( strchr(classname.c_str(),':')!=0 ) {
               // We have a namespace and we have to check it first

               int slen = classname.size();
               for(int k=0;k<slen;++k) {
                  if (classname[k]==':') {
                     if (k+1>=slen || classname[k+1]!=':') {
                        // we expected another ':'
                        break;
                     }
                     if (k) {
                        string base = classname.substr(0,k);
                        if (base=="std") {
                           // std is not declared but is also ignored by CINT!
                           break;
                        } else {
                           gAutoloads[base] = ""; // We never load namespaces on their own.
                           if (sbuffer < base.size()+20) {
                              delete [] buffer;
                              buffer = new char[base.size()+20];
                              sbuffer = base.size()+20;
                           }
                           strcpy(buffer,base.c_str());
                           G__set_class_autoloading_table(buffer,""); // We never load namespaces on their own.
                        }
                        ++k;
                     }
                  } else if (classname[k] == '<') {
                     // We do not want to look at the namespace inside the template parameters!
                     break;
                  }
               }
            }

            if (strncmp("ROOT::TImpProxy",classname.c_str(),strlen("ROOT::TImpProxy"))==0) {
               // Do not register the ROOT::TImpProxy so that they can be instantiated.
               continue;
            }
            gAutoloads[classname] = line;
            if (sbuffer < classname.size()+20) {
               delete [] buffer;
               buffer = new char[classname.size()+20];
               sbuffer = classname.size()+20;
            }
            strcpy(buffer,classname.c_str());
            G__set_class_autoloading_table(buffer,(char*)line.c_str());
         }
      }
   }
}

extern "C" {
   typedef void G__parse_hook_t ();
   G__parse_hook_t* G__set_beforeparse_hook (G__parse_hook_t* hook);
}

void EnableAutoLoading() {
   G__set_class_autoloading_table("ROOT","libCore.so");
   LoadLibraryMap();
   G__set_class_autoloading_callback(&AutoLoadCallback);
}

//______________________________________________________________________________
bool CheckInputOperator(G__ClassInfo &cl)
{
   // Check if the operator>> has been properly declared if the user has
   // resquested a custom version.

   bool has_input_error = false;

   // Need to find out if the operator>> is actually defined for
   // this class.
   G__ClassInfo gcl;
   long offset;

   char *proto = new char[strlen(cl.Fullname())+13];
   sprintf(proto,"TBuffer&,%s*&",cl.Fullname());

   G__MethodInfo methodinfo = gcl.GetMethod("operator>>",proto,&offset);

   Info(0, "Class %s: Do not generate operator>>()\n",
        cl.Fullname());
   if (!methodinfo.IsValid() ||
        methodinfo.ifunc()->para_p_tagtable[methodinfo.Index()][1] != cl.Tagnum() ||
        strstr(methodinfo.FileName(),"TBuffer.h")!=0 ||
        strstr(methodinfo.FileName(),"Rtypes.h" )!=0) {

      Error(0,
            "in this version of ROOT, the option '!' used in a linkdef file\n"
            "       implies the actual existence of customized operators.\n"
            "       The following declaration is now required:\n"
            "   TBuffer &operator>>(TBuffer &,%s *&);\n",cl.Fullname());

      has_input_error = true;
   } else {
    // Warning(0, "TBuffer &operator>>(TBuffer &,%s *&); defined at line %s %d \n",cl.Fullname(),methodinfo.FileName(),methodinfo.LineNumber());
   }
   // fprintf(stderr, "DEBUG: %s %d\n",methodinfo.FileName(),methodinfo.LineNumber());

   methodinfo = gcl.GetMethod("operator<<",proto,&offset);
   if (!methodinfo.IsValid() ||
        methodinfo.ifunc()->para_p_tagtable[methodinfo.Index()][1] != cl.Tagnum() ||
        strstr(methodinfo.FileName(),"TBuffer.h")!=0 ||
        strstr(methodinfo.FileName(),"Rtypes.h" )!=0) {

      Error(0,
            "in this version of ROOT, the option '!' used in a linkdef file\n"
            "       implies the actual existence of customized operator.\n"
            "       The following declaration is now required:\n"
            "   TBuffer &operator<<(TBuffer &,const %s *);\n",cl.Fullname());

      has_input_error = true;
   } else {
      //fprintf(stderr, "DEBUG: %s %d\n",methodinfo.FileName(),methodinfo.LineNumber());
   }

   delete [] proto;
   return has_input_error;
}

string FixSTLName(const string& cintName) {

   const char *s = cintName.c_str();
   char type[kMaxLen];
   strcpy(type, s);

#if 0 // (G__GNUC<3) && !defined (G__KCC)
   if (!strncmp(type, "vector",6)   ||
       !strncmp(type, "list",4)     ||
       !strncmp(type, "deque",5)    ||
       !strncmp(type, "map",3)      ||
       !strncmp(type, "multimap",8) ||
       !strncmp(type, "set",3)      ||
       !strncmp(type, "multiset",8) ) {

      // we need to remove this type of construct ",__malloc_alloc_template<0>"
      // in case of older gcc
      string result;
      unsigned int i;
      unsigned int start;
      unsigned int next = 0;
      unsigned int nesting = 0;
      unsigned int end;
      const string toReplace = "__malloc_alloc_template<0>";
      const string replacement = "alloc";
      for(i=0; i<cintName.length(); i++) {
         switch (cintName[i]) {
         case '<':
            if (nesting==0) {
               start = next;
               next = i+1;
               end = i;
               result += cintName.substr(start,end-start);
               result += "< ";
            }
            nesting++;
            break;
         case '>':
            nesting--;
            if (nesting==0) {
               start = next;
               next = i+1;
               end = i;
               string param = cintName.substr(start,end-start-1); // the -1 removes the space we know is there
               if (param==toReplace) {
                  result += replacement;
               } else {
                  result += FixSTLName(param);
               }
               result += " >";
            }
            break;
         case ',':
            if (nesting==1) {
               start = next;
               next = i+1;
               end = i;
               string param = cintName.substr(start,end-start);
               if (param==toReplace) {
                  result += replacement;
               } else {
                  result += FixSTLName(param);
               }
               result += ',';
            }
         }
      }
      return result;
   }
#endif
   return cintName;
}

//______________________________________________________________________________
bool CheckClassDef(G__ClassInfo &cl)
{
   // Return false if the class does not have ClassDef even-though it should.


   // Detect if the class has a ClassDef
   bool hasClassDef = cl.HasMethod("Class_Version");

   /*
      The following could be use to detect whether one of the
      class' parent class has a ClassDef

   long offset;
   const char *proto = "";
   const char *name = "IsA";

   G__MethodInfo methodinfo = cl.GetMethod(name,proto,&offset);
   bool parentHasClassDef = methodinfo.IsValid() && (methodinfo.Property() & G__BIT_ISPUBLIC);
   */

   bool inheritsFromTObject = cl.IsBase("TObject");
   bool inheritsFromTSelector = cl.IsBase("TSelector");

   bool result = true;
   if (!inheritsFromTSelector && inheritsFromTObject && !hasClassDef) {
      Error(cl.Name(),"%s inherits from TObject but does not have its own ClassDef\n",cl.Name());
      // We do want to always output the message (hence the Error level)
      // but still want rootcint to succeed.
      result = true;
   }

   // This check is disabled for now.
   return result;
}

//______________________________________________________________________________
int GetClassVersion(G__ClassInfo &cl)
{
   // Return the version number of the class or -1
   // if the function Class_Version does not exist.

   if (!cl.HasMethod("Class_Version")) return -1;

   const char *function = "::Class_Version()";
   string funcname = GetLong64_Name( cl.Fullname() ) + function;
   int version = (int)G__int(G__calc(funcname.c_str()));
   return version;
}

//______________________________________________________________________________
string GetNonConstMemberName(G__DataMemberInfo &m, const string &prefix = "")
{
   // Return the name of the data member so that it can be used
   // by non-const operation (so it includes a const_cast if necessary).

   if (m.Property() & (G__BIT_ISCONSTANT|G__BIT_ISPCONSTANT)) {
      string ret = "const_cast< ";
      ret += G__ShadowMaker::GetNonConstTypeName(m);
      ret += " &>( ";
      ret += prefix;
      ret += m.Name();
      ret += " )";
      return ret;
   } else {
      return prefix+m.Name();
   }
}

//______________________________________________________________________________
bool NeedShadowClass(G__ClassInfo& cl)
{
   if (G__ShadowMaker::IsStdPair(cl)) return true;
   if (G__ShadowMaker::IsSTLCont(cl.Name())) return false;
   if (strcmp(cl.Name(),"string") == 0 ) return false;

   if (cl.FileName() && !strncmp(cl.FileName(),"prec_stl",8))
       return false;

   // This means templated classes hiding members won't have
   // a proper shadow class, and the use has no change of
   // vetoring a shadow, as we need it for ShowMembers :-/
   if (cl.HasMethod("ShowMembers"))
      return dict_type == kDictTypeReflex || cl.IsTmplt();

   // no streamer, no shadow
   if (cl.RootFlag() == G__NOSTREAMER) return false;

   if (dict_type == kDictTypeReflex) return true;

   return ((cl.RootFlag() & G__USEBYTECOUNT));
}

//______________________________________________________________________________
bool NeedTypedefShadowClass(G__ClassInfo& cl)
{
   // shadow class is a typedef if the class has a ClassDef, and is not a templated class
   return (cl.HasMethod("Class_Name") && !cl.IsTmplt());
}

//______________________________________________________________________________
int NeedTemplateKeyword(G__ClassInfo &cl)
{
   if (cl.IsTmplt()) {
      char *templatename = StrDup(cl.Fullname());
      char *loc = strstr(templatename, "<");
      if (loc) *loc = 0;
      struct G__Definedtemplateclass *templ = G__defined_templateclass(templatename);
      if (templ) {

         int current = cl.Tagnum();
         G__IntList * ilist = templ->instantiatedtagnum;
         while(ilist) {
            if (ilist->i == current) {
               delete [] templatename;
               // This is an automatically instantiated templated class.
#ifdef __KCC
               // for now KCC works better without it !
               return 0;
#else
               return 1;
#endif
            }
            ilist = ilist->next;
         }

         delete [] templatename;
         // This is a specialized templated class
         return 0;

      } else {

         delete [] templatename;
         // It might be a specialization without us seeing the template definition
         return 0;
      }
   }
   return 0;
}

bool HasCustomOperatorNew(G__ClassInfo& cl);
bool HasCustomOperatorNewPlacement(G__ClassInfo& cl);
bool HasCustomOperatorNewArrayPlacement(G__ClassInfo& cl);
bool HasDefaultConstructor(G__ClassInfo& cl,string *args=0);
bool NeedConstructor(G__ClassInfo& cl);

//______________________________________________________________________________
bool HasCustomOperatorNew(G__ClassInfo& cl)
{
   // return true if we can find a custom operator new

   // Look for a custom operator new
   bool custom = false;
   G__ClassInfo gcl;
   long offset;
   const char *name = "operator new";
   const char *proto = "size_t";

   // first in the global namespace:
   G__MethodInfo methodinfo = gcl.GetMethod(name,proto,&offset);
   if  (methodinfo.IsValid()) {
      custom = true;
   }

   // in nesting space:
   gcl = cl.EnclosingSpace();
   methodinfo = gcl.GetMethod(name,proto,&offset);
   if  (methodinfo.IsValid()) {
      custom = true;
   }

   // in class
   methodinfo = cl.GetMethod(name,proto,&offset);
   if  (methodinfo.IsValid()) {
      custom = true;
   }

   return custom;
}

//______________________________________________________________________________
bool HasCustomOperatorNewPlacement(G__ClassInfo& cl)
{
   // return true if we can find a custom operator new

   // Look for a custom operator new
   bool custom = false;
   G__ClassInfo gcl;
   long offset;
   const char *name = "operator new";
   const char *proto = "size_t";
   const char *protoPlacement = "size_t,void*";

   // first in the global namespace:
   G__MethodInfo methodinfo = gcl.GetMethod(name,proto,&offset);
   G__MethodInfo methodinfoPlacement = gcl.GetMethod(name,protoPlacement,&offset);
   if  (methodinfoPlacement.IsValid()) {
      // We have a custom new placement in the global namespace
      custom = true;
   }

   // in nesting space:
   gcl = cl.EnclosingSpace();
   methodinfo = gcl.GetMethod(name,proto,&offset);
   methodinfoPlacement = gcl.GetMethod(name,protoPlacement,&offset);
   if  (methodinfoPlacement.IsValid()) {
      custom = true;
   }

   // in class
   methodinfo = cl.GetMethod(name,proto,&offset);
   methodinfoPlacement = cl.GetMethod(name,protoPlacement,&offset);
   if  (methodinfoPlacement.IsValid()) {
      // We have a custom operator new with placement in the class
      // hierarchy.  We still need to check that it has not been
      // overloaded by a simple operator new.

      G__ClassInfo clNew(methodinfo.ifunc()->tagnum);
      G__ClassInfo clPlacement(methodinfoPlacement.ifunc()->tagnum);

      if (clNew.IsBase(clPlacement)) {
         // the operator new hides the operator new with placement
         custom = false;
      } else {
         custom = true;
      }
   }

   return custom;
}

//______________________________________________________________________________
bool HasCustomOperatorNewArrayPlacement(G__ClassInfo& cl)
{
   // return true if we can find a custom operator new

   // Look for a custom operator new[]
   bool custom = false;
   G__ClassInfo gcl;
   long offset;
   const char *name = "operator new[]";
   const char *proto = "size_t";
   const char *protoPlacement = "size_t,void*";

   // first in the global namespace:
   G__MethodInfo methodinfo = gcl.GetMethod(name,proto,&offset);
   G__MethodInfo methodinfoPlacement = gcl.GetMethod(name,protoPlacement,&offset);
   if  (methodinfoPlacement.IsValid()) {
      // We have a custom new[] placement in the global namespace
      custom = true;
   }

   // in nesting space:
   gcl = cl.EnclosingSpace();
   methodinfo = gcl.GetMethod(name,proto,&offset);
   methodinfoPlacement = gcl.GetMethod(name,protoPlacement,&offset);
   if  (methodinfoPlacement.IsValid()) {
      custom = true;
   }

   // in class
   methodinfo = cl.GetMethod(name,proto,&offset);
   methodinfoPlacement = cl.GetMethod(name,protoPlacement,&offset);
   if  (methodinfoPlacement.IsValid()) {
      // We have a custom operator new[] with placement in the class
      // hierarchy.  We still need to check that it has not been
      // overloaded by a simple operator new.

      G__ClassInfo clNew(methodinfo.ifunc()->tagnum);
      G__ClassInfo clPlacement(methodinfoPlacement.ifunc()->tagnum);

      if (clNew.IsBase(clPlacement)) {
         // the operator new[] hides the operator new with placement
         custom = false;
      } else {
         custom = true;
      }
   }

   return custom;
}

//______________________________________________________________________________
bool CheckConstructor(G__MethodInfo &methodinfo, int argRequested)
{
   // Return true if the method is a valid, public constructor.

   if (    methodinfo.NArg()==(argRequested+methodinfo.NDefaultArg())
      && (methodinfo.Property() & G__BIT_ISPUBLIC)) {
      if (argRequested) {
         if (methodinfo.ifunc()) {
            // filter out constructor taking a void* or char*
            G__MethodArgInfo args( methodinfo );
            args.Next();
            if (args.Type()->Tagnum() == -1 ) {
               return false;
            }
            //switch ( args.Type()->Type() ) {
            //   case 'B': /* unsigned char* */
            //   case 'C': /* signed char* */
            //   case 'Y': /* void* */
            //      return false;
            //   default:
            //      ;
            //};
         }
      }
      return true;
   }
   return false;
}



//______________________________________________________________________________
bool HasDefaultConstructor(G__ClassInfo& cl, string *arg)
{
   // return true if we can find an constructor calleable without any arguments

   bool result = false;
   long offset;

   if (cl.Property() & G__BIT_ISNAMESPACE) return false;
   if (cl.Property() & G__BIT_ISABSTRACT) return false;

   for(unsigned int i=0; i<gIoConstructorTypes.size(); ++i) {
      string proto( gIoConstructorTypes[i] );
      int extra = (proto.size()==0) ? 0 : 1;
      if (extra==0) {
         // Looking for default constructor
         result = true;
      } else {
         proto += '*';
      }
      G__MethodInfo methodinfo  = cl.GetMethod(cl.TmpltName(),proto.c_str(),&offset,G__ClassInfo::ExactMatch,G__ClassInfo::InThisScope);
      G__MethodInfo tmethodinfo = cl.GetMethod(cl.Name(),     proto.c_str(),&offset,G__ClassInfo::ExactMatch,G__ClassInfo::InThisScope);

      if (methodinfo.IsValid()) {

         result = CheckConstructor( methodinfo, extra);
         if (result && extra && arg) {
            *arg = "( (";
            *arg += proto;
            *arg += ")0 )";
         }

      } else if (tmethodinfo.IsValid()) {

         // exactly same as above with a function with the full template name
         result = CheckConstructor( tmethodinfo, extra);
         if (result && extra && arg) {
            *arg = "( (";
            *arg += proto;
            *arg += ")0 )";
         }

      } else if (extra==0) {
         // Case where the default constructor is explicitly
         // declared but we could not get a hold of it (i.e. it is not
         // accessible.
         if (cl.HasMethod(cl.TmpltName())) result = false;
         if (cl.HasMethod(cl.Name())) result = false;
      }

      // Check for private operator new
      if (result) {
         const char *name = "operator new";
         proto = "size_t";
         methodinfo = cl.GetMethod(name,proto.c_str(),&offset);
         if  (methodinfo.IsValid() && !(methodinfo.Property() & G__BIT_ISPUBLIC) ) {
            result = false;
         }
         if (result) return true;
      }
   }
   return result;
}

//______________________________________________________________________________
bool NeedConstructor(G__ClassInfo& cl)
{
   // We need a constructor if:
   //   the class is not abstract
   //   the class is not an stl container
   //   the class version is greater than 0
   //   or (the option + has been specified and ShowMembers is missing)

   if (cl.Property() & G__BIT_ISNAMESPACE) return false;

   bool res= ((GetClassVersion(cl)>0
               || (!cl.HasMethod("ShowMembers") && (cl.RootFlag() & G__USEBYTECOUNT)
                  && strncmp(cl.FileName(),"prec_stl",8)!=0 )
               ) && !(cl.Property() & G__BIT_ISABSTRACT));
   return res;
}

//______________________________________________________________________________
bool CheckConstructor(G__ClassInfo& cl)
{
   // Return false if the constructor configuration is invalid

   bool result = true;
   if (NeedConstructor(cl)) {

      bool custom = HasCustomOperatorNew(cl);
      if (custom && cl.IsBase("TObject")) {
         custom = false;
      }
      // if (custom) fprintf(stderr,"%s has custom operator new\n",cl.Name());

      result = !HasDefaultConstructor(cl);
   }

   // For now we never issue a warning at rootcint time.
   // There will be a warning at run-time.
   result = true;

   if (!result) {
      //Error(cl.Fullname(), "I/O has been requested but there is no constructor calleable without arguments\n"
      //      "\tand a custom operator new has been defined.\n"
      //      "\tEither disable the I/O or add an explicit default constructor.\n",cl.Fullname());
      Warning(cl.Fullname(), "I/O has been requested but is missing an explicit default constructor.\n"
               "\tEither disable the I/O or add an explicit default constructor.\n",cl.Fullname());
   }

   return result;
}

//______________________________________________________________________________
bool NeedDestructor(G__ClassInfo& cl)
{
   long offset;
   const char *proto = "";
   string name = "~";
   name += cl.TmpltName();

   if (cl.Property() & G__BIT_ISNAMESPACE) return false;

   G__MethodInfo methodinfo = cl.GetMethod(name.c_str(),proto,&offset);

   // fprintf(stderr,"testing %s and has %d",name.c_str(),methodinfo.IsValid());
   if (methodinfo.IsValid() && !(methodinfo.Property() & G__BIT_ISPUBLIC) ) {
      return false;
   }
   return true;
   /* (GetClassVersion(cl)>0
      || (!cl.HasMethod("ShowMembers") && (cl.RootFlag() & G__USEBYTECOUNT)
      && strncmp(cl.FileName(),"prec_stl",8)!=0 ) );
   */
}

//______________________________________________________________________________
int IsSTLContainer(G__DataMemberInfo &m)
{
   // Is this an STL container?

   const char *s = m.Type()->TrueName();
   if (!s) return kNotSTL;

   string type(s);
   int k = TClassEdit::IsSTLCont(type.c_str(),1);

//    if (k) printf(" %s==%d\n",type.c_str(),k);

   return k;
}

//______________________________________________________________________________
int IsSTLContainer(G__BaseClassInfo &m)
{
   // Is this an STL container?

   const char *s = m.Name();
   if (!s) return kNotSTL;

   string type(s);
   int k = TClassEdit::IsSTLCont(type.c_str(),1);
//   if (k) printf(" %s==%d\n",type.c_str(),k);
   return k;
}

//______________________________________________________________________________
int IsStreamable(G__DataMemberInfo &m)
{
   // Is this member a Streamable object?

   const char* mTypeName = ShortTypeName(m.Type()->Name());


   if ((m.Property() & G__BIT_ISSTATIC) ||
         strncmp(m.Title(), "!", 1) == 0        ||
         strcmp(m.Name(), "G__virtualinfo") == 0) return 0;

   if (((m.Type())->Property() & G__BIT_ISFUNDAMENTAL) ||
       ((m.Type())->Property() & G__BIT_ISENUM)) return 0;

   if (m.Property() & G__BIT_ISREFERENCE) return 0;

   if (IsSTLContainer(m)) {
      return 1;
   }

   if (!strcmp(mTypeName, "string") || !strcmp(mTypeName, "string*")) return 1;

   if ((m.Type())->HasMethod("Streamer")) {
      if (!(m.Type())->HasMethod("Class_Version")) return 1;
      int version = GetClassVersion(*m.Type());
      if (version > 0) return 1;
   }
   return 0;
}

//______________________________________________________________________________
G__TypeInfo &TemplateArg(G__DataMemberInfo &m, int count = 0)
{
   // Returns template argument. When count = 0 return first argument,
   // 1 second, etc.

   static G__TypeInfo ti;
   char arg[2048], *current, *next;

   strcpy(arg, m.Type()->TmpltArg());
   // arg is now a comma separated list of type names, and we want
   // to return the 'count+1'-th element in the list.
   int len = strlen(arg);
   int nesting = 0;
   int i = 0;
   current = 0;
   next = &(arg[0]);
   for (int c = 0; c<len && i<=count; c++) {
      switch (arg[c]) {
      case '<': nesting++; break;
      case '>': nesting--; break;
      case ',': if (nesting==0) {
                   arg[c]=0;
                   i++;
                   current = next;
                   next = &(arg[c+1]);
                }
                break;
      }
   }
   if (current) ti.Init(current);

   return ti;
}

//______________________________________________________________________________
G__TypeInfo &TemplateArg(G__BaseClassInfo &m, int count = 0)
{
   // Returns template argument. When count = 0 return first argument,
   // 1 second, etc.

   static G__TypeInfo ti;
   char arg[2048], *current, *next;

   strcpy(arg, m.Name());
   // arg is now is the name of class template instantiation.
   // We first need to find the start of the list of its template arguments
   // then we have a comma separated list of type names.  We want to return
   // the 'count+1'-th element in the list.
   int len = strlen(arg);
   int nesting = 0;
   int i = 0;
   current = 0;
   next = &(arg[0]);
   for (int c = 0; c<len && i<=count; c++) {
      switch (arg[c]) {
      case '<': if (nesting==0) {
                   arg[c]=0;
                   current = next;
                   next = &(arg[c+1]);
                }
                nesting++;
                break;
      case '>': nesting--; break;
      case ',': if (nesting==1) {
                   arg[c]=0;
                   i++;
                   current = next;
                   next = &(arg[c+1]);
                }
                break;
      }
   }
   if (current) ti.Init(current);

   return ti;
}

//______________________________________________________________________________
void WriteAuxFunctions(G__ClassInfo &cl)
{
   // Write the functions that are need for the TGenericClassInfo.
   // This includes
   //    IsA
   //    operator new
   //    operator new[]
   //    operator delete
   //    operator delete[]

   string classname( GetLong64_Name(RStl::DropDefaultArg( cl.Fullname() ) ) );
   string mappedname = G__map_cpp_name((char*)classname.c_str());

   if ( ! TClassEdit::IsStdClass( classname.c_str() ) ) {

      // Prefix the full class name with '::' except for the STL
      // containers and std::string.  This is to request the
      // real class instead of the class in the namespace ROOT::Shadow
      classname.insert(0,"::");
   }

   (*dictSrcOut) << "namespace ROOT {" << std::endl;

   string args;
   if (HasDefaultConstructor(cl,&args)) {
      // write the constructor wrapper only for concrete classes
      (*dictSrcOut) << "   // Wrappers around operator new" << std::endl
          << "   static void *new_" << mappedname.c_str() << "(void *p) {" << std::endl
          << "      return  p ? ";
      if (HasCustomOperatorNewPlacement(cl)) {
         (*dictSrcOut) << "new(p) " << classname.c_str() << args << " : ";
      } else {
         (*dictSrcOut) << "::new((::ROOT::TOperatorNewHelper*)p) " << classname.c_str() << args << " : ";
      }
      (*dictSrcOut) << "new " << classname.c_str() << args << ";" << std::endl
          << "   }" << std::endl;

      if (args.size()==0) {
         (*dictSrcOut) << "   static void *newArray_" << mappedname.c_str() << "(Long_t nElements, void *p) {" << std::endl;
         (*dictSrcOut) << "      return p ? ";
         if (HasCustomOperatorNewArrayPlacement(cl)) {
            (*dictSrcOut) << "new(p) " << classname.c_str() << "[nElements] : ";
         } else {
            (*dictSrcOut) << "::new((::ROOT::TOperatorNewHelper*)p) " << classname.c_str() << "[nElements] : ";
         }
         (*dictSrcOut) << "new " << classname.c_str() << "[nElements];" << std::endl;
         (*dictSrcOut) << "   }" << std::endl;
      }
   }

   if (NeedDestructor(cl)) {
      (*dictSrcOut) << "   // Wrapper around operator delete" << std::endl
          << "   static void delete_" << mappedname.c_str() << "(void *p) {" << std::endl
          << "      delete ((" << classname.c_str() << "*)p);" << std::endl
          << "   }" << std::endl

          << "   static void deleteArray_" << mappedname.c_str() << "(void *p) {" << std::endl
          << "      delete [] ((" << classname.c_str() << "*)p);" << std::endl
          << "   }" << std::endl

          << "   static void destruct_" << mappedname.c_str() << "(void *p) {" << std::endl
          << "      typedef " << classname.c_str() << " current_t;" << std::endl
          << "      ((current_t*)p)->~current_t();" << std::endl
          << "   }" << std::endl;
   }

   (*dictSrcOut) << "} // end of namespace ROOT for class " << classname.c_str() << std::endl << std::endl;
}

//______________________________________________________________________________
void WriteStringOperators(FILE *fd)
{
   // Write static ANSI C++ string to TBuffer operators.

   fprintf(fd, "//_______________________________________");
   fprintf(fd, "_______________________________________\n");
   fprintf(fd, "static TBuffer &operator>>(TBuffer &b, string &s)\n{\n");
   fprintf(fd, "   // Reading string object.\n\n");
   fprintf(fd, "   R__ASSERT(b.IsReading());\n");
   fprintf(fd, "   char ch;\n");
   fprintf(fd, "   do {\n");
   fprintf(fd, "      b >> ch;\n");
   fprintf(fd, "      if (ch) s.append(1, ch);\n");
   fprintf(fd, "   } while (ch != 0);\n");
   fprintf(fd, "   return b;\n");
   fprintf(fd, "}\n");
   fprintf(fd, "//_______________________________________");
   fprintf(fd, "_______________________________________\n");
   fprintf(fd, "static TBuffer &operator<<(TBuffer &b, string s)\n{\n");
   fprintf(fd, "   // Writing string object.\n\n");
   fprintf(fd, "   R__ASSERT(b.IsWriting());\n");
   fprintf(fd, "   b.WriteString(s.c_str());\n");
   fprintf(fd, "   return b;\n");
   fprintf(fd, "}\n");
}

//______________________________________________________________________________
int ElementStreamer(G__TypeInfo &ti, const char *R__t,int rwmode,const char *tcl=0)
{
   enum {
      R__BIT_ISTOBJECT   = 0x10000000,
      R__BIT_HASSTREAMER = 0x20000000,
      R__BIT_ISSTRING    = 0x40000000
   };

   long prop = ti.Property();
   char tiName[kMaxLen],tiFullname[kMaxLen],objType[kMaxLen];
   strcpy(tiName,ti.Name());
   strcpy(objType,ShortTypeName(tiName));
   if (ti.Fullname())
      strcpy(tiFullname,ti.Fullname());
   else
      tiFullname[0] = 0;
   int isTObj = (ti.IsBase("TObject") || !strcmp(tiFullname, "TObject"));
   int isStre = (ti.HasMethod("Streamer"));

   long kase = prop & (G__BIT_ISPOINTER|G__BIT_ISFUNDAMENTAL|G__BIT_ISENUM);
   if (isTObj)                      kase |= R__BIT_ISTOBJECT;
   if (strcmp("string" ,tiName)==0) kase |= R__BIT_ISSTRING;
   if (strcmp("string*",tiName)==0) kase |= R__BIT_ISSTRING;
   if (isStre)                      kase |= R__BIT_HASSTREAMER;

//    if (strcmp(objType,"string")==0) RStl::inst().GenerateTClassFor( "string"  );

   if (rwmode == 0) {  //Read mode

      if (R__t) (*dictSrcOut) << "            " << tiName << " " << R__t << ";" << std::endl;
      switch (kase) {

         case G__BIT_ISFUNDAMENTAL:
            if (!R__t)  return 0;
            (*dictSrcOut) << "            R__b >> " << R__t << ";" << std::endl;
            break;

         case G__BIT_ISPOINTER|R__BIT_ISTOBJECT|R__BIT_HASSTREAMER:
            if (!R__t)  return 1;
            (*dictSrcOut) << "            " << R__t << " = (" << tiName << ")R__b.ReadObjectAny(" << tcl << ");"  << std::endl;
            break;

         case G__BIT_ISENUM:
            if (!R__t)  return 0;
            //             fprintf(fp, "            R__b >> (Int_t&)%s;\n",R__t);
            // On some platforms enums and not 'Int_t' and casting to a reference to Int_t
            // induces the silent creation of a temporary which is 'filled' __instead of__
            // the desired enum.  So we need to take it one step at a time.
            (*dictSrcOut) << "            Int_t readtemp;" << std::endl
                << "            R__b >> readtemp;" << std::endl
                << "            " << R__t << " = static_cast<" << tiName << ">(readtemp);" << std::endl;
            break;

         case R__BIT_HASSTREAMER:
         case R__BIT_HASSTREAMER|R__BIT_ISTOBJECT:
            if (!R__t)  return 0;
            (*dictSrcOut) << "            " << R__t << ".Streamer(R__b);" << std::endl;
            break;

         case R__BIT_HASSTREAMER|G__BIT_ISPOINTER:
            if (!R__t)  return 1;
            //fprintf(fp, "            fprintf(stderr,\"info is %%p %%d\\n\",R__b.GetInfo(),R__b.GetInfo()?R__b.GetInfo()->GetOldVersion():-1);\n");
            (*dictSrcOut) << "            if (R__b.GetInfo() && R__b.GetInfo()->GetOldVersion()<=3) {" << std::endl;
            if (ti.Property() & G__BIT_ISABSTRACT) {
               (*dictSrcOut) << "               R__ASSERT(0);// " << objType << " is abstract. We assume that older file could not be produced using this streaming method." << std::endl;
            } else {
               (*dictSrcOut) << "               " << R__t << " = new " << objType << ";" << std::endl
                   << "               " << R__t << "->Streamer(R__b);" << std::endl;
            }
            (*dictSrcOut) << "            } else {" << std::endl
                << "               " << R__t << " = (" << tiName << ")R__b.ReadObjectAny(" << tcl << ");" << std::endl
                << "            }" << std::endl;
            break;

         case R__BIT_ISSTRING:
            if (!R__t)  return 0;
            (*dictSrcOut) << "            {TString R__str;" << std::endl
                << "             R__str.Streamer(R__b);" << std::endl
                << "             " << R__t << " = R__str.Data();}" << std::endl;
            break;

         case R__BIT_ISSTRING|G__BIT_ISPOINTER:
            if (!R__t)  return 0;
            (*dictSrcOut) << "            {TString R__str;"  << std::endl
                << "             R__str.Streamer(R__b);" << std::endl
                << "             " << R__t << " = new string(R__str.Data());}" << std::endl;
            break;

         case G__BIT_ISPOINTER:
            if (!R__t)  return 1;
            (*dictSrcOut) << "            " << R__t << " = (" << tiName << ")R__b.ReadObjectAny(" << tcl << ");" << std::endl;
            break;

         default:
            if (!R__t) return 1;
            (*dictSrcOut) << "            R__b.StreamObject(&" << R__t << "," << tcl << ");" << std::endl;
            break;
      }

   } else {     //Write case

      switch (kase) {

         case G__BIT_ISFUNDAMENTAL:
         case G__BIT_ISPOINTER|R__BIT_ISTOBJECT|R__BIT_HASSTREAMER:
            if (!R__t)  return 0;
            (*dictSrcOut) << "            R__b << " << R__t << ";" << std::endl;
            break;

         case G__BIT_ISENUM:
            if (!R__t)  return 0;
            (*dictSrcOut) << "            R__b << (Int_t&)" << R__t << ";" << std::endl;
            break;

         case R__BIT_HASSTREAMER:
         case R__BIT_HASSTREAMER|R__BIT_ISTOBJECT:
            if (!R__t)  return 0;
            (*dictSrcOut) << "            ((" << objType << "&)" << R__t << ").Streamer(R__b);" << std::endl;
            break;

         case R__BIT_HASSTREAMER|G__BIT_ISPOINTER:
            if (!R__t)  return 1;
            (*dictSrcOut) << "            R__b.WriteObjectAny(" << R__t << "," << tcl << ");" << std::endl;
            break;

         case R__BIT_ISSTRING:
            if (!R__t)  return 0;
            (*dictSrcOut) << "            {TString R__str(" << R__t << ".c_str());" << std::endl
                << "             R__str.Streamer(R__b);};" << std::endl;
            break;

         case R__BIT_ISSTRING|G__BIT_ISPOINTER:
            if (!R__t)  return 0;
            (*dictSrcOut) << "            {TString R__str(" << R__t << "->c_str());" << std::endl
                << "             R__str.Streamer(R__b);}" << std::endl;
            break;

         case G__BIT_ISPOINTER:
            if (!R__t)  return 1;
            (*dictSrcOut) << "            R__b.WriteObjectAny(" << R__t << "," << tcl <<");" << std::endl;
            break;

         default:
            if (!R__t)  return 1;
            (*dictSrcOut) << "            R__b.StreamObject((" << objType << "*)&" << R__t << "," << tcl << ");" << std::endl;
            break;
      }
   }
   return 0;
}

//______________________________________________________________________________
int STLContainerStreamer(G__DataMemberInfo &m, int rwmode)
{
   // Create Streamer code for an STL container. Returns 1 if data member
   // was an STL container and if Streamer code has been created, 0 otherwise.

   int stltype = abs(IsSTLContainer(m));
   if (stltype!=0) {
//        fprintf(stderr,"Add %s (%d) which is also %s\n",
//                m.Type()->Name(), stltype, m.Type()->TrueName() );
      RStl::inst().GenerateTClassFor( m.Type()->Name() );
   }
   if (!m.Type()->IsTmplt() || stltype<=0) return 0;

   int isArr = 0;
   int len = 1;
   if (m.Property() & G__BIT_ISARRAY) {
      isArr = 1;
      for (int dim = 0; dim < m.ArrayDim(); dim++) len *= m.MaxIndex(dim);
   }

   // string stlType( RStl::DropDefaultArg( m.Type()->Name() ) );
//    string stlType( TClassEdit::ShortType(m.Type()->Name(),
//                                          TClassEdit::kDropTrailStar|
//                                          TClassEdit::kDropStlDefault) );
   string stlType( ShortTypeName(m.Type()->Name()) );
   string stlName;
   stlName = ShortTypeName(m.Name());

   string fulName1,fulName2;
   const char *tcl1=0,*tcl2=0;
   G__TypeInfo &ti = TemplateArg(m);
   if (ElementStreamer(ti, 0, rwmode)) {
      tcl1="R__tcl1";
      const char *name = ti.Fullname();
      if (name) {
         // the value return by ti.Fullname is a static buffer
         // so we have to copy it immeditately
         fulName1 = name;
      } else {
         // ti is a simple type name
         fulName1 = ti.TrueName();
      }
   }
   if (stltype==kMap || stltype==kMultiMap) {
      G__TypeInfo &ti = TemplateArg(m,1);
      if (ElementStreamer(ti, 0, rwmode)) {
         tcl2="R__tcl2";
         const char *name = ti.Fullname();
         if (name) {
            // the value return by ti.Fullname is a static buffer
            // so we have to copy it immeditately
            fulName2 = name;
         } else {
            // ti is a simple type name
            fulName2 = ti.TrueName();
         }
      }
   }

   int pa = isArr;
   if (m.Property() & G__BIT_ISPOINTER) pa+=2;
   if (rwmode == 0) {
      // create read code
      (*dictSrcOut) << "      {" << std::endl;
      if (isArr) {
         (*dictSrcOut) << "         for (Int_t R__l = 0; R__l < " << len << "; R__l++) {" << std::endl;
      }

      switch (pa) {
         case 0:         //No pointer && No array
            (*dictSrcOut) << "         " << stlType.c_str() << " &R__stl =  " << stlName.c_str() << ";" << std::endl;
            break;
         case 1:         //No pointer && array
            (*dictSrcOut) << "         " << stlType.c_str() << " &R__stl =  " << stlName.c_str() << "[R__l];" << std::endl;
            break;
         case 2:         //pointer && No array
            (*dictSrcOut) << "         delete *" << stlName.c_str() << ";"<< std::endl
                << "         *" << stlName.c_str() << " = new " << stlType.c_str() << ";" << std::endl
                << "         " << stlType.c_str() << " &R__stl = **" << stlName.c_str() << ";" << std::endl;
            break;
         case 3:         //pointer && array
            (*dictSrcOut) << "         delete " << stlName.c_str() << "[R__l];" << std::endl
                << "         " << stlName.c_str() << "[R__l] = new " << stlType.c_str() << ";" << std::endl
                << "         " << stlType.c_str() << " &R__stl = *" << stlName.c_str() << "[R__l];" << std::endl;
            break;
      }

      (*dictSrcOut) << "         R__stl.clear();" << std::endl;

      if (tcl1) {
         (*dictSrcOut) << "         TClass *R__tcl1 = TBuffer::GetClass(typeid(" << fulName1.c_str() << "));" << std::endl
             << "         if (R__tcl1==0) {" << std::endl
             << "            Error(\"" << stlName.c_str() << " streamer\",\"Missing the TClass object for "
             << fulName1.c_str() << "!\");"  << std::endl
             << "            return;" << std::endl
             << "         }" << std::endl;
      }
      if (tcl2) {
         (*dictSrcOut) << "         TClass *R__tcl2 = TBuffer::GetClass(typeid(" << fulName2.c_str() << "));" << std::endl
             << "         if (R__tcl2==0) {" << std::endl
             << "            Error(\"" << stlName.c_str() << " streamer\",\"Missing the TClass object for "
             << fulName2.c_str() <<"!\");" << std::endl
             << "            return;" << std::endl
             << "         }" << std::endl;
      }

      (*dictSrcOut) << "         int R__i, R__n;" << std::endl
          << "         R__b >> R__n;" << std::endl;

      if (stltype==kVector) {
         (*dictSrcOut) << "         R__stl.reserve(R__n);" << std::endl;
      }
      (*dictSrcOut) << "         for (R__i = 0; R__i < R__n; R__i++) {" << std::endl;

      ElementStreamer(TemplateArg(m), "R__t", rwmode, tcl1);
      if (stltype == kMap || stltype == kMultiMap) {     //Second Arg
         ElementStreamer(TemplateArg(m,1), "R__t2", rwmode, tcl2);
      }

      /* Need to go from
         type R__t;
         R__t.Stream;
         vec.push_back(R__t);
         to
         vec.push_back(type());
         R__t_p = &(vec.last());
         *R__t_p->Stream;

      */
      switch (stltype) {

         case kMap:
         case kMultiMap: {
            std::string keyName( TemplateArg(m).Name() );
            (*dictSrcOut) << "            typedef " << keyName << " Value_t;" << std::endl
                << "            std::pair<Value_t const, " << TemplateArg(m,1).Name() << " > R__t3(R__t,R__t2);" << std::endl
                << "            R__stl.insert(R__t3);" << std::endl;
          //fprintf(fp, "            R__stl.insert(%s::value_type(R__t,R__t2));\n",stlType.c_str());
            break;
         }
         case kSet:
         case kMultiSet:
            (*dictSrcOut) << "            R__stl.insert(R__t);" << std::endl;
            break;
         case kVector:
         case kList:
         case kDeque:
            (*dictSrcOut) << "            R__stl.push_back(R__t);" << std::endl;
            break;

         default:
            assert(0);
      }
      (*dictSrcOut) << "         }" << std::endl
          << "      }" << std::endl;
      if (isArr) (*dictSrcOut) << "    }" << std::endl;

   } else {

      // create write code
      if (isArr) {
         (*dictSrcOut) << "         for (Int_t R__l = 0; R__l < " << len << "; R__l++) {" << std::endl;
      }
      (*dictSrcOut) << "      {" << std::endl;
      switch (pa) {
         case 0:         //No pointer && No array
            (*dictSrcOut) << "         " << stlType.c_str() << " &R__stl =  " << stlName.c_str() << ";" << std::endl;
            break;
         case 1:         //No pointer && array
            (*dictSrcOut) << "         " << stlType.c_str() << " &R__stl =  " << stlName.c_str() << "[R__l];" << std::endl;
            break;
         case 2:         //pointer && No array
            (*dictSrcOut) << "         " << stlType.c_str() << " &R__stl = **" << stlName.c_str() << ";" << std::endl;
            break;
         case 3:         //pointer && array
            (*dictSrcOut) << "         " << stlType.c_str() << " &R__stl = *" << stlName.c_str() << "[R__l];" << std::endl;
            break;
      }

      (*dictSrcOut) << "         int R__n=(&R__stl) ? int(R__stl.size()) : 0;" << std::endl
          << "         R__b << R__n;" << std::endl
          << "         if(R__n) {" << std::endl;

      if (tcl1) {
         (*dictSrcOut) << "         TClass *R__tcl1 = TBuffer::GetClass(typeid(" << fulName1.c_str() << "));" << std::endl
             << "         if (R__tcl1==0) {" << std::endl
             << "            Error(\"" << stlName.c_str() << " streamer\",\"Missing the TClass object for "
             << fulName1.c_str() << "!\");" << std::endl
             << "            return;" << std::endl
             << "         }" << std::endl;
      }
      if (tcl2) {
         (*dictSrcOut) << "         TClass *R__tcl2 = TBuffer::GetClass(typeid(" << fulName2.c_str() << "));" << std::endl
             << "         if (R__tcl2==0) {" << std::endl
             << "            Error(\"" << stlName.c_str() << "streamer\",\"Missing the TClass object for " << fulName2.c_str() << "!\");" << std::endl
             << "            return;" << std::endl
             << "         }" << std::endl;
      }

      (*dictSrcOut) << "            " << stlType.c_str() << "::iterator R__k;" << std::endl
          << "            for (R__k = R__stl.begin(); R__k != R__stl.end(); ++R__k) {" << std::endl;
      if (stltype == kMap || stltype == kMultiMap) {
         ElementStreamer(TemplateArg(m,0), "((*R__k).first )",rwmode,tcl1);
         ElementStreamer(TemplateArg(m,1), "((*R__k).second)",rwmode,tcl2);
      } else {
         ElementStreamer(TemplateArg(m,0), "(*R__k)"         ,rwmode,tcl1);
      }

      (*dictSrcOut) << "            }" << std::endl
          << "         }" << std::endl
          << "      }" << std::endl;
      if (isArr) (*dictSrcOut) << "    }" << std::endl;
   }
   return 1;
}

//______________________________________________________________________________
int STLStringStreamer(G__DataMemberInfo &m, int rwmode)
{
   // Create Streamer code for a standard string object. Returns 1 if data
   // member was a standard string and if Streamer code has been created,
   // 0 otherwise.

   const char *mTypeName = ShortTypeName(m.Type()->Name());
   if (!strcmp(mTypeName, "string")) {
      if (rwmode == 0) {
         // create read mode
         if ((m.Property() & G__BIT_ISPOINTER) &&
             (m.Property() & G__BIT_ISARRAY)) {

         } else if (m.Property() & G__BIT_ISARRAY) {

         } else {
            (*dictSrcOut) << "      { TString R__str; R__str.Streamer(R__b); ";
            if (m.Property() & G__BIT_ISPOINTER)
               (*dictSrcOut) << "if (*" << m.Name() << ") delete *" << m.Name() << "; (*"
                   << m.Name() << " = new string(R__str.Data())); }" << std::endl;
            else
               (*dictSrcOut) << m.Name() << " = R__str.Data(); }" << std::endl;
         }
      } else {
         // create write mode
         if (m.Property() & G__BIT_ISPOINTER)
            (*dictSrcOut) << "      { TString R__str; if (*" << m.Name() << ") R__str = (*"
                << m.Name() << ")->c_str(); R__str.Streamer(R__b);}" << std::endl;
         else
            (*dictSrcOut) << "      { TString R__str = " << m.Name() << ".c_str(); R__str.Streamer(R__b);}" << std::endl;
      }
      return 1;
   }
   return 0;
}

//______________________________________________________________________________
#ifdef OLDSTREAMER
int STLBaseStreamer(G__BaseClassInfo &m, int rwmode)
{
   // Create Streamer code for an STL base class. Returns 1 if base class
   // was an STL container and if Streamer code has been created, 0 otherwise.

   int stltype = abs(IsSTLContainer(m));
   if (m.IsTmplt() && stltype>0) {
      char ss[kMaxLen];strcpy(ss,TemplateArg(m).Name());char *s=ss;

      if (rwmode == 0) {
         // create read code
         (*dictSrcOut) << "      {" << std::endl;
         char tmparg[kMaxLen];
         strcpy(tmparg,m.Name());
         int lenarg = strlen(tmparg);
         if (tmparg[lenarg-1] == '*') {tmparg[lenarg-1] = 0; lenarg--;}
         if (tmparg[lenarg-1] == '*') {tmparg[lenarg-1] = 0; lenarg--;}
         if (!strncmp(s, "const ", 6)) s += 6;
         (*dictSrcOut) << "         clear();" << std::endl
             << "         int R__i, R__n;" << std::endl
             << "         R__b >> R__n;" << std::endl
             << "         for (R__i = 0; R__i < R__n; R__i++) {" << std::endl
             << "            " << s << " R__t;" << std::endl;
         if ((TemplateArg(m).Property() & G__BIT_ISPOINTER) ||
             (TemplateArg(m).Property() & G__BIT_ISFUNDAMENTAL) ||
             (TemplateArg(m).Property() & G__BIT_ISENUM)) {
            if (TemplateArg(m).Property() & G__BIT_ISENUM)
               (*dictSrcOut) << "            R__b >> (Int_t&)R__t;" << std::endl;
            else {
               if (stltype == kMap || stltype == kMultiMap) {
                  (*dictSrcOut) << "            R__b >> R__t;" << std::endl;
                  if ((TemplateArg(m,1).Property() & G__BIT_ISPOINTER) ||
                  (TemplateArg(m,1).Property() & G__BIT_ISFUNDAMENTAL) ||
                  (TemplateArg(m,1).Property() & G__BIT_ISENUM)) {
                     (*dictSrcOut) << "            " << TemplateArg(m,1).Name() << " R__t2;" << std::endl
                         << "            R__b >> R__t2;" << std::endl;
                  } else {
                     if (strcmp(TemplateArg(m,1).Name(),"string") == 0) {
                        (*dictSrcOut) << "            TString R__str;" << std::endl
                            << "            R__str.Streamer(R__b);" << std::endl
                            << "            string R__t2 = R__str.Data();" << std::endl;
                     } else {
                        (*dictSrcOut) << "            " << TemplateArg(m,1).Name() << " R__t2;" << std::endl
                            << "            R__t2.Streamer(R__b);" << std::endl;
                     }
                  }
               } else if (stltype == kSet || stltype == kMultiSet) {
                  (*dictSrcOut) << "            R__b >> R__t;" << std::endl;
               } else {
                  if (strcmp(s,"string*") == 0) {
                     (*dictSrcOut) << "            TString R__str;" << std::endl
                         << "            R__str.Streamer(R__b);" << std::endl
                         << "            R__t = new string(R__str.Data());" << std::endl;
                  } else {
                     (*dictSrcOut) << "            R__b >> R__t;" << std::endl;
                  }
               }
            }
         } else {
            if (TemplateArg(m).HasMethod("Streamer")) {
               if (stltype == kMap || stltype == kMultiMap) {
                  (*dictSrcOut) << "            R__t.Streamer(R__b);" << std::endl;
                  if ((TemplateArg(m,1).Property() & G__BIT_ISPOINTER) ||
                      (TemplateArg(m,1).Property() & G__BIT_ISFUNDAMENTAL) ||
                      (TemplateArg(m,1).Property() & G__BIT_ISENUM)) {
                     (*dictSrcOut) << "            " << TemplateArg(m,1).Name() << " R__t2;" << std::endl;
                         << "            R__b >> R__t2;" << std::endl;
                  } else {
                     if (strcmp(TemplateArg(m,1).Name(),"string") == 0) {
                        (*dictSrcOut) << "            TString R__str;" << std::endl
                            << "            R__str.Streamer(R__b); << std::endl"
                            << "            string R__t2 = R__str.Data();" << std::endl;
                     } else {
                        (*dictSrcOut) << "            " << TemplateArg(m,1).Name() << " R__t2;" << std::endl
                            << "            R__t2.Streamer(R__b);" << std::endl;
                     }
                  }
               } else {
                  (*dictSrcOut) << "            R__t.Streamer(R__b);" << std::endl;
               }
            } else {
               if (strcmp(s,"string") == 0) {
                  (*dictSrcOut) << "            TString R__str;" << std::endl
                      << "            R__str.Streamer(R__b);" << std::endl;
                      << "            R__t = R__str.Data();" << std::endl;
               } else {
                  (*dictSrcOut) << "R__b.StreamObject(&R__t,typeid(" << s << "));" << std::endl;
//VP                 Error(0, "*** Baseclass %s: template arg %s has no Streamer()"
//VP                          " method (need manual intervention)\n",
//VP                          m.Name(), TemplateArg(m).Name());
//VP                 (*dictSrcOut) << "            //R__t.Streamer(R__b);" << std::endl;
               }
            }
         }
         if (m.Property() & G__BIT_ISPOINTER) {
            if (stltype == kMap || stltype == kMultiMap) {
               (*dictSrcOut) << "            insert(make_pair(R__t,R__t2));" << std::endl;
            } else if (stltype == kSet || stltype == kMultiSet) {
               (*dictSrcOut) << "            insert(R__t);" << std::endl;
            } else {
               (*dictSrcOut) << "            push_back(R__t);" << std::endl;
            }
         } else {
            if (stltype == kMap || stltype == kMultiMap) {
               (*dictSrcOut) << "            insert(make_pair(R__t,R__t2));");
            } else if (stltype == kSet || stltype == kMultiSet) {
               fprintf(fp, "            insert(R__t);\n");
            } else {
               fprintf(fp, "            push_back(R__t);\n");
            }
         }
         fprintf(fp, "         }\n");
         fprintf(fp, "      }\n");
      } else {
         // create write code
         fprintf(fp, "      {\n");
         fprintf(fp, "         R__b << int(size());\n");
         char tmparg[kMaxLen];
         strcpy(tmparg,m.Name());
         int lenarg = strlen(tmparg);
         if (tmparg[lenarg-1] == '*') {tmparg[lenarg-1] = 0; lenarg--;}
         if (tmparg[lenarg-1] == '*') {tmparg[lenarg-1] = 0; lenarg--;}
         fprintf(fp, "         %s::iterator R__k;\n", tmparg);
         fprintf(fp, "         for (R__k = begin(); R__k != end(); ++R__k) {\n");
         if ((TemplateArg(m).Property() & G__BIT_ISPOINTER) ||
             (TemplateArg(m).Property() & G__BIT_ISFUNDAMENTAL) ||
             (TemplateArg(m).Property() & G__BIT_ISENUM)) {
            if (TemplateArg(m).Property() & G__BIT_ISENUM)
               fprintf(fp, "            R__b << (Int_t)*R__k;\n");
            else {
               if (stltype == kMap || stltype == kMultiMap) {
                  fprintf(fp, "            R__b << (*R__k).first;\n");
                  if ((TemplateArg(m,1).Property() & G__BIT_ISPOINTER) ||
                  (TemplateArg(m,1).Property() & G__BIT_ISFUNDAMENTAL) ||
                  (TemplateArg(m,1).Property() & G__BIT_ISENUM)) {
                     fprintf(fp, "            R__b << (*R__k).second;\n");
                  } else {
                     if (strcmp(TemplateArg(m,1).Name(),"string") == 0) {
                        fprintf(fp, "            TString R__str = ((%s&)((*R__k).second)).c_str();\n",TemplateArg(m,1).Name());
                        fprintf(fp, "            R__str.Streamer(R__b);\n");
                     } else {
                        fprintf(fp, "            ((%s&)((*R__k).second)).Streamer(R__b);\n",TemplateArg(m,1).Name());
                     }
                  }
               } else if (stltype == kSet || stltype == kMultiSet) {
                  fprintf(fp, "            R__b << *R__k;\n");
               } else {
                  if (strcmp(TemplateArg(m).Name(),"string*") == 0) {
                     fprintf(fp,"            TString R__str = (*R__k)->c_str();\n");
                     fprintf(fp,"            R__str.Streamer(R__b);\n");
                  } else {
                     if (strcmp(TemplateArg(m).Name(),"(unknown)") == 0) {
                        Error(0, "cannot process template argument1 %s\n",tmparg);
                        fprintf(fp, "            //R__b << *R__k;\n");
                     } else {
                        fprintf(fp, "            R__b << *R__k;\n");
                     }
                  }
               }
            }
         } else {
            if (TemplateArg(m).HasMethod("Streamer")) {
               if (stltype == kMap || stltype == kMultiMap) {
                  fprintf(fp, "            ((%s&)((*R__k).first)).Streamer(R__b);\n",TemplateArg(m).Name());
                  if ((TemplateArg(m,1).Property() & G__BIT_ISPOINTER) ||
                      (TemplateArg(m,1).Property() & G__BIT_ISFUNDAMENTAL) ||
                      (TemplateArg(m,1).Property() & G__BIT_ISENUM)) {
                     fprintf(fp, "            R__b << (*R__k).second;\n");
                  } else {
                     if (strcmp(TemplateArg(m,1).Name(),"string") == 0) {
                        fprintf(fp, "            TString R__str = ((%s&)((*R__k).second)).c_str();\n",TemplateArg(m,1).Name());
                        fprintf(fp, "            R__str.Streamer(R__b);\n");
                     } else {
                        fprintf(fp, "            ((%s&)((*R__k).second)).Streamer(R__b);\n",TemplateArg(m,1).Name());
                     }
                  }
               } else if (stltype == kSet || stltype == kMultiSet) {
                  fprintf(fp, "            (*R__k).Streamer(R__b);\n");
               } else {
                  fprintf(fp, "            (*R__k).Streamer(R__b);\n");
               }
            } else {
               if (strcmp(TemplateArg(m).Name(),"string") == 0) {
                  fprintf(fp,"            TString R__str = (*R__k).c_str();\n");
                  fprintf(fp,"            R__str.Streamer(R__b);\n");
               } else {
                  if (strcmp(TemplateArg(m).Name(),"(unknown)") == 0) {
                     Error(0, "cannot process template argument2 %s\n",tmparg);
                     fprintf(fp, "            //(*R__k).Streamer(R__b);\n");
                  } else {
                     fprintf(fp, "R__b.StreamObject(R__k,typeid(%s));\n",s);               //R__t.Streamer(R__b);\n");
                     //VP                    fprintf(fp, "            //(*R__k).Streamer(R__b);\n");
                  }
               }
            }
         }
         fprintf(fp, "         }\n");
         fprintf(fp, "      }\n");
      }
      return 1;
   }
   return 0;
}
#endif
//______________________________________________________________________________
int PointerToPointer(G__DataMemberInfo &m)
{
   if (strstr(m.Type()->Name(), "**")) return 1;
   return 0;
}

//______________________________________________________________________________
void WriteArrayDimensions(int dim)
{
   for (int i = 0; i < dim-1; i++)
      (*dictSrcOut) << "[0]";
}

//______________________________________________________________________________
void WriteInputOperator(G__ClassInfo &cl)
{
   if (cl.IsBase("TObject") || !strcmp(cl.Fullname(), "TObject"))
      return;

   (*dictSrcOut) << "//_______________________________________"
       << "_______________________________________" << std::endl;

   char space_prefix[kMaxLen] = "";
#ifdef WIN32
   G__ClassInfo space = cl.EnclosingSpace();
   if (space.Property() & G__BIT_ISNAMESPACE)
      sprintf(space_prefix,"%s::",space.Fullname());
#endif

   if (cl.IsTmplt()) {
      // Produce specialisation for templates:
      (*dictSrcOut) << "template<> TBuffer &operator>>"
          << "(TBuffer &buf, " << cl.Fullname() << " *&obj)" << std::endl
          << "{" << std::endl;
   } else {
      (*dictSrcOut) << "template<> TBuffer &" << space_prefix << "operator>>(TBuffer &buf, "
          << cl.Fullname() << " *&obj)" << std::endl
          << "{" << std::endl;
   }
   (*dictSrcOut) << "   // Read a pointer to an object of class " << cl.Fullname() << "."
       << std::endl << std::endl;

   if (cl.IsBase("TObject") || !strcmp(cl.Fullname(), "TObject")) {
      (*dictSrcOut) << "   obj = (" << cl.Fullname() << " *) buf.ReadObjectAny("
          << cl.Fullname() << "::Class());" << std::endl;
   } else {
      (*dictSrcOut) << "   ::Error(\"" << cl.Fullname() << "::operator>>\", \"objects not inheriting"
                  " from TObject need a specialized operator>>"
                  " function\"); if (obj) { }" << std::endl;
   }
   (*dictSrcOut) << "   return buf;" << std::endl << "}" << std::endl << std::endl;
}

//______________________________________________________________________________
void WriteClassFunctions(G__ClassInfo &cl, int /*tmplt*/ = 0)
{
   // Write the code to set the class name and the initialization object.

   int add_template_keyword = NeedTemplateKeyword(cl);

   G__ClassInfo ns = cl.EnclosingSpace();
   string clsname = cl.Fullname();
   string nsname;
   if (ns.IsValid()) {
     nsname = ns.Fullname();
     clsname.erase (0, nsname.size() + 2);
   }

   int enclSpaceNesting = 0;
   if (!nsname.empty()) {
     G__ShadowMaker nestTempShadowMaker(*dictSrcOut, "");
     enclSpaceNesting = nestTempShadowMaker.WriteNamespaceHeader(cl);
   }

   (*dictSrcOut) << "//_______________________________________"
       << "_______________________________________" << std::endl;
   if (add_template_keyword) (*dictSrcOut) << "template <> ";
   (*dictSrcOut) << "TClass *" << clsname.c_str() << "::fgIsA = 0;  // static to hold class pointer" << std::endl
       << std::endl

       << "//_______________________________________"
       << "_______________________________________" << std::endl;
   if (add_template_keyword) (*dictSrcOut) << "template <> ";
   (*dictSrcOut) << "const char *" << clsname.c_str() << "::Class_Name()" << std::endl << "{" << std::endl
       << "   return \"" << cl.Fullname() << "\";"  << std::endl <<"}" << std::endl << std::endl;

   if (1 || !cl.IsTmplt()) {
      // If the class is not templated and has a ClassDef,
      // a ClassImp is required and already defines those function:

      (*dictSrcOut) << "//_______________________________________"
          << "_______________________________________" << std::endl;
      if (add_template_keyword) (*dictSrcOut) << "template <> ";
      (*dictSrcOut) << "const char *" << clsname.c_str() << "::ImplFileName()"  << std::endl << "{" << std::endl
          << "   return ::ROOT::GenerateInitInstanceLocal((const ::" << cl.Fullname()
          << "*)0x0)->GetImplFileName();" << std::endl << "}" << std::endl << std::endl

          << "//_______________________________________"
          << "_______________________________________" << std::endl;
      if (add_template_keyword) (*dictSrcOut) <<"template <> ";
      (*dictSrcOut) << "int " << clsname.c_str() << "::ImplFileLine()" << std::endl << "{" << std::endl
          << "   return ::ROOT::GenerateInitInstanceLocal((const ::" << cl.Fullname()
          << "*)0x0)->GetImplFileLine();" << std::endl << "}" << std::endl << std::endl

          << "//_______________________________________"
          << "_______________________________________" << std::endl;
      if (add_template_keyword) (*dictSrcOut) << "template <> ";
      (*dictSrcOut) << "void " << clsname.c_str() << "::Dictionary()" << std::endl << "{" << std::endl
          << "   fgIsA = ::ROOT::GenerateInitInstanceLocal((const ::" << cl.Fullname()
          << "*)0x0)->GetClass();" << std::endl
          << "}" << std::endl << std::endl

          << "//_______________________________________"
          << "_______________________________________" << std::endl;
      if (add_template_keyword) (*dictSrcOut) << "template <> ";
      (*dictSrcOut) << "TClass *" << clsname.c_str() << "::Class()" << std::endl << "{" << std::endl;
      (*dictSrcOut) << "   if (!fgIsA) fgIsA = ::ROOT::GenerateInitInstanceLocal((const ::";
      (*dictSrcOut) << cl.Fullname() << "*)0x0)->GetClass();" << std::endl
                    << "   return fgIsA;" << std::endl
                    << "}" << std::endl << std::endl;
   }

   while (enclSpaceNesting) {
     (*dictSrcOut) << "} // namespace " << nsname << std::endl;
     --enclSpaceNesting;
   }
}

//______________________________________________________________________________
void WriteClassInit(G__ClassInfo &cl)
{
   // Write the code to initialize the class name and the initialization object.

   string classname = GetLong64_Name( RStl::DropDefaultArg( cl.Fullname() ) );
   string mappedname = G__map_cpp_name((char*)classname.c_str());
   string csymbol = classname;
   string args;

   if ( ! TClassEdit::IsStdClass( classname.c_str() ) ) {

      // Prefix the full class name with '::' except for the STL
      // containers and std::string.  This is to request the
      // real class instead of the class in the namespace ROOT::Shadow
      csymbol.insert(0,"::");
   }

   int stl = TClassEdit::IsSTLCont(classname.c_str());

   (*dictSrcOut) << "namespace ROOT {" << std::endl
       << "   void " << mappedname.c_str() << "_ShowMembers(void *obj, TMemberInspector &R__insp, char *R__parent);"
       << std::endl;

   if (!cl.HasMethod("Dictionary") || cl.IsTmplt())
      (*dictSrcOut) << "   static void " << mappedname.c_str() << "_Dictionary();" << std::endl;

   if (HasDefaultConstructor(cl,&args)) {
      (*dictSrcOut) << "   static void *new_" << mappedname.c_str() << "(void *p = 0);" << std::endl;
      if (args.size()==0)
         (*dictSrcOut) << "   static void *newArray_" << mappedname.c_str()
                       << "(Long_t size, void *p);" << std::endl;
   }
   if (NeedDestructor(cl)) {
      (*dictSrcOut) << "   static void delete_" << mappedname.c_str() << "(void *p);" << std::endl
          << "   static void deleteArray_" << mappedname.c_str() << "(void *p);" << std::endl
          << "   static void destruct_" << mappedname.c_str() << "(void *p);" << std::endl;
   }
   (*dictSrcOut) << std::endl

       << "   // Function generating the singleton type initializer" << std::endl;

#if 0
   fprintf(fp, "#if defined R__NAMESPACE_TEMPLATE_IMP_BUG\n");
   fprintf(fp, "   template <> ::ROOT::TGenericClassInfo *::ROOT::GenerateInitInstanceLocal< %s >(const %s*)\n   {\n",
           cl.Fullname(), cl.Fullname() );
   fprintf(fp, "#else\n");
   fprintf(fp, "   template <> ::ROOT::TGenericClassInfo *GenerateInitInstanceLocal< %s >(const %s*)\n   {\n",
           classname.c_str(), classname.c_str() );
   fprintf(fp, "#endif\n");
#endif


   (*dictSrcOut) << "   static TGenericClassInfo *GenerateInitInstanceLocal(const " << csymbol.c_str() << "*)" << std::endl
                    << "   {" << std::endl;

   if (NeedShadowClass(cl)) {
      (*dictSrcOut) << "      // Make sure the shadow class has the right sizeof" << std::endl;
      if (G__ShadowMaker::IsStdPair(cl)) {
         // Some compiler don't recognize ::pair even after a 'using namespace std;'
         // and there is not risk of confusion since it is a template.
         //fprintf(fp, "      R__ASSERT(sizeof(%s)", classname.c_str() );
      } else {
         std::string clfullname;
         shadowMaker->GetFullShadowName(cl, clfullname);
         (*dictSrcOut) << "      R__ASSERT(sizeof(" << csymbol.c_str() << ")"
             << " == sizeof(" << clfullname.c_str() << "));" << std::endl;
      }
   }

   (*dictSrcOut) << "      " << csymbol.c_str() << " *ptr = 0;" << std::endl;

   //fprintf(fp, "      static ::ROOT::ClassInfo< %s > \n",classname.c_str());
   if ( cl.HasMethod("IsA") ) {
      (*dictSrcOut) << "      static ::TVirtualIsAProxy* isa_proxy = new ::TInstrumentedIsAProxy< "
          << csymbol.c_str() << " >(0);" << std::endl;
   }
   else {
      (*dictSrcOut) << "      static ::TVirtualIsAProxy* isa_proxy = new ::TIsAProxy(typeid("
          << csymbol.c_str() << "),0);" << std::endl;
   }
   (*dictSrcOut) << "      static ::ROOT::TGenericClassInfo " << std::endl

       << "         instance(\"" << classname.c_str() << "\", ";

   if (cl.HasMethod("Class_Version")) {
      (*dictSrcOut) << csymbol.c_str() << "::Class_Version(), ";
   } else if (stl) {
      (*dictSrcOut) << "::TStreamerInfo::Class_Version(), ";
   } else { // if (cl.RootFlag() & G__USEBYTECOUNT ) {

      // Need to find out if the operator>> is actually defined for this class.
      G__ClassInfo gcl;
      long offset;
      const char *versionFunc = "GetClassVersion";
      char *funcname= new char[strlen(classname.c_str())+strlen(versionFunc)+5];
      sprintf(funcname,"%s<%s >",versionFunc,classname.c_str());
      char *proto = new char[strlen(classname.c_str())+ 10 ];
      sprintf(proto,"%s*",classname.c_str());
      G__MethodInfo methodinfo = gcl.GetMethod(versionFunc,proto,&offset);
      delete [] funcname;
      delete [] proto;

      if (methodinfo.IsValid() &&
          //          methodinfo.ifunc()->para_p_tagtable[methodinfo.Index()][0] == cl.Tagnum() &&
          strstr(methodinfo.FileName(),"Rtypes.h") == 0) {

         // GetClassVersion was defined in the header file.
         //fprintf(fp, "GetClassVersion((%s *)0x0), ",classname.c_str());
         (*dictSrcOut) << "GetClassVersion< " << classname.c_str() << " >(), ";
      }
      //static char temporary[1024];
      //sprintf(temporary,"GetClassVersion<%s>( (%s *) 0x0 )",classname.c_str(),classname.c_str());
      //fprintf(stderr,"DEBUG: %s has value %d\n",classname.c_str(),(int)G__int(G__calc(temporary)));
   }

   char *filename = (char*)cl.FileName();
   if (filename) {
      for (unsigned int i=0; i<strlen(filename); i++) {
         if (filename[i]=='\\') filename[i]='/';
      }
   } else {
      filename = "";
   }
   (*dictSrcOut) << "\"" << filename << "\", " << cl.LineNumber() << "," << std::endl
       << "                  typeid(" << csymbol.c_str() << "), DefineBehavior(ptr, ptr)," << std::endl
   //   fprintf(fp, "                  (::ROOT::ClassInfo< %s >::ShowMembersFunc_t)&::ROOT::ShowMembers,%d);\n", classname.c_str(),cl.RootFlag());
       << "                  ";
   if (!NeedShadowClass(cl)) {
      if (!cl.HasMethod("ShowMembers")) (*dictSrcOut) << "0, ";
   } else {
      if (!cl.HasMethod("ShowMembers"))
         (*dictSrcOut) << "(void*)&" << mappedname.c_str() << "_ShowMembers, ";
   }

   if (cl.HasMethod("Dictionary") && !cl.IsTmplt()) {
      (*dictSrcOut) << "&" << csymbol.c_str() << "::Dictionary, ";
   } else {
      (*dictSrcOut) << "&" << mappedname.c_str() << "_Dictionary, ";
   }

   (*dictSrcOut) << "isa_proxy, " << cl.RootFlag() << "," << std::endl
       << "                  sizeof(" << csymbol.c_str() << ") );" << std::endl;
   if (HasDefaultConstructor(cl,&args)) {
      (*dictSrcOut) << "      instance.SetNew(&new_" << mappedname.c_str() << ");" << std::endl;
      if (args.size()==0)
         (*dictSrcOut) << "      instance.SetNewArray(&newArray_" << mappedname.c_str() << ");" << std::endl;
   }
   if (NeedDestructor(cl)) {
      (*dictSrcOut) << "      instance.SetDelete(&delete_" << mappedname.c_str() << ");" << std::endl
          << "      instance.SetDeleteArray(&deleteArray_" << mappedname.c_str() << ");" << std::endl
          << "      instance.SetDestructor(&destruct_" << mappedname.c_str() << ");" << std::endl;
   }
   if (stl != 0 && ((stl>0 && stl<8) || (stl<0 && stl>-8)) )  {
      int idx = classname.find("<");
      int stlType = (idx!=(int)std::string::npos) ? TClassEdit::STLKind(classname.substr(0,idx).c_str()) : 0;
      const char* methodTCP=0;
      switch(stlType)  {
         case TClassEdit::kVector:
         case TClassEdit::kList:
         case TClassEdit::kDeque:
            methodTCP="Pushback";
            break;
         case TClassEdit::kMap:
         case TClassEdit::kMultiMap:
            methodTCP="MapInsert";
            break;
         case TClassEdit::kSet:
         case TClassEdit::kMultiSet:
            methodTCP="Insert";
            break;
      }
      (*dictSrcOut) << "      instance.AdoptStreamer(TCollectionProxy::GenClassStreamer(TCollectionProxy::"
          << methodTCP << "< " << classname.c_str() << " >()));" << std::endl
          << "      instance.AdoptCollectionProxy(TCollectionProxy::GenProxy(TCollectionProxy::"
          << methodTCP << "< " << classname.c_str() << " >()));" << std::endl;
   }
   (*dictSrcOut) << "      return &instance;"  << std::endl
                 << "   }" << std::endl;

   if (!stl) {
      // The GenerateInitInstance for STL are not unique and should not be externally accessible
      (*dictSrcOut) << "   TGenericClassInfo *GenerateInitInstance(const " << csymbol.c_str() << "*)" << std::endl
                    << "   {\n      return GenerateInitInstanceLocal((" <<  csymbol.c_str() << "*)0);\n   }" 
                    << std::endl;
   }
    
   (*dictSrcOut) << "   // Static variable to force the class initialization" << std::endl;
   // must be one long line otherwise R__UseDummy does not work


   (*dictSrcOut) 
      << "   static ::ROOT::TGenericClassInfo *_R__UNIQUE_(Init) = GenerateInitInstanceLocal((const "
      << csymbol.c_str() << "*)0x0); R__UseDummy(_R__UNIQUE_(Init));" << std::endl;
 
   if (!cl.HasMethod("Dictionary") || cl.IsTmplt()) {
      (*dictSrcOut) <<  std::endl << "   // Dictionary for non-ClassDef classes" << std::endl
                    << "   static void " << mappedname.c_str() << "_Dictionary() {" << std::endl;
      (*dictSrcOut) << "      ::ROOT::GenerateInitInstanceLocal((const " << csymbol.c_str();
      (*dictSrcOut) << "*)0x0)->GetClass();" << std::endl
                    << "   }" << std::endl << std::endl;
   }

   (*dictSrcOut) << "} // end of namespace ROOT" << std::endl << std::endl;
}

//______________________________________________________________________________
void WriteNamespaceInit(G__ClassInfo &cl)
{
   // Write the code to initialize the namespace name and the initialization object.

   if (! (cl.Property() & G__BIT_ISNAMESPACE) ) return;

   string classname = GetLong64_Name( RStl::DropDefaultArg( cl.Fullname() ) );
   string mappedname = G__map_cpp_name((char*)classname.c_str());

   int nesting = 0;
   // We should probably unwind the namespace to properly nest it.
   if (classname!="ROOT") {
      string right = classname;
      int pos = right.find(":");
      if (pos==0) {
         right = right.substr(2);
         pos = right.find(":");
      }
      while(pos>=0) {
         string left = right.substr(0,pos);
         right = right.substr(pos+2);
         pos = right.find(":");
         ++nesting;
         (*dictSrcOut) << "namespace " << left << " {" << std::endl;
      }

      ++nesting;
      (*dictSrcOut) << "namespace " << right << " {" << std::endl;
   }

   (*dictSrcOut) << "   namespace ROOT {" << std::endl;

#if !defined(R__SGI) && !defined(R__AIX)
   (*dictSrcOut) << "      inline ::ROOT::TGenericClassInfo *GenerateInitInstance();" << std::endl;
#endif

   if (!cl.HasMethod("Dictionary") || cl.IsTmplt())
      (*dictSrcOut) << "      static void " << mappedname.c_str() << "_Dictionary();" << std::endl;

   (*dictSrcOut) << std::endl

       << "      // Function generating the singleton type initializer" << std::endl

#if !defined(R__SGI) && !defined(R__AIX)
       << "      inline ::ROOT::TGenericClassInfo *GenerateInitInstance()" << std::endl
       << "      {" << std::endl
#else
       << "      ::ROOT::TGenericClassInfo *GenerateInitInstance()" << std::endl
       << "      {" << std::endl
#endif

       << "         static ::ROOT::TGenericClassInfo " << std::endl

       << "            instance(\"" << classname.c_str() << "\", ";

   if (cl.HasMethod("Class_Version")) {
      (*dictSrcOut) << "::" << classname.c_str() << "::Class_Version(), ";
   } else {

      // Need to find out if the operator>> is actually defined for this class.
      G__ClassInfo gcl;
      long offset;
      const char *versionFunc = "GetClassVersion";
      char *funcname= new char[strlen(classname.c_str())+strlen(versionFunc)+5];
      sprintf(funcname,"%s<%s >",versionFunc,classname.c_str());
      char *proto = new char[strlen(classname.c_str())+ 10 ];
      sprintf(proto,"%s*",classname.c_str());
      G__MethodInfo methodinfo = gcl.GetMethod(versionFunc,proto,&offset);
      delete [] funcname;
      delete [] proto;

      if (methodinfo.IsValid() &&
          strstr(methodinfo.FileName(),"Rtypes.h") == 0) {
         (*dictSrcOut) << "GetClassVersion< " << classname.c_str() << " >(), ";
      } else {
         (*dictSrcOut) << "0 /*version*/, ";
      }
   }

   char *filename = (char*)cl.FileName();
   for (unsigned int i=0; i<strlen(filename); i++) {
      if (filename[i]=='\\') filename[i]='/';
   }
   (*dictSrcOut) << "\"" << filename << "\", " << cl.LineNumber() << "," << std::endl
       << "                     ::ROOT::DefineBehavior((void*)0,(void*)0)," << std::endl
       << "                     ";

   if (cl.HasMethod("Dictionary") && !cl.IsTmplt()) {
      (*dictSrcOut) << "&::" << classname.c_str() << "::Dictionary, ";
   } else {
      (*dictSrcOut) << "&" << mappedname.c_str() << "_Dictionary, ";
   }

   (*dictSrcOut) << cl.RootFlag() << ");" << std::endl

       << "         return &instance;" << std::endl
       << "      }" << std::endl
       << "      // Static variable to force the class initialization" << std::endl
   // must be one long line otherwise R__UseDummy does not work
       << "      static ::ROOT::TGenericClassInfo *_R__UNIQUE_(Init) = GenerateInitInstance();"
       << " R__UseDummy(_R__UNIQUE_(Init));" << std::endl;

   if (!cl.HasMethod("Dictionary") || cl.IsTmplt()) {
      (*dictSrcOut) <<  std::endl << "      // Dictionary for non-ClassDef classes" << std::endl
          << "      static void " << mappedname.c_str() << "_Dictionary() {" << std::endl
          << "         GenerateInitInstance()->GetClass();" << std::endl
          << "      }" << std::endl << std::endl;
   }

   (*dictSrcOut) << "   }" << std::endl;
   while(nesting--) {
      (*dictSrcOut) << "}" << std::endl;
   }
   (*dictSrcOut) <<  std::endl;
}

//______________________________________________________________________________
const char *ShortTypeName(const char *typeDesc)
{
   // Return the absolute type of typeDesc.
   // E.g.: typeDesc = "class TNamed**", returns "TNamed".
   // we remove * and const keywords. (we do not want to remove & ).
   // You need to use the result immediately before it is being overwritten.

   static char t[1024];
   static const char* constwd = "const ";
   static const char* constwdend = "const";

   const char *s;
   char *p=t;
   int lev=0;
   for (s=typeDesc;*s;s++) {
      if (*s=='<') lev++;
      if (*s=='>') lev--;
      if (lev==0 && *s=='*') continue;
      if (lev==0 && (strncmp(constwd,s,strlen(constwd))==0
                     ||strcmp(constwdend,s)==0 ) ) {
         s+=strlen(constwd)-1; // -1 because the loop adds 1
         continue;
      }
      if (lev==0 && *s==' ' && *(s+1)!='*') { p = t; continue;}
      *p++ = *s;
   }
   p[0]=0;

   return t;
}

//______________________________________________________________________________
const char *GrabIndex(G__DataMemberInfo &member, int printError)
{
   // GrabIndex returns a static string (so use it or copy it immediatly, do not
   // call GrabIndex twice in the same expression) containing the size of the
   // array data member.
   // In case of error, or if the size is not specified, GrabIndex returns 0.

   int error;
   char *where = 0;

   const char *index = member.ValidArrayIndex(&error, &where);
   if (index==0 && printError) {
      const char *errorstring;
      switch (error) {
         case G__DataMemberInfo::NOT_INT:
            errorstring = "is not an integer";
            break;
         case G__DataMemberInfo::NOT_DEF:
            errorstring = "has not been defined before the array";
            break;
         case G__DataMemberInfo::IS_PRIVATE:
            errorstring = "is a private member of a parent class";
            break;
         case G__DataMemberInfo::UNKNOWN:
            errorstring = "is not known";
            break;
         default:
            errorstring = "UNKNOWN ERROR!!!!";
      }

      if (where==0) {
         Error(0, "*** Datamember %s::%s: no size indication!\n",
                     member.MemberOf()->Name(), member.Name());
      } else {
         Error(0,"*** Datamember %s::%s: size of array (%s) %s!\n",
                  member.MemberOf()->Name(), member.Name(), where, errorstring);
      }
   }
   return index;
}

//______________________________________________________________________________
void WriteStreamer(G__ClassInfo &cl)
{
   int add_template_keyword = NeedTemplateKeyword(cl);

   G__ClassInfo ns = cl.EnclosingSpace();
   string clsname = cl.Fullname();
   string nsname;
   if (ns.IsValid()) {
     nsname = ns.Fullname();
     clsname.erase (0, nsname.size() + 2);
   }

   int enclSpaceNesting = 0;
   if (!nsname.empty()) {
     G__ShadowMaker nestTempShadowMaker(*dictSrcOut, "");
     enclSpaceNesting = nestTempShadowMaker.WriteNamespaceHeader(cl);
   }

   (*dictSrcOut) << "//_______________________________________"
       << "_______________________________________" << std::endl;
   if (add_template_keyword) (*dictSrcOut) << "template <> ";
   (*dictSrcOut) << "void " << clsname << "::Streamer(TBuffer &R__b)"  << std::endl << "{" << std::endl
       << "   // Stream an object of class " << cl.Fullname() << "." << std::endl << std::endl;

   // In case of VersionID<=0 write dummy streamer only calling
   // its base class Streamer(s). If no base class(es) let Streamer
   // print error message, i.e. this Streamer should never have been called.
   int version = GetClassVersion(cl);
   if (version <= 0) {
      G__BaseClassInfo b(cl);

      int basestreamer = 0;
      while (b.Next())
         if (b.HasMethod("Streamer")) {
            if (strstr(b.Fullname(),"::")) {
               // there is a namespace involved, trigger MS VC bug workaround
               (*dictSrcOut) << "   //This works around a msvc bug and should be harmless on other platforms" << std::endl
                   << "   typedef " << b.Fullname() << " baseClass" << basestreamer << ";" << std::endl
                   << "   baseClass" << basestreamer << "::Streamer(R__b);" << std::endl;
            }
            else
               (*dictSrcOut) << "   " << b.Fullname() << "::Streamer(R__b);" << std::endl;
            basestreamer++;
         }
      if (!basestreamer) {
         (*dictSrcOut) << "   ::Error(\"" << cl.Fullname() << "::Streamer\", \"version id <=0 in ClassDef,"
                 " dummy Streamer() called\"); if (R__b.IsReading()) { }" << std::endl;
      }
      (*dictSrcOut) << "}" << std::endl << std::endl;
      while (enclSpaceNesting) {
         (*dictSrcOut) << "} // namespace " << nsname.c_str() << std::endl;
         --enclSpaceNesting;
      }
      return;
   }

   // see if we should generate Streamer with extra byte count code
   int ubc = 0;
   //if ((cl.RootFlag() & G__USEBYTECOUNT)) ubc = 1;
   ubc = 1;   // now we'll always generate byte count streamers

   // loop twice: first time write reading code, second time writing code
   string classname = cl.Fullname();
   if (strstr(cl.Fullname(),"::")) {
      // there is a namespace involved, trigger MS VC bug workaround
      (*dictSrcOut) << "   //This works around a msvc bug and should be harmless on other platforms" << std::endl
          << "   typedef ::" << cl.Fullname() << " thisClass;" << std::endl;
      classname = "thisClass";
   }
   for (int i = 0; i < 2; i++) {

      int decli = 0;

      if (i == 0) {
         if (ubc) (*dictSrcOut) << "   UInt_t R__s, R__c;" << std::endl;
         (*dictSrcOut) << "   if (R__b.IsReading()) {" << std::endl;
         if (ubc)
            (*dictSrcOut) << "      Version_t R__v = R__b.ReadVersion(&R__s, &R__c); if (R__v) { }" << std::endl;
         else
            (*dictSrcOut) << "      Version_t R__v = R__b.ReadVersion(); if (R__v) { }" << std::endl;
      } else {
         if (ubc) (*dictSrcOut) << "      R__b.CheckByteCount(R__s, R__c, " << classname.c_str() << "::IsA());" << std::endl;
         (*dictSrcOut) << "   } else {" << std::endl;
         if (ubc)
            (*dictSrcOut) << "      R__c = R__b.WriteVersion(" << classname.c_str() << "::IsA(), kTRUE);" << std::endl;
         else
            (*dictSrcOut) << "      R__b.WriteVersion(" << classname.c_str() << "::IsA());" << std::endl;
      }

      // Stream base class(es) when they have the Streamer() method
      G__BaseClassInfo b(cl);

      int base=0;
      while (b.Next()) {
         if (b.HasMethod("Streamer")) {
            if (strstr(b.Fullname(),"::")) {
               // there is a namespace involved, trigger MS VC bug workaround
               (*dictSrcOut) << "      //This works around a msvc bug and should be harmless on other platforms" << std::endl
                   << "      typedef " << b.Fullname() << " baseClass" << base << ";" << std::endl
                   << "      baseClass" << base << "::Streamer(R__b);" << std::endl;
               ++base;
            }
            else
               (*dictSrcOut) << "      " << b.Fullname() << "::Streamer(R__b);" << std::endl;
         }
      }
      // Stream data members
      G__DataMemberInfo m(cl);

      while (m.Next()) {

         // we skip:
         //  - static members
         //  - members with an ! as first character in the title (comment) field
         //  - the member G__virtualinfo inserted by the CINT RTTI system

         //special case for Double32_t
         int isDouble32=0;
         if (strstr(m.Type()->Name(),"Double32_t")) isDouble32=1;

         if (!(m.Property() & G__BIT_ISSTATIC) &&
             strncmp(m.Title(), "!", 1)        &&
             strcmp(m.Name(), "G__virtualinfo")) {

            // fundamental type: short, int, long, etc....
            if (((m.Type())->Property() & G__BIT_ISFUNDAMENTAL) ||
                ((m.Type())->Property() & G__BIT_ISENUM)) {
               if (m.Property() & G__BIT_ISARRAY &&
                   m.Property() & G__BIT_ISPOINTER) {
                  int s = 1;
                  for (int dim = 0; dim < m.ArrayDim(); dim++)
                     s *= m.MaxIndex(dim);
                  if (!decli) {
                     (*dictSrcOut) << "      int R__i;" << std::endl;
                     decli = 1;
                  }
                  (*dictSrcOut) << "      for (R__i = 0; R__i < " << s << "; R__i++)" << std::endl;
                  if (i == 0) {
                     Error(0, "*** Datamember %s::%s: array of pointers to fundamental type (need manual intervention)\n", cl.Fullname(), m.Name());
                     (*dictSrcOut) << "         ;//R__b.ReadArray(" << m.Name() << ");" << std::endl;
                  } else {
                     (*dictSrcOut) << "         ;//R__b.WriteArray(" << m.Name() << ", __COUNTER__);" << std::endl;
                  }
               } else if (m.Property() & G__BIT_ISPOINTER) {
                  const char *indexvar = GrabIndex(m, i==0);
                  if (indexvar==0) {
                     if (i == 0) {
                        Error(0,"*** Datamember %s::%s: pointer to fundamental type (need manual intervention)\n", cl.Fullname(), m.Name());
                        (*dictSrcOut) << "      //R__b.ReadArray(" << m.Name() << ");" << std::endl;
                     } else {
                        (*dictSrcOut) << "      //R__b.WriteArray(" << m.Name() << ", __COUNTER__);" << std::endl;
                     }
                  } else {
                     if (i == 0) {
                        (*dictSrcOut) << "      delete [] " << m.Name() << ";" << std::endl
                            << "      " << GetNonConstMemberName(m) << " = new "
                            << ShortTypeName(m.Type()->Name()) << "[" << indexvar << "];" << std::endl;
                        if (isDouble32) {
                           (*dictSrcOut) << "      R__b.ReadFastArrayDouble32(" <<  GetNonConstMemberName(m)
                               << "," << indexvar << ");" << std::endl;
                        } else {
                           (*dictSrcOut) << "      R__b.ReadFastArray(" << GetNonConstMemberName(m)
                               << "," << indexvar << ");" << std::endl;
                        }
                     } else {
                        if (isDouble32) {
                           (*dictSrcOut) << "      R__b.WriteFastArrayDouble32("
                               << m.Name() << "," << indexvar << ");" << std::endl;
                        } else {
                           (*dictSrcOut) << "      R__b.WriteFastArray("
                              << m.Name() << "," << indexvar << ");" << std::endl;
                        }
                     }
                  }
               } else if (m.Property() & G__BIT_ISARRAY) {
                  if (i == 0) {
                     if (m.ArrayDim() > 1) {
                        if ((m.Type())->Property() & G__BIT_ISENUM)
                           (*dictSrcOut) << "      R__b.ReadStaticArray((Int_t*)" << m.Name() << ");" << std::endl;
                        else
                           if (isDouble32) {
                              (*dictSrcOut) << "      R__b.ReadStaticArrayDouble32((" << m.Type()->TrueName()
                                  << "*)" << m.Name() << ");" << std::endl;
                           } else {
                              (*dictSrcOut) << "      R__b.ReadStaticArray((" << m.Type()->TrueName()
                                  << "*)" << m.Name() << ");" << std::endl;
                           }
                     } else {
                        if ((m.Type())->Property() & G__BIT_ISENUM)
                           (*dictSrcOut) << "      R__b.ReadStaticArray((Int_t*)" << m.Name() << ");" << std::endl;
                        else
                           if (isDouble32) {
                              (*dictSrcOut) << "      R__b.ReadStaticArrayDouble32(" << m.Name() << ");" << std::endl;
                           } else {
                              (*dictSrcOut) << "      R__b.ReadStaticArray((" << m.Type()->TrueName()
                                  << "*)" << m.Name() << ");" << std::endl;
                           }
                     }
                  } else {
                     int s = 1;
                     for (int dim = 0; dim < m.ArrayDim(); dim++)
                        s *= m.MaxIndex(dim);
                     if (m.ArrayDim() > 1) {
                        if ((m.Type())->Property() & G__BIT_ISENUM)
                           (*dictSrcOut) << "      R__b.WriteArray((Int_t*)" << m.Name() << ", "
                               << s << ");" << std::endl;
                        else
                           if (isDouble32) {
                              (*dictSrcOut) << "      R__b.WriteArrayDouble32((" << m.Type()->TrueName()
                                  << "*)" << m.Name() << ", " << s << ");" << std::endl;
                           } else {
                              (*dictSrcOut) << "      R__b.WriteArray((" << m.Type()->TrueName()
                                  << "*)" << m.Name() << ", " << s << ");" << std::endl;
                           }
                     } else {
                        if ((m.Type())->Property() & G__BIT_ISENUM)
                           (*dictSrcOut) << "      R__b.WriteArray((Int_t*)" << m.Name() << ", " << s << ");" << std::endl;
                        else
                           if (isDouble32) {
                              (*dictSrcOut) << "      R__b.WriteArrayDouble32(" << m.Name() << ", " << s << ");" << std::endl;
                           } else {
                              (*dictSrcOut) << "      R__b.WriteArray(" << m.Name() << ", " << s << ");" << std::endl;
                           }
                     }
                  }
               } else if ((m.Type())->Property() & G__BIT_ISENUM) {
                  if (i == 0)
                     (*dictSrcOut) << "      R__b >> (Int_t&)" << m.Name() << ";" << std::endl;
                  else
                     (*dictSrcOut) << "      R__b << (Int_t)" << m.Name() << ";" << std::endl;
               } else {
                  if (isDouble32) {
                     if (i == 0)
                        (*dictSrcOut) << "      {float R_Dummy; R__b >> R_Dummy; " << GetNonConstMemberName(m)
                            << "=Double32_t(R_Dummy);}" << std::endl;
                     else
                        (*dictSrcOut) << "      R__b << float(" << GetNonConstMemberName(m) << ");" << std::endl;
                  } else {
                     if (i == 0)
                        (*dictSrcOut) << "      R__b >> " << GetNonConstMemberName(m) << ";" << std::endl;
                     else
                        (*dictSrcOut) << "      R__b << " << GetNonConstMemberName(m) << ";" << std::endl;
                  }
               }
            } else {
               // we have an object...

               // check if object is a standard string
               if (STLStringStreamer(m, i))
                  continue;

               // check if object is an STL container
               if (STLContainerStreamer(m, i))
                  continue;

               // handle any other type of objects
               if (m.Property() & G__BIT_ISARRAY &&
                   m.Property() & G__BIT_ISPOINTER) {
                  int s = 1;
                  for (int dim = 0; dim < m.ArrayDim(); dim++)
                     s *= m.MaxIndex(dim);
                  if (!decli) {
                     (*dictSrcOut) << "      int R__i;" << std::endl;
                     decli = 1;
                  }
                  (*dictSrcOut) << "      for (R__i = 0; R__i < " << s << "; R__i++)" << std::endl;
                  if (i == 0)
                     (*dictSrcOut) << "         R__b >> " << GetNonConstMemberName(m);
                  else {
                     if (m.Type()->IsBase("TObject") && m.Type()->IsBase("TArray"))
                        (*dictSrcOut) << "         R__b << (TObject*)" << m.Name();
                     else
                        (*dictSrcOut) << "         R__b << " << GetNonConstMemberName(m);
                  }
                  WriteArrayDimensions(m.ArrayDim());
                  (*dictSrcOut) << "[R__i];" << std::endl;
               } else if (m.Property() & G__BIT_ISPOINTER) {
                  // This is always good. However, in case of a pointer
                  // to an object that is guarenteed to be there and not
                  // being referenced by other objects we could use
                  //     xx->Streamer(b);
                  // Optimize this with control statement in title.
                  if (PointerToPointer(m)) {
                     if (i == 0) {
                        Error(0, "*** Datamember %s::%s: pointer to pointer (need manual intervention)\n", cl.Fullname(), m.Name());
                        (*dictSrcOut) << "      //R__b.ReadArray(" << m.Name() << ");" << std::endl;
                     } else {
                        (*dictSrcOut) << "      //R__b.WriteArray(" << m.Name() << ", __COUNTER__);";
                     }
                  } else {
                     if (strstr(m.Type()->Name(), "TClonesArray")) {
                        (*dictSrcOut) << "      " << m.Name() << "->Streamer(R__b);" << std::endl;
                     } else {
                        if (i == 0) {
                           // The following:
                           //    if (strncmp(m.Title(),"->",2) != 0) fprintf(fp, "      delete %s;\n", GetNonConstMemberName(m).c_str());
                           // could be used to prevent a memory leak since the next statement could possibly create a new object.
                           // In the TStreamerInfo based I/O we made the previous statement conditional on TStreamerInfo::CanDelete
                           // to allow the user to prevent some inadvisable deletions.  So we should be offering this flexibility
                           // here to and should not (technically) rely on TStreamerInfo for it, so for now we leave it as is.
                           // Note that the leak should happen from here only if the object is stored in an unsplit object
                           // and either the user request an old branch or the streamer has been customized.
                           (*dictSrcOut) << "      R__b >> " << GetNonConstMemberName(m) << ";" << std::endl;
                        } else {
                           if (m.Type()->IsBase("TObject") && m.Type()->IsBase("TArray"))
                              (*dictSrcOut) << "      R__b << (TObject*)" << m.Name() << ";" << std::endl;
                           else
                              (*dictSrcOut) << "      R__b << " << GetNonConstMemberName(m) << ";" << std::endl;
                        }
                     }
                  }
               } else if (m.Property() & G__BIT_ISARRAY) {
                  int s = 1;
                  for (int dim = 0; dim < m.ArrayDim(); dim++)
                     s *= m.MaxIndex(dim);
                  if (!decli) {
                     (*dictSrcOut) << "      int R__i;" << std::endl;
                     decli = 1;
                  }
                  (*dictSrcOut) << "      for (R__i = 0; R__i < " << s << "; R__i++)" << std::endl;
                  const char *mTypeName = m.Type()->Name();
                  const char *constwd = "const ";
                  if (strncmp(constwd,mTypeName,strlen(constwd))==0) {
                     mTypeName += strlen(constwd);
                     (*dictSrcOut) << "         const_cast< " << mTypeName << " &>(" << m.Name();
                     WriteArrayDimensions(m.ArrayDim());
                     (*dictSrcOut) << "[R__i]).Streamer(R__b);" << std::endl;
                  } else {
                     (*dictSrcOut) << "         " << GetNonConstMemberName(m);
                     WriteArrayDimensions(m.ArrayDim());
                     (*dictSrcOut) << "[R__i].Streamer(R__b);" << std::endl;
                  }
               } else {
                  if ((m.Type())->HasMethod("Streamer"))
                     (*dictSrcOut) << "      " << GetNonConstMemberName(m) << ".Streamer(R__b);" << std::endl;
                  else {
                     (*dictSrcOut) << "      R__b.StreamObject(&(" << m.Name() << "),typeid("
                         << m.Type()->Name() << "));" << std::endl;               //R__t.Streamer(R__b);\n");
//VP                     if (i == 0)
//VP                        Error(0, "*** Datamember %s::%s: object has no Streamer() method (need manual intervention)\n",
//VP                                  cl.Fullname(), m.Name());
//VP                     fprintf(fp, "      //%s.Streamer(R__b);\n", m.Name());
                  }
               }
            }
         }
      }
   }
   if (ubc) (*dictSrcOut) << "      R__b.SetByteCount(R__c, kTRUE);" << std::endl;
   (*dictSrcOut) << "   }" << std::endl
       << "}" << std::endl << std::endl;

   while (enclSpaceNesting) {
     (*dictSrcOut) << "} // namespace " << nsname.c_str() << std::endl;
     --enclSpaceNesting;
   }
}

//______________________________________________________________________________
void WriteAutoStreamer(G__ClassInfo &cl)
{

   // Write Streamer() method suitable for automatic schema evolution.

   int add_template_keyword = NeedTemplateKeyword(cl);

   G__BaseClassInfo base(cl);
   while (base.Next()) {
      if (IsSTLContainer(base)) {
         RStl::inst().GenerateTClassFor( base.Name() );
      }
   }

   G__ClassInfo ns = cl.EnclosingSpace();
   string clsname = cl.Fullname();
   string nsname;
   if (ns.IsValid()) {
     nsname = ns.Fullname();
     clsname.erase (0, nsname.size() + 2);
   }

   int enclSpaceNesting = 0;
   if (!nsname.empty()) {
     G__ShadowMaker nestTempShadowMaker(*dictSrcOut, "");
     enclSpaceNesting = nestTempShadowMaker.WriteNamespaceHeader(cl);
   }

   (*dictSrcOut) << "//_______________________________________"
       << "_______________________________________" << std::endl;
   if (add_template_keyword) (*dictSrcOut) << "template <> ";
   (*dictSrcOut) << "void " << clsname << "::Streamer(TBuffer &R__b)" << std::endl
       << "{" << std::endl
       << "   // Stream an object of class " << cl.Fullname() << "." << std::endl << std::endl
       << "   if (R__b.IsReading()) {" << std::endl
       << "      " << cl.Fullname() << "::Class()->ReadBuffer(R__b, this);" << std::endl
       << "   } else {" << std::endl
       << "      " << cl.Fullname() << "::Class()->WriteBuffer(R__b, this);" << std::endl
       << "   }" << std::endl
       << "}" << std::endl << std::endl;

   while (enclSpaceNesting) {
     (*dictSrcOut) << "} // namespace " << nsname << std::endl;
     --enclSpaceNesting;
   }
}

//______________________________________________________________________________
void WriteStreamerBases(G__ClassInfo &cl)
{
   // Write Streamer() method for base classes of cl (unused)

   (*dictSrcOut) << "//_______________________________________"
       << "_______________________________________" << std::endl
       << "void " << cl.Fullname() << "_StreamerBases(TBuffer &R__b, void *pointer)" << std::endl
       << "{" << std::endl
       << "   // Stream base classes of class " << cl.Fullname() << "." << std::endl << std::endl
       << "   " << cl.Fullname() << " *obj = (" << cl.Fullname() << "*)pointer;" << std::endl
       << "   if (R__b.IsReading()) {" << std::endl;
   G__BaseClassInfo br(cl);
   while (br.Next())
      if (br.HasMethod("Streamer")) {
         (*dictSrcOut) << "      obj->" << br.Name() << "::Streamer(R__b);" << std::endl;
      }
   (*dictSrcOut) << "   } else {" << std::endl;
   G__BaseClassInfo bw(cl);
   while (bw.Next())
      if (bw.HasMethod("Streamer")) {
         (*dictSrcOut) << "      obj->" << bw.Name() << "::Streamer(R__b);" << std::endl;
      }
   (*dictSrcOut) << "   }" << std::endl
       << "}" << std::endl << std::endl;
}

//______________________________________________________________________________
void WritePointersSTL(G__ClassInfo &cl)
{
   // Write interface function for STL members

   // char a[80],fun[80];
   char a[G__LONGLINE];
   char clName[G__LONGLINE];
   strcpy(clName, G__map_cpp_name((char *)cl.Fullname()));
   int version = GetClassVersion( cl);
   if (version == 0) return;
   if (version < 0 && !(cl.RootFlag() & G__USEBYTECOUNT) ) return;


   // We also need to look at the base classes.
   G__BaseClassInfo base(cl);
   while (base.Next()) {
      int k = IsSTLContainer(base);
      if (k!=0) {
         RStl::inst().GenerateTClassFor( base.Name() );
      }
   }

   // Look at the data members
   G__DataMemberInfo m(cl);
   while (m.Next()) {

      if ((m.Property() & G__BIT_ISSTATIC)) continue;
      int pCounter = 0;
      if (m.Property() & G__BIT_ISPOINTER) {
         const char *leftb = strchr(m.Title(),'[');
         if (leftb) {
            const char *rightb = strchr(leftb,']');
            if (rightb) {
               pCounter++;
               strcpy(a,m.Type()->Name());
               char *astar = (char*)strchr(a,'*');
               *astar = 0;
               if (strstr(m.Type()->Name(),"**")) pCounter++;
            }
         }
      }


      //member is a string
      {
         const char*shortTypeName = ShortTypeName(m.Type()->Name());
         if (!strcmp(shortTypeName, "string")) {
            continue;
         }
      }

      if (!IsStreamable(m)) continue;

      int k = IsSTLContainer(m);
      if (k!=0) {
//          fprintf(stderr,"Add %s which is also",m.Type()->Name());
//          fprintf(stderr," %s\n",m.Type()->TrueName() );
         RStl::inst().GenerateTClassFor( m.Type()->Name() );
      }
      if (k<0) continue;
      else if (k>0) continue; // do not generate the member streamer for STL containers anymore.

      // Check whether we need a streamer function.
      // For now we use it only for variable size array of objects (well maybe ... it is not really tested!)
      if (!pCounter) continue;

      // We no longer need a custom streamer for variable size arrays of objects.
      continue;

      {
         string fun ( string("R__")+ clName +"_" + m.Name() );
         // sprintf(fun,"R__%s_%s",clName,m.Name());
         SetFun(fun);
      }

      (*dictSrcOut) << "//_______________________________________"
          << "_______________________________________" << std::endl
          << "void R__" << clName << "_" << m.Name() << "(TBuffer &R__b, void *R__p, int";
      if (pCounter) {
         (*dictSrcOut) << " R__n";
      }
      (*dictSrcOut) << ")" << std::endl
          << "{" << std::endl;
      // remove all 'const' keyword.
      string mTypeName = G__ShadowMaker::GetNonConstTypeName(m);
      // Define a variable for easy access to the data member.
      if (m.Property() & G__BIT_ISARRAY) {
         (*dictSrcOut) << "   " << mTypeName << "* " << m.Name() << " = (" << mTypeName << "*)R__p;" << std::endl;
      } else {
         if (m.Property() & G__BIT_ISPOINTER) {
            (*dictSrcOut) << "   " << mTypeName << "* " << m.Name() << " = (" << mTypeName << "*)R__p;" << std::endl;
         } else {
            (*dictSrcOut) << "   " << mTypeName << " &" << m.Name() << " = *(" << mTypeName << " *)R__p;" << std::endl;
         }
      }
      (*dictSrcOut) << "   if (R__b.IsReading()) {" << std::endl;
      if (m.Type()->IsTmplt() && IsSTLContainer(m)) {
         STLContainerStreamer(m, 0);
      } else {
         if (m.Property() & G__BIT_ISARRAY) {
            int len = 1;
            for (int dim = 0; dim < m.ArrayDim(); dim++) len *= m.MaxIndex(dim);
            (*dictSrcOut) << "      for (Int_t R__l = 0; R__l < " << len << "; R__l++) {" << std::endl;
            if (m.Property() & G__BIT_ISPOINTER) {
               (*dictSrcOut) << "         R__b >> " << m.Name() << "[R__l];" << std::endl;
            } else {
               (*dictSrcOut) << "         " << m.Name() << "[R__l].Streamer(R__b);" << std::endl;
            }
            (*dictSrcOut) << "      }" << std::endl;
         } else {
            if (m.Property() & G__BIT_ISPOINTER) {
               if (pCounter == 2) {
                  (*dictSrcOut) << "      delete [] *" << m.Name() << ";" << std::endl
                      << "      if (!R__n) return;" << std::endl
                      << "      *" << m.Name() << " = new " << a << "*[R__n];" << std::endl
                      << "      " << a << "** R__s = *" << m.Name() << ";" << std::endl
                      << "      for (Int_t R__l = 0; R__l < R__n; R__l++) {" << std::endl
                      << "         R__s[R__l] = new " << a << "();" << std::endl
                      << "         R__s[R__l]->Streamer(R__b);" << std::endl
                      << "      }" << std::endl;
               } else if (pCounter == 1) {
                  (*dictSrcOut) << "      delete [] *" << m.Name() << ";" << std::endl
                      << "      if (!R__n) return;"  << std::endl
                      << "      *" << m.Name() << " = new " << a << "[R__n];" << std::endl
                      << "      " << a << "* R__s = *" << m.Name() << ";" << std::endl
                      << "      for (Int_t R__l = 0; R__l < R__n; R__l++) {" << std::endl
                      << "         R__s[R__l].Streamer(R__b);" << std::endl
                      << "      }" << std::endl;
               } else {
                  if (strncmp(m.Title(),"->",2) == 0)
                      (*dictSrcOut) << "      (*" << m.Name() << ")->Streamer(R__b);" << std::endl;
                  else
                      (*dictSrcOut) << "      R__b >> *" << m.Name() << ";" << std::endl;
               }
            } else {
               (*dictSrcOut) << "      " << m.Name() << ".Streamer(R__b);" << std::endl;
            }
         }
      }
      (*dictSrcOut) << "   } else {" << std::endl;
      if (m.Type()->IsTmplt() && IsSTLContainer(m)) {
         STLContainerStreamer(m, 1);
      } else {
         if (m.Property() & G__BIT_ISARRAY) {
            int len = 1;
            for (int dim = 0; dim < m.ArrayDim(); dim++) len *= m.MaxIndex(dim);
            (*dictSrcOut) << "      for (Int_t R__l = 0; R__l < " << len << "; R__l++) {" << std::endl;
            if (m.Property() & G__BIT_ISPOINTER) {
               if (m.Type()->IsBase("TObject"))
                  (*dictSrcOut) << "         R__b << (TObject*)" << m.Name() << "[R__l];" << std::endl;
               else
                  (*dictSrcOut) << "         R__b << " << m.Name() << "[R__l];" << std::endl;
            } else {
                (*dictSrcOut) << "         " << m.Name() << "[R__l].Streamer(R__b);" << std::endl;
            }
            (*dictSrcOut) << "      }" << std::endl;
         } else {
            if (m.Property() & G__BIT_ISPOINTER) {
               if (pCounter == 2) {
                  (*dictSrcOut) << "      " << a << "** R__s = *" << m.Name() << ";" << std::endl
                      << "      for (Int_t R__l = 0; R__l < R__n; R__l++) {" << std::endl
                      << "         R__s[R__l]->Streamer(R__b);" << std::endl
                      << "      }" << std::endl;
               } else if(pCounter == 1) {
                  (*dictSrcOut) << "      " << a << "* R__s = *" << m.Name() << ";" << std::endl
                      << "      for (Int_t R__l = 0; R__l < R__n; R__l++) {" << std::endl
                      << "         R__s[R__l].Streamer(R__b);" << std::endl
                      << "      }" << std::endl;
               } else {
                  if (strncmp(m.Title(),"->",2) == 0)
                     (*dictSrcOut) << "      (*" << m.Name() << ")->Streamer(R__b);" << std::endl;
                  else {
                     if (m.Type()->IsBase("TObject"))
                        (*dictSrcOut) << "      R__b << (TObject*)*" << m.Name() << ";" << std::endl;
                     else
                        (*dictSrcOut) << "      R__b << *" << m.Name() << ";" << std::endl;
                  }
               }
            } else {
               (*dictSrcOut) << "      " << m.Name() << ".Streamer(R__b);" << std::endl;
            }
         }
      }
      (*dictSrcOut) << "   }" << std::endl
          << "}" << std::endl << std::endl;
   }
}


//______________________________________________________________________________
void WriteBodyShowMembers(G__ClassInfo& cl, bool outside)
{
   string csymbol = cl.Fullname();
   if ( ! TClassEdit::IsStdClass( csymbol.c_str() ) ) {

      // Prefix the full class name with '::' except for the STL
      // containers and std::string.  This is to request the
      // real class instead of the class in the namespace ROOT::Shadow
      csymbol.insert(0,"::");
   }

   const char *prefix = "";

   (*dictSrcOut) << "      // Inspect the data members of an object of class " << cl.Fullname() << "." << std::endl;

   std::string clfullname;
   shadowMaker->GetFullShadowName(cl, clfullname);
   if (outside) {
      (*dictSrcOut) << "      typedef " << clfullname.c_str() << " ShadowClass;" << std::endl
          << "      ShadowClass *sobj = (ShadowClass*)obj;" << std::endl
          << "      if (sobj) { } // Dummy usage just in case there is no datamember." << std::endl << std::endl;
      prefix = "sobj->";
   }

   if (cl.HasMethod("IsA") && !outside) {
#ifdef  WIN32
      // This is to work around a bad msvc C++ bug.
      // This code would work in the general case, but why bother....and
      // we want to remember to eventually remove it ...

      if (strstr(csymbol.c_str(),"::")) {
         // there is a namespace involved, trigger MS VC bug workaround
         (*dictSrcOut) << "      typedef " << csymbol.c_str() << " msvc_bug_workaround;" << std::endl
             << "      TClass *R__cl = msvc_bug_workaround::IsA();" << std::endl;
      } else
         (*dictSrcOut) << "      TClass *R__cl = " << csymbol.c_str() << "::IsA();" << std::endl;
#else
      (*dictSrcOut) << "      TClass *R__cl = " << csymbol.c_str() << "::IsA();" << std::endl;
#endif
   } else {
      (*dictSrcOut) << "      TClass *R__cl  = ::ROOT::GenerateInitInstanceLocal((const " << csymbol.c_str() << "*)0x0)->GetClass();" << std::endl;
   }
   (*dictSrcOut) << "      Int_t R__ncp = strlen(R__parent);" << std::endl
       << "      if (R__ncp || R__cl || R__insp.IsA()) { }" << std::endl;

   // Inspect data members
   G__DataMemberInfo m(cl);
   char cdim[24], cvar[128];
   char clName[G__LONGLINE];
   string fun;
   strcpy(clName,G__map_cpp_name((char *)cl.Fullname()));
   int version = GetClassVersion(cl);
   int clflag = 1;
   if (version == 0 || cl.RootFlag() == 0) clflag = 0;
   if (version < 0 && !(cl.RootFlag() & G__USEBYTECOUNT) ) clflag = 0;

   while (m.Next()) {

      // we skip:
      //  - static members
      //  - the member G__virtualinfo inserted by the CINT RTTI system

      fun = string("R__") + clName + "_" + m.Name(); // sprintf(fun,"R__%s_%s",clName,m.Name());
      if (!(m.Property() & G__BIT_ISSTATIC) &&
          strcmp(m.Name(), "G__virtualinfo")) {

         // fundamental type: short, int, long, etc....
         if (((m.Type())->Property() & G__BIT_ISFUNDAMENTAL) ||
             ((m.Type())->Property() & G__BIT_ISENUM)) {
            if (m.Property() & G__BIT_ISARRAY &&
                m.Property() & G__BIT_ISPOINTER) {
               cvar[0] = '*';
               strcpy(cvar+1, m.Name());
               for (int dim = 0; dim < m.ArrayDim(); dim++) {
                  sprintf(cdim, "[%d]", m.MaxIndex(dim));
                  strcat(cvar, cdim);
               }
               (*dictSrcOut) << "      R__insp.Inspect(R__cl, R__parent, \"" << cvar << "\", &"
                   << prefix << m.Name() << ");" << std::endl;
            } else if (m.Property() & G__BIT_ISPOINTER) {
               (*dictSrcOut) << "      R__insp.Inspect(R__cl, R__parent, \"*" << m.Name() << "\", &"
                   << prefix << m.Name() << ");" << std::endl;
            } else if (m.Property() & G__BIT_ISARRAY) {
               strcpy(cvar, m.Name());
               bool vardim = false;
               for (int dim = 0; dim < m.ArrayDim(); dim++) {
                  int maxInd = m.MaxIndex(dim);
                  if (maxInd < 0) {
                     strcpy(cdim,"[]");
                     vardim = true;
                  } else {
                     sprintf(cdim, "[%d]", maxInd);
                  }
                  strcat(cvar, cdim);
               }
               if (vardim) {
                  (*dictSrcOut) << "      R__insp.Inspect(R__cl, R__parent, \"" << cvar << "\", &"
                      << prefix << m.Name() << ");" << std::endl;
               } else {
                  (*dictSrcOut) << "      R__insp.Inspect(R__cl, R__parent, \"" << cvar << "\", "
                      << prefix << m.Name() << ");" << std::endl;
               }

            } else {
               (*dictSrcOut) << "      R__insp.Inspect(R__cl, R__parent, \"" << m.Name() << "\", &"
                   << prefix << m.Name() << ");" << std::endl;
            }
         } else {
            // we have an object

            //string
            if (!strcmp(m.Type()->Name(), "string") || !strcmp(m.Type()->Name(), "string*")) {
               if (m.Property() & G__BIT_ISPOINTER) {
                  (*dictSrcOut) << "      R__insp.Inspect(R__cl, R__parent, \"*" << m.Name() << "\", &"
                      << prefix <<  m.Name()<< ");" << std::endl;
                  if (clflag && IsStreamable(m) && GetFun(fun))
                     (*dictSrcOut) << "   R__cl->SetMemberStreamer(\"*" << m.Name() << "\",R__"
                         << clName << "_" << m.Name() << ");" << std::endl;
               } else {
                  (*dictSrcOut) << "      R__insp.Inspect(R__cl, R__parent, \"" << m.Name() << "\", &"
                      << prefix << m.Name() << ");" << std::endl;
                  if (clflag && IsStreamable(m) && GetFun(fun))
                     (*dictSrcOut) << "      R__cl->SetMemberStreamer(\"" << m.Name() << "\",R__"
                         << clName << "_" << m.Name() << ");" << std::endl;
               }
               continue;
            }

            if (m.Property() & G__BIT_ISARRAY &&
                m.Property() & G__BIT_ISPOINTER) {
               cvar[0] = '*';
               strcpy(cvar+1, m.Name());
               for (int dim = 0; dim < m.ArrayDim(); dim++) {
                  sprintf(cdim, "[%d]", m.MaxIndex(dim));
                  strcat(cvar, cdim);
               }
               (*dictSrcOut) << "      R__insp.Inspect(R__cl, R__parent, \"" << cvar << "\", &"
                   << prefix << m.Name() << ");" << std::endl;
               if (clflag && IsStreamable(m) && GetFun(fun))
                  (*dictSrcOut) << "      R__cl->SetMemberStreamer(\"" << cvar << "\",R__"
                      << clName << "_" << m.Name() << ");" << std::endl;
            } else if (m.Property() & G__BIT_ISPOINTER) {
               (*dictSrcOut) << "      R__insp.Inspect(R__cl, R__parent, \"*" << m.Name() << "\", &" << prefix << m.Name() << ");" << std::endl;
               if (clflag && IsStreamable(m) && GetFun(fun))
                  (*dictSrcOut) << "      R__cl->SetMemberStreamer(\"*" << m.Name() << "\",R__"
                      << clName << "_" << m.Name() << ");" << std::endl;
            } else if (m.Property() & G__BIT_ISARRAY) {
               strcpy(cvar, m.Name());
               for (int dim = 0; dim < m.ArrayDim(); dim++) {
                  sprintf(cdim, "[%d]", m.MaxIndex(dim));
                  strcat(cvar, cdim);
               }
               (*dictSrcOut) << "      R__insp.Inspect(R__cl, R__parent, \"" << cvar << "\", "
                   << prefix << m.Name() << ");" << std::endl;
               if (clflag && IsStreamable(m) && GetFun(fun))
                  (*dictSrcOut) << "      R__cl->SetMemberStreamer(\"" << cvar << "\",R__"
                      << clName << "_" << m.Name() << ");"  << std::endl;
            } else if (m.Property() & G__BIT_ISREFERENCE) {
               // For reference we do not know what do not ... let's do nothing (hopefully the referenced objects is saved somewhere else!

            } else {
               if ((m.Type())->HasMethod("ShowMembers")) {
                  (*dictSrcOut) << "      R__insp.Inspect(R__cl, R__parent, \"" << m.Name() << "\", &"
                      << prefix << m.Name() << ");" << std::endl;
                  (*dictSrcOut) << "      " << GetNonConstMemberName(m,prefix)
                      << ".ShowMembers(R__insp, strcat(R__parent,\""
                      << m.Name() << ".\")); R__parent[R__ncp] = 0;"  << std::endl;
                  if (clflag && IsStreamable(m) && GetFun(fun))
                     //fprintf(fp, "      R__cl->SetMemberStreamer(strcat(R__parent,\"%s\"),R__%s_%s); R__parent[R__ncp] = 0;\n", m.Name(), clName, m.Name());
                     (*dictSrcOut) << "      R__cl->SetMemberStreamer(\"" << m.Name() << "\",R__"
                         << clName << "_" << m.Name() << ");" << std::endl;
               } else {
                  // NOTE: something to be added here!
                  (*dictSrcOut) << "      R__insp.Inspect(R__cl, R__parent, \"" << m.Name()
                      << "\", (void*)&" << prefix << m.Name() << ");" << std::endl;
                  /* if (can call ShowStreamer) */

                  char compareName[G__LONGLINE];
                  strcpy(compareName,clName);
                  strcat(compareName,"::");

                  if (strlen(m.Type()->Name()) &&
                      strcmp(compareName,m.Type()->Name())!=0 ) {
                     // Filter out the unamed type from with a the class.

                     string typeWithDefaultStlName( RStl::DropDefaultArg(m.Type()->Name()) );
                     //TClassEdit::ShortType(m.Type()->Name(),TClassEdit::kRemoveDefaultAlloc) );
                     string typeName( GetLong64_Name( m.Type()->Name() ) );

                     (*dictSrcOut) << "      ::ROOT::GenericShowMembers(\"" << typeName << "\", (void*)&"
                         << prefix << m.Name() << ", R__insp, strcat(R__parent,\""
                         << m.Name() << ".\")," << (!strncmp(m.Title(), "!", 1)?"true":"false")
                         <<  ");" << std::endl
                         << "      R__parent[R__ncp] = 0;" << std::endl;
                  }
                  if (clflag && IsStreamable(m) && GetFun(fun))
                     (*dictSrcOut) << "      R__cl->SetMemberStreamer(\"" << m.Name() << "\",R__"
                         << clName << "_" << m.Name() << ");" << std::endl;
               }
            }
         }
      }
   }

   // Write ShowMembers for base class(es) when they have the ShowMember() method
   G__BaseClassInfo b(cl);

   int base = 0;
   while (b.Next()) {
      base++;
      if (b.HasMethod("ShowMembers")) {
         if (outside) {
            (*dictSrcOut) << "      sobj->" << b.Fullname() << "::ShowMembers(R__insp, R__parent);" << std::endl;
         } else {
            if (strstr(b.Fullname(),"::")) {
               // there is a namespace involved, trigger MS VC bug workaround
               (*dictSrcOut) << "      //This works around a msvc bug and should be harmless on other platforms" << std::endl
                   << "      typedef " << b.Fullname() << " baseClass" << base << ";" << std::endl
                   << "      baseClass" << base << "::ShowMembers(R__insp, R__parent);" << std::endl;
            } else {
               (*dictSrcOut) << "      " << b.Fullname() << "::ShowMembers(R__insp, R__parent);" << std::endl;
            }
         }
      } else {
         string baseclass = FixSTLName(b.Fullname());
         // We used to use a dynamic_cast for cast to the parent class.  This
         // was not necessary and actually crippling.  In this situations
         // (casting from child to parent) the C-style cast is returning the
         // same result as the dynamic_cast but has the advantage (for us) of
         // being able to apply the case even if the parent is inherited from
         // privately.

         //string baseclassWithDefaultStlName( m.Type()->Name()); //  RStl::DropDefaultArg(m.Type()->Name()) );
         //string baseclassWithDefaultStlName( TClassEdit::ShortType(baseclass.c_str(),
         //                                                          TClassEdit::kRemoveDefaultAlloc) );
         if (outside) {
            (*dictSrcOut) << "      ::ROOT::GenericShowMembers(\"" << baseclass.c_str() << "\", ( ::" << baseclass.c_str()
                << " * )( (::" << cl.Fullname() << "*) obj ), R__insp, R__parent, false);" << std::endl;
         } else {
            (*dictSrcOut) << "      ::ROOT::GenericShowMembers(\"" << baseclass.c_str() << "\", ( ::" << baseclass.c_str()
                << " *) (this ), R__insp, R__parent, false);" << std::endl;
         }
      }
   }

}

//______________________________________________________________________________
void WriteShowMembers(G__ClassInfo &cl, bool outside = false)
{
   (*dictSrcOut) << "//_______________________________________";
   (*dictSrcOut) << "_______________________________________" << std::endl;

   string classname = GetLong64_Name( RStl::DropDefaultArg( cl.Fullname() ) );
   string mappedname = G__map_cpp_name((char*)classname.c_str());

   if (outside || cl.IsTmplt()) {
      (*dictSrcOut) << "namespace ROOT {" << std::endl

          << "   void " << mappedname.c_str() << "_ShowMembers(void *obj, TMemberInspector &R__insp, char *R__parent)"
          << std::endl << "   {" << std::endl;
      WriteBodyShowMembers(cl, outside || cl.IsTmplt());
      (*dictSrcOut) << "   }" << std::endl << std::endl;
      (*dictSrcOut) << "}" << std::endl << std::endl;
   }

   if (!outside) {
      G__ClassInfo ns = cl.EnclosingSpace();
      string clsname = cl.Fullname();
      string nsname;
      if (ns.IsValid()) {
        nsname = ns.Fullname();
        clsname.erase (0, nsname.size() + 2);
      }
      int add_template_keyword = NeedTemplateKeyword(cl);
      int enclSpaceNesting = 0;
      if (!nsname.empty()) {
         G__ShadowMaker nestTempShadowMaker(*dictSrcOut, "");
         enclSpaceNesting = nestTempShadowMaker.WriteNamespaceHeader(cl);
      }
      if (add_template_keyword) (*dictSrcOut) << "template <> ";
      (*dictSrcOut) << "void " << clsname << "::ShowMembers(TMemberInspector &R__insp, char *R__parent)"
          << std::endl << "{" << std::endl;
      if (!cl.IsTmplt()) {
         WriteBodyShowMembers(cl, outside);
      } else {
         string classname = GetLong64_Name( RStl::DropDefaultArg( cl.Fullname() ) );
         string mappedname = G__map_cpp_name((char*)classname.c_str());

         (*dictSrcOut) <<  "   ::ROOT::" << mappedname.c_str() << "_ShowMembers(this, R__insp, R__parent);" << std::endl;
      }
      (*dictSrcOut) << "}" << std::endl << std::endl;

      while (enclSpaceNesting) {
         (*dictSrcOut) << "} // namespace " << nsname << std::endl;
         --enclSpaceNesting;
      }
  }
}

//______________________________________________________________________________
void WriteClassCode(G__ClassInfo &cl, bool force = false)
{
   if ((cl.Property() & (G__BIT_ISCLASS|G__BIT_ISSTRUCT)) && (force || cl.Linkage() == G__CPPLINK) ) {

      if ( TClassEdit::IsSTLCont(cl.Name()) ) {
         RStl::inst().GenerateTClassFor( cl.Name() );
         return;
      }

      if (cl.HasMethod("Streamer")) {
         //WriteStreamerBases(cl);
         if (cl.RootFlag()) WritePointersSTL(cl);
         if (!(cl.RootFlag() & G__NOSTREAMER)) {
            if ((cl.RootFlag() & G__USEBYTECOUNT /*G__AUTOSTREAMER*/)) {
               WriteAutoStreamer(cl);
            } else {
              WriteStreamer(cl);
            }
         } else
            Info(0, "Class %s: Do not generate Streamer() [*** custom streamer ***]\n", cl.Fullname());
      } else {
         Info(0, "Class %s: Streamer() not declared\n", cl.Fullname());

         if (cl.RootFlag() & G__USEBYTECOUNT) WritePointersSTL(cl);
      }
      if (cl.HasMethod("ShowMembers")) {
         WriteShowMembers(cl);
         WriteAuxFunctions(cl);
      } else {
         if (NeedShadowClass(cl)) {
           WriteShowMembers(cl, true);
         }
         WriteAuxFunctions(cl);
      }
   }
}

//______________________________________________________________________________
void GenerateLinkdef(int *argc, char **argv, int iv)
{
   FILE *fl = fopen(autold, "w");
   if (fl==0) {
      Error(0, "Could not write the automatically generated Linkdef: %s\n", autold);
      exit(1);
   }

   fprintf(fl, "#ifdef __CINT__\n\n");
   fprintf(fl, "#pragma link off all globals;\n");
   fprintf(fl, "#pragma link off all classes;\n");
   fprintf(fl, "#pragma link off all functions;\n\n");

   for (int i = iv; i < *argc; i++) {
      char *s, trail[3];
      int   nostr = 0, noinp = 0, bcnt = 0, l = strlen(argv[i])-1;
      for (int j = 0; j < 3; j++) {
         if (argv[i][l] == '-') {
            argv[i][l] = '\0';
            nostr = 1;
            l--;
         }
         if (argv[i][l] == '!') {
            argv[i][l] = '\0';
            noinp = 1;
            l--;
         }
         if (argv[i][l] == '+') {
            argv[i][l] = '\0';
            bcnt = 1;
            l--;
         }
      }
      if (nostr || noinp) {
         trail[0] = 0;
         if (nostr) strcat(trail, "-");
         if (noinp) strcat(trail, "!");
      }
      if (bcnt) {
         strcpy(trail, "+");
         if (nostr)
            Error(0, "option + mutual exclusive with -\n");
      }
      char *cls = strrchr(argv[i], '/');
      if (!cls) cls = strrchr(argv[i], '\\');
      if (cls)
         cls++;
      else
         cls = argv[i];
      if ((s = strrchr(cls, '.'))) *s = '\0';
      if (nostr || noinp || bcnt)
         fprintf(fl, "#pragma link C++ class %s%s;\n", cls, trail);
      else
         fprintf(fl, "#pragma link C++ class %s;\n", cls);
      if (s) *s = '.';
   }

   fprintf(fl, "\n#endif\n");
   fclose(fl);
}

//______________________________________________________________________________
const char *Which(const char *fname)
{
   // Find file name in path specified via -I statements to CINT.
   // Can be only called after G__main(). Return pointer to static
   // space containing full pathname or 0 in case file not found.

   static char pname[1024];
   FILE *fp = 0;

   strcpy(pname, fname);
#ifdef WIN32
   fp = fopen(pname, "rb");
#else
   fp = fopen(pname, "r");
#endif
   if (fp) {
      fclose(fp);
      return pname;
   }

   struct G__includepath *ipath = G__getipathentry();

   while (!fp && ipath->pathname) {
#ifdef WIN32
      sprintf(pname, "%s\\%s", ipath->pathname, fname);
      fp = fopen(pname, "rb");
#else
      sprintf(pname, "%s/%s", ipath->pathname, fname);
      fp = fopen(pname, "r");
#endif
      ipath = ipath->next;
   }
   if (fp) {
      fclose(fp);
      return pname;
   }
   return 0;
}

//______________________________________________________________________________
char *StrDup(const char *str)
{
   // Duplicate the string str. The returned string has to be deleted by
   // the user.

   if (!str) return 0;

   // allocate 20 extra characters in case of eg, vector<vector<T>>
   char *s = new char[strlen(str)+20];
   if (s) strcpy(s, str);

   return s;
}

//______________________________________________________________________________
char *Compress(const char *str)
{
   // Remove all blanks from the string str. The returned string has to be
   // deleted by the user.

   if (!str) return 0;

   const char *p = str;
   // allocate 20 extra characters in case of eg, vector<vector<T>>
   char *s, *s1 = new char[strlen(str)+20];
   s = s1;

   while (*p) {
      // keep space for A<const B>!
      if (*p != ' ' || (p-str>6 && !strncmp(p-5,"const",5)))
         *s++ = *p;
      p++;
   }
   *s = '\0';

   return s1;
}

//______________________________________________________________________________
void StrcpyWithEsc(char *escaped, const char *original)
{
   // Copy original into escaped BUT make sure that the \ characters
   // are properly escaped (on Windows temp files have \'s).

   int j, k;
   j = 0; k = 0;
   while (original[j] != '\0') {
      if (original[j] == '\\')
         escaped[k++] = '\\';
      escaped[k++] = original[j++];
   }
   escaped[k] = '\0';
}

void ReplaceFile(const char *tmpdictname, const char *dictname)
{
   // Unlink dictname and move tmpdictname into dictname

#ifdef WIN32
   int tries=0;
   bool success=false;
   while (!success && ++tries<51) {
      success = (unlink(dictname) != -1);
      if (!success && tries<50)
         if (errno!=EACCES) break;
         else Sleep(200);
   }
   if (success) {
      success=false;
      tries=0;
      while (!success && ++tries<52) {
         success = (rename(tmpdictname, dictname) != -1);
         if (!success && tries<51)
            if (errno!=EACCES) break;
            else Sleep(200);
      }
   }
   if (!success)
      Error(0, "rootcint: failed to rename %s to %s in ReplaceBundleInDict() after %d tries (error is %d)\n",
               tmpdictname, dictname, tries, errno);
#else
   if (unlink(dictname) == -1 || rename(tmpdictname, dictname) == -1)
      Error(0, "rootcint: failed to rename %s to %s in ReplaceBundleInDict()\n",
               tmpdictname, dictname);
#endif

}

//______________________________________________________________________________
void ReplaceBundleInDict(const char *dictname, const string &bundlename)
{
   // Replace the bundlename in the dict.cxx and .h file by the contents
   // of the bundle.

   // First patch dict.cxx. Create tmp file and copy dict.cxx to this file.
   // When discovering a line like:
   //   G__add_compiledheader("bundlename");
   // replace it by the appropriate number of lines contained in the bundle.

   FILE *fpd = fopen(dictname, "r");
   if (!fpd) {
      Error(0, "rootcint: failed to open %s in ReplaceBundleInDict()\n",
               dictname);
      return;
   }

   char tmpdictname[512];
   sprintf(tmpdictname, "%s_+_+_+rootcinttmp", dictname);
   FILE *tmpdict = fopen(tmpdictname, "w");
   if (!tmpdict) {
      Error(0, "rootcint: failed to open %s in ReplaceBundleInDict()\n",
               tmpdictname);
      fclose(fpd);
      return;
   }

   char esc_bundlename[512];
   StrcpyWithEsc(esc_bundlename, bundlename.c_str());

   char checkline[kMaxLen];
   sprintf(checkline, "  G__add_compiledheader(\"%s\");", esc_bundlename);
   int clen = strlen(checkline);

   char line[BUFSIZ];
   if (tmpdict && fpd) {
      while (fgets(line, BUFSIZ, fpd)) {
         if (!strncmp(line, checkline, clen)) {
            FILE *fb = fopen(bundlename.c_str(), "r");
            if (!fb) {
               Error(0, "rootcint: failed to open %s in ReplaceBundleInDict()\n",
                        bundlename.c_str());
               fclose(fpd);
               fclose(tmpdict);
               remove(tmpdictname);
               return;
            }
            while (fgets(line, BUFSIZ, fb)) {
               char *s = strchr(line, '"');
               if (!s) continue;
               s++;
               char *s1 = strrchr(s, '"');
               if (((strstr(s,"LinkDef") || strstr(s,"Linkdef") ||
                     strstr(s,"linkdef")) && strstr(s,".h"))) {
                  s1 = 0;
               }
               if (s1) {
                  *s1 = 0;
                  fprintf(tmpdict, "  G__add_compiledheader(\"%s\");\n", s);
               }
            }
            fclose(fb);
         } else
            fprintf(tmpdict, "%s", line);
      }
   }

   fclose(tmpdict);
   fclose(fpd);

   ReplaceFile(tmpdictname,dictname);

   // Next patch dict.h. Create tmp file and copy dict.h to this file.
   // When discovering a line like:
   //   #include "bundlename"
   // replace it by the appropriate number of lines contained in the bundle.

   // make dict.h
   char dictnameh[kMaxLen];
   strcpy(dictnameh, dictname);
   char *s = strrchr(dictnameh, '.');
   if (s) {
      *(s+1) = 'h';
      *(s+2) = 0;
   } else {
      Error(0, "rootcint: failed create dict.h in ReplaceBundleInDict()\n");
      return;
   }

   fpd = fopen(dictnameh, "r");
   if (!fpd) {
      Error(0, "rootcint: failed to open %s in ReplaceBundleInDict()\n",
              dictnameh);
      return;
   }
   tmpdict = fopen(tmpdictname, "w");
   if (!tmpdict) {
      Error(0, "rootcint: failed to open %s in ReplaceBundleInDict()\n",
               tmpdictname);
      fclose(fpd);
      return;
   }

   sprintf(checkline, "#include \"%s\"", esc_bundlename);
   clen = strlen(checkline);

   if (tmpdict && fpd) {
      while (fgets(line, BUFSIZ, fpd)) {
         if (!strncmp(line, checkline, clen)) {
            FILE *fb = fopen(bundlename.c_str(), "r");
            if (!fb) {
               Error(0, "rootcint: failed to open %s in ReplaceBundleInDict()\n",
                     bundlename.c_str());
               fclose(tmpdict);
               fclose(fpd);
               return;
            }
            while (fgets(line, BUFSIZ, fb))
               if (!((strstr(line,"LinkDef") || strstr(line,"Linkdef") || strstr(line,"linkdef")) &&
                     strstr(line,".h")))
                  fprintf(tmpdict, "%s", line);
            fclose(fb);
         } else
            fprintf(tmpdict, "%s", line);
      }
   }

   fclose(tmpdict);
   fclose(fpd);

   ReplaceFile(tmpdictname,dictnameh);
}

string bundlename;
string tname;
string dictsrc;

//______________________________________________________________________________
void CleanupOnExit(int code) {
   // removes tmp files, and (if code!=0) output files
   if (!bundlename.empty()) unlink(bundlename.c_str());
   if (!tname.empty()) unlink(tname.c_str());
   if (autold[0]) unlink(autold);
   if (code) {
      if (!dictsrc.empty()) {
         unlink(dictsrc.c_str());
         // also remove the .d file belonging to dictsrc
         size_t posExt=dictsrc.rfind('.');
         if (posExt!=string::npos) {
            dictsrc.replace(posExt, dictsrc.length(), ".d");
            unlink(dictsrc.c_str());
         }
      }
   }
   // also remove the .def file created by CINT.
   {
      size_t posExt=dictsrc.rfind('.');
      if (posExt!=string::npos) {
         dictsrc.replace(posExt, dictsrc.length(), ".def");
         unlink(dictsrc.c_str());

         size_t posSlash=dictsrc.rfind('/');
         if (posSlash==string::npos) {
            posSlash=dictsrc.rfind('\\');
         }
         if (posSlash!=string::npos) {
            dictsrc.replace(0,posSlash+1,"");
            unlink(dictsrc.c_str());
         }
      }
   }

}


//______________________________________________________________________________
int main(int argc, char **argv)
{
#ifdef __MWERKS__
   argc = ccommand(&argv);
#endif

   if (argc < 2) {
      fprintf(stderr,
       "Usage: %s [-v][-v0-4] [-cint|-reflex|-gccxml] [-l] [-f] [out.cxx] [-c] file1.h[+][-][!] file2.h[+][-][!]...[LinkDef.h]\n",
              argv[0]);
      fprintf(stderr, "For more extensive help type: %s -h\n", argv[0]);
      return 1;
   }

   char dictname[512];
   int i, j, ic, ifl, force;
   int icc = 0;
   int use_preprocessor = 0;
   int longheadername = 0;
   string dictpathname;
   string libfilename;
   const char *env_dict_type=getenv("ROOTDICTTYPE");

   if (env_dict_type)
      if (!strcmp(env_dict_type, "cint"))
         dict_type=kDictTypeCint;
      else if (!strcmp(env_dict_type, "reflex"))
         dict_type=kDictTypeReflex;
      else if (!strcmp(env_dict_type, "gccxml"))
         dict_type=kDictTypeGCCXML;

   sprintf(autold, autoldtmpl, getpid());

   ic = 1;
   if (!strcmp(argv[ic], "-v")) {
      gErrorIgnoreLevel = kInfo; // The default is kError
      ic++;
   } else if (!strcmp(argv[ic], "-v0")) {
      gErrorIgnoreLevel = kFatal; // Explicitly remove all messages
      ic++;
   } else if (!strcmp(argv[ic], "-v1")) {
      gErrorIgnoreLevel = kError; // Only error message (default)
      ic++;
   } else if (!strcmp(argv[ic], "-v2")) {
      gErrorIgnoreLevel = kWarning; // error and warning message
      ic++;
   } else if (!strcmp(argv[ic], "-v3")) {
      gErrorIgnoreLevel = kNote; // error, warning and note
      ic++;
   } else if (!strcmp(argv[ic], "-v4")) {
      gErrorIgnoreLevel = kInfo; // Display all information (same as -v)
      ic++;
   }
   if (!strcmp(argv[ic], "-cint")) {
      dict_type = kDictTypeCint;
      ic++;
   } else if (!strcmp(argv[ic], "-reflex")) {
      dict_type = kDictTypeReflex;
      ic++;
   } else if (!strcmp(argv[ic], "-gccxml")) {
      dict_type = kDictTypeGCCXML;
      ic++;
   }

   if (dict_type==kDictTypeGCCXML) {
      int rc =  system("genreflex-rootcint --gccxml-available");
      if (rc) dict_type=kDictTypeReflex; // fall back to reflex
   }

   const char* libprefix = "--lib-list-prefix=";

   ifl = 0;
   while (strncmp(argv[ic], "-",1)==0
          && strcmp(argv[ic], "-f")!=0 ) {
      if (!strcmp(argv[ic], "-l")) {

         longheadername = 1;
         ic++;
      } else if (!strncmp(argv[ic],libprefix,strlen(libprefix))) {

         gLiblistPrefix = argv[ic]+strlen(libprefix);

         string filein = gLiblistPrefix + ".in";
         FILE *fp;
         if ((fp = fopen(filein.c_str(), "r")) == 0) {
            Error(0, "%s: The input list file %s does not exist\n", argv[0], filein.c_str());
            return 1;
         }
         fclose(fp);

         ic++;
      } else {
         break;
      }
   }


   if (!strcmp(argv[ic], "-f")) {
      force = 1;
      ic++;
   } else if (!strcmp(argv[1], "-?") || !strcmp(argv[1], "-h")) {
      fprintf(stderr, "%s\n", help);
      return 1;
   } else if (!strncmp(argv[ic], "-",1)) {
       fprintf(stderr,"Usage: %s [-v][-v0-4] [-reflex] [-l] [-f] [out.cxx] [-c] file1.h[+][-][!] file2.h[+][-][!]...[LinkDef.h]\n",
              argv[0]);
      fprintf(stderr,"Only one verbose flag is authorized (one of -v, -v0, -v1, -v2, -v3, -v4)\n"
                     "and must be before the -f flags\n");
      fprintf(stderr,"For more extensive help type: %s -h\n", argv[0]);
      return 1;
   } else {
      force = 0;
   }

   if (strstr(argv[ic],".C")  || strstr(argv[ic],".cpp") ||
       strstr(argv[ic],".cp") || strstr(argv[ic],".cxx") ||
       strstr(argv[ic],".cc") || strstr(argv[ic],".c++")) {
      FILE *fp;
      if ((fp = fopen(argv[ic], "r")) != 0) {
         fclose(fp);
         if (!force) {
            Error(0, "%s: output file %s already exists\n", argv[0], argv[ic]);
            return 1;
         }
      }
      string header( argv[ic] );
      int loc = strrchr(argv[ic],'.') - argv[ic];
      header[loc+1] = 'h';
      header[loc+2] = '\0';
      if ((fp = fopen(header.c_str(), "r")) != 0) {
         fclose(fp);
         if (!force) {
            Error(0, "%s: output file %s already exists\n", argv[0], header.c_str());
            return 1;
         } else {
            for (int k = ic+1; k < argc; ++k) {
               if (*argv[k] != '-' && *argv[k] != '+') {
                  if (strcmp(header.c_str(),argv[k])==0) {
                     Error(0, "%s: output file %s would overwrite one of the input files!\n", argv[0], header.c_str());
                     return 1;
                  }
                  if (strcmp(argv[ic],argv[k])==0) {
                     Error(0, "%s: output file %s would overwrite one of the input files!\n", argv[0],argv[ic]);
                     return 1;
                  }
               }
            }
         }
      }

      dictsrc=argv[ic];
      fp = fopen(argv[ic], "w");
      if (fp) fclose(fp);    // make sure file is created and empty
      ifl = ic;
      ic++;

      // remove possible pathname to get the dictionary name
      strcpy(dictname, argv[ifl]);
      char *p = 0;
      // find the right part of then name.
      for (p = dictname + strlen(dictname)-1;p!=dictname;--p) {
         if (*p =='/' ||  *p =='\\') {
            *p = 0;
            if (p == dictname) {
               dictpathname = "/";
            } else {
               dictpathname = dictname;
            }
            break;
         }
      }
      if (!p)
         p = dictname;
      else if (p != dictname) {
         p++;
         memmove(dictname, p, strlen(p)+1);
      }
   } else if (!strcmp(argv[1], "-?") || !strcmp(argv[1], "-h")) {
      fprintf(stderr, "%s\n", help);
      return 1;
   } else {
      ic = 1;
      if (force) ic = 2;
      ifl = 0;
   }

   // If the user request use of a preprocessor we are going to bundle
   // all the files into one so that cint considers them one compilation
   // unit and so that each file that contains code guard is really
   // included only once.
   for (i = 1; i < argc; i++)
      if (strcmp(argv[i], "-p") == 0) use_preprocessor = 1;

#ifndef __CINT__
   int   argcc, iv, il;
   char  path[16][128];
   char *argvv[500];

   for (i = 0; i < 16; i++)
      path[i][0] = 0;

#ifndef ROOTINCDIR
# ifndef ROOTBUILD
   if (getenv("ROOTSYS")) {
#  ifdef __MWERKS__
      sprintf(path[0], "-I%s:include", getenv("ROOTSYS"));
      sprintf(path[1], "-I%s:src", getenv("ROOTSYS"));
#  else
      sprintf(path[0], "-I%s/include", getenv("ROOTSYS"));
      sprintf(path[1], "-I%s/src", getenv("ROOTSYS"));
#  endif
   } else {
      Error(0, "%s: environment variable ROOTSYS not defined\n", argv[0]);
      return 1;
   }
# else
   sprintf(path[0], "-Ibase/inc");
   sprintf(path[1], "-Icont/inc");
   sprintf(path[2], "-Iinclude");
# endif
#else
   sprintf(path[0], "-I%s", ROOTINCDIR);
#endif

   argvv[0] = argv[0];
   argcc = 1;

   if (!strcmp(argv[ic], "-c")) {
      icc++;
      if (ifl) {
         char *s;
         ic++;
         argvv[argcc++] = "-q0";
         argvv[argcc++] = "-n";
         argvv[argcc] = (char *)calloc(strlen(argv[ifl])+1, 1);
         strcpy(argvv[argcc], argv[ifl]); argcc++;
         argvv[argcc++] = "-N";
         s = strrchr(dictname,'.');
         argvv[argcc] = (char *)calloc(strlen(dictname), 1);
         strncpy(argvv[argcc], dictname, s-dictname); argcc++;

         while (ic < argc && (*argv[ic] == '-' || *argv[ic] == '+')) {
            if (strcmp("+P", argv[ic]) == 0 ||
                strcmp("+V", argv[ic]) == 0 ||
                strcmp("+STUB", argv[ic]) == 0) {
               // break when we see positional options
               break;
            }
            argvv[argcc++] = argv[ic++];
         }

         for (i = 0; path[i][0]; i++)
            argvv[argcc++] = path[i];

#ifdef __hpux
         argvv[argcc++] = "-I/usr/include/X11R5";
#endif
         switch (gErrorIgnoreLevel) {
            case kInfo:     argvv[argcc++] = "-J4"; break;
            case kNote:     argvv[argcc++] = "-J3"; break;
            case kWarning:  argvv[argcc++] = "-J2"; break;
            case kError:    argvv[argcc++] = "-J1"; break;
            case kSysError:
            case kFatal:    argvv[argcc++] = "-J0"; break;
            default:        argvv[argcc++] = "-J1"; break;
         }

         if (!use_preprocessor) {
            // If the compiler's preprocessor is not used
            // we still need to declare the compiler specific flags
            // so that the header file are properly parsed.
#ifdef __KCC
            argvv[argcc] = (char *)calloc(64, 1);
            sprintf(argvv[argcc], "-D__KCC=%ld", (long)__KCC); argcc++;
#endif
#ifdef __INTEL_COMPILER
            argvv[argcc] = (char *)calloc(64, 1);
            sprintf(argvv[argcc], "-D__INTEL_COMPILER=%ld", (long)__INTEL_COMPILER); argcc++;
#endif
#ifdef __xlC__
            argvv[argcc] = (char *)calloc(64, 1);
            sprintf(argvv[argcc], "-D__xlC__=%ld", (long)__xlC__); argcc++;
#endif
#ifdef __GNUC__
            argvv[argcc] = (char *)calloc(64, 1);
            sprintf(argvv[argcc], "-D__GNUC__=%ld", (long)__GNUC__); argcc++;
#endif
#ifdef __GNUC_MINOR__
            argvv[argcc] = (char *)calloc(64, 1);
            sprintf(argvv[argcc], "-D__GNUC_MINOR__=%ld", (long)__GNUC_MINOR__); argcc++;
#endif
#ifdef __HP_aCC
            argvv[argcc] = (char *)calloc(64, 1);
            sprintf(argvv[argcc], "-D__HP_aCC=%ld", (long)__HP_aCC); argcc++;
#endif
#ifdef __sun
            argvv[argcc] = (char *)calloc(64, 1);
            sprintf(argvv[argcc], "-D__sun=%ld", (long)__sun); argcc++;
#endif
#ifdef __SUNPRO_CC
            argvv[argcc] = (char *)calloc(64, 1);
            sprintf(argvv[argcc], "-D__SUNPRO_CC=%ld", (long)__SUNPRO_CC); argcc++;
#endif
#ifdef __ia64__
            argvv[argcc] = (char *)calloc(64, 1);
            sprintf(argvv[argcc], "-D__ia64__=%ld", (long)__ia64__); argcc++;
#endif
#ifdef __x86_64__
            argvv[argcc] = (char *)calloc(64, 1);
            sprintf(argvv[argcc], "-D__x86_64__=%ld", (long)__x86_64__); argcc++;
#endif
#ifdef R__B64
            argvv[argcc] = (char *)calloc(64, 1);
            sprintf(argvv[argcc], "-DR__B64"); argcc++;
#endif
#ifdef _WIN32
            argvv[argcc] = (char *)calloc(64, 1);
            sprintf(argvv[argcc], "-D_WIN32=%ld",(long)_WIN32); argcc++;
#endif
#ifdef WIN32
            argvv[argcc] = (char *)calloc(64, 1);
            sprintf(argvv[argcc], "-DWIN32=%ld",(long)WIN32); argcc++;
#endif
#ifdef _MSC_VER
            argvv[argcc] = (char *)calloc(64, 1);
            sprintf(argvv[argcc], "-D_MSC_VER=%ld",(long)_MSC_VER); argcc++;
#endif
         }
#ifdef ROOTBUILD
         argvv[argcc++] = "-DG__NOCINTDLL";
#endif
         argvv[argcc++] = "-DTRUE=1";
         argvv[argcc++] = "-DFALSE=0";
         argvv[argcc++] = "-Dexternalref=extern";
         argvv[argcc++] = "-DSYSV";
         argvv[argcc++] = "-D__MAKECINT__";
         argvv[argcc++] = "-V";        // include info on private members
         if (dict_type==kDictTypeReflex) {
            argvv[argcc++] = "-c-3";
         }
         else argvv[argcc++] = "-c-10";
         argvv[argcc++] = "+V";        // turn on class comment mode
         if (!use_preprocessor) {
#ifdef ROOTBUILD
            argvv[argcc++] = "base/inc/TROOT.h";
            argvv[argcc++] = "base/inc/TMemberInspector.h";
#else
            argvv[argcc++] = "TROOT.h";
            argvv[argcc++] = "TMemberInspector.h";
#endif
         }
      } else {
         Error(0, "%s: option -c can only be used when an output file has been specified\n", argv[0]);
         return 1;
      }
   }

   iv = 0;
   il = 0;

   char esc_arg[512];
   bool insertedBundle = false;
   FILE *bundle = 0;
   if (use_preprocessor) {
      bundlename = R__tmpnam();
      bundlename += ".h";
      bundle = fopen(bundlename.c_str(), "w");
      if (!bundle) {
         Error(0, "%s: failed to open %s, usage of external preprocessor by CINT is not optimal\n",
                 argv[0], bundlename.c_str());
         use_preprocessor = 0;
      } else {
         fprintf(bundle,"#include \"TROOT.h\"\n");
         fprintf(bundle,"#include \"TMemberInspector.h\"\n");
      }
   }
   for (i = ic; i < argc; i++) {
      if (!iv && *argv[i] != '-' && *argv[i] != '+') {
         if (!icc) {
            for (j = 0; path[j][0]; j++)
               argvv[argcc++] = path[j];
            argvv[argcc++] = "+V";
         }
         iv = i;
      }
      if ((strstr(argv[i],"LinkDef") || strstr(argv[i],"Linkdef") ||
           strstr(argv[i],"linkdef")) && strstr(argv[i],".h")) {
         il = i;
         if (i != argc-1) {
            Error(0, "%s: %s must be last file on command line\n", argv[0], argv[i]);
            return 1;
         }
      }
      if (!strcmp(argv[i], "-c")) {
         Error(0, "%s: option -c must come directly after the output file\n", argv[0]);
         return 1;
      }
      if (use_preprocessor && *argv[i] != '-' && *argv[i] != '+') {
         StrcpyWithEsc(esc_arg, argv[i]);
         fprintf(bundle,"#include \"%s\"\n", esc_arg);
         if (!insertedBundle) {
            argvv[argcc++] = (char*)bundlename.c_str();
            insertedBundle = true;
         }
      } else
         argvv[argcc++] = argv[i];
   }
   if (use_preprocessor) {
      fclose(bundle);
   }

   if (!iv) {
      Error(0, "%s: no input files specified\n", argv[0]);
      return 1;
   }

   if (!il) {
      GenerateLinkdef(&argc, argv, iv);
      argvv[argcc++] = autold;
   }
   G__ShadowMaker::VetoShadow(); // we create them ourselves
   G__setothermain(2);
   G__set_ioctortype_handler( (int (*)(const char*))AddConstructorType );
   if (gLiblistPrefix.length()) G__set_beforeparse_hook (EnableAutoLoading);
   if (G__main(argcc, argvv) < 0) {
      Error(0, "%s: error loading headers...\n", argv[0]);
      CleanupOnExit(1);
      return 1;
   } else {
      if (ifl) {
         FILE *fpd = fopen(argv[ifl], "r");
         if (fpd==0) {
            // The dictionary file was not created by CINT.
            // There mush have been an error.
            Error(0, "%s: error loading headers...\n", argv[0]);
            CleanupOnExit(1);
            return 1;
         }
         fclose(fpd);
      }
   }
   G__setglobalcomp(0);  // G__NOLINK
#endif

   // We ran cint to load the in-memory database,
   // so that the I/O code can be properly generated.
   // So now let's call GCCXML if requested
   if (dict_type==kDictTypeGCCXML) {
      string gccxml_rootcint_call;
#ifndef ROOTBUILD
      if (getenv("ROOTSYS")) {
         gccxml_rootcint_call=getenv("ROOTSYS");
# ifdef WIN32
         gccxml_rootcint_call+="\\bin\\";
# else
         gccxml_rootcint_call+="/bin/";
# endif
      }
#else
# ifdef WIN32
      gccxml_rootcint_call="bin\\";
# else
      gccxml_rootcint_call="bin/";
# endif
#endif
      gccxml_rootcint_call+="genreflex-rootcint";

      for (int iarg=1; iarg<argc; ++iarg) {
         gccxml_rootcint_call+=" ";
         gccxml_rootcint_call+=argv[iarg];

         if (!strcmp(argv[iarg], "-c")) {
            for (i = 0; path[i][0]; i++) {
               gccxml_rootcint_call+=" ";
               gccxml_rootcint_call+=path[i];
            }
            gccxml_rootcint_call+=" -DR__GCCXML";
#ifdef ROOTBUILD
            gccxml_rootcint_call+=" -DG__NOCINTDLL";
#endif
            gccxml_rootcint_call+=" -DTRUE=1";
            gccxml_rootcint_call+=" -DFALSE=0";
            gccxml_rootcint_call+=" -DR__EXTERN=extern";
            gccxml_rootcint_call+=" -Dexternalref=extern";
            gccxml_rootcint_call+=" -DSYSV";
#ifdef ROOTBUILD
            gccxml_rootcint_call+=" base/inc/TROOT.h";
            gccxml_rootcint_call+=" base/inc/TMemberInspector.h";
#else
            gccxml_rootcint_call+=" TROOT.h";
            gccxml_rootcint_call+=" TMemberInspector.h";
#endif
         }
      }
      //printf("Calling %s\n", gccxml_rootcint_call.c_str());
      int rc=system(gccxml_rootcint_call.c_str());
      if (rc) {
         CleanupOnExit(rc);
         return rc;
      }
   }

   if (use_preprocessor && icc)
      ReplaceBundleInDict(argv[ifl], bundlename);

   // Check if code goes to stdout or cint file, use temporary file
   // for prepending of the rootcint generated code (STK)
   std::ofstream fileout;
   if (ifl) {
      tname = R__tmpnam();
      fileout.open(tname.c_str());
      dictSrcOut = &fileout;
      if (!(*dictSrcOut)) {
         Error(0, "rootcint: failed to open %s in main\n",
               tname.c_str());
         CleanupOnExit(1);
         return 1;
      }
   } else
      dictSrcOut = &std::cout;

   time_t t = time(0);
   (*dictSrcOut) << "//"  << std::endl
       << "// File generated by " << argv[0] << " at " << ctime(&t) << std::endl
       << "// Do NOT change. Changes will be lost next time file is generated" << std::endl
       << "//" << std::endl << std::endl

       << "#include \"RConfig.h\"" << std::endl
       << "#if !defined(R__ACCESS_IN_SYMBOL)" << std::endl
       << "//Break the privacy of classes -- Disabled for the moment" << std::endl
       << "#define private public" << std::endl
       << "#define protected public" << std::endl
       << "#endif" << std::endl << std::endl;
#ifndef R__SOLARIS
   (*dictSrcOut) << "// Since CINT ignores the std namespace, we need to do so in this file." << std::endl
       << "namespace std {} using namespace std;" << std::endl << std::endl;
   int linesToSkip = 15; // number of lines up to here.
#else
   int linesToSkip = 12; // number of lines up to here.
#endif

   (*dictSrcOut) << "#include \"TClass.h\"" << std::endl
       << "#include \"TBuffer.h\"" << std::endl
       << "#include \"TStreamerInfo.h\"" << std::endl
       << "#include \"TMemberInspector.h\"" << std::endl
       << "#include \"TError.h\"" << std::endl << std::endl
       << "#ifndef G__ROOT" << std::endl
       << "#define G__ROOT" << std::endl
       << "#endif" << std::endl << std::endl
       << "#include \"RtypesImp.h\"" << std::endl
       << "#include \"TCollectionProxy.h\"" << std::endl
       << "#include \"TIsAProxy.h\"" << std::endl;
#ifdef R__SOLARIS
   (*dictSrcOut) << "// Since CINT ignores the std namespace, we need to do so in this file." << std::endl
       << "namespace std {} using namespace std;" << std::endl << std::endl;
#endif

   // Loop over all command line arguments and write include statements.
   // Skip options and any LinkDef.h.
   if (ifl && !icc) {
      for (i = ic; i < argc; i++) {
         if (*argv[i] != '-' && *argv[i] != '+' &&
             !((strstr(argv[i],"LinkDef") || strstr(argv[i],"Linkdef") ||
                strstr(argv[i],"linkdef")) && strstr(argv[i],".h")))
            (*dictSrcOut) << "#include \"" << argv[i] << "\"" << std::endl;
      }
      (*dictSrcOut) << std::endl;
   }


   //
   // We will loop over all the classes several times.
   // In order we will call
   //
   //     WriteShadowClass
   //     WriteClassInit (code to create the TGenericClassInfo)
   //     check for constructor and operator input
   //     WriteClassFunctions (declared in ClassDef)
   //     WriteClassCode (Streamer,ShowMembers,Auxiliary functions)
   //

   //
   // Loop over all classes and write the Shadow class if needed
   //

   // Open LinkDef file for reading, so that we can process classes
   // in order of appearence in this file (STK)
   FILE *fpld = 0;
   if (!il) {
      // Open auto-generated file
      fpld = fopen(autold, "r");
   } else {
      // Open file specified on command line
      const char* filename=Which(argv[il]);
      if (!filename) {
         Error(0, "%s: cannot open file %s\n", argv[0], argv[il]);
         CleanupOnExit(1);
         return 1;
      }
      fpld = fopen(filename, "r");
   }
   if (!fpld) {
      Error(0, "%s: cannot open file %s\n", argv[0], il ? argv[il] : autold);
      CleanupOnExit(1);
      return 1;
   }

   // Read LinkDef file and process the #pragma link C++ ioctortype
   char consline[256];
   while (fgets(consline, 256, fpld)) {
      bool constype = false;
      if ((strcmp(strtok(consline, " "), "#pragma") == 0) &&
          (strcmp(strtok(0, " "), "link") == 0) &&
          (strcmp(strtok(0, " "), "C++") == 0) &&
          (strcmp(strtok(0, " " ), "ioctortype") == 0)) {

         constype = true;
      }

      if (constype) {

         char *request = strtok(0, "-!+;");
         // just in case remove trailing space and tab
         while (*request == ' ') request++;
         int len = strlen(request)-1;
         while (request[len]==' ' || request[len]=='\t') request[len--] = '\0';
         request = Compress(request); //no space between tmpl arguments allowed
         AddConstructorType(request);

      }
   }
   rewind(fpld);
   AddConstructorType("TRootIOCtor");
   AddConstructorType("");

   const char* shadowNSName="ROOT";
   if (dict_type != kDictTypeCint)
      shadowNSName = "ROOT::Reflex";
   G__ShadowMaker myShadowMaker((*dictSrcOut), shadowNSName, NeedShadowClass,
      dict_type==kDictTypeCint ? NeedTypedefShadowClass : 0);
   shadowMaker = &myShadowMaker;

   G__ShadowMaker::VetoShadow(false);
   shadowMaker->WriteAllShadowClasses();

   //
   // Loop over all classes and create Streamer() & Showmembers() methods
   //

   G__ClassInfo cl;
   cl.Init();
   while (cl.Next()) {
      if (cl.Linkage() == G__CPPLINK && !cl.IsLoaded()) {
         Error(0,"A dictionary has been requested for %s but there is no declaration!\n",cl.Name());
         continue;
      }
      if ((cl.Property() & (G__BIT_ISCLASS|G__BIT_ISSTRUCT)) && cl.Linkage() == G__CPPLINK) {

         // Write Code for initialization object (except for STL containers)
         if ( TClassEdit::IsSTLCont(cl.Name()) ) {
            RStl::inst().GenerateTClassFor( cl.Name() );
         } else {
            WriteClassInit(cl);
         }
      } else if (((cl.Property() & (G__BIT_ISNAMESPACE)) && cl.Linkage() == G__CPPLINK)) {
         WriteNamespaceInit(cl);
      }
   }

   cl.Init();
   bool has_input_error = false;
   while (cl.Next()) {
      if ((cl.Property() & (G__BIT_ISCLASS|G__BIT_ISSTRUCT)) && cl.Linkage() == G__CPPLINK) {
         if (!cl.IsLoaded()) {
            continue;
         }
         if (cl.HasMethod("Streamer")) {
            if (!(cl.RootFlag() & G__NOINPUTOPERATOR)) {
               // We do not write out the input operator anymore, it is a template
#if defined R__CONCRETE_INPUT_OPERATOR
               WriteInputOperator(cl);
#endif
            } else {
               int version = GetClassVersion(cl);
               if (version!=0) {
                  // Only Check for input operator is the object is I/O has
                  // been requested.
                  has_input_error |= CheckInputOperator(cl);
               }
            }
         }
         bool res = CheckConstructor(cl);
         if (!res) {
            // has_input_error = true;
         }
         has_input_error |= !CheckClassDef(cl);
      }
   }

   if (has_input_error) {
      // Be a little bit makefile friendly and remove the dictionary in case of error.
      // We could add an option -k to keep the file even in case of error.
      CleanupOnExit(1);
      exit(1);
   }

   //
   // Write all TBuffer &operator>>(...), Class_Name(), Dictionary(), etc.
   // first to allow template specialisation to occur before template
   // instantiation (STK)
   //
   cl.Init();
   while (cl.Next()) {
      if (!cl.IsLoaded()) {
         continue;
      }
      if ((cl.Property() & (G__BIT_ISCLASS|G__BIT_ISSTRUCT)) && cl.Linkage() == G__CPPLINK) {
         // Write Code for Class_Name() and static variable
         if (cl.HasMethod("Class_Name")) {
            WriteClassFunctions(cl, cl.IsTmplt());
         }
      }
   }

   // Keep track of classes processed by reading Linkdef file.
   // When all classes in LinkDef are done, loop over all classes known
   // to CINT output the ones that were not in the LinkDef. This can happen
   // in case "#pragma link C++ defined_in" is used.
   //const int kMaxClasses = 2000;
   //char *clProcessed[kMaxClasses];
   vector<string> clProcessed;
   int   ncls = 0;

   // Read LinkDef file and process valid entries (STK)
   char line[256];
   char cline[256];
   char nline[256];
   while (fgets(line, 256, fpld)) {

      bool skip = true;
      bool force = false;
      strcpy(cline,line);
      strcpy(nline,line);
      int len = strlen(line);

      // Check if the line contains a "#pragma link C++ class" specification,
      // if so, process the class (STK)
      if ((strcmp(strtok(line, " "), "#pragma") == 0) &&
          (strcmp(strtok(0, " "), "link") == 0) &&
          (strcmp(strtok(0, " "), "C++") == 0) &&
          (strcmp(strtok(0, " " ), "class") == 0)) {

         skip = false;
         force = false;

      } else if ((strcmp(strtok(cline, " "), "#pragma") == 0) &&
                 (strcmp(strtok(0, " "), "create") == 0) &&
                 (strcmp(strtok(0, " "), "TClass") == 0)) {

         skip = false;
         force = true;

      } else if ((strcmp(strtok(nline, " "), "#pragma") == 0) &&
          (strcmp(strtok(0, " "), "link") == 0) &&
          (strcmp(strtok(0, " "), "C++") == 0) &&
          (strcmp(strtok(0, " " ), "namespace") == 0)) {

         skip = false;
         force = false;

      }

      if (!skip) {

         // Create G__ClassInfo object for this class and process. Be
         // careful with the hardcoded string of trailing options in case
         // these change (STK)

         int extraRootflag = 0;
         if (force && len>2) {
            char *endreq = line+len-2;
            bool ending = false;
            while (!ending) {
               switch ( (*endreq) ) {
                  case ';': break;
                  case '+': extraRootflag |= G__USEBYTECOUNT; break;
                  case '!': extraRootflag |= G__NOINPUTOPERATOR; break;
                  case '-': extraRootflag |= G__NOSTREAMER; break;
                  case ' ':
                  case '\t': break;
                  default:
                     ending = true;
               }
               --endreq;
            }
            if ( extraRootflag & (G__USEBYTECOUNT | G__NOSTREAMER) ) {
               Warning(line,"option + mutual exclusive with -, + prevails\n");
               extraRootflag &= ~G__NOSTREAMER;
            }
         }

         char *request = strtok(0, "-!+;");
         // just in case remove trailing space and tab
         while (*request == ' ') request++;
         int len = strlen(request)-1;
         while (request[len]==' ' || request[len]=='\t') request[len--] = '\0';
         request = Compress(request); //no space between tmpl arguments allowed
         G__ClassInfo cl(request);

         string fullname;
         if (cl.IsValid())
            fullname = cl.Fullname();
         else {
            fullname = request;
         }
         // In order to upgrade the pragma create TClass we would need a new function in
         // CINT's G__ClassInfo.
         // if (force && extraRootflag) cl.SetRootFlag(extraRootflag);
//          fprintf(stderr,"DEBUG: request==%s processed==%s rootflag==%d\n",request,fullname.c_str(),extraRootflag);
         delete [] request;

         // Avoid requesting the creation of a class infrastructure twice.
         // This could happen if one of the request link C++ class XXX is actually a typedef.
         int nxt = 0;
         for (i = 0; i < ncls; i++) {
            if ( clProcessed[i] == fullname ) {
               nxt++;
               break;
            }
         }
         if (nxt) continue;

         clProcessed.push_back( fullname );
         ncls++;

         if (force) {
            if ((cl.Property() & (G__BIT_ISCLASS|G__BIT_ISSTRUCT)) && cl.Linkage() != G__CPPLINK) {
               if (NeedShadowClass(cl)) {
                  (*dictSrcOut) << "namespace ROOT {" << std::endl
                      << "   namespace Shadow {" << std::endl;
                  shadowMaker->WriteShadowClass(cl);
                  (*dictSrcOut) << "   } // Of namespace ROOT::Shadow" << std::endl
                      << "} // Of namespace ROOT" << std::endl << std::endl;
               }
               if (G__ShadowMaker::IsSTLCont(cl.Name()) == 0 ) {
                  WriteClassInit(cl);
               }
            }
         }
         WriteClassCode(cl, force);
      }
   }

   // Loop over all classes and create Streamer() & ShowMembers() methods
   // for classes not in clProcessed list (exported via
   // "#pragma link C++ defined_in")
   cl.Init();

   while (cl.Next()) {
      int nxt = 0;
      // skip utility class defined in ClassImp
      if (!strncmp(cl.Fullname(), "R__Init", 7) ||
           strstr(cl.Fullname(), "::R__Init"))
         continue;
      string fullname( cl.Fullname() );
      for (i = 0; i < ncls; i++) {
         if ( clProcessed[i] == fullname ) {
            nxt++;
            break;
         }
      }
      if (nxt) continue;

      WriteClassCode(cl);
   }

   //RStl::inst().WriteStreamer(fp); //replaced by new Markus code
   RStl::inst().WriteClassInit(0);

   fclose(fpld);

   if (!il) remove(autold);
   if (use_preprocessor) remove(bundlename.c_str());

   // Append CINT dictionary to file containing Streamers and ShowMembers
   if (ifl) {
      char line[BUFSIZ];
      FILE *fpd = fopen(argv[ifl], "r");
      FILE* fp = fopen(tname.c_str(), "a");

      if (fp && fpd)
         while (fgets(line, BUFSIZ, fpd))
            fprintf(fp, "%s", line);

      if (fp)  fclose(fp);
      if (fpd) fclose(fpd);

      // copy back to dictionary file
      fpd = fopen(argv[ifl], "w");
      fp  = fopen(tname.c_str(), "r");

      if (fp && fpd) {

         // make name of dict include file "aapDict.cxx" -> "aapDict.h"
         int  nl = 0;
         char inclf[kMaxLen];
         char *s = strrchr(dictname, '.');
         if (s) *s = 0;
         sprintf(inclf, "%s.h", dictname);
         if (s) *s = '.';

         // during copy put dict include on top and remove later reference
         while (fgets(line, BUFSIZ, fp)) {
            if (!strncmp(line, "#include", 8) && strstr(line, inclf))
               continue;
            fprintf(fpd, "%s", line);
            // 'linesToSkip' is because we want to put it after #defined private/protected
            if (++nl == linesToSkip && icc) {
               switch (dict_type) {
                  case kDictTypeGCCXML:
                     fprintf(fpd, "#define G__DICTIONARY gccxml\n");
                     break;
                  case kDictTypeReflex:
                     fprintf(fpd, "#define G__DICTIONARY reflex\n");
                     break;
                  default:;
               }
               if (longheadername && dictpathname.length() ) {
                  fprintf(fpd, "#include \"%s/%s\"\n", dictpathname.c_str(), inclf);
               } else {
                  fprintf(fpd, "#include \"%s\"\n", inclf);
               }
            }
         }
      }

      if (fp)  fclose(fp);
      if (fpd) fclose(fpd);
      remove(tname.c_str());
   }

   if (gLiblistPrefix.length()) {
      string liblist_filename = gLiblistPrefix + ".out";

      ofstream outputfile( liblist_filename.c_str(), ios::out );
      if (!outputfile) {
        Error(0,"%s: Unable to open output lib file %s\n",
              argv[0], liblist_filename.c_str());
      } else outputfile << gLibsNeeded << endl;
   }

   G__setglobalcomp(-1);  // G__CPPLINK
   CleanupOnExit(0);
   G__exit(0);

   return 0;
}
