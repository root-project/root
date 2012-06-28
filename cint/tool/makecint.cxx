/* /% C %/ */
/***********************************************************************
 * makecint (C/C++ interpreter-compiler)
 ************************************************************************
 * Source file makecint.c
 ************************************************************************
 * Description:
 *  This tool creates Makefile for encapsurating arbitrary C/C++ object
 * into Cint as Dynamic Link Library or archived library
 ************************************************************************
 * Copyright(c) 1995~2010  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#ifdef G__ROOT
#include "RConfigure.h"
#define G__EXTRA_TOPDIR "cint"
#else
#define G__EXTRA_TOPDIR ""
#endif
#define G__CINT_LIBNAME "Cint"
#if defined(G__HAVE_CONFIG)
#include "configcint.h"
#endif

#ifndef EXIT_FAILURE
#define EXIT_FAILURE 1
#endif
#ifndef EXIT_SUCCESS
#define EXIT_SUCCESS 0
#endif

#include <string>
#include <iostream>
#include <fstream>
#include <list>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

std::string G__DLLID;
std::string G__MACRO;
std::string G__IPATH;
std::string G__CHDR;
std::string G__CXXHDR;
std::string G__LIB;
std::string G__COMPFLAGS;
std::string G__CIOPT;
std::string G__CSTUB;
std::string G__CXXSTUB;
std::string G__CSTUBCINT;
std::string G__CXXSTUBCINT;
std::string G__INITFUNC;
std::string G__preprocess;
std::string G__makefile;
std::string G__object;
bool G__quiet = false;

enum G__C_OR_CXX { G__unknown_MODE, G__C_MODE, G__CXX_MODE };

enum G__SOMEFLAGS { G__ismain = 1,
                    G__isDLL = 2,
                    G__genReflexCode = 4 };

int G__flags; // bit pattern of G__SOMEFLAGS

enum G__MODE { G__IDLE, G__CHEADER, G__CSOURCE, G__CXXHEADER, G__CXXSOURCE
	     , G__LIBRARY , G__CSTUBFILE , G__CXXSTUBFILE
	     , G__COMPILEROPT, G__CINTOPT
};

class G__SourceFile {
public:
  G__SourceFile(const std::string& source, G__C_OR_CXX mode=G__unknown_MODE);
  const std::string& GetSource() const { return fSource; }
  const std::string& GetObject() const { return fObject; }
  bool               IsCxx()     const { return fCOrCxx==G__CXX_MODE; }
  operator bool() { return !fObject.empty() && fCOrCxx!=G__unknown_MODE; }

  static const std::string& GetAllObjects(G__C_OR_CXX mode) {
    return mode==G__C_MODE?fgAllCObjects:fgAllCxxObjects; }

private:
  std::string fSource;
  std::string fObject;
  G__C_OR_CXX fCOrCxx;
  static std::string fgAllCObjects;
  static std::string fgAllCxxObjects;
};

G__SourceFile::G__SourceFile(const std::string& source, G__C_OR_CXX mode): fSource(source) {
  size_t extpos = source.find('.');
  if (extpos == std::string::npos) {
    std::cerr << "ERROR in G__SourceFile::G__SourceFile: source " << source << " has no extension!" << std::endl;
    return;
  }
  std::string ext = source.substr(extpos+1);
  std::string extup = ext;
  for (size_t i = 0; i < extup.length(); ++i)
    if (extup[i] >= 'a' && extup[i] <= 'z') extup[i] += 'A'-'a';
  if (extup != "C" && extup != "CC" && extup != "CXX" && extup != "CPP") {
    std::cerr << "ERROR in G__SourceFile::G__SourceFile: source " << source
              << " has unrecognized extension " << ext << "!" << std::endl;
    return;
  }
  if (mode==G__unknown_MODE)
    if (ext=="c") mode=G__C_MODE; // not extup - .C can be C++!
    else mode=G__CXX_MODE;
  else if ((ext == "c") != (mode == G__C_MODE))
    std::cerr << "WARNING in G__SourceFile::G__SourceFile: " <<  (mode==G__C_MODE?"C":"CXX")
              << " mode does not match source file " << source << std::endl;

  fObject=source.substr(0, extpos);
  fObject+=G__CFG_OBJEXT;
  fCOrCxx=mode;
  if (mode==G__C_MODE)
    fgAllCObjects+=fObject+" ";
  else
    fgAllCxxObjects+=fObject+" ";
}

std::string G__SourceFile::fgAllCObjects;
std::string G__SourceFile::fgAllCxxObjects;

std::list<G__SourceFile> G__SOURCEFILES;

/****************************************************************
* G__printtitle()
****************************************************************/
void G__printtitle()
{
  printf("##########################################################################\n");
#if defined(G__DJGPP)
  printf("# makecint : interpreter-compiler for cint (MS-DOS DJGPP version)\n");
#elif defined(G__CYGWIN)
  printf("# makecint : interpreter-compiler for cint (Windows Cygwin DLL version)\n");
#elif defined(G__MINGW)
  printf("# makecint : interpreter-compiler for cint (Windows Mingw version)\n");
#elif defined(G__WIN32)
  printf("# makecint : interpreter-compiler for cint (Windows VisualC++ version)\n");
#else
  printf("# makecint : interpreter-compiler for cint (UNIX version)\n");
#endif
  printf("# Copyright(c) 1995~2010 Masaharu Goto. Mailing list: root-cint@cern.ch\n");
  printf("##########################################################################\n");
}

/****************************************************************
* G__displayhelp()
****************************************************************/
void G__displayhelp()
{
  printf("Usage :\n");
  printf(" makecint -mk [Makefile] -o [Object] -H [C++header] -C++ [C++source]\n");
  printf("          <-m> <-p>      -dl [DLL]   -h [Cheader]   -C   [Csource]\n");
  printf("                          -l [Lib] -i [StubC] -i++ [StubC++]\n");
  printf("  -o [obj]      :Object name\n");
  printf("  -dl [dynlib]  :Generate dynamic link library object\n");
  printf("  -mk [mkfile]  :Create makefile (no actual compilation)\n");
  printf("  -p            :Use preprocessor for header files\n");
  printf("  -m            :Needed if main() is included in the source file\n");
  printf("  -D [macro]    :Define macro\n");
  printf("  -I [incldpath]:Set Include file search path\n");
  printf("  -H [sut].h    :C++ header as parameter information file\n");
  printf("  -h [sut].h    :C header as parameter information file\n");
  printf("    +P          :Turn on preprocessor mode for following header files\n");
  printf("    -P          :Turn off preprocessor mode for following header files\n");
  printf("    +V          :Turn on class title loading for following header files\n");
  printf("    -V          :Turn off class title loading for following header files\n");
  printf("  -C++ [sut].C  :Link C++ object. Not accessed unless -H [sut].h is given\n");
  printf("  -C [sut].c    :Link C object. Not accessed unless -h [sut].h is given\n");
  printf("  -i++ [stub].h :C++ STUB function parameter information file\n");
  printf("  -i [stub].h   :C STUB function parameter information file\n");
  printf("  -c [sut].c    :Same as '-h [sut].c -C [sut].c'\n");
  printf("  -l -l[lib]    :Compiled object, Library or linker options\n");
#ifndef G__OLDIMPLEMENTATION1452
  printf("  -u [file]     :Generate dummy class for undefined typename\n");
  printf("  -U [dir]      :Directory to disable interface method generation\n");
  printf("  -Y [0|1]      :Ignore std namespace (default=1:ignore)\n");
  printf("  -Z [0|1]      :Automatic loading of standard header files\n");
#endif
  printf("  -cc   [opt]   :Compiler option\n");
  printf("  -cint [opt]   :Cint option\n");
  printf("  -B [funcname] :Initialization function name\n");
  printf("  -y [LIBNAME]  :Name of CINT core DLL, LIBCINT or WILDC(WinNT/95 only)\n");
  printf("  -q            :quiet, reduce output to warnings, errors\n");
}

/****************************************************************
* G__displaytodo()
****************************************************************/
void G__displaytodo()
{
  printf("Run 'make -f %s' to compile the object\n",G__makefile.c_str());
}

/******************************************************************
Print a object file target to out
******************************************************************/
void G__printsourcecompile(std::ostream& out)
{
  for (std::list<G__SourceFile>::iterator iSource=G__SOURCEFILES.begin();
       iSource!=G__SOURCEFILES.end(); ++iSource) {
    if (!*iSource) {
      std::cerr << "Warning in G__printsourcecompile: source "
                << iSource->GetSource() << "not initialized." << std::endl;
      continue;
    }
    out << iSource->GetObject() << ": " << iSource->GetSource();
    if (!iSource->IsCxx())
      out << " $(CHEADER)" << std::endl
          << "\t $(CC) $(IPATH) $(CMACRO) " << G__CFG_CFLAGS;
    else
      out << " $(CXXHEADER)" << std::endl
          << "\t $(CXX) $(IPATH) $(CXXMACRO) " << G__CFG_CXXFLAGS;

    out << " " << G__CFG_COUT << iSource->GetObject() << " "
        << G__CFG_COMP << " " << iSource->GetSource() << std::endl;
  }
}

/******************************************************************
* G__readargument
******************************************************************/
void G__readargument(int argc, char **argv)
{
  G__MODE mode=G__IDLE;
  G__flags = 0;
  const std::string space(" ");
  int i=1;
  while(i<argc) {
    /*************************************************************************
    * options with no argument
    *************************************************************************/
    if(strcmp(argv[i],"-p")==0) {
      G__preprocess=argv[i];
      mode = G__IDLE;
    }
#ifndef G__OLDIMPLEMENTATION1452
    /*************************************************************************/
    else if(strcmp(argv[i],"-u") == 0 ||
            strcmp(argv[i],"-U") == 0 ||
            strcmp(argv[i],"-Y") == 0 ||
            strcmp(argv[i],"-Z") == 0) {
      G__CIOPT += space + argv[i];
      G__CIOPT += argv[++i];
      mode = G__IDLE;
    }
    else if(strncmp(argv[i],"-u",2) == 0 ||
            strncmp(argv[i],"-U",2) == 0 ||
            strncmp(argv[i],"-Y",2) == 0 ||
            strncmp(argv[i],"-Z",2) == 0) {
      G__CIOPT += space + argv[i];
      mode = G__IDLE;
    }
#endif
    /*************************************************************************/
    else if(strcmp(argv[i],"-m")==0) {
      G__flags |= G__ismain;
      mode = G__IDLE;
    }
    /*************************************************************************/
    else if(strcmp(argv[i],"-?")==0) {
      G__displayhelp();
      mode = G__IDLE;
      exit(EXIT_SUCCESS);
    }
    /*************************************************************************/
    else if((strcmp(argv[i],"--reflex") == 0) || strcmp(argv[i],"-r") == 0) {
      G__flags |= G__genReflexCode;
      mode = G__IDLE;
    }
    /*************************************************************************
    * options with 1 argument
    *************************************************************************/
    else if(strcmp(argv[i],"-D") == 0) {
      G__MACRO += space + argv[i];
      G__MACRO += argv[++i];
      mode = G__IDLE;
    }
    else if(strncmp(argv[i],"-D",2)==0
            && G__COMPILEROPT!=mode && G__CINTOPT!=mode
            ) {
      G__MACRO += space + argv[i];
      mode = G__IDLE;
    }
    /*************************************************************************/
    else if(strcmp(argv[i],"-I")==0) {
      G__IPATH += space + argv[i];
      G__IPATH += argv[++i];
      mode = G__IDLE;
    }
    else if(strncmp(argv[i],"-I",2)==0
            && G__COMPILEROPT!=mode && G__CINTOPT!=mode
            ) {
      G__IPATH += space + argv[i];
      mode = G__IDLE;
    }
    /*************************************************************************/
    else if(strcmp(argv[i],"-B")==0) {
      G__INITFUNC = "-B";
      G__INITFUNC += argv[++i];
    }
    else if(strcmp(argv[i],"-y")==0) {
      ++i;
      /* WinNT/95 only  - ignored! */
    }
    /*************************************************************************/
    else if(strcmp(argv[i],"-o")==0) {
      G__object=argv[++i];
      size_t pos_slash = G__object.rfind('/');
      size_t pos_backslash = G__object.rfind('\\');
      size_t posfile=pos_slash;
      if (posfile == std::string::npos
          || (pos_backslash != std::string::npos && pos_backslash > pos_slash))
        posfile = pos_backslash;

      if (posfile == std::string::npos) posfile=0;
      else ++posfile;
      G__DLLID=G__object.substr(posfile);
      size_t posext=G__DLLID.rfind('.');
      if (posext!=std::string::npos)
        G__DLLID.erase(posext);
      G__flags |= !G__isDLL;
      mode = G__IDLE;
#ifdef G__DJGPP
      if(G__object.find(".EXE")==std::string::npos &&
         G__object.find(".exe")==std::string::npos) {
        G__object+=".exe";
      }
#endif
    }
    /*************************************************************************/
    else if(strcmp(argv[i],"-dl")==0 || strcmp(argv[i],"-sl")==0 ||
	    strcmp(argv[i],"-dll")==0 || strcmp(argv[i],"-DLL")==0 ||
	    strcmp(argv[i],"-so")==0) {
      G__object = argv[++i];
      size_t pos_slash = G__object.rfind('/');
      size_t pos_backslash = G__object.rfind('\\');
      size_t posfile=pos_slash;
      if (posfile == std::string::npos
          || (pos_backslash != std::string::npos && pos_backslash > pos_slash))
        posfile = pos_backslash;

      if (posfile == std::string::npos) posfile=0;
      else ++posfile;
      G__DLLID=G__object.substr(posfile);
      size_t posext=G__DLLID.rfind('.');
      if (posext!=std::string::npos)
        G__DLLID.erase(posext);
      G__flags |= G__isDLL;
      mode = G__IDLE;
    }
    /*************************************************************************/
    else if(strcmp(argv[i],"-mk")==0) {
      G__makefile = argv[++i];
      mode = G__IDLE;
    }
    /*************************************************************************
    * options with multiple argument
    *************************************************************************/
    else if(strcmp(argv[i],"-h")==0) {
      mode = G__CHEADER;
    }
    else if(strcmp(argv[i],"-H")==0) {
      mode = G__CXXHEADER;
    }
    else if(strcmp(argv[i],"-C")==0) {
      mode = G__CSOURCE;
    }
    else if(strcmp(argv[i],"-C++")==0) {
      mode = G__CXXSOURCE;
    }
    else if(strcmp(argv[i],"-l")==0) {
      mode = G__LIBRARY;
    }
    else if(strcmp(argv[i],"-cc")==0) {
      mode = G__COMPILEROPT;
    }
    else if(strcmp(argv[i],"-cint")==0) {
      mode = G__CINTOPT;
    }
    else if(strcmp(argv[i],"-i")==0) {
      mode = G__CSTUBFILE;
    }
    else if(strcmp(argv[i],"-i++")==0) {
      mode = G__CXXSTUBFILE;
    }
    else if(strcmp(argv[i],"-c")==0) {
      /* fprintf(stderr,"makecint: -c being obsoleted. no guarantee\n"); */
      mode = G__CHEADER;
    } else if(strcmp(argv[i],"-q")==0) {
      G__quiet = true;
    }
    /*************************************************************************/
    else {
      switch(mode) {
      case G__CHEADER:
	G__CHDR += space + argv[i];
	break;
      case G__CSOURCE:
	G__SOURCEFILES.push_back(G__SourceFile(argv[i], G__C_MODE));
	break;
      case G__CXXHEADER:
	G__CXXHDR += space + argv[i];
	break;
      case G__CXXSOURCE:
	G__SOURCEFILES.push_back(G__SourceFile(argv[i], G__CXX_MODE));
	break;
      case G__LIBRARY:
	G__LIB += space + argv[i];
	break;
      case G__COMPILEROPT:
	G__COMPFLAGS += space + argv[i];
	break;
      case G__CINTOPT:
	G__CIOPT += space + argv[i];
	break;
      case G__CSTUBFILE:
	G__CSTUB += space + argv[i];
	break;
      case G__CXXSTUBFILE:
	G__CXXSTUB += space + argv[i];
	break;
      case G__IDLE:
      default:
	break;
      } // switch mode
    } // if not a flag
    ++i;
  } // while args
}

void G__remove_cintopts(std::string& opts) {
  size_t pos=0;
  while (std::string::npos != (pos=opts.find("+P")))
    opts.erase(pos,2);
  while (std::string::npos != (pos=opts.find("-P")))
    opts.erase(pos,2);
  while (std::string::npos != (pos=opts.find("+V")))
    opts.erase(pos,2);
  while (std::string::npos != (pos=opts.find("-V")))
    opts.erase(pos,2);
}

/******************************************************************
* G__check
******************************************************************/
int G__check(const std::string& what, const char *name, const char *where)
{
  if(what.empty()) {
    std::cerr << "Error: " << name << " must be set " << where << std::endl;
    return(1);
  }
  return(0);
}

/******************************************************************
* G__checksetup
******************************************************************/
int G__checksetup()
{
  int error=0;
  if(G__flags & G__isDLL) {
    error+=G__check(G__object,"'-dl [DLL]'","in the command line");
  }
  else {
    error+=G__check(G__object,"'-o [Object]'","in the command line");
  }
  error+=G__check(G__makefile,"'-mk [Makefile]'","in the command line");
  return(error);
}

/******************************************************************
* G__outputmakefile
******************************************************************/
void G__outputmakefile(int argc,char **argv)
{
  int i;
  int minusCOption = 1;  // GOHERE //

  std::ofstream out(G__makefile.c_str());
  if(!out) {
    std::cerr << "Error: can not create " << G__makefile << std::endl;
    exit(EXIT_FAILURE);
  }

  out << "############################################################" << std::endl
      << "# Automatically created makefile for " << G__object << std::endl
      << "############################################################" << std::endl
      << std::endl;

  std::string cintexCompFlags;
  if (G__flags & G__genReflexCode) cintexCompFlags = "-DCINTEX";

  /***************************************************************************
   * Print out variables
   ***************************************************************************/
  // check if the variable __CINT_BUILDDIR is set. This means that we are
  // within an initial cint build, and cint is not yet installed, i.e.
  // cint system directories should be taken from the build directory
  // instead of the final destination
  char *builddir = getenv("__CINT_BUILDDIR");

  out << "# Set variables ############################################" << std::endl
      << "CXX         := " << G__CFG_CXX << std::endl
      << "CC          := " << G__CFG_CC << std::endl
      << "LD          := " << G__CFG_LD << std::endl
      << "CINT        := $(shell which cint" << G__CFG_EXEEXT << ")" << std::endl;
  // changes start ---------------------------------------------
  //<< "CINTSYSDIRU := $(patsubst %/bin/,%/,$(dir $(CINT)))" << std::endl
  //<< "CINTSYSDIRW := $(shell " << G__CFG_MANGLEPATHS << " $(CINTSYSDIRU) )" << std::endl;
  // changes end -----------------------------------------------

  // if G__CFG_INCLUDEDIRCINT is set, this means that ./configure --with-prefix
  // was called, i.e. a standard linux install is going to be done.
#ifdef G__CFG_INCLUDEDIRCINT
  if(builddir)
  {
      // initial build, take includedir from the builddir (temporarily)
      out << "CINTINCDIRU := $(shell " << G__CFG_MANGLEPATHSU << " " << builddir << "/" << G__EXTRA_TOPDIR << "/"
	  << G__CFG_COREVERSION << "/inc )" << std::endl
	  << "CINTINCDIRW := " << builddir << "/" << G__EXTRA_TOPDIR << "/"
	  << G__CFG_COREVERSION << "/inc" << std::endl;
  }
  else
  {
      // already the installed version is running. Take the includedir
      // using cint-config

      // changes start ------------------------------------------

      // old code
      // out << "CINTINCDIRU := " << G__CFG_INCLUDEDIRCINT << std::endl
      // << "CINTINCDIRW := " << G__CFG_INCLUDEDIRCINT << std::endl;

      // new code
      out << "CINTINCDIRU := $(shell cint-config --incdir)" << std::endl;
      out << "CINTINCDIRW := $(shell " << G__CFG_MANGLEPATHS << " $(CINTINCDIRU) )" << std::endl;

      //???
      // The old code was buggy??? CINTINCDIRW should have been set
      // with G__CFG_MANGLEPATHS as well? Probably nobody installed
      // cint on windows/cygwin via ./configure --with-prefix ???

      // changes end --------------------------------------------
  }

  // In the other case, the compile-and-leave-in-place policy was used
  // i.e. every cint system directory is within the toplevel CINTSYSDIR
#else

  // changes start ------------------------------------------

  // old code
  //out << "CINTINCDIRU := $(CINTSYSDIRU)" << G__EXTRA_TOPDIR << "/"
  //<< G__CFG_COREVERSION << "/inc" << std::endl
  //<< "CINTINCDIRW := $(CINTSYSDIRW)" << G__EXTRA_TOPDIR << "/"
  //<< G__CFG_COREVERSION << "/inc" << std::endl;

  // new code
  out << "CINTINCDIRU := $(shell cint-config --unix --incdir)" << std::endl;
  out << "CINTINCDIRW := $(shell " << G__CFG_MANGLEPATHS << " $(CINTINCDIRU) )" << std::endl;

  // changes end --------------------------------------------


#endif // G__CFG_INCLUDEDIRCINT

#ifdef G__CFG_LIBDIR
  if(builddir)
  {
#if defined(G__WIN32)
      out << "CINTLIB     := $(shell " << G__CFG_MANGLEPATHSU << " " << builddir << "/bin/lib" << G__CINT_LIBNAME << G__CFG_SOEXT << " )\n";
#else
      out << "CINTLIB     := " << builddir << "/lib/lib" << G__CINT_LIBNAME << G__CFG_SOEXT << std::endl;
#endif
  }
  else
  {
      // changes start ------------------------------------------

      // old code
      //out << "CINTLIB     := " << G__CFG_LIBDIR << "/lib" << G__CINT_LIBNAME << G__CFG_SOEXT << std::endl;      

      // new code
#if defined(G__WIN32)
      out << "CINTLIB     := $(shell cint-config --unix --bindir)/lib" << G__CINT_LIBNAME << G__CFG_SOEXT << std::endl;
#else
      out << "CINTLIB     := $(shell cint-config --unix --libdir)/lib" << G__CINT_LIBNAME << G__CFG_SOEXT << std::endl;
#endif
      // changes end --------------------------------------------
      
  }
#else

  // changes start ------------------------------------------

  // old code
  //out << "CINTLIB     := $(CINTSYSDIRU)/lib/lib" << G__CINT_LIBNAME << G__CFG_SOEXT << std::endl;

  // new code
#if defined(G__WIN32)
  out << "CINTLIB     := $(shell cint-config --unix --bindir)/lib" << G__CINT_LIBNAME << G__CFG_SOEXT << std::endl;
#else
  out << "CINTLIB     := $(shell cint-config --unix --libdir)/lib" << G__CINT_LIBNAME << G__CFG_SOEXT << std::endl;
#endif

  // changes end --------------------------------------------
  
#endif

  if (!strcmp(G__CFG_COREVERSION,"cint7"))
    out << "CINTLIB     := $(CINTLIB) $(subst lib" << G__CINT_LIBNAME << ",libReflex,$(CINTLIB))" << std::endl;
  out << "IPATH       := " << G__IPATH;
  out << std::endl;

  out << "CMACRO      := " << G__CFG_CMACROS << " " << G__MACRO << std::endl
      << "CXXMACRO    := " << G__CFG_CXXMACROS << " " << G__MACRO << std::endl
      << "CFLAGS      := " << G__CFG_CFLAGS << " "
      << G__COMPFLAGS << " " << cintexCompFlags << std::endl
      << "CXXFLAGS    := " << G__CFG_CXXFLAGS << " " 
      << G__COMPFLAGS << " " << cintexCompFlags << std::endl;
  out << "CINTIPATH   := "<< G__CFG_INCP << "$(CINTINCDIRW)" << std::endl
      << "OBJECT      := " << G__object << std::endl

      << "LINKSPEC    := ";
  if(!G__CHDR.empty()) out << " -DG__CLINK_ON";
  if(!G__CXXHDR.empty()) out << " -DG__CXXLINK_ON";
  out << std::endl << std::endl;

  out << "# Set File names ###########################################" << std::endl;

  if(!G__CHDR.empty()) {
    out << "CIFC        := G__c_" << G__DLLID << ".c" << std::endl
        << "CIFH        := G__c_" << G__DLLID << ".h" << std::endl
        << "CIFO        := G__c_" << G__DLLID << G__CFG_OBJEXT << std::endl;
  }
  else {
    out << "CIFC        :=" << std::endl
        << "CIFH        :=" << std::endl
        << "CIFO        :=" << std::endl;
  }
  if(!G__CXXHDR.empty()) {
    out << "CXXIFC      := G__cpp_" << G__DLLID << ".cxx" << std::endl
        << "CXXIFH      := G__cpp_" << G__DLLID << ".h" << std::endl
        << "CXXIFO      := G__cpp_" << G__DLLID << G__CFG_OBJEXT << std::endl;
  }
  else {
    out << "CXXIFC      :=" << std::endl
        << "CXXIFH      :=" << std::endl
        << "CXXIFO      :=" << std::endl;
  }
  out << std::endl;

  out << "LIBS        := ";
  // changes start ----------------------------------------------
  // old code
  //out << G__CFG_LIBP << "$(CINTSYSDIRW)/lib $(subst @imp@," << G__CINT_LIBNAME << "," << G__CFG_LIBL << ") ";
  // new code
  if(builddir)
  {
     out << G__CFG_LIBP << "\"" << builddir << "/lib\" $(subst @imp@," << G__CINT_LIBNAME << "," << G__CFG_LIBL << ") ";
  }
  else
  {
      out << G__CFG_LIBP << "\"$(shell cint-config --libdir)\" $(subst @imp@," << G__CINT_LIBNAME << "," << G__CFG_LIBL << ") ";
  }
  // changes end ------------------------------------------------
  if (!strcmp(G__CFG_COREVERSION,"cint7"))
    out << " $(subst @imp@,Reflex," << G__CFG_LIBL << ") ";
  out << G__CFG_DEFAULTLIBS << " " << G__LIB << " " << std::endl
      << std::endl;

  out << "CINTOPT     := " << G__CIOPT << std::endl
      << "COFILES     := " << G__SourceFile::GetAllObjects(G__C_MODE) 
      << std::endl << std::endl;

  std::string without_cintopts(G__CHDR);
  G__remove_cintopts(without_cintopts);
  out << "CHEADER     := " << without_cintopts << std::endl
      << "CHEADERCINT := " << G__CHDR << std::endl;

  without_cintopts=G__CSTUB;
  G__remove_cintopts(without_cintopts);
  out << "CSTUB       := " << without_cintopts << std::endl
      << "CSTUBCINT   := " << G__CSTUB << std::endl
      << std::endl;

  out << "CXXOFILES   := " << G__SourceFile::GetAllObjects(G__CXX_MODE) 
      << std::endl << std::endl;

  without_cintopts=G__CXXHDR;
  G__remove_cintopts(without_cintopts);
  out << "CXXHEADER   := " << without_cintopts << std::endl
      << "CXXHEADERCINT := " << G__CXXHDR << std::endl
      << std::endl;

  without_cintopts=G__CXXSTUB;
  G__remove_cintopts(without_cintopts);
  out << "CXXSTUB     := " << without_cintopts << std::endl
      << "CXXSTUBCINT := " << G__CXXSTUB << std::endl
      << std::endl;

# ifdef G__CFG_DATADIRCINT
  std::string maindiru(G__CFG_DATADIRCINT);
  std::string maindirw(G__CFG_DATADIRCINT);
# else
  // changes start -------------------------------------------------
  // old code
  //std::string maindiru("$(CINTSYSDIRU)");
  //std::string maindirw("$(CINTSYSDIRW)");
  // new code
  std::string maindiru = "$(patsubst %/bin/,%/,$(dir $(CINT)))";
  std::string maindirw =
        std::string("$(shell ") + std::string(G__CFG_MANGLEPATHS)
      + std::string(" $(patsubst %/bin/,%/,$(dir $(shell which cint") 
      + std::string(G__CFG_EXEEXT) + std::string("))))");
# endif
  std::string maindir2("/");
  maindir2 += G__CFG_COREVERSION;
  maindir2 += "/main/";
  out << "MAINDIRU    := " << maindiru << maindir2 << std::endl;
  out << "MAINDIRW    := " << maindirw << maindir2 << std::endl;

#if !defined(G__CFG_EXPLLINK)
#define G__CFG_EXPLLINK 0
#endif
  if (!G__CFG_EXPLLINK || !strlen(G__CFG_EXPLLINK)) {
    // if the libs are not linked against readline then the executable needs to
    out << "READLINEA   := "
#ifdef G__CFG_READLINELIB
        << G__CFG_READLINELIB << " "
#endif
#ifdef G__CFG_CURSESLIB
        << G__CFG_CURSESLIB
#endif
        << std::endl;
  }

  /***************************************************************************
   * Link Object
   ***************************************************************************/
  out << "# Link Object #############################################" << std::endl;
  if(G__flags & G__isDLL) {
    out << "$(OBJECT) : $(CINTLIB) $(COFILES) $(CXXOFILES) $(CIFO) $(CXXIFO)"
        << std::endl
        << "\t";
    if (G__quiet) out << "@";
    out << "$(LD) $(subst @so@,$(OBJECT:" << G__CFG_SOEXT << "=),"
        << G__CFG_SOFLAGS << ") "<< G__CFG_SOOUT
        << "$(OBJECT) $(COFILES) $(CIFO) $(CXXIFO) $(CXXOFILES) $(LIBS)"
        << std::endl;
  }
  else if(G__flags & G__ismain) {
#ifdef _AIX
TODO!
  cout << "$(OBJECT) : $(CINTLIB) $(READLINEA) $(DLFCN) G__setup" << G__CFG_OBJEXT << " $(COFILES) $(CXXOFILES) $(CIFO) $(CXXIFO)";
 out << "\t";
   if (G__quiet) out << "@";
   out << "$(LD) $(IPATH) $(MACRO) $(CCOPT) -o $(OBJECT) $(CIFO) $(CXXIFO) $(COFILES) $(CXXOFILES) G__setup" << G__CFG_OBJEXT << " $(READLINEA) $(DLFCN) $(LIBS)" 
     << std::endl;
#else
  out << "$(OBJECT) : $(CINTLIB) $(READLINEA) G__setup" << G__CFG_OBJEXT
      << " $(COFILES) $(CXXOFILES) $(CIFO) $(CXXIFO)" << std::endl
      << "\t";
   if (G__quiet) out << "@";
   out << "$(LD) " << G__CFG_LDFLAGS << " $(CCOPT) " 
      << G__CFG_LDOUT << "$(OBJECT) $(CIFO) $(CXXIFO) $(COFILES) $(CXXOFILES) G__setup" << G__CFG_OBJEXT << " $(LIBS) $(READLINEA)"  << std::endl;
#endif
  }
  else {
#ifdef _AIX
    TODO!
    out << "$(OBJECT) : G__main" << G__CFG_OBJEXT << " $(CINTLIB) $(READLINEA) $(DLFCN) G__setup" << G__CFG_OBJEXT << " $(COFILES) $(CXXOFILES) $(CIFO) $(CXXIFO) \n";
    out << "\trm -f shr.o $(OBJECT).nm $(OBJECT).exp\n";
    out << "\t$(NM) G__main" << G__CFG_OBJEXT << " $(CIFO) $(CXXIFO) $(COFILES) $(CXXOFILES) G__setup" << G__CFG_OBJEXT << " $(READLINEA) $(DLFCN) $(LIBS) $(NMOPT)\n";
    out << "\trm -f shr.o\n";
    out << "\techo \"#!\" > $(OBJECT).exp ; cat $(OBJECT).nm >> $(OBJECT).exp\n";
    out << "\trm -f $(OBJECT).nm\n";
    out << "\t$(LD) -bE:$(OBJECT).exp -bM:SRE  $(IPATH) $(MACRO) $(CCOPT) -o $(OBJECT) G__main" << G__CFG_OBJEXT << " $(CIFO) $(CXXIFO) $(COFILES) $(CXXOFILES) G__setup" << G__CFG_OBJEXT << " $(READLINEA) $(DLFCN) $(LIBS)\n";
#else
    out << "$(OBJECT) : G__main" << G__CFG_OBJEXT << " $(CINTLIB) $(READLINEA) G__setup" << G__CFG_OBJEXT << " "
        << " $(COFILES) $(CXXOFILES) $(CIFO) $(CXXIFO)" << std::endl
        << "\t";
   if (G__quiet) out << "@";
   out << "$(LD) $(CCOPT) " << G__CFG_LDFLAGS << " " 
        << G__CFG_LDOUT << "$(OBJECT) G__main" << G__CFG_OBJEXT << " $(CIFO) $(CXXIFO) $(COFILES) $(CXXOFILES) G__setup" << G__CFG_OBJEXT << " $(LIBS) $(READLINEA)" << std::endl;
#endif
  }
  out << std::endl;

  /***************************************************************************
   * Compile user source
   ***************************************************************************/
  out << "# Compile User source files ##############################" << std::endl;
  G__printsourcecompile(out);
  out << std::endl;
    
  /***************************************************************************
   * Compile Initialization routine
   ***************************************************************************/
  out << "# Compile main function  #################################" << std::endl;
  out << "G__main" << G__CFG_OBJEXT << ": ";
#if defined(G__CYGWIN) || defined(_MSC_VER) || \
  defined(__BORLANDC__) || defined(__BCPLUSPLUS__) || defined(G__BORLANDCC5)
  out << "G__main.cxx" << std::endl;
#else
  out << "$(MAINDIRU)/G__main.c" << std::endl;
#endif
  out << "\t";
  if (G__quiet) out << "@";
#if defined(G__CYGWIN) || defined(_MSC_VER) || \
  defined(__BORLANDC__) || defined(__BCPLUSPLUS__) || defined(G__BORLANDCC5)
  out << "$(CXX) $(CXXMACRO) $(CXXFLAGS) $(CCOPT) "
#else
  out << "$(CC)  $(CMACRO) $(CFLAGS) $(CCOPT) "
#endif
      << "$(LINKSPEC) $(CINTIPATH) "
      << G__CFG_COUT << "$@ -c $<\n" << std::endl;

  out << "# Compile dictionary setup routine #######################" << std::endl;
  out << "G__setup" << G__CFG_OBJEXT << ": $(MAINDIRU)/G__setup.c $(CINTINCDIRU)/G__ci.h" << std::endl
      << "\t";
  if (G__quiet) out << "@";
  out << "$(CC) $(LINKSPEC) $(CINTIPATH) $(CMACRO) $(CFLAGS) "
      << G__CFG_COUT << "$@ " 
      << G__CFG_COMP << " $(MAINDIRW)/G__setup.c\n" << std::endl;

  /***************************************************************************
   * Interface routine
   ***************************************************************************/
  if(!G__CHDR.empty()) {
    out << "# Compile C Interface routine ############################" << std::endl
        << "$(CIFO) : $(CIFC)" << std::endl
        << "\t";
    if (G__quiet) out << "@";
    out << "$(CC) $(CINTIPATH) $(IPATH) $(CMACRO) $(CFLAGS) $(CCOPT) "
        << G__CFG_COMP << " $(CIFC)" << std::endl
        << std::endl
        << "# Create C Interface routine #############################" << std::endl

    // changes start ---------------------------------------------------------

	// old code
        //<< "$(CIFC) : $(CHEADER) $(CSTUB) $(CINTSYSDIRU)/cint" << G__CFG_EXEEXT << std::endl;

	// new code
        <<   "$(CIFC) : $(CHEADER) $(CSTUB) $(CINT)" << std::endl;

    // changes end ------------------------------------------------------------

    /* Following line needs explanation. -K is used at the beginning and
     * later again $(KRMODE) may be set to -K. When -K is given after -c-2
     * it will set G__clock flags so that it will create K&R compatible
     * function headers. This is not a good manner but -K -c-2 and -c-2 -K
     * has different meaning. */
    out << "\t";
    if (G__quiet) out << "@";
    out << "$(CINT) " << G__INITFUNC << " -K -w" << (int)(G__flags & G__isDLL) << " -z" << G__DLLID 
        << " -n$(CIFC) $(DLLSPEC) -D__MAKECINT__ -DG__MAKECINT $(CMACRO) "
        << G__preprocess
        << " -c-2 $(KRMODE) $(CINTIPATH) $(IPATH) $(CMACRO) $(CINTOPT) $(CHEADERCINT)" ;
    if(!G__CSTUB.empty()) out << " +STUB $(CSTUBCINT) -STUB";
  }
  if(!G__CXXHDR.empty()) {
    out << "# Compile C++ Interface routine ##########################" << std::endl
        << "$(CXXIFO) : $(CXXIFC)" << std::endl
        << "\t";
   if (G__quiet) out << "@";
   out << "$(CXX) $(CINTIPATH) $(IPATH) $(CXXMACRO) $(CXXFLAGS) $(CCOPT) "
        << G__CFG_COMP <<" $(CXXIFC)" << std::endl
        << std::endl
        << "# Create C++ Interface routine ###########################" << std::endl
        << "$(CXXIFC) : $(CXXHEADER) $(CXXSTUB) $(CINT)" << std::endl;

    if (G__flags & G__genReflexCode) minusCOption = 3;
    out << "\t";
   if (G__quiet) out << "@";
   out << "$(CINT) " << G__INITFUNC << " -w" << (int)(G__flags & G__isDLL) << " -z" << G__DLLID 
        << " -n$(CXXIFC) $(DLLSPEC) -D__MAKECINT__ -DG__MAKECINT "
        << G__preprocess << " -c-" << minusCOption
        << " -A $(CINTIPATH) $(IPATH) $(CXXMACRO) $(CINTOPT) $(CXXHEADERCINT)";
    if(!G__CXXSTUB.empty())
      out << " +STUB $(CXXSTUBCINT) -STUB";
  }
  out << std::endl
      << std::endl;

  out << "# Clean up #################################################\n" << std::endl;
  out << "clean :" << std::endl;
  if(G__flags & G__isDLL) {
    out << "\t";
   if (G__quiet) out << "@";
   out << "$(RM) $(OBJECT) core $(CIFO) $(CIFC) $(CIFH) $(CXXIFO) $(CXXIFC) $(CXXIFH) $(RMCOFILES) $(RMCXXOFILES)" << std::endl;
  }
  else {
#ifdef _AIX
    out << "\t";
   if (G__quiet) out << "@";
   out << "$(RM) $(OBJECT) $(OBJECT).exp $(OBJECT).nm shr.o core $(CIFO) $(CIFC) $(CIFH) $(CXXIFO) $(CXXIFC) $(CXXIFH) $(COFILES) $(CXXOFILES) G__setup" << G__CFG_OBJEXT << std::endl;
#else
    out << "\t";
   if (G__quiet) out << "@";
   out << "$(RM) $(OBJECT) core $(CIFO) $(CIFC) $(CIFH) $(CXXIFO) $(CXXIFC) $(CXXIFH) $(RMCOFILES) $(RMCXXOFILES) G__main" << G__CFG_OBJEXT
        << " G__setup" << G__CFG_OBJEXT << std::endl;
#endif
  }
  out << std::endl;

  out << "# re-makecint ##############################################" << std::endl;
  out << "makecint :" << std::endl;
  out << "\tmakecint ";
  for(i=1;i<argc;i++)
    out << " " << argv[i];
  out << std::endl
      << std::endl;

}

#if defined(G__CYGWIN) || defined(_MSC_VER) || \
  defined(__BORLANDC__) || defined(__BCPLUSPLUS__) || defined(G__BORLANDCC5)
/******************************************************************
* G__outputmain()
******************************************************************/
void G__outputmain()
{
  FILE *mainfp;
  char G__DLLID[10] = "";
  /*****************************************************************
  * creating G__main.cxx
  *****************************************************************/
  mainfp = fopen("G__main.cxx","w");
  fprintf(mainfp,"/******************************************************\n");
  fprintf(mainfp,"* G__main.cxx\n");
  fprintf(mainfp,"*  automatically generated main() function for cint\n");
  fprintf(mainfp,"*  Cygwin environment\n");
  fprintf(mainfp,"******************************************************/\n");
  fprintf(mainfp,"#include <stdio.h>\n");
  fprintf(mainfp,"extern \"C\" {\n");
  fprintf(mainfp,"extern void G__setothermain(int othermain);\n");
  fprintf(mainfp,"extern int G__main(int argc,char **argv);\n");
  fprintf(mainfp,"extern void G__set_p2fsetup(void (*p2f)());\n");
  fprintf(mainfp,"extern void G__free_p2fsetup();\n");
  if(!G__CHDR.empty()) fprintf(mainfp,"extern void G__c_setup%s();\n",G__DLLID);
  if(!G__CXXHDR.empty()) fprintf(mainfp,"extern void G__cpp_setup%s();\n",G__DLLID);
  fprintf(mainfp,"}\n");
  fprintf(mainfp,"\n");
#ifndef G__OLDIMPLEMENTATION874
  if(G__flags & G__ismain) {
    fprintf(mainfp,"class G__DMYp2fsetup {\n");
    fprintf(mainfp," public:\n");
    fprintf(mainfp,"  G__DMYp2fsetup() { \n");
    if(!G__CHDR.empty()) fprintf(mainfp,"    G__set_p2fsetup(G__c_setup%s);\n",G__DLLID);
    if(!G__CXXHDR.empty()) fprintf(mainfp,"    G__set_p2fsetup(G__cpp_setup%s);\n",G__DLLID);
    fprintf(mainfp,"  }\n");
    fprintf(mainfp,"} G__DMY;\n");
  }
  else {
#endif
    fprintf(mainfp,"int main(int argc,char **argv)\n");
    fprintf(mainfp,"{\n");
    fprintf(mainfp,"  int result;\n");
    if(!G__CHDR.empty()) fprintf(mainfp,"  G__set_p2fsetup(G__c_setup%s);\n",G__DLLID);
    if(!G__CXXHDR.empty()) fprintf(mainfp,"  G__set_p2fsetup(G__cpp_setup%s);\n",G__DLLID);
    fprintf(mainfp,"  G__setothermain(0);\n");
    fprintf(mainfp,"  result=G__main(argc,argv);\n");
    fprintf(mainfp,"  G__free_p2fsetup();\n");
    fprintf(mainfp,"  return(result);\n");
    fprintf(mainfp,"}\n");
#ifndef G__OLDIMPLEMENTATION874
  }
#endif

  fclose(mainfp);
  /*****************************************************************
  * end of creating G__main.cxx
  *****************************************************************/
}
#endif

/******************************************************************
* G__makecint
******************************************************************/
int G__makecint(int argc, char **argv)
{
  G__readargument(argc,argv);
  if (!G__quiet) G__printtitle();
  if(G__checksetup()) {
    std::cerr << "!!!makecint aborted!!!  makecint -? for help\n" << std::endl;
    exit(EXIT_FAILURE);
  }
#if defined(G__CYGWIN) || defined(_MSC_VER) || \
  defined(__BORLANDC__) || defined(__BCPLUSPLUS__) || defined(G__BORLANDCC5)
  if(!(G__flags & G__isDLL)) G__outputmain();
#endif
  G__outputmakefile(argc,argv);
  if (!G__quiet) G__displaytodo();
  return(EXIT_SUCCESS);
}

/******************************************************************
* main
******************************************************************/
int main(int argc, char **argv)
{
  return(G__makecint(argc,argv));
}

/* *-*-
 * Local Variables:
 * c-tab-always-indent:nil
 * c-basic-offset:2
 * c-continued-statement-offset:2
 * c-brace-offset:-2
 * c-brace-imaginary-offset:0
 * c-argdecl-indent:0
 * c-label-offset:-2
 * compile-command:"make -k"
 * End:
 */
