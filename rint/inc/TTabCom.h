// @(#)root/rint:$Name$:$Id$
// Author: Christian Lacunza <lacunza@cdfsg6.lbl.gov>   27/04/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TTabCom
#define ROOT_TTabCom


////////////////////////////////////////////////////////////////////////////
//                                                                        //
// TTabCom                                                                //
//                                                                        //
// This class performs basic tab completion.                              //
// You should be able to hit [TAB] to complete a partially typed:         //
//                                                                        //
//   username                                                             //
//   environment variable                                                 //
//   preprocessor directive                                               //
//   pragma                                                               //
//   filename (with a context-sensitive path)                             //
//   public member function or data member                                //
//   global variable, function, or class name                             //
//                                                                        //
// Also, something like gWhatever->Func([TAB] will print the appropriate  //
// list of prototypes. For a list of some limitations see the source.     //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObjString
#include "TObjString.h"
#endif
#ifndef ROOT_TRegExp
#include "TRegexp.h"
#endif


#define MAX_LEN_PAT 1024               // maximum length of a pattern
#define dblquote(x) "\"" << x << "\""

// forward declarations
class TList;
class TListIter;
class TSeqCollection;
class TClass;

// save typing
typedef TSeqCollection TSeqCol;



class TTabCom
{
 public: // constructors
     TTabCom();

 public: // typedefs
     typedef TList     TContainer;
     typedef TListIter TContIter;

 public: // member functions
     const TSeqCol* GetListOfClasses();
     const TSeqCol* GetListOfCppDirectives();
     const TSeqCol* GetListOfFilesInPath( const char path[] );
     const TSeqCol* GetListOfEnvVars();
     const TSeqCol* GetListOfGlobalFunctions();
     const TSeqCol* GetListOfGlobals();
     const TSeqCol* GetListOfPragmas();
     const TSeqCol* GetListOfSysIncFiles();
     const TSeqCol* GetListOfUsers();

     void ClearClasses();
     void ClearCppDirectives();
     void ClearEnvVars();
     void ClearFiles();
     void ClearGlobalFunctions();
     void ClearGlobals();
     void ClearPragmas();
     void ClearSysIncFiles();
     void ClearUsers();

     void ClearAll();

     void RehashClasses();
     void RehashCppDirectives();
     void RehashEnvVars();
     void RehashFiles();
     void RehashGlobalFunctions();
     void RehashGlobals();
     void RehashPragmas();
     void RehashSysIncFiles();
     void RehashUsers();

     void RehashAll();

 public: // static utility functions
     static Char_t   AllAgreeOnChar( int i, const TSeqCol* pList, Int_t& nGoodStrings );
     static void     AppendListOfFilesInDirectory( const char dirName[], TSeqCol* pList );
     static TString  DetermineClass( const char varName[] ); // TROOT
     static Bool_t   ExcludedByFignore( TString s );
     static TString  GetSysIncludePath(); // TROOT
     static Bool_t   IsDirectory( const char fileName[] ); // TSystem
     static TSeqCol* NewListOfFilesInPath( const char path[] );
     static Bool_t   PathIsSpecifiedInFileName( const TString& fileName );
     static void     NoMsg( Int_t errorLevel );

 public: // enums
     enum {kDebug = 17}; // set gDebug==TTabCom::kDebug for debugging output

     enum EContext_t
     {
          kUNKNOWN_CONTEXT=-1,
          // first enum (not counting "kUNKNOWN_CONTEXT") must
          // cast to zero because these enums will be used to
          // index arrays of size "kNUM_PAT"
          // ---------------

          // user names and environment variables should come first
          kSYS_UserName,
          kSYS_EnvVar, // environment variables

          // file descriptor redirection should almost come first
          kCINT_stdout,  // stdout
          kCINT_stderr,  // stderr
          kCINT_stdin,   // stdin

          // CINT "." instructions
          // the position of these guys is irrelevant since each of
          // these commands will always be the only command on the line.
          kCINT_Load,  // .L
          kCINT_Exec,  // .x

          // specific preprocessor directives.
          kCINT_pragma,
          kCINT_includeSYS,  // system files
          kCINT_includePWD,  // local files

          // specific preprocessor directives
          // must come before general preprocessor directives

          // general preprocessor directives
          // must come after specific preprocessor directives
          kCINT_cpp,

          // specific member accessing
          // should come before general member accessing
          kROOT_Load,

          // random files
          /******************************************************************/
          /*                                                                */
          /* file names should come before member accessing                 */
          /*                                                                */
          /* (because otherwise "/tmp/a.cc" might look like you're trying   */
          /* to access member "cc" of some object "a")                      */
          /*                                                                */
          /* but after anything that requires a specific path.              */
          /*                                                                */
          /******************************************************************/
          kSYS_FileName,

          // general member access
          // should come after specific member access
          kCXX_ScopeMember,
          kCXX_DirectMember,
          kCXX_IndirectMember,

          // time to print prototype
          kCXX_ScopeProto,
          kCXX_DirectProto,
          kCXX_IndirectProto,
          kCXX_NewProto, // kCXX_NewProto must come before kCXX_ConstructorProto
          kCXX_ConstructorProto, // kCXX_ConstructorProto this must come before kCXX_GlobalProto

          // arbitrary global identifiers
          // should really come last
          kCXX_Global,
          kCXX_GlobalProto,


          // ------- make additions above this line ---------
          kNUM_PAT // kNUM_PAT must be last. (to fix array size)
     };

 private: // member functions
     TTabCom(const TTabCom &);  //private and not implemented

     Int_t      Complete( const TRegexp& re, const TSeqCol* pListOfCandidates, const char appendage[] );
     void       CopyMatch( char dest[], const char localName[], const char appendage[]=0, const char fullName[]=0 ) const;
     EContext_t DetermineContext() const;
     TString    DeterminePath( const TString& fileName, const char defaultPath[] ) const;
     TString    ExtendPath( const char originalPath[], TString newBase ) const;
     Int_t      Hook( char* buf, int* pLoc );
     void       InitPatterns();
     TClass*    MakeClassFromClassName( const char className[] ) const;
     TClass*    MakeClassFromVarName( const char varName[], EContext_t& context );
     void       SetPattern( EContext_t handle, const char regexp[] );

 private: // friends
     friend int gl_root_tab_hook(char* buf, int prompt_width, int* pLoc);

 private: // data members
     TSeqCol* fpClasses;
     TSeqCol* fpDirectives;
     TSeqCol* fpEnvVars;
     TSeqCol* fpFiles;
     TSeqCol* fpGlobals;
     TSeqCol* fpGlobalFuncs;
     TSeqCol* fpPragmas;
     TSeqCol* fpSysIncFiles;
     TSeqCol* fpUsers;

     char* fBuf;  // initialized by Hook()
     int*  fpLoc; // initialized by Hook()

     Pattern_t   fPat[ kNUM_PAT ][ MAX_LEN_PAT ];  // array of patterns
     const char* fRegExp[ kNUM_PAT ];              // corresponding regular expression plain text

     ClassDef(TTabCom,0)  //Perform comand line completion when hitting <TAB>
};

extern TTabCom* gTabCom;

#endif
