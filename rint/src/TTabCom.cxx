// @(#)root/rint:$Name$:$Id$
// Author: Christian Lacunza <lacunza@cdfsg6.lbl.gov>   27/04/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

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
//   public member function or data member (including base classes)       //
//   global variable, function, or class name                             //
//                                                                        //
// Also, something like                                                   //
//                                                                        //
//   someObject->Func([TAB]                                               //
//   someObject.Func([TAB]                                                //
//   someClass::Func([TAB]                                                //
//   someClass var([TAB]                                                  //
//   new someClass([TAB]                                                  //
//                                                                        //
// will print a list of prototypes for the indicated                      //
// method or constructor.                                                 //
//                                                                        //
// Current limitations and bugs:                                          //
//                                                                        //
//  1. you can only use one member access operator at a time.             //
//     eg, this will work: gROOT->GetListOfG[TAB]                         //
//     but this will not:  gROOT->GetListOfGlobals()->Conta[TAB]          //
//                                                                        //
//  2. nothing is guaranteed to work on windows or VMS                    //
//     (for one thing, /bin/env and /etc/passwd are hardcoded)            //
//                                                                        //
//  3. CINT shortcut #2 is deliberately not supported.                    //
//     (using "operator.()" instead of "operator->()")                    //
//                                                                        //
//  4. most identifiers (including C++ identifiers, usernames,            //
//     environment variables, etc)                                        //
//     are restriceted to this character set: [_a-zA-Z0-9]                //
//     therefore, you won't be able to complete things like               //
//                                                                        //
//          operator new                                                  //
//          operator+                                                     //
//          etc                                                           //
//                                                                        //
//  5. ~whatever[TAB] always tries to complete a username.                //
//     use whitespace (~ whatever[TAB]) if you want to complete a global  //
//     identifier.                                                        //
//                                                                        //
//  6. CINT shortcut #3 is not supported when trying to complete          //
//     the name of a global object.  (it is supported when trying to      //
//     complete a member of a global object)                              //
//                                                                        //
//  7. the list of #pragma's is hardcoded                                 //
//     (ie not obtained from the interpreter at runtime)                  //
//     ==> user-defined #pragma's will not be recognized                  //
//                                                                        //
//  8. the system include directories are also hardcoded                  //
//     because i don't know how to get them from the interpreter.         //
//     fons, maybe they should be #ifdef'd for the different sytems?      //
//                                                                        //
//  9. the TabCom.FileIgnore resource is always applied, even if you      //
//     are not trying to complete a filename.                             //
//                                                                        //
// 10. anything in quotes is assumed to be a filename                     //
//     so (among other things) you can't complete a quoted class name:    //
//     eg, TClass class1( "TDict[TAB]                                     //
//     this won't work... looks for a file in pwd starting with TDict     //
//                                                                        //
// 11. the prototypes tend to omit the word "const" a lot.                //
//     this is a problem with ROOT or CINT.                               //
//                                                                        //
// 12. when listing ambiguous matches, only one column is used,           //
//     even if there are many completions.                                //
//                                                                        //
// 13. anonymous objects are not currently identified                     //
//     so, for example,                                                   //
//                                                                        //
//          root> printf( TString([TAB                                    //
//                                                                        //
//     gives an error message instead of listing TString's constructors.  //
//     (this could be fixed)                                              //
//                                                                        //
// 14. the routine that adds the "appendage" isn't smart enough to know   //
//     if it's already there:                                             //
//                                                                        //
//          root> TCanvas::Update()                                       //
//              press [TAB] here ^                                        //
//          root> TCanvas::Update()()                                     //
//     (this could be fixed)                                              //
//                                                                        //
// 15. the appendage is only applied if there is exactly 1 match.         //
//     eg, this                                                           //
//                                                                        //
//          root> G__at[TAB]                                              //
//          root> G__ateval                                               //
//                                                                        //
//     happens instead of this                                            //
//                                                                        //
//          root> G__at[TAB]                                              //
//          root> G__ateval(                                              //
//                                                                        //
//     because there are several overloaded versions of G__ateval().      //
//     (this could be fixed)                                              //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

#ifdef HAVE_CONFIG
#include "config.h"
#endif

#include "TTabCom.h"
#include "TClass.h"
#include "TSystem.h"
#include "TROOT.h"
#include "TMethod.h"
#include "TEnv.h"
#include "TBenchmark.h"
#include "TError.h"
#include "TGlobal.h"
#include "TList.h"
#include "Getline.h"
#include "TFunction.h"
#include "TMethodArg.h"

//Direct CINT include
#include "DataMbr.h"


#include <stdio.h>
#include <iostream.h>
#ifndef WIN32
#  include <strstream.h>
#else
#  include <strstrea.h>
#endif
#include <fstream.h>
#include <iomanip.h>     // setw()


#define BUF_SIZE    1024 // must match value in C_Getline.c (for bounds checking)
#define IfDebug(x)  if(gDebug==TTabCom::kDebug) x


ClassImp(TTabCom)

// ----------------------------------------------------------------------------
//
//             global/file scope variables
//

TTabCom* gTabCom=0;


int gl_root_tab_hook(char* buf, int /*prompt_width*/, int* pLoc)
{
     return gTabCom ? gTabCom->Hook( buf, pLoc ) : -1;
}


// ----------------------------------------------------------------------------
//
//              constructors
//

TTabCom::TTabCom()
{
     fpDirectives = 0;
     fpPragmas = 0;
     fpGlobals = 0;
     fpGlobalFuncs = 0;
     fpClasses = 0;
     fpUsers = 0;
     fpEnvVars = 0;
     fpFiles = 0;
     fpSysIncFiles = 0;

     InitPatterns();

     gl_tab_hook = gl_root_tab_hook;
}

//
//              constructors
//
// ----------------------------------------------------------------------------


// ----------------------------------------------------------------------------
//
//              public member functions
//


void TTabCom::ClearClasses()        { if( !fpClasses     ) return; fpClasses->Delete(0);     delete fpClasses;     fpClasses = 0;     }
void TTabCom::ClearCppDirectives()  { if( !fpDirectives  ) return; fpDirectives->Delete(0);  delete fpDirectives;  fpDirectives = 0;  }
void TTabCom::ClearEnvVars()        { if( !fpEnvVars     ) return; fpEnvVars->Delete(0);     delete fpEnvVars;     fpEnvVars = 0;     }
void TTabCom::ClearFiles()          { if( !fpFiles       ) return; fpFiles->Delete(0);       delete fpFiles;       fpFiles = 0;       }
void TTabCom::ClearGlobalFunctions(){ if( !fpGlobalFuncs ) return; fpGlobalFuncs->Delete(0); delete fpGlobalFuncs; fpGlobalFuncs = 0; }
void TTabCom::ClearGlobals()        { if( !fpGlobals     ) return; fpGlobals->Delete(0);     delete fpGlobals;     fpGlobals = 0;     }
void TTabCom::ClearPragmas()        { if( !fpPragmas     ) return; fpPragmas->Delete(0);     delete fpPragmas;     fpPragmas = 0;     }
void TTabCom::ClearSysIncFiles()    { if( !fpSysIncFiles ) return; fpSysIncFiles->Delete(0); delete fpSysIncFiles; fpSysIncFiles = 0; }
void TTabCom::ClearUsers()          { if( !fpUsers       ) return; fpUsers->Delete(0);       delete fpUsers;       fpUsers = 0;       }

void TTabCom::ClearAll()
{
     // clears all lists
     // except for user names and system include files.

     ClearClasses();
     ClearCppDirectives();
     ClearEnvVars();
     ClearFiles();
     ClearGlobalFunctions();
     ClearGlobals();
     ClearPragmas();
//   ClearSysIncFiles(); <-- this one stays cached
//   ClearUsers();       <-- this one stays cached
}

void TTabCom::RehashClasses()         { ClearClasses();         GetListOfClasses();         }
void TTabCom::RehashCppDirectives()   { ClearCppDirectives();   GetListOfCppDirectives();   }
void TTabCom::RehashEnvVars()         { ClearEnvVars();         GetListOfEnvVars();         }
void TTabCom::RehashFiles()           { ClearFiles();           /* path unknown */          } // think about this
void TTabCom::RehashGlobalFunctions() { ClearGlobalFunctions(); GetListOfGlobalFunctions(); }
void TTabCom::RehashGlobals()         { ClearGlobals();         GetListOfGlobals();         }
void TTabCom::RehashPragmas()         { ClearPragmas();         GetListOfPragmas();         }
void TTabCom::RehashSysIncFiles()     { ClearSysIncFiles();     GetListOfSysIncFiles();     }
void TTabCom::RehashUsers()           { ClearUsers();           GetListOfUsers();           }

void TTabCom::RehashAll()
{
     // clears and then rebuilds all lists
     // except for user names and system include files.

     RehashClasses();
     RehashCppDirectives();
     RehashEnvVars();
     RehashFiles();
     RehashGlobalFunctions();
     RehashGlobals();
     RehashPragmas();
//   RehashSysIncFiles(); <-- this one stays cached
//   RehashUsers();       <-- this one stays cached
}

const TSeqCol* TTabCom::GetListOfClasses( void )
{
     if( !fpClasses )
     {
          // generate a text list of classes on disk
          strstream   cmd;
          const char* tmpfilename = tmpnam(0);
          cmd << ".class > " << tmpfilename << endl;
          gROOT->ProcessLineSync( cmd.str() ); // memory leak cmd.str()

          // open the file
          ifstream file1( tmpfilename );
          if( !file1 ) {
               Error("TTabCom::GetListOfClasses", "could not open file \"%s\"", tmpfilename);
               gSystem->Unlink( tmpfilename );
               return 0;
          }

          // skip the first 2 lines (which are just header info)
          file1.ignore(32000,'\n');
          file1.ignore(32000,'\n');

          // parse file, add to list
          fpClasses = new TContainer;
          TString line;
          while( file1 )
          {
               line = "";
               line.ReadLine( file1, kFALSE ); // kFALSE ==> don't skip whitespace
               line = line( 23, 32000 );
// old way...
//             if (line.Index("class") >= 0)
//                  line = line(6, 32000);
//             else if (line.Index("enum") >= 0)
//                  line = line(5, 32000);
//             else if (line.Index("(unknown)") >= 0)
//                  line = line(10, 32000);
//             line = line("[^ ]*");
// new way...
               int index;
               if(0);
               else if ((index=line.Index(" class ")    ) >= 0) line = line(1+index+6, 32000);
               else if ((index=line.Index(" struct ")   ) >= 0) line = line(1+index+7, 32000);
               else if ((index=line.Index(" enum ")     ) >= 0) line = line(1+index+5, 32000);
               else if ((index=line.Index(" (unknown) ")) >= 0) line = line(1+index+10, 32000);
               // 2 changes: 1. use spaces ^         ^          2. use offset ^^^^^ in case of long
               //               to reduce probablility that        filename which overflows
               //               these keywords will occur in       its field.
               //               filename or classname.
               line = line("[^ ]*");
               fpClasses->Add( new TObjString(line.Data()) );
          }

          // done with this file
          file1.close();
          gSystem->Unlink( tmpfilename );
     }

     return fpClasses;
}
const TSeqCol* TTabCom::GetListOfCppDirectives()
{
     if( !fpDirectives )
     {
          fpDirectives = new TContainer;

          fpDirectives->Add(  new TObjString("if")       );
          fpDirectives->Add(  new TObjString("ifdef")    );
          fpDirectives->Add(  new TObjString("ifndef")   );
          fpDirectives->Add(  new TObjString("elif")     );
          fpDirectives->Add(  new TObjString("else")     );
          fpDirectives->Add(  new TObjString("endif")    );
          fpDirectives->Add(  new TObjString("include")  );
          fpDirectives->Add(  new TObjString("define")   );
          fpDirectives->Add(  new TObjString("undef")    );
          fpDirectives->Add(  new TObjString("line")     );
          fpDirectives->Add(  new TObjString("error")    );
          fpDirectives->Add(  new TObjString("pragma")   );
     }

     return fpDirectives;
}
const TSeqCol* TTabCom::GetListOfFilesInPath( const char path[] )
{
     // "path" should be initialized with a colon separated list of
     // system directories

     static TString previousPath;

     if( path && fpFiles && strcmp(path,previousPath)==0 )
     {
          return fpFiles;
     }
     else
     {
          ClearFiles();

          fpFiles = NewListOfFilesInPath( path );
     }

     return fpFiles;
}
const TSeqCol* TTabCom::GetListOfEnvVars()
{
     // calls "/bin/env"

     if( !fpEnvVars )
     {
          const char* tmpfilename = tmpnam(0);
          strstream  cmd;

#ifndef WIN32
         cmd << "/bin/env > " << tmpfilename << endl;
#else
         cmd << "set > " << tmpfilename << endl;
#endif
          gSystem->Exec( cmd.str() ); // memory leak cmd.str()

          // open the file
          ifstream file1( tmpfilename );
          if( !file1 ) {
               Error( "TTabCom::GetListOfEnvVars", "could not open file \"%s\"", tmpfilename );
               gSystem->Unlink( tmpfilename );
               return 0;
          }


          // parse, add
          fpEnvVars = new TContainer;
          TString line;
          while( file1 ) // i think this loop goes one time extra which
                         // results in an empty string in the list, but i don't think it causes any
                         // problems.
          {
               line.ReadToDelim( file1, '=' );
               file1.ignore(32000,'\n');
               fpEnvVars->Add( new TObjString(line.Data()) );
          }

          file1.close();
          gSystem->Unlink( tmpfilename );
     }

     return fpEnvVars;
}
//______________________________________________________________________________
const TSeqCol* TTabCom::GetListOfGlobals()
{
     if( !fpGlobals ) {

          fpGlobals = new TContainer;

          G__DataMemberInfo *a;
          int last  = 0;
          int nglob = 0;

          // find the number of global objects
          G__DataMemberInfo t;
          while (t.Next()) nglob++;

          for (int i = 0; i < nglob; i++) {
               a = new G__DataMemberInfo();
               a->Next();   // initial positioning

               for (int j = 0; j < last; j++)
                    a->Next();

               // if name cannot be obtained no use to put in list
               if (a->IsValid() && a->Name()) {
                    fpGlobals->Add(new TGlobal(a));
               } else
                    delete a;

               last++;
          }
     }

     return fpGlobals;
}
//______________________________________________________________________________
const TSeqCol* TTabCom::GetListOfGlobalFunctions()
{
     if( !fpGlobalFuncs ) {

          fpGlobalFuncs = new TContainer;

          G__MethodInfo *a;
          int last  = 0;
          int nglob = 0;

          // find the number of global functions
          G__MethodInfo t;
          while (t.Next()) nglob++;

          for (int i = 0; i < nglob; i++) {
               a = new G__MethodInfo();
               a->Next();   // initial positioning

               for (int j = 0; j < last; j++)
                    a->Next();

               // if name cannot be obtained no use to put in list
               if (a->IsValid() && a->Name()) {
                    fpGlobalFuncs->Add(new TFunction(a));
               } else
                    delete a;

               last++;
          }
     }

     return fpGlobalFuncs;
}
const TSeqCol* TTabCom::GetListOfPragmas()
{
     if( !fpPragmas )
     {
          fpPragmas = new TContainer;

          fpPragmas->Add(  new TObjString("ANSI "          )  );
          fpPragmas->Add(  new TObjString("autocompile "   )  );
          fpPragmas->Add(  new TObjString("bytecode "      )  );
          fpPragmas->Add(  new TObjString("compile "       )  );
          fpPragmas->Add(  new TObjString("endbytecode "   )  );
          fpPragmas->Add(  new TObjString("endcompile "    )  );
          fpPragmas->Add(  new TObjString("include "       )  );
          fpPragmas->Add(  new TObjString("includepath "   )  );
          fpPragmas->Add(  new TObjString("K&R "           )  );
          fpPragmas->Add(  new TObjString("link "          )  );
          fpPragmas->Add(  new TObjString("preprocess "    )  );
          fpPragmas->Add(  new TObjString("preprocessor "  )  );
          fpPragmas->Add(  new TObjString("security level" )  );
          // "setertti "  omitted. Ordinaly user should not use this statement
          // "setstdio "  omitted. Ordinaly user should not use this statement
          // "setstream " omitted. Ordinaly user should not use this statement
          // "stub"       omitted. Ordinaly user should not use this statement

     }

     return fpPragmas;
}
const TSeqCol* TTabCom::GetListOfSysIncFiles()
{
     if( !fpSysIncFiles )
     {
          fpSysIncFiles = NewListOfFilesInPath( GetSysIncludePath() );
     }

     return fpSysIncFiles;

}
const TSeqCol* TTabCom::GetListOfUsers()
{
     // reads from "/etc/passwd"

     if( !fpUsers )
     {
          fpUsers = new TContainer;

          ifstream passwd;
          TString  user;

          passwd.open("/etc/passwd");
          while( passwd )
          {
               user.ReadToDelim( passwd, ':' );
               fpUsers->Add( new TObjString(user) );
               passwd.ignore( 32000, '\n' );
          }
          passwd.close();
     }

     return fpUsers;
}

//
//              public member functions
//
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
//
//                           static utility funcitons
//

Char_t TTabCom::AllAgreeOnChar( int i, const TSeqCol* pList, Int_t& nGoodStrings )
{
     //[static utility function]///////////////////////////////////////////
     //
     //  if all the strings in "*pList" have the same ith character,
     //  that character is returned.
     //  otherwise 0 is returned.
     //
     //  any string "s" for which "ExcludedByFignore(s)" is true
     //  will be ignored unless All the strings in "*pList"
     //  are "ExcludedByFignore()"
     //
     //  in addition, the number of strings which were not
     //  "ExcludedByFignore()" is returned in "nGoodStrings".
     //
     /////////////////////////////////////////////////////////////////////////

     assert( pList != 0 );

     TIter       next  (pList);
     TObject*    pObj;
     const char* s;
     char        ch0;
     Bool_t      isGood;
     Bool_t      atLeast1GoodString;

     // init
     nGoodStrings = 0;
     atLeast1GoodString = kFALSE;

     // first look for a good string
     do
     {
          if(( pObj=next() )) {
               s = pObj->GetName();
               isGood = !ExcludedByFignore(s);
               if( isGood ) {
                    atLeast1GoodString = kTRUE;
                    nGoodStrings += 1;
               }
          }
          else {
               // reached end of list without finding a single good string.
               // just use the first one.
               next.Reset();
               pObj = next();
               s = pObj->GetName();
               break;
          }
     }
     while( !isGood );

     // found a good string...
     ch0 = s[i];

     // all subsequent good strings must have the same ith char
     do
     {
          if(( pObj=next() ))
          {
               s = pObj->GetName();
               isGood = !ExcludedByFignore(s);
               if( isGood ) nGoodStrings += 1;
          }
          else
               return ch0;
     }
     while( ((int)strlen(s)>=i && s[i] == ch0) ||
            (atLeast1GoodString && !isGood) );

     return 0;
}
void TTabCom::AppendListOfFilesInDirectory( const char dirName[], TSeqCol* pList )
{
     //[static utility function]/////////////////////////////
     //
     //  adds a TObjString to "*pList"
     //  for each entry found in the system directory "dirName"
     //
     //  directories that do not exist are silently ignored.
     //
     //////////////////////////////////////////////////////////

     assert( dirName != 0 );
     assert( pList != 0 );

     // open the directory
     void* dir = gSystem->OpenDirectory( dirName );

     // it is normal for this function to receive names of directories that do not exist.
     // they should be ignored and should not generate any error messages.
     if( !dir ) return;

     // put each filename in the list
     const char*  tmp_ptr; // gSystem->GetDirEntry() returns NULL when no more files.
     TString      fileName;

     while(( tmp_ptr = gSystem->GetDirEntry(dir) ))
     {
          fileName = tmp_ptr;

          // skip "." and ".."
          if( fileName == "." || fileName == ".." ) continue;

          // add to list
          pList->Add( new TObjString(dirName+fileName.Prepend("/")) );
     }
     // NOTE:
     // with a path like "/usr/include:/usr/include/CC:$ROOTDIR/include:$ROOTDIR/cint/include:..."
     // the above loop could get traversed 700 times or more.
     // ==> keep it minimal or it could cost whole seconds on slower machines.
     // also: TClonesArray doesn't help.

     // close the directory
     gSystem->FreeDirectory(dir);
}
// -----\/-------- homemade RTTI ---------------\/------------------------
TString TTabCom::DetermineClass( const char varName[] )
{
     //[static utility function]/////////////////////////////
     //
     //  returns empty string on failure.
     //  otherwise returns something like this: "TROOT*".
     //  fails for non-class types (ie, int, char, etc).
     //  fails for pointers to functions.
     //
     ///////////////////////////////////


     ///////////////////////////////////
     //
     //  note that because of the strange way this function works,
     //  CINT will print
     //
     //     Error: No symbol asdf in current scope  FILE:/var/tmp/gaaa001HR LINE:1
     //
     //  if "varName" is not defined. (in this case, varName=="asdf")
     //  i don't know how to suppress this.
     //
     ///////////////////////////////////

     assert(varName != 0);
     IfDebug(cerr << "DetermineClass(\"" << varName << "\");" << endl);

     strstream   cmd;
     const char* tmpfile = tmpnam(0);
     cmd << "gROOT->ProcessLine(\"" << varName << "\"); > " << tmpfile << endl;
     gROOT->ProcessLineSync(cmd.str()); // memory leak cmd.str()
     // the type of the variable whose name is "varName"
     // should now be stored on disk in the file "tmpfile"

     TString type = "";
     int c;

     // open the file
     ifstream file1( tmpfile );
     if( !file1 ) {
          Error( "TTabCom::DetermineClass", "could not open file \"%s\"", tmpfile );
          goto cleanup;
     }

     // first char should be '(', which we can ignore.
     c = file1.get();
     if( !file1 || c<=0 || c=='*' || c!='(' ) {
          Error( "TTabCom::DetermineClass", "variable \"%s\" not defined?", varName );
          goto cleanup;
     }
     IfDebug(cerr << (char)c << flush);

     // in case of success, "class TClassName*)0x12345" remains,
     // since the opening '(' was removed.
     file1 >> type; // ignore "class"

     // non-class type ==> failure
     if( type != "class" && type != "struct" ) {
          type = ""; // empty return string indicates failure.
          goto cleanup; //* RETURN *//
     }

     // ignore ' '
     c = file1.get();
     IfDebug(cerr << (char)c << flush);

     // this is what we want
     type.ReadToDelim( file1, ')' );
     IfDebug(cerr << type << endl);

cleanup:
     // done reading from file
     file1.close();
     gSystem->Unlink( tmpfile );

     return type;
}
Bool_t TTabCom::ExcludedByFignore( TString s )
{
     //[static utility function]/////////////////////////////
     //
     //  returns true iff "s" ends with one of
     //  the strings listed in the "TabCom.FileIgnore" resource.
     //
     /////////////////////////////////////////////////////////////

     const char* fignore  = gEnv->GetValue("TabCom.FileIgnore", (char*)0);

     if( !fignore )
     {
          return kFALSE;
     }
     else
     {
          istrstream endings((char*)fignore); // do i need to make a copy first?
          TString    ending;

          ending.ReadToDelim( endings, ':' );

          while( !ending.IsNull() ) {
               if( s.EndsWith(ending) )
                    return kTRUE;
               else
                    ending.ReadToDelim( endings, ':' ); // next
          }
          return kFALSE;
     }
}
TString TTabCom::GetSysIncludePath( void )
{
     //[static utility function]/////////////////////////////
     //
     //  returns a colon-separated string of directories
     //  that CINT will search when you call #include<...>
     //
     //  returns empty string on failure.
     //
     ///////////////////////////////////////////////////////////

     // >i noticed that .include doesn't list the standard directories like
     // >/usr/include or /usr/include/CC.
     // >
     // >how can i get a list of all the directories the interpreter will
     // >search through when the user does a #include<...> ?
     //
     // Right now, there is no easy command to tell you about it.  Instead, I can
     // describe it here.
     //
     // 1) CINT first searches current working directory for #include "xxx"
     //   (#include <xxx> does not)
     //
     // 2) CINT searches include path directories given by -I option
     //
     // 3) CINT searches following standard include directories.
     //    $CINTSYSDIR/include
     //    $CINTSYSDIR/stl
     //    $CINTSYSDIR/msdev/include   if VC++4.0
     //    $CINTSYSDIR/sc/include      if Symantec C++
     //    /usr/include
     //    /usr/include/g++            if gcc,g++
     //    /usr/include/CC             if HP-UX
     //    /usr/include/codelibs       if HP-UX
     //
     // .include command only displays 2).
     //
     // Thank you
     // Masaharu Goto

     // 1) current dir
     // ----------------------------------------------
     // N/A


     // 2) -I option (and #pragma includepath)
     // ----------------------------------------------

     // get this part of the include path from the interpreter
     // and stick it in a tmp file.
     const char* tmpfilename = tmpnam(0);
     strstream cmd;
     cmd << "gROOT->ProcessLine(\".include\"); > " << tmpfilename << endl;
     gROOT->ProcessLineSync( cmd.str() ); // memory leak cmd.str()

     // open the tmp file
     ifstream file1( tmpfilename );
     if( !file1 ) { // error
          Error( "TTabCom::GetSysIncludePath", "could not open file \"%s\"", tmpfilename );
          gSystem->Unlink( tmpfilename );
          return "";
     }

     // parse it.
     TString token; // input buffer
     TString path;  // all directories so far (colon-separated)
     file1 >> token; // skip "include"
     file1 >> token; // skip "path:"
     while( file1 ) {
          file1 >> token;
          if( !token.IsNull() ) {
               if( path.Length() > 0 ) path.Append(":");
               path.Append(token.Data()+2); // +2 skips "-I"
          }
     }

     // done with the tmp file
     file1.close();
     gSystem->Unlink( tmpfilename );

     // 3) standard directories
     // ----------------------------------------------

#ifndef CINTINCDIR
     TString CINTSYSDIR("$ROOTSYS/cint");
#else
     TString CINTSYSDIR(CINTINCDIR);
#endif
     path.Append(":"+CINTSYSDIR+"/include");
//   path.Append(":"+CINTSYSDIR+"/stl");
//   path.Append(":"+CINTSYSDIR+"/msdev/include");
//   path.Append(":"+CINTSYSDIR+"/sc/include");
     path.Append(":/usr/include");
//   path.Append(":/usr/include/g++");
//   path.Append(":/usr/include/CC");
//   path.Append(":/usr/include/codelibs");

     return path;
}
Bool_t TTabCom::IsDirectory( const char fileName[] )
{
     //[static utility function]/////////////////////////////
     //
     //  calls TSystem::GetPathInfo() to see if "fileName"
     //  is a system directory.
     //
     ///////////////////////////////////////////////////////

     Long_t flags = 0;
     gSystem->GetPathInfo( fileName, 0, 0, &flags, 0 );
     return (int)flags & 2;
}
TSeqCol* TTabCom::NewListOfFilesInPath( const char path1[] )
{
     //[static utility function]/////////////////////////////
     //
     //  creates a list containing the full path name for each file
     //  in the (colon separated) string "path1"
     //
     //  memory is allocated with "new", so
     //  whoever calls this function takes responsibility for deleting it.
     //
     //////////////////////////////////////////////////////////////////////

     assert( path1 != 0 );

     TContainer* pList = new TContainer; // maybe use RTTI here? (since its a static function)
     istrstream  path((char*)path1);
     TString     dirName;

     dirName.ReadToDelim( path, ':' );

     while( !dirName.IsNull() )
     {
          IfDebug(cerr << "NewListOfFilesInPath(): dirName = " << dirName << endl);

          AppendListOfFilesInDirectory( dirName, pList );

          // next
          dirName.ReadToDelim( path, ':' );
     }

     return pList;
}
Bool_t TTabCom::PathIsSpecifiedInFileName( const TString& fileName )
{
     //[static utility function]/////////////////////////////
     //
     //  true if "fileName"
     //  1. is an absolute path ("/tmp/a")
     //  2. is a relative path  ("../whatever", "./test")
     //  3. starts with user name ("~/mail")
     //  4. starts with an environment variable ("$ROOTSYS/bin")
     //
     //////////////////////////////////////////////////////////////////////////

     char c1 = (fileName.Length()>0) ? fileName[0] : 0;
     return c1=='/' || c1=='~' || c1=='$' || fileName.BeginsWith("./") || fileName.BeginsWith("../");
}
void TTabCom::NoMsg( Int_t errorLevel )
{
     //[static utility function]/////////////////////////////
     //
     //  calling "NoMsg( errorLevel )",
     //  sets "gErrorIgnoreLevel" to "errorLevel+1" so that
     //  all errors with "level < errorLevel" will be ignored.
     //
     //  calling the function with a negative argument
     //  (e.g., "NoMsg( -1 )")
     //  resets gErrorIgnoreLevel to its previous value.
     //
     //////////////////////////////////////////////////////////////////

     ////////////////////////////////////////////////////////////////
     //
     // if you call the function twice with a non-negative argument
     // (without an intervening call with a negative argument)
     // it will complain because it is almost certainly an error
     // that will cause the function to loose track of the previous
     // value of gErrorIgnoreLevel.
     //
     // most common causes: 1. suspiciously placed "return;" statement
     //                     2. calling a function that calls "NoMsg()"
     //
     //////////////////////////////////////////////////////////////////

     const  Int_t kNotDefined = -1;
     static Int_t old_level   = kNotDefined;

     if( errorLevel < 0 ) // reset
     {
          if( old_level==kNotDefined ) {
               cerr << "NoMsg(): ERROR 1. old_level==" << old_level << endl;
               return;
          }

          gErrorIgnoreLevel = old_level; // resore
          old_level = kNotDefined;
     }
     else // set
     {
          if( old_level!=kNotDefined ) {
               cerr << "NoMsg(): ERROR 2. old_level==" << old_level << endl;
               return;
          }

          old_level = gErrorIgnoreLevel;
          if( gErrorIgnoreLevel <= errorLevel ) gErrorIgnoreLevel = errorLevel+1;
     }
}

//
//                           static utility funcitons
//
// ----------------------------------------------------------------------------


// ----------------------------------------------------------------------------
//
//                       private member functions
//
//


Int_t TTabCom::Complete( const TRegexp& re, const TSeqCol* pListOfCandidates, const char appendage[] )
{
     // [private]

     // returns position of first change in buffer
     // ------------------------------------------
     // -2 ==> new line altogether (whole thing needs to be redrawn, including prompt)
     // -1 ==> no changes
     //  0 ==> beginning of line
     //  1 ==> after 1st char
     //  n ==> after nth char

     IfDebug(cerr << "TTabCom::Complete() ..." << endl);
     assert( fpLoc != 0 );
     assert( pListOfCandidates != 0 );

     Int_t     pos;          // position of first change
     const int loc = *fpLoc; // location where TAB was pressed

     // -----------------------------------------
     //
     // 1. get the substring we need to complete
     //
     // NOTES:
     // s1 = original buffer
     // s2 = sub-buffer from 0 to wherever the user hit TAB
     // s3 = the actual text that needs completing
     //
     // -----------------------------------------
     TString s1( fBuf );
     TString s2 = s1( 0, loc );
     TString s3 = s2( re );

     int start = s2.Index( re );

     IfDebug(cerr << "   s1: " << s1    << endl);
     IfDebug(cerr << "   s2: " << s2    << endl);
     IfDebug(cerr << "   s3: " << s3    << endl);
     IfDebug(cerr << "start: " << start << endl);
     IfDebug(cerr << endl);

     // -----------------------------------------
     // 2. go through each possible completion,
     //    keeping track of the number of matches
     // -----------------------------------------
     TList       listOfMatches;   // list of matches (local filenames only) (insertion order must agree across these 3 lists)
     TList       listOfFullPaths; // list of matches (full filenames)       (insertion order must agree across these 3 lists)

     int         nMatches=0;      // number of matches
     TObject*    pObj;            // pointer returned by iterator
     TIter       next_candidate  (pListOfCandidates);
     TIter       next_match      (&listOfMatches);
     TIter       next_fullpath   (&listOfFullPaths);

     // stick all matches into "listOfMatches"
     while(( pObj = next_candidate() ))
     {
          // get the full filename
          const char* s4 = pObj->GetName();

          assert( s4 != 0 );

          // pick off tail
          const char* s5 = strrchr(s4,'/');
          if( !s5 )
               s5 = s4; // no '/' found
          else
               s5 += 1; // advance past '/'

          // check for match
          if( strstr( s5, s3 ) == s5 ) {
               nMatches += 1;
               listOfMatches.Add( new TObjString(s5) );
               listOfFullPaths.Add( new TObjString(s4) );
               IfDebug(cerr << "adding " << s5 << '\t' << s4 << endl);
          }
          else {
               IfDebug(cerr << "considered " << s5 << '\t' << s4 << endl);
          }
     }

     // -----------------------------------------
     // 3. beep, list, or complete
     //    depending on how many matches were found
     // -----------------------------------------

     // 3a. no matches ==> bell
     TString partialMatch = "";

     if( nMatches == 0 )
     {
          cout << "\a" << flush;
          pos = -1;
          goto done; //* RETURN *//
     }

     // 3b. one or more matches.
     char match[1024];

     if( nMatches == 1 )
     {
          // get the (lone) match
          const char* short_name = next_match()->GetName();
          const char* full_name  = next_fullpath()->GetName();

          pObj = pListOfCandidates->FindObject( short_name );
          if( pObj ) {
               IfDebug(cerr << endl << "class: " << pObj->ClassName() << endl);
               TString className = pObj->ClassName();
               if( 0 );
               else if( className == "TMethod" || className == "TFunction" )
               {
                    TFunction* pFunc = (TFunction*)pObj;
                    if( pFunc->GetNargsOpt() == pFunc->GetNargs() )
                         appendage = "()"; // all args have default values
                    else
                         appendage = "(";  // user needs to supply some args
               }
               else if( className == "TDataMember" )
               {
                    appendage = " ";
               }
          }

          CopyMatch( match, short_name, appendage, full_name );
     }
     else
     {
          // multiple matches ==> complete as far as possible
          Char_t ch;
          Int_t  nGoodStrings;

          for( int i=0;
               (ch=AllAgreeOnChar( i, &listOfMatches, nGoodStrings ));
               i+=1 )
          {
               IfDebug(cerr << " i=" << i << " ch=" << ch << endl);
               partialMatch.Append( ch );
          }

          const char* s;
          const char* s0;

          // multiple matches, but maybe only 1 of them is any good.
          if( nGoodStrings == 1 ) {

               // find the 1 good match
               do {
                    s  = next_match()->GetName();
                    s0 = next_fullpath()->GetName();
               }
               while( ExcludedByFignore(s) );

               // and use it.
               CopyMatch( match, s, appendage, s0 );
          }
          else {
               IfDebug(cerr << "more than 1 GoodString" << endl);

//             if( partialMatch.Length() > (int)strlen(s3) )
               if( partialMatch.Length() > s3.Length() )
                    // this partial match is our (partial) completion.
               {
                    CopyMatch( match, partialMatch.Data() );
               }
               else
                    // couldn't do any completing at all,
                    // print a list of all the ambiguous matches
                    // (except for those excluded by "FileIgnore")
               {
                    IfDebug(cerr << "printing ambiguous matches" << endl);
                    cout << endl;
                    while(( pObj = next_match() )) {
                         s  = pObj->GetName();
                         s0 = next_fullpath()->GetName();
                         if( !ExcludedByFignore(s) || nGoodStrings==0 )
                         {
                              if( IsDirectory(s0) )
                                   cout << s << "/" << endl;
                              else
                                   cout << s << endl;
                         }
                    }
                    pos = -2;
                    goto done; //* RETURN *//
               }
          }
     }


     // ---------------------------------------
     // 4. finally write text into the buffer.
     // ---------------------------------------
     {
          int i = strlen(fBuf);                   // old EOL position is i
          int L = strlen(match) - (loc-start);    // new EOL position will be i+L

          // first check for overflow
          if( strlen(fBuf)+strlen(match)+1 > BUF_SIZE ) {
               Error("TTabCom::Complete", "buffer overflow");
               pos = -2;
               goto done; /* RETURN */
          }

          // debugging output
          IfDebug(cerr << "  i=" << i   << endl);
          IfDebug(cerr << "  L=" << L   << endl);
          IfDebug(cerr << "loc=" << loc << endl);

          // slide everything (including the null terminator) over to make space
          for( ; i>=loc; i-=1 ) {
               fBuf[i+L] = fBuf[i];
          }

          // insert match
          strncpy( fBuf+start, match, strlen(match) );

          pos    = loc;     // position of first change in "fBuf"
          *fpLoc = loc + L; // new cursor position
     }

 done: // <----- goto label
     // un-init
     fpLoc = 0;
     fBuf  = 0;

     return pos;
}
void TTabCom::CopyMatch( char dest[], const char localName[], const char appendage[], const char fullName[] ) const
{
     // [private]

     // if "appendage" is NULL, no appendage is applied.
     //
     // if "appendage" is of the form "filenameXXX" then,
     // "filename" is ignored and "XXX" is taken to be the appendage,
     // but it will only be applied if the file is not a directory...
     // if the file is a directory, a "/" will be used for the appendage instead.
     //
     // if "appendage" is of the form "XXX" then "XXX" will be appended to the match.

     assert( dest != 0 );
     assert( localName != 0 );

     // potential buffer overflow.
     strcpy( dest, localName );

     const char* key = "filename";
     const int   key_len = strlen(key);

     IfDebug(cerr << "CopyMatch()." << endl);
     IfDebug(cerr << "localName: " << (localName?localName:"NULL") << endl);
     IfDebug(cerr << "appendage: " << (appendage?appendage:"NULL") << endl);
     IfDebug(cerr << " fullName: " << (fullName ?fullName :"NULL") << endl);


     // check to see if "appendage" starts with "key"
     if( appendage && strncmp(appendage, key, key_len)==0 )
     {
          // filenames get special treatment
          appendage += key_len;
          IfDebug(cerr << "new appendage: " << appendage << endl);
          if( IsDirectory(fullName) )
          {
               if( fullName ) strcpy( dest+strlen(localName), "/" );
          }
          else
          {
               if( appendage ) strcpy( dest+strlen(localName), appendage );
          }
     }
     else
     {
          if( appendage ) strcpy( dest+strlen(localName), appendage );
     }
}
TTabCom::EContext_t TTabCom::DetermineContext() const
{
     // [private]

     assert( fBuf != 0 );

     const char* pStart; // start of match
     const char* pEnd;   // end of match

     for( int context=0; context<kNUM_PAT; ++context )
     {
          pEnd = Matchs( fBuf, *fpLoc, fPat[context], &pStart );
          if( pEnd )
          {
               IfDebug(cerr << endl
                       << "context=" << context << " "
                       << "RegExp="  << fRegExp[context]
                       << endl);
               return EContext_t( context ); //* RETURN *//
          }
     }

     return kUNKNOWN_CONTEXT; //* RETURN *//
}
TString TTabCom::DeterminePath( const TString& fileName, const char defaultPath[] ) const
{
     // [private]

     if( PathIsSpecifiedInFileName( fileName ) )
     {
          TString path = fileName;
          gSystem->ExpandPathName( path );
          path = gSystem->DirName( path );

          return path;
     }
     else
     {
          TString newBase;
          TString extendedPath;
          if( fileName.Contains("/") )
          {
               newBase      = gSystem->DirName(fileName);
               extendedPath = ExtendPath( defaultPath, newBase );
          }
          else
          {
               newBase      = "";
               extendedPath = defaultPath;
          }
          IfDebug(cerr << endl);
          IfDebug(cerr << "    fileName: " << fileName      << endl);
          IfDebug(cerr << "    pathBase: " << newBase       << endl);
          IfDebug(cerr << " defaultPath: " << defaultPath   << endl);
          IfDebug(cerr << "extendedPath: " << extendedPath  << endl);
          IfDebug(cerr << endl);

          return extendedPath;
     }
}
TString TTabCom::ExtendPath( const char originalPath[], TString newBase ) const
{
     // [private]

     if( newBase.BeginsWith("/") ) newBase = newBase.Strip( TString::kLeading, '/');
     strstream str;
     TString   dir;
     TString   newPath;
     str << originalPath;

#ifndef WIN32
     while( str )
#else
     while( 1 )
#endif
     {
          dir = "";
          dir.ReadToDelim( str, ':' );
          if( dir.IsNull() ) continue; // ignore blank entries
          newPath.Append( dir );
          if( !newPath.EndsWith("/") ) newPath.Append("/");
          newPath.Append( newBase );
          newPath.Append( ':' );
     }

     return newPath.Strip( TString::kTrailing, ':' );
}
Int_t TTabCom::Hook( char* buf, int* pLoc )
{
     // [private]

     // initialize
     fBuf  = buf;
     fpLoc = pLoc;

     // default
     Int_t pos = -2; // position of the first character that was changed in the buffer (needed for redrawing)

     // get the context this tab was triggered in.
     EContext_t context = DetermineContext();

     // get the substring that triggered this tab (as defined by "SetPattern()")
     const char dummy[] = ".";
     TRegexp re1(context==kUNKNOWN_CONTEXT ? dummy : fRegExp[ context ]);
     TString s1( fBuf );
     TString s2 = s1( 0, *fpLoc );
     TString s3 = s2( re1 );

     switch( context ) {
     case kUNKNOWN_CONTEXT:
          cerr << endl << "tab completion not implemented for this context" << endl;
          pos = -2;
          break;

     case kSYS_UserName:
          {
               const TSeqCol* pListOfUsers = GetListOfUsers();

               pos = Complete( "[^~]*$", pListOfUsers, "/" );
          }
          break;
     case kSYS_EnvVar:
          {
               const TSeqCol* pEnv = GetListOfEnvVars();

               pos = Complete( "[^$]*$", pEnv, "" );
          }
          break;

     case kCINT_stdout:
     case kCINT_stderr:
     case kCINT_stdin:
          {
               auto  TString  fileName     = s3("[^ ><]*$"); gSystem->ExpandPathName(fileName);
               const TString  filePath     = gSystem->DirName(fileName);
               const TSeqCol* pListOfFiles = GetListOfFilesInPath( filePath.Data() );

//             pos = Complete( "[^ /]*$", pListOfFiles, " " );
               pos = Complete( "[^ /]*$", pListOfFiles, "filename " );
          }
          break;

     case kCINT_Exec:
     case kCINT_Load:
          {
               const TString   fileName     = s3("[^ ]*$");
               const TString   macroPath    = DeterminePath( fileName, TROOT::GetMacroPath() );
               const TSeqCol*  pListOfFiles = GetListOfFilesInPath( macroPath.Data() );

//             pos = Complete( "[^ /]*$", pListOfFiles, " " );
               pos = Complete( "[^ /]*$", pListOfFiles, "filename " );
          }
          break;

     case kCINT_pragma:
          {
               pos = Complete( "[^ ]*$", GetListOfPragmas(), "" );
          }
          break;
     case kCINT_includeSYS:
          {
               TString fileName = s3("[^<]*$");
               if( PathIsSpecifiedInFileName( fileName ) || fileName.Contains("/") )
               {
                    TString includePath = DeterminePath( fileName, GetSysIncludePath() );

//                  pos = Complete( "[^</]*$", GetListOfFilesInPath( includePath ), "> " );
                    pos = Complete( "[^</]*$", GetListOfFilesInPath( includePath ), "filename> " );
               }
               else
               {
//                  pos = Complete( "[^</]*$", GetListOfSysIncFiles(), "> " );
                    pos = Complete( "[^</]*$", GetListOfSysIncFiles(), "filename> " );
               }
          }
          break;
     case kCINT_includePWD:
          {
               const TString  fileName     = s3("[^\"]*$");
               const TString  includePath  = DeterminePath( fileName, "." );
               const TSeqCol* pListOfFiles = GetListOfFilesInPath( includePath.Data() );

//             pos = Complete( "[^\"/]*$", pListOfFiles, "\" " );
               pos = Complete( "[^\"/]*$", pListOfFiles, "filename\" " );
          }
          break;

     case kCINT_cpp:
          {
               pos = Complete( "[^# ]*$", GetListOfCppDirectives(), " " );
          }
          break;

     case kROOT_Load:
          {
               const TString  fileName     = s3("[^\"]*$");
//             const TString  dynamicPath  = DeterminePath( fileName, TROOT::GetDynamicPath() ); /* should use this one */
               const TString  dynamicPath  = DeterminePath( fileName, gEnv->GetValue("Root.DynamicPath",(char*)0) );
               const TSeqCol* pListOfFiles = GetListOfFilesInPath( dynamicPath );

//             pos = Complete( "[^\"/]*$", pListOfFiles, "\");" );
               pos = Complete( "[^\"/]*$", pListOfFiles, "filename\");" );
          }
          break;

     case kSYS_FileName:
          {
               auto  TString  fileName     = s3("[^ \"]*$"); gSystem->ExpandPathName(fileName);
               const TString  filePath     = gSystem->DirName(fileName);
               const TSeqCol* pListOfFiles = GetListOfFilesInPath( filePath.Data() );

//             pos = Complete( "[^\" /]*$", pListOfFiles, "\"" );
               pos = Complete( "[^\" /]*$", pListOfFiles, "filename\"" );
          }
          break;

     case kCXX_ScopeMember:
     case kCXX_DirectMember:
     case kCXX_IndirectMember:
          {
               const EContext_t original_context = context; // save this for later

               TClass* pClass;
               TString name = s3("^[_a-zA-Z][_a-zA-Z0-9]*"); // may be a class, object, or pointer

               IfDebug(cerr << endl);
               IfDebug(cerr << "name: " << '"' << name << '"' << endl);

               switch( context )
               {
               case kCXX_ScopeMember:    pClass = MakeClassFromClassName( name );        break;
               case kCXX_DirectMember:   pClass = MakeClassFromVarName( name, context ); break;
               case kCXX_IndirectMember: pClass = MakeClassFromVarName( name, context ); break;
               default:                  assert(0);                                      break;
               }
               if( !pClass ) { pos = -2; break; }

               TContainer* pList = new TContainer;

               pList->AddAll( (TCollection*) pClass->GetListOfAllPublicMethods() );
               pList->AddAll( (TCollection*) pClass->GetListOfAllPublicDataMembers() );

               switch( context )
               {
               case kCXX_ScopeMember:    pos = Complete( "[^: ]*$", pList, "(" ); break;
               case kCXX_DirectMember:   pos = Complete( "[^. ]*$", pList, "(" ); break;
               case kCXX_IndirectMember: pos = Complete( "[^> ]*$", pList, "(" ); break;
               default:                  assert(0);                               break;
               }

               delete pList;
               delete pClass;

               if( context != original_context) pos = -2;
          }
          break;

     case kCXX_ScopeProto:
     case kCXX_DirectProto:
     case kCXX_IndirectProto:
     case kCXX_NewProto:
     case kCXX_ConstructorProto:
          {
               const EContext_t original_context = context; // save this for later

               // get class
               TClass* pClass;
               TString name;
               if( context == kCXX_NewProto )
               {
                    name = s3("[_a-zA-Z][_a-zA-Z0-9]* *($", 3);
                    name.Chop();
//                  name.Remove( TString::kTrailing, ' ' ); // fons: don't you think this would be nice?
                    name = name.Strip( TString::kTrailing, ' ' );
                    // "name" should now be the name of a class
               }
               else {
                    name = s3("^[_a-zA-Z][_a-zA-Z0-9]*");
                    // "name" may now be the name of a class, object, or pointer
               }
               IfDebug(cerr << endl);
               IfDebug(cerr << "name: " << '"' << name << '"' << endl);
               switch( context )
               {
               case kCXX_ScopeProto:       pClass = MakeClassFromClassName( name );        break;
               case kCXX_DirectProto:      pClass = MakeClassFromVarName( name, context ); break;
               case kCXX_IndirectProto:    pClass = MakeClassFromVarName( name, context ); break;
               case kCXX_NewProto:         pClass = MakeClassFromClassName( name );        break;
               case kCXX_ConstructorProto: pClass = MakeClassFromClassName( name );        break;
               default:                    assert(0);                                      break;
               }
               if( !pClass ) { pos = -2; break; }

               // get method name
               TString methodName;
               if( context == kCXX_ConstructorProto || context == kCXX_NewProto )
               {
                    // (constructor)
                    methodName = name;
               }
               else {
                    // (normal member function)
                    methodName = s3("[^:>\\.(]*($");
                    methodName.Chop();
//                  methodName.Remove( TString::kTrailing, ' ' ); // fons: don't you think this would be nice?
                    methodName = methodName.Strip( TString::kTrailing, ' ' );
               }
               IfDebug(cerr << methodName << endl);

               // get methods
               TContainer* pList = new TContainer;
               pList->AddAll( (TCollection*) pClass->GetListOfAllPublicMethods() );

               // print prototypes
               Bool_t   foundOne = kFALSE;
               TIter    nextMethod( pList );
               TMethod* pMethod;
               while(( pMethod = (TMethod*) nextMethod() ))
               {
                    if( methodName == pMethod->GetName() ) {
                         foundOne = kTRUE;
                         cout << endl
                              << pMethod->GetReturnTypeName()
                              << " "
                              << pMethod->GetName()
                              << pMethod->GetSignature();
                         const char* comment = pMethod->GetCommentString();
                         if( comment && comment[0]!='\0'  ) {
                              cout << " \t// " << comment;
                         }
                    }
               }

               // done
               if( foundOne ) {
                    cout << endl;
                    pos = -2;
               }
               else {
                    cout << "\a" << flush;
                    pos = -1;
               }

               // cleanup
               delete pList;
               delete pClass;

               if( context!=original_context ) pos = -2;
          }
          break;

     case kCXX_Global:
          {
               // first need to veto a few possibilities.
               int L2 = s2.Length(), L3 = s3.Length();

               // "abc().whatever[TAB]"
               if( L2>L3  &&  s2[ L2-L3-1 ]=='.' )
               {
                    cerr << endl << "tab completion not implemented for this context" << endl;
                    break; // veto
               }

               // "abc()->whatever[TAB]"
               if( L2>L3+1  &&  s2( L2-L3-2, 2 )=="->" )
               {
                   cerr << endl << "tab completion not implemented for this context" << endl;
                   break; // veto
               }

               TContainer* pList = new TContainer;

               // shouldn't TCollection::AddAll( TCollection* ) take a const argument?
               const TSeqCol* pL2 = GetListOfClasses();
               pList->AddAll(  (TSeqCol*) pL2 );
               //
               const TSeqCol* pC1 = GetListOfGlobals();
               pList->AddAll(  (TSeqCol*) pC1 );
               //
               const TSeqCol* pC3 = GetListOfGlobalFunctions();
               pList->AddAll(  (TSeqCol*) pC3 );

               pos = Complete( "[_a-zA-Z][_a-zA-Z0-9]*$", pList, "" );


               delete pList;
          }
          break;

     case kCXX_GlobalProto:
          {
               // get function name
               TString functionName = s3("[_a-zA-Z][_a-zA-Z0-9]*");
               IfDebug(cerr << functionName << endl);

               TContainer listOfMatchingGlobalFuncs;
               TIter      nextGlobalFunc (GetListOfGlobalFunctions());
               TObject*   pObj;
               while(( pObj = nextGlobalFunc() ))
               {
                    if( strcmp(pObj->GetName(), functionName)==0 )
                    {
                         listOfMatchingGlobalFuncs.Add( pObj );
                    }
               }

               if( listOfMatchingGlobalFuncs.IsEmpty() )
               {
                    cerr << endl << "no such function: " << dblquote(functionName) << endl;
               }
               else
               {
                    cout << endl;
                    TIter next (&listOfMatchingGlobalFuncs);
                    TFunction* pFunction;
                    while(( pFunction = (TFunction*) next() ))
                    {
                         cout
                              << pFunction->GetReturnTypeName()
                              << " "
                              << pFunction->GetName()
                              << pFunction->GetSignature()
                              << endl;
                    }
               }

               pos = -2;
          }
          break;

          /******************************************************************/
          /*                                                                */
          /* default: should never happen                                   */
          /*                                                                */
          /******************************************************************/
     default:
          assert(0);
          break;
     }

     return pos;
}
void TTabCom::InitPatterns( void )
{
     // [private]

     // add more patterns somewhere below.
     // add corresponding enum to "EContext_t"
     //
     // note:
     // 1. in some cases order is important ...
     //
     //    the order of the "case" statements in "switch( context )" in "TTabCom::Hook()" is Not important.
     //
     //    the order of the "SetPattern()" function calls below is Not important.
     //
     //    the order of the initializers in the "EContext_t" enumeration Is important
     //    because DetermineContext() goes through the array in order, and returns at the first match.
     //
     // 2. below, "$" will match cursor position

     SetPattern( kSYS_UserName, "~[_a-zA-Z0-9]*$" );
     SetPattern( kSYS_EnvVar,   "$[_a-zA-Z0-9]*$" );

     SetPattern( kCINT_stdout, "; *>>?.*$"  );   // stdout
     SetPattern( kCINT_stderr, "; *2>>?.*$" );   // stderr
     SetPattern( kCINT_stdin,  "; *<.*$"    );   // stdin

     SetPattern( kCINT_Load,  "^ *\\.L .*$" );
     SetPattern( kCINT_Exec,  "^ *\\.x [-0-9_a-zA-Z~$./]*$" );

     SetPattern( kCINT_pragma,     "^# *pragma +[_a-zA-Z0-9]*$" );
     SetPattern( kCINT_includeSYS, "^# *include *<[^>]*$"  );   // system files
     SetPattern( kCINT_includePWD, "^# *include *\"[^\"]*$" );  // local files

     SetPattern( kCINT_cpp, "^# *[_a-zA-Z0-9]*$"  );

     SetPattern( kROOT_Load, "gSystem *-> *Load *( *\"[^\"]*$" );

     SetPattern( kCXX_ScopeMember,    "[_a-zA-Z][_a-zA-Z0-9]* *:: *[_a-zA-Z0-9]*$" );
     SetPattern( kCXX_DirectMember,   "[_a-zA-Z][_a-zA-Z0-9]* *\\. *[_a-zA-Z0-9]*$" );
     SetPattern( kCXX_IndirectMember, "[_a-zA-Z][_a-zA-Z0-9]* *-> *[_a-zA-Z0-9]*$" );

     SetPattern( kCXX_ScopeProto,        "[_a-zA-Z][_a-zA-Z0-9]* *:: *[_a-zA-Z0-9]* *($" );
     SetPattern( kCXX_DirectProto,       "[_a-zA-Z][_a-zA-Z0-9]* *\\. *[_a-zA-Z0-9]* *($" );
     SetPattern( kCXX_IndirectProto,     "[_a-zA-Z][_a-zA-Z0-9]* *-> *[_a-zA-Z0-9]* *($" );
     SetPattern( kCXX_NewProto,          "new +[_a-zA-Z][_a-zA-Z0-9]* *($" );
     SetPattern( kCXX_ConstructorProto,  "[_a-zA-Z][_a-zA-Z0-9]* +[_a-zA-Z][_a-zA-Z0-9]* *($" );

     SetPattern( kSYS_FileName, "\"[-0-9_a-zA-Z~$./]*$" );
     SetPattern( kCXX_Global, "[_a-zA-Z][_a-zA-Z0-9]*$" );
     SetPattern( kCXX_GlobalProto, "[_a-zA-Z][_a-zA-Z0-9]* *($" );
}
TClass* TTabCom::MakeClassFromClassName( const char className[] ) const
{
     // [private]
     //   (does some specific error handling that makes the function unsuitable for general use.)
     //   returns a new'd TClass given the name of a class.
     //   user must delete.
     //   returns 0 in case of error.

     // the TClass constructor will print a Warning message for classes that don't exist
     // so, ignore warnings temporarily.
     NoMsg(kWarning);
     TClass* pClass = new TClass( className, 0 );
     NoMsg(-1);

     // make sure "className" exists
     if( pClass->Size()==0 )
     {
          // i'm assuming this happens iff there was some error.
          // (misspelled the class name, for example)
          cerr << endl << "class " << dblquote(className) << " not defined." << endl;
          return 0;
     }

     return pClass;
}
TClass* TTabCom::MakeClassFromVarName( const char varName[], EContext_t& context )
{
     // [private]
     //   (does some specific error handling that makes the function unsuitable for general use.)
     //   returns a new'd TClass given the name of a variable.
     //   user must delete.
     //   returns 0 in case of error.
     //   if user has operator.() or operator->() bacwards, will modify: context, *fpLoc and fBuf.
     //   context sensitive behevior.


     // need to make sure "varName" exists
     // because "DetermineClass()" prints clumsy error message otherwise.
     Bool_t varName_exists =
          GetListOfGlobals()->Contains(varName) || // check in list of globals first.
          (gROOT->FindObject(varName) != 0);       // then check CINT "shortcut #3"

     // not found...
     if( !varName_exists  ) {
          cerr << endl << "variable " << dblquote(varName) << " not defined." << endl;
          return 0; //* RETURN *//
     }

     /*****************************************************************************************/
     /*                                                                                       */
     /*  this section is really ugly.                                                         */
     /*  and slow.                                                                            */
     /*  it could be made a lot better if there was some way to tell whether or not a given   */
     /*  variable is a pointer or a pointer to a pointer.                                     */
     /*                                                                                       */
     /*****************************************************************************************/

     TString className = DetermineClass( varName );

     if( className.Length()<1 )
     {
          // this will happen if "varName" is a fundamental type (as opposed to class type).
          // or a pointer to a pointer.
          // or a function pointer.
          cerr << endl
               << "problem determining class of " << dblquote(varName)
               << endl;
          return 0; //* RETURN *//
     }

     Bool_t varIsPointer = className[ className.Length()-1 ]=='*';

     if( varIsPointer &&
         (context == kCXX_DirectMember ||
          context == kCXX_DirectProto     ))
     {
          // user is using operator.() instead of operator->()
          // ==>
          //      1. we are in wrong context.
          //      2. user is lazy
          //      3. or maybe confused

          // 1. fix the context
          switch( context )
          {
          case kCXX_DirectMember:
               context = kCXX_IndirectMember;
               break;
          case kCXX_DirectProto:
               context = kCXX_IndirectProto;
               break;
          default:
               assert(0);
               break;
          }

          // 2. fix the operator.
          int i;
          for( i=*fpLoc; fBuf[i]!='.'; i-=1 ) {
          }
          int loc = i;
          for( i=strlen(fBuf); i>=loc; i-=1 ) {
               fBuf[i+1] = fBuf[i];
          }
          fBuf[loc]='-';
          fBuf[loc+1]='>';
          *fpLoc += 1;

          // 3. inform the user.
          cerr << endl << dblquote(varName) << " is of pointer type. Use this operator: ->" << endl;
     }

     if( context == kCXX_IndirectMember ||
         context == kCXX_IndirectProto     )
     {
          if( varIsPointer ) {
               className.Chop(); // remove the '*'

               if( className[ className.Length()-1 ]=='*' ) {
                    cerr << endl << "can't handle pointers to pointers." << endl;
                    return 0; //* RETURN *//
               }
          }
          else {
               // user is using operator->() instead of operator.()
               // ==>
               //      1. we are in wrong context.
               //      2. user is lazy
               //      3. or maybe confused

               // 1. fix the context
               switch( context )
                    {
                    case kCXX_IndirectMember:
                         context = kCXX_DirectMember;
                         break;
                    case kCXX_IndirectProto:
                         context = kCXX_DirectProto;
                         break;
                    default:
                         assert(0);
                         break;
                    }

               // 2. fix the operator.
               int i;
               for( i=*fpLoc; fBuf[i-1]!='-' && fBuf[i]!='>'; i-=1 ) {
               }
               fBuf[i-1]='.';
               int len = strlen(fBuf);
               for( ; i<len; i+=1 ) {
                    fBuf[i] = fBuf[i+1];
               }
               *fpLoc -= 1;

               // 3. inform the user.
               cerr << endl << dblquote(varName) << " is not of pointer type. Use this operator: ." << endl;
          }
     }

     return new TClass( className, 0 );
}
void TTabCom::SetPattern( EContext_t handle, const char regexp[] )
{
     // [private]

     // prevent overflow
     if( handle >= kNUM_PAT ) {
          cerr
               << endl
               << "ERROR: handle="
               << (int)handle
               << " >= kNUM_PAT="
               << (int)kNUM_PAT
               << endl;
          return;
     }

     fRegExp[ handle ]  = regexp;
     Makepat( regexp, fPat[ handle ], MAX_LEN_PAT );
}
