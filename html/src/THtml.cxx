// @(#)root/html:$Name:  $:$Id: THtml.cxx,v 1.6 2001/02/06 17:35:55 brun Exp $
// Author: Nenad Buncic   18/10/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TROOT.h"
#include "TBaseClass.h"
#include "TVirtualPad.h"
#include "TClass.h"
#include "TClassTable.h"
#include "TDataMember.h"
#include "TDataType.h"
#include "TDatime.h"
#include "TEnv.h"
#include "TError.h"
#include "THtml.h"
#include "TMethod.h"
#include "TSystem.h"
#include "TString.h"
#include "TInterpreter.h"
#include "TRegexp.h"

#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <fstream.h>

THtml *gHtml = 0;

const Int_t   kSpaceNum      = 1;
const char   *formatStr      = "%12s %5s %s";

enum ESortType {kCaseInsensitive, kCaseSensitive};
enum EFileType {kSource, kInclude, kTree};


////////////////////////////////////////////////////////////////////////////////
//
//   The HyperText Markup Language (HTML) is a simple data format used to
// create hypertext documents that are portable from one platform to another.
// HTML documents are SGML documents with generic semantics that are
// appropriate for representing information from a wide range of domains.
//
//   The THtml class is designed to provide an easy way for converting ROOT
// classes, and files as well, into HTML documents. Here is the few rules
// and suggestions for a configuration, coding and usage.
//
//
// Configuration:
// -------------
//
//   The output directory could be specified using the Root.Html.OutputDir
// environment variable ( default value: "html/" ). Also it is necessary to
// define Root.Html.SourceDir to point to directories containing .cxx and .h
// files ( see: TEnv ).
//
//       Examples:
//                Root.Html.OutputDir: html
//                Root.Html.SourceDir: src:include:.:/usr/user/source
//                Root.Html.Root:      http://root.cern.ch/root/html
//
//
//   During the conversion, THtml will look for the certain number of
// user defined strings, i.e. author's name, copyright note, etc.
// This could be defined with following environment variables:
//
//       Root.Html.Author     ( default: // Author:)
//       Root.Html.LastUpdate ( default: // @(#))
//       Root.Html.Copyright  ( default:  * Copyright)
//
//
//
// Coding rules:
// ------------
//
//   A class description block, which must be placed before the first
// member function, has a following form:
//
//       ////////////////////////////////////////////////////////////////
//       //                                                            //
//       // TMyClass                                                   //
//       //                                                            //
//       // This is the description block.                             //
//       //                                                            //
//       ////////////////////////////////////////////////////////////////
//
//   The environment variable Root.Html.Description ( see: TEnv ) contents
// the delimiter string ( default value: //_________________ ). It means
// that you can also write your class description block like this:
//
//       //_____________________________________________________________
//       // A description of the class starts with the line above, and
//       // will take place here !
//       //
//
//   Note that EVERYTHING until the first non-commented line is considered
// as a valid class description block.
//
//   A member function description block starts immediately after '{'
// and looks like this:
//
//       void TWorld::HelloWorldFunc( string *text )
//       {
//       // This is an example of description for the
//       // TWorld member function
//
//          helloWorld.Print( text );
//       }
//
//   Like in a class description block, EVERYTHING until the first
// non-commented line is considered as a valid member function
// description block.
//
//   ==> The "Begin_Html" and "End_Html" special keywords <=========
//       --------------------------------------------
// You can insert pure html code in your comment lines. During the
// generation of the documentation, this code will be inserted as is
// in the html file.
// Pure html code must be inserted between the keywords "Begin_Html"
// and "End_Html" starting/finishing anywhere in the comment lines.
// Examples of pure html code are given in many Root classes.
// See for example the classes TDataMember and TMinuit.
//
//   ==> The escape character
//       --------------------
// Outside blocks starting with "Begin_Html" and finishing with "End_Html"
// one can prevent the automatic translation of symbols like "<" and ">"
// to "&lt;" and "&gt;" by using the escape character in front.
// The default escape character is backslash and can be changed
// via the member function SetEscape.
//
// Usage:
// -----
//
//     Root> gHtml.MakeAll               // invoke a make for all classes
//     Root> gHtml.MakeClass( TMyClass ) // create a HTML files for that class only
//     Root> gHtml.MakeIndex()           // creates an index files only
//     Root> gHtml.MakeTree( TMyClass )  // creates an inheritance tree for a class
//
//     Root> gHtml.Convert( hist1.mac, "Histogram example" )
//
//
// Environment variables:
// ---------------------
//
//   Root.Html.OutputDir    ( default: htmldoc/)
//   Root.Html.SourceDir    ( default: .:src/:include/)
//   Root.Html.Author       ( default: // Author:)
//   Root.Html.LastUpdate   ( default: // @(#))
//   Root.Html.Copyright    ( default:  * Copyright)
//   Root.Html.Description  ( default: //____________________ )
//   Root.Html.HomePage     ( URL to the user defined home page )
//   Root.Html.SearchEngine ( link to the search engine )
//
////////////////////////////////////////////////////////////////////////////////

ClassImp( THtml )

//______________________________________________________________________________
THtml::THtml()
{
   // Create a THtml object. Use object directly or via the global
   // pointer gHtml. In case output directory does not exist an error
   // will be printed and gHtml stays 0 also zombie bit will be set.

   fLen       = 1024;
   fLine      = new char [fLen];
   fCounter   = new char [6]; for (Int_t i=0;i<6;i++) fCounter[i] = 0;
   fEscFlag   = kFALSE;
   SetEscape();

   // get prefix for source directory
   fSourcePrefix = gEnv->GetValue( "Root.Html.SourcePrefix", "");

   // check for source directory
   fSourceDir = gEnv->GetValue( "Root.Html.SourceDir", "./:src/:include/" );

   // check for output directory
   fOutputDir = gEnv->GetValue( "Root.Html.OutputDir", "htmldoc/" );

   fXwho      = "http://consult.cern.ch/xwho/people?";

   Int_t  st;
   Long_t sId, sSize, sFlags, sModtime;
   if ((st = gSystem->GetPathInfo(fOutputDir, &sId, &sSize, &sFlags, &sModtime)) ||
       !(sFlags & 2)) {
      if (st == 0) {
         Error("THtml", "output directory %s is an existing file", fOutputDir);
         MakeZombie();
         return;
      }
      // Try creating directory
      if (gSystem->MakeDirectory(fOutputDir) == -1) {
         Error("THtml", "output directory %s does not exist", fOutputDir);
         MakeZombie();
         return;
      }
   }

   // insert html object in the list of special ROOT objects
   gHtml = this;
   gROOT->GetListOfSpecials()->Add(gHtml);
}


//______________________________________________________________________________
THtml::~THtml()
{
// Default destructor

    if( fLine    ) delete [] fLine;
    if( fCounter ) delete [] fCounter;

    fSourceDir = 0;
    fLen       = 0;
}


//______________________________________________________________________________
int CaseSensitiveSort( const void *name1, const void *name2 )
{
// Friend function for sorting strings, case sensitive
//
//
// Input: name1 - pointer to the first string
//        name2 - pointer to the second string
//
//  NOTE: This function compares its arguments and returns an integer less
//        than, equal to, or greater than zero, depending on whether name1
//        is lexicographically less than, equal to, or greater than name2.
//
//

    return( strcmp( *( (char **) name1 ), *( (char **) name2 )) );
}


//______________________________________________________________________________
int CaseInsensitiveSort( const void *name1, const void *name2 )
{
// Friend function for sorting strings, case insensitive
//
//
// Input: name1 - pointer to the first string
//        name2 - pointer to the second string
//
//  NOTE: This function compares its arguments and returns an integer less
//        than, equal to, or greater than zero, depending on whether name1
//        is lexicographically less than, equal to, or greater than name2,
//        but characters are forced to lower-case prior to comparison.
//
//

    return( strcasecmp( *( (char **) name1 ), *( (char **) name2 )) );
}


//______________________________________________________________________________
void THtml::Class2Html( TClass *classPtr, Bool_t force )
{
// It creates HTML file for a single class
//
//
// Input: classPtr - pointer to the class


    const char *tab = "<!--TAB-->";
    const char *tab2 = "<!--TAB2-->  ";
    const char *tab4 = "<!--TAB4-->    ";
    const char *tab6 = "<!--TAB6-->      ";

    gROOT->GetListOfGlobals( kTRUE );

    // create a filename
    char *tmp1 = gSystem->ExpandPathName( fOutputDir );
    char *tmp2 = gSystem->ConcatFileName( tmp1, classPtr->GetName() );

    char *filename = StrDup( tmp2, 6 );
    strcat( filename, ".html" );

    if( tmp1 ) delete [] tmp1;
    if( tmp2 ) delete [] tmp2;
    tmp1 = tmp2 = 0;

    if( IsModified( classPtr, kSource ) || force ) {

        // open class file
        ofstream classFile;
        classFile.open( filename, ios::out );

        Bool_t classFlag = kFALSE;


        if( classFile.good() ) {

            Printf( formatStr, "", fCounter, filename );

            // write a HTML header for the classFile file
            WriteHtmlHeader( classFile, classPtr->GetName() );

            // make a link to the description
            classFile << "<!--BEGIN-->" << endl;
            classFile << "<center>" << endl;
            classFile << "<h1>" << classPtr->GetName() << "</h1>" << endl;
            classFile << "<hr width=300>" << endl;
            classFile << "<!--SDL--><em><a href=#" << classPtr->GetName()
                      << ":description>class description</a>";

            // make a link to the '.cxx' file
            classFile << " - <a href=\"src/" << classPtr->GetName() << ".cxx.html\"";
            classFile << ">source file</a>";

            // make a link to the inheritance tree
            classFile << " - <a href=\"" << classPtr->GetName() << "_Tree.ps\"";
            classFile << ">inheritance tree</a>";


            classFile << "</em>" << endl;
            classFile << "<hr width=300>" << endl;
            classFile << "</center>" << endl;


            // make a link to the '.h' file
            classFile << "<h2>" << "class <a name=\"" << classPtr->GetName() << "\" href=\"";
            classFile << GetFileName( (const char * ) classPtr->GetDeclFileName() ) << "\"";
            classFile << ">" << classPtr->GetName() << "</a> ";

            // copy .h file to the Html output directory
            char *declf = GetSourceFileName(classPtr->GetDeclFileName());
            CopyHtmlFile(declf);
            delete [] declf;

            // make a loop on base classes
            Bool_t first = kTRUE;
            TBaseClass *inheritFrom;
            TIter nextBase( classPtr->GetListOfBases() );

            while (( inheritFrom = ( TBaseClass * ) nextBase() )) {
                if( first ) {
                    classFile << ": ";
                    first = kFALSE;
                }
                else classFile << ", ";
                classFile << "public ";

                // get a class
                TClass *classInh = GetClass( (const char * ) inheritFrom->GetName() );

                char *htmlFile = GetHtmlFileName( classInh );

                if( htmlFile ) {
                    classFile << "<a href=\"";

                    // make a link to the base class
                    classFile << htmlFile;
                    classFile << "\">" << inheritFrom->GetName() << "</a>";
                    delete [] htmlFile;
                    htmlFile = 0;
                }
                else classFile << inheritFrom->GetName();
            }

            classFile << "</h2>" << endl;
            classFile << "<pre>" << endl;


            // make a loop on member functions
            TMethod *method;
            TIter nextMethod( classPtr->GetListOfMethods() );

            Int_t len, maxLen[3];
            len = maxLen[0] = maxLen[1] = maxLen[2] = 0;

            // loop to get a pointers to a method names
            const Int_t nMethods = classPtr->GetNmethods();
            const char **methodNames = new const char*[3*2*nMethods];

            Int_t mtype, num[3];
            mtype = num[0] = num[1] = num[2] = 0;

            while (( method = ( TMethod * ) nextMethod() )) {

                if(
                    !strcmp( method->GetName(), "Dictionary"    ) ||
                    !strcmp( method->GetName(), "Class_Version" ) ||
                    !strcmp( method->GetName(), "Class_Name"    ) ||
                    !strcmp( method->GetName(), "DeclFileName"  ) ||
                    !strcmp( method->GetName(), "DeclFileLine"  ) ||
                    !strcmp( method->GetName(), "ImplFileName"  ) ||
                    !strcmp( method->GetName(), "ImplFileLine"  )
                ) continue;


                if( kIsPrivate & method->Property() )
                    mtype = 0;
                else if( kIsProtected & method->Property() )
                    mtype = 1;
                else if( kIsPublic & method->Property() )
                    mtype = 2;

                methodNames[mtype*2*nMethods+2*num[mtype]] = method->GetName();

                if (method->GetReturnTypeName() ) len = strlen( method->GetReturnTypeName() );
                else len = 0;

                if( kIsVirtual & method->Property() ) len += 8;
                if( kIsStatic & method->Property() ) len += 7;

                maxLen[mtype] = maxLen[mtype] > len ? maxLen[mtype] : len;

                const char* type = strrchr( method->GetReturnTypeName(), ' ' );
                if( ! type ) type = method->GetReturnTypeName();
                else type++;

                if( classPtr && !strcmp( type, classPtr->GetName() ))
                    methodNames[mtype*2*nMethods+2*num[mtype]] = "A00000000";

                // if this is the destructor
                while( '~' == *methodNames[mtype*2*nMethods+2*num[mtype]] )
                    methodNames[mtype*2*nMethods+2*num[mtype]] = "A00000001";

                methodNames[mtype*2*nMethods+2*num[mtype]+1] = (char *) method;

                num[mtype]++;
            }

            Int_t i, j;

            for( j = 0; j < 3; j ++ ) {
                if( *( methodNames+j*2*nMethods ) ) {
                    qsort( methodNames+j*2*nMethods, num[j], 2*sizeof( methodNames ), CaseInsensitiveSort );

                    const char *ftitle = 0;
                    switch( j ) {
                        case 0: ftitle = "private:";
                                break;
                        case 1: ftitle = "protected:";
                                break;
                        case 2: ftitle = "public:";
                                break;
                    }
                    if( j ) classFile << endl;
                    classFile << tab4 << "<b>" << ftitle << "</b><br>" << endl;

                    for( i = 0; i < num[j]; i++ ) {
                        method = (TMethod *) methodNames[j*2*nMethods+2*i+1];

                        if( method ) {
                            Int_t w = 0;
                            if( method->GetReturnTypeName() ) len = strlen( method->GetReturnTypeName() );
                            else len = 0;

                            if( kIsVirtual & method->Property() ) len += 8;
                            if( kIsStatic & method->Property() ) len += 7;

                            classFile << tab6;
                            for( w = 0; w < ( maxLen[j]-len ); w++ )
                                classFile << " ";

                            if( kIsVirtual & method->Property() )
                                classFile << "virtual ";

                            if( kIsStatic & method->Property() )
                                classFile << "static ";

                            strcpy( fLine, method->GetReturnTypeName() );
                            ExpandKeywords( classFile, fLine, classPtr, classFlag );

                            classFile << " " << tab << "<!--BOLD-->";
                            classFile << "<a href=\"#" << classPtr->GetName();
                            classFile << ":";
                            ReplaceSpecialChars( classFile, method->GetName() );
                            classFile << "\">";
                            ReplaceSpecialChars( classFile, method->GetName() );
                            classFile << "</a><!--PLAIN-->";

                            strcpy( fLine, method->GetSignature() );
                            ExpandKeywords( classFile, fLine, classPtr, classFlag );
                            classFile << endl;
                        }
                    }
                }
            }

            delete [] methodNames;

            // make a loop on data members
            first = kFALSE;
            TDataMember *member;
            TIter nextMember( classPtr->GetListOfDataMembers() );


            Int_t len1, len2, maxLen1[3], maxLen2[3];
            len1 = len2 = maxLen1[0] = maxLen1[1] = maxLen1[2] = 0;
            maxLen2[0] = maxLen2[1] = maxLen2[2] = 0;
            mtype = num[0] = num[1] = num[2] = 0;

            Int_t ndata = classPtr->GetNdata();

            // if data member exist
            if( ndata ) {
                TDataMember **memberArray = new TDataMember*[3*ndata];

                if( memberArray ) {
                    while (( member = ( TDataMember * ) nextMember() )) {

                        if(
                            !strcmp( member->GetName(), "fgIsA" )
                        ) continue;

                        if( kIsPrivate & member->Property() )
                            mtype = 0;
                        else if( kIsProtected & member->Property() )
                            mtype = 1;
                        else if( kIsPublic & member->Property() )
                            mtype = 2;

                        memberArray[mtype*ndata+num[mtype]] = member;
                        num[mtype]++;

                        if( member->GetFullTypeName() )
                             len1 = strlen( (char * ) member->GetFullTypeName() );
                        else len1 = 0;
                        if( member->GetName() )
                             len2 = strlen( member->GetName() );
                        else len2 = 0;

                        if( kIsStatic & member->Property() ) len1 += 7;

                       // Take in account the room the array index will occupy

                        Int_t dim = member->GetArrayDim();
                        for (Int_t indx = 0; indx < dim; indx++ ) {
                              len2 += Int_t(TMath::Log10(member->GetMaxIndex(indx))) + 3;
                         }

                        maxLen1[mtype] = maxLen1[mtype] > len1 ? maxLen1[mtype] : len1;
                        maxLen2[mtype] = maxLen2[mtype] > len2 ? maxLen2[mtype] : len2;
                    }

                    classFile << endl;
                    classFile << "<h3>" << tab2 << "<a name=\"";
                    classFile << classPtr->GetName();
                    classFile << ":Data Members\">Data Members</a></h3>" << endl;

                    for( j = 0; j < 3; j++ ) {
                        if( memberArray[j*ndata] ) {
                            const char *ftitle = 0;
                            switch( j ) {
                                case 0: ftitle = "private:";
                                        break;
                                case 1: ftitle = "protected:";
                                        break;
                                case 2: ftitle = "public:";
                                        break;
                            }
                            if( j ) classFile << endl;
                            classFile << tab4 << "<b>" << ftitle << "</b><br>" << endl;

                            for( i = 0; i < num[j]; i++ ) {
                                Int_t w = 0;
                                member = memberArray[j*ndata+i];

                                classFile << tab6;
                                if ( member->GetFullTypeName() ) len1 = strlen( member->GetFullTypeName() );
                                else len1 = 0;

                                if( kIsStatic & member->Property() ) len1 += 7;

                                for( w = 0; w < ( maxLen1[j]-len1 ); w++ )
                                    classFile << " ";

                                if( kIsStatic & member->Property() )
                                    classFile << "static ";

                                strcpy( fLine, member->GetFullTypeName() );
                                ExpandKeywords( classFile, fLine, classPtr, classFlag );

                                classFile << " " << tab << "<!--BOLD-->";
                                classFile << "<a name=\"" << classPtr->GetName() << ":";
                                classFile << member->GetName();
                                classFile << "\">" << member->GetName();

                                // Add the dimensions to "array" members

                                Int_t dim = member->GetArrayDim();
                                Int_t indx = 0;
                                Int_t indxlen = 0;
                                while (indx < dim ){
                                     if (member->GetMaxIndex(indx) <=0) break;
                                     classFile <<  "[" << member->GetMaxIndex(indx)<<"]";
                                     // Take in account the room this index will occupy
                                     indxlen += Int_t(TMath::Log10(member->GetMaxIndex(indx))) + 3;
                                     indx++;
                                }

                                classFile << "</a><!--PLAIN--> ";

                                len2 = 0;
                                if( member->GetName() )
                                      len2 = strlen( member->GetName() ) + indxlen;

                                for( w = 0; w < ( maxLen2[j]-len2 ); w++ )
                                    classFile << " ";
                                classFile << " " << tab;

                                classFile << "<i><a name=\"Title:";
                                classFile << member->GetName();

                                classFile << "\">";

                                strcpy( fLine, member->GetTitle() );
                                ReplaceSpecialChars( classFile, fLine );
                                classFile << "</a></i>" << endl;
                            }
                        }
                    }
                    classFile << "</pre>" << endl;
                    delete [] memberArray;
                }
            }

            classFile << "<!--END-->" << endl;

            // create a 'See also' part
            DerivedClasses( classFile, classPtr );

            // process a '.cxx' file
            ClassDescription( classFile, classPtr, classFlag );


            // close a file
            classFile.close();

        }
        else Error( "Make", "Can't open file '%s' !", filename );
    }
    else Printf( formatStr, "-no change-", fCounter, filename );

    if( filename ) delete [] filename;
    filename = 0;
}


//______________________________________________________________________________
void THtml::ClassDescription( ofstream &out, TClass *classPtr, Bool_t &flag )
{
// This function builds the description of the class
//
//
// Input: out      - output file stream
//        classPtr - pointer to the class
//        flag     - this is a 'begin_html/end_html' flag
//

    char  *ptr, *key;
    Bool_t tempFlag = kFALSE;
    char  *filename = 0;


    // allocate memory
    char *nextLine    = new char [256];
    char *pattern     = new char [80];

    char *lastUpdate  = new char [256];
    char *author      = new char [80];
    char *copyright   = new char [80];


    char *funcName   = new char [64];

    const char *lastUpdateStr;
    const char *authorStr;
    const char *copyrightStr;
    const char *descriptionStr;


    // just in case
    *lastUpdate = *author = *copyright = 0;


    // define pattern
    strcpy( pattern, classPtr->GetName() );
    strcat( pattern, "::" );
    Int_t len = strlen( pattern );


    // get environment variables
    lastUpdateStr  = gEnv->GetValue( "Root.Html.LastUpdate", "// @(#)" );
    authorStr      = gEnv->GetValue( "Root.Html.Author", "// Author:" );
    copyrightStr   = gEnv->GetValue( "Root.Html.Copyright", " * Copyright" );
    descriptionStr = gEnv->GetValue( "Root.Html.Description", "//____________________" );


    // find a .cxx file
    char *tmp1 = GetSourceFileName(classPtr->GetImplFileName());
    char *realFilename = StrDup( tmp1, 16 );
    if( !realFilename ) Error( "Make", "Can't find file '%s' !", tmp1 );

    if( tmp1 ) delete [] tmp1;
    tmp1 = 0;

    Bool_t classDescription    = kTRUE;

    Bool_t foundLastUpdate     = kFALSE;
    Bool_t foundAuthor         = kFALSE;
    Bool_t foundCopyright      = kFALSE;

    Bool_t firstCommentLine    = kTRUE;
    Bool_t extractComments     = kFALSE;
    Bool_t thisLineIsCommented = kFALSE;
    Bool_t thisLineIsPpLine    = kFALSE;

    // Class Description Title
    out << "<hr>" << endl;
    out << "<!--DESCRIPTION-->";
    out << "<h2><a name=\"" << classPtr->GetName();
    out << ":description\">Class Description</a></h2>" << endl;


    // open source file
    ifstream sourceFile;
    sourceFile.open( realFilename, ios::in );


    if( sourceFile.good() ) {
        // open a .cxx.html file
        tmp1 = gSystem->ExpandPathName( fOutputDir );
        char *tmp2 = gSystem->ConcatFileName( tmp1, "src" );
        char *dirname = StrDup( tmp2 );

        if( tmp1 ) delete [] tmp1;
        if( tmp2 ) delete [] tmp2;
        tmp1 = tmp2 = 0;

        // create directory if necessary
        if( gSystem->AccessPathName( dirname ))
            gSystem->MakeDirectory( dirname );

        tmp1 = gSystem->ConcatFileName( dirname, classPtr->GetName() );
        filename = StrDup( tmp1, 16 );
        strcat( filename, ".cxx.html" );

        ofstream tempFile;
        tempFile.open( filename, ios::out );

        if( dirname ) delete [] dirname;

        if( tmp1 ) delete [] tmp1;
        tmp1 = 0;

        if( tempFile.good() ) {


            // create an array of method names
            Int_t i = 0;
            TMethod *method;
            TIter nextMethod( classPtr->GetListOfMethods() );
            Int_t numberOfMethods = classPtr->GetNmethods();
            const char **methodNames = new const char* [2*numberOfMethods];
            while (( method = ( TMethod * ) nextMethod() )) {
                methodNames[2*i]   = method->GetName();
                methodNames[2*i+1] = ( const char * ) method;
                i++;
            }


            // write a HTML header
            char *sourceTitle = StrDup( classPtr->GetName(), 16 );
            strcat( sourceTitle, " - source file" );
            WriteHtmlHeader( tempFile, sourceTitle );
            if( sourceTitle ) delete [] sourceTitle;


            tempFile << "<pre>" << endl;

            while( !sourceFile.eof() ) {

                sourceFile.getline( fLine, fLen-1 );
                if( sourceFile.eof() ) break;


                // set start & end of the line
                if (!fLine) {
                   fLine = (char *) " ";
                   Warning("ClassDescription", "found an empty line");
                }
                char *startOfLine = fLine;
                char *endOfLine   = fLine + strlen( fLine ) - 1;


                // remove leading spaces
                while( isspace( *startOfLine )) startOfLine++;

                // remove trailing spaces
                while( isspace( *endOfLine )) endOfLine--;
                if( *startOfLine == '#' && !tempFlag )
                    thisLineIsPpLine = kTRUE;

                // if this line is a comment line
                else if( !strncmp( startOfLine, "//", 2 )) {

                    thisLineIsCommented = kTRUE;
                    thisLineIsPpLine    = kFALSE;

                    // remove a repeating characters from the end of the line
                    while( (*endOfLine == *startOfLine ) &&
                            ( endOfLine >= startOfLine )) endOfLine--;
                    endOfLine++;
                    char tempChar = *endOfLine;
                    *endOfLine = 0;


                    if( extractComments ) {
                        if( firstCommentLine ) {
                            out << "<pre>";
                            firstCommentLine = kFALSE;
                        }
                        if( endOfLine >= startOfLine+2 )
                            ExpandKeywords( out, startOfLine+2, classPtr, flag );
                        out << endl;
                    }

                    *endOfLine = tempChar;

                    // if line is composed of the same characters
                    if( (endOfLine == startOfLine ) && *( startOfLine+2 ) && classDescription ) {
                        extractComments  = kTRUE;
                        classDescription = kFALSE;
                    }
                }
                else {
                    thisLineIsCommented = kFALSE;
                    if( flag ) {
                        out << fLine << endl;
                    }
                    else {
                        extractComments = kFALSE;
                        if( !firstCommentLine ) {
                            out << "</pre>";
                            firstCommentLine = kTRUE;
                        }
                    }
                }


                // if NOT member function
                key = strstr( fLine, pattern );
                if( !key ) {
                    // check for a lastUpdate string
                    if( !foundLastUpdate && lastUpdateStr) {
                        if( !strncmp( fLine, lastUpdateStr, strlen( lastUpdateStr )) ) {
                            strcpy( lastUpdate, fLine+strlen( lastUpdateStr ));
                            foundLastUpdate = kTRUE;
                        }
                    }

                    // check for an author string
                    if( !foundAuthor && authorStr) {
                        if( !strncmp( fLine, authorStr, strlen( authorStr )) ) {
                            strcpy( author, fLine+strlen( authorStr ));
                            foundAuthor = kTRUE;
                        }
                    }

                    // check for a copyright string
                    if( !foundCopyright && copyrightStr) {
                        if( !strncmp( fLine, copyrightStr, strlen( copyrightStr )) ) {
                            strcpy( copyright, fLine+strlen( copyrightStr ));
                            foundCopyright = kTRUE;
                        }
                    }

                    // check for a description comments
                    if( descriptionStr && !strncmp( fLine, descriptionStr, strlen( descriptionStr )) ) {
                        if( classDescription ) {
                            // write description out
                            classDescription = kFALSE;
                            extractComments = kTRUE;
                        }
                    }
                }
                else {
                    Bool_t found = kFALSE;
                    // find method name
                    char *funcName = key + len;

                    while( *funcName && isspace( *funcName ) )
                        funcName++;
                    char *nameEndPtr = funcName;

                    // In case of destructor
                    if( *nameEndPtr == '~' ) nameEndPtr++;

                    while( *nameEndPtr && IsName( *nameEndPtr ) )
                        nameEndPtr++;

                    char c1 = *nameEndPtr;
                    char pe = 0;

                    char *params, *paramsEnd;
                    params = nameEndPtr;
                    paramsEnd = NULL;

                    while( *params  && isspace( *params ) ) params++;
                    if( *params != '(') params = NULL;
                    else params++;
                    paramsEnd = params;

                    // if signature exist, try to find the ending character
                    if( paramsEnd ) {
                        Int_t count = 1;
                        while( *paramsEnd ) {
                            if( *paramsEnd == '(') count++;
                            if( *paramsEnd == ')')
                                if( !--count ) break;
                            paramsEnd++;
                        }
                        pe = *paramsEnd;
                        *paramsEnd = 0;
                    }
                    *nameEndPtr = 0;

                    // get method
                    TMethod *method;
                    method = classPtr->GetMethodAny( funcName );

                    // restore characters
                    if( paramsEnd )  *paramsEnd = pe;
                    if( nameEndPtr ) *nameEndPtr = c1;

                    if( method ) {
                        char *typeEnd = NULL;
                        char c2 = 0;

                        found = kFALSE;

                        // try to get type
                        typeEnd = key-1;
                        while( ( typeEnd > fLine ) && (isspace( *typeEnd ) || *typeEnd == '*') )
                            typeEnd--;
                        typeEnd++;
                        c2 = *typeEnd;
                        *typeEnd = 0;
                        char *type = typeEnd - 1;
                        while( IsName( *type ) && ( type > fLine ))
                            type--;
                        if( !IsWord( *type )) type++;

                        while( (type > fLine ) && isspace( *( type-1 )) )
                            type--;
                        if( type > fLine ) {
                            if( !strncmp( type-5, "const", 5 ))
                                found = kTRUE;
                            else found = kFALSE;
                        }
                        else if( type == fLine )
                            found = kTRUE;

                        if( !strcmp( type, "void" ) && ( *funcName == '~' ) )
                            found = kTRUE;

                        *typeEnd = c2;

                        if( found ) {
                            ptr = strchr( nameEndPtr, '{');
                            char *semicolon = strchr( nameEndPtr, ';');
                            if( semicolon )
                                if( !ptr || ( semicolon < ptr )) found = kFALSE;

                            if( !ptr && found ) {
                                found = kFALSE;
                                while( sourceFile.getline( nextLine, 255 ) && fLine && nextLine &&
                                ( strlen( fLine ) < ( fLen-strlen( nextLine )) )) {
                                    strcat( fLine, "\n" );
                                    strcat( fLine, nextLine );
                                    if (( ptr = strchr( fLine, '{') )) {
                                        found = kTRUE;
                                        *ptr = 0;
                                        break;
                                    }
                                }
                            }
                            else if( ptr ) *ptr = 0;

                            if( found ) {
                                char *colonPtr = strrchr( fLine, ':');
                                if( colonPtr > funcName ) *colonPtr = 0;
                                if( found ) {
                                    out << "<hr>" << endl;
                                    out << "<!--FUNCTION-->";
                                    if( typeEnd ) {
                                        c2 = *typeEnd;
                                        *typeEnd = 0;
                                        ExpandKeywords( out, fLine, classPtr, flag );
                                        *typeEnd = c2;
                                        while( typeEnd < key ) {
                                            if( *typeEnd == '*')
                                                out << *typeEnd;
                                            typeEnd++;
                                        }
                                    }
                                    *nameEndPtr = 0;

                                    out << " <a name=\"" << classPtr->GetName() << ":";
                                    out << funcName << "\" href=\"src/";
                                    out << classPtr->GetName() << ".cxx.html#" << classPtr->GetName() << ":";
                                    ReplaceSpecialChars( out, funcName );
                                    out << "\">";
                                    ReplaceSpecialChars( out, funcName );
                                    out << "</a>";

                                    tempFile << "<a name=\"" << classPtr->GetName() << ":";
                                    ReplaceSpecialChars( tempFile, funcName );
                                    tempFile << "\"> </a>";

                                    // remove this method name from the list of methods
                                    i = 0;
                                    while( i < numberOfMethods ) {
                                        const char *mptr = methodNames[2*i];
                                        if( mptr ) {
                                           while( *mptr == '*') mptr++;
                                           if( !strcmp( mptr, funcName )) {
                                               methodNames[2*i] = NULL;
                                               break;
                                           }
                                        }
                                        i++;
                                    }

                                    *nameEndPtr = c1;
                                    if( colonPtr ) *colonPtr = ':';
                                    ExpandKeywords( out, nameEndPtr, classPtr, flag );
                                    out << "<br>" << endl;
                                    extractComments = kTRUE;
                                }
                            }
                            if( ptr ) *ptr = '{';
                        }
                    }
                }

                // write to '.cxx.html' file
                if( thisLineIsPpLine )
                    ExpandPpLine( tempFile, fLine );
                else {
                    if( thisLineIsCommented ) tempFile << "<b>";
                    ExpandKeywords( tempFile, fLine, classPtr, tempFlag, "../" );
                    if( thisLineIsCommented ) tempFile << "</b>";
                }
                tempFile << endl;
            }
            tempFile << "</pre>" << endl;

            // do some checking
            Bool_t inlineFunc = kFALSE;
            i = 0;
            while( i++ < numberOfMethods ) {
                if( methodNames[2*i] ) {
                    inlineFunc = kTRUE;
                    break;
                }
            }


            if( inlineFunc ) {
                out << "<br><br><br>" << endl;
                out << "<h3>Inline Functions</h3>" << endl;
                out << "<hr>" << endl;
                out << "<pre>" << endl;

                Int_t maxlen = 0, len = 0;
                for( i = 0; i < numberOfMethods; i++ ) {
                    if( methodNames[2*i] ) {
                        method = ( TMethod * ) methodNames[2*i+1];
                        if ( method->GetReturnTypeName() ) len = strlen( method->GetReturnTypeName() );
                        else len = 0;
                        maxlen = len > maxlen ? len : maxlen;
                    }
                }


                // write out an inline functions
                for( i = 0; i < numberOfMethods; i++ ) {
                    if( methodNames[2*i] ) {

                        method = ( TMethod * ) methodNames[2*i+1];

                        if( method ) {

                            if(
                                !strcmp( method->GetName(), "Dictionary"    ) ||
                                !strcmp( method->GetName(), "Class_Version" ) ||
                                !strcmp( method->GetName(), "Class_Name"    ) ||
                                !strcmp( method->GetName(), "DeclFileName"  ) ||
                                !strcmp( method->GetName(), "DeclFileLine"  ) ||
                                !strcmp( method->GetName(), "ImplFileName"  ) ||
                                !strcmp( method->GetName(), "ImplFileLine"  )
                            ) continue;

                            out << "<!--INLINE FUNCTION-->";
                            if(method->GetReturnTypeName() ) len = strlen( method->GetReturnTypeName() );
                            else len = 0;

                            out << "<!--TAB6-->      ";
                            while( len++ < maxlen+2 ) out << " ";

                            char *tmpstr = StrDup( method->GetReturnTypeName() );
                            if( tmpstr ) {
                                ExpandKeywords( out, tmpstr, classPtr, flag );
                                delete [] tmpstr;
                            }

                            out << " <a name=\"" << classPtr->GetName();
                            out << ":" << method->GetName() << "\" href=\"";
                            out << GetFileName( classPtr->GetDeclFileName() ) << "\">";
                            out << method->GetName() << "</a>";

                            strcpy( fLine, method->GetSignature() );
                            ExpandKeywords( out, fLine, classPtr, flag );
                            out << endl;
                        }
                    }
                }
                out << "</pre>" << endl;
            }


            // write tempFile footer
            WriteHtmlFooter( tempFile, "../" );

            // close a temp file
            tempFile.close();

            delete [] methodNames;
        }
        else Error( "MakeClass", "Can't open file '%s' !", filename );

        // close a source file
        sourceFile.close();

    }
    else Error( "Make", "Can't open file '%s' !", realFilename );


    // write classFile footer
    WriteHtmlFooter( out, "",  lastUpdate, author, copyright );

    // free memory
    if( nextLine )     delete [] nextLine;
    if( pattern )      delete [] pattern;

    if( lastUpdate )   delete [] lastUpdate;
    if( author )       delete [] author;
    if( copyright )    delete [] copyright;
    if( funcName )     delete [] funcName;

    if( realFilename ) delete [] realFilename;
    if( filename )     delete [] filename;
}


//______________________________________________________________________________
void THtml::ClassTree( TVirtualPad *psCanvas, TClass *classPtr, Bool_t force )
{
// It makes a class tree
//
//
// Input: psCanvas - pointer to the current canvas
//        classPtr - pointer to the class
//

    if( psCanvas && classPtr ) {
        char *tmp1 = gSystem->ConcatFileName( gSystem->ExpandPathName( fOutputDir ), classPtr->GetName() );
        char *filename = StrDup( tmp1 , 16 );

        strcat( filename, "_Tree.ps" );

        if( tmp1 ) delete [] tmp1;
        tmp1 = 0;

        if( IsModified( classPtr, kTree ) || force ) {
            Printf( formatStr, "", "", filename );
            classPtr->Draw( "same" );
            psCanvas->SaveAs(filename);
        }
        else Printf( formatStr, "-no change-", "", filename );

        if( filename ) delete [] filename;
    }
}


//______________________________________________________________________________
void THtml::Convert( const char *filename, const char *title, const char *dirname )
{
// It converts a single text file to HTML
//
//
// Input: filename - name of the file to convert
//        title    - title which will be placed at the top of the HTML file
//        dirname  - optional parameter, if it's not specified, output will
//                   be placed in html/examples directory.
//
//  NOTE: Output file name is the same as filename, but with extension .html
//

    const char *dir;
    char *ptr;

    Bool_t isCommentedLine = kFALSE;
    Bool_t tempFlag = kFALSE;

    // if it's not defined, make the "examples" as a default directory
    if( !*dirname ) {
        dir = gSystem->ConcatFileName( gSystem->ExpandPathName( fOutputDir ), "examples" );

        // create directory if necessary
        if( gSystem->AccessPathName( dir ))
            gSystem->MakeDirectory( dir );
    }
    else dir = dirname;


    // find a file
    char *realFilename = gSystem->Which( fSourceDir, filename, kReadPermission );

    if( realFilename ) {

        // open source file
        ifstream sourceFile;
        sourceFile.open( realFilename, ios::in );

        delete [] realFilename;
        realFilename = 0;

        if( sourceFile.good() ) {

            // open temp file with extension '.html'
            if( !gSystem->AccessPathName( dir )) {
                char *tmp1 = gSystem->ConcatFileName( dir, GetFileName( filename ));
                char *htmlFilename = StrDup( tmp1, 16 );
                strcat( htmlFilename, ".html" );

                if( tmp1 ) delete [] tmp1;
                tmp1 = 0;

                ofstream tempFile;
                tempFile.open( htmlFilename, ios::out );

                if( tempFile.good() ) {

                    Printf( "Convert: %s", htmlFilename );

                    // write a HTML header
                    WriteHtmlHeader( tempFile, title );

                    tempFile << "<h1>" << title << "</h1>" << endl;
                    tempFile << "<pre>" << endl;

                    while( !sourceFile.eof() ) {
                        sourceFile.getline( fLine, fLen-1 );
                        if( sourceFile.eof() ) break;


                        // remove leading spaces
                        ptr = fLine;
                        while( isspace( *ptr )) ptr++;


                        // check for a commented line
                        if( !strncmp( ptr, "//", 2 )) isCommentedLine = kTRUE;
                        else isCommentedLine = kFALSE;


                        // write to a '.html' file
                        if( isCommentedLine ) tempFile << "<b>";
                        gROOT->GetListOfGlobals(kTRUE); // force update of this list
                        ExpandKeywords( tempFile, fLine, NULL, tempFlag, "../" );
                        if( isCommentedLine ) tempFile << "</b>";
                        tempFile << endl;
                    }
                    tempFile << "</pre>" << endl;


                    // write a HTML footer
                    WriteHtmlFooter( tempFile, "../" );


                    // close a temp file
                    tempFile.close();

                }
                else Error( "Convert", "Can't open file '%s' !", htmlFilename );

                // close a source file
                sourceFile.close();
                if( htmlFilename ) delete [] htmlFilename;
                htmlFilename = 0;
            }
            else Error( "Convert", "Directory '%s' doesn't exist, or it's write protected !", dir );
        }
        else Error( "Convert", "Can't open file '%s' !", realFilename );
    }
    else Error( "Convert", "Can't find file '%s' !", filename );
}


//______________________________________________________________________________
Bool_t THtml::CopyHtmlFile( const char *sourceName, const char *destName )
{
// Copy file to HTML directory
//
//
//  Input: sourceName - source file name
//         destName   - optional destination name, if not
//                      specified it would be the same
//                      as the source file name
//
// Output: TRUE if file is successfully copied, or
//         FALSE if it's not
//
//
//   NOTE: The destination directory is always fOutputDir
//

    Bool_t ret = kFALSE;
    Int_t check = 0;

    // source file name
    char *tmp1 = gSystem->Which( fSourceDir, sourceName, kReadPermission );
    char *sourceFile = StrDup( tmp1, 16 );

    if( tmp1 ) delete [] tmp1;
    tmp1 = 0;

    if( sourceFile ) {

        // destination file name
        char *tmpstr = 0;
        if( !*destName ) tmpstr = StrDup( GetFileName( sourceFile ), 16 );
        else tmpstr = StrDup( GetFileName( destName ), 16 );
        destName = tmpstr;

        tmp1 = gSystem->ConcatFileName( gSystem->ExpandPathName( fOutputDir ), destName );
        char *filename = StrDup( tmp1, 16 );

        if( tmp1 ) delete [] tmp1;
        tmp1 = 0;

        // Get info about a file
        Long_t sId, sSize, sFlags, sModtime;
        Long_t dId, dSize, dFlags, dModtime;
        if( !( check = gSystem->GetPathInfo( sourceFile, &sId, &sSize, &sFlags, &sModtime )) )
            check = gSystem->GetPathInfo( filename, &dId, &dSize, &dFlags, &dModtime );


        if( (sModtime != dModtime ) || check ) {

            char *cmd = new char[256];

#ifdef R__UNIX
            strcpy( cmd, "/bin/cp " );
            strcat( cmd, sourceFile );
            strcat( cmd, " " );
            strcat( cmd, filename );
#endif

#ifdef WIN32
            strcpy( cmd, "copy \"" );
            strcat( cmd, sourceFile );
            strcat( cmd, "\" \"" );
            strcat( cmd, filename );
            strcat( cmd, "\"");
            char *bptr = 0;
            while( bptr = strchr( cmd, '/') )
                *bptr = '\\';
#endif

            ret = !gSystem->Exec( cmd );

            delete [] cmd;
            delete [] filename;
            delete [] tmpstr;
            delete [] sourceFile;
        }
    }
    else Error( "Copy", "Can't copy file '%s' to '%s' directory !", sourceName, fOutputDir );

    return( ret );
}



//______________________________________________________________________________
void THtml::CreateIndex( const char **classNames, Int_t numberOfClasses )
{
// Create an index
//
//
// Input: classNames      - pointer to an array of class names
//        numberOfClasses - number of elements
//

    Int_t i, len, maxLen = 0;

    char *tmp1 = gSystem->ConcatFileName( gSystem->ExpandPathName( fOutputDir ), "ClassIndex.html" );
    char *filename = StrDup( tmp1 );

    if( tmp1 ) delete [] tmp1;
    tmp1 = 0;

    // open indexFile file
    ofstream indexFile;
    indexFile.open( filename, ios::out );

    for( i = 0; i < numberOfClasses; i++ ) {
        len    = strlen( classNames[i] );
        maxLen = maxLen > len ? maxLen : len;
    }

    if( indexFile.good() ) {

        Printf( formatStr, "", fCounter, filename );

        // write indexFile header
        WriteHtmlHeader( indexFile, "Class Index" );
        indexFile << "<h1>Index</h1>" << endl;

        // check for a search engine
        const char *searchEngine = gEnv->GetValue( "Root.Html.SearchEngine", "" );

        // if exists ...
        if( *searchEngine ) {

            // create link to search engine page
            indexFile << "<h2><a href=\"" << searchEngine
                      << "\">Search the Class Reference Guide</a></h2>" << endl;

        }

        indexFile << "<hr>" << endl;
        indexFile << "<pre>" << endl;
        indexFile << "<ul>" << endl;

        // loop on all classes
        for( i = 0; i < numberOfClasses; i++ ) {

            // get class
            TClass *classPtr = GetClass( (const char * ) classNames[i] );

            indexFile << "<li>";
            char *htmlFile = GetHtmlFileName( classPtr );
            if( htmlFile ) {
                indexFile << "<a name=\"";
                indexFile << classNames[i];
                indexFile << "\" href=\"";
                indexFile << htmlFile;
                indexFile << "\">";
                indexFile << classNames[i];
                indexFile << "</a> ";
                delete [] htmlFile;
                htmlFile = 0;
            }
            else indexFile << classNames[i];


            // write title
            len = strlen( classNames[i] );
            for( Int_t w = 0; w < ( maxLen-len+2 ); w++ )
                indexFile << ".";
            indexFile << " ";

            indexFile << "<a name=\"Title:";
            indexFile << classPtr->GetName();
            indexFile << "\">";
            ReplaceSpecialChars( indexFile, classPtr->GetTitle() );
            indexFile << "</a>" << endl;
        }

        indexFile << "</ul>" << endl;
        indexFile << "</pre>" << endl;


        // write indexFile footer
        TDatime date;
        WriteHtmlFooter( indexFile, "", date.AsString() );


        // close file
        indexFile.close();

    }
    else Error( "MakeIndex", "Can't open file '%s' !", filename );

    if( filename ) delete [] filename;
}


//______________________________________________________________________________
void THtml::CreateIndexByTopic( char **fileNames, Int_t numberOfNames, Int_t maxLen )
{
// It creates several index files
//
//
// Input: fileNames     - pointer to an array of file names
//        numberOfNames - number of elements in the fileNames array
//        maxLen        - maximum length of a single name
//

    ofstream outputFile;
    char *filename = NULL;
    Int_t i;

    for( i = 0; i < numberOfNames; i++ ) {
        if( !filename ) {

            // create a filename
            char *tmp1 = gSystem->ConcatFileName( gSystem->ExpandPathName( fOutputDir ), fileNames[i] );
            filename = StrDup( tmp1, 16);

            if( tmp1 ) delete [] tmp1;
            tmp1 = 0;

            char *underlinePtr = strrchr( filename, '_');
            *underlinePtr = 0;

            strcat( filename, "_Index.html" );

            // open a file
            outputFile.open( filename, ios::out );

            // check if it's OK
            if( outputFile.good() ) {

                Printf( formatStr, "", fCounter, filename );

                // write outputFile header
                WriteHtmlHeader( outputFile, "Index" );
                outputFile << "<h2>" << "Index" << "</h2><hr>" << endl;
                outputFile << "<pre>" << endl;
                outputFile << "<ul>" << endl;
            }
            else Error( "MakeIndex", "Can't open file '%s' !", filename );
            delete [] filename;
        }

        // get a class
        TClass *classPtr = GetClass( (const char * ) strrchr( fileNames[i], '_')+1 );
        if( classPtr ) {

            // write a classname to an index file
            outputFile << "<li>";

            char *htmlFile = GetHtmlFileName( classPtr );

            if( htmlFile ) {
                outputFile << "<a name=\"";
                outputFile << classPtr->GetName();
                outputFile << "\" href=\"";
                outputFile << htmlFile;
                outputFile << "\">";
                outputFile << classPtr->GetName();
                outputFile << "</a> ";
                delete [] htmlFile;
                htmlFile = 0;
            }
            else outputFile << classPtr->GetName();


            // write title
            Int_t len = strlen( classPtr->GetName() );
            for( Int_t w = 0; w < maxLen-len; w++ )
                outputFile << ".";
            outputFile << " ";

            outputFile << "<a name=\"Title:";
            outputFile << classPtr->GetName();
            outputFile << "\">";
            ReplaceSpecialChars( outputFile, classPtr->GetTitle() );
            outputFile << "</a>" << endl;
        }
        else Error( "MakeIndex", "Unknown class '%s' !", strchr( fileNames[i], '_')+1 );


        // first base name
        char *first  = strrchr( fileNames[i], '_');
        if( first ) *first = 0;

        // second base name
        char *second = NULL;
        if( i < ( numberOfNames - 1 )) {
            second = strrchr( fileNames[i+1], '_');
            if( second ) *second = 0;
        }

        // check and close the file if necessary
        if( !first || !second || strcmp( fileNames[i], fileNames[i+1] )) {

            if( outputFile.good() ) {

                outputFile << "</ul>" << endl;
                outputFile << "</pre>" << endl;

                // write outputFile footer
                TDatime date;
                WriteHtmlFooter( outputFile, "", date.AsString() );

                // close file
                outputFile.close();

                filename = NULL;
            }
            else Error( "MakeIndex", "Corrupted file '%s' !", filename );
        }

        if( first )  *first  = '_';
        if( second ) *second = '_';
    }

    // free memory
    for( i = 0; i < numberOfNames; i++ )
        if( *fileNames[i] ) delete [] fileNames[i];
}


//______________________________________________________________________________
void THtml::CreateListOfTypes()
{
// Create list of all data types

    Int_t maxLen = 0;
    Int_t len;

    // open file
    ofstream typesList;

    char *outFile = gSystem->ConcatFileName( gSystem->ExpandPathName( fOutputDir ), "ListOfTypes.html" );
    typesList.open( outFile, ios::out );


    if( typesList.good() ) {
        Printf( formatStr, "", "", outFile );

        // write typesList header
        WriteHtmlHeader( typesList, "List of data types" );
        typesList << "<h2> List of data types </h2><hr>" << endl;

        typesList << "<dl><dd>" << endl;
        typesList << "<pre>" << endl;

        // make loop on data types
        TDataType *type;
        TIter nextType( gROOT->GetListOfTypes() );

        while (( type = ( TDataType * ) nextType() )) {
            if( *type->GetTitle() && !strchr( type->GetName(), '(' ) ) {
                if( type->GetName() ) len = strlen( type->GetName() );
                else len = 0;
                maxLen = maxLen > len ? maxLen : len;
            }
        }
        nextType.Reset();

        maxLen += kSpaceNum;

        while (( type = ( TDataType * ) nextType() )) {
            if( *type->GetTitle() && !strchr( type->GetName(), '(' ) ) {
                typesList << "<b><a name=\"";
                typesList << type->GetName();
                typesList << "\">" << type->GetName();
                typesList << "</a></b>";

                if( type->GetName() ) len = strlen( type->GetName() );
                else len = 0;
                typesList << " ";
                for( Int_t j = 0; j < ( maxLen-len ); j++ )
                    typesList << ".";
                typesList << " ";

                typesList << "<a name=\"Title:";
                typesList << type->GetTitle();
                typesList << "\">";
                char *tempstr = StrDup( type->GetTitle() );
                ReplaceSpecialChars( typesList, tempstr );
                typesList << "</a>" << endl;

                if( tempstr ) delete [] tempstr;
            }
        }

        typesList << "</pre>" << endl;
        typesList << "</dl>" << endl;

        // write typesList footer
        TDatime date;
        WriteHtmlFooter( typesList, "", date.AsString() );

        // close file
        typesList.close();

    }
    else Error( "Make", "Can't open file '%s' !", outFile );

    if (outFile) delete [] outFile;
}


//______________________________________________________________________________
void THtml::DerivedClasses( ofstream &out, TClass *classPtr )
{
// It creates a list of derived classes
//
//
// Input: out      - output file stream
//        classPtr - pointer to the class
//

    Bool_t first = kTRUE;
    Bool_t found = kFALSE;


    // get total number of classes
    Int_t numberOfClasses = gClassTable->Classes();

    // start from begining
    gClassTable->Init();

    // get class names
    TClass *derivedClassPtr;
    const char *derivedClassName;
    for( Int_t i = 0; i < numberOfClasses; i++ ) {

        // get class name
        derivedClassName = gClassTable->Next();

        // get class pointer
        derivedClassPtr = GetClass( derivedClassName );

        if ( !derivedClassPtr ) {
           Warning("DerivedClasses","Can not find a definition for class <%s>",derivedClassName);
           continue;
        }
        // make a loop on base classes
        TBaseClass *inheritFrom;
        TIter nextBase( derivedClassPtr->GetListOfBases() );

        while (( inheritFrom = ( TBaseClass * ) nextBase() )) {
            if( !strcmp( inheritFrom->GetName(), classPtr->GetName() )) {
                if( first ) {
                    out << "<br><hr>" << endl;
                    out << "<!--SEE ALSO-->";
                    out << "<h2>See also</h2><dl><dd>" << endl;
                }
                if( !first ) out << ", ";

                char *htmlFile = GetHtmlFileName( derivedClassPtr );

                if( htmlFile ) {
                    out << "<a href=\"";
                    out << htmlFile;
                    out << "\">";
                    out << derivedClassPtr->GetName() << "</a>";
                    delete [] htmlFile;
                    htmlFile = 0;
                }
                else out << derivedClassPtr->GetName();

                if( first ) {
                    first = kFALSE;
                    found = kTRUE;
                }
            }
        }
    }
    if( found ) out << "</dl>" << endl;
}


//______________________________________________________________________________
void THtml::ExpandKeywords( ofstream &out, char *text, TClass *ptr2class,
                            Bool_t &flag, const char *dir )
{
// Find keywords in text & create URLs
//
//
// Input: out       - output file stream
//        text      - pointer to the array of the characters to process
//        ptr2class - pointer to the class
//        flag      - this is a 'html_begin/html_end' flag
//        dir       - usually "" or "../", depends of current file
//                    directory position
//

    char *keyword = text;
    char *end;
    char *funcName;
    char *funcNameEnd;
    char *funcSig;
    char *funcSigEnd;
    char c, c2, c3;
    char *tempEndPtr;
    c2 = c3 = 0;

    Bool_t hide;
    Bool_t mmf = 0;

    do {
        tempEndPtr = end = funcName = funcNameEnd = funcSig = funcSigEnd = NULL;

        hide = kFALSE;

        // skip until start of the word
        while( !IsWord( *keyword ) && *keyword ) {
            if( !flag ) ReplaceSpecialChars( out, *keyword );
            else out << *keyword;
            keyword++;
        }

        // get end of the word
        end = keyword;
        while( IsName( *end ) && *end ) end++;

        // put '\0' at the end of the keyword
        c = *end;
        *end = 0;

        if( strlen( keyword ) > 50 ) {
            out << keyword;
            *end = c;
            keyword = end;
            continue;
        }

        // check if this is a HTML block
        if( flag ) {
            if( !strcasecmp( keyword, "end_html" ) && *( keyword-1 ) != '\"') {
                flag = kFALSE;
                hide = kTRUE;
            }
        }
        else {
            if( !strcasecmp( keyword, "begin_html" ) && *( keyword-1 ) != '\"') {
                flag = kTRUE;
                hide  = kTRUE;
            }
            else {
                *end = c;
                tempEndPtr = end;

                // skip leading spaces
                while( *tempEndPtr && isspace( *tempEndPtr ) ) tempEndPtr++;


                // check if we have a something like a 'name[arg].name'
                Int_t count = 0;
                if( *tempEndPtr == '[') {
                    count++;
                    tempEndPtr++;
                }

                // wait until the last ']'
                while( count && *tempEndPtr ) {
                    switch( *tempEndPtr ) {
                        case '[': count++;
                                  break;
                        case ']': count--;
                                  break;
                    }
                    tempEndPtr++;
                }

                if( !strncmp( tempEndPtr, "::", 2 ) || !strncmp( tempEndPtr, "->", 2 ) || ( *tempEndPtr == '.') ) {
                    funcName = tempEndPtr;

                    // skip leading spaces
                    while( isspace( *funcName )) funcName++;

                    // check if we have a '.' or '->'
                    if( *tempEndPtr == '.') funcName++;
                    else funcName += 2;

                    if( !strncmp( tempEndPtr, "::", 2 )) mmf = kTRUE;
                    else mmf = kFALSE;

                    // skip leading spaces
                    while( *funcName && isspace( *funcName )) funcName++;

                    // get the end of the word
                    if( !IsWord( *funcName )) funcName = NULL;

                    if( funcName ) {
                        funcNameEnd = funcName;

                        // find the end of the function name part
                        while( IsName( *funcNameEnd ) && *funcNameEnd )
                            funcNameEnd++;
                        c2 = *funcNameEnd;
                        if( !mmf ) {

                            // try to find a signature
                            funcSig = funcNameEnd;

                            // skip leading spaces
                            while( *funcSig && isspace( *funcSig )) funcSig++;
                            if( *funcSig != '(') funcSig = NULL;
                            else funcSig++;
                            funcSigEnd = funcSig;

                            // if signature exist, try to find the ending character
                            if( funcSigEnd ) {
                                Int_t count = 1;
                                while( *funcSigEnd ) {
                                    if( *funcSigEnd == '(') count++;
                                    if( *funcSigEnd == ')')
                                        if( !--count ) break;
                                    funcSigEnd++;
                                }
                                c3 = *funcSigEnd;
                                *funcSigEnd = 0;
                            }
                        }
                        *funcNameEnd = 0;
                    }
                }
                *end = 0;
            }
        }

        if( !flag && !hide && *keyword ) {

            // get class
            TClass *classPtr = GetClass( (const char * ) keyword );

            if( classPtr ) {

                char *htmlFile = GetHtmlFileName( classPtr );

                if( htmlFile ) {
                    out << "<a href=\"";
                    if( *dir && strncmp( htmlFile, "http://", 7 )) out << dir;
                    out << htmlFile;

                    if( funcName && mmf ) {

                        // make a link to the member function
                        out << "#" << classPtr->GetName() << ":";
                        out << funcName;
                        out << "\">";
                        out << classPtr->GetName() << "::";
                        out << funcName;
                        out << "</a>";

                        *funcNameEnd = c2;
                        keyword = funcNameEnd;
                    }
                    else {
                        // make a link to the class
                        out << "\">";
                        out << classPtr->GetName();
                        out << "</a>";

                        keyword = end;
                    }
                    delete [] htmlFile;
                    htmlFile = 0;

                }
                else {
                    out << keyword;
                    keyword = end;
                }
                *end = c;
                if( funcName ) *funcNameEnd = c2;
                if( funcSig )  *funcSigEnd  = c3;
            }
            else {
                // get data type
                TDataType *type = gROOT->GetType( (const char *) keyword );

                if( type ) {

                    // make a link to the data type
                    out << "<a href=\"";
                    if( *dir ) out << dir;
                    out << "ListOfTypes.html#";
                    out << keyword << "\">";
                    out << keyword << "</a>";

                    *end = c;
                    keyword = end;
                }
                else {
                    // look for '('
                    Bool_t isfunc = ( (*tempEndPtr == '(') || c == '(')? kTRUE: kFALSE;
                    if( !isfunc ) {
                        char *bptr = tempEndPtr + 1;
                        while( *bptr && isspace( *bptr ) ) bptr++;
                        if( *bptr == '(') isfunc = kTRUE;
                    }

                    if( isfunc && ptr2class && ( ptr2class->GetMethodAny( keyword )) ) {
                        out << "<a href=\"#";
                        out << ptr2class->GetName();
                        out << ":" << keyword << "\">";
                        out << keyword << "</a>";
                        *end = c;
                        keyword = end;
                    }
                    else {
                        const char *anyname = gROOT->FindObjectClassName( keyword );

                        const char *namePtr = NULL;
                        TClass *cl  = 0;
                        TClass *cdl = 0;

                        if( anyname ) {
                            cl = GetClass( anyname );
                            namePtr = ( const char * ) anyname;
                            cdl = cl;
                        }
                        else if( ptr2class ) {
                            cl = ptr2class->GetBaseDataMember( keyword );
                            if( cl ) {
                                namePtr = cl->GetName();
                                TDataMember *member = cl->GetDataMember( keyword );
                                if( member )
                                    cdl = GetClass( member->GetTypeName() );
                            }
                        }

                        if( cl ) {
                            char *htmlFile = GetHtmlFileName( cl );

                            if( htmlFile ) {
                                out << "<a href=\"";
                                if( *dir  && strncmp( htmlFile, "http://", 7 )) out << dir;
                                out << htmlFile;
                                if( cl->GetDataMember( keyword ) ) {
                                    out << "#" << namePtr << ":";
                                    out << keyword;
                                }
                                out << "\">";
                                out << keyword;
                                out << "</a>";
                                delete [] htmlFile;
                                htmlFile = 0;
                            }
                            else out << keyword;

                            if( funcName ) {
                                char *ptr = end;
                                ptr++;
                                ReplaceSpecialChars( out, c );
                                while( ptr < funcName )
                                    ReplaceSpecialChars( out, *ptr++ );

                                TMethod *method = NULL;
                                if( cdl ) method = cdl->GetMethodAny( funcName );
                                if( method ) {
                                    TClass *cm = method->GetClass();
                                    if( cm ) {
                                        char *htmlFile2 = GetHtmlFileName( cm );
                                        if( htmlFile2 ) {
                                            out << "<a href=\"";
                                            if( *dir  && strncmp( htmlFile2, "http://", 7 )) out << dir;
                                            out << htmlFile2;
                                            out << "#" << cm->GetName() << ":";
                                            out << funcName;
                                            out << "\">";
                                            out << funcName;
                                            out << "</a>";
                                            delete [] htmlFile2;
                                            htmlFile2 = 0;
                                        }
                                        else out << funcName;

                                        keyword = funcNameEnd;
                                    }
                                    else keyword = funcName;
                                }
                                else keyword = funcName;

                                *funcNameEnd = c2;
                                if( funcSig ) *funcSigEnd = c3;
                            }
                            else keyword = end;
                            *end = c;
                        }
                        else {
                            if( funcName ) *funcNameEnd = c2;
                            if( funcSig )  *funcSigEnd  = c3;
                            out << keyword;
                            *end = c;
                            keyword = end;
                        }
                    }
                }
            }
        }
        else {
            if( !hide && *keyword )
                out << keyword;
            *end = c;
            keyword = end;
        }
    } while( *keyword );
}


//______________________________________________________________________________
void THtml::ExpandPpLine( ofstream &out, char *line )
{
// Expand preprocessor statements
//
//
// Input: out  - output file stream
//        line - pointer to the array of characters,
//               usually one line from the source file
//
//  NOTE: Looks for the #include statements and
//        creates link to the corresponding file
//        if such file exists
//

    const char *ptr;
    const char *ptrStart;
    const char *ptrEnd;
    char *fileName;

    Bool_t linkExist = kFALSE;

    ptrEnd = strstr( line, "include" );
    if( ptrEnd ) {
        ptrEnd += 7;
        if (( ptrStart = strpbrk( ptrEnd, "<\"" ))) {
            ptrStart++;
            ptrEnd = strpbrk( ptrStart, ">\"" );
            if( ptrEnd ) {
                Int_t len = ptrEnd - ptrStart;
                fileName = new char [len + 1];
                strncpy( fileName, ptrStart, len );

                char *tmpstr = gSystem->Which( fSourceDir, fileName, kReadPermission );
                if( tmpstr ) {
                    char *realFileName = StrDup( tmpstr );

                    if( realFileName ) {
                        CopyHtmlFile( realFileName );

                        ptr = line;
                        while( ptr < ptrStart )
                            ReplaceSpecialChars( out, *ptr++ );
                        out << "<a href=\"../" << GetFileName( realFileName ) << "\">";
                        out << fileName << "</a>";
                        out << ptrEnd;

                        linkExist = kTRUE;
                    }
                    if( realFileName ) delete [] realFileName;
                    if( fileName )     delete [] fileName;
                    delete [] tmpstr;
                }
            }
        }
    }

    if( !linkExist ) ReplaceSpecialChars( out, line );
}

//______________________________________________________________________________
const char *THtml::GetFileName( const char *filename )
{
// It discards any directory information inside filename
//
//
//  Input: filename - pointer to the file name
//
// Output: pointer to the string containing just a file name
//         without any other directory information, i.e.
//         '/usr/root/test.dat' will return 'test.dat'
//

    return( gSystem->BaseName( gSystem->UnixPathName( filename )) );
}

//______________________________________________________________________________
char *THtml::GetSourceFileName(const char *filename)
{
   // Find the source file. If filename contains a path it will be used
   // together with the possible source prefix. If not found we try
   // old algorithm, by stripping off the path and trying to find it in the
   // specified source search path. Returned string must be deleted by the
   // user. In case filename is not found 0 is returned.

   char *tmp1;
#ifdef WIN32
   if (strchr(filename, '/') || strchr(filename, '\\')) {
#else
   if (strchr(filename, '/')) {
#endif
      char *tmp;
      if (strlen(fSourcePrefix) > 0)
         tmp = gSystem->ConcatFileName(fSourcePrefix, filename);
      else
         tmp = StrDup(filename);
      if ((tmp1 = gSystem->Which(fSourceDir, tmp, kReadPermission))) {
         delete [] tmp;
         return tmp1;
      }
      delete [] tmp;
   }

   if ((tmp1 = gSystem->Which(fSourceDir, GetFileName(filename), kReadPermission)))
      return tmp1;

   return 0;
}

//______________________________________________________________________________
char *THtml::GetHtmlFileName( TClass *classPtr )
{
// Return real HTML filename
//
//
//  Input: classPtr - pointer to a class
//
// Output: pointer to the string containing a full name
//         of the corresponding HTML file. The string must be deleted by the user.
//

    char htmlFileName [128];

    char *ret  = 0;
    Bool_t found = kFALSE;

    if( classPtr ) {

        const char *filename = classPtr->GetImplFileName();

        char varName[80];
        const char *colon = strchr( filename, ':');


        // this should be a prefix
        strcpy( varName, "Root.Html." );


        if( colon )
            strncat( varName, filename, colon-filename );
        else strcat( varName, "Root" );

        char *tmp;
        if( !(tmp = gSystem->Which( fSourceDir, filename, kReadPermission ))) {
            strcpy( htmlFileName, gEnv->GetValue( varName, "" ));
            if( !*htmlFileName ) found = kFALSE;
            else found = kTRUE;
        }
        else {
            strcpy( htmlFileName, "." );
            found = kTRUE;
        }
        delete [] tmp;

        if( found ) {
            char *tmp1 = gSystem->ConcatFileName( htmlFileName, classPtr->GetName() );
            ret = StrDup( tmp1, 16 );
            strcat( ret, ".html" );

            if( tmp1 ) delete [] tmp1;
            tmp1 = 0;
        }
        else ret = 0;

    }

    return ret;
}

//______________________________________________________________________________
TClass *THtml::GetClass(const char *name1, Bool_t load)
{
//*-*-*-*-*Return pointer to class with name*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*      =================================
   Int_t n = strlen(name1);
   char *name = new char[n+1];
   strcpy(name, name1);
   char *t = name+n-1;
   while(*t == ' ') {
      *t = 0;
      if (t == name) break;
      t--;
   }
   t = name;
   while(*t == ' ') t++;

   TClass *cl = gROOT->GetClass(t,load);
   delete [] name;
   return cl;
}


//______________________________________________________________________________
Bool_t THtml::IsModified( TClass *classPtr, const Int_t type )
{
// Check if file is modified
//
//
//  Input: classPtr - pointer to the class
//         type     - file type to compare with
//                    values: kSource, kInclude, kTree
//
// Output: TRUE     - if file is modified since last time
//         FALSE    - if file is up to date
//

    Bool_t ret = kTRUE;

    char  sourceFile[1024], filename[1024];
    char *strPtr, *strPtr2;

    switch( type ) {
       case kSource:
          strPtr2 = GetSourceFileName(classPtr->GetImplFileName());
          if (strPtr2) strcpy( sourceFile, strPtr2 );
          strPtr = gSystem->ConcatFileName( gSystem->ExpandPathName( fOutputDir ), "src" );
          strcpy( filename, strPtr );
          delete [] strPtr;
          delete [] strPtr2;
#ifdef WIN32
          strcat( filename, "\\" );
#else
          strcat( filename, "/" );
#endif
          strcat( filename, classPtr->GetName() );
          strcat( filename, ".cxx.html" );
          break;

       case kInclude:
          strPtr2 = GetSourceFileName(classPtr->GetDeclFileName());
          if (strPtr2) strcpy( sourceFile, strPtr2 );
          strPtr = gSystem->ConcatFileName( gSystem->ExpandPathName( fOutputDir ), GetFileName( classPtr->GetDeclFileName() ));
          strcpy( filename,strPtr );
          delete [] strPtr;
          delete [] strPtr2;
          break;

       case kTree:
          strPtr2 = GetSourceFileName(classPtr->GetDeclFileName());
          if (strPtr2) strcpy( sourceFile, strPtr2 );
          strPtr = gSystem->ConcatFileName( gSystem->ExpandPathName( fOutputDir ), GetFileName( classPtr->GetName() ));
          strcpy( filename, strPtr);
          delete [] strPtr;
          delete [] strPtr2;
          strcat( filename, "_Tree.ps" );
          break;

       default:
          Error( "IsModified", "Unknown file type !" );
    }

    // Get info about a file
    Long_t sId, sSize, sFlags, sModtime;
    Long_t dId, dSize, dFlags, dModtime;

    if( !( gSystem->GetPathInfo( sourceFile, &sId, &sSize, &sFlags, &sModtime )) )
       if( !( gSystem->GetPathInfo( filename, &dId, &dSize, &dFlags, &dModtime )) )
          ret = ( sModtime > dModtime ) ? kTRUE : kFALSE;

    return( ret );
}


//______________________________________________________________________________
Bool_t THtml::IsName( Int_t c )
{
// Check if c is a valid C++ name character
//
//
//  Input: c - a single character
//
// Output: TRUE if c is a valid C++ name character
//         and FALSE if it's not.
//
//   NOTE: Valid name characters are [a..zA..Z0..9_],
//

    Bool_t ret = kFALSE;

    if( isalnum( c ) || c == '_') ret = kTRUE;

    return ret;
}


//______________________________________________________________________________
Bool_t THtml::IsWord( Int_t c )
{
// Check if c is a valid first character for C++ name
//
//
//  Input: c - a single character
//
// Output: TRUE if c is a valid first character for C++ name,
//         and FALSE if it's not.
//
//   NOTE: Valid first characters are [a..zA..Z_]
//

    Bool_t ret = kFALSE;

    if( isalpha( c ) || c == '_') ret = kTRUE;

    return ret;
}


//______________________________________________________________________________
void THtml::MakeAll( Bool_t force, const char *filter)
{
// It makes all the classes specified in the filter (by default "*")
// To process all classes having a name starting with XX, do:
//        html.MakeAll(kFALSE,"XX*");
// if force=kFALSE (default), only the classes that have been modified since
// the previous call to this function will be generated.
// if force=kTRUE, all classes passing the filter will be processed.
//

    Int_t i;

    TString reg = filter;
    TRegexp re(reg, kTRUE);
    Int_t nOK = 0;

    MakeIndex(filter);

    Int_t numberOfClasses    = gClassTable->Classes();
    const char **className = new const char* [numberOfClasses];

    // start from begining
    gClassTable->Init();


    for( i = 0; i < numberOfClasses; i++ ) {       
      const char *cname = gClassTable->Next();
      TString s = cname;
      if (s.Index(re) == kNPOS) continue;
      className[nOK] = cname;
      nOK++;
    }

    for( i = 0; i < nOK; i++ ) {
        sprintf( fCounter, "%5d", nOK - i );
        MakeClass( (char * ) className[i], force );
    }

    *fCounter = 0;

    delete [] className;
}


//______________________________________________________________________________
void THtml::MakeClass(const char *className, Bool_t force )
{
// Make HTML files for a single class
//
//
// Input: className - name of the class to process
//

    TClass *classPtr = GetClass( className );

    if( classPtr ) {
        char *htmlFile = GetHtmlFileName( classPtr );
        if( htmlFile && !strncmp( htmlFile, "http://", 7 )) {
           delete [] htmlFile;
           htmlFile = 0;
        }
        if( htmlFile ) {
            Class2Html( classPtr, force );
            MakeTree( className, force );
            delete [] htmlFile;
            htmlFile = 0;
        }
        else Printf( formatStr, "-skipped-", fCounter, className );
    }
    else Error( "MakeClass", "Unknown class '%s' !", className );

}


//______________________________________________________________________________
void THtml::MakeIndex(const char *filter)
{
   // It makes an index files
   // by default makes an index of all classes (if filter="*")
   // To generate an index for all classes starting with "XX", do
   //    html.MakeIndex("XX*");

   CreateListOfTypes();

   // get total number of classes
   Int_t numberOfClasses = gClassTable->Classes();


   // allocate memory
   const char **classNames = new const char *[numberOfClasses];
   char       **fileNames  = new       char *[numberOfClasses];

   // start from begining
   gClassTable->Init();

   // get class names
   Int_t len = 0;
   Int_t maxLen = 0;
   Int_t numberOfImpFiles = 0;

   TString reg = filter;
   TRegexp re(reg, kTRUE);
   Int_t nOK = 0;
   
   for( Int_t i = 0; i < numberOfClasses; i++ ) {

      // get class name
      const char *cname = gClassTable->Next();
      TString s = cname;
      if (s.Index(re) == kNPOS) continue;
      classNames[nOK] = cname;
      len    = strlen( classNames[nOK] );
      maxLen = maxLen > len ? maxLen : len;

      // get class & filename
      TClass *classPtr = GetClass( (const char * ) classNames[nOK] );
      const char *impname = classPtr->GetImplFileName();

      if( impname ) {
         fileNames[numberOfImpFiles] = StrDup( impname, 64 );

         char *underline = strchr( fileNames[numberOfImpFiles], '_');
//         if( underline )
//            strcpy( underline + 1, classNames[nOK] );
//         else {
            // for new ROOT install the impl file name has the form: base/src/TROOT.cxx
            char *srcdir = strstr(fileNames[numberOfImpFiles], "/src/");
            if (srcdir) {
               strcpy(srcdir, "_");
               for (char *t = fileNames[numberOfImpFiles]; (t[0] = toupper(t[0])); t++) ;
               strcat(srcdir, classNames[nOK]);
            } else {
               strcpy( fileNames[nOK], "USER_" );
               strcat( fileNames[nOK], classNames[nOK] );
            }
//         }
         numberOfImpFiles++;
      }
      else cout << "WARNING class:" << classNames[i] << " has no implementation file name !" << endl;

      nOK++;
   }
   maxLen += kSpaceNum;

   // quick sort
   SortNames( classNames, nOK );
   SortNames( (const char ** ) fileNames,  numberOfImpFiles );

   // create an index
   CreateIndex( classNames, nOK);
   CreateIndexByTopic( fileNames, nOK, maxLen );

   // free allocated memory
   delete [] classNames;
   delete [] fileNames;
}


//______________________________________________________________________________
void THtml::MakeTree(const char *className, Bool_t force )
{
// Make an inheritance tree
//
//
// Input: className - name of the class to process
//

    // create canvas & set fill color
    TVirtualPad *psCanvas = 0;
    gROOT->ProcessLineFast("new TCanvas(\"\",\"psCanvas\",0,0,1000,750);");
    psCanvas = gPad->GetVirtCanvas();

    TClass *classPtr = GetClass( className );

    if( classPtr ) {

        char *htmlFile = GetHtmlFileName( classPtr );
        if( htmlFile && !strncmp( htmlFile, "http://", 7 )) {
           delete [] htmlFile;
           htmlFile = 0;
        }
        if( htmlFile ) {

            // make a class tree
            ClassTree( psCanvas, classPtr, force );
            delete [] htmlFile;
            htmlFile = 0;
        }
        else Printf( formatStr, "-skipped-", "", className );

    }
    else Error( "MakeTree", "Unknown class '%s' !", className );

    // close canvas
    psCanvas->Close();
    delete psCanvas;

}


//______________________________________________________________________________
void THtml::ReplaceSpecialChars( ofstream &out, const char c )
{
// Replace ampersand, less-than and greater-than character
//
//
// Input: out - output file stream
//        c   - single character
//

    if (fEscFlag) {
      out << c;
      fEscFlag = kFALSE;
    }
    else if (c == fEsc)
      fEscFlag = kTRUE;
    else
    {
      switch( c ) {
          case '<':
              out << "&lt;";
              break;
          case '&':
              out << "&amp;";
              break;
          case '>':
              out << "&gt;";
              break;
          default:
             out << c;
       }
    }
}


//______________________________________________________________________________
void THtml::ReplaceSpecialChars( ofstream &out, const char *string )
{
// Replace ampersand, less-than and greater-than characters
//
//
// Input: out    - output file stream
//        string - pointer to an array of characters
//

    if( string ) {
        char *data  = StrDup( string );
        if( data ) {
            char *ptr   = NULL;
            char *start = data;

            while (( ptr = strpbrk( start, "<&>" ))) {
               char c = *ptr;
               *ptr = 0;
               out << start;
               ReplaceSpecialChars( out, c );
               start = ptr+1;
            }
            out << start;
            delete [] data;
        }
    }
}

//______________________________________________________________________________
void THtml::SortNames( const char **strings, Int_t num, Bool_t type )
{
// Sort strings
//
//
// Input: strings - pointer to an array of strings
//        type    - sort type
//                  values : kCaseInsensitive, kCaseSensitive
//                  default: kCaseInsensitive
//

    if( type == kCaseSensitive )
        qsort( strings, num, sizeof( strings ), CaseSensitiveSort );
    else
        qsort( strings, num, sizeof( strings ), CaseInsensitiveSort );
}


//______________________________________________________________________________
char *THtml::StrDup( const char *s1, Int_t n )
{
// Returns a pointer to a new string which is a duplicate
// of the string to which 's1' points.  The space for the
// new string is obtained using the 'new' operator. The new
// string has the length of 'strlen(s1) + n'.


   char *str = 0;

   if( s1 ) {
       if( n < 0 ) n = 0;
       str = new char[ strlen( s1 ) + n + 1 ];
       if( str ) strcpy( str, s1 );
   }

   return( str );
}

//______________________________________________________________________________
void THtml::WriteHtmlHeader( ofstream &out, const char *title )
{
// Write HTML header
//
//
// Input: out   - output file stream
//        title - title for the HTML page
//

    TDatime date;

    out << "<!DOCTYPE HTML PUBLIC \"-// IETF/DTD HTML 2.0// EN\">" << endl;
    out << "<html>" << endl;
    out << "<!--                                             -->" << endl;
    out << "<!-- Author: ROOT team (rootdev@hpsalo.cern.ch)  -->" << endl;
    out << "<!--                                             -->" << endl;
    out << "<!--   Date: "<< date.AsString() << "            -->" << endl;
    out << "<!--                                             -->" << endl;
    out << "<head>" << endl;
    out << "<title>";
    ReplaceSpecialChars( out, title );
    out << "</title>" << endl;
    out << "<link rev=made href=\"mailto:rootdev@root.cern.ch\">" << endl;
    out << "<meta name=\"rating\" content=\"General\">" << endl;
    out << "<meta name=\"objecttype\" content=\"Manual\">" << endl;
    out << "<meta name=\"keywords\" content=\"software development, oo, object oriented, ";
    out << "unix, x11, windows, c++, html, rene brun, fons rademakers\">" << endl;
    out << "<meta name=\"description\" content=\"ROOT - An Object Oriented Framework For Large Scale Data Analysis.\">" << endl;
    out << "</head>" << endl;

    out << "<body BGCOLOR=\"#ffffff\" LINK=\"#0000ff\" VLINK=\"#551a8b\" ALINK=\"#ff0000\" TEXT=\"#000000\">" << endl;
    out << "<a name=\"TopOfPage\"></a>" << endl;
}


//______________________________________________________________________________
void THtml::WriteHtmlFooter( ofstream &out, const char *dir, const char *lastUpdate,
                             const char *author, const char *copyright )
{
// Write HTML footer
//
//
// Input: out        - output file stream
//        dir        - usually equal to "" or "../", depends of
//                     current file directory position, i.e. if
//                     file is in the fOutputDir, then dir will be ""
//        lastUpdate - last update string
//        author     - author's name
//        copyright  - copyright note
//

    out << endl;

    if( *author || *lastUpdate || *copyright ) out << "<hr><br>" << endl;

    out << "<!--SIGNATURE-->" << endl;

    // get the author( s )
    if( *author )  {

        out << "<em>Author: ";

        char *auth = StrDup(author);

        char *name = strtok( auth, "," );

        Bool_t firstAuthor = kTRUE;

        do {
            char *ptr = name;
            char c;

            // remove leading spaces
            while( *ptr && isspace( *ptr ) ) ptr++;

            if( !firstAuthor ) out << ", ";

            if( !strncmp( ptr, "Nicolas", 7 ) ) {
                out << "<a href=http://pcbrun.cern.ch/nicolas/index.html";
                ptr += 12;
            } else {
                out << "<a href="<<GetXwho();
            }
            while( *ptr ) {
                // Valery's specific case
                if( !strncmp( ptr, "Valery", 6 ) ) {
                    out << "Valeri";
                    ptr += 6;
                }
                else if( !strncmp( ptr, "Fine", 4 ) ) {
                    out << "Faine";
                    ptr += 4;
                }
                while( *ptr && !isspace( *ptr ) )
                    out << *ptr++;

                if( isspace( *ptr ) ) {
                    while( *ptr && isspace( *ptr ) ) ptr++;
                    if( isalpha( *ptr) ) out << '+';
                    else break;
                }
                else break;
            }
            c = *ptr;
            *ptr = 0;
            out << ">" << name << "</a>";
            *ptr = c;
            out << ptr;

            firstAuthor = kFALSE;

        } while (( name = strtok( NULL, "," )));
        out << "</em><br>" << endl;
        delete [] auth;
    }

    if( *lastUpdate ) out << "<em>Last update: " << lastUpdate << "</em><br>" << endl;
    if( *copyright )  out << "<em>Copyright " << copyright << "</em><br>" << endl;


    // this is a menu
    out << "<br>" << endl;
    out << "<address>" << endl;
    out << "<hr>" << endl;
    out << "<center>" << endl;

    // link to the ROOT home page
    out << "<a href=\"http://root.cern.ch/root/Welcome.html\">ROOT page</a> - ";

    // link to the user home page( if exist )
    const char *userHomePage = gEnv->GetValue( "Root.Html.HomePage", "" );
    if( *userHomePage ) {
        out << "<a href=\"";
        if( *dir ) {
            if( strncmp( userHomePage, "http://", 7 ))
                out << dir;
        }
        out << userHomePage;
        out << "\">Home page</a> - ";
    }

    // link to the index file
    out << "<a href=\"";
    if( *dir ) out << dir;
    out << "ClassIndex.html\">Class index</a> - ";

    // link to the top of the page
    out << "<a href=\"#TopOfPage\">Top of the page</a><br>" << endl;

    out << "</center>" << endl;

    out << "<hr>This page has been automatically generated. If you have any comments or suggestions ";
    out << "about the page layout send a mail to <a href=\"mailto:rootdev@root.cern.ch\">ROOT support</a>, or ";
    out << "contact <a href=\"mailto:rootdev@root.cern.ch\">the developers</a> with any questions or problems regarding ROOT." << endl;
    out << "</address>" << endl;
    out << "</body>" << endl;
    out << "</html>" << endl;
}

