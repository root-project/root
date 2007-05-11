/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooHtml.cc,v 1.22 2006/12/08 15:50:40 wverkerke Exp $
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

// -- CLASS DESCRIPTION [MISC] --

#include "RooFit.h"
#include "RooHtml.h"

#include "TDatime.h"
#include "TClassTable.h"
#include "TRegexp.h"
#include "TClass.h"
#include "TSystem.h"
#include "TObjString.h"

#include <ctype.h>
#include "Riostream.h"
#include <fstream>
#include <string.h>

#ifndef _WIN32
#include <strings.h>
#endif


#if (__GNUC__==3&&__GNUC_MINOR__==2&&__GNUC_PATCHLEVEL__==3)
char* operator+( streampos&, char* );
#endif

const Int_t   kSpaceNum      = 1;
const char   *formatStr      = "%12s %5s %s";

ClassImp(RooHtml)
  ;

void RooHtml::WriteHtmlHeader(ofstream &out, const char *title, const char* dir, TClass* /*cls*/)
{
  // Write a custom html header for RooFit class documentation to the specified output stream.  

  out
    << "<!doctype html public \"-//dtd html 4.0 transitional// en\">" << endl
    << "<html>" << endl
    << "<head>" << endl
    << "  <meta http-equiv=\"Content-Type\" content=\"text/html; charset=iso-8859-1\">" << endl
    << "  <meta name=\"Author\" content=\"David Kirkby & Wouter Verkerke\">" << endl
    << "  <meta name=\"GENERATOR\" content=\"RooHtml Class in ROOT Environment\">" << endl
    << "  <title>";
  ReplaceSpecialChars(out, title);
  out
    << "</title>" << endl
    << "  <meta name=\"rating\" content=\"General\">" << endl
    << "  <meta name=\"objecttype\" content=\"Manual\">" << endl
    << "  <meta name=\"keywords\" content=\"software development, oo, object oriented, "
    << "  unix, x11, windows, c++, html, rene brun, fons rademakers, david kirkby, wouter verkerke\">" << endl
    << "  <meta name=\"description\" content=\"RooFit - A Toolkit for Physics Data Modeling.\">" << endl
    << "</head>" << endl
    << "<body text=\"#000000\" bgcolor=\"#FFFFFF\" link=\"#0000FF\" vlink=\"#FF0000\" alink=\"#000088\">"
    << endl
    << "<a name=\"TopOfPage\"></a>" << endl;
  out
    << "<center><table BORDER=0 CELLSPACING=0 COLS=2 WIDTH=\"100%\" BGCOLOR=\"" << _hfColor << "\" NOSAVE >" << endl
    << "  <tr NOSAVE><td NOSAVE><b><i><font color=\"#000000\">" << endl
    //-------------------------------------------------
    << "    <a href=\"http://roofit.sourceforge.net\"" << endl
    << "      title=\"Visit RooFit Home Page\">" << endl
    << "    RooFit Toolkit for Data Modeling</a>" << endl
    //-------------------------------------------------
    << "  </font></i></b></td>" << endl
    << "  <td><div align=right><b><i><font color=\"#000000\">" << endl
    //-------------------------------------------------
    << "    <a href=\"" << dir << "IndexByTopic.html\"" << endl
    << "      title=\"Visit List of Classes\">" << endl;
  out << getVersion() << " Version</a>" << endl;
  //-------------------------------------------------
  out
    << "  </font></i></b></div></td></tr>" << endl
    << "</table></center>" << endl;
}

void RooHtml::WriteHtmlFooter(ofstream &out, const char * /*dir*/, const char *lastUpdate,
			      const char * /*author*/, const char * /*copyright*/) {
  // Write a custom html footer for RooFit class documentation to the specified output stream.  

  // lastUpdate will be the CVS tag in case of .rdl and .cc files: clean it up a bit

// Matthew D. Langston  <langston@SLAC.Stanford.EDU>
// There is no index function in Windows (nor is there a strings.h).
#ifndef _WIN32
   const char *comma= index(lastUpdate,',');
  TString update = comma ? comma+1 : lastUpdate;
#else
  TString update = lastUpdate;
#endif

  if(update.EndsWith(" Exp $")) update.Remove(update.Length()-6,6);

  out << "<center><table BORDER=0 CELLSPACING=0 COLS=2 WIDTH=\"100%\" BGCOLOR=\"" << _hfColor << "\" NOSAVE >" << endl
      << "<tr NOSAVE>" << endl
      << "<td>Last CVS Update: " << update << "</td>" << endl
      << "<td NOSAVE align=right><b><a href=\"#TopOfPage\">Top</a></b></td>" << endl
      << "</tr></table></center>" << endl
      << "<center>Copyright &copy 2000-2005 University of California, Stanford University</center>" << endl
      << "</body>" << endl ;
}


void RooHtml::GetModuleName(TString& module, const char* filename) const
{
  // Return module name based on tag in 'CLASS DESCRIPTION' 
  // instead of source file subdirectory
   module = getClassGroup(filename);
}

char* RooHtml::getClassGroup(const char* fileName) const
{
  // Scan file for 'CLASS DESCRIPTION [<tag>]' sequence
  // If found, return <tag>, otherwise return "USER"

  // Initialize buffer to default group name
  static char buffer[1024] = "" ;
  strcpy(buffer,"USER") ;

  const char* fullName = gSystem->Which(fSourceDir, fileName, kReadPermission) ;
  if (!fullName) {
    return buffer ;
  }

  // Scan file contents
  ifstream ifs(fullName) ;

  char line[1024] ;
  while(ifs.good()) {
    ifs.getline(line,sizeof(line),'\n') ;
    
    // Find magic word
    char *ptr ;
    if ((ptr = strstr(line,"CLASS DESCRIPTION"))) {
      char* start = strchr(ptr,'[') ;
      if (start) {
	// Must have closing bracket to proceed
	if (!strchr(start,']')) break ;

	// Extract keyword between square brackets
	char* group = strtok(start+1,"]") ;

	// Group name must be non-empty
	if (group && strlen(group)) strcpy(buffer,group) ;
      }
      break ;
    }
  }

  return buffer ;
}




void RooHtml::addTopic(const char* tag, const char* description) 
{
  _topicTagList.Add(new TObjString(tag)) ;
  _topicDescList.Add(new TObjString(description)) ;
}




void RooHtml::MakeIndexOfTopics() 
{
  TString idxFileName(fOutputDir) ;
  idxFileName.Append("/") ;
  idxFileName.Append("IndexByTopic.html") ;

  ofstream ofs(idxFileName) ;
  WriteHtmlHeader(ofs,"RooFit Index by Topic") ;

  TIterator* tagIter = _topicTagList.MakeIterator() ;
  TIterator* descIter = _topicDescList.MakeIterator() ;
  TObjString* tag ;
  TObjString* desc ;

  ofs << "<H2>" << endl << "<UL>" << endl ;

  while((tag=(TObjString*)tagIter->Next())) {
    desc=(TObjString*)descIter->Next() ;
    ofs << "<LI> <A HREF=" << tag->String() << "_Index.html>" << desc->String() << "</A>" << endl ;
  }
  ofs << "</UL>" << endl << "</H2>" << endl ;

  TDatime now;
  WriteHtmlFooter(ofs,"",now.AsString()) ;

  delete tagIter ;
  delete descIter ;

  
}

