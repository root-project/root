/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooHtml.cc,v 1.1 2001/10/04 00:37:18 david Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   26-Sep-2001 DK Created initial version
 *
 * Copyright (C) 2001 Stanford University
 *****************************************************************************/

#include "RooFitCore/RooHtml.hh"

#include "TDatime.h"

#include <iostream.h>
#include <fstream.h>
#include <string.h>

ClassImp(RooHtml)
  ;

void RooHtml::WriteHtmlHeader(ofstream &out, const char *title) {
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
    << "<center><table BORDER=0 CELLSPACING=0 COLS=2 WIDTH=\"100%\" BGCOLOR=\"#FFCC00\" NOSAVE >" << endl
    << "  <tr NOSAVE><td NOSAVE><b><i><font color=\"#000000\"><font size=+2>" << endl
    //-------------------------------------------------
    << "    <a href=\"http://www.slac.stanford.edu/BFROOT/www/Computing/Offline/ROOT/RooFit\"" << endl
    << "      title=\"Visit RooFit Home Page\">" << endl
    << "    RooFit Toolkit for Data Modeling</a>" << endl
    //-------------------------------------------------
    << "  </font></font></i></b></td>" << endl
    << "  <td><div align=right><b><i><font color=\"#000000\"><font size=+2>" << endl
    //-------------------------------------------------
    << "    <a href=\"ClassIndex.html\"" << endl
    << "      title=\"Visit List of Classes\">" << endl;
  out << getVersion() << " Version</a>" << endl;
  //-------------------------------------------------
  out
    << "  </font></font></i></b></div></td></tr>" << endl
    << "</table></center>" << endl;
}

void RooHtml::WriteHtmlFooter(ofstream &out, const char *dir, const char *lastUpdate,
			      const char *author, const char *copyright) {
  // Write a custom html footer for RooFit class documentation to the specified output stream.  

  // lastUpdate will be the CVS tag in case of .rdl and .cc files: clean it up a bit
  const char *comma= index(lastUpdate,',');
  TString update = comma ? comma+1 : lastUpdate;
  if(update.EndsWith(" Exp $")) update.Remove(update.Length()-6,6);

  out << "<center><table BORDER=0 CELLSPACING=0 COLS=2 WIDTH=\"100%\" BGCOLOR=\"#FFCC00\" NOSAVE >" << endl
      << "<tr NOSAVE>" << endl
      << "<td>Last CVS Update: " << update << "</td>" << endl
      << "<td NOSAVE align=right><b><a href=\"#TopOfPage\">Top</a></b></td>" << endl
      << "</tr></table></center>" << endl
      << "<center>Copyright &copy 2001 University of California</center>" << endl;
}
