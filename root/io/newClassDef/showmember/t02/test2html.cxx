// @(#)root/test:$Name:  $:$Id: test2html.cxx,v 1.2 2000/07/11 18:05:26 rdm Exp $
// Author: Rene Brun   23/08/96

{
   // this macro can be called from an interactive ROOT session via the command:
   //      Root > .x test2html.cxx
   // It generates the html files for some of the ROOT test programs.

   gROOT.Reset();
   new THtml();   // ctor sets gHtml
   gHtml.SetSourceDir(".");
   gHtml.Convert("Event.cxx","Creation of a ROOT Tree");
   gHtml.Convert("tcollex.cxx","Example of use of collection classes");
   gHtml.Convert("tstring.cxx","Example of use of the TString classes");
   gHtml.Convert("hsimple.cxx","A simple program making histograms and ntuples");
   gHtml.Convert("minexam.cxx","An example of use of the minimization class TMinuit");
}
