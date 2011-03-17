// @(#)root/tmva $Id$
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Peter Speckmayer

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : VariableRearrangeTransform                                            *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation (see header for description)                               *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Peter Speckmayer <Peter.Speckmayer@cern.ch> - CERN, Switzerland           *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland                                                         *
 *      MPI-K Heidelberg, Germany                                                 *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#include <iostream>
#include <iomanip>
#include <stdexcept>

#ifndef ROOT_TMVA_MsgLogger
#include "TMVA/MsgLogger.h"
#endif
#ifndef ROOT_TMVA_VariableRearrangeTransform
#include "TMVA/VariableRearrangeTransform.h"
#endif
#ifndef ROOT_TMVA_Tools
#include "TMVA/Tools.h"
#endif
#ifndef ROOT_TMVA_DataSet
#include "TMVA/DataSet.h"
#endif

ClassImp(TMVA::VariableRearrangeTransform)

//_______________________________________________________________________
TMVA::VariableRearrangeTransform::VariableRearrangeTransform( DataSetInfo& dsi )
:  VariableTransformBase( dsi, Types::kRearranged, "Rearrange" )
{ 
   // constructor
}

//_______________________________________________________________________
TMVA::VariableRearrangeTransform::~VariableRearrangeTransform() {
}

//_______________________________________________________________________
void TMVA::VariableRearrangeTransform::Initialize()
{
   // initialization of the rearrangement transformation
   // (nothing to do)
}

//_______________________________________________________________________
Bool_t TMVA::VariableRearrangeTransform::PrepareTransformation( const std::vector<Event*>& /*events*/ )
{
   // prepare transformation --> (nothing to do)
   if (!IsEnabled() || IsCreated()) return kTRUE;

   UInt_t nvars = 0, ntgts = 0, nspcts = 0;
   CountVariableTypes( nvars, ntgts, nspcts );
//   std::cout << "vartypes&varrearrtransf: " << nvars << " " << ntgts << " " << nspcts << std::endl;
//   events[0]->Print(std::cout);
   if( ntgts>0 )
      Log() << kFATAL << "Targets used in Rearrange-transformation." << Endl;

   SetCreated( kTRUE );
   return kTRUE;
}

//_______________________________________________________________________
const TMVA::Event* TMVA::VariableRearrangeTransform::Transform( const TMVA::Event* const ev, Int_t /*cls*/ ) const
{
   if( !IsEnabled() )
      return ev;

   // apply the normalization transformation
   if (!IsCreated()) Log() << kFATAL << "Transformation not yet created" << Endl;

   if (fTransformedEvent==0 ) 
      fTransformedEvent = new Event();

   FloatVector input; // will be filled with the selected variables, (targets)
   std::vector<Char_t> mask; // masked variables
//    std::cout << "========" << std::endl;
//    UInt_t nvars = 0, ntgts = 0, nspcts = 0;
//    CountVariableTypes( nvars, ntgts, nspcts );
//    std::cout << "vartypes&varrearrtransf/trnsfrm: " << nvars << " " << ntgts << " " << nspcts << std::endl;
//    ev->Print(std::cout);
   GetInput( ev, input, mask );
//    for( std::vector<Float_t>::iterator it = input.begin(), itEnd = input.end(); it != itEnd; ++it ){
//       std::cout << (*it) << "  ";
//    }
//    std::cout << std::endl;
   SetOutput( fTransformedEvent, input, mask, ev );
//    std::cout << "transformed ---" << std::endl;
//    fTransformedEvent->Print(std::cout);


   return fTransformedEvent;
}

//_______________________________________________________________________
const TMVA::Event* TMVA::VariableRearrangeTransform::InverseTransform( const TMVA::Event* const ev, Int_t /*cls*/ ) const
{
   if( !IsEnabled() )
      return ev;

   // apply the inverse transformation
   if (!IsCreated()) Log() << kFATAL << "Transformation not yet created" << Endl;

   if (fBackTransformedEvent==0) 
      fBackTransformedEvent = new Event( *ev );

   FloatVector input;  // will be filled with the selected variables, targets, (spectators)
   std::vector<Char_t> mask; // masked variables
//   std::cout << "inv =====" << std::endl;
   GetInput( ev, input, mask, kTRUE );
//   ev->Print(std::cout);
   SetOutput( fBackTransformedEvent, input, mask, ev, kTRUE );
//    std::cout << "inv ---" << std::endl;
//    fBackTransformedEvent->Print(std::cout);


   return fBackTransformedEvent;
}


//_______________________________________________________________________
std::vector<TString>* TMVA::VariableRearrangeTransform::GetTransformationStrings( Int_t /*cls*/ ) const
{
//    // creates string with variable transformations applied

//    // if cls (the class chosen by the user) not existing, assume that user wants to 
//    // have the matrix for all classes together. 
//    if (cls < 0 || cls > GetNClasses()) cls = GetNClasses();

//    Float_t min, max;

   const UInt_t size = fGet.size();
   std::vector<TString>* strVec = new std::vector<TString>(size);

//    UInt_t iinp = 0;
//    for( ItVarTypeIdxConst itGet = fGet.begin(), itGetEnd = fGet.end(); itGet != itGetEnd; ++itGet ) {
//       min = fMin.at(cls).at(iinp);
//       max = fMax.at(cls).at(iinp);

//       Char_t type = (*itGet).first;
//       UInt_t idx  = (*itGet).second;

//       Float_t offset = min;
//       Float_t scale  = 1.0/(max-min);      
//       TString str("");
//       VariableInfo& varInfo = (type=='v'?fDsi.GetVariableInfo(idx):(type=='t'?fDsi.GetTargetInfo(idx):fDsi.GetSpectatorInfo(idx)));

//       if (offset < 0) str = Form( "2*%g*([%s] + %g) - 1", scale, varInfo.GetLabel().Data(), -offset );
//       else            str = Form( "2*%g*([%s] - %g) - 1", scale, varInfo.GetLabel().Data(),  offset );
//       (*strVec)[iinp] = str;

//       ++iinp;
//    }


   return strVec;
}

//_______________________________________________________________________
void TMVA::VariableRearrangeTransform::AttachXMLTo(void* parent) 
{
//    // create XML description of Rearrange transformation
   void* trfxml = gTools().AddChild(parent, "Transform");
   gTools().AddAttr(trfxml, "Name", "Rearrange");

   VariableTransformBase::AttachXMLTo( trfxml );

//    Int_t numC = (GetNClasses()<= 1)?1:GetNClasses()+1;


//    for( Int_t icls=0; icls<numC; icls++ ) {
//       void* clsxml = gTools().AddChild(trfxml, "Class");
//       gTools().AddAttr(clsxml, "ClassIndex", icls);
//       void* inpxml = gTools().AddChild(clsxml, "Ranges");
//       UInt_t iinp = 0;
//       for( ItVarTypeIdx itGet = fGet.begin(), itGetEnd = fGet.end(); itGet != itGetEnd; ++itGet ) {
//          void* mmxml = gTools().AddChild(inpxml, "Range");
//          gTools().AddAttr(mmxml, "Index", iinp);
//          gTools().AddAttr(mmxml, "Min", fMin.at(icls).at(iinp) );
//          gTools().AddAttr(mmxml, "Max", fMax.at(icls).at(iinp) );
// 	 ++iinp;
//       }
//    }
}

//_______________________________________________________________________
void TMVA::VariableRearrangeTransform::ReadFromXML( void* trfnode ) 
{
//    // Read the transformation matrices from the xml node


   Bool_t newFormat = kFALSE;

   void* inpnode = NULL;

   inpnode = gTools().GetChild(trfnode, "Selection"); // new xml format
   if(inpnode == NULL)
      Log() << kFATAL << "Unknown weight file format for transformations. (tried to read in 'rearrange' transform)" << Endl;
   newFormat = kTRUE;
   
   VariableTransformBase::ReadFromXML( inpnode );
   
   SetCreated();

//    Bool_t newFormat = kFALSE;

//    void* inpnode = NULL;
//    try{
//       inpnode = gTools().GetChild(trfnode, "Input"); // new xml format
//       newFormat = kTRUE;
//    }catch( std::logic_error& excpt ){
//       newFormat = kFALSE; // old xml format
//    }
//    if( newFormat ){
//       // ------------- new format --------------------
//       // read input
//       VariableTransformBase::ReadFromXML( inpnode );

//       // read transformation information
      
//       UInt_t size = fGet.size();
//       UInt_t classindex, idx;

//       void* ch = gTools().GetChild( trfnode );
//       while(ch) {
// 	 Int_t ci = 0;
// 	 gTools().ReadAttr(ch, "ClassIndex", ci);
// 	 classindex = UInt_t(ci);

// 	 fMin.resize(classindex+1);
// 	 fMax.resize(classindex+1);
	 
// 	 fMin[classindex].resize(size,Float_t(0));
// 	 fMax[classindex].resize(size,Float_t(0));

// 	 void* clch = gTools().GetChild( ch );
// 	 while(clch) {
// 	    TString nodeName(gTools().GetName(clch));
// 	    if(nodeName=="Ranges") {
// 	       void* varch = gTools().GetChild( clch );
// 	       while(varch) {
// 		  gTools().ReadAttr(varch, "Index", idx);
// 		  gTools().ReadAttr(varch, "Min",      fMin[classindex][idx]);
// 		  gTools().ReadAttr(varch, "Max",      fMax[classindex][idx]);
// 		  varch = gTools().GetNextChild( varch );
// 	       }
// 	    }
// 	    clch = gTools().GetNextChild( clch );
// 	 }
// 	 ch = gTools().GetNextChild( ch );
//       }

//       SetCreated();
//       return;
//    }
   
//    // ------------- old format --------------------
//    UInt_t classindex, varindex, tgtindex, nvars, ntgts;

//    gTools().ReadAttr(trfnode, "NVariables", nvars);
//    gTools().ReadAttr(trfnode, "NTargets",   ntgts);

//    for( UInt_t ivar = 0; ivar < nvars; ++ivar ){
//       fGet.push_back(std::make_pair<Char_t,UInt_t>('v',ivar));
//    }
//    for( UInt_t itgt = 0; itgt < ntgts; ++itgt ){
//       fGet.push_back(std::make_pair<Char_t,UInt_t>('t',itgt));
//    }

//    void* ch = gTools().GetChild( trfnode );
//    while(ch) {
//       gTools().ReadAttr(ch, "ClassIndex", classindex);

//       fMin.resize(classindex+1);
//       fMax.resize(classindex+1);
//       fMin[classindex].resize(nvars+ntgts,Float_t(0));
//       fMax[classindex].resize(nvars+ntgts,Float_t(0));

//       void* clch = gTools().GetChild( ch );
//       while(clch) {
//          TString nodeName(gTools().GetName(clch));
//          if(nodeName=="Variables") {
//             void* varch = gTools().GetChild( clch );
//             while(varch) {
//                gTools().ReadAttr(varch, "VarIndex", varindex);
//                gTools().ReadAttr(varch, "Min",      fMin[classindex][varindex]);
//                gTools().ReadAttr(varch, "Max",      fMax[classindex][varindex]);
//                varch = gTools().GetNextChild( varch );
//             }
//          } else if (nodeName=="Targets") {
//             void* tgtch = gTools().GetChild( clch );
//             while(tgtch) {
//                gTools().ReadAttr(tgtch, "TargetIndex", tgtindex);
//                gTools().ReadAttr(tgtch, "Min",      fMin[classindex][nvars+tgtindex]);
//                gTools().ReadAttr(tgtch, "Max",      fMax[classindex][nvars+tgtindex]);
//                tgtch = gTools().GetNextChild( tgtch );
//             }
//          }
//          clch = gTools().GetNextChild( clch );
//       }
//       ch = gTools().GetNextChild( ch );
//    }
//    SetCreated();
}



//_______________________________________________________________________
void TMVA::VariableRearrangeTransform::PrintTransformation( ostream& ) 
{
//    // prints the transformation ranges

//    Int_t numC = GetNClasses()+1;
//    if (GetNClasses() <= 1 ) numC = 1;

//    for (Int_t icls = 0; icls < numC; icls++ ) {
//       Log() << "Transformation for class " << icls << " based on these ranges:" << Endl;
      
//       UInt_t iinp = 0;
//       for( ItVarTypeIdxConst itGet = fGet.begin(), itGetEnd = fGet.end(); itGet != itGetEnd; ++itGet ){
// 	 Char_t type = (*itGet).first;
// 	 UInt_t idx  = (*itGet).second;

// 	 TString typeString = (type=='v'?"Variable: ": (type=='t'?"Target : ":"Spectator : ") );
// 	 Log() << typeString.Data() << std::setw(20) << fMin[icls][idx] << std::setw(20) << fMax[icls][idx] << Endl;
	 
// 	 ++iinp;
//       }
//    }
}

//_______________________________________________________________________
void TMVA::VariableRearrangeTransform::MakeFunction( std::ostream& /*fout*/, const TString& /*fcncName*/, 
                                                     Int_t /*part*/, UInt_t /*trCounter*/, Int_t ) 
{
//    // creates a normalizing function

//    UInt_t numC = fMin.size();
//    if (part==1) {
//       fout << std::endl;
//       fout << "   double fMin_"<<trCounter<<"["<<numC<<"]["<<fGet.size()<<"];" << std::endl;
//       fout << "   double fMax_"<<trCounter<<"["<<numC<<"]["<<fGet.size()<<"];" << std::endl;
//    }

//    if (part==2) {
//       fout << std::endl;
//       fout << "//_______________________________________________________________________" << std::endl;
//       fout << "inline void " << fcncName << "::InitTransform_"<<trCounter<<"()" << std::endl;
//       fout << "{" << std::endl;
      
//       for (UInt_t ivar=0; ivar<GetNVariables(); ivar++) {
//          Float_t min = FLT_MAX;
//          Float_t max = -FLT_MAX;
//          for (UInt_t icls = 0; icls < numC; icls++) {
//             min = TMath::Min(min, fMin.at(icls).at(ivar) );
//             max = TMath::Max(max, fMax.at(icls).at(ivar) );
//             fout << "   fMin_"<<trCounter<<"["<<icls<<"]["<<ivar<<"] = " << std::setprecision(12)
//                  << min << ";" << std::endl;
//             fout << "   fMax_"<<trCounter<<"["<<icls<<"]["<<ivar<<"] = " << std::setprecision(12)
//                  << max << ";" << std::endl;
//          }
//       }
//       fout << "}" << std::endl;
//       fout << std::endl;
//       fout << "//_______________________________________________________________________" << std::endl;
//       fout << "inline void " << fcncName << "::Transform_"<<trCounter<<"( std::vector<double>& iv, int cls) const" << std::endl;
//       fout << "{" << std::endl;
//       fout << "if (cls < 0 || cls > "<<GetNClasses()<<") {"<< std::endl;
//       fout << "   if ("<<GetNClasses()<<" > 1 ) cls = "<<GetNClasses()<<";"<< std::endl;
//       fout << "   else cls = "<<(fMin.size()==1?0:2)<<";"<< std::endl;
//       fout << "}"<< std::endl;
//       fout << "   for (int ivar=0;ivar<"<<GetNVariables()<<";ivar++) {" << std::endl;
//       fout << "      double offset = fMin_"<<trCounter<<"[cls][ivar];" << std::endl;
//       fout << "      double scale  = 1.0/(fMax_"<<trCounter<<"[cls][ivar]-fMin_"<<trCounter<<"[cls][ivar]);" << std::endl;
//       fout << "      iv[ivar] = (iv[ivar]-offset)*scale * 2 - 1;" << std::endl;
//       fout << "   }" << std::endl;
//       fout << "}" << std::endl;
//    }
}
