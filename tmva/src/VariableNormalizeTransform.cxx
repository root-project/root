// @(#)root/tmva $Id$
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Peter Speckmayer

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : VariableNormalizeTransform                                            *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation (see header for description)                               *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker  <Andreas.Hocker@cern.ch>   - CERN, Switzerland           *
 *      Joerg Stelzer    <Joerg.Stelzer@cern.ch>    - CERN, Switzerland           *
 *      Peter Speckmayer <Peter:Speckmayer@cern.ch> - CERN, Switzerland           *
 *      Helge Voss       <Helge.Voss@cern.ch>       - MPI-K Heidelberg, Germany   *
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

#include "TVectorF.h"
#include "TVectorD.h"
#include "TMatrixD.h"
#include "TMatrixDBase.h"

#ifndef ROOT_TMVA_MsgLogger
#include "TMVA/MsgLogger.h"
#endif
#ifndef ROOT_TMVA_VariableNormalizeTransform
#include "TMVA/VariableNormalizeTransform.h"
#endif
#ifndef ROOT_TMVA_Tools
#include "TMVA/Tools.h"
#endif
#ifndef ROOT_TMVA_DataSet
#include "TMVA/DataSet.h"
#endif

ClassImp(TMVA::VariableNormalizeTransform)

//_______________________________________________________________________
TMVA::VariableNormalizeTransform::VariableNormalizeTransform( DataSetInfo& dsi )
   : VariableTransformBase( dsi, Types::kNormalized, "Norm" )
{ 
   // constructor
}

//_______________________________________________________________________
TMVA::VariableNormalizeTransform::~VariableNormalizeTransform() {
}

//_______________________________________________________________________
void TMVA::VariableNormalizeTransform::Initialize()
{
   // initialization of the normalization transformation

   UInt_t nvar = Variables().size();
   UInt_t ntgts = Targets().size();
   Int_t numC = GetNClasses()+1;
   if (GetNClasses() <= 1 ) numC = 1;

   fMin.resize( numC ); 
   fMax.resize( numC ); 
   for (Int_t i=0; i<numC; i++) {
      fMin.at(i).resize(nvar+ntgts);
      fMax.at(i).resize(nvar+ntgts);
      fMin.at(i).assign(nvar+ntgts, 0);
      fMax.at(i).assign(nvar+ntgts, 0);
   }
}

//_______________________________________________________________________
Bool_t TMVA::VariableNormalizeTransform::PrepareTransformation( const std::vector<Event*>& events )
{
   // prepare transformation
   if (!IsEnabled() || IsCreated()) return kTRUE;

   Log() << kINFO << "Preparing the transformation." << Endl;

   Initialize();

   CalcNormalizationParams( events );

   SetCreated( kTRUE );

   return kTRUE;
}

//_______________________________________________________________________
const TMVA::Event* TMVA::VariableNormalizeTransform::Transform( const TMVA::Event* const ev, Int_t cls ) const
{

   // apply the decorrelation transformation
   if (!IsCreated()) Log() << kFATAL << "Transformation not yet created" << Endl;

   // if cls (the class chosen by the user) not existing, 
   // assume that he wants to have the matrix for all classes together. 
   // if (cls < 0 || cls > GetNClasses()) {
   //       if (GetNClasses() > 1 ) cls = GetNClasses();
   //       else cls = (fMin.size()==1?0:2);
   //    }
   // EVT this is a workaround to address the reader problem with transforma and EvaluateMVA(std::vector<float/double> ,...) 
   if (cls < 0 || cls >= (int) fMin.size()) cls = fMin.size()-1;
   // EVT workaround end

   const UInt_t nvars = GetNVariables();
   const UInt_t ntgts = ev->GetNTargets();
   if (nvars != ev->GetNVariables()) {
      Log() << kFATAL << "Transformation defined for a different number of variables (defined for: " << GetNVariables() 
            << ", event contains:  " << ev->GetNVariables() << ")" << Endl;
   }
   if (ntgts != ev->GetNTargets()) {
      Log() << kFATAL << "Transformation defined for a different number of targets (defined for: " << GetNTargets() 
            << ", event contains:  " << ev->GetNTargets() << ")" << Endl;
   }

   if (fTransformedEvent==0) fTransformedEvent = new Event();

   Float_t min,max;
   for (Int_t ivar=nvars-1; ivar>=0; ivar--) {
      min = fMin.at(cls).at(ivar); 
      max = fMax.at(cls).at(ivar);
      Float_t offset = min;
      Float_t scale  = 1.0/(max-min);

      Float_t valnorm = (ev->GetValue(ivar)-offset)*scale * 2 - 1;
      fTransformedEvent->SetVal(ivar,valnorm);  
   }
   for (Int_t itgt=ntgts-1; itgt>=0; itgt--) {
      min = fMin.at(cls).at(nvars+itgt); 
      max = fMax.at(cls).at(nvars+itgt);
      Float_t offset = min;
      Float_t scale  = 1.0/(max-min);

      Float_t original = ev->GetTarget(itgt);
      Float_t valnorm = (original-offset)*scale * 2 - 1;
      fTransformedEvent->SetTarget(itgt,valnorm);
   }
   
   fTransformedEvent->SetWeight     ( ev->GetWeight() );
   fTransformedEvent->SetBoostWeight( ev->GetBoostWeight() );
   fTransformedEvent->SetClass      ( ev->GetClass() );
   return fTransformedEvent;
}

//_______________________________________________________________________
const TMVA::Event* TMVA::VariableNormalizeTransform::InverseTransform( const TMVA::Event* const ev, Int_t cls ) const
{
   // apply the inverse transformation
   if (!IsCreated()) Log() << kFATAL << "Transformation not yet created" << Endl;

   // if cls (the class chosen by the user) not existing, 
   // assume that user wants to have the matrix for all classes together. 
   if (cls < 0 || cls > GetNClasses()) {
      if (GetNClasses() > 1 ) cls = GetNClasses();
      else cls = 0;
   }

   const UInt_t nvars = GetNVariables();
   const UInt_t ntgts = GetNTargets();
   if (nvars != ev->GetNVariables()) {
      Log() << kFATAL << "Transformation defined for a different number of variables " << GetNVariables() << "  " << ev->GetNVariables() 
            << Endl;
   }

   if (fBackTransformedEvent==0) fBackTransformedEvent = new Event( *ev );

   Float_t min,max;
   for (Int_t ivar=nvars-1; ivar>=0; ivar--) {
      min = fMin.at(cls).at(ivar); 
      max = fMax.at(cls).at(ivar);
      Float_t offset = min;
      Float_t scale  = 1.0/(max-min);

      Float_t valnorm = offset+((ev->GetValue(ivar)+1)/(scale * 2));
      fBackTransformedEvent->SetVal(ivar,valnorm);
   }

   for (Int_t itgt=ntgts-1; itgt>=0; itgt--) {
      min = fMin.at(cls).at(nvars+itgt); 
      max = fMax.at(cls).at(nvars+itgt);
      Float_t offset = min;
      Float_t scale  = 1.0/(max-min);

      Float_t original = ev->GetTarget(itgt);
      Float_t valnorm = offset+((original+1.0)/(scale * 2));
      fBackTransformedEvent->SetTarget(itgt,valnorm);
   }

   return fBackTransformedEvent;
}

//_______________________________________________________________________
void TMVA::VariableNormalizeTransform::CalcNormalizationParams( const std::vector<Event*>& events )
{
   // compute offset and scale from min and max
   if (events.size() <= 1) 
      Log() << kFATAL << "Not enough events (found " << events.size() << ") to calculate the normalization" << Endl;
   
   UInt_t nvars = GetNVariables();
   UInt_t ntgts = GetNTargets();

   Int_t numC = GetNClasses()+1;
   if (GetNClasses() <= 1 ) numC = 1;

   for (UInt_t ivar=0; ivar<nvars+ntgts; ivar++) {
      for (Int_t ic = 0; ic < numC; ic++) {
         fMin.at(ic).at(ivar) = FLT_MAX;
         fMax.at(ic).at(ivar) = -FLT_MAX;
      }
   }

   const Int_t all = GetNClasses();
   std::vector<Event*>::const_iterator evIt = events.begin();
   for (;evIt!=events.end();evIt++) {
      for (UInt_t ivar=0; ivar<nvars; ivar++) {
         Float_t val = (*evIt)->GetValue(ivar);
         UInt_t cls = (*evIt)->GetClass();

         if (fMin.at(cls).at(ivar) > val) fMin.at(cls).at(ivar) = val;
         if (fMax.at(cls).at(ivar) < val) fMax.at(cls).at(ivar) = val;

         if (GetNClasses() != 1) {
            if (fMin.at(all).at(ivar) > val) fMin.at(all).at(ivar) = val;
            if (fMax.at(all).at(ivar) < val) fMax.at(all).at(ivar) = val;
         }
      }
      for (UInt_t itgt=0; itgt<ntgts; itgt++) {
         Float_t val = (*evIt)->GetTarget(itgt);
         UInt_t cls = (*evIt)->GetClass();

         if (fMin.at(cls).at(nvars+itgt) > val) fMin.at(cls).at(nvars+itgt) = val;
         if (fMax.at(cls).at(nvars+itgt) < val) fMax.at(cls).at(nvars+itgt) = val;

         if (GetNClasses() != 1) {
            if (fMin.at(all).at(nvars+itgt) > val) fMin.at(all).at(nvars+itgt) = val;
            if (fMax.at(all).at(nvars+itgt) < val) fMax.at(all).at(nvars+itgt) = val;
         }
      }
   }

   return;
}

//_______________________________________________________________________
std::vector<TString>* TMVA::VariableNormalizeTransform::GetTransformationStrings( Int_t cls ) const
{
   // creates string with variable transformations applied

   // if cls (the class chosen by the user) not existing, assume that user wants to 
   // have the matrix for all classes together. 
   if (cls < 0 || cls > GetNClasses()) cls = GetNClasses();

   const UInt_t nvar = GetNVariables();
   std::vector<TString>* strVec = new std::vector<TString>(nvar);

   Float_t min, max;
   for (Int_t ivar=nvar-1; ivar>=0; ivar--) {
      min = fMin.at(cls).at(ivar); 
      max = fMax.at(cls).at(ivar);
      Float_t offset = min;
      Float_t scale  = 1.0/(max-min);      
      TString str("");
      if (offset < 0) str = Form( "2*%g*([%s] + %g) - 1", scale, Variables()[ivar].GetLabel().Data(), -offset );
      else            str = Form( "2*%g*([%s] - %g) - 1", scale, Variables()[ivar].GetLabel().Data(),  offset );
      (*strVec)[ivar] = str;
   }

   return strVec;
}

//_______________________________________________________________________
void TMVA::VariableNormalizeTransform::WriteTransformationToStream( std::ostream& o ) const
{
   // write the decorrelation matrix to the stream
   o << "# min max for all variables for all classes one after the other and as a last entry for all classes together" << std::endl;

   Int_t numC = GetNClasses()+1;
   if (GetNClasses() <= 1 ) numC = 1;

   UInt_t nvars = GetNVariables();
   UInt_t ntgts = GetNTargets();

   for (Int_t icls = 0; icls < numC; icls++ ) {
      o << icls << std::endl;
      for (UInt_t ivar=0; ivar<nvars; ivar++)
         o << std::setprecision(12) << std::setw(20) << fMin.at(icls).at(ivar) << " " 
           << std::setprecision(12) << std::setw(20) << fMax.at(icls).at(ivar) << std::endl;
      for (UInt_t itgt=0; itgt<ntgts; itgt++)
         o << std::setprecision(12) << std::setw(20) << fMin.at(icls).at(nvars+itgt) << " " 
           << std::setprecision(12) << std::setw(20) << fMax.at(icls).at(nvars+itgt) << std::endl;
   }
   o << "##" << std::endl;
}

//_______________________________________________________________________
void TMVA::VariableNormalizeTransform::AttachXMLTo(void* parent) 
{
   // create XML description of Normalize transformation
   Int_t numC = (GetNClasses()<= 1)?1:GetNClasses()+1;
   UInt_t nvars = GetNVariables();
   UInt_t ntgts = GetNTargets();

   void* trfxml = gTools().AddChild(parent, "Transform");
   gTools().AddAttr(trfxml, "Name", "Normalize");
   gTools().AddAttr(trfxml, "NVariables", nvars);
   gTools().AddAttr(trfxml, "NTargets",   ntgts);

   for( Int_t icls=0; icls<numC; icls++ ) {
      void* clsxml = gTools().AddChild(trfxml, "Class");
      gTools().AddAttr(clsxml, "ClassIndex", icls);
      void* varsxml = gTools().AddChild(clsxml, "Variables");
      for (UInt_t ivar=0; ivar<nvars; ivar++) {
         void* varxml = gTools().AddChild(varsxml, "Variable");
         gTools().AddAttr(varxml, "VarIndex", ivar);
         gTools().AddAttr(varxml, "Min",      fMin.at(icls).at(ivar) );
         gTools().AddAttr(varxml, "Max",      fMax.at(icls).at(ivar) );
      }
      void* tgtsxml = gTools().AddChild(clsxml, "Targets");
      for (UInt_t itgt=0; itgt<ntgts; itgt++) {
         void* tgtxml = gTools().AddChild(tgtsxml, "Target");
         gTools().AddAttr(tgtxml, "TargetIndex", itgt);
         gTools().AddAttr(tgtxml, "Min",         fMin.at(icls).at(nvars+itgt) );
         gTools().AddAttr(tgtxml, "Max",         fMax.at(icls).at(nvars+itgt) );
      }
   }
}

//_______________________________________________________________________
void TMVA::VariableNormalizeTransform::ReadFromXML( void* trfnode ) 
{
   // Read the transformation matrices from the xml node
   UInt_t classindex, varindex, tgtindex, nvars, ntgts;

   gTools().ReadAttr(trfnode, "NVariables", nvars);
   gTools().ReadAttr(trfnode, "NTargets",   ntgts);

   void* ch = gTools().GetChild( trfnode );
   while(ch) {
      gTools().ReadAttr(ch, "ClassIndex", classindex);

      fMin.resize(classindex+1);
      fMax.resize(classindex+1);
      fMin[classindex].resize(nvars+ntgts,Float_t(0));
      fMax[classindex].resize(nvars+ntgts,Float_t(0));

      void* clch = gTools().GetChild( ch );
      while(clch) {
         TString nodeName(gTools().GetName(clch));
         if(nodeName=="Variables") {
            void* varch = gTools().GetChild( clch );
            while(varch) {
               gTools().ReadAttr(varch, "VarIndex", varindex);
               gTools().ReadAttr(varch, "Min",      fMin[classindex][varindex]);
               gTools().ReadAttr(varch, "Max",      fMax[classindex][varindex]);
               varch = gTools().GetNextChild( varch );
            }
         } else if (nodeName=="Targets") {
            void* tgtch = gTools().GetChild( clch );
            while(tgtch) {
               gTools().ReadAttr(tgtch, "TargetIndex", tgtindex);
               gTools().ReadAttr(tgtch, "Min",      fMin[classindex][nvars+tgtindex]);
               gTools().ReadAttr(tgtch, "Max",      fMax[classindex][nvars+tgtindex]);
               tgtch = gTools().GetNextChild( tgtch );
            }
         }
         clch = gTools().GetNextChild( clch );
      }
      ch = gTools().GetNextChild( ch );
   }
   SetCreated();
}

//_______________________________________________________________________
void
TMVA::VariableNormalizeTransform::BuildTransformationFromVarInfo( const std::vector<TMVA::VariableInfo>& var ) {
   // this method is only used when building a normalization transformation
   // from old text files
   // in this case regression didn't exist and there were no targets

   UInt_t nvars = GetNVariables();

   if(var.size() != nvars)
      Log() << kFATAL << "<BuildTransformationFromVarInfo> can't build transformation,"
            << " since the number of variables disagree" << Endl;

   UInt_t numC = (GetNClasses()<=1)?1:GetNClasses()+1;
   fMin.clear();fMin.resize( numC );
   fMax.clear();fMax.resize( numC );


   for(UInt_t cls=0; cls<numC; ++cls) {
      fMin[cls].resize(nvars+GetNTargets(),0);
      fMax[cls].resize(nvars+GetNTargets(),0);
      UInt_t vidx(0);
      for(std::vector<TMVA::VariableInfo>::const_iterator v = var.begin(); v!=var.end(); ++v, ++vidx) {
         fMin[cls][vidx] = v->GetMin();
         fMax[cls][vidx] = v->GetMax();
      }
   }
   SetCreated();
}

//_______________________________________________________________________
void TMVA::VariableNormalizeTransform::ReadTransformationFromStream( std::istream& istr, const TString& )
{
   // Read the variable ranges from an input stream

   UInt_t nvars = GetNVariables();
   UInt_t ntgts = GetNTargets();
   char buf[512];
   char buf2[512];
   istr.getline(buf,512);
   TString strvar, dummy;
   Int_t icls;
   TString test;
   while (!(buf[0]=='#'&& buf[1]=='#')) { // if line starts with ## return
      char* p = buf;
      while (*p==' ' || *p=='\t') p++; // 'remove' leading whitespace
      if (*p=='#' || *p=='\0') {
         istr.getline(buf,512);
         continue; // if comment or empty line, read the next line
      }
      std::stringstream sstr(buf);
      sstr >> icls;
      for (UInt_t ivar=0;ivar<nvars;ivar++) {
         istr.getline(buf2,512); // reading the next line
         std::stringstream sstr2(buf2);
         sstr2 >> fMin[icls][ivar] >> fMax[icls][ivar];
      }
      for (UInt_t itgt=0;itgt<ntgts;itgt++) {
         istr.getline(buf2,512); // reading the next line
         std::stringstream sstr2(buf2);
         sstr2 >> fMin[icls][nvars+itgt] >> fMax[icls][nvars+itgt];
      }
      istr.getline(buf,512); // reading the next line
   }
   SetCreated();
}

//_______________________________________________________________________
void TMVA::VariableNormalizeTransform::PrintTransformation( ostream& o ) 
{
   // prints the transformation ranges

   Int_t numC = GetNClasses()+1;
   if (GetNClasses() <= 1 ) numC = 1;

   UInt_t nvars = GetNVariables();
   UInt_t ntgts = GetNTargets();
   for (Int_t icls = 0; icls < numC; icls++ ) {
      Log() << kINFO << "Transformation for class " << icls << " based on these ranges:" << Endl;
      Log() << kINFO << "Variables:" << Endl;
      for (UInt_t ivar=0; ivar<nvars; ivar++)
         o << std::setw(20) << fMin[icls][ivar] << std::setw(20) << fMax[icls][ivar] << std::endl;
      Log() << kINFO << "Targets:" << Endl;
      for (UInt_t itgt=0; itgt<ntgts; itgt++)
         o << std::setw(20) << fMin[icls][nvars+itgt] << std::setw(20) << fMax[icls][nvars+itgt] << std::endl;
   }
}

//_______________________________________________________________________
void TMVA::VariableNormalizeTransform::MakeFunction( std::ostream& fout, const TString& fcncName, 
                                                     Int_t part, UInt_t trCounter, Int_t ) 
{
   // creates a normalizing function
   // TODO include target-transformation into makefunction
   UInt_t numC = fMin.size();
   if (part==1) {
      fout << std::endl;
      fout << "   double fMin_"<<trCounter<<"["<<numC<<"]["<<GetNVariables()<<"];" << std::endl;
      fout << "   double fMax_"<<trCounter<<"["<<numC<<"]["<<GetNVariables()<<"];" << std::endl;
   }

   if (part==2) {
      fout << std::endl;
      fout << "//_______________________________________________________________________" << std::endl;
      fout << "inline void " << fcncName << "::InitTransform_"<<trCounter<<"()" << std::endl;
      fout << "{" << std::endl;
      for (UInt_t ivar=0; ivar<GetNVariables(); ivar++) {
         Float_t min = FLT_MAX;
         Float_t max = -FLT_MAX;
         for (UInt_t icls = 0; icls < numC; icls++) {
            min = TMath::Min(min, fMin.at(icls).at(ivar) );
            max = TMath::Max(max, fMax.at(icls).at(ivar) );
            fout << "   fMin_"<<trCounter<<"["<<icls<<"]["<<ivar<<"] = " << std::setprecision(12)
                 << min << ";" << std::endl;
            fout << "   fMax_"<<trCounter<<"["<<icls<<"]["<<ivar<<"] = " << std::setprecision(12)
                 << max << ";" << std::endl;
         }
      }
      fout << "}" << std::endl;
      fout << std::endl;
      fout << "//_______________________________________________________________________" << std::endl;
      fout << "inline void " << fcncName << "::Transform_"<<trCounter<<"( std::vector<double>& iv, int cls) const" << std::endl;
      fout << "{" << std::endl;
      fout << "if (cls < 0 || cls > "<<GetNClasses()<<") {"<< std::endl;
      fout << "   if ("<<GetNClasses()<<" > 1 ) cls = "<<GetNClasses()<<";"<< std::endl;
      fout << "   else cls = "<<(fMin.size()==1?0:2)<<";"<< std::endl;
      fout << "}"<< std::endl;
      fout << "   for (int ivar=0;ivar<"<<GetNVariables()<<";ivar++) {" << std::endl;
      fout << "      double offset = fMin_"<<trCounter<<"[cls][ivar];" << std::endl;
      fout << "      double scale  = 1.0/(fMax_"<<trCounter<<"[cls][ivar]-fMin_"<<trCounter<<"[cls][ivar]);" << std::endl;
      fout << "      iv[ivar] = (iv[ivar]-offset)*scale * 2 - 1;" << std::endl;
      fout << "   }" << std::endl;
      fout << "}" << std::endl;
   }
}
