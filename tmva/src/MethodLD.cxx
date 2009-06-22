// @(#)root/tmva $Id$ 
// Author: Krzysztof Danielowski, Kamil Kraszewski, Maciej Kruk, Jan Therhaag

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodLD                                                              *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Linear Discriminant - Simple Linear Regression and Classification 	       *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Krzysztof Danielowski   <danielow@cern.ch>      - IFJ PAN & AGH, Poland   *
 *      Kamil Kraszewski        <kalq@cern.ch>          - IFJ PAN & UJ, Poland    *
 *      Maciej Kruk             <mkruk@cern.ch>         - IFJ PAN & AGH, Poland   *
 *      Jan Therhaag            <therhaag@physik.uni-bonn.de> - Uni Bonn, Germany *
 *                                                                                *
 * Copyright (c) 2008:                                                            *
 *      CERN, Switzerland                                                         *
 *      PAN, Poland                                                               *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 *                                                                                *
 **********************************************************************************/

#include <iomanip>

#include "TMath.h"
#include "Riostream.h"
#include "TMatrix.h"
#include "TMatrixD.h"
#include "TXMLEngine.h"

#include "TMVA/VariableTransformBase.h"
#include "TMVA/MethodLD.h"
#include "TMVA/Tools.h"
#include "TMVA/Ranking.h"
#include "TMVA/Types.h"
#include "TMVA/PDF.h"
#include "TMVA/ClassifierFactory.h"

REGISTER_METHOD(LD)

ClassImp(TMVA::MethodLD)

//_______________________________________________________________________
TMVA::MethodLD::MethodLD( const TString& jobName,
                          const TString& methodTitle,
                          DataSetInfo& dsi, 
                          const TString& theOption,
                          TDirectory* theTargetDir ) :
   MethodBase( jobName, Types::kLD, methodTitle, dsi, theOption, theTargetDir ),
   fSumMatx   ( 0 ), 
   fSumValMatx( 0 ),
   fCoeffMatx ( 0 ),
   fLDCoeff   ( 0 )
{
	// standard constructor for the LD 
}

//_______________________________________________________________________
TMVA::MethodLD::MethodLD( DataSetInfo& theData, const TString& theWeightFile, TDirectory* theTargetDir )
   : MethodBase( Types::kLD, theData, theWeightFile, theTargetDir ),
     fSumMatx   ( 0 ), 
     fSumValMatx( 0 ),
     fCoeffMatx ( 0 ),
     fLDCoeff   ( 0 )
{
	// constructor from weight file
}

//_______________________________________________________________________
void TMVA::MethodLD::Init( void )
{
   // default initialization called by all constructors

   if (DoRegression()) fNRegOut = DataInfo().GetNTargets();
   else                fNRegOut = 1;

   fLDCoeff = new vector< vector< Double_t >* >(fNRegOut);
   for (Int_t iout = 0; iout<fNRegOut; iout++) (*fLDCoeff)[iout] = new std::vector<Double_t>( GetNvar()+1 );

   // the minimum requirement to declare an event signal-like
   SetSignalReferenceCut( 0.0 );
}

//_______________________________________________________________________
TMVA::MethodLD::~MethodLD( void )
{
   // destructor
   if (fSumMatx)    { delete fSumMatx;    fSumMatx    = 0; }
   if (fSumValMatx) { delete fSumValMatx; fSumValMatx = 0; }
   if (fCoeffMatx)  { delete fCoeffMatx;  fCoeffMatx  = 0; }
   if (fLDCoeff) { 
      for (vector< vector< Double_t >* >::iterator vi=fLDCoeff->begin(); vi!=fLDCoeff->end(); vi++)
         if (*vi) {	delete *vi;	*vi = 0;	}
      delete fLDCoeff; fLDCoeff = 0; 
   }
}

//_______________________________________________________________________
Bool_t TMVA::MethodLD::HasAnalysisType( Types::EAnalysisType type, UInt_t numberClasses, UInt_t numberTargets )
{
   // LD can handle classification with 2 classes and regression with one regression-target
   if      (type == Types::kClassification && numberClasses == 2) return kTRUE;
   else if (type == Types::kRegression     && numberTargets == 1) {
     log() << "regression with " << numberTargets << " targets.";
     return kTRUE;
   }
   else return kFALSE;
}


//_______________________________________________________________________
void TMVA::MethodLD::Train( void )
{
   // compute fSumMatx
   GetSum();

   // compute fSumValMatx
   GetSumVal();

   // compute fCoeffMatx and fLDCoeff
   GetLDCoeff();
   
   // nice output
   PrintCoefficients();
}

//_______________________________________________________________________
Double_t TMVA::MethodLD::GetMvaValue( Double_t* err )
{
   //Returns the MVA classification output
   const Event* ev = GetEvent();

   if (fRegressionReturnVal == NULL) fRegressionReturnVal = new vector< Float_t >();
   fRegressionReturnVal->resize( fNRegOut );
   
   for (Int_t iout = 0; iout<fNRegOut; iout++) {
      (*fRegressionReturnVal)[iout] = (*(*fLDCoeff)[iout])[0] ;

      int icoeff=0;
      for (std::vector<Float_t>::const_iterator it = ev->GetValues().begin();it!=ev->GetValues().end();++it){
         (*fRegressionReturnVal)[iout] += (*(*fLDCoeff)[iout])[++icoeff] * (*it);
      }
   }

   // cannot determine error
   if (err != 0) *err = -1;

   return (*fRegressionReturnVal)[0];
}
		  
//_______________________________________________________________________
const std::vector< Float_t >& TMVA::MethodLD::GetRegressionValues()
{
  //Calculates the regression output
   const Event* ev = GetEvent();

   if (fRegressionReturnVal == NULL) fRegressionReturnVal = new vector< Float_t >();
   fRegressionReturnVal->resize( fNRegOut );
   
   for (Int_t iout = 0; iout<fNRegOut; iout++) {
      (*fRegressionReturnVal)[iout] = (*(*fLDCoeff)[iout])[0] ;

      int icoeff = 0;      
      for (std::vector<Float_t>::const_iterator it = ev->GetValues().begin();it!=ev->GetValues().end();++it){
         (*fRegressionReturnVal)[iout] += (*(*fLDCoeff)[iout])[++icoeff] * (*it);
      }
   }

   // perform inverse transformation 
   Event* evT = new Event(*ev);
   for (Int_t iout = 0; iout<fNRegOut; iout++) evT->SetTarget(iout,(*fRegressionReturnVal)[iout]);

   const Event* evT2 = GetTransformationHandler().InverseTransform( evT );
   fRegressionReturnVal->clear();
   for (Int_t iout = 0; iout<fNRegOut; iout++) fRegressionReturnVal->push_back(evT2->GetTarget(iout));

   delete evT;
   return (*fRegressionReturnVal);
}

//_______________________________________________________________________
void TMVA::MethodLD::InitMatrices( void )
{
   // Initializaton method; creates global matrices and vectors

   fSumMatx    = new TMatrixD( GetNvar()+1, GetNvar()+1 );
   fSumValMatx = new TMatrixD( GetNvar()+1, fNRegOut ); 
   fCoeffMatx  = new TMatrixD( GetNvar()+1, fNRegOut ); 

}

//_______________________________________________________________________
void TMVA::MethodLD::GetSum( void )
{
   // Calculates the matrix transposed(X)*W*X with W being the diagonal weight matrix 
   // and X the coordinates values
   const UInt_t nvar = DataInfo().GetNVariables();

   for (UInt_t ivar = 0; ivar<=nvar; ivar++)
      for (UInt_t jvar = 0; jvar<=nvar; jvar++) (*fSumMatx)( ivar, jvar ) = 0;

   // compute sample means
   Long64_t nevts = Data()->GetNEvents();
   for (Int_t ievt=0; ievt<nevts; ievt++) {
      const Event * ev = GetEvent(ievt);
      Double_t weight = ev->GetWeight();

      if (IgnoreEventsWithNegWeightsInTraining() && weight <= 0) continue;

      // Sum of weights		
      (*fSumMatx)( 0, 0 ) += weight;

      // Sum of coordinates
      for (UInt_t ivar=0; ivar<nvar; ivar++) {
         (*fSumMatx)( ivar+1, 0 ) += ev->GetVal( ivar ) * weight;
         (*fSumMatx)( 0, ivar+1 ) += ev->GetVal( ivar ) * weight;
      }
		
      // Sum of products of coordinates
      for (UInt_t ivar=0; ivar<nvar; ivar++)
         for (UInt_t jvar=0; jvar<nvar; jvar++)
            (*fSumMatx)( ivar+1, jvar+1 ) += ev->GetVal( ivar ) * ev->GetVal( jvar ) * weight;
   }
}

//_______________________________________________________________________
void TMVA::MethodLD::GetSumVal( void )
{
   //Calculates the vector transposed(X)*W*Y with Y being the target vector
   const UInt_t nvar = DataInfo().GetNVariables();
	
   for (Int_t ivar = 0; ivar<fNRegOut; ivar++)
      for (UInt_t jvar = 0; jvar<=nvar; jvar++)
         (*fSumValMatx)(jvar,ivar) = 0;
		
   // Sum of coordinates multiplied by values
   for (Int_t ievt=0; ievt<Data()->GetNEvents(); ievt++) {

      // retrieve the event
      const Event* ev = GetEvent(ievt);
      Double_t weight = ev->GetWeight();

      // in case event with neg weights are to be ignored
      if (IgnoreEventsWithNegWeightsInTraining() && weight <= 0) continue; 

	
      for (Int_t ivar=0; ivar<fNRegOut; ivar++) {

         Double_t val = weight;

         if (!DoRegression())
            val *= ev->IsSignal();
         else	//for regression
            val *= ev->GetTarget( ivar ); 
	 
         (*fSumValMatx)( 0,ivar ) += val; 
         for (UInt_t jvar=0; jvar<nvar; jvar++) 
            (*fSumValMatx)(jvar+1,ivar ) += ev->GetVal(jvar) * val;
      }
   }
}

//_______________________________________________________________________
void TMVA::MethodLD::GetLDCoeff( void )
{
   //Calculates the coeffiecients used for classification/regression
   const UInt_t nvar = DataInfo().GetNVariables();

   for (Int_t ivar = 0; ivar<fNRegOut; ivar++){
	
      TMatrixD invSum( *fSumMatx );
      if ( TMath::Abs(invSum.Determinant()) < 10E-24 ) {
         log() << kWARNING << "<GetCoeff> matrix is almost singular with determinant="
                 << TMath::Abs(invSum.Determinant()) 
                 << " did you use the variables that are linear combinations or highly correlated?" 
                 << Endl;
      }
      if ( TMath::Abs(invSum.Determinant()) < 10E-120 ) {
         log() << kFATAL << "<GetCoeff> matrix is singular with determinant="
                 << TMath::Abs(invSum.Determinant())  
                 << " did you use the variables that are linear combinations?" 
                 << Endl;
      }
      invSum.Invert();

      fCoeffMatx = new TMatrixD( invSum * (*fSumValMatx));
      for (UInt_t jvar = 0; jvar<nvar+1; jvar++) {
         (*(*fLDCoeff)[ivar])[jvar] = (*fCoeffMatx)(jvar, ivar );
      }
      if (!DoRegression()) {
         (*(*fLDCoeff)[ivar])[0]=0.0;
         for (UInt_t jvar = 1; jvar<nvar+1; jvar++)
            (*(*fLDCoeff)[ivar])[0]+=(*fCoeffMatx)(jvar,ivar)*(*fSumMatx)(0,jvar)/(*fSumMatx)( 0, 0 );
         (*(*fLDCoeff)[ivar])[0]/=-2.0;
      }
      
   }
}

//_______________________________________________________________________
void  TMVA::MethodLD::WriteWeightsToStream( std::ostream& o ) const
{  
   // save the weights
   for (Int_t iout=0; iout<fNRegOut; iout++){
      for (UInt_t icoeff=0; icoeff<GetNvar()+1; icoeff++)
         o << std::setprecision(12) << (*(*fLDCoeff)[iout])[icoeff] << endl;
   }
}

//_______________________________________________________________________
void  TMVA::MethodLD::ReadWeightsFromStream( istream& istr )
{
   // read LD coefficients from weight file
   for (Int_t iout=0; iout<fNRegOut; iout++) {
     for (UInt_t icoeff=0; icoeff<GetNvar()+1; icoeff++) {
	  istr >> (*(*fLDCoeff)[iout])[icoeff];
     }
   }
}

//_______________________________________________________________________
void TMVA::MethodLD::AddWeightsXMLTo( void* parent ) const 
{
   // create XML description for LD classification and regression 
   // (for arbitrary number of output classes/targets)

   void* wght = gTools().xmlengine().NewChild(parent, 0, "Weights");
   gTools().AddAttr( wght, "NOut",   fNRegOut    );
   gTools().AddAttr( wght, "NCoeff", GetNvar()+1 );
   for (Int_t iout=0; iout<fNRegOut; iout++) {
      for (UInt_t icoeff=0; icoeff<GetNvar()+1; icoeff++) {
         void* coeffxml = gTools().xmlengine().NewChild( wght, 0, "Coefficient" );
         gTools().AddAttr( coeffxml, "IndexOut",   iout   );
         gTools().AddAttr( coeffxml, "IndexCoeff", icoeff );
         gTools().AddAttr( coeffxml, "Value",      (*(*fLDCoeff)[iout])[icoeff] );
      }
   }
}
  
//_______________________________________________________________________
void TMVA::MethodLD::ReadWeightsFromXML( void* wghtnode ) 
{
   // read coefficients from xml weight file
   UInt_t ncoeff;
   gTools().ReadAttr( wghtnode, "NOut",   fNRegOut );
   gTools().ReadAttr( wghtnode, "NCoeff", ncoeff   );
   
   // sanity checks
   if (ncoeff != GetNvar()+1) log() << kFATAL << "Mismatch in number of output variables/coefficients: " 
                                      << ncoeff << " != " << GetNvar()+1 << Endl;

   // create vector with coefficients (double vector due to arbitrary output dimension)
   if (fLDCoeff) { 
      for (vector< vector< Double_t >* >::iterator vi=fLDCoeff->begin(); vi!=fLDCoeff->end(); vi++)
         if (*vi) {	delete *vi;	*vi = 0;	}
      delete fLDCoeff; fLDCoeff = 0; 
   }
   fLDCoeff = new vector< vector< Double_t >* >(fNRegOut);
   for (Int_t ivar = 0; ivar<fNRegOut; ivar++) (*fLDCoeff)[ivar] = new std::vector<Double_t>( ncoeff );

   void* ch = gTools().xmlengine().GetChild(wghtnode);
   Double_t coeff;
   Int_t iout, icoeff;
   while (ch) {
      gTools().ReadAttr( ch, "IndexOut",   iout   );
      gTools().ReadAttr( ch, "IndexCoeff", icoeff );
      gTools().ReadAttr( ch, "Value",      coeff  );

      (*(*fLDCoeff)[iout])[icoeff] = coeff;

      ch = gTools().xmlengine().GetNext(ch);
   }
}

//_______________________________________________________________________
void TMVA::MethodLD::MakeClassSpecific( std::ostream& fout, const TString& className ) const
{
   // write LD-specific classifier response
   fout << "   std::vector<double> fLDCoefficients;" << endl;
   fout << "};" << endl;
   fout << "" << endl;
   fout << "inline void " << className << "::Initialize() " << endl;
   fout << "{" << endl;
   for (UInt_t ivar=0; ivar<GetNvar()+1; ivar++) {
      fout << "   fLDCoefficients.push_back( " << std::setprecision(12) << (*(*fLDCoeff)[0])[ivar] << " );" << endl;
   }
   fout << endl;
   fout << "   // sanity check" << endl;
   fout << "   if (fLDCoefficients.size() != fNvars+1) {" << endl;
   fout << "      std::cout << \"Problem in class \\\"\" << fClassName << \"\\\"::Initialize: mismatch in number of input values\"" << endl;
   fout << "                << fLDCoefficients.size() << \" != \" << fNvars+1 << std::endl;" << endl;
   fout << "      fStatusIsClean = false;" << endl;
   fout << "   }         " << endl;
   fout << "}" << endl;
   fout << endl;
   fout << "inline double " << className << "::GetMvaValue__( const std::vector<double>& inputValues ) const" << endl;
   fout << "{" << endl;
   fout << "   double retval = fLDCoefficients[0];" << endl;
   fout << "   for (size_t ivar = 1; ivar < fNvars+1; ivar++) {" << endl;
   fout << "      retval += fLDCoefficients[ivar]*inputValues[ivar-1];" << endl;
   fout << "   }" << endl;
   fout << endl;
   fout << "   return retval;" << endl;
   fout << "}" << endl;
   fout << endl;
   fout << "// Clean up" << endl;
   fout << "inline void " << className << "::Clear() " << endl;
   fout << "{" << endl;
   fout << "   // clear coefficients" << endl;
   fout << "   fLDCoefficients.clear(); " << endl;
   fout << "}" << endl;
}
//_______________________________________________________________________
const TMVA::Ranking* TMVA::MethodLD::CreateRanking() 
{
   // computes ranking of input variables

   // create the ranking object
   fRanking = new Ranking( GetName(), "Discr. power" );

   for (UInt_t ivar=0; ivar<GetNvar(); ivar++) {
      fRanking->AddRank( *new Rank( GetInputLabel(ivar), TMath::Abs((* (*fLDCoeff)[0])[ivar+1] )) );
   }

   return fRanking;
}

//_______________________________________________________________________
void TMVA::MethodLD::DeclareOptions()
{
   //MethodLD options
	AddPreDefVal(TString("LD"));
}

//_______________________________________________________________________
void TMVA::MethodLD::ProcessOptions()
{
   // this is the preparation for training
   if (HasTrainingTree()) InitMatrices();
}

//_______________________________________________________________________
void TMVA::MethodLD::PrintCoefficients( void ) 
{
   //Display the classification/regression coefficients for each variable
   log() << kINFO << "Results for LD coefficients:" << Endl;

   if (GetTransformationHandler().GetTransformationList().GetSize() != 0) {
      log() << kINFO << "NOTE: The coefficients must be applied to TRANFORMED variables" << Endl;
      log() << kINFO << "      List of the transformation: " << Endl;
      TListIter trIt(&GetTransformationHandler().GetTransformationList());
      while (VariableTransformBase *trf = (VariableTransformBase*) trIt() ) {
         log() << kINFO << "  -- " << trf->GetName() << Endl;
      }
   }
   std::vector<TString>  vars;
   std::vector<Double_t> coeffs;
   for (UInt_t ivar=0; ivar<GetNvar(); ivar++) {
      vars  .push_back( GetInputLabel(ivar) );
      coeffs.push_back( (* (*fLDCoeff)[0])[ivar+1] );
   }
   vars  .push_back( "(offset)" );
   coeffs.push_back((* (*fLDCoeff)[0])[0] );
   TMVA::gTools().FormattedOutput( coeffs, vars, "Variable" , "Coefficient", log() );
   if (IsNormalised()) {
      log() << kINFO << "NOTE: You have chosen to use the \"Normalise\" booking option. Hence, the" << Endl;
      log() << kINFO << "      coefficients must be applied to NORMALISED (') variables as follows:" << Endl;
      Int_t maxL = 0;
      for (UInt_t ivar=0; ivar<GetNvar(); ivar++) if (GetInputLabel(ivar).Length() > maxL) maxL = GetInputLabel(ivar).Length();

      // Print normalisation expression (see Tools.cxx): "2*(x - xmin)/(xmax - xmin) - 1.0"
      for (UInt_t ivar=0; ivar<GetNvar(); ivar++) {
         log() << kINFO 
                 << setw(maxL+9) << TString("[") + GetInputLabel(ivar) + "]' = 2*(" 
                 << setw(maxL+2) << TString("[") + GetInputLabel(ivar) + "]"
                 << setw(3) << (GetXmin(ivar) > 0 ? " - " : " + ")
                 << setw(6) << TMath::Abs(GetXmin(ivar)) << setw(3) << ")/"
                 << setw(6) << (GetXmax(ivar) -  GetXmin(ivar) )
                 << setw(3) << " - 1"
                 << Endl;
      }
      log() << kINFO << "The TMVA Reader will properly account for this normalisation, but if the" << Endl;
      log() << kINFO << "LD classifier is applied outside the Reader, the transformation must be" << Endl;
      log() << kINFO << "implemented -- or the \"Normalise\" option is removed and LD retrained." << Endl;
      log() << kINFO << Endl;
   }
}

//_______________________________________________________________________
void TMVA::MethodLD::GetHelpMessage() const
{
   // get help message text
   //
   // typical length of text line: 
   //         "|--------------------------------------------------------------|"
   log() << Endl;
   log() << gTools().Color("bold") << "--- Short description:" << gTools().Color("reset") << Endl;
   log() << Endl;
   log() << "Linear discriminants select events by distinguishing the mean " << Endl;
   log() << "values of the signal and background distributions in a trans- " << Endl;
   log() << "formed variable space where linear correlations are removed." << Endl;
   log() << "The LD implementation here is equivalent to the \"Fisher\" discriminant" << Endl;
   log() << "for classification, but also provides linear regression." << Endl;
   log() << Endl;
   log() << "   (More precisely: the \"linear discriminator\" determines" << Endl;
   log() << "    an axis in the (correlated) hyperspace of the input " << Endl;
   log() << "    variables such that, when projecting the output classes " << Endl;
   log() << "    (signal and background) upon this axis, they are pushed " << Endl;
   log() << "    as far as possible away from each other, while events" << Endl;
   log() << "    of a same class are confined in a close vicinity. The  " << Endl;
   log() << "    linearity property of this classifier is reflected in the " << Endl;
   log() << "    metric with which \"far apart\" and \"close vicinity\" are " << Endl;
   log() << "    determined: the covariance matrix of the discriminating" << Endl;
   log() << "    variable space.)" << Endl;
   log() << Endl;
   log() << gTools().Color("bold") << "--- Performance optimisation:" << gTools().Color("reset") << Endl;
   log() << Endl;
   log() << "Optimal performance for the linear discriminant is obtained for " << Endl;
   log() << "linearly correlated Gaussian-distributed variables. Any deviation" << Endl;
   log() << "from this ideal reduces the achievable separation power. In " << Endl;
   log() << "particular, no discrimination at all is achieved for a variable" << Endl;
   log() << "that has the same sample mean for signal and background, even if " << Endl;
   log() << "the shapes of the distributions are very different. Thus, the linear " << Endl;
   log() << "discriminant often benefits from a suitable transformation of the " << Endl;
   log() << "input variables. For example, if a variable x in [-1,1] has a " << Endl;
   log() << "a parabolic signal distributions, and a uniform background" << Endl;
   log() << "distributions, their mean value is zero in both cases, leading " << Endl;
   log() << "to no separation. The simple transformation x -> |x| renders this " << Endl;
   log() << "variable powerful for the use in a linear discriminant." << Endl;
   log() << Endl;
   log() << gTools().Color("bold") << "--- Performance tuning via configuration options:" << gTools().Color("reset") << Endl;
   log() << Endl;
   log() << "<None>" << Endl;
}
