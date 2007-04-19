// @(#)root/tmva $Id: Tools.h,v 1.10 2007/01/16 09:37:03 brun Exp $   
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : Tools                                                                 *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Global auxiliary applications and data treatment routines                 *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Xavier Prudent  <prudent@lapp.in2p3.fr>  - LAPP, France                   *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland,                                                        *
 *      U. of Victoria, Canada,                                                   *
 *      MPI-K Heidelberg, Germany ,                                               *
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#ifndef ROOT_TMVA_Tools
#define ROOT_TMVA_Tools

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Tools (namespace)                                                    //
//                                                                      //
// Global auxiliary applications and data treatment routines            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <vector>
#include <string>

#ifndef ROOT_TMVA_TMatrixDSymEigen
#include "TMatrixDSymEigen.h"
#endif
#ifndef ROOT_TMVA_MsgLogger
#include "TMVA/MsgLogger.h"
#endif

class TList;
class TTree;
class TString;
class TH1;
class TSpline;

#define __N__(a1,a2,a3) Tools::NormVariable(a1,a2,a3)

namespace TMVA {

   class Event;
   
   namespace Tools {

      // simple statistics operations on tree entries
      void  ComputeStat( TTree* theTree, const TString& theVarName,
                         Double_t&, Double_t&, Double_t&, 
                         Double_t&, Double_t&, Double_t&, Bool_t norm = kFALSE );
  
      // creates histograms normalized to one
      TH1* projNormTH1F( TTree* theTree, TString theVarName,
                         TString name, Int_t nbins, 
                         Double_t xmin, Double_t xmax, TString cut );

      // normalize histogram by its integral
      Double_t NormHist( TH1* theHist, Double_t norm = 1.0 );

      // parser for TString phrase with items separated by a character
      TList* ParseFormatLine( TString theString, const char * sep = ":" );

      // parse option string for ANN methods
      std::vector<Int_t>* ParseANNOptionString( TString theOptions, Int_t nvar,
                                                std::vector<Int_t>* nodes );
      
      // returns the square-root of a symmetric matrix: symMat = sqrtMat*sqrtMat
      void GetSQRootMatrix( TMatrixDSym* symMat, TMatrixD*& sqrtMat );

      // turns covariance into correlation matrix
      const TMatrixD* GetCorrelationMatrix( const TMatrixD* covMat );

      // check spline quality by comparison with initial histogram
      Bool_t CheckSplines( TH1*, TSpline* );

      // normalization of variable output
      Double_t NormVariable( Double_t x, Double_t xmin, Double_t xmax );

      // return separation of two histograms
      Double_t GetSeparation( TH1* S, TH1* B );

      // vector rescaling
      std::vector<Double_t> MVADiff( std::vector<Double_t>&, std::vector<Double_t>& );
      void Scale     ( std::vector<Double_t>&, Double_t );
      void Scale     ( std::vector<Float_t> &, Float_t  );
  
      // re-arrange a vector of arrays (vectors) in a way such that the first array
      // is ordered, and the other arrays reshuffeld accordingly
      void UsefulSortDescending( std::vector< std::vector<Double_t> >&, std::vector<TString>* vs = 0 );
      void UsefulSortAscending ( std::vector< std::vector<Double_t> >& );
    
      void UsefulSortDescending( std::vector<Double_t>& );
      void UsefulSortAscending ( std::vector<Double_t>& );

      Int_t GetIndexMaxElement ( std::vector<Double_t>& );
      Int_t GetIndexMinElement ( std::vector<Double_t>& );

      // check if input string contains regular expression
      Bool_t  ContainsRegularExpression( const TString& s );
      TString ReplaceRegularExpressions( const TString& s, TString replace = "+" );

      // routines for formatted output -----------------
      void FormattedOutput( const std::vector<Double_t>&, const std::vector<TString>&, 
                            const TString titleVars, const TString titleValues, MsgLogger& logger,
                            TString format = "%+1.3f" );
      void FormattedOutput( const TMatrixD&, const std::vector<TString>&, MsgLogger& logger );

      // output logger
      MsgLogger& Logger();

      const TString __regexp__ = "!%^&()'<>?= ";

      const TString& Color(const TString &);

      // print welcome message (to be called from, eg, .TMVAlogon)
      enum EWelcomeMessage {
         kStandardWelcomeMsg = 1,
         kIsometricWelcomeMsg,
         kBlockWelcomeMsg,
         kLeanWelcomeMsg,
         kLogoWelcomeMsg,
         kSmall1WelcomeMsg,
         kSmall2WelcomeMsg,
         kOriginalWelcomeMsgColor,
         kOriginalWelcomeMsgBW
      };

      void TMVAWelcomeMessage();
      void TMVAWelcomeMessage( MsgLogger& logger, EWelcomeMessage m = kStandardWelcomeMsg );
      void TMVAVersionMessage( MsgLogger& logger );
   }

} // namespace TMVA

#endif

