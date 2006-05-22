// @(#)root/tmva $Id: Tools.h,v 1.5 2006/05/22 08:04:39 andreas.hoecker Exp $   
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : Tools                                                                 *
 *                                                                                *
 * Description:                                                                   *
 *      Global auxiliary applications and data treatment routines                 *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Xavier Prudent  <prudent@lapp.in2p3.fr>  - LAPP, France                   *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-KP Heidelberg, Germany     *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland,                                                        * 
 *      U. of Victoria, Canada,                                                   * 
 *      MPI-KP Heidelberg, Germany,                                               * 
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://mva.sourceforge.net/license.txt)                                       *
 *                                                                                *
 * File and Version Information:                                                  *
 * $Id: Tools.h,v 1.5 2006/05/22 08:04:39 andreas.hoecker Exp $        
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

#include "TList.h"

#ifndef ROOT_TMVA_Event
#include "TMVA/Event.h"
#endif

#ifndef ROOT_TMVA_TMatrixDSymEigen
#include "TMatrixDSymEigen.h"
#endif

class TTree;
class TH1;
class TSpline;

#define __N__(a1,a2,a3) Tools::NormVariable(a1,a2,a3)

using std::vector;

namespace TMVA {

  namespace Tools {

    //  TString GetName( void ) { return "Tools"; }

    // simple statistics operations on tree entries
    void  ComputeStat( TTree* theTree, TString theVarName,
		       Double_t&, Double_t&, Double_t&, 
		       Double_t&, Double_t&, Double_t&, Bool_t norm = kFALSE );
  
    // simple statistics operations on Events in a Vector collection class 
    void  ComputeStat( std::vector<Event*>, Int_t ivar,
		       Double_t&, Double_t&, Double_t&, 
		       Double_t&, Double_t&, Double_t&, Bool_t norm = kFALSE );
  


    // creates histograms normalized to one
    TH1* projNormTH1F( TTree* theTree, TString theVarName,
		       TString name, Int_t nbins, 
		       Double_t xmin, Double_t xmax, TString cut );

    // normalize histogram by its integral
    Double_t NormHist( TH1* theHist, Double_t norm = 1.0 );

    // parser for TString phrase with items separated by ':'
    TList* ParseFormatLine( TString theString );

    // returns covariance matrix for all variables in tree
    void GetCovarianceMatrix( TTree* theTree, TMatrixDBase *theMatrix, 
			      vector<TString>* theVars, Int_t theType, Bool_t norm = kFALSE );

    // returns correlation matrix for all variables in tree
    void GetCorrelationMatrix( TTree* theTree, TMatrixDBase *theMatrix, 
			       vector<TString>* theVars, Int_t theType );

    // returns the square-root of a symmetric matrix: symMat = sqrtMat*sqrtMat
    void GetSQRootMatrix( TMatrixDSym* symMat, TMatrixD* sqrtMat );

    // type-safe accessor to TTree elements
    Double_t GetValue( TTree *theTree, Int_t entry, TString varname );

    // check spline quality by comparison with initial histogram
    Bool_t CheckSplines( TH1*, TSpline* );

    // normalization of variable output
    Double_t NormVariable( Double_t x, Double_t xmin, Double_t xmax );

    // vector rescaling
    vector<Double_t> MVADiff( vector<Double_t>&, vector<Double_t>& );
    void Scale     ( vector<Double_t>&, Double_t );
    void Scale     ( vector<Float_t> &, Float_t  );
  
    // re-arrange a vector of arrays (vectors) in a way such that the first array
    // is ordered, and the other arrays reshuffeld accordingly
    void UsefulSortDescending( vector< vector<Double_t> >&, vector<TString>* vs = 0 );
    void UsefulSortAscending ( vector< vector<Double_t> >& );
    
    void UsefulSortDescending( vector<Double_t>& );
    void UsefulSortAscending ( vector<Double_t>& );

    Int_t GetIndexMaxElement ( vector<Double_t>& );
    Int_t GetIndexMinElement ( vector<Double_t>& );

  }

} // namespace TMVA

#endif

