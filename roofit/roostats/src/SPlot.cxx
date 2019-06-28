// @(#)root/roostats:$Id$
// Author: Kyle Cranmer   28/07/2008

/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/*****************************************************************************
 * Project: RooStats
 * Package: RooFit/RooStats
 *
 * Authors:
 *   Original code from M. Pivk as part of MLFit package from BaBar.
 *   Modifications:
 *     Giacinto Piacquadio, Maurizio Pierini: modifications for new RooFit version
 *     George H. Lewis, Kyle Cranmer: generalized for weighted events
 *
 * Porting to RooStats (with permission) by Kyle Cranmer, July 2008
 *
 *****************************************************************************/


/** \class RooStats::SPlot
    \ingroup Roostats

   This class calculates sWeights used to create an sPlot.
   The code is based on
   ``SPlot: A statistical tool to unfold data distributions,''
   Nucl. Instrum. Meth. A 555, 356 (2005) [arXiv:physics/0402083].

   An SPlot gives us  the distribution of some variable, x in our
   data sample for a given species (eg. signal or background).
   The result is similar to a likelihood projection plot, but no cuts are made,
   so every event contributes to the distribution.

   To use this class, you first must have a pdf that includes
   yields for (possibly several) different species.
   Create an instance of the class by supplying a data set,
   the pdf, and a list of the yield variables.  The SPlot Class
   will calculate SWeights and include these as columns in the RooDataSet.

*/

#include <vector>
#include <map>

#include "RooStats/SPlot.h"
#include "RooAbsPdf.h"
#include "RooDataSet.h"
#include "RooRealVar.h"
#include "RooGlobalFunc.h"
#include "TTree.h"
#include "RooStats/RooStatsUtils.h"


#include "TMatrixD.h"


ClassImp(RooStats::SPlot); ;

using namespace RooStats;
using namespace std;

////////////////////////////////////////////////////////////////////////////////

SPlot::~SPlot()
{
   if(TestBit(kOwnData) && fSData)
      delete fSData;

}

////////////////////////////////////////////////////////////////////////////////
/// Default constructor

SPlot::SPlot():
  TNamed()
{
  RooArgList Args;

  fSWeightVars = Args;

  fSData = NULL;

}

////////////////////////////////////////////////////////////////////////////////

SPlot::SPlot(const char* name, const char* title):
  TNamed(name, title)
{
  RooArgList Args;

  fSWeightVars = Args;

  fSData = NULL;

}

////////////////////////////////////////////////////////////////////////////////
///Constructor from a RooDataSet
///No sWeighted variables are present

SPlot::SPlot(const char* name, const char* title, const RooDataSet &data):
  TNamed(name, title)
{
  RooArgList Args;

  fSWeightVars = Args;

  fSData = (RooDataSet*) &data;
}

////////////////////////////////////////////////////////////////////////////////
/// Copy Constructor from another SPlot

SPlot::SPlot(const SPlot &other):
  TNamed(other)
{
  RooArgList Args = (RooArgList) other.GetSWeightVars();

  fSWeightVars.addClone(Args);

  fSData = (RooDataSet*) other.GetSDataSet();

}

////////////////////////////////////////////////////////////////////////////////
///Construct a new SPlot instance, calculate sWeights, and include them
///in the RooDataSet held by this instance.
///
/// The constructor automatically calls AddSWeight() to add s weights to the dataset.
/// These can be retrieved later using GetSWeight() or GetSDataSet().
///\param[in] name Name of the instance.
///\param[in] title Title of the instance.
///\param[in] data Dataset to fit to.
///\param[in] pdf PDF to compute s weights for.
///\param[in] yieldsList List of parameters in `pdf` that are yields.
///\param[in] projDeps Don't normalise over these parameters when calculating the sWeights. Will be passed on to AddSWeight().
///\param[in] includeWeights Whether or not to include the weights in `data`. Passed on to AddSWeight().
///\param[in] cloneData Make a clone of the incoming data before adding weights.
///\param[in] newName New name for the data.
///\param[in] argX Additional arguments for the fitting step in AddSWeight().
SPlot::SPlot(const char* name, const char* title, RooDataSet& data, RooAbsPdf* pdf,
        const RooArgList &yieldsList, const RooArgSet &projDeps,
        bool includeWeights, bool cloneData, const char* newName,
        const RooCmdArg& arg5, const RooCmdArg& arg6, const RooCmdArg& arg7, const RooCmdArg& arg8):
  TNamed(name, title)
{
   if(cloneData == 1) {
    fSData = (RooDataSet*) data.Clone(newName);
    SetBit(kOwnData);
   }
  else
    fSData = (RooDataSet*) &data;

  // Add check that yieldsList contains all RooRealVars
  for (const auto arg : yieldsList) {
    if (!dynamic_cast<const RooRealVar*>(arg)) {
      coutE(InputArguments) << "SPlot::SPlot(" << GetName() << ") input argument "
             << arg->GetName() << " is not of type RooRealVar " << endl ;
      throw string(Form("SPlot::SPlot(%s) input argument %s is not of type RooRealVar",GetName(),arg->GetName())) ;
    }
  }

  //Construct a new SPlot class,
  //calculate sWeights, and include them
  //in the RooDataSet of this class.

  this->AddSWeight(pdf, yieldsList, projDeps, includeWeights, arg5, arg6, arg7, arg8);
}

////////////////////////////////////////////////////////////////////////////////
/// Set dataset (if not passed in constructor).
RooDataSet* SPlot::SetSData(RooDataSet* data)
{
  if(data)    {
    fSData = (RooDataSet*) data;
    return fSData;
  }  else
    return NULL;
}

////////////////////////////////////////////////////////////////////////////////
/// Retrieve s-weighted data.
/// It does **not** automatically call AddSWeight(). This needs to be done manually.
RooDataSet* SPlot::GetSDataSet() const
{
  return fSData;
}

////////////////////////////////////////////////////////////////////////////////
/// Retrieve an s weight.
/// \param[in] numEvent Event number to retrieve s weight for.
/// \param[in] sVariable The yield parameter to retrieve the s weight for.
Double_t SPlot::GetSWeight(Int_t numEvent, const char* sVariable) const
{
  if(numEvent > fSData->numEntries() )
    {
      coutE(InputArguments)  << "Invalid Entry Number" << endl;
      return -1;
    }

  if(numEvent < 0)
    {
      coutE(InputArguments)  << "Invalid Entry Number" << endl;
      return -1;
    }

  Double_t totalYield = 0;

  std::string varname(sVariable);
  varname += "_sw";


  if(fSWeightVars.find(sVariable) )
    {
      RooArgSet Row(*fSData->get(numEvent));
      totalYield += Row.getRealValue(sVariable);

      return totalYield;
    }

  if( fSWeightVars.find(varname.c_str())  )
    {

      RooArgSet Row(*fSData->get(numEvent));
      totalYield += Row.getRealValue(varname.c_str() );

      return totalYield;
    }

  else
    coutE(InputArguments) << "InputVariable not in list of sWeighted variables" << endl;

  return -1;
}


////////////////////////////////////////////////////////////////////////////////
/// Sum the SWeights for a particular event.
/// This sum should equal the total weight of that event.
/// This method is intended to be used as a check.

Double_t SPlot::GetSumOfEventSWeight(Int_t numEvent) const
{
  if(numEvent > fSData->numEntries() )
    {
      coutE(InputArguments)  << "Invalid Entry Number" << endl;
      return -1;
    }

  if(numEvent < 0)
    {
      coutE(InputArguments)  << "Invalid Entry Number" << endl;
      return -1;
    }

  Int_t numSWeightVars = this->GetNumSWeightVars();

  Double_t eventSWeight = 0;

  RooArgSet Row(*fSData->get(numEvent));

  for (Int_t i = 0; i < numSWeightVars; i++)
    eventSWeight += Row.getRealValue(fSWeightVars.at(i)->GetName() );

  return  eventSWeight;
}

////////////////////////////////////////////////////////////////////////////////
/// Sum the SWeights for a particular species over all events.
/// This should equal the total (weighted) yield of that species.
/// This method is intended as a check.

Double_t SPlot::GetYieldFromSWeight(const char* sVariable) const
{

  Double_t totalYield = 0;

  std::string varname(sVariable);
  varname += "_sw";


  if(fSWeightVars.find(sVariable) )
    {
      for(Int_t i=0; i < fSData->numEntries(); i++)
   {
     RooArgSet Row(*fSData->get(i));
     totalYield += Row.getRealValue(sVariable);
   }

      return totalYield;
    }

  if( fSWeightVars.find(varname.c_str())  )
    {
      for(Int_t i=0; i < fSData->numEntries(); i++)
   {
     RooArgSet Row(*fSData->get(i));
     totalYield += Row.getRealValue(varname.c_str() );
   }

      return totalYield;
    }

  else
    coutE(InputArguments) << "InputVariable not in list of sWeighted variables" << endl;

  return -1;
}


////////////////////////////////////////////////////////////////////////////////
/// Return a RooArgList containing all paramters that have s weights.

RooArgList SPlot::GetSWeightVars() const
{

  RooArgList Args = fSWeightVars;

  return  Args;

}

////////////////////////////////////////////////////////////////////////////////
/// Return the number of SWeights
/// In other words, return the number of
/// species that we are trying to extract.

Int_t SPlot::GetNumSWeightVars() const
{
  RooArgList Args = fSWeightVars;

  return Args.getSize();
}

////////////////////////////////////////////////////////////////////////////////
/// Method which adds the sWeights to the dataset.
///
/// The SPlot will contain two new variables for each yield parameter:
/// - `L_<varname>` is the value of the pdf for the variable "varname" for each event.
/// - `<varname>_sw` is the value of the sWeight for the variable "varname" for each event.
///
/// Find Parameters in the PDF to be considered fixed when calculating the SWeights
/// and be sure to NOT include the yields in that list.
///
/// After fixing non-yield parameters, this function will start a fit by calling
/// ```
/// pdf->fitTo(*fSData, RooFit::Extended(kTRUE), RooFit::SumW2Error(kTRUE), RooFit::PrintLevel(-1), RooFit::PrintEvalErrors(-1)).
/// ```
/// One can pass additional arguments to `fitTo`, such as `RooFit::Range("fitrange")`, as `arg5`, `arg6`, `arg7`, `arg8`.
///
/// \note A `RooFit::Range` may be necessary to get expected results if you initially fit in a range
/// and/or called `pdf->fixCoefRange("fitrange")` on `pdf`.
/// Pass `arg5`, `arg6`, `arg7`, `arg8` AT YOUR OWN RISK.
///
/// \param[in] pdf PDF to fit to data to compute s weights.
/// \param[in] yieldsTmp Yields to use to compute s weights.
/// \param[in] projDeps These will not be normalized over when calculating the sWeights,
/// and will be considered parameters, not observables.
/// \param[in] includeWeights
/// \param[in] argX Optional additional arguments for the fitting step.
void SPlot::AddSWeight( RooAbsPdf* pdf, const RooArgList &yieldsTmp,
         const RooArgSet &projDeps, bool includeWeights,
         const RooCmdArg& arg5, const RooCmdArg& arg6, const RooCmdArg& arg7, const RooCmdArg& arg8)
{

  RooFit::MsgLevel currentLevel =  RooMsgService::instance().globalKillBelow();

  // Find Parameters in the PDF to be considered fixed when calculating the SWeights
  // and be sure to NOT include the yields in that list
  RooArgList* constParameters = (RooArgList*)pdf->getParameters(fSData) ;
  constParameters->remove(yieldsTmp, kTRUE, kTRUE);


  // Set these parameters constant and store them so they can later
  // be set to not constant
  std::vector<RooRealVar*> constVarHolder;

  for(Int_t i = 0; i < constParameters->getSize(); i++)
    {
      RooRealVar* varTemp = ( dynamic_cast<RooRealVar*>( constParameters->at(i) ) );
      if(varTemp &&  varTemp->isConstant() == 0 )
   {
     varTemp->setConstant();
     constVarHolder.push_back(varTemp);
   }
    }

  // Fit yields to the data with all other variables held constant
  // This is necessary because SPlot assumes the yields minimise -Log(likelihood)

  pdf->fitTo(*fSData, RooFit::Extended(kTRUE), RooFit::SumW2Error(kTRUE), RooFit::PrintLevel(-1), RooFit::PrintEvalErrors(-1), arg5, arg6, arg7, arg8);

  // Hold the value of the fitted yields
  std::vector<double> yieldsHolder;

  for(Int_t i = 0; i < yieldsTmp.getSize(); i++)
    yieldsHolder.push_back( ((RooRealVar*) yieldsTmp.at(i))->getVal());

  Int_t nspec = yieldsTmp.getSize();
  RooArgList yields = *(RooArgList*)yieldsTmp.snapshot(kFALSE);

  if(currentLevel <= RooFit::DEBUG)
    {
      coutI(InputArguments) << "Printing Yields" << endl;
      yields.Print();
    }

  // The list of variables to normalize over when calculating PDF values.

  RooArgSet vars(*fSData->get() );
  vars.remove(projDeps, kTRUE, kTRUE);

  // Attach data set

  // const_cast<RooAbsPdf*>(pdf)->attachDataSet(*fSData);

  pdf->attachDataSet(*fSData);

  // first calculate the pdf values for all species and all events
  std::vector<RooRealVar*> yieldvars ;
  RooArgSet* parameters = pdf->getParameters(fSData) ;

  std::vector<Double_t> yieldvalues ;
  for (Int_t k = 0; k < nspec; ++k)
    {
      RooRealVar* thisyield = dynamic_cast<RooRealVar*>(yields.at(k)) ;
      if (thisyield) {
         RooRealVar* yieldinpdf = dynamic_cast<RooRealVar*>(parameters->find(thisyield->GetName() )) ;

         if (yieldinpdf) {
            coutI(InputArguments)<< "yield in pdf: " << yieldinpdf->GetName() << " " << thisyield->getVal() << endl;

            yieldvars.push_back(yieldinpdf) ;
            yieldvalues.push_back(thisyield->getVal()) ;
         }
      }
    }

  Int_t numevents = fSData->numEntries() ;

  std::vector<std::vector<Double_t> > pdfvalues(numevents,std::vector<Double_t>(nspec,0)) ;


  // set all yield to zero
  for(Int_t m=0; m<nspec; ++m) yieldvars[m]->setVal(0) ;


  // For every event and for every specie,
  // calculate the value of the component pdf for that specie
  // by setting the yield of that specie to 1
  // and all others to 0.  Evaluate the pdf for each event
  // and store the values.

  RooArgSet * pdfvars = pdf->getVariables();

  for (Int_t ievt = 0; ievt <numevents; ievt++)
    {
      //   if (ievt % 100 == 0)
      //  coutP(Eval)  << ".";


      //FIX THIS PART, EVALUATION PROGRESS!!

      RooStats::SetParameters(fSData->get(ievt), pdfvars);

      //   RooArgSet row(*fSData->get(ievt));

      for(Int_t k = 0; k < nspec; ++k)
   {
     //Check that range of yields is at least (0,1), and fix otherwise
     if(yieldvars[k]->getMin() > 0)
       {
         coutW(InputArguments)  << "Minimum Range for " << yieldvars[k]->GetName() << " must be 0.  ";
         coutW(InputArguments)  << "Setting min range to 0" << std::endl;
         yieldvars[k]->setMin(0);
       }

     if(yieldvars[k]->getMax() < 1)
       {
         coutW(InputArguments)  << "Maximum Range for " << yieldvars[k]->GetName() << " must be 1.  ";
         coutW(InputArguments)  << "Setting max range to 1" << std::endl;
         yieldvars[k]->setMax(1);
       }

     // set this yield to 1
     yieldvars[k]->setVal( 1 ) ;
     // evaluate the pdf
     Double_t f_k = pdf->getVal(&vars) ;
     pdfvalues[ievt][k] = f_k ;
     if( !(f_k>1 || f_k<1) )
       coutW(InputArguments) << "Strange pdf value: " << ievt << " " << k << " " << f_k << std::endl ;
     yieldvars[k]->setVal( 0 ) ;
   }
    }
  delete pdfvars;

  // check that the likelihood normalization is fine
  std::vector<Double_t> norm(nspec,0) ;
  for (Int_t ievt = 0; ievt <numevents ; ievt++)
    {
      Double_t dnorm(0) ;
      for(Int_t k=0; k<nspec; ++k) dnorm += yieldvalues[k] * pdfvalues[ievt][k] ;
      for(Int_t j=0; j<nspec; ++j) norm[j] += pdfvalues[ievt][j]/dnorm ;
    }

  coutI(Contents) << "likelihood norms: "  ;

  for(Int_t k=0; k<nspec; ++k)  coutI(Contents) << norm[k] << " " ;
  coutI(Contents) << std::endl ;

  // Make a TMatrixD to hold the covariance matrix.
  TMatrixD covInv(nspec, nspec);
  for (Int_t i = 0; i < nspec; i++) for (Int_t j = 0; j < nspec; j++) covInv(i,j) = 0;

  coutI(Contents) << "Calculating covariance matrix";


  // Calculate the inverse covariance matrix, using weights
  for (Int_t ievt = 0; ievt < numevents; ++ievt)
    {

      fSData->get(ievt) ;

      // Calculate contribution to the inverse of the covariance
      // matrix. See BAD 509 V2 eqn. 15

      // Sum for the denominator
      Double_t dsum(0);
      for(Int_t k = 0; k < nspec; ++k)
   dsum += pdfvalues[ievt][k] * yieldvalues[k] ;

      for(Int_t n=0; n<nspec; ++n)
   for(Int_t j=0; j<nspec; ++j)
     {
       if(includeWeights == kTRUE)
         covInv(n,j) +=  fSData->weight()*pdfvalues[ievt][n]*pdfvalues[ievt][j]/(dsum*dsum) ;
       else
         covInv(n,j) +=  pdfvalues[ievt][n]*pdfvalues[ievt][j]/(dsum*dsum) ;
     }

      //ADDED WEIGHT ABOVE

    }

  // Covariance inverse should now be computed!

  // Invert to get the covariance matrix
  if (covInv.Determinant() <=0)
    {
      coutE(Eval) << "SPlot Error: covariance matrix is singular; I can't invert it!" << std::endl;
      covInv.Print();
      return;
    }

  TMatrixD covMatrix(TMatrixD::kInverted,covInv);

  //check cov normalization
  if(currentLevel <= RooFit::DEBUG)
    {
      coutI(Eval) << "Checking Likelihood normalization:  " << std::endl;
      coutI(Eval) << "Yield of specie  Sum of Row in Matrix   Norm" << std::endl;
      for(Int_t k=0; k<nspec; ++k)
   {
     Double_t covnorm(0) ;
     for(Int_t m=0; m<nspec; ++m) covnorm += covInv[k][m]*yieldvalues[m] ;
     Double_t sumrow(0) ;
     for(Int_t m = 0; m < nspec; ++m) sumrow += covMatrix[k][m] ;
     coutI(Eval)  << yieldvalues[k] << " " << sumrow << " " << covnorm << endl ;
   }
    }

  // calculate for each event the sWeight (BAD 509 V2 eq. 21)
  coutI(Eval) << "Calculating sWeight" << std::endl;
  std::vector<RooRealVar*> sweightvec ;
  std::vector<RooRealVar*> pdfvec ;
  RooArgSet sweightset ;

  // Create and label the variables
  // used to store the SWeights

  fSWeightVars.Clear();

  for(Int_t k=0; k<nspec; ++k)
    {
       std::string wname = std::string(yieldvars[k]->GetName()) + "_sw";
       RooRealVar* var = new RooRealVar(wname.c_str(),wname.c_str(),0) ;
       sweightvec.push_back( var) ;
       sweightset.add(*var) ;
       fSWeightVars.add(*var);

       wname = "L_" + std::string(yieldvars[k]->GetName());
       var = new RooRealVar(wname.c_str(),wname.c_str(),0) ;
       pdfvec.push_back( var) ;
       sweightset.add(*var) ;
    }

  // Create and fill a RooDataSet
  // with the SWeights

  RooDataSet* sWeightData = new RooDataSet("dataset", "dataset with sWeights", sweightset);

  for(Int_t ievt = 0; ievt < numevents; ++ievt)
    {

      fSData->get(ievt) ;

      // sum for denominator
      Double_t dsum(0);
      for(Int_t k = 0; k < nspec; ++k)   dsum +=  pdfvalues[ievt][k] * yieldvalues[k] ;
      // covariance weighted pdf for each specief
      for(Int_t n=0; n<nspec; ++n)
   {
     Double_t nsum(0) ;
     for(Int_t j=0; j<nspec; ++j) nsum += covMatrix(n,j) * pdfvalues[ievt][j] ;


     //Add the sWeights here!!
     //Include weights,
     //ie events weights are absorbed into sWeight


     if(includeWeights == kTRUE) sweightvec[n]->setVal(fSData->weight() * nsum/dsum) ;
     else  sweightvec[n]->setVal( nsum/dsum) ;

     pdfvec[n]->setVal( pdfvalues[ievt][n] ) ;

     if( !(fabs(nsum/dsum)>=0 ) )
       {
         coutE(Contents) << "error: " << nsum/dsum << endl ;
         return;
       }
   }

      sWeightData->add(sweightset) ;
    }


  // Add the SWeights to the original data set

  fSData->merge(sWeightData);

  delete sWeightData;

  //Restore yield values

  for(Int_t i = 0; i < yieldsTmp.getSize(); i++)
    ((RooRealVar*) yieldsTmp.at(i))->setVal(yieldsHolder.at(i));

  //Make any variables that were forced to constant no longer constant

  for(Int_t i=0; i < (Int_t) constVarHolder.size(); i++)
    constVarHolder.at(i)->setConstant(kFALSE);

  return;

}
