// @(#)root/roostats:$Id$
// Author: Sven Kreiss    January 2012
// Author: Kyle Cranmer, Lorenzo Moneta, Gregory Schott, Wouter Verkerke
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class RooStats::ToyMCImportanceSampler
    \ingroup Roostats

ToyMCImportanceSampler is an extension of the ToyMCSampler for Importance Sampling.

Implementation based on a work by   Cranmer, Kreiss, Read (in Preparation)
*/

#include "RooStats/ToyMCImportanceSampler.h"

#include "RooMsgService.h"

#include "RooCategory.h"
#include "TMath.h"

using namespace RooFit;
using namespace std;


ClassImp(RooStats::ToyMCImportanceSampler);

namespace RooStats {

////////////////////////////////////////////////////////////////////////////////

ToyMCImportanceSampler::~ToyMCImportanceSampler() {
   for( unsigned int i=0; i < fImportanceSnapshots.size(); i++ ) if(fImportanceSnapshots[i]) delete fImportanceSnapshots[i];
   for( unsigned int i=0; i < fNullSnapshots.size(); i++ ) if(fNullSnapshots[i]) delete fNullSnapshots[i];
}

////////////////////////////////////////////////////////////////////////////////

void ToyMCImportanceSampler::ClearCache(void) {
   ToyMCSampler::ClearCache();

   for( unsigned int i=0; i < fImpNLLs.size(); i++ ) if(fImpNLLs[i]) { delete fImpNLLs[i]; fImpNLLs[i] = NULL; }
   for( unsigned int i=0; i < fNullNLLs.size(); i++ ) if(fNullNLLs[i]) { delete fNullNLLs[i]; fNullNLLs[i] = NULL; }
}

////////////////////////////////////////////////////////////////////////////////

RooDataSet* ToyMCImportanceSampler::GetSamplingDistributionsSingleWorker(RooArgSet& paramPoint) {
   if( fNToys == 0 ) return NULL;

   // remember original #toys, but overwrite it temporarily with the #toys per distribution
   Int_t allToys = fNToys;

   // to keep track of which dataset entry comes form which density, define a roocategory as a label
   RooCategory densityLabel( "densityLabel", "densityLabel" );
   densityLabel.defineType( "null", -1 );
   for( unsigned int i=0; i < fImportanceDensities.size(); i++ )
      densityLabel.defineType( TString::Format( "impDens_%d", i ), i );


   RooDataSet* fullResult = NULL;

   // generate null (negative i) and imp densities (0 and positive i)
   for( int i = -1; i < (int)fImportanceDensities.size(); i++ ) {
      if( i < 0 ) {
         // generate null toys
         oocoutP(nullptr,Generation) << endl << endl << "   GENERATING FROM NULL DENSITY " << endl << endl;
         SetDensityToGenerateFromByIndex( 0, true ); // true = generate from null
      }else{
         oocoutP(nullptr,Generation) << endl << endl << "   GENERATING IMP DENS/SNAP "<<i+1<<"  OUT OF "<<fImportanceDensities.size()<<endl<<endl;
         SetDensityToGenerateFromByIndex( i, false ); // false = generate not from null
      }

      RooRealVar reweight( "reweight", "reweight", 1.0 );
      // apply strategy for how to distribute the #toys between the distributions
      if( fToysStrategy == EQUALTOYSPERDENSITY ) {
         // assuming alltoys = one null + N imp densities. And round up.
         fNToys = TMath::CeilNint(  double(allToys)/(fImportanceDensities.size()+1)  );
      }else if(fToysStrategy == EXPONENTIALTOYDISTRIBUTION ) {
         // for N densities, split the toys into (2^(N+1))-1 parts, and assign 2^0 parts to the first
         // density (which is the null), 2^1 to the second (first imp dens), etc, up to 2^N
         fNToys = TMath::CeilNint(  double(allToys) * pow( double(2) , i+1 )  /  (pow( double(2), int(fImportanceDensities.size()+1) )-1)  );

         int largestNToys = TMath::CeilNint(  allToys * pow( double(2), int(fImportanceDensities.size()) )  /  (pow( double(2), int(fImportanceDensities.size()+1) )-1)  );
         reweight.setVal( ((double)largestNToys) / fNToys );
      }

      ooccoutI(nullptr,InputArguments) << "Generating " << fNToys << " toys for this density." << endl;
      ooccoutI(nullptr,InputArguments) << "Reweight is " << reweight.getVal() << endl;


      RooDataSet* result = ToyMCSampler::GetSamplingDistributionsSingleWorker( paramPoint );

      if (result->get()->getSize() > Int_t(fTestStatistics.size())) {
         // add label
         densityLabel.setIndex( i );
         result->addColumn( densityLabel );
         result->addColumn( reweight );
      }

      if( !fullResult ) {
         RooArgSet columns( *result->get() );
         RooRealVar weightVar ( "weight", "weight", 1.0 );
         columns.add( weightVar );
//       cout << endl << endl << "Reweighted data columns: " << endl;
//       columns.Print("v");
//       cout << endl;
         fullResult = new RooDataSet( result->GetName(), result->GetTitle(), columns, "weight" );
      }

      for( int j=0; j < result->numEntries(); j++ ) {
//          cout << "entry: " << j << endl;
//          result->get(j)->Print();
//          cout << "weight: " << result->weight() << endl;
//          cout << "reweight: " << reweight.getVal() << endl;
         const RooArgSet* row = result->get(j);
         fullResult->add( *row, result->weight()*reweight.getVal() );
      }
      delete result;
   }

   // restore #toys
   fNToys = allToys;

   return fullResult;
}

////////////////////////////////////////////////////////////////////////////////

RooAbsData* ToyMCImportanceSampler::GenerateToyData(
   RooArgSet& paramPoint,
   double& weight
) const {
   if( fNullDensities.size() > 1 ) {
      ooccoutI(nullptr,InputArguments) << "Null Densities:" << endl;
      for( unsigned int i=0; i < fNullDensities.size(); i++) {
         ooccoutI(nullptr,InputArguments) << "  null density["<<i<<"]: " << fNullDensities[i] << " \t null snapshot["<<i<<"]: " << fNullSnapshots[i] << endl;
      }
      ooccoutE(nullptr,InputArguments) << "Cannot use multiple null densities and only ask for one weight." << endl;
      return NULL;
   }

   if( fNullDensities.size() == 0  &&  fPdf ) {
      ooccoutI(nullptr,InputArguments) << "No explicit null densities specified. Going to add one based on the given paramPoint and the global fPdf. ... but cannot do that inside const function." << endl;
      //AddNullDensity( fPdf, &paramPoint );
   }

   // do not do anything if the given parameter point if fNullSnapshots[0]
   // ... which is the most common case
   if( fNullSnapshots[0] != &paramPoint ) {
      ooccoutD(nullptr,InputArguments) << "Using given parameter point. Replaces snapshot for the only null currently defined." << endl;
      if(fNullSnapshots[0]) delete fNullSnapshots[0];
      fNullSnapshots.clear();
      fNullSnapshots.push_back( (RooArgSet*)paramPoint.snapshot() );
   }

   vector<double> weights;
   weights.push_back( weight );

   vector<double> impNLLs;
   for( unsigned int i=0; i < fImportanceDensities.size(); i++ ) impNLLs.push_back( 0.0 );
   vector<double> nullNLLs;
   for( unsigned int i=0; i < fNullDensities.size(); i++ ) nullNLLs.push_back( 0.0 );

   RooAbsData *d = GenerateToyData( weights, impNLLs, nullNLLs );
   weight = weights[0];
   return d;
}

////////////////////////////////////////////////////////////////////////////////

RooAbsData* ToyMCImportanceSampler::GenerateToyData(
   RooArgSet& paramPoint,
   double& weight,
   vector<double>& impNLLs,
   double& nullNLL
) const {
   if( fNullDensities.size() > 1 ) {
      ooccoutI(nullptr,InputArguments) << "Null Densities:" << endl;
      for( unsigned int i=0; i < fNullDensities.size(); i++) {
         ooccoutI(nullptr,InputArguments) << "  null density["<<i<<"]: " << fNullDensities[i] << " \t null snapshot["<<i<<"]: " << fNullSnapshots[i] << endl;
      }
      ooccoutE(nullptr,InputArguments) << "Cannot use multiple null densities and only ask for one weight and NLL." << endl;
      return NULL;
   }

   if( fNullDensities.size() == 0  &&  fPdf ) {
      ooccoutI(nullptr,InputArguments) << "No explicit null densities specified. Going to add one based on the given paramPoint and the global fPdf. ... but cannot do that inside const function." << endl;
      //AddNullDensity( fPdf, &paramPoint );
   }

   ooccoutI(nullptr,InputArguments) << "Using given parameter point. Overwrites snapshot for the only null currently defined." << endl;
   if(fNullSnapshots[0]) delete fNullSnapshots[0];
   fNullSnapshots.clear();
   fNullSnapshots.push_back( (const RooArgSet*)paramPoint.snapshot() );

   vector<double> weights;
   weights.push_back( weight );

   vector<double> nullNLLs;
   nullNLLs.push_back( nullNLL );

   RooAbsData *d = GenerateToyData( weights, impNLLs, nullNLLs );
   weight = weights[0];
   nullNLL = nullNLLs[0];
   return d;
}

////////////////////////////////////////////////////////////////////////////////

RooAbsData* ToyMCImportanceSampler::GenerateToyData(
   vector<double>& weights
) const {
   if( fNullDensities.size() != weights.size() ) {
      ooccoutI(nullptr,InputArguments) << "weights.size() != nullDesnities.size(). You need to provide a vector with the correct size." << endl;
      //AddNullDensity( fPdf, &paramPoint );
   }

   vector<double> impNLLs;
   for( unsigned int i=0; i < fImportanceDensities.size(); i++ ) impNLLs.push_back( 0.0 );
   vector<double> nullNLLs;
   for( unsigned int i=0; i < fNullDensities.size(); i++ ) nullNLLs.push_back( 0.0 );

   RooAbsData *d = GenerateToyData( weights, impNLLs, nullNLLs );
   return d;
}

////////////////////////////////////////////////////////////////////////////////
/// This method generates a toy data set for importance sampling for the given parameter point taking
/// global observables into account.
/// The values of the generated global observables remain in the pdf's variables.
/// They have to have those values for the subsequent evaluation of the
/// test statistics.

RooAbsData* ToyMCImportanceSampler::GenerateToyData(
   vector<double>& weights,
   vector<double>& impNLLVals,
   vector<double>& nullNLLVals
) const {


   ooccoutD(nullptr,InputArguments) << endl;
   ooccoutD(nullptr,InputArguments) << "GenerateToyDataImportanceSampling" << endl;

   if(!fObservables) {
      ooccoutE(nullptr,InputArguments) << "Observables not set." << endl;
      return NULL;
   }

   if( fNullDensities.size() == 0 ) {
      oocoutE(nullptr,InputArguments) << "ToyMCImportanceSampler: Need to specify the null density explicitly." << endl;
      return NULL;
   }

   // catch the case when NLLs are not created (e.g. when ToyMCSampler was streamed for Proof)
   if( fNullNLLs.size() == 0  &&  fNullDensities.size() > 0 ) {
      for( unsigned int i = 0; i < fNullDensities.size(); i++ ) fNullNLLs.push_back( NULL );
   }
   if( fImpNLLs.size() == 0  &&  fImportanceDensities.size() > 0 ) {
      for( unsigned int i = 0; i < fImportanceDensities.size(); i++ ) fImpNLLs.push_back( NULL );
   }

   if( fNullDensities.size() != fNullNLLs.size() ) {
      oocoutE(nullptr,InputArguments) << "ToyMCImportanceSampler: Something wrong. NullNLLs must be of same size as null densities." << endl;
      return NULL;
   }

   if( (!fGenerateFromNull  &&  fIndexGenDensity >= fImportanceDensities.size()) ||
       (fGenerateFromNull   &&  fIndexGenDensity >= fNullDensities.size())
   ) {
      oocoutE(nullptr,InputArguments) << "ToyMCImportanceSampler: no importance density given or index out of range." << endl;
      return NULL;
   }


   // paramPoint used to be given as parameter
   // situation is clear when there is only one null.
   // WHAT TO DO FOR MANY NULL DENSITIES?
   RooArgSet paramPoint( *fNullSnapshots[0] );
   //cout << "paramPoint: " << endl;
   //paramPoint.Print("v");


   // assign input paramPoint
   RooArgSet* allVars = fPdf->getVariables();
   allVars->assign(paramPoint);


   // create nuisance parameter points
   if(!fNuisanceParametersSampler && fPriorNuisance && fNuisancePars)
      fNuisanceParametersSampler = new NuisanceParametersSampler(fPriorNuisance, fNuisancePars, fNToys, fExpectedNuisancePar);

   // generate global observables
   RooArgSet observables(*fObservables);
   if(fGlobalObservables  &&  fGlobalObservables->getSize()) {
      observables.remove(*fGlobalObservables);
      // WHAT TODO FOR MANY NULL DENSITIES?
      GenerateGlobalObservables(*fNullDensities[0]);
   }

   // save values to restore later.
   // but this must remain after(!) generating global observables
   if( !fGenerateFromNull ) {
      RooArgSet* allVarsImpDens = fImportanceDensities[fIndexGenDensity]->getVariables();
      allVars->add(*allVarsImpDens);
      delete allVarsImpDens;
   }
   const RooArgSet* saveVars = (const RooArgSet*)allVars->snapshot();

   double globalWeight = 1.0;
   if(fNuisanceParametersSampler) { // use nuisance parameters?
      // Construct a set of nuisance parameters that has the parameters
      // in the input paramPoint removed. Therefore, no parameter in
      // paramPoint is randomized.
      // Therefore when a parameter is given (should be held fixed),
      // but is also in the list of nuisance parameters, the parameter
      // will be held fixed. This is useful for debugging to hold single
      // parameters fixed although under "normal" circumstances it is
      // randomized.
      RooArgSet allVarsMinusParamPoint(*allVars);
      allVarsMinusParamPoint.remove(paramPoint, false, true); // match by name

      // get nuisance parameter point and weight
      fNuisanceParametersSampler->NextPoint(allVarsMinusParamPoint, globalWeight);
   }
   // populate input weights vector with this globalWeight
   for( unsigned int i=0; i < weights.size(); i++ ) weights[i] = globalWeight;

   RooAbsData* data = NULL;
   if( fGenerateFromNull ) {
      //cout << "gen from null" << endl;
      allVars->assign(*fNullSnapshots[fIndexGenDensity]);
      data = Generate(*fNullDensities[fIndexGenDensity], observables);
   }else{
      // need to be careful here not to overwrite the current state of the
      // nuisance parameters, ie they must not be part of the snapshot
      //cout << "gen from imp" << endl;
      if(fImportanceSnapshots[fIndexGenDensity]) {
        allVars->assign(*fImportanceSnapshots[fIndexGenDensity]);
      }
      data = Generate(*fImportanceDensities[fIndexGenDensity], observables);
   }
   //cout << "data generated: " << data << endl;

   if (!data) {
      oocoutE(nullptr,InputArguments) << "ToyMCImportanceSampler: error generating data" << endl;
      return NULL;
   }



   // Importance Sampling: adjust weight
   // Sources: Alex Read, presentation by Michael Woodroofe

   ooccoutD(nullptr,InputArguments) << "About to create/calculate all nullNLLs." << endl;
   for( unsigned int i=0; i < fNullDensities.size(); i++ ) {
      //oocoutI(nullptr,InputArguments) << "Setting variables to nullSnapshot["<<i<<"]"<<endl;
      //fNullSnapshots[i]->Print("v");

      allVars->assign(*fNullSnapshots[i]);
      if( !fNullNLLs[i] ) {
         RooArgSet* allParams = fNullDensities[i]->getParameters(*data);
         fNullNLLs[i] = fNullDensities[i]->createNLL(*data, RooFit::CloneData(false), RooFit::Constrain(*allParams),
                                                     RooFit::ConditionalObservables(fConditionalObs));
         delete allParams;
      }else{
         fNullNLLs[i]->setData( *data, false );
      }
      nullNLLVals[i] = fNullNLLs[i]->getVal();
      // FOR DEBuGGING!!!!!!!!!!!!!!!!!
      if( !fReuseNLL ) { delete fNullNLLs[i]; fNullNLLs[i] = NULL; }
   }


   // for each null: find minNLLVal of null and all imp densities
   ooccoutD(nullptr,InputArguments) << "About to find the minimum NLLs." << endl;
   vector<double> minNLLVals;
   for( unsigned int i=0; i < nullNLLVals.size(); i++ ) minNLLVals.push_back( nullNLLVals[i] );

   for( unsigned int i=0; i < fImportanceDensities.size(); i++ ) {
      //oocoutI(nullptr,InputArguments) << "Setting variables to impSnapshot["<<i<<"]"<<endl;
      //fImportanceSnapshots[i]->Print("v");

      if( fImportanceSnapshots[i] ) {
        allVars->assign(*fImportanceSnapshots[i]);
      }
      if( !fImpNLLs[i] ) {
         RooArgSet* allParams = fImportanceDensities[i]->getParameters(*data);
         fImpNLLs[i] = fImportanceDensities[i]->createNLL(*data, RooFit::CloneData(false), RooFit::Constrain(*allParams),
                                                          RooFit::ConditionalObservables(fConditionalObs));
         delete allParams;
      }else{
         fImpNLLs[i]->setData( *data, false );
      }
      impNLLVals[i] = fImpNLLs[i]->getVal();
      // FOR DEBuGGING!!!!!!!!!!!!!!!!!
      if( !fReuseNLL ) { delete fImpNLLs[i]; fImpNLLs[i] = NULL; }

      for( unsigned int j=0; j < nullNLLVals.size(); j++ ) {
         if( impNLLVals[i] < minNLLVals[j] ) minNLLVals[j] = impNLLVals[i];
         ooccoutD(nullptr,InputArguments) << "minNLLVals["<<j<<"]: " << minNLLVals[j] << "  nullNLLVals["<<j<<"]: " << nullNLLVals[j] << "    impNLLVals["<<i<<"]: " << impNLLVals[i] << endl;
      }
   }

   // veto toys: this is a sort of "overlap removal" of the various distributions
   // if not vetoed: apply weight
   ooccoutD(nullptr,InputArguments) << "About to apply vetos and calculate weights." << endl;
   for( unsigned int j=0; j < nullNLLVals.size(); j++ ) {
      if     ( fApplyVeto  &&  fGenerateFromNull  &&  minNLLVals[j] != nullNLLVals[j] ) weights[j] = 0.0;
      else if( fApplyVeto  &&  !fGenerateFromNull  &&  minNLLVals[j] != impNLLVals[fIndexGenDensity] ) weights[j] = 0.0;
      else if( !fGenerateFromNull ) {
         // apply (for fImportanceGenNorm, the weight is one, so nothing needs to be done here)

         // L(pdf) / L(imp)  =  exp( NLL(imp) - NLL(pdf) )
         weights[j] *= exp(minNLLVals[j] - nullNLLVals[j]);
      }

      ooccoutD(nullptr,InputArguments) << "weights["<<j<<"]: " << weights[j] << endl;
   }



   allVars->assign(*saveVars);
   delete allVars;
   delete saveVars;

   return data;
}

////////////////////////////////////////////////////////////////////////////////
/// poi has to be fitted beforehand. This function expects this to be the muhat value.

int ToyMCImportanceSampler::CreateImpDensitiesForOnePOIAdaptively( RooAbsPdf& pdf, const RooArgSet& allPOI, RooRealVar& poi, double nStdDevOverlap, double poiValueForBackground ) {
   // these might not necessarily be the same thing.
   double impMaxMu = poi.getVal();

   // this includes the null
   int n = 1;

   // check whether error is trustworthy
   if( poi.getError() > 0.01  &&  poi.getError() < 5.0 ) {
      n = TMath::CeilNint( poi.getVal() / (2.*nStdDevOverlap*poi.getError()) ); // round up
      oocoutI(nullptr,InputArguments) << "Using fitFavoredMu and error to set the number of imp points" << endl;
      oocoutI(nullptr,InputArguments) << "muhat: " << poi.getVal() << "    optimize for distance: " << 2.*nStdDevOverlap*poi.getError() << endl;
      oocoutI(nullptr,InputArguments) << "n = " << n << endl;
      oocoutI(nullptr,InputArguments) << "This results in a distance of: " << impMaxMu / n << endl;
   }

   // exclude the null, just return the number of importance snapshots
   return CreateNImpDensitiesForOnePOI( pdf, allPOI, poi, n-1, poiValueForBackground);
}

////////////////////////////////////////////////////////////////////////////////
/// n is the number of importance densities

int ToyMCImportanceSampler::CreateNImpDensitiesForOnePOI( RooAbsPdf& pdf, const RooArgSet& allPOI, RooRealVar& poi, int n, double poiValueForBackground ) {

   // these might not necessarily be the same thing.
   double impMaxMu = poi.getVal();

   // create imp snapshots
   if( impMaxMu > poiValueForBackground  &&  n > 0 ) {
      for( int i=1; i <= n; i++ ) {
         poi.setVal( poiValueForBackground + (double)i/(n)*(impMaxMu - poiValueForBackground) );
         oocoutI(nullptr,InputArguments) << endl << "create point with poi: " << endl;
         poi.Print();

         // impSnaps without first snapshot because that is null hypothesis

         AddImportanceDensity( &pdf, &allPOI );
      }
   }

   return n;
}

} // end namespace RooStats
