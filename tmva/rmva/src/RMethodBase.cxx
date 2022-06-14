// @(#)root/tmva/rmva $Id$
// Author: Omar Zapata,Lorenzo Moneta, Sergei Gleyzer 2015


/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : RMethodBase                                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Virtual base class for all MVA method based on ROOTR                      *
 *                                                                                *
 **********************************************************************************/

#include<TMVA/RMethodBase.h>
#include <TMVA/DataSetInfo.h>
#include<TApplication.h>
using namespace TMVA;

ClassImp(RMethodBase);

//_______________________________________________________________________
RMethodBase::RMethodBase(const TString &jobName,
                         Types::EMVA methodType,
                         const TString &methodTitle,
                         DataSetInfo &dsi,
                         const TString &theOption , ROOT::R::TRInterface &_r): MethodBase(jobName, methodType, methodTitle, dsi, theOption),
   r(_r)
{
   LoadData();
}

//_______________________________________________________________________
RMethodBase::RMethodBase(Types::EMVA methodType,
                         DataSetInfo &dsi,
                         const TString &weightFile,ROOT::R::TRInterface &_r): MethodBase(methodType, dsi, weightFile),
   r(_r)
{
   LoadData();
}

//_______________________________________________________________________
void RMethodBase::LoadData()
{
   ///////////////////////////
   //Loading Training Data  //
   ///////////////////////////
   const UInt_t nvar = DataInfo().GetNVariables();

   const UInt_t ntrains = Data()->GetNTrainingEvents();

   //array of columns for every var to create a dataframe for training
   std::vector<std::vector<Float_t> > fArrayTrain(nvar);
//    Data()->SetCurrentEvent(1);
//    Data()->SetCurrentType(Types::ETreeType::kTraining);

   fWeightTrain.ResizeTo(ntrains);
   for (UInt_t j = 0; j < ntrains; j++) {
      const Event *ev = Data()->GetEvent(j, Types::ETreeType::kTraining);
//        const Event *ev=Data()->GetEvent( j );
      //creating array with class type(signal or background) for factor required
      if (ev->GetClass() == Types::kSignal) fFactorTrain.push_back("signal");
      else fFactorTrain.push_back("background");

      fWeightTrain[j] = ev->GetWeight();

      //filling vector of columns for training
      for (UInt_t i = 0; i < nvar; i++) {
         fArrayTrain[i].push_back(ev->GetValue(i));
      }

   }
   for (UInt_t i = 0; i < nvar; i++) {
      fDfTrain[DataInfo().GetListOfVariables()[i].Data()] = fArrayTrain[i];
   }
   ////////////////////////
   //Loading Test  Data  //
   ////////////////////////

   const UInt_t ntests = Data()->GetNTestEvents();
   const UInt_t nspectators = DataInfo().GetNSpectators(kTRUE);

   //array of columns for every var to create a dataframe for testing
   std::vector<std::vector<Float_t> > fArrayTest(nvar);
   //array of columns for every spectator to create a dataframe for testing
   std::vector<std::vector<Float_t> > fArraySpectators(nvar);
   fWeightTest.ResizeTo(ntests);
//    Data()->SetCurrentType(Types::ETreeType::kTesting);
   for (UInt_t j = 0; j < ntests; j++) {
      const Event *ev = Data()->GetEvent(j, Types::ETreeType::kTesting);
//        const Event *ev=Data()->GetEvent(j);
      //creating array with class type(signal or background) for factor required
      if (ev->GetClass() == Types::kSignal) fFactorTest.push_back("signal");
      else fFactorTest.push_back("background");

      fWeightTest[j] = ev->GetWeight();

      for (UInt_t i = 0; i < nvar; i++) {
         fArrayTest[i].push_back(ev->GetValue(i));
      }
      for (UInt_t i = 0; i < nspectators; i++) {
         fArraySpectators[i].push_back(ev->GetSpectator(i));
      }
   }
   for (UInt_t i = 0; i < nvar; i++) {
      fDfTest[DataInfo().GetListOfVariables()[i].Data()] = fArrayTest[i];
   }
   for (UInt_t i = 0; i < nspectators; i++) {
      fDfSpectators[DataInfo().GetSpectatorInfo(i).GetLabel().Data()] = fArraySpectators[i];
   }

}
