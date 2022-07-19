/**
 *  \ingroup HistFactory
 */

// A set of utils for navegating HistFactory models
#include <stdexcept>
#include <typeinfo>

#include "RooStats/HistFactory/ParamHistFunc.h"
#include "TIterator.h"
#include "RooAbsArg.h"
#include "RooAbsPdf.h"
#include "RooArgSet.h"
#include "RooArgList.h"
#include "RooSimultaneous.h"
#include "RooCategory.h"
#include "RooRealVar.h"
#include "RooProdPdf.h"
#include "TH1.h"

#include "RooStats/HistFactory/HistFactoryModelUtils.h"

namespace RooStats{
namespace HistFactory{


  std::string channelNameFromPdf( RooAbsPdf* channelPdf ) {
    std::string channelPdfName = channelPdf->GetName();
    std::string ChannelName = channelPdfName.substr(6, channelPdfName.size() );
    return ChannelName;
  }

  RooAbsPdf* getSumPdfFromChannel( RooAbsPdf* sim_channel ) {

    bool verbose=false;

    if(verbose) std::cout << "Getting the RooRealSumPdf for the channel: "
           << sim_channel->GetName() << std::endl;

    std::string channelPdfName = sim_channel->GetName();
    std::string ChannelName = channelPdfName.substr(6, channelPdfName.size() );

    // Now, get the RooRealSumPdf
    // ie the channel WITHOUT constraints
    std::string realSumPdfName = ChannelName + "_model";

    RooAbsPdf* sum_pdf = nullptr;
    bool FoundSumPdf=false;
    for (auto *sum_pdf_arg : *sim_channel->getComponents()) {
        std::string NodeClassName = sum_pdf_arg->ClassName();
        if( NodeClassName == std::string("RooRealSumPdf") ) {
            FoundSumPdf=true;
           sum_pdf = static_cast<RooAbsPdf*>(sum_pdf_arg);
        break;
      }
    }
    if( ! FoundSumPdf ) {
      if(verbose) {
   std::cout << "Failed to find RooRealSumPdf for channel: " << sim_channel->GetName() << std::endl;
   sim_channel->getComponents()->Print("V");
      }
      sum_pdf=nullptr;
      //throw std::runtime_error("Failed to find RooRealSumPdf for channel");
    }
    else {
      if(verbose) std::cout << "Found RooRealSumPdf: " << sum_pdf->GetName() << std::endl;
    }

    return sum_pdf;

  }


 void FactorizeHistFactoryPdf(const RooArgSet &observables, RooAbsPdf &pdf, RooArgList &obsTerms, RooArgList &constraints) {
   // utility function to factorize constraint terms from a pdf
   // (from G. Petrucciani)
   const std::type_info & id = typeid(pdf);
   if (id == typeid(RooProdPdf)) {
      RooProdPdf *prod = dynamic_cast<RooProdPdf *>(&pdf);
      RooArgList list(prod->pdfList());
      for (int i = 0, n = list.getSize(); i < n; ++i) {
         RooAbsPdf *pdfi = (RooAbsPdf *) list.at(i);
            FactorizeHistFactoryPdf(observables, *pdfi, obsTerms, constraints);
         }
      } else if (id == typeid(RooSimultaneous)) {    //|| id == typeid(RooSimultaneousOpt)) {
         RooSimultaneous *sim  = dynamic_cast<RooSimultaneous *>(&pdf);
         RooAbsCategoryLValue *cat = (RooAbsCategoryLValue *) sim->indexCat().Clone();
         for (int ic = 0, nc = cat->numBins((const char *)0); ic < nc; ++ic) {
            cat->setBin(ic);
            FactorizeHistFactoryPdf(observables, *sim->getPdf(cat->getCurrentLabel()), obsTerms, constraints);
         }
         delete cat;
      } else if (pdf.dependsOn(observables)) {
         if (!obsTerms.contains(pdf)) obsTerms.add(pdf);
      } else {
         if (!constraints.contains(pdf)) constraints.add(pdf);
      }
   }


  bool getStatUncertaintyFromChannel( RooAbsPdf* channel, ParamHistFunc*& paramfunc, RooArgList* gammaList ) {

    bool verbose=false;

    // Find the servers of this channel
    bool FoundParamHistFunc=false;
    for( auto *paramfunc_arg : *channel->getComponents() ) {
      std::string NodeName = paramfunc_arg->GetName();
      std::string NodeClassName = paramfunc_arg->ClassName();
      if( NodeClassName != std::string("ParamHistFunc") ) continue;
      if( NodeName.find("mc_stat_") != std::string::npos ) {
        FoundParamHistFunc=true;
        paramfunc = static_cast<ParamHistFunc*>(paramfunc_arg);
        break;
      }
    }
    if( ! FoundParamHistFunc || !paramfunc ) {
      if(verbose) std::cout << "Failed to find ParamHistFunc for channel: " << channel->GetName() << std::endl;
      return false;
    }

    // Now, get the set of gamma's
    gammaList = (RooArgList*) &( paramfunc->paramList());
    if(verbose) gammaList->Print("V");

    return true;

  }


  void getDataValuesForObservables( std::map< std::string, std::vector<double> >& ChannelBinDataMap,
                RooAbsData* data, RooAbsPdf* pdf ) {

    bool verbose=false;

    RooSimultaneous* simPdf = (RooSimultaneous*) pdf;

    // get category label
    RooCategory* cat = nullptr;
    for (auto* temp : *data->get()) {
      if( strcmp(temp->ClassName(),"RooCategory")==0){
          cat = static_cast<RooCategory*>(temp);
        break;
      }
    }
    if(verbose) {
      if(!cat) std::cout <<"didn't find category"<< std::endl;
      else std::cout <<"found category"<< std::endl;
    }

    if (!cat) {
       std::cerr <<"Category not found"<< std::endl;
       return;
    }

    // split dataset
    TList* dataByCategory = data->split(*cat);
    if(verbose) dataByCategory->Print();
    // note :
    // RooAbsData* dataForChan = (RooAbsData*) dataByCategory->FindObject("");

    // loop over channels
    RooCategory* channelCat = (RooCategory*) (&simPdf->indexCat());
    for (const auto& nameIdx : *channelCat) {

      // Get pdf associated with state from simpdf
      RooAbsPdf* pdftmp = simPdf->getPdf(nameIdx.first.c_str());

      std::string ChannelName = pdftmp->GetName(); //tt->GetName();
      if(verbose) std::cout << "Getting data for channel: " << ChannelName << std::endl;
      ChannelBinDataMap[ ChannelName ] = std::vector<double>();

      RooAbsData* dataForChan = (RooAbsData*) dataByCategory->FindObject(nameIdx.first.c_str());
      if(verbose) dataForChan->Print();

      // Generate observables defined by the pdf associated with this state
      RooArgSet* obstmp = pdftmp->getObservables(*dataForChan->get()) ;
      RooRealVar* obs = ((RooRealVar*)obstmp->first());
      if(verbose) obs->Print();

      //double expected = pdftmp->expectedEvents(*obstmp);

      // set value to desired value (this is just an example)
      // double obsVal = obs->getVal();
      // set obs to desired value of observable
      // obs->setVal( obsVal );
      //double fracAtObsValue = pdftmp->getVal(*obstmp);

      // get num events expected in bin for obsVal
      // double nu = expected * fracAtObsValue;

      // multidimensional way to get n
      // credit goes to P. Hamilton
      for(int i=0; i<dataForChan->numEntries(); i++)
      {
        const RooArgSet* tmpargs=dataForChan->get(i);
        if(verbose) tmpargs->Print();
        const double n = dataForChan->weight();
        if(verbose) std::cout << "n" << i << " = " << n << std::endl;
        ChannelBinDataMap[ ChannelName ].push_back( n );
      }
      
    } // End Loop Over Categories

    dataByCategory->Delete();
    delete dataByCategory;
  }


  int getStatUncertaintyConstraintTerm( RooArgList* constraints, RooRealVar* gamma_stat,
               RooAbsReal*& pois_nom, RooRealVar*& tau ) {
    // Given a set of constraint terms,
    // find the poisson constraint for the
    // given gamma and return the mean
    // as well as the 'tau' parameter

    bool verbose=false;

    // To get the constraint term, loop over all constraint terms
    // and look for the gamma_stat name as well as '_constraint'

    bool FoundConstraintTerm=false;
    RooAbsPdf* constraintTerm=nullptr;
    for (auto *term_constr : *constraints) {
      std::string TermName = term_constr->GetName();
      if( term_constr->dependsOn( *gamma_stat) ) {
        if( TermName.find("_constraint")!=std::string::npos ) {
          FoundConstraintTerm=true;
          constraintTerm = static_cast<RooAbsPdf*>(term_constr);
          break;
        }
      }
    }
    if( FoundConstraintTerm==false ) {
      std::cout << "Error: Couldn't find constraint term for parameter: " << gamma_stat->GetName()
      << " among constraints: " << constraints->GetName() <<  std::endl;
      constraints->Print("V");
      throw std::runtime_error("Failed to find Gamma ConstraintTerm");
      return -1;
    }

    /*
    RooAbsPdf* constraintTerm = (RooAbsPdf*) constraints->find( constraintTermName.c_str() );
    if( constraintTerm == nullptr ) {
      std::cout << "Error: Couldn't find constraint term: " << constraintTermName
      << " for parameter: " << gamma_stat->GetName()
      << std::endl;
      throw std::runtime_error("Failed to find Gamma ConstraintTerm");
      return -1;
    }
    */

    // Find the "data" of the poisson term
    // This is the nominal value
    bool FoundNomMean=false;
    TIter iter_pois = constraintTerm->serverIterator(); //constraint_args
    RooAbsArg* term_pois ;
    while((term_pois=(RooAbsArg*)iter_pois.Next())) {
      std::string serverName = term_pois->GetName();
      //std::cout << "Checking Server: " << serverName << std::endl;
      if( serverName.find("nom_")!=std::string::npos ) {
   FoundNomMean = true;
   pois_nom = (RooRealVar*) term_pois;
      }
    }
    if( !FoundNomMean || !pois_nom ) {
      std::cout << "Error: Did not find Nominal Pois Mean parameter in gamma constraint term PoissonMean: "
      << constraintTerm->GetName() << std::endl;
      throw std::runtime_error("Failed to find Nom Pois Mean");
    }
    else {
      if(verbose) std::cout << "Found Poisson 'data' term: " << pois_nom->GetName() << std::endl;
    }

    // Taking the constraint term (a Poisson), find
    // the "mean" which is the product: gamma*tau
    // Then, from that mean, find tau
    TIter iter_constr = constraintTerm->serverIterator(); //constraint_args
    RooAbsArg* pois_mean_arg=nullptr;
    bool FoundPoissonMean = false;
    while(( pois_mean_arg = (RooAbsArg*) iter_constr.Next() )) {
      std::string serverName = pois_mean_arg->GetName();
      if( pois_mean_arg->dependsOn( *gamma_stat ) ) {
   FoundPoissonMean=true;
   // pois_mean = (RooAbsReal*) pois_mean_arg;
   break;
      }
    }
    if( !FoundPoissonMean || !pois_mean_arg ) {
      std::cout << "Error: Did not find PoissonMean parameter in gamma constraint term: "
      << constraintTerm->GetName() << std::endl;
      throw std::runtime_error("Failed to find PoissonMean");
      return -1;
    }
    else {
      if(verbose) std::cout << "Found Poisson 'mean' term: " << pois_mean_arg->GetName() << std::endl;
    }

    TIter iter_product = pois_mean_arg->serverIterator(); //constraint_args
    RooAbsArg* term_in_product ;
    bool FoundTau=false;
    while((term_in_product=(RooAbsArg*)iter_product.Next())) {
      std::string serverName = term_in_product->GetName();
      //std::cout << "Checking Server: " << serverName << std::endl;
      if( serverName.find("_tau")!=std::string::npos ) {
   FoundTau = true;
   tau = (RooRealVar*) term_in_product;
      }
    }
    if( !FoundTau || !tau ) {
      std::cout << "Error: Did not find Tau parameter in gamma constraint term PoissonMean: "
      << pois_mean_arg->GetName() << std::endl;
      throw std::runtime_error("Failed to find Tau");
    }
    else {
      if(verbose) std::cout << "Found Poisson 'tau' term: " << tau->GetName() << std::endl;
    }

    return 0;

  }



} // close RooStats namespace
} // close HistFactory namespace
