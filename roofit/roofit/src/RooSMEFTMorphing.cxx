#include "RooSMEFTMorphing.h"
#include "RooFormulaVar.h"

///////////////////////////////////////////////////////////////////////////////
// Higgs Characterization Model ///////////////////////////////////////////////
// https://arxiv.org/pdf/1306.6464.pdf ////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

 ///////////////////////////////////////////////////////////////////////////////
 /// find and, if necessary, create a parameter from a list
 template< class T >
 inline RooAbsArg& get(T& operators, const char* name, double defaultval=0)
 {
   RooAbsArg* kappa = operators.find(name);
   if(kappa) return *kappa;
   RooRealVar* newKappa = new RooRealVar(name,name,defaultval);
 //  double minVal = 0.9*defaultval;
 //  double maxVal = 1.1*defaultval;
 //  newKappa->setRange(std::min(minVal,maxVal),std::max(minVal,maxVal));
   newKappa->setConstant(false);
   operators.add(*newKappa);
   return *newKappa;
 }
 ///////////////////////////////////////////////////////////////////////////////
 /// find and, if necessary, create a parameter from a list
 template< class T >
 inline RooAbsArg& get(T& operators, const std::string& name, double defaultval=0)
 {
   return get(operators,name.c_str(),defaultval);
 }

  ////////////////////////////////////////////////////////////////////////////////
  /// create a new coupling and add it to the set

  template< class T >
  inline void addCoupling(T& set, const TString& name, const TString& formula, const RooArgList& components, bool isNP)
  {
    if(!set.find(name)){
      RooFormulaVar* c = new RooFormulaVar(name,formula,components);
      c->setAttribute("NP",isNP);
      set.add(*c);
    }
  }


namespace {

////////////////////////////////////////////////////////////////////////////////
/// create the couplings needed for SMEFT

  RooArgSet makeSMEFTCouplings(RooAbsCollection& operators, const char* label, const std::vector<std::string>& names)
  {
    DEBUG("creating SMEFT " << label << " couplings");
    RooArgSet couplings(label);
    DEBUG("adding Lambda");
    RooAbsArg& Lambda = get(operators,"Lambda",1000);
    DEBUG("adding SM");
    RooAbsArg& sm = get(operators,"SM",1.);
    couplings.add(sm);
    for(const auto& op:names){
      DEBUG("adding "+op);
      addCoupling(couplings,TString::Format("_g%s",op.c_str()) ,TString::Format("c%s/Lambda/Lambda",op.c_str()),RooArgList(Lambda,get(operators,TString::Format("c%s",op.c_str()))),true);
    }
    return couplings;
  }
}

////////////////////////////////////////////////////////////////////////////////
/// create the couplings needed for SMEFT

RooArgSet RooLagrangianMorphing::makeSMEFTCouplings(RooAbsCollection& operators) {
  return ::makeSMEFTCouplings(operators,"all",{"dH","eH","G","HB","Hbox","Hd","HD","He","HG","HGtil","Hl1","Hl3","Hq1","Hq3","Hu","HW","HWtil","HWB","ll","uG","uH","W"});
}

////////////////////////////////////////////////////////////////////////////////
/// create the couplings needed for SMEFT ggF

RooArgSet RooLagrangianMorphing::makeSMEFTggFCouplings(RooAbsCollection& operators) {
  return ::makeSMEFTCouplings(operators,"ggF",{"HG"});
}

////////////////////////////////////////////////////////////////////////////////
/// create the couplings needed for SMEFT VBF

RooArgSet RooLagrangianMorphing::makeSMEFTVBFCouplings(RooAbsCollection& operators) {
  return ::makeSMEFTCouplings(operators,"VBF",{"HW","Hq3","Hu","ll1","HDD","HW","Hl3"});
}

////////////////////////////////////////////////////////////////////////////////
/// create the couplings needed for SMEFT ttH

//RooArgSet RooLagrangianMorphing::makeSMEFTtthCouplings(RooAbsCollection& operators) {
//  return ::makeSMEFTCouplings(operators,"ttH",{"Hd","Hq1","uu1","HWB","uWAbs","Hl3","uu","H","ud1","uGAbs","qd1","uBAbs","HDD","qd8","qq11","Hq3","qu1","HB","Hu","qu8","qq3","q     q1","uHAbs","HG","qq31","ud8","HW"});
//}

////////////////////////////////////////////////////////////////////////////////
/// create the couplings needed for SMEFT H->WW

RooArgSet RooLagrangianMorphing::makeSMEFTHWWCouplings(RooAbsCollection& operators) {
  return ::makeSMEFTCouplings(operators,"HWW",{"HW","HWtil","Hbox","HDD"});
}

////////////////////////////////////////////////////////////////////////////////
/// create the couplings needed for SMEFT H->yy

RooArgSet RooLagrangianMorphing::makeSMEFTHyyCouplings(RooAbsCollection& operators) {
  return ::makeSMEFTCouplings(operators,"Hyy",{"HB"});
}
