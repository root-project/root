#include "RooHCMorphing.h"
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

////////////////////////////////////////////////////////////////////////////////
/// create the couplings needed for ggF vertices

RooArgSet makeHCggFCouplings(RooAbsCollection& operators)
{
  //DEBUG("creating ggF couplings");
  RooArgSet prodCouplings("ggF");
  RooAbsArg& cosa = get(operators,"cosa",1);
  addCoupling(  prodCouplings,"_gHgg" ,"cosa*kHgg",                       RooArgList(cosa,get(operators,"kHgg")),false);
  addCoupling(  prodCouplings,"_gAgg" ,"sqrt(1-(cosa*cosa))*kAgg",        RooArgList(cosa,get(operators,"kAgg")),true);
  return prodCouplings;
}

////////////////////////////////////////////////////////////////////////////////
/// create the couplings needed for VBF vertices

RooArgSet makeHCVBFCouplings(RooAbsCollection& operators)
{
  RooArgSet prodCouplings("VBF");
  RooAbsArg& cosa = get(operators,"cosa",1);
  RooAbsArg& lambda = get(operators,"Lambda",1000);
  addCoupling(prodCouplings,"_gSM"  ,"cosa*kSM",                        RooArgList(cosa,get(operators,"kSM")),false);
  addCoupling(prodCouplings,"_gHaa" ,"cosa*kHaa",                       RooArgList(cosa,get(operators,"kHaa")),true);
  addCoupling(prodCouplings,"_gAaa" ,"sqrt(1-(cosa*cosa))*kAaa",        RooArgList(cosa,get(operators,"kAaa")),true);
  addCoupling(prodCouplings,"_gHza" ,"cosa*kHza",                       RooArgList(cosa,get(operators,"kHza")),true);
  addCoupling(prodCouplings,"_gAza" ,"sqrt(1-(cosa*cosa))*kAza",        RooArgList(cosa,get(operators,"kAza")),true);
  addCoupling(prodCouplings,"_gHzz" ,"cosa*kHzz/Lambda",                RooArgList(cosa,get(operators,"kHzz"),lambda),true);
  addCoupling(prodCouplings,"_gAzz" ,"sqrt(1-(cosa*cosa))*kAzz/Lambda", RooArgList(cosa,get(operators,"kAzz"),lambda),true);
  addCoupling(prodCouplings,"_gHdz","cosa*kHdz/Lambda",                 RooArgList(cosa,get(operators,"kHdz"),lambda),true);
  addCoupling(prodCouplings,"_gHww" ,"cosa*kHww/Lambda",                RooArgList(cosa,get(operators,"kHww"),lambda),true);
  addCoupling(prodCouplings,"_gAww" ,"sqrt(1-(cosa*cosa))*kAww/Lambda", RooArgList(cosa,get(operators,"kAww"),lambda),true);
  addCoupling(prodCouplings,"_gHdwR","cosa*kHdwR/Lambda",               RooArgList(cosa,get(operators,"kHdwR"),lambda),true);
  addCoupling(prodCouplings,"_gHdwI","cosa*kHdwI/Lambda",               RooArgList(cosa,get(operators,"kHdwI"),lambda),true);
  addCoupling(prodCouplings,"_gHda","cosa*kHda/Lambda",                 RooArgList(cosa,get(operators,"kHda"),lambda),true);
  return prodCouplings;
}

////////////////////////////////////////////////////////////////////////////////
/// create the couplings needed for HWW vertices

RooArgSet makeHCHWWCouplings(RooAbsCollection& operators)
{
  //DEBUG("creating HWW couplings");
  RooArgSet decCouplings("HWW");
  RooAbsArg& cosa = get(operators,"cosa",1);
  RooAbsArg& lambda = get(operators,"Lambda",1000);
  addCoupling(decCouplings,"_gSM"  ,"cosa*kSM",                        RooArgList(cosa,get(operators,"kSM")),false);
  addCoupling(decCouplings,"_gHww" ,"cosa*kHww/Lambda",                RooArgList(cosa,get(operators,"kHww"),lambda),true);
  addCoupling(decCouplings,"_gAww" ,"sqrt(1-(cosa*cosa))*kAww/Lambda", RooArgList(cosa,get(operators,"kAww"),lambda),true);
  addCoupling(decCouplings,"_gHdwR","cosa*kHdwR/Lambda",               RooArgList(cosa,get(operators,"kHdwR"),lambda),true);
  addCoupling(decCouplings,"_gHdwI","cosa*kHdwI/Lambda",               RooArgList(cosa,get(operators,"kHdwI"),lambda),true);
  return decCouplings;
}

////////////////////////////////////////////////////////////////////////////////
/// create the couplings needed for HZZ vertices

RooArgSet makeHCHZZCouplings(RooAbsCollection& operators)
{
  RooArgSet decCouplings("HZZ");
  RooAbsArg& cosa = get(operators,"cosa",1);
  RooAbsArg& lambda = get(operators,"Lambda",1000);
  addCoupling(decCouplings,"_gSM"  ,"cosa*kSM",                        RooArgList(cosa,get(operators,"kSM")),true);
  addCoupling(decCouplings,"_gHzz" ,"cosa*kHzz/Lambda",                RooArgList(cosa,get(operators,"kHzz"),lambda),true);
  addCoupling(decCouplings,"_gAzz" ,"sqrt(1-(cosa*cosa))*kAzz/Lambda", RooArgList(cosa,get(operators,"kAzz"),lambda),true);
  addCoupling(decCouplings,"_gHdz","cosa*kHdz/Lambda",                 RooArgList(cosa,get(operators,"kHdz"),lambda),true);
  addCoupling(decCouplings,"_gHaa" ,"cosa*kHaa",                       RooArgList(cosa,get(operators,"kHaa")),true);
  addCoupling(decCouplings,"_gAaa" ,"sqrt(1-(cosa*cosa))*kAaa",        RooArgList(cosa,get(operators,"kAaa")),true);
  addCoupling(decCouplings,"_gHza" ,"cosa*kHza",                       RooArgList(cosa,get(operators,"kHza")),true);
  addCoupling(decCouplings,"_gAza" ,"sqrt(1-(cosa*cosa))*kAza",        RooArgList(cosa,get(operators,"kAza")),true);
  addCoupling(decCouplings,"_gHda","cosa*kHda/Lambda",                 RooArgList(cosa,get(operators,"kHda"),lambda),true);
  return decCouplings;
}

////////////////////////////////////////////////////////////////////////////////
/// create the couplings needed for Hll vertices

RooArgSet makeHCHllCouplings(RooAbsCollection& operators)
{
  RooArgSet decCouplings("Hmumu");
  RooAbsArg& cosa = get(operators,"cosa",1);
  addCoupling(decCouplings,"_gHll" ,"cosa*kHll",                       RooArgList(cosa,get(operators,"kHll")),false);
  return decCouplings;
}