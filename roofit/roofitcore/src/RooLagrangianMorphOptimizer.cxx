#include "TParameter.h"
#include "TMinuit.h"
#include <TParameter.h>
#include <TKey.h>
#include <TFile.h>
#include <TCanvas.h>
#include <TLine.h>
#include <TFolder.h>
#include <TH1F.h>
#include <TDirectory.h>

#include <Math/ProbFuncMathCore.h>

#include "TAxis.h"
#include "TGraph.h"

#include <iostream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <cfloat>

#include <RooStringVar.h>
#include <RooFormulaVar.h>

#include <RooLagrangianMorphing/RooLagrangianMorphOptimizer.h>
#include <RooLagrangianMorphing/RooLagrangianMorphing.h>

#define ERROR(arg){                                                     \
  if(RooLagrangianMorphing::gAllowExceptions){                                \
    std::stringstream err; err << arg << std::endl; throw(std::runtime_error(err.str())); \
  } else {                                                              \
    std::cerr << arg << std::endl;                                      \
  }}
#define INFO(arg) std::cout << arg << std::endl;

RooLagrangianMorphOptimizer* RooLagrangianMorphOptimizer::gActiveInstance = NULL;

RooLagrangianMorphOptimizer::ParamCard RooLagrangianMorphOptimizer::RandomLagrangianGenerator::generate(){
  RooLagrangianMorphOptimizer::ParamCard pc;
  for(const auto& c:this->fCouplings){
    pc.insert(std::make_pair(c.first,c.second.generate()));
  }
  return pc;
}
void RooLagrangianMorphOptimizer::RandomLagrangianGenerator::print(){
  for(const auto&gen:this->fCouplings){
    std::cout << gen.first << " " << gen.second.fLower << " -- " << gen.second.fUpper << std::endl;
  }
}
void RooLagrangianMorphOptimizer::RandomLagrangianGenerator::addCoupling(const std::string& name, double min, double max){
  this->fCouplings.insert(std::make_pair(name,RandomCouplingGenerator(min,max)));
}
RooLagrangianMorphOptimizer::RandomCouplingGenerator::RandomCouplingGenerator(double low, double high) : fLower(low),fUpper(high) {}
double RooLagrangianMorphOptimizer::RandomCouplingGenerator::generate() const {
  return this->fLower + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(this->fUpper-this->fLower)));
}

void RooLagrangianMorphOptimizer::setCrossSection(TFolder* f, const double xs, const double xsunc){
  if(this->fXSContainerType == TH1::Class()){
    // TH1 version
    TH1* xsection = (TH1*)(f->FindObject("xsection"));
    xsection->SetBinContent(1,xs);
    xsection->SetBinError(1,xsunc);
  } else if(this->fXSContainerType == TPair::Class()){
    // TPair version
    const TPair* p = static_cast<const TPair*>(f->FindObject("xsection"));
    (static_cast<TParameter<double>*>(p->Key()))->SetVal(xs);
    (static_cast<TParameter<double>*>(p->Value()))->SetVal(xsunc);
  }
}

void RooLagrangianMorphOptimizer::setupMorphFunc(){
  if(this->morphFunc){
    this->morphFunc->updateCoefficients();
  } else {
    // empty string as filename makes RooLagrangianMorphFunc use the gDirectory for input
    if(this->_nonInterfering.size()!=0){
      this->morphFunc = new RooLagrangianMorphPdf("morphing","morphing","","xsection",this->vertices,this->_nonInterfering,this->temporaries_list);
    }else{
      this->morphFunc = new RooLagrangianMorphPdf("morphing","morphing","","xsection",this->vertices,this->temporaries_list);
    }
  }
}
  
void RooLagrangianMorphOptimizer::setupMorphing(const std::vector<double>& par){
  // setup the temporary morphing function
  int iSample = 0;
  for(const auto& name:this->temporaries){
    TFolder* f = dynamic_cast<TFolder*>(gDirectory->Get(name.Data()));
    if(!f) continue;
    TH1* param_card = (TH1*)(f->FindObject("param_card"));
    for(size_t i=0; i<this->fnfreeParameters; ++i){
      // find bin in param_card with name parameternames[iSample][i]
      int thisbin = -1;
      for(size_t iBin=1; iBin<this->fnParameters+1; ++iBin){
        if(this->parameternames[iSample][i].compare(param_card->GetXaxis()->GetBinLabel(iBin))==0){
          thisbin = iBin;
          break;
        }
      }
      param_card->SetBinContent(thisbin,par[iSample*this->fnfreeParameters+i]);
    }
    this->xsHelper->setParameters(param_card);
    
    const double xs = this->xsHelper->expectedEvents();
    const double xsunc = (this->presetUncertainty > 0 ? this->presetUncertainty * xs : this->xsHelper->expectedUncertainty());

    this->setCrossSection(f,xs,xsunc);
    ++iSample;
  }
  this->setupMorphFunc();
}

void RooLagrangianMorphOptimizer::setEvaluator(Evaluator* eval, double preset){
  this->evaluator = eval;
  this->presetUncertainty = preset;
}

double RooLagrangianMorphOptimizer::testMorphing(){
  // evaluate the temporary morphing function
  double score = 0.;
  for(const auto& b:this->benchmarks){
    this->morphFunc->setParameters(b.name.Data());
    
    const double addScore = (*(this->evaluator))(this->morphFunc->expectedEvents(),
                                                 this->morphFunc->expectedUncertainty(),
                                                 b.xsection,
                                                 b.uncertainty);
    
    //    std::cout<<"evts="<<evts<<"+-"<<unc<< ", evts_bm="<<evts_benchmark<<"+-"<<unc_benchmark<<" addScore=" << addScore <<std::endl;
    
    score += addScore;
    
  }
  return score;
}

void RooLagrangianMorphOptimizer::printResult(const std::vector<Double_t>&par,Double_t&score){
  std::cout<<std::endl<<"it: "<<this->iterations<<" score "<<std::setprecision(100)<<score<<std::endl;
  for(size_t i=0; i<this->fnSamples;++i){
    for(size_t j=0; j<this->fnfreeParameters;++j){
      std::cout<<std::setprecision(100)<<par[i*this->fnfreeParameters+j]<<" ";
    }
    std::cout<<std::endl;
  }
}

std::vector<double> RooLagrangianMorphOptimizer::getParameterBounds(const std::vector<double>& pars){
  std::vector<double> pars_limit(pars);
  for(size_t i=0; i<this->fnSamples;++i){
    for(size_t j=0; j<this->fnfreeParameters;++j){
      std::string parname = this->parameternames[i][j];
      RooRealVar* var = this->xsHelper->getParameter(parname.c_str());
      int ipar = i*this->fnfreeParameters+j;
      if(pars[ipar]<var->getMin()) pars_limit[ipar] = var->getMin();
      if(pars[ipar]>var->getMax()) pars_limit[ipar] = var->getMax();
    }
  }
  return pars_limit;
}

void RooLagrangianMorphOptimizer::targetFcn(Int_t&npar, Double_t* /*gin*/, Double_t&f, Double_t*par, Int_t /*flag*/){
  // putting it all together
  std::vector<double> pars(par, par + npar);
  try {
    std::vector<double> pars_limit = RooLagrangianMorphOptimizer::gActiveInstance->getParameterBounds(pars);
    RooLagrangianMorphOptimizer::gActiveInstance->setupMorphing(pars_limit);
    f = RooLagrangianMorphOptimizer::gActiveInstance->testMorphing();
    double penalty = 0.;
    for(size_t ipar=0; ipar<pars.size();ipar++){
      penalty += std::pow(pars_limit[ipar] - pars[ipar],2);
    }
    f *= (1+penalty/npar);
    if(std::isinf(f) || std::isnan(f)){
      std::cout << "error: obtained non-numeric result" << std::endl;
      f = std::numeric_limits<double>::max();
    }
    if(f<RooLagrangianMorphOptimizer::gActiveInstance->bestScore) RooLagrangianMorphOptimizer::gActiveInstance->bestScore = f;
  } catch(std::exception& e){
    std::cout << "error: " << e.what() << std::endl;
    f = std::numeric_limits<double>::max();
  }
  if(RooLagrangianMorphOptimizer::gActiveInstance->iterations % 1000 == 0){
    std::cout<<"processing iteration "<<RooLagrangianMorphOptimizer::gActiveInstance->iterations<<"..."<<std::endl;
  }
  if(f==RooLagrangianMorphOptimizer::gActiveInstance->bestScore){
    RooLagrangianMorphOptimizer::gActiveInstance->printResult(pars,f);
  }

  ++RooLagrangianMorphOptimizer::gActiveInstance->iterations;
}

void RooLagrangianMorphOptimizer::splitpath(const TString& input, TString& filename, TString& subpath){
  size_t split = input.First(":");
  filename = input(0,split);
  subpath = input(split+1,input.Length()-split);
}


namespace {
  std::vector<std::string> split(const std::string& s,const char& delim){
    std::stringstream ss(s);
    std::vector<std::string> elems;
    std::string item;
    while (std::getline(ss, item, delim)){
      if (!item.empty()) elems.push_back(item);
    }
    return elems;
  }

  int ndigits(int num){
    int length = 1;
    while(num /= 10) length++;
    return length;
  }

  void trim(std::string& s){
    while ( s.front() == '"' || s.front() == ' ' || s.front() == '{') {
      s.erase( 0, 1 ); // erase the first character
    }
    while ( s.back() == '"' || s.back() == ' ' || s.back() == '}') {
      s.erase( s.size()-1, 1 ); // erase the last character
    }
  }
}

RooLagrangianMorphOptimizer::RooLagrangianMorphOptimizer(const char* input, const char* benchmarksArg, const std::vector<RooArgList>& verticesArg, TClass* containerType) : 
  fXSContainerType(containerType)
{
  // internal private constructor
  RooLagrangianMorphOptimizer::gActiveInstance = this;
  RooLagrangianMorphOptimizer::splitpath(input,this->inputfilename,this->inputobservable);
  RooLagrangianMorphOptimizer::splitpath(benchmarksArg,this->benchmarkfilename,this->benchmarkobservable);
  for(const auto& v:verticesArg){
    this->vertices.push_back(v);
  }
  this->cloneFileContents(this->inputfilename,false);
  this->cloneFileContents(this->benchmarkfilename,true);
}

RooArgList RooLagrangianMorphOptimizer::initInputs(const std::vector<std::string>& xsInputsArg){
  // init helper
  RooArgList inputs;
  for(auto const& sample: xsInputsArg) {
    RooStringVar* v = new RooStringVar(sample.c_str(),sample.c_str(),sample.c_str());
    this->xsInputs.push_back(sample);
    inputs.add(*v);
  }
  return inputs;
}

RooLagrangianMorphOptimizer::RooLagrangianMorphOptimizer(const char* input, const char* benchmarksArg, const std::vector<RooArgList>& verticesArg, const std::vector<std::string>& xsInputsArg, TClass* containerType, const ParamCardSet& startvalues, const std::vector<std::vector<const char*> >& nonInterfering) :
  RooLagrangianMorphOptimizer(input,benchmarksArg,verticesArg,containerType)
{
  // constructor with interference setting
  this->_nonInterfering = nonInterfering;
  this->xsHelper = new RooLagrangianMorphPdf("xsHelper","xsHelper","",inputobservable.Data(),this->vertices,this->_nonInterfering,this->initInputs(xsInputsArg));
  this->setup(startvalues);
}

RooLagrangianMorphOptimizer::RooLagrangianMorphOptimizer(const char* input, const char* benchmarksArg, const std::vector<RooArgList>& verticesArg, const std::vector<std::string>& xsInputsArg, TClass* containerType, const ParamCardSet& startvalues) : 
  RooLagrangianMorphOptimizer(input,benchmarksArg,verticesArg,containerType)
{
  // constructor without interference setting
  this->xsHelper = new RooLagrangianMorphPdf("xsHelper","xsHelper","",inputobservable.Data(),this->vertices,this->initInputs(xsInputsArg));
  this->setup(startvalues);
}

RooLagrangianMorphOptimizer::~RooLagrangianMorphOptimizer(){
  delete this->morphFunc;
  delete this->xsHelper;
  delete this->ptMinuit;
  gDirectory->Clear();
}

double RooLagrangianMorphOptimizer::evaluate(const ParamCardSet& pcset){
  double a,b;
  return evaluate(pcset,a,b);
}
double RooLagrangianMorphOptimizer::evaluate(const ParamCardSet& pcset, double& condition, double& l2norm){
  int i=0;
  for(const auto& pc:pcset){
    TFolder* f = dynamic_cast<TFolder*>(gDirectory->Get(this->temporaries[i]));
    i++;
    if(!f){
      throw std::runtime_error("unable to access temporary folder!");
    }
    TH1* param_card = dynamic_cast<TH1*>(f->FindObject("param_card"));
    if(!param_card){
      throw std::runtime_error("unable to access param_card!");
    }
    for(const auto&p:pc.second){
      int thisbin = -1;
      for(int iBin=1; iBin<=param_card->GetNbinsX(); ++iBin){
        if(p.first.compare(param_card->GetXaxis()->GetBinLabel(iBin))==0){
          thisbin = iBin;
          break;
        }
      }
      param_card->SetBinContent(thisbin,p.second);
    }
    this->xsHelper->setParameters(param_card);
    const double xs = this->xsHelper->expectedEvents();
    const double xsunc = (this->presetUncertainty > 0 ? this->presetUncertainty * xs : this->xsHelper->expectedUncertainty());
    this->setCrossSection(f,xs,xsunc);
  }
  this->setupMorphFunc();
  condition = this->morphFunc->getCondition();
  l2norm = this->morphFunc->getInvertedMatrix().E2Norm();
  double score = this->testMorphing();
  return score;
}

void RooLagrangianMorphOptimizer::setup(const ParamCardSet& startvalues){
  const RooArgList* parameterlist = this->xsHelper->getParameterSet();

  this->fnSamples = this->xsHelper->nSamples();
  this->fnParameters = this->xsHelper->nParameters();

  this->fnfreeParameters = 0;
  RooFIter itr(parameterlist->fwdIterator());
  TObject* obj;
  while((obj = itr.next())){
    RooRealVar* param = dynamic_cast<RooRealVar*>(obj);
    if(!param) continue;
    if(!this->xsHelper->isParameterConstant(param->GetName())){
      ++this->fnfreeParameters;
    }
  }

  // setup minuit
  this->ptMinuit = new TMinuit(this->fnfreeParameters * this->fnSamples);

  // initialize the storage
  size_t nparams = 0;
  for(size_t i=0; i<this->fnSamples; ++i){
    // create the TFolder structure
    TString name(TString::Format("sample%03d",int(i)));
    this->temporaries.push_back(name);
    this->temporaries_list.add(*(new RooStringVar(name.Data(),name.Data(),name.Data())));
    TFolder* f = new TFolder(name,name);
    gDirectory->Add(f);
    // create the param_card histogram
    TH1F* hist = new TH1F("param_card","param_card",this->fnParameters,0,this->fnParameters);
    hist->SetDirectory(0);
    f->Add(hist);
    // create the cross-section object
    
    if(this->fXSContainerType == TH1::Class()){
      // TH1 version
      TH1F* xs = new TH1F("xsection","xsection",1,0.,1.);
      xs->SetDirectory(0);
      f->Add(xs);
    } else if(this->fXSContainerType == TPair::Class()){
      // TPair version
      TPair* p = RooLagrangianMorphing::makeCrosssectionContainer(1,0);
      f->Add(p);
    }

    // loop over the parameters
    this->xsHelper->setParameters(this->xsInputs[i].c_str());
    const RooArgList* parset = this->xsHelper->getParameterSet();
    RooFIter itrpar(parset->fwdIterator());
    TObject* objpar = NULL;
    size_t iBin = 1;
    std::vector<std::string> parnames;
    while((objpar = itrpar.next())){
      RooRealVar* p = dynamic_cast<RooRealVar*>(objpar);
      if(!p) continue;
      // fill the histogram bins
      double value = p->getVal();
      hist->GetXaxis()->SetBinLabel(iBin,p->GetName());
      hist->SetBinContent(iBin,value);
      ++iBin;
      // add the parameter to minuit
      if(!p->isConstant()){
        TString parname = TString::Format("sample%03d_%s",int(i),p->GetName());
        double step = fabs(0.5*(p->getMax()-p->getMin())); // need to fiddle around with this number
        if(startvalues.find(name.Data()) != startvalues.end()){
          const ParamCard& pcard = startvalues.at(name.Data());
          if(pcard.find(p->GetName()) != pcard.end()){
            value = pcard.at(p->GetName());
          }
        }
        //ptMinuit->mnparm(nparams, parname.Data(),value, step, p->getMin(),p->getMax(),ierflg); // pars with limits
        ptMinuit->mnparm(nparams, parname.Data(),value, step, 0,0,ierflg); // pars with no limit

        parnames.push_back(p->GetName());
        nparams++;
      }
    }
    this->parameternames.insert(std::pair<int,std::vector<std::string> >(i,parnames));
  }
  this->bestScore = std::numeric_limits<double>::infinity();
}


void RooLagrangianMorphOptimizer::cloneFileContents(const TString& filename, bool addbenchmarks){
  TDirectory* storage = gDirectory;
  TFile *f = TFile::Open(filename,"READ");
  if (!f || f->IsZombie()) {
    ERROR("Cannot open file '" << filename << "!");
  }
  TKey *key;
  TIter nextkey(f->GetListOfKeys());
  while ((key = (TKey*)nextkey())) {
    TClass* cl = TClass::GetClass(key->GetClassName());
    if (!cl || !cl->InheritsFrom(TFolder::Class())) continue;
    f->cd();
    TFolder *folder = dynamic_cast<TFolder*>(key->ReadObj());
    TString name(folder->GetName());
    f->Remove(folder);
    storage->Add(folder);
    if(addbenchmarks){
      TH1* hist = (TH1*)(folder->FindObject(benchmarkobservable));
      if(!hist){
	ERROR("unable to read hist '"<<benchmarkobservable<<"' in folder '"<<folder->GetName()<<"' from file '"<<filename<<"'!");
      }
      double error;
      double xsec = hist->IntegralAndError(0, hist->GetNbinsX()+1,error);
      std::cout<<"benchmark xsec for "<<name<<" "<<xsec<<" "<<error<<std::endl;
      this->benchmarks.push_back(Benchmark(name,xsec,error));
    }
  }
  f->Close();
  delete f;

  storage->cd();
}

// plot likelihoods/FCNs

std::vector<double> RooLagrangianMorphOptimizer::getCurrentPars(){
  int npars = this->temporaries.size()*this->fnfreeParameters;
  std::vector<double> pars(npars,0.);
  int iSample = 0;
  for(const auto& name:this->temporaries){
    TFolder* f = dynamic_cast<TFolder*>(gDirectory->Get(name.Data()));
    if(!f) continue;
    TH1* param_card = (TH1*)(f->FindObject("param_card"));
    for(size_t i=0; i<this->fnfreeParameters; ++i){
      int thisbin = -1;
      for(size_t iBin=1; iBin<this->fnParameters+1; ++iBin){
        if(this->parameternames[iSample][i].compare(param_card->GetXaxis()->GetBinLabel(iBin))==0){
          thisbin = iBin;
          break;
        }
      }
      pars[iSample*this->fnfreeParameters+i] = param_card->GetBinContent(thisbin);
    }
    ++iSample;
  }
  return pars;
}

TGraph* RooLagrangianMorphOptimizer::makeLikelihoodGraph(const int& sample, const TString& parametername, const int n, const double modus){
  // modus: e.g. modus =  0.01 -> plot in 1% of the maximal parameter range around the minimum
  //             modus <= 0.   -> plot whole range
  std::vector<double> pars = getCurrentPars();
  std::vector<double> pars_start(pars);
  int npars = this->temporaries.size()*this->fnfreeParameters;
  int index = -1;
  for(size_t i=0; i<parameternames[sample].size();++i){
    if(parameternames[sample][i].compare(parametername.Data())==0){
      index = i;
      break;
    }
  }
  if(index==-1) ERROR("Did not find parametername "<<parametername<<" in list of used parameters!");
  int ipar = sample*this->fnfreeParameters + index;
  double parametervalue = pars[ipar];
  RooRealVar* parameter = this->xsHelper->getParameter(parametername.Data());
  double min = std::min(parameter->getMin(),parametervalue);
  double max = std::max(parameter->getMax(),parametervalue);
  if(modus<=0. || modus >=1.){
    double margin = (max-min)*0.05;
    min = min-margin;
    max = max+margin;
  }else{
    double margin = (max-min)*modus;
    min = parametervalue-margin;
    max = parametervalue+margin;
  }
  std::vector<Double_t> x = std::vector<Double_t>(n);
  std::vector<Double_t> y = std::vector<Double_t>(n);
  for (Int_t ipoint=0;ipoint<n;++ipoint){
    x[ipoint] = ipoint*(max-min)/n+min;
    pars[ipar] = x[ipoint];
    double f = 0;
    targetFcn(npars, NULL, f, &(pars[0]), 0);
    y[ipoint] = f;
    // std::cout<<"sample: "<<sample<<" freepar: "<<parametername<<",i: "<<ipoint<<" "<<x[ipoint]<<" "<<y[ipoint]<<std::endl;
  }
  // reset values in param_card
  double f;
  targetFcn(npars, NULL, f, &(pars_start[0]), 0);
  return new TGraph(n,&x[0],&y[0]);
}

RooLagrangianMorphOptimizer::ParamCardSet RooLagrangianMorphOptimizer::readParamCards(const char* ifname, const ParamCard& defaultvalues){
  ParamCardSet params;
  std::ifstream infile(ifname);
  while(infile.good() && infile.get() != '{'){
    // do nothing;
  }
  std::string line;
  while (std::getline(infile, line)){
    std::istringstream iss(line);
    std::string name;
    iss >> name;
    trim(name);
    if(name.size() < 1) continue;
    ParamCard pcard = defaultvalues;
    while(iss.good() && iss.get() != '{'){
      // do nothing;
    }
    while(iss.good()){
      std::string pname,pval;
      if(!std::getline(iss, pname, ':')) {
        continue;
      }
      if(!std::getline(iss, pval, ',')) {
        continue;
      }
      trim(pname);
      trim(pval);

      char * e;
      errno = 0;
      double v = std::strtod(pval.c_str(), &e);
      if (*e != '\0' || errno != 0 ){
        if(defaultvalues.find(pname) != defaultvalues.end()){
          v = defaultvalues.at(pname);
        } else {
          ERROR("failed to parse definition of '" << pname << "' and no default value given!");
        }
      }
      pcard[pname] = v;
    }
    params[name] = pcard;
  }
  return params;
}


RooLagrangianMorphOptimizer::ParamCardSet RooLagrangianMorphOptimizer::getParamCards(const std::string& correlationstring, const std::string& addparamsstring){
  // format correlation vector
  std::vector<std::vector<std::string>> correlations;
  std::vector<std::string> corr_tmp = split(correlationstring,',');
  for(auto const& thiscorr: corr_tmp){
    correlations.push_back(split(thiscorr,'='));
  }  

  // format additional parameter vector
  std::vector<std::vector<std::string>> addparams;
  std::vector<std::string> addparam_tmp = split(addparamsstring,',');
  for(auto const& thisaddparam: addparam_tmp){
    addparams.push_back(split(thisaddparam,'='));
  }  
  
  // write param map
  ParamCardSet parammap;

  // get parameters
  std::vector<double> pars = getCurrentPars();

  int samplenumwidth = ndigits(this->fnSamples);
  for(size_t i=0; i<this->fnSamples;++i){

    // get this samplename
    std::stringstream ss_samplename;
    ss_samplename << "s" << std::setw(samplenumwidth) << std::setfill('0') << i;
    std::string samplename = ss_samplename.str();

    // add additional fixed parameters to sample
    ParamCard thispars;
    for(auto const& addparam: addparams){
      if(addparam.size()!=2){
	std::cout<<"Wrong formated additionalpars option: "<<addparamsstring<<"\nShould be in the form: kBSM1=value,kBSM2=value,...";
	exit(0);
      }
      thispars[addparam[0]] = std::atof(addparam[1].c_str());
    }

    // get parameters of this sample
    for(size_t j=0; j<this->fnfreeParameters;++j){
      std::string thisparname = this->parameternames[i][j];
      double val = pars[i*this->fnfreeParameters+j];
      thispars[thisparname] = val;
      
      // check for correlated parameters
      for(const auto& corr: correlations){
        if(std::find(corr.begin(),corr.end(),thisparname) != corr.end()){
          for(auto const& corrpar: corr){
            thispars[corrpar] = val;
          }
        }
      }
    }
    parammap[samplename] = thispars;
  }
  return parammap;
}

void RooLagrangianMorphOptimizer::printParamCards(const RooLagrangianMorphOptimizer::ParamCardSet& parammap, const ParamCard& addParams){
  // write param file
  RooLagrangianMorphOptimizer::writeParamCards(parammap, std::cout, addParams);
  std::cout.flush();
}

void RooLagrangianMorphOptimizer::printParamCard(const RooLagrangianMorphOptimizer::ParamCard& paramcard, const ParamCard& addParams){
  // write param file
  RooLagrangianMorphOptimizer::writeParamCard(paramcard, std::cout, addParams);
  std::cout << std::endl;
}


void RooLagrangianMorphOptimizer::writeParamCards(const RooLagrangianMorphOptimizer::ParamCardSet& parammap, const char*outname, const ParamCard& addParams){
  // write param file
  std::ofstream fout(outname);
  RooLagrangianMorphOptimizer::writeParamCards(parammap, fout, addParams);
  fout.close();
}

void RooLagrangianMorphOptimizer::writeParamCard(const RooLagrangianMorphOptimizer::ParamCard& sample, std::ostream& fout, const ParamCard& addParams){
  // write param card
  for(auto const& par: addParams){
    fout << "\"" << par.first << "\":" << par.second;
    fout << ", ";
  }
  size_t j = sample.size();
  for(auto const& par:sample){
    fout << "\"" << par.first << "\":" << par.second;
    j--;
    if(j!=0) fout << ", ";
  }
}

void RooLagrangianMorphOptimizer::writeParamCards(const RooLagrangianMorphOptimizer::ParamCardSet& parammap, std::ostream& fout, const ParamCard& addParams){
  // write param file
  fout << "param_config = {\n";
  size_t i = parammap.size();
  for(auto const& sample: parammap){
    fout << "  \"" <<  sample.first << "\" : {";
    RooLagrangianMorphOptimizer::writeParamCard(sample.second,fout,addParams);
    fout << "}";
    i--;
    if(i!=0) fout << ",\n";
  }
  fout << "\n}\n";
}


// putting everything together

void RooLagrangianMorphOptimizer::printBestParameters(){
  // execute once targetFcn to initialize morphfunc and get score
  std::vector<double> pars = getCurrentPars();
  double f = 0;
  int npars = this->temporaries.size()*this->fnfreeParameters;
  targetFcn(npars, NULL, f, &(pars[0]), 0);
  printResult(pars,f);
}

double RooLagrangianMorphOptimizer::getBestParameters(const int& sample, const TString& parametername){
  // execute once targetFcn to initialize morphfunc and get score
  std::vector<double> pars = getCurrentPars();
  int index = -1;
  for(size_t i=0; i<parameternames[sample].size();++i){
    if(parameternames[sample][i].compare(parametername.Data())==0){
      index = i;
      break;
    }
  }
  if(index==-1) ERROR("Did not find parametername "<<parametername<<" in list of used parameters!");
  int ipar = sample*this->fnfreeParameters + index;
  return pars[ipar];
}

double RooLagrangianMorphOptimizer::getBestScore(){
  // execute once targetFcn to initialize morphfunc and get score
  std::vector<double> pars = getCurrentPars();
  double f = 0;
  int npars = this->temporaries.size()*this->fnfreeParameters;
  targetFcn(npars, NULL, f, &(pars[0]), 0);
  return f;
}

int RooLagrangianMorphOptimizer::optimize(){
  // finish configuring minuit
  //  select verbose level:
  //    default :     (58 lines in this test)
  //    -1 : minimum  ( 4 lines in this test)
  //     0 : low      (31 lines)
  //     1 : medium   (61 lines)
  //     2 : high     (89 lines)
  //     3 : maximum (199 lines in this test)
  //
  ptMinuit->SetPrintLevel();
  // set the user function that calculates chi_square (the value to minimize)
  ptMinuit->SetFCN(targetFcn);

  // DISCLAIMER:
  // I don't understand any of this arcane BS. I stole it from here: 
  // http://tesla.desy.de/~pcastro/example_progs/fit/minuit_c_style/fit_minuit.cxx

  ptMinuit->mnsimp();
  ptMinuit->mnmigr();

  // Print results
  std::cout << "\nPrint results from minuit\n";
  for(auto p:parameters){
    double fParamVal;
    double fParamErr;
    ptMinuit->GetParameter(p.first,fParamVal,fParamErr);
    std::cout << p.second << "=" << fParamVal << "+/-" << fParamErr << std::endl;
  }

  // if you want to access to these parameters, use:
  Double_t amin,edm,errdef;
  Int_t nvpar,nparx,icstat;
  ptMinuit->mnstat(amin,edm,errdef,nvpar,nparx,icstat);
  //void mnstat(Double_t &fmin, Double_t &fedm, Double_t &errdef, Int_t &npari, Int_t &nparx, Int_t &istat) 
  //*-*-*-*-*Returns concerning the current status of the minimization*-*-*-*-*
  //*-*      =========================================================
  //*-*       User-called
  //*-*          Namely, it returns:
  //*-*        FMIN: the best function value found so far
  //*-*        FEDM: the estimated vertical distance remaining to minimum
  //*-*        ERRDEF: the value of UP defining parameter uncertainties
  //*-*        NPARI: the number of currently variable parameters
  //*-*        NPARX: the highest (external) parameter number defined by user
  //*-*        ISTAT: a status integer indicating how good is the covariance
  //*-*           matrix:  0= not calculated at all
  //*-*                    1= approximation only, not accurate
  //*-*                    2= full matrix, but forced positive-definite
  //*-*                    3= full accurate covariance matrix
  //*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
  std::cout << "\n";
  std::cout << " Minimum target function square = " << amin << "\n";
  std::cout << " Estimated vert. distance to min. = " << edm << "\n";
  std::cout << " Number of variable parameters = " << nvpar << "\n";
  std::cout << " Highest number of parameters defined by user = " << nparx << "\n";
  std::cout << " Status of covariance matrix = " << icstat << "\n";

  std::cout << "\n";
  ptMinuit->mnprin(3,amin);
  //*-*-*-*Prints the values of the parameters at the time of the call*-*-*-*-*
  //*-*    ===========================================================
  //*-*        also prints other relevant information such as function value,
  //*-*        estimated distance to minimum, parameter errors, step sizes.
  //*-*
  //*-*         According to the value of IKODE, the printout is:
  //*-*    IKODE=INKODE= 0    only info about function value
  //*-*                  1    parameter values, errors, limits
  //*-*                  2    values, errors, step sizes, internal values
  //*-*                  3    values, errors, step sizes, first derivs.
  //*-*                  4    values, parabolic errors, MINOS errors
  //*-*    when INKODE=5, MNPRIN chooses IKODE=1,2, or 3, according to ISW(2)
  //*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

  return 0;
}
