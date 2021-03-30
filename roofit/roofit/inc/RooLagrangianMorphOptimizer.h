/* -*- mode: c++ -*- *********************************************************
 * Project: RooFit                                                           *
 *                                                                           *
 * authors:                                                                  *
 *  Lydia Brenner (lbrenner@cern.ch), Carsten Burgard (cburgard@cern.ch)     *
 *  Katharina Ecker (kecker@cern.ch), Adam Kaluza      (akaluza@cern.ch)     *
 *****************************************************************************/


#ifndef ROO_LAGRANGIAN_MORPHING_OPTIMIZER
#define ROO_LAGRANGIAN_MORPHING_OPTIMIZER

#include "RooLagrangianMorphFunc.h"
#include <map>

class TMinuit;
class TClass;
class TFolder;
class RooLagrangianMorphFunc;

class RooLagrangianMorphOptimizer {

protected:
  static RooLagrangianMorphOptimizer* gActiveInstance;
  static void targetFcn(Int_t&npar, Double_t* /*gin*/, Double_t&f, Double_t*par, Int_t /*flag*/);
  RooLagrangianMorphOptimizer(const char* input, const char* benchmarks, const std::vector<RooArgList>& vertices, TClass* containerType);
  RooArgList initInputs(const std::vector<std::string>& xsInputs);

public: 
  typedef RooLagrangianMorphFunc::ParamSet ParamCard;
  typedef std::map<const std::string, ParamCard> ParamCardSet;

  class RandomLagrangianGenerator;
  class RandomCouplingGenerator {
    double fLower; double fUpper;
    friend RandomLagrangianGenerator;
  public:
    RandomCouplingGenerator(double low, double high);
    double generate() const;
  };
  
  class RandomLagrangianGenerator {
    std::map<const std::string, const RandomCouplingGenerator> fCouplings;
  public:
    void addCoupling(const std::string& name, double min, double max);
    RooLagrangianMorphOptimizer::ParamCard generate();
    void print();
  };
    
  class Evaluator {
  public:
    virtual double operator() (double val_morphed, double unc_morphed, double val_benchmark, double unc_benchmark) = 0;
  };
protected:
  Evaluator* evaluator = NULL;
  double presetUncertainty = 0;

  class Benchmark {
  public:
    TString name;
    double xsection;
    double uncertainty;
    Benchmark(const TString& n, double xs, double unc) :
      name(n), xsection(xs), uncertainty(unc)
    {}
  };
public:
  
  RooLagrangianMorphOptimizer(const char* input, const char* benchmarks, const std::vector<RooArgList>& vertices, const std::vector<std::string>& xsInputs, TClass* containerType, const ParamCardSet& startvalues, const std::vector<std::vector<const char*> >& nonInterfering);
  RooLagrangianMorphOptimizer(const char* input, const char* benchmarks, const std::vector<RooArgList>& vertices, const std::vector<std::string>& xsInputs, TClass* containerType, const ParamCardSet& startvalues);
  virtual ~RooLagrangianMorphOptimizer();

  void setEvaluator(Evaluator* eval, double presetUncertainty = 0);
  int optimize();
  TGraph* makeLikelihoodGraph(const int& sample, const TString& , const int n = 1000, const double modus = 0.);
  double evaluate(const ParamCardSet& pcset, double& condition, double& l2norm);
  double evaluate(const ParamCardSet& pcset);

  static ParamCardSet readParamCards(const char* filename, const ParamCard& defaultvalues);
  static void writeParamCards(const ParamCardSet& set, const char* filename, const ParamCard& addParams = ParamCard());
  static void writeParamCards(const ParamCardSet& set, std::ostream& fout, const ParamCard& addParams = ParamCard());
  static void printParamCards(const ParamCardSet& set, const ParamCard& addParams = ParamCard());
  static void writeParamCard(const ParamCard& sample, std::ostream& fout, const ParamCard& addParams = ParamCard());
  static void printParamCard(const ParamCard& paramcard, const ParamCard& addParams = ParamCard());
  ParamCardSet getParamCards(const std::string& correlationstring = "", const std::string& addparamsstring = "");
  
  void printBestParameters();
  double getBestParameters(const int& sample, const TString& parametername);
  double getBestScore();
  
  static void splitpath(const TString& input, TString& filename, TString& subpath);
  
protected:

  void setCrossSection(TFolder* f, const double xs, const double xsunc);
  std::vector<double> getCurrentPars();
  void setup(const ParamCardSet& startvalues);
  void cloneFileContents(const TString& filename, bool addbenchmarks);
  std::vector<double> getParameterBounds(const std::vector<double>& pars);
  void printResult(const std::vector<Double_t>&par,Double_t&score);
  double testMorphing();
  void setupMorphing(const std::vector<double>& par);
  void setupMorphFunc();

protected:
  // inputs
  TString inputfilename;       
  TString benchmarkfilename;
  TString inputobservable; 
  TString benchmarkobservable;
  TClass* fXSContainerType = NULL;
  
  RooArgList sampleinputs;
  std::map<const int, std::string> parameters;
  std::vector<std::vector<const char*> > _nonInterfering;
  
protected:
  // temporaries
  RooLagrangianMorphFunc* morphFunc = NULL;
  RooLagrangianMorphFunc* xsHelper = NULL;
  TMinuit* ptMinuit = NULL;
  int ierflg;
  
  std::vector<RooArgList> vertices;
  
  std::vector<Benchmark> benchmarks;

  std::vector<TString> temporaries;
  
  RooArgList temporaries_list;
  
  size_t fnParameters;
  size_t fnfreeParameters;
  size_t fnSamples;
  
  std::map<const int, std::vector<std::string>> parameternames; // sample, parameternames

  int iterations = 0;
  std::vector<std::string> xsInputs;
  double bestScore;

public:
  double nSamples(){
    return this->fnSamples;
  }
  double nFreeParameters(){
    return this->fnfreeParameters;
  }
  double nParameters(){
    return this->fnParameters;
  }

  std::vector<std::string> freeParameterNames(){
    return this->parameternames[0];
  }
 
  ClassDef(RooLagrangianMorphOptimizer,0)

};

#endif
