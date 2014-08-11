#include "Math/GeneticMinimizer.h"

#include "TMVA/GeneticAlgorithm.h"
#include "TMVA/IFitterTarget.h"

#include "Math/IFunction.h"
#include "Math/GenAlgoOptions.h"

#include "TError.h"

#include <cassert>

namespace ROOT {
namespace Math {


// wrapper class for TMVA interface to evaluate objective function
class MultiGenFunctionFitness : public TMVA::IFitterTarget {
private:
   unsigned int fNCalls;
   unsigned int fNFree;
   const ROOT::Math::IMultiGenFunction& fFunc;
   std::vector<int> fFixedParFlag;
   mutable std::vector<double> fValues;

public:
   MultiGenFunctionFitness(const ROOT::Math::IMultiGenFunction& function) : fNCalls(0),
                                                                            fFunc(function)
   { fNFree = fFunc.NDim(); }

   unsigned int NCalls() const { return fNCalls; }
   unsigned int NDims() const { return fNFree; }

   unsigned int NTotal() const { return fFunc.NDim(); }

   void FixParameter(unsigned int ipar, double value, bool fix = true) {

      if (fValues.size() != fFunc.NDim() ) {
         fValues.resize(fFunc.NDim() );
         fFixedParFlag.resize(fFunc.NDim());
      }

      if (ipar >= fValues.size() ) return;

      // first find if it has been already fixed
      fFixedParFlag[ipar] = fix;
      fValues[ipar] = value;
      // count number of fixed params
      for (unsigned int i = 0; i < fFixedParFlag.size(); ++i)
         if (!fFixedParFlag[i] ) fNFree++;

   }

   // transfrom from internal parameters (not fixed to external vector which include the fixed ones)
   const std::vector<double> & Transform( const std::vector<double> & factors) const {
      unsigned int n = fValues.size();
      if (n == 0 || fNFree == n )
         return factors;

      // in case of fixed parameters
      for (unsigned int i = 0, j = 0; i < n ; ++i) {
         if (!fFixedParFlag[i] ) {
            assert (j < fNFree);
            fValues[i] = factors[j];
            j++;
         }
      }
      return fValues;
   }

   Double_t Evaluate(const std::vector<double> & factors ) const {
      const std::vector<double> & x = Transform( factors);
      return fFunc(&x[0]);
   }

   Double_t EstimatorFunction(std::vector<double> & factors ){
      fNCalls += 1;
      return Evaluate( factors);
   }
};

GeneticMinimizerParameters::GeneticMinimizerParameters()
{
   // constructor of parameters with default values (use 100 is max iterations is not defined)
   int defmaxiter = ROOT::Math::MinimizerOptions::DefaultMaxIterations();
   fNsteps   =  (defmaxiter > 0) ?  defmaxiter : 100;
   fPopSize  =300;
   fCycles   = 3;
   fSC_steps =10;
   fSC_rate  =5;
   fSC_factor=0.95;
   fConvCrit =10.0 * ROOT::Math::MinimizerOptions::DefaultTolerance(); // default is 0.001
   if (fConvCrit <=0 ) fConvCrit = 0.001;
   fSeed=0;  // random seed
}

// genetic minimizer class

GeneticMinimizer::GeneticMinimizer(int ):
   fFitness(0),
   fMinValue(0),
   fParameters(GeneticMinimizerParameters() )
{

   // check with default minimizer options
   ROOT::Math::IOptions * geneticOpt = ROOT::Math::MinimizerOptions::FindDefault("Genetic");
   if (geneticOpt) {
      ROOT::Math::MinimizerOptions opt; // create using default options
      opt.SetExtraOptions(*geneticOpt);
      this->SetOptions(opt);
   }

   // set the parameters
   SetTolerance(0.1 * fParameters.fConvCrit);
   SetMaxIterations( fParameters.fNsteps);
 }

GeneticMinimizer::~GeneticMinimizer()
{
   if ( fFitness )
   {
      delete fFitness;
      fFitness = 0;
   }
}

void GeneticMinimizer::Clear()
{
   fRanges.clear();
   fResult.clear();
   if ( fFitness )
   {
      delete fFitness;
      fFitness = 0;
   }
}

void GeneticMinimizer::SetFunction(const ROOT::Math::IMultiGenFunction & func)
{
   Clear();

   fFitness = new MultiGenFunctionFitness(func);
   fResult = std::vector<double>(func.NDim() );
   assert(fResult.size() == NDim() );
}

bool GeneticMinimizer::SetLimitedVariable(unsigned int , const std::string & , double , double , double lower , double upper )
{
   fRanges.push_back( new TMVA::Interval(lower,upper) );

   return true;
}

bool GeneticMinimizer::SetVariable(unsigned int, const std::string& name, double value, double step)
{
   //It does nothing! As there is no variable if it has no limits!
   double lower = value - (50 * step);
   double upper = value + (50 * step);
   Info("GeneticMinimizer::SetVariable", "Variables should be limited - set automatic range to 50 times step size for %s : [%f, %f]",
        name.c_str(),lower,upper);
   fRanges.push_back( new TMVA::Interval(lower, upper ) );

   return true;
}

bool GeneticMinimizer::SetFixedVariable(unsigned int par, const std::string& name, double value) {
   // set a fixed variable
   if (!fFitness) {
      Error("GeneticMinimizer::SetFixedVariable", "Function has not been set - cannot set fixed variables %s",name.c_str());
      return false;
   }

   static_cast<MultiGenFunctionFitness*>(fFitness)->FixParameter(par, value);
   return true;
}


void GeneticMinimizer::SetParameters(const GeneticMinimizerParameters & params )
{
   fParameters  = params;
   // set also the one defined in Minimizer
   SetTolerance( 0.1 * fParameters.fConvCrit);
   SetMaxIterations( fParameters.fNsteps );
}

ROOT::Math::MinimizerOptions GeneticMinimizer::Options() const {
   ROOT::Math::MinimizerOptions opt;
   GetGeneticOptions(opt);
   return opt;
}

void  GeneticMinimizer::GetGeneticOptions(ROOT::Math::MinimizerOptions & opt) const {
   // get  the genetic options of the class and return them in the MinimizerOptions class
   opt.SetTolerance(fParameters.fConvCrit/10); // use a factor of 10 to have default as Minuit
   opt.SetPrintLevel(PrintLevel() );
   opt.SetMaxIterations(fParameters.fNsteps);
   // use fixed or dammy value for the other options
   opt.SetMinimizerType("Genetic");
   opt.SetMaxFunctionCalls(0);
   opt.SetStrategy(-1);
   opt.SetErrorDef(0);
   opt.SetPrecision(0);
   opt.SetMinimizerAlgorithm("");

   ROOT::Math::GenAlgoOptions geneticOpt;
   geneticOpt.SetValue("PopSize",fParameters.fPopSize);
   geneticOpt.SetValue("Steps",fParameters.fNsteps);
   geneticOpt.SetValue("Cycles",fParameters.fCycles);
   geneticOpt.SetValue("SC_steps",fParameters.fSC_steps);
   geneticOpt.SetValue("SC_rate",fParameters.fSC_rate);
   geneticOpt.SetValue("SC_factor",fParameters.fSC_factor);
   geneticOpt.SetValue("ConvCrit",fParameters.fConvCrit);
   geneticOpt.SetValue("RandomSeed",fParameters.fSeed);

   opt.SetExtraOptions(geneticOpt);
}

void GeneticMinimizer::SetOptions(const ROOT::Math::MinimizerOptions & opt)
{
   SetTolerance(opt.Tolerance() );
   SetPrintLevel(opt.PrintLevel() );
   //SetMaxFunctionCalls(opt.MaxFunctionCalls() );
   SetMaxIterations(opt.MaxIterations() );

   fParameters.fConvCrit = 10.*opt.Tolerance(); // use a factor of 10 to have default as Minuit

   // set genetic parameter from minimizer options
   const ROOT::Math::IOptions * geneticOpt = opt.ExtraOptions();
   if (!geneticOpt) {
      Warning("GeneticMinimizer::SetOptions", "No specific genetic minimizer options have been set");
      return;
   }

   // if options are not existing values will not be set
   geneticOpt->GetValue("PopSize",fParameters.fPopSize);
   geneticOpt->GetValue("Steps",fParameters.fNsteps);
   geneticOpt->GetValue("Cycles",fParameters.fCycles);
   geneticOpt->GetValue("SC_steps",fParameters.fSC_steps);
   geneticOpt->GetValue("SC_rate",fParameters.fSC_rate);
   geneticOpt->GetValue("SC_factor",fParameters.fSC_factor);
   geneticOpt->GetValue("ConvCrit",fParameters.fConvCrit);
   geneticOpt->GetValue("RandomSeed",fParameters.fSeed);

   // use same of options in base class
   int maxiter = opt.MaxIterations();
   if (maxiter > 0 && fParameters.fNsteps > 0 && maxiter !=  fParameters.fNsteps  )    {
      Warning("GeneticMinimizer::SetOptions", "max iterations value given different than  than Steps - set equal to Steps %d",fParameters.fNsteps);
   }
   if (fParameters.fNsteps > 0) SetMaxIterations(fParameters.fNsteps);

}

bool GeneticMinimizer::Minimize()
{

   if (!fFitness) {
      Error("GeneticMinimizer::Minimize","Fitness function has not been set");
      return false;
   }

   // sync parameters
   if (MaxIterations() > 0) fParameters.fNsteps = MaxIterations();
   if (Tolerance() > 0) fParameters.fConvCrit = 10* Tolerance();

   TMVA::GeneticAlgorithm mg( *fFitness, fParameters.fPopSize, fRanges, fParameters.fSeed );

   if (PrintLevel() > 0) {
      std::cout << "GeneticMinimizer::Minimize  - Start iterating - max iterations = " <<  MaxIterations()
                << " conv criteria (tolerance) =  "   << fParameters.fConvCrit << std::endl;
   }

   fStatus = 0;
   unsigned int niter = 0;
   do {
      mg.Init();

      mg.CalculateFitness();

      // Just for debugging options
      //mg.GetGeneticPopulation().Print(0);

      mg.GetGeneticPopulation().TrimPopulation();

      mg.SpreadControl( fParameters.fSC_steps, fParameters.fSC_rate, fParameters.fSC_factor );

      if (PrintLevel() > 2) {
         std::cout << "New Iteration " << niter << " with  parameter values :" << std::endl;
         TMVA::GeneticGenes* genes = mg.GetGeneticPopulation().GetGenes( 0 );
         if (genes) {
            std::vector<Double_t> gvec;
            gvec = genes->GetFactors();
            for (unsigned int i = 0; i < gvec.size(); ++i) {
               std::cout << gvec[i] << "    ";
            }
            std::cout << std::endl;
            std::cout << "\tFitness function value = " <<  static_cast<MultiGenFunctionFitness*>(fFitness)->Evaluate(gvec) << std::endl;
         }
      }
      niter++;
      if ( niter > MaxIterations() && MaxIterations() > 0) {
         if (PrintLevel() > 0) {
            Info("GeneticMinimizer::Minimize","Max number of iterations %d reached - stop iterating",MaxIterations());
         }
         fStatus = 1;
         break;
      }

   } while (!mg.HasConverged( fParameters.fNsteps, fParameters.fConvCrit ));  // converged if: fitness-improvement < CONVCRIT within the last CONVSTEPS loops

   TMVA::GeneticGenes* genes = mg.GetGeneticPopulation().GetGenes( 0 );
   std::vector<Double_t> gvec;
   gvec = genes->GetFactors();


   // transform correctly gvec on fresult in case there are fixed parameters
   const std::vector<double> & transVec = static_cast<MultiGenFunctionFitness*>(fFitness)->Transform(gvec);
   std::copy(transVec.begin(), transVec.end(), fResult.begin() );
   fMinValue = static_cast<MultiGenFunctionFitness*>(fFitness)->Evaluate(gvec);


   if (PrintLevel() > 0) {
      if (PrintLevel() > 2) std::cout << std::endl;
          std::cout << "Finished Iteration (niter = " << niter << "  with fitness function value = " << MinValue() << std::endl;
      for (unsigned int i = 0; i < fResult.size(); ++i) {
         std::cout << " Parameter-" << i << "\t=\t" << fResult[i] << std::endl;
      }
   }

   return true;
}

double GeneticMinimizer::MinValue() const
{
   return (fFitness) ? fMinValue : 0;
}

const double *  GeneticMinimizer::X() const {
   return (fFitness) ? &fResult[0] : 0;
}

unsigned int GeneticMinimizer::NCalls() const
{
   if ( fFitness )
      return static_cast<MultiGenFunctionFitness*>(fFitness)->NCalls();
   else
      return 0;
}

unsigned int GeneticMinimizer::NDim() const
{
   if ( fFitness )
      return static_cast<MultiGenFunctionFitness*>(fFitness)->NTotal();
   else
      return 0;
}
unsigned int GeneticMinimizer::NFree() const
{
   if ( fFitness )
      return static_cast<MultiGenFunctionFitness*>(fFitness)->NDims();
   else
      return 0;
}

// Functions we don't need...
const double *  GeneticMinimizer::MinGradient() const { return 0; }
bool GeneticMinimizer::ProvidesError() const { return false; }
const double * GeneticMinimizer::Errors() const { return 0; }
double GeneticMinimizer::Edm() const { return 0; }
double GeneticMinimizer::CovMatrix(unsigned int, unsigned int) const { return 0; }

}
}
