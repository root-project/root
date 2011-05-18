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
   const ROOT::Math::IMultiGenFunction& fFunc;

public:
   MultiGenFunctionFitness(const ROOT::Math::IMultiGenFunction& function) : fFunc(function) { fNCalls = 0; }

   unsigned int NCalls() const { return fNCalls; }
   unsigned int NDims() const { return fFunc.NDim(); }

   Double_t Evaluate(const std::vector<double> & factors ) const {
      return fFunc(&factors[0]);
   }

   Double_t EstimatorFunction(std::vector<double> & factors ){
      fNCalls += 1;
      return fFunc(&factors[0]);
   }
};

GeneticMinimizerParameters::GeneticMinimizerParameters() 
{ 
   // constructor of parameters with default values
   fNsteps   = 40;
   fPopSize  =300;
   fCycles   = 3;
   fSC_steps =10;
   fSC_rate  =5;
   fSC_factor=0.95;
   fConvCrit =10.0 * ROOT::Math::MinimizerOptions::DefaultTolerance(); // default is 0.001
   if (fConvCrit <=0 ) fConvCrit = 0.001; 

}

   GeneticMinimizer::GeneticMinimizer(int ): 
      fFitness(0), 
      fParameters(GeneticMinimizerParameters() )
{

   // check with default minimizer options
   ROOT::Math::IOptions * geneticOpt = ROOT::Math::MinimizerOptions::FindDefault("Genetic");
   if (geneticOpt) { 
      ROOT::Math::MinimizerOptions opt; // create using default options
      opt.SetExtraOptions(*geneticOpt);
      this->SetOptions(opt);
   } 
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
}  

bool GeneticMinimizer::SetLimitedVariable(unsigned int , const std::string & , double , double , double lower , double upper ) 
{ 
   fRanges.push_back( new TMVA::Interval(lower,upper) );

   return true;
}

bool GeneticMinimizer::SetVariable(unsigned int, const std::string&, double value, double step) 
{
   //It does nothing! As there is no variable if it has no limits!
   Info("GeneticMinimizer::SetVariable", "Variables should be limited on a Genetic Minimizer - set automatic range to 50 times step size");
   fRanges.push_back( new TMVA::Interval(value - (50 * step), value + (50 * step)) );
   
   return true;
}

void GeneticMinimizer::SetParameters(const GeneticMinimizerParameters & params )
{
   fParameters  = params; 
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
   opt.SetMaxFunctionCalls(MaxIterations());  // this is different than nsteps
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
   ROOT::Math::IOptions * geneticOpt = opt.ExtraOptions(); 
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

   // use same of options in base class
   int maxiter = MaxIterations();
   if ( maxiter > 0 &&  maxiter < fParameters.fNsteps )    {
      Warning("GeneticMinimizer::SetOptions", "max iterations smaller than Steps - set equal to steps %d",fParameters.fNsteps); 
      SetMaxIterations(fParameters.fNsteps);
   }


}

bool GeneticMinimizer::Minimize() 
{

   if (!fFitness) {
      Error("GeneticMinimizer::Minimize","Fitness function has not been set"); 
      return false; 
   }

   TMVA::GeneticAlgorithm mg( *fFitness, fParameters.fPopSize, fRanges );

   if (PrintLevel() > 0) { 
      Info("GeneticMinimizer::Minimize","Start iterating - max iterations = %d , conv criteria (tolerance) = %10e6 ",
           MaxIterations() ,  fParameters.fConvCrit );
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

   fResult = gvec;   


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
   if ( fFitness )
      return static_cast<MultiGenFunctionFitness*>(fFitness)->Evaluate(fResult);
   else
      return 0;
}  

const double *  GeneticMinimizer::X() const { return &fResult[0]; }  

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
      return static_cast<MultiGenFunctionFitness*>(fFitness)->NDims();
   else
      return 0;
}   
unsigned int GeneticMinimizer::NFree() const 
{
   // They should be the same in this case!
   return NDim();
}   

// Functions we don't need...
const double *  GeneticMinimizer::MinGradient() const { return 0; }   
bool GeneticMinimizer::ProvidesError() const { return false; }  
const double * GeneticMinimizer::Errors() const { return 0; }
double GeneticMinimizer::Edm() const { return 0; }
double GeneticMinimizer::CovMatrix(unsigned int, unsigned int) const { return 0; }

}
}
