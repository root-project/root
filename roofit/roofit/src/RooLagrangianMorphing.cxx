/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooLagrangianMorphing                                            *
 * @(#)root/roofit:$Id$
 * Authors:                                                                  *
 *  Lydia Brenner (lbrenner@cern.ch), Carsten Burgard (cburgard@cern.ch)     *
 *  Katharina Ecker (kecker@cern.ch), Adam Kaluza      (akaluza@cern.ch)     *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/


/** \class RooLagrangianMorphing
    \ingroup Roofit
Class RooLagrangianMorphing is a implementation of the method of Effective
Lagrangian Morphing, descibed in ATL-PHYS-PUB-2015-047.
Effective Lagrangian Morphing is a method to construct a continuous signal
model in the coupling parameter space. Basic assumption is that shape and
cross section of a physical distribution is proportional to it's
squared matrix element.
The signal model is constructed by a weighted sum over N input distributions.
The calculation of the weights is based on Matrix Elements evaluated for the
different input scenarios.
The number of input files depends on the number of couplings in production
and decay vertices, and also whether the decay and production vertices
describe the same process or not.
**/

// uncomment to force UBLAS multiprecision matrices
// #define USE_UBLAS 1
// #undef USE_UBLAS

#include "Riostream.h"

#include "RooLagrangianMorphing.h"
#include "RooDataHist.h"
#include "RooHistFunc.h"
#include "RooParamHistFunc.h"
#include "RooRealSumPdf.h"
#include "RooAbsArg.h"
#include "RooAbsCollection.h"
#include "RooLinkedList.h"
#include "RooArgSet.h"
#include "RooArgList.h"
#include "RooStringVar.h"
#include "RooRealVar.h"
#include "RooFormulaVar.h"
#include "RooHistFunc.h"
#include "RooConstVar.h"
#include "RooProdPdf.h"
#include "RooProduct.h"
#include "RooFitResult.h"
#include "RooHistConstraint.h"
#include "RooUniformBinning.h"
#include "RooBinning.h"
#include "RooWorkspace.h"
#include "RooWrapperPdf.h"
#include "RooSimultaneous.h"
#include "LinearCombination.h"
#include "RooMsgService.h"
//#include "RooStats/HistFactory/PiecewiseInterpolation.h"
#include "TH1.h"
#include "TParameter.h"
#include "TFile.h"
#include "TMap.h"
#include "TFolder.h"
#include "TVirtualPad.h"
//#include "TCanvas.h"
#include "TRandom3.h"
#include "TMatrixD.h"
#include "TRegexp.h"
// stl includes
#include <map>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <cstddef>
#include <cmath>
#include <iostream>
#include <limits>
#include <type_traits>
#include <iostream>
#include <typeinfo>


ClassImp(RooLagrangianMorphing::RooLagrangianMorphFunc);

//#define _DEBUG_

///////////////////////////////////////////////////////////////////////////////
// PREPROCESSOR MAGIC /////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

// various preprocessor helpers
#define NaN std::numeric_limits<double>::quiet_NaN()
#define UNUSED(expr) do { (void)(expr); } while (0)
#define NODEBUG(arg) std::cout << arg << std::endl;
#ifdef _DEBUG_
#define DEBUG(arg) std::cout << arg << std::endl;
//<typename ... T>
//const char* ype(const T& ... args )
//{
//  return __PRETTY_FUNCTION__;
//}
////cxcoutD(Eval) << arg << std::endl
#else
#define DEBUG(arg)
#endif

bool RooLagrangianMorphing::gAllowExceptions = true;
#define ERROR(arg){                                                     \
  if(RooLagrangianMorphing::gAllowExceptions){                          \
    std::stringstream err; err << arg << std::endl; throw(std::runtime_error(err.str())); \
  } else {                                                              \
    std::cerr << arg << std::endl;                                      \
  }}
#define INFO(arg) std::cout << arg << std::endl;

///////////////////////////////////////////////////////////////////////////////
// TEMPLATE MAGIC /////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

template<typename Test, template<typename...> class Ref>
struct is_specialization : std::false_type {};

template<template<typename...> class Ref, typename... Args>
struct is_specialization<Ref<Args...>, Ref>: std::true_type {};

///////////////////////////////////////////////////////////////////////////////
// LINEAR ALGEBRA HELPERS /////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
/// retrieve the size of a square matrix

template<class MatrixT>
inline size_t size(const MatrixT& matrix);
template <> inline size_t size<TMatrixD> (const TMatrixD& mat)
{
  return mat.GetNrows();
}

////////////////////////////////////////////////////////////////////////////////
/// write a matrix to a stream

template<class MatrixT>
inline void writeMatrixToStreamT(const MatrixT& matrix, std::ostream& stream)
{
  for(size_t i=0; i<size(matrix); ++i){
    for(size_t j=0; j<size(matrix); ++j){
#ifdef USE_UBLAS
      stream << std::setprecision(RooLagrangianMorphing::SuperFloatPrecision::digits10) << matrix(i,j) << "\t";
#else
      stream << matrix(i,j) << "\t";
#endif
    }
    stream << std::endl;
  }
}

////////////////////////////////////////////////////////////////////////////////
/// write a matrix to a text file

template<class MatrixT>
inline void writeMatrixToFileT(const MatrixT& matrix, const char* fname){
  std::ofstream of(fname);
  if(!of.good()){
    ERROR("unable to read file '"<<fname<<"'!");
  }
  writeMatrixToStreamT(matrix,of);
  of.close();
}

#ifdef USE_UBLAS

// boost includes
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#include <boost/numeric/ublas/symmetric.hpp> //inc diag
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/lu.hpp>
#pragma GCC diagnostic pop

typedef boost::numeric::ublas::matrix<RooLagrangianMorphing::SuperFloat> Matrix;

////////////////////////////////////////////////////////////////////////////////
/// write a matrix

inline void printMatrix(const Matrix& mat)
{
  for(size_t i=0; i<mat.size1(); ++i){
    for(size_t j=0; j<mat.size2(); ++j){
      std::cout << std::setprecision(RooLagrangianMorphing::SuperFloatPrecision::digits10) << mat(i,j) << " ,\t";
    }
    std::cout << std::endl;
  }
}

////////////////////////////////////////////////////////////////////////////////
/// retrieve the size of a square matrix

<> inline size_t size<Matrix> (const Matrix& matrix)
{
  return matrix.size1();
}

////////////////////////////////////////////////////////////////////////////////
/// create a new diagonal matrix of size n

inline Matrix diagMatrix(size_t n)
{
  return boost::numeric::ublas::identity_matrix<RooLagrangianMorphing::SuperFloat>(n);
}

////////////////////////////////////////////////////////////////////////////////
/// convert a matrix into a TMatrixD

inline TMatrixD makeRootMatrix(const Matrix& in)
{
  size_t n = size(in);
  TMatrixD mat(n,n);
  for(size_t i=0; i<n; ++i){
    for(size_t j=0; j<n; ++j){
      mat(i,j) = (double)(in(i,j));
    }
  }
  return mat;
}

////////////////////////////////////////////////////////////////////////////////
/// convert a TMatrixD into a Matrix

inline Matrix makeSuperMatrix(const TMatrixD& in)
{
  size_t n = in.GetNrows();
  Matrix mat(n,n);
  for(size_t i=0; i<n; ++i){
    for(size_t j=0; j<n; ++j){
      mat(i,j) = in(i,j);
    }
  }
  return mat;
}

////////////////////////////////////////////////////////////////////////////////
/// calculate the inverse of a matrix, returning the condition

inline RooLagrangianMorphing::SuperFloat invertMatrix(const Matrix& matrix, Matrix& inverse){
  boost::numeric::ublas::permutation_matrix<size_t> pm(size(matrix));
  RooLagrangianMorphing::SuperFloat mnorm = norm_inf(matrix);
  Matrix lu(matrix);
  try {
    int res = lu_factorize(lu,pm);
    if( res != 0 ){
      std::stringstream ss;
      ss << "lu_factorize error: matrix is not invertible:\n";
      ::writeMatrixToStreamT(matrix,ss);
      ERROR(ss.str());
    }
    // backsubstitute to get the inverse
    lu_substitute(lu, pm, inverse);
  } catch (boost::numeric::ublas::internal_logic& error){
    ERROR("boost::numberic::ublas error: matrix is not invertible!");
  }
  RooLagrangianMorphing::SuperFloat inorm = norm_inf(inverse);
  RooLagrangianMorphing::SuperFloat condition = mnorm * inorm;
  return condition;
}
inline Matrix operator* (const Matrix&m, const Matrix& otherM){
  return prod(m,otherM);
}
#else
#include "TDecompLU.h"
typedef TMatrixD Matrix;

////////////////////////////////////////////////////////////////////////////////
/// convert a matrix into a TMatrixD

inline TMatrixD makeRootMatrix(const Matrix& in)
{
  return TMatrixD(in);
}

////////////////////////////////////////////////////////////////////////////////
/// convert a TMatrixD into a Matrix

inline Matrix makeSuperMatrix(const TMatrixD& in)
{
  return in;
}

////////////////////////////////////////////////////////////////////////////////
/// create a new diagonal matrix of size n

inline Matrix diagMatrix(size_t n)
{
  TMatrixD mat(n,n);
  mat.UnitMatrix();
  return mat;
}

////////////////////////////////////////////////////////////////////////////////
/// write a matrix

inline void printMatrix(const TMatrixD& mat)
{
  writeMatrixToStreamT(mat,std::cout);
}

////////////////////////////////////////////////////////////////////////////////
// calculate the inverse of a matrix, returning the condition

inline double invertMatrix(const Matrix& matrix, Matrix& inverse)
{
  TDecompLU lu(matrix);
  bool status = lu.Invert(inverse);
  // check if the matrix is invertible
  if(!status){
    std::cout << std::endl;
    printMatrix(matrix);
    ERROR("Error: matrix is not invertible!");
  }
  double condition = lu.GetCondition();
  const size_t n = size(inverse);
  // sanitize numeric problems
  for(size_t i= 0; i<n; ++i)
    for(size_t j=0; j<n; ++j)
      if(fabs(inverse(i,j)) < 1e-9) inverse(i,j) = 0;
  return condition;
}
#endif


////////////////////////////////////////////////////////////////////////////////
/// print the contents of a RooDataHist

void RooLagrangianMorphing::printDataHistogram(RooDataHist* hist, RooRealVar* obs){
  for(Int_t i=0; i<obs->getBins(); ++i){
    hist->get(i);
    obs->setBin(i);
    std::cout << hist->weight() << " +/- " << hist->weightSquared() << std::endl;
  }
}

/////////////////////////////////////////////////////////////////////////////////
// LOCAL FUNCTIONS AND DEFINITIONS //////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////
/// anonymous namespace to prohibit use of these functions outside the class itself
namespace
{
  ///////////////////////////////////////////////////////////////////////////////
  // HELPERS ////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////

  typedef std::map<const std::string,RooLagrangianMorphing::ParamSet > ParamMap;
  typedef std::vector<std::vector<bool> > FeynmanDiagram;
  typedef std::vector<std::vector<int> > MorphFuncPattern;
  typedef std::map<int,RooAbsReal*> FormulaList;

  ///////////////////////////////////////////////////////////////////////////////
  /// check if a std::string begins with the given character set

  inline bool begins_with(const std::string& input, const std::string& match)
  {
    return input.size() >= match.size()
      && equal(match.begin(), match.end(), input.begin());
  }
 
  ///////////////////////////////////////////////////////////////////////////////
  /// (-?-)

  inline TString makeValidName(const char* input){
    TString retval(input);
    retval.ReplaceAll("/","_");
    retval.ReplaceAll("^","");
    retval.ReplaceAll("*","X");
    retval.ReplaceAll("[","");
    retval.ReplaceAll("]","");
    return retval;
  }


  //////////////////////////////////////////////////////////////////////////////
  /// concatenate the names of objects in a collection to a single string

  template<class List>
  std::string concatNames(const List& c, const char* sep)
  {
    std::stringstream ss;
    bool first = true;
    for(auto itr = c.begin(); itr != c.end(); ++itr){
      if(!first) ss << sep;
      ss << (*itr)->GetName();
      first = false;
    }
    return ss.str();
  }
  
  ///////////////////////////////////////////////////////////////////////////////
  /// this is a workaround for the missing implicit conversion from SuperFloat<>double

  template<class A,class B> inline void assignElement(A& a,const B& b)
  {
    a = static_cast<A>(b);
  }
  ///////////////////////////////////////////////////////////////////////////////
  // read a matrix from a stream

  template<class MatrixT>
  inline MatrixT readMatrixFromStreamT(std::istream& stream)
  {
    std::vector<std::vector<RooLagrangianMorphing::SuperFloat> > matrix;
    std::vector<RooLagrangianMorphing::SuperFloat> line;
    while(!stream.eof()){
      if(stream.peek() == '\n'){
        stream.get();
        stream.peek();
        continue;
      }
      RooLagrangianMorphing::SuperFloat val;
      stream>>val;
      line.push_back(val);
      while(stream.peek() == ' ' || stream.peek() == '\t'){
        stream.get();
      }
      if(stream.peek() == '\n'){
        matrix.push_back(line);
        line.clear();
      }
    }
    MatrixT retval(matrix.size(),matrix.size());
    for(size_t i=0; i<matrix.size(); ++i){
      if(matrix[i].size() != matrix.size()){
        ERROR("matrix read from stream doesn't seem to be square!");
      }
      for(size_t j=0; j<matrix[i].size(); ++j){
        assignElement(retval(i,j),matrix[i][j]);
      }
    }
    return retval;
  }

  ///////////////////////////////////////////////////////////////////////////////
  /// read a matrix from a text file

  template<class MatrixT>
  inline MatrixT readMatrixFromFileT(const char* fname)
  {
    std::ifstream in(fname);
    if(!in.good()){
      ERROR("unable to read file '"<<fname<<"'!");
    }
    MatrixT mat = readMatrixFromStreamT<MatrixT>(in);
    in.close();
    return mat;
  }

  ///////////////////////////////////////////////////////////////////////////////
  /// convert a TH1* param hist into the corresponding ParamSet object

  template<class T>
  inline std::map<const std::string,T> readValues(TH1* h_pc)
  {
    std::map<const std::string,T> point;
    if(h_pc){
      // loop over all bins of the param_card histogram
      for(int ibx = 1; ibx <= h_pc->GetNbinsX(); ++ibx){
        // read the value of one parameter
        const std::string s_coup(h_pc->GetXaxis()->GetBinLabel(ibx));
        double coup_val = h_pc->GetBinContent(ibx);
        // add it to the map
        if(!s_coup.empty()){
          point[s_coup] = T(coup_val);
        }
      }
    }
    return point;
  }

  ///////////////////////////////////////////////////////////////////////////////
  /// retrieve a param_hist from a certain subfolder 'name' of the file

  inline TH1F* getParamHist(TDirectory* file, const std::string& name, const std::string& objkey = "param_card", bool notFoundError = true)
  {
    TFolder* f_tmp = dynamic_cast<TFolder*>(file->Get(name.c_str()));
    if(!f_tmp) ERROR("unable to retrieve folder '"<<name<<"' from file '"<<file->GetName()<<"'!");
    // retrieve the histogram param_card which should live directly in the folder
    TH1F* h_pc = dynamic_cast<TH1F*>(f_tmp->FindObject(objkey.c_str()));
    if(h_pc){
      DEBUG("found " << objkey << " for '" << name << "'");
      return h_pc;
    }
    if(notFoundError){
      ERROR("unable to retrieve " << objkey << " histogram from folder '"<<name<<"'");
    }
    return NULL;
  }

  ///////////////////////////////////////////////////////////////////////////////
  /// retrieve a ParamSet from a certain subfolder 'name' of the file

  template<class T>
  inline std::map<const std::string,T> readValues(TDirectory* file, const std::string& name, const std::string& key = "param_card",bool notFoundError=true)
  {
    TH1F* h_pc = getParamHist(file,name,key,notFoundError);
    return readValues<T>(h_pc);
  }

  ///////////////////////////////////////////////////////////////////////////////
  /// retrieve the param_hists file and return a map of the parameter values
  /// by providing a list of names, only the param_hists of those subfolders are read
  /// leaving the list empty is interpreted as meaning 'read everyting'
   
  template<class T>
  inline std::map<const std::string,std::map<const std::string,T> > readValues(TDirectory* f, const std::vector<std::string>& names, const std::string& key = "param_card",bool notFoundError = true)
  {
    std::map<const std::string,std::map<const std::string,T> > inputParameters;
    // if the list of names is empty, we assume that this means 'all'
    // loop over all folders in the file
    for(size_t i=0; i<names.size(); i++){
      const std::string name(names[i]);
      // actually read an individual param_hist
      DEBUG("reading " << key << " '" << name << "'!");
      inputParameters[name] = readValues<T>(f,name,key,notFoundError);
    }
    
    // return the map of all parameter values found for all samples
    return inputParameters;
  }
  
  ///////////////////////////////////////////////////////////////////////////////
  /// open the file and return a file pointer

  inline TDirectory* openFile(const std::string& filename)
  {
    if(filename.empty()){
      return gDirectory;
    } else {
      TFile *file= TFile::Open(filename.c_str(),"READ");
      if (!file|| !file->IsOpen()) {
        if(file) delete file;
        ERROR("could not open file '"<<filename<<"'!");
      }
      return file;
    }
  }

  ///////////////////////////////////////////////////////////////////////////////
  /// open the file and return a file pointer

  inline void closeFile(TDirectory*& d)
  {
    TFile* f = dynamic_cast<TFile*>(d);
    if(f){
      f->Close();
      delete f;
      d=NULL;
    }
  }

  ///////////////////////////////////////////////////////////////////////////////
  /// extract the operators from a single coupling
  template<class T2> inline void extractServers(const RooAbsArg& coupling, T2& operators)
  {
    int nservers = 0;
    for (const auto server: coupling.servers()) {
      extractServers(*server, operators);
      nservers++;
    }
    if(nservers==0){operators.add(coupling);}
  }

  ///////////////////////////////////////////////////////////////////////////////
  /// extract the operators from a list of couplings
  
  template< class T1, class T2, typename std::enable_if<!is_specialization<T1,std::vector>::value,T1>::type* = nullptr>
  inline void extractOperators(const T1& couplings, T2& operators)
  {
    DEBUG("extracting operators from "<<couplings.getSize()<<" couplings");
    RooAbsArg* obj;
    for(auto itr = couplings.begin(); itr != couplings.end(); ++itr){
      obj = *itr;
      extractServers(*obj,operators);
    }
  }
  
  ///////////////////////////////////////////////////////////////////////////////
  /// extract the operators from a list of vertices

  template< class T1, class T2, typename std::enable_if<is_specialization<T1,std::vector>::value,T1>::type* = nullptr>
  inline void extractOperators(const T1& vec, T2& operators)
  {
    for(const auto& v:vec){
      extractOperators(v,operators);
    }
  }

  ///////////////////////////////////////////////////////////////////////////////
  /// extract the couplings from a given set and copy them to a new one

  template< class T1, class T2 >
  inline void extractCouplings(const T1& inCouplings, T2& outCouplings)
  {
    RooAbsArg* obj;
    for(auto itr = inCouplings.begin(); itr != inCouplings.end(); ++itr){
      obj = *itr;
      if(!outCouplings.find(obj->GetName())){
        DEBUG("adding parameter "<< obj->GetName());
        outCouplings.add(*obj);
      }
    }
  }

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
  /// set parameter values first set all values to defaultVal (if value not
  /// present in param_card then it should be 0)

  inline bool setParam(RooRealVar* p, double val, bool force)
  {
    DEBUG("setparam for "<<p->GetName()<<" to "<<val);
    bool ok = true;
    if(val > p->getMax()){
      if(force){
        p->setMax(val);
      } else {
        std::cerr << "ERROR: parameter " << p->GetName() << " out of bounds: " << val << " > " << p->getMax() << std::endl;
        ok=false;
      }
    } else if(val < p->getMin()){
      if(force){
        p->setMin(val);
      } else {
        std::cerr << "ERROR: parameter " << p->GetName() << " out of bounds: " << val << " < " << p->getMin() << std::endl;
        ok=false;
      }
    }
    if(ok) p->setVal(val);
    return ok;
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// set parameter values first set all values to defaultVal (if value not
  /// present in param_card then it should be 0)

  template< class T1, class T2>
  inline bool setParams(const T2& args,T1 val)
  {
    for (auto itr = args.begin(); itr != args.end(); ++itr){
      RooRealVar* param = dynamic_cast<RooRealVar*>(*itr);
      if(!param) continue;
      setParam(param,val,true);
    }
    return true;
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// set parameter values first set all values to defaultVal (if value not
  /// present in param_card then it should be 0)

  template< class T1, class T2>
  inline bool setParams(const std::map<const std::string,T1>& point,const T2& args,bool force=false,T1 defaultVal=0)
  {
    bool ok = true;    
    for(auto itr = args.begin(); itr != args.end(); ++itr){
      RooRealVar* param = dynamic_cast<RooRealVar*>(*itr);
      if(!param || param->isConstant()) continue;
      ok = setParam(param,defaultVal,force) && ok;
    }
    // set all parameters to the values in the param_card histogram
    for(auto paramit=point.begin(); paramit!=point.end(); ++paramit){
      // loop over all the parameters
      const std::string param(paramit->first);
      // retrieve them from the map
      RooRealVar* p = dynamic_cast<RooRealVar*>(args.find(param.c_str()));
      if(!p) continue;
      // set them to their nominal value
      ok = setParam(p,paramit->second,force) && ok;
    }
    return ok;
  }
 
  ////////////////////////////////////////////////////////////////////////////////
  /// set parameter values first set all values to defaultVal (if value not
  /// present in param_card then it should be 0)
 
  template< class T>
  inline bool setParams(TH1* hist,const T& args,bool force=false)
  {
    bool ok = true;

    for(auto itr = args.begin(); itr != args.end(); ++itr){
      RooRealVar* param = dynamic_cast<RooRealVar*>(*itr);
      if(!param) continue;
      ok = setParam(param,0.,force) && ok;
    }

    // set all parameters to the values in the param_card histogram
    TAxis* ax = hist->GetXaxis();
    for(int i=1; i<=ax->GetNbins(); ++i){
      // loop over all the parameters
      RooRealVar* p = dynamic_cast<RooRealVar*>(args.find(ax->GetBinLabel(i)));
      if(!p) continue;
      // set them to their nominal value
      ok = setParam(p,hist->GetBinContent(i),force) && ok;
    }
    return ok;
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// create a set of parameters

  template< class T>
  inline RooLagrangianMorphing::ParamSet getParams(const T& parameters)
  {
    RooLagrangianMorphing::ParamSet retval;
    for(auto itr = parameters.begin(); itr != parameters.end(); ++itr){
      RooRealVar* param = dynamic_cast<RooRealVar*>(*itr);
      if(!param) continue;
      retval[param->GetName()] = param->getVal();
    }
    return retval;
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// collect the histograms from the input file and convert them to RooFit objects

  void collectHistograms(const char* name,TDirectory* file, std::map<std::string,int>& list_hf, RooArgList& physics, RooRealVar& var, const std::string& varname, const std::string& /*basefolder*/, const RooLagrangianMorphing::ParamMap& inputParameters)
  {
    DEBUG("building list of histogram functions");
    bool binningOK = false;
    for(auto sampleit=inputParameters.begin(); sampleit!=inputParameters.end(); ++sampleit){
      const std::string sample(sampleit->first);
      TFolder* folder = dynamic_cast<TFolder*>(file->Get(sample.c_str()));
      if(!folder){
        ERROR("Error: unable to access data from folder '" << sample << "'!");
        continue;
      }
      TH1* hist = dynamic_cast<TH1*>(folder->FindObject(varname.c_str()));
      if(!hist){
        std::stringstream errstr;
        errstr << "Error: unable to retrieve histogram '" << varname << "' from folder '" << sample << "'. contents are:";
        TIter next(folder->GetListOfFolders()->begin());
        TFolder* f;
        while ((f = (TFolder*)next())) {
          errstr << " " << f->GetName();
        }
        ERROR(errstr.str());
      }
      
      auto it = list_hf.find(sample);
      if(it != list_hf.end()){
        RooHistFunc* hf = (RooHistFunc*)(physics.at(it->second));
        hf->setValueDirty();
        RooDataHist* dh = &(hf->dataHist());
       // RooLagrangianMorphing::setDataHistogram(hist,&var,dh);
       // RooArgSet vars;
       // vars.add(var);
       // dh->importTH1(vars,*hist,1.,kFALSE);
      } else {
        if(!binningOK){
          int n = hist->GetNbinsX();
          std::vector<double> bins;
          for(int i =1 ; i < n+1 ; ++i){
            bins.push_back(hist->GetBinLowEdge(i));
          }
          bins.push_back(hist->GetBinLowEdge(n)+hist->GetBinWidth(n));
          var.setBinning(RooBinning(n,&(bins[0])));
        }
        
        // generate the mean value
        TString histname = makeValidName(TString::Format("dh_%s_%s",sample.c_str(),name));
        TString funcname = makeValidName(TString::Format("phys_%s_%s",sample.c_str(),name));
        RooArgSet vars;
        vars.add(var);

        RooDataHist* dh = new RooDataHist(histname,histname,vars,hist);
        // add it to the list
        RooHistFunc* hf = new RooHistFunc(funcname,funcname,var,*dh);
        int idx = physics.getSize();
        list_hf[sample] = idx;
        physics.add(*hf);
        assert(hf = (RooHistFunc*)physics.at(idx));
      }
      DEBUG("found histogram " << hist->GetName() << " with integral " << hist->Integral());
    }
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// collect the RooAbsReal objects from the input directory

  void collectRooAbsReal(const char* /*name*/,TDirectory* file, std::map<std::string,int>& list_hf, RooArgList& physics, const std::string& varname, const RooLagrangianMorphing::ParamMap& inputParameters)
  {
    DEBUG("building list of RooAbsReal objects");
    for(auto sampleit=inputParameters.begin(); sampleit!=inputParameters.end(); ++sampleit){
      const std::string sample(sampleit->first);
      TFolder* folder = dynamic_cast<TFolder*>(file->Get(sample.c_str()));
      if(!folder){
        ERROR("Error: unable to access data from folder '" << sample << "'!");
        continue;
      }

      RooAbsReal* obj = dynamic_cast<RooAbsReal*>(folder->FindObject(varname.c_str()));
      if(!obj){
        std::stringstream errstr;
        errstr << "Error: unable to retrieve RooAbsArg '" << varname << "' from folder '" << sample << "'. contents are:";
        TIter next(folder->GetListOfFolders()->begin());
        TFolder* f;
        while ((f = (TFolder*)next())) {
          errstr << " " << f->GetName();
        }
        ERROR(errstr.str());
      }
      auto it = list_hf.find(sample);
      if(it == list_hf.end()){
        int idx = physics.getSize();
        list_hf[sample] = idx;
        physics.add(*obj);
        assert(obj == physics.at(idx));
      }
    }
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// collect the TParameter objects from the input file and convert them to RooFit objects

  template< class T>
  void collectCrosssections(const char* name, TDirectory* file, std::map<std::string,int>& list_xs, RooArgList& physics, const std::string& varname, const std::string& /*basefolder*/, const RooLagrangianMorphing::ParamMap& inputParameters)
  {
    DEBUG("building list of histogram functions");
    for(auto sampleit=inputParameters.begin(); sampleit!=inputParameters.end(); ++sampleit){
      const std::string sample(sampleit->first);
      TFolder* folder = dynamic_cast<TFolder*>(file->Get(sample.c_str()));
      if(!folder) ERROR("unable to access data from folder '" << sample << "'!");
      TObject* obj = folder->FindObject(varname.c_str());
      TParameter<T>* xsection = NULL;
      TParameter<T>* error = NULL;
      TParameter<T>* p = dynamic_cast<TParameter<T>*>(obj);
      if(p){
        xsection = p;
      }
      TPair* pair = dynamic_cast<TPair*>(obj);
      if(pair){
        xsection = dynamic_cast<TParameter<T>*>(pair->Key());
        error = dynamic_cast<TParameter<T>*>(pair->Value());
      }
      if(!xsection){
        std::stringstream errstr;
        errstr << "Error: unable to retrieve cross section '" << varname << "' from folder '" << sample << "'. contents are:";
                TIter next(folder->GetListOfFolders()->begin());
                TFolder* f;
                while ((f = (TFolder*)next())) {
                    errstr << " " << f->GetName();
                }
        ERROR(errstr.str());
      }

      auto it = list_xs.find(sample.c_str());
      RooRealVar* xs;
      if(it != list_xs.end()){
        xs = (RooRealVar*)(physics.at(it->second));
        xs->setVal(xsection->GetVal());
      } else {
        TString objname = TString::Format("phys_%s_%s",name,sample.c_str());
        xs = new RooRealVar(objname,objname,xsection->GetVal());
        xs->setConstant(true);
        int idx = physics.getSize();
        list_xs[sample] = idx;
        physics.add(*xs);
        assert(physics.at(idx) == xs);
      }
      if(error) xs->setError(error->GetVal());
    }
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// collect the TPair<TParameter,TParameter> objects from the input file and
  /// convert them to RooFit objects

  void collectCrosssectionsTPair(const char* name, TDirectory* file, std::map<std::string,int>& list_xs, RooArgList& physics, const std::string& varname, const std::string& basefolder, const RooLagrangianMorphing::ParamMap& inputParameters)
  {
    TFolder* folder = (dynamic_cast<TFolder*>(file->Get(basefolder.c_str())));
    TPair* pair = dynamic_cast<TPair*>(folder->FindObject(varname.c_str()));
    TParameter<double>* xsec_double = dynamic_cast<TParameter<double>*>(pair->Key());
    if(xsec_double){
      collectCrosssections<double>(name, file, list_xs, physics, varname, basefolder, inputParameters);
    } else {
      TParameter<float>* xsec_float = dynamic_cast<TParameter<float>*>(pair->Key());
      if(xsec_float) {
        collectCrosssections<float>(name, file, list_xs, physics, varname, basefolder, inputParameters);
      } else {
        ERROR("cannot morph objects of class 'TPair' if parameter is not double or float!");
      }
    }
  }
  
  ///////////////////////////////////////////////////////////////////////////////
  // FORMULA CALCULATION ////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////


  ////////////////////////////////////////////////////////////////////////////////
  /// recursive function to determine polynomials

  void collectPolynomialsHelper(const FeynmanDiagram& diagram, MorphFuncPattern& morphfunc, std::vector<int>& term, int vertexid, bool first)
  {
    if(vertexid > 0){
      for(size_t i=0; i<diagram[vertexid-1].size(); ++i){
        if(!diagram[vertexid-1][i]) continue;
        std::vector<int> newterm(term);
        newterm[i]++;
        if(first){
          ::collectPolynomialsHelper(diagram,morphfunc,newterm,vertexid,false);
        } else {
          ::collectPolynomialsHelper(diagram,morphfunc,newterm,vertexid-1,true);
        }
      }
    } else {
      bool found = false;
      for(size_t i=0; i<morphfunc.size(); ++i){
        bool thisfound = true;
        for(size_t j=0; j<morphfunc[i].size(); ++j){
          if(morphfunc[i][j] != term[j]){
            thisfound=false;
            break;
          }
        }
        if(thisfound) {
          found = true;
          break;
        }
      }
      if(!found){
        morphfunc.push_back(term);
      }
    }
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// calculate the morphing function pattern based on a vertex map

  void collectPolynomials(MorphFuncPattern& morphfunc, const FeynmanDiagram& diagram)
  {
    int nvtx(diagram.size());
    std::vector<int> term(diagram[0].size(),0);

    ::collectPolynomialsHelper(diagram,morphfunc,term,nvtx,true);
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// build a vertex map based on vertices and couplings appearing

  template< class List>
  inline void fillFeynmanDiagram(FeynmanDiagram& diagram, const std::vector<List*>& vertices,RooArgList& couplings)
  {
    const int ncouplings = couplings.getSize();
    DEBUG("ncouplings " << ncouplings);
    for(size_t i=0; i<vertices.size(); ++i){
      const List* vertex = vertices[i];
      std::vector<bool> vertexCouplings(ncouplings,false);
      int idx = -1;
      RooAbsReal* coupling;
      for(auto citr = couplings.begin(); citr != couplings.end(); ++citr){
        coupling = dynamic_cast<RooAbsReal*>(*citr);
        coupling->Print();
        idx++;
        if(!coupling){
          ERROR("encountered invalid list of couplings in vertex!");
        }
        if(vertex->find(coupling->GetName())){
          vertexCouplings[idx] = true;
        }
      }
      diagram.push_back(vertexCouplings);
    }
  }


  ////////////////////////////////////////////////////////////////////////////////
  /// fill the matrix of coefficients

  template< class MatrixT, class T1, class T2>
  inline MatrixT buildMatrixT(const RooLagrangianMorphing::ParamMap& inputParameters, const FormulaList& formulas, const T1& args, const RooLagrangianMorphing::FlagMap& flagValues, const T2& flags)
  {
    const size_t dim = inputParameters.size();
    MatrixT matrix(dim,dim);
    int row = 0;
    for(auto sampleit=inputParameters.begin(); sampleit!=inputParameters.end(); ++sampleit){
      const std::string sample(sampleit->first);
      // set all vars to value stored in input file
      if(!setParams<double>(sampleit->second,args,true,0)){
        ERROR("unable to set parameters for sample "<<sample<<"!");
      }
      auto flagit = flagValues.find(sample);
      if(flagit != flagValues.end() && !setParams<int>(flagit->second,flags,true,1)){
        ERROR("unable to set parameters for sample "<<sample<<"!");
      }
      // loop over all the formulas
      int col = 0;
      for(auto formulait=formulas.begin(); formulait!=formulas.end(); ++formulait){
        RooAbsReal* formula = formulait->second;
        if(!formula){
          ERROR("Error: invalid formula encountered!");
        }
        matrix(row,col) = formula->getVal();
        DEBUG(formula->getVal() << " = " << formula->GetTitle() << " for " << sample);
        col++;
      }
      row++;
    }
    return matrix;
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// check if the matrix is square

  inline void checkMatrix(const RooLagrangianMorphing::ParamMap& inputParameters, const FormulaList& formulas)
  {
    if(inputParameters.size() != formulas.size()){
      std::stringstream ss;
      ss << "ERROR: matrix is not square, consistency check failed: " <<
        inputParameters.size() << " samples, " <<
        formulas.size() << " expressions:" << std::endl;
      ss << "formulas: " << std::endl;
      for(auto formula : formulas){
        ss << formula.second->GetTitle() << std::endl;
      }
      ss << "samples: " << std::endl;
      for(auto sample : inputParameters){
        ss << sample.first << std::endl;
      }
      ERROR(ss.str());
    }
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// check if the entries in the inverted matrix are sensible

  inline void inverseSanity(const Matrix& matrix, const Matrix& inverse, double& unityDeviation, double& largestWeight)
  {
    DEBUG("multiplying for sanity check");
    Matrix unity(inverse * matrix);
    DEBUG("matrix operations done");

    unityDeviation = 0.;
    largestWeight = 0.;
    const size_t dim = size(unity);
    for(size_t i=0; i<dim; ++i){
      for(size_t j=0; j<dim; ++j){
        if(inverse(i,j) > largestWeight){
          largestWeight = (double)inverse(i,j);
        }
        if(fabs(unity(i,j) - (i==j)) > unityDeviation){
          unityDeviation = fabs((double)unity(i,j)) - (i==j);
        }
      }
    }
    DEBUG("found deviation of " << unityDeviation << ", largest weight is " << largestWeight << ".");
  }
 
  ////////////////////////////////////////////////////////////////////////////////
  /// check for name conflicts between the input samples and an argument set
  template< class List>
  inline void checkNameConflict(const RooLagrangianMorphing::ParamMap& inputParameters, List& args)
  {
    for(auto sampleit=inputParameters.begin(); sampleit!=inputParameters.end(); ++sampleit){
      const std::string sample(sampleit->first);
      RooAbsArg* arg = args.find(sample.c_str());
      if(arg){
        ERROR("detected name conflict: cannot use sample '" << sample << "' - a parameter with the same name of type '" << arg->ClassName() << "' is present in set '" << args.GetName() << "'!");
      }
    }
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// build the formulas corresponding to the given set of input files and
  ///  the physics process

  template< class List>
  inline FormulaList buildFormulas(const char* mfname,const RooLagrangianMorphing::ParamMap& inputParameters, const RooLagrangianMorphing::FlagMap& inputFlags, const MorphFuncPattern& morphfunc, const RooArgList& couplings, const List& flags, const std::vector<List*>& nonInterfering)
  {
    // example vbf hww:
    //                        Operators kSM,  kHww, kAww, kHdwR,kHzz, kAzz
    // std::vector<bool> vertexProd  = {true, true, true, true, true, true };
    // std::vector<bool> vertexDecay = {true, true, true, true, false,false};
    // diagram.push_back(vertexProd);
    // diagram.push_back(vertexDecay);

    const int ncouplings = couplings.getSize();
    std::vector<bool> couplingsZero(ncouplings,true);
    std::map<TString,bool> flagsZero;

    RooArgList operators;
    extractOperators(couplings,operators);
    size_t nOps = operators.getSize();

#ifdef _DEBUG_
    operators.Print("v");
#endif

    for(auto sampleit=inputParameters.begin(); sampleit!=inputParameters.end(); ++sampleit){
      const std::string sample(sampleit->first);
      if(!setParams(sampleit->second,operators,true)){
        ERROR("unable to set parameters for sample '"<<sample<<"'!");
      }

      if((int)nOps!=(operators.getSize())){
        ERROR("internal error, number of operators inconsistent!");
      }

      RooAbsReal* obj0;
      int idx = 0;

      for(auto itr1 = couplings.begin(); itr1 != couplings.end(); ++itr1){
          obj0 = dynamic_cast<RooAbsReal*>(*itr1);
          if(obj0->getVal() != 0){
          DEBUG(obj0->GetName() << " is non-zero for sample " << sample << " (idx=" << idx << ")!");
          couplingsZero[idx] = false;
        } else {
          DEBUG(obj0->GetName() << " is zero for sample " << sample << " (idx=" << idx << ")!");
        }
        idx++;
      }
    }


    RooAbsReal* obj1;
    for(auto itr2 = flags.begin(); itr2 != flags.end(); ++itr2){
      obj1 = dynamic_cast<RooAbsReal*>(*itr2);
      int nZero = 0;
      int nNonZero = 0;
      for(auto sampleit=inputFlags.begin(); sampleit!=inputFlags.end(); ++sampleit){
        const auto& flag = sampleit->second.find(obj1->GetName());
        if(flag != sampleit->second.end()){
          if(flag->second == 0.) nZero++;
          else nNonZero++;
          //          std::cout << "flag found " << obj->GetName() << ", value = " << flag->second <<  std::endl;
        } else {
          //          std::cout << "flag not found " << obj->GetName() << std::endl;
        }
      }
      if(nZero > 0 && nNonZero == 0) flagsZero[obj1->GetName()] = true;
      else flagsZero[obj1->GetName()] = false;
    }

    #ifdef _DEBUG_
    {
      int idx = 0;
      RooAbsReal* obj2;
      for(auto itr = couplings.begin(); itr != couplings.end(); ++itr){
        obj2 = dynamic_cast<RooAbsReal*>(*itr);
        if(couplingsZero[idx]){
          DEBUG(obj2->GetName() << " is zero (idx=" << idx << ")");
        } else {
          DEBUG(obj2->GetName() << " is non-zero (idx=" << idx << ")");
        }
        idx++;
      }
    }
    #endif

    FormulaList formulas;
    for(size_t i=0; i<morphfunc.size(); ++i){
      RooArgList ss;
      bool isZero = false;
      std::string reason;
      // check if this is a blacklisted interference term
      for(const auto&group:nonInterfering){
        int nInterferingOperators = 0;
        for(size_t j=0; j<morphfunc[i].size(); ++j){
          if(morphfunc[i][j]%2==0) continue; // even exponents are not interference terms
          if(group->find(couplings.at(j)->GetName())){ // if the coupling is part of a "pairwise non-interfering group"
            nInterferingOperators++;
          }
        }
        if(nInterferingOperators>1){
          isZero=true;
          reason = "blacklisted interference term!";
        }
      }
      int nNP = 0;
      if(!isZero){
        // prepare the term
        for(size_t j=0; j<morphfunc[i].size(); ++j){
          const int exponent = morphfunc[i][j];
          if(exponent == 0) continue;
          RooAbsReal* coupling = dynamic_cast<RooAbsReal*>(couplings.at(j));
          for(int k=0; k<exponent; ++k){
            ss.add(*coupling);
            if(coupling->getAttribute("NP")){
              nNP++;
            }
          }
          std::string cname(coupling->GetName());
          if(coupling->getAttribute("LO") && exponent > 1){
            isZero = true;
            reason = "coupling "+cname+" was listed as leading-order-only";
          }
          // mark the term as zero if any of the couplings are zero
          if(!isZero && couplingsZero[j]){
            isZero = true;
            reason = "coupling "+cname+" is zero!";
          }
        }
      }
      // check and apply flags
      bool removedByFlag = false;
      RooAbsReal* obj;

      for(auto itr = flags.begin(); itr != flags.end(); ++itr){
        obj = dynamic_cast<RooAbsReal*>(*itr);
        if(!obj) continue;
        TString sval(obj->getStringAttribute("NP"));
        int val = atoi(sval);
        if(val == nNP){
          if(flagsZero.find(obj->GetName()) != flagsZero.end() && flagsZero.at(obj->GetName())){
            removedByFlag = true;
            reason = TString::Format("flag %s is zero",obj->GetName());
          }
          ss.add(*obj);
        }
      }


      // create and add the formula
      if(!isZero && !removedByFlag){
        // build the name
        const TString name = TString::Format("%s_pol%lu",mfname,i);
        RooProduct* prod = new RooProduct(name.Data(),::concatNames(ss," * ").c_str(),ss);
        formulas[i] = prod;
        DEBUG("creating formula " << name << ": " << prod->GetTitle());
      } else {
        // print a message and continue withou,mt doing anything
        DEBUG("killing formula " << ::concatNames(ss," * ") << " because " << reason);
      }
    }
    return formulas;
  }
 
  ////////////////////////////////////////////////////////////////////////////////
  /// create the weight formulas required for the morphing

  template<class T>
  inline FormulaList createFormulas(const char* name,const RooLagrangianMorphing::ParamMap& inputs, const RooLagrangianMorphing::FlagMap& inputFlags, const std::vector<std::vector<T*> >& diagrams, RooArgList& couplings, const T& flags, const std::vector<T*>& nonInterfering)
  {
    MorphFuncPattern morphfuncpattern;
    for(const auto& vertices:diagrams){
      FeynmanDiagram d;
      DEBUG("building vertex map");
      ::fillFeynmanDiagram<T>(d,vertices,couplings);
      DEBUG("collecting polynomials for diagram of size " << d.size());
      ::collectPolynomials(morphfuncpattern,d);
//      if (d.size() != 0) ::collectPolynomials(morphfuncpattern,d);
    }
    DEBUG("building formulas");
    FormulaList retval = buildFormulas(name,inputs,inputFlags,morphfuncpattern,couplings,flags,nonInterfering);
    if(retval.size() == 0){
      ERROR("no formulas are non-zero, check if any if your couplings is floating and missing from your param_cards!");
    }
    DEBUG("checking matrix consistency");
    checkMatrix(inputs,retval);
    return retval;
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// (-?-)
  //
  template<class T1>
  inline void buildSampleWeights(T1& weights, const char* fname,const RooLagrangianMorphing::ParamMap& inputParameters, FormulaList& formulas, const Matrix& inverse)
  {
    int sampleidx = 0;
    //printMatrix(inverse);
    for(auto sampleit=inputParameters.begin(); sampleit!=inputParameters.end(); ++sampleit){
      const std::string sample(sampleit->first);
      std::stringstream title;
      DEBUG("building formula for sample '" << sample << "'");
      TString name_full(makeValidName(sample.c_str()));
      if(fname){
        name_full.Append("_");
        name_full.Append(fname);
        name_full.Prepend("w_");
      }
      int formulaidx = 0;
      // build the formula with the correct normalization
//#ifdef USE_MULTIPRECISION_LC
      RooLagrangianMorphing::LinearCombination* sampleformula = new RooLagrangianMorphing::LinearCombination(name_full.Data());
      for(auto formulait=formulas.begin(); formulait!=formulas.end(); ++formulait){
        const RooLagrangianMorphing::SuperFloat val(inverse(formulaidx,sampleidx));
        RooAbsReal* formula = formulait->second;
        sampleformula->add(val,formula);
        formulaidx++;
        title <<" + "<<double(val) << "*(" << formula->GetTitle()<<")";
      }
//#else
//      RooArgList numbers;
//      RooArgList formulalist;
//      for(auto formulait=formulas.begin(); formulait!=formulas.end(); ++formulait){
//        TString idx = TString::Format("c_%d_%d",(int)sampleidx,(int)formulaidx);
//        double val = double(inverse(formulaidx,sampleidx));
//        RooConstVar* constVal = new RooConstVar(idx,idx,val);
//        RooAbsReal* formula = formulait->second;
//        numbers.add(*constVal);
//        formulalist.add(*formula);
//        formulaidx++;
//        title <<" + "<<double(val) << "*(" << formula->GetTitle()<<")";
//      }
//      RooRealSumFunc* sampleformula = new RooRealSumFunc(name_full.Data(),title.str().c_str(),numbers,formulalist);
//#endif
      weights.add(*sampleformula);
      sampleidx++;
    }
    DEBUG("done building sample weights");
  }

  inline std::map<std::string,std::string> buildSampleWeightStrings(const RooLagrangianMorphing::ParamMap& inputParameters, FormulaList& formulas, const Matrix& inverse){
    int sampleidx = 0;
    std::map<std::string,std::string> weights;
    for(auto sampleit=inputParameters.begin(); sampleit!=inputParameters.end(); ++sampleit){
      const std::string sample(sampleit->first);
      std::stringstream str;
      DEBUG("building formula for sample '" << sample << "'");
      int formulaidx = 0;
      // build the formula with the correct normalization
      for(auto formulait=formulas.begin(); formulait!=formulas.end(); ++formulait){
        double val(inverse(formulaidx,sampleidx));
        RooAbsReal* formula = formulait->second;
        if(val != 0.){
          if(formulaidx > 0 && val > 0) str << " + ";
          str << val << "*(" << formula->GetTitle()<<")";
        }
        formulaidx++;
      }
      weights[sample] = str.str();
      sampleidx++;
    }
    return weights;
  }
}

///////////////////////////////////////////////////////////////////////////////
// CacheElem magic ////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

class RooLagrangianMorphing::RooLagrangianMorphFunc::CacheElem : public RooAbsCacheElement {
public:
  
  RooRealSumFunc* _sumFunc = 0 ;
  RooArgList _couplings;
  
  FormulaList _formulas;
  RooArgList _weights;
  
  Matrix _matrix;
  Matrix _inverse;
  double _condition;
  
  CacheElem(){ };
  virtual void operModeHook(RooAbsArg::OperMode) override {};


  //////////////////////////////////////////////////////////////////////////////
  /// retrieve the list of contained args

  virtual RooArgList containedArgs(Action)
  {
    RooArgList args(*_sumFunc);
    args.add(_weights);
    args.add(_couplings);
    for(auto it:_formulas){
      args.add(*(it.second));
    }
    return args;
  }

  //////////////////////////////////////////////////////////////////////////////
  // default destructor

  virtual ~CacheElem()
  {
    // the sumfunc owns all its contents
    delete _sumFunc;
    for(auto it:_formulas){
      delete it.second;
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  /// a factory function as a wrapper around the constructors of RooRealSum*

  static inline RooRealSumFunc* makeSum(const char* name, const char* title, const RooArgList &funcList, const RooArgList &coefList);

  //////////////////////////////////////////////////////////////////////////////
  /// create the basic objects required for the morphing

  inline void createComponents(const RooLagrangianMorphing::ParamMap& inputParameters,const RooLagrangianMorphing::FlagMap& inputFlags,const char* funcname,const std::vector<std::vector<RooListProxy*> >& diagrams,const std::vector<RooListProxy*>& nonInterfering, const RooListProxy& flags)
  {
    RooArgList operators;
    DEBUG("collecting couplings");
    for(const auto& diagram:diagrams){
      for(const auto& vertex : diagram){
        extractCouplings(*vertex,this->_couplings);
      }
    }
    extractOperators(this->_couplings,operators);
    this->_formulas = ::createFormulas(funcname,inputParameters,inputFlags,diagrams,this->_couplings,flags,nonInterfering);
  }

  //////////////////////////////////////////////////////////////////////////////
  /// build and invert the morphing matrix
  template<class List>
  inline void buildMatrix(const RooLagrangianMorphing::ParamMap& inputParameters,const RooLagrangianMorphing::FlagMap& inputFlags,const List& flags)
  {
    RooArgList operators;
    extractOperators(this->_couplings,operators);
    DEBUG("filling matrix");
    Matrix matrix(buildMatrixT<Matrix>(inputParameters,this->_formulas,operators,inputFlags,flags));
    if(size(matrix) < 1 ){
      ERROR("input matrix is empty, please provide suitable input samples!");
    }
    Matrix inverse(diagMatrix(size(matrix)));
#ifdef _DEBUG_
    printMatrix(matrix);
#endif

    double condition = (double)(invertMatrix(matrix,inverse));
    DEBUG("inverse matrix (condition " << condition << ") is:");
#ifdef _DEBUG_
    std::cout << "Condition of the matrix :" << condition << std::endl;
    printMatrix(inverse);
#endif
    
    double unityDeviation, largestWeight;
    inverseSanity(matrix, inverse, unityDeviation, largestWeight);
    bool weightwarning(largestWeight > 10e7 ? true : false);
    bool unitywarning(unityDeviation > 10e-6 ? true : false);

    // if(unitywarning || weightwarning){
    if(false){
      if(unitywarning){
        std::cerr << "Warning: The matrix inversion seems to be unstable. This can be a result to input samples that are not sufficiently different to provide any morphing power." << std::endl;
      } else if(weightwarning){
        std::cerr << "Warning: Some weights are excessively large. This can be a result to input samples that are not sufficiently different to provide any morphing power." << std::endl;
      }
      std::cerr << "         Please consider the couplings encoded in your samples to cross-check:" << std::endl;
      for(auto sampleit=inputParameters.begin(); sampleit!=inputParameters.end(); ++sampleit){
        const std::string sample(sampleit->first);
        std::cerr << "         " << sample << ": ";
        // set all vars to value stored in input file
        setParams(sampleit->second,operators,true);
        bool first = true;
        RooAbsReal* obj;

        for(auto itr = this->_couplings.begin(); itr != this->_couplings.end(); ++itr){
          obj = dynamic_cast<RooAbsReal*>(*itr);
          if(!first) std::cerr << ", ";
          std::cerr << obj->GetName() << "=" << obj->getVal();
          first = false;
        }
        std::cerr << std::endl;
      }
    }
#ifndef USE_UBLAS
    this->_matrix.ResizeTo(matrix.GetNrows(),matrix.GetNrows());
    this->_inverse.ResizeTo(matrix.GetNrows(),matrix.GetNrows());
#endif
    this->_matrix  = matrix;
    this->_inverse = inverse;
    this->_condition = condition;
  }

////////////////////////////////////////////////////////////////////////////////
/// build the final morphing function

  inline void buildMorphingFunction(const char* name,const RooLagrangianMorphing::ParamMap& inputParameters,const std::map<std::string,int>& storage, const RooArgList& physics,
                                    bool allowNegativeYields,RooRealVar* observable,RooRealVar* binWidth)
  {
    if(!binWidth){
      ERROR("invalid bin width given!");
      return;
    }
    if(!observable){
      ERROR("invalid observable given!");
      return;
    }
    
    RooArgList operators;
    extractOperators(this->_couplings,operators);

    // retrieve the weights
    DEBUG("creating Sample Weights");
    ::buildSampleWeights(this->_weights,name,inputParameters,this->_formulas,this->_inverse);

    DEBUG("creating RooProducts");
    // build the products of element and weight for each sample
    size_t i=0;
    RooArgList sumElements;
    RooArgList scaleElements;
    for(auto sampleit=inputParameters.begin(); sampleit!=inputParameters.end(); ++sampleit){
      // for now, we assume all the lists are nicely ordered
      TString prodname (makeValidName(sampleit->first.c_str()));
      DEBUG("   for " << prodname);
      RooAbsReal* obj = (RooAbsReal*)(physics.at(storage.at(prodname.Data())));
      if(!obj) ERROR("unable to access physics object for " << prodname);
      RooAbsReal* weight = (RooAbsReal*)(this->_weights.at(i));
      if(!weight) ERROR("unable to access weight object for " << prodname);
      prodname.Append("_");
      prodname.Append(name);
      RooArgList prodElems(*weight,*obj);
      RooProduct* prod = new RooProduct(prodname,prodname,prodElems);
      if(!allowNegativeYields){
        TString maxname(prodname);
        maxname.Append("_max0");
        RooArgSet prodset(*prod);
       RooFormulaVar* max = new RooFormulaVar(maxname,"max(0,"+prodname+")",prodset);
        sumElements.add(*max);
      } else {
        sumElements.add(*prod);
          }
      scaleElements.add(*(binWidth));
      i++;
    }

    // put everything together
    DEBUG("creating RooRealSumPdf");
    // as RooRealSum* does not have a common constructor format, we need to provide a factory wrapper
    RooRealSumFunc* morphfunc = new RooRealSumFunc(TString::Format("%s_morphfunc",name),name,sumElements,scaleElements);

    DEBUG("ownership handling");
    DEBUG("... adding observable");
    if(!observable) ERROR("unable to access observable");
    morphfunc->addServer(*observable);
    DEBUG("... adding bin width");
    if(!binWidth) ERROR("unable to access bin width");
    morphfunc->addServer(*binWidth);
    DEBUG("... adding params");
    if(operators.getSize() < 1) ERROR("no operators listed");
    morphfunc->addServerList(operators);
    DEBUG("... adding weights");
    if(this->_weights.getSize() < 1) ERROR("unable to access weight objects");
    morphfunc->addOwnedComponents(this->_weights);

    DEBUG("... adding temporary objects");
    morphfunc->addOwnedComponents(sumElements);
    morphfunc->addServerList(sumElements);
    morphfunc->addServerList(scaleElements);

#ifdef USE_UBLAS
    std::cout.precision(std::numeric_limits<double>::digits);
#endif
#ifdef DEBUG_
    morphfunc->Print();
#endif
    DEBUG("successfully created morphing function");

    // fill the this
    this->_sumFunc = morphfunc;
  }
  //////////////////////////////////////////////////////////////////////////////
  /// create all the temporary objects required by the class

  static RooLagrangianMorphFunc::CacheElem* createCache(const RooLagrangianMorphing::RooLagrangianMorphFunc* func)
  {
    std::string obsName = func->_config._obsName;
    DEBUG("creating cache for basePdf " << func);
    RooLagrangianMorphing::ParamSet values = getParams(func->_operators);

    RooLagrangianMorphFunc::CacheElem* cache = new RooLagrangianMorphFunc::CacheElem();
    cache->createComponents(func->_paramCards,func->_flagValues,func->GetName(),func->_diagrams,func->_config._nonInterfering,func->_flags);

    DEBUG("performing matrix operations");
    cache->buildMatrix(func->_paramCards,func->_flagValues,func->_flags);
    if(obsName.size() == 0){
      ERROR("Matrix inversion succeeded, but no observable was supplied. quitting...");
      return cache;
    }
    
    DEBUG("building morphing function");
    #ifdef _DEBUG_
    DEBUG("observable: " << func->getObservable()->GetName());
    DEBUG("binWidth: " << func->getBinWidth()->GetName());
    #endif
    
    setParams(func->_flags,1);
    cache->buildMorphingFunction(func->GetName(),func->_paramCards,func->_sampleMap,func->_physics,
                                 func->_allowNegativeYields,func->getObservable(),func->getBinWidth());
    setParams(values,func->_operators,true);
    setParams(func->_flags,1);
    return cache;
  }

  //////////////////////////////////////////////////////////////////////////////
  /// create all the temporary objects required by the class
  /// function variant with precomputed inverse matrix

  static RooLagrangianMorphFunc::CacheElem* createCache(const RooLagrangianMorphing::RooLagrangianMorphFunc* func, const Matrix& inverse)
  {
    DEBUG("creating cache for basePdf = " << func << " with matrix");
    RooLagrangianMorphing::ParamSet values = getParams(func->_operators);

    RooLagrangianMorphFunc::CacheElem* cache = new RooLagrangianMorphFunc::CacheElem();
    cache->createComponents(func->_paramCards,func->_flagValues,func->GetName(),func->_diagrams,func->_config._nonInterfering,func->_flags);

#ifndef USE_UBLAS
    cache->_inverse.ResizeTo(inverse.GetNrows(),inverse.GetNrows());
#endif
    cache->_inverse = inverse;
    cache->_condition = NaN;

    DEBUG("building morphing function");
    setParams(func->_flags,1);
    cache->buildMorphingFunction(func->GetName(),func->_paramCards,func->_sampleMap,func->_physics,
                                 func->_allowNegativeYields,func->getObservable(),func->getBinWidth());
    setParams(values,func->_operators,true);
    setParams(func->_flags,1);
    return cache;
  }
};

///////////////////////////////////////////////////////////////////////////////
// Class Implementation ///////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
/// insert an object into a workspace (wrapper for RooWorkspace::import)
 
void RooLagrangianMorphing::importToWorkspace(RooWorkspace* ws, const RooAbsReal* object)
{
  if(!ws) return;
  if(!object) return;
  ws->import(*object,RooFit::RecycleConflictNodes());
}

////////////////////////////////////////////////////////////////////////////////
/// insert an object into a workspace (wrapper for RooWorkspace::import)

void RooLagrangianMorphing::importToWorkspace(RooWorkspace* ws, RooAbsData* object)
{
  if(!ws) return;
  if(!object) return;
  ws->import(*object);
}

////////////////////////////////////////////////////////////////////////////////
/// append the parameter map with a parmeter set

void RooLagrangianMorphing::append(RooLagrangianMorphing::ParamMap& map, const char* str, RooLagrangianMorphing::ParamSet& set)
{
  map[str]=set;
}

////////////////////////////////////////////////////////////////////////////////
/// set values to paramerter set (-?-)

void RooLagrangianMorphing::append(RooLagrangianMorphing::ParamSet& set, const char* str, double val)
{
  set[str]=val;
}

////////////////////////////////////////////////////////////////////////////////
/// insert this object into a workspace

void RooLagrangianMorphing::RooLagrangianMorphFunc::insert(RooWorkspace* ws)
{
  RooLagrangianMorphing::importToWorkspace(ws,this);
}

////////////////////////////////////////////////////////////////////////////////
/// length of floating point digits precision supported by implementation

double RooLagrangianMorphing::implementedPrecision()
{
  return RooLagrangianMorphing::SuperFloatPrecision::digits10;
}

////////////////////////////////////////////////////////////////////////////////
/// write a matrix to a file

void RooLagrangianMorphing::writeMatrixToFile(const TMatrixD& matrix, const char* fname)
{
  writeMatrixToFileT(matrix,fname);
}

////////////////////////////////////////////////////////////////////////////////
/// write a matrix to a stream

void RooLagrangianMorphing::writeMatrixToStream(const TMatrixD& matrix, std::ostream& stream)
{
  writeMatrixToStreamT(matrix,stream);
}

////////////////////////////////////////////////////////////////////////////////
/// read a matrix from a text file

TMatrixD RooLagrangianMorphing::readMatrixFromFile(const char* fname)
{
  return readMatrixFromFileT<TMatrixD>(fname);
}

////////////////////////////////////////////////////////////////////////////////
/// read a matrix from a stream

TMatrixD RooLagrangianMorphing::readMatrixFromStream(std::istream& stream)
{
  return readMatrixFromStreamT<TMatrixD>(stream);
}

////////////////////////////////////////////////////////////////////////////////
/// setup observable, recycle existing observable if defined

RooRealVar* RooLagrangianMorphing::RooLagrangianMorphFunc::setupObservable(const char* obsname,TClass* mode,TObject* inputExample)
{
  DEBUG("setting up observable");
  RooRealVar* obs = NULL;
  Bool_t obsExists(false) ;
  if (this->_observable.at(0)!=0) {
    obs = (RooRealVar*)this->_observable.at(0) ;
    obsExists = true ;
  }
  if(mode && mode->InheritsFrom(RooHistFunc::Class())){
    obs = (RooRealVar*)(dynamic_cast<RooHistFunc*>(inputExample)->getObservables().first());
    obsExists = true ;
    this->_observable.add(*obs) ;
  } else if(mode && mode->InheritsFrom(RooParamHistFunc::Class())){
    obs = (RooRealVar*)(dynamic_cast<RooParamHistFunc*>(inputExample)->paramList().first());
    obsExists = true ;
    this->_observable.add(*obs) ;
  }
    
  // obtain the observable
  if (!obsExists){
    if(mode && mode->InheritsFrom(TH1::Class())){
      DEBUG("getObservable: creating new multi-bin observable object " << obsname);
      TH1* hist = (TH1*)(inputExample);
      obs = new RooRealVar(obsname,obsname,hist->GetXaxis()->GetXmin(),hist->GetXaxis()->GetXmax());
      obs->setBins(hist->GetNbinsX());
    } else {
      DEBUG("getObservable: creating new single-bin observable object " << obsname);
      obs = new RooRealVar(obsname,obsname,0,1);
      obs->setBins(1);
    }
    this->_observable.add(*obs) ;
  } else {
    DEBUG("getobservable: recycling existing observable object " << this->_observable.at(0));
    if (strcmp(obsname,obs->GetName())!=0 ) {
      std::cerr << "WARNING: name of existing observable " << this->_observable.at(0)->GetName() << " does not match expected name " << obsname << std::endl ;
    }
  }

  
  DEBUG("managing bin width");
  TString sbw = TString::Format("binWidth_%s",makeValidName(obs->GetName()).Data());
  RooRealVar* binWidth = new RooRealVar(sbw.Data(),sbw.Data(),1.);
  double bw = obs->numBins()/(obs->getMax() - obs->getMin());
  binWidth->setVal(bw);
  binWidth->setConstant(true);
  this->_binWidths.add(*binWidth);
  
  return obs;
}
  
//#ifndef USE_MULTIPRECISION_LC
//#pragma GCC diagnostic push
//#pragma GCC diagnostic ignored "-Wunused-parameter"
//#endif

////////////////////////////////////////////////////////////////////////////////
/// update sample weight (-?-)

inline void RooLagrangianMorphing::RooLagrangianMorphFunc::updateSampleWeights()
{
//#ifdef USE_MULTIPRECISION_LC
  int sampleidx = 0;
  auto cache = this->getCache(_curNormSet);
  const size_t n(size(cache->_inverse));
  for(auto sampleit=this->_paramCards.begin(); sampleit!=this->_paramCards.end(); ++sampleit){
    const std::string sample(sampleit->first);
    // build the formula with the correct normalization
    RooLagrangianMorphing::LinearCombination* sampleformula = dynamic_cast<RooLagrangianMorphing::LinearCombination*>(this->getSampleWeight(sample.c_str()));
    if(!sampleformula){
      ERROR(TString::Format("unable to access formula for sample '%s'!",sample.c_str()).Data());
    }
    DEBUG("updating formula for sample '" << sample << "'");
    for(size_t formulaidx = 0; formulaidx<n; ++formulaidx){
      const RooLagrangianMorphing::SuperFloat val(cache->_inverse(formulaidx,sampleidx));
      if(val != val){
        ERROR("refusing to propagate NaN!");
      }
      DEBUG("   " << formulaidx << ":" << sampleformula->getCoefficient(formulaidx) << " -> " << val);
      sampleformula->setCoefficient(formulaidx,val);
      assert(sampleformula->getCoefficient(formulaidx) == val);
    }
    sampleformula->setValueDirty();
    ++sampleidx;
  }
//#else
//  ERROR("updating sample weights currently not possible without boost!");
//#endif
}
//#ifndef USE_MULTIPRECISION_LC
//#pragma GCC diagnostic pop
//#endif

////////////////////////////////////////////////////////////////////////////////
/// read the parameters from the input file

void RooLagrangianMorphing::RooLagrangianMorphFunc::readParameters(TDirectory* f)
{
  this->_paramCards = readValues<double>(f,this->_folderNames,"param_card",true);
  this->_flagValues = readValues<int>(f,this->_folderNames,"flags",false);
}


////////////////////////////////////////////////////////////////////////////////
/// retrieve the physics inputs

void RooLagrangianMorphing::RooLagrangianMorphFunc::collectInputs(TDirectory* file)
{
  std::string obsName = this->_config._obsName;
  DEBUG("initializing physics inputs from file " << file->GetName() << " with object name(s) '" << obsName << "'");
    
  TFolder* base = dynamic_cast<TFolder*>(file->Get(this->_folderNames[0].c_str()));
  TObject* obj = base->FindObject(obsName.c_str());
  if(!obj) ERROR("unable to locate object '"<<obsName<<"' in folder '" << base << "'!");
  TClass* mode = TClass::GetClass(obj->ClassName());

  RooRealVar* observable = this->setupObservable(obsName.c_str(),mode,obj);
  if(mode->InheritsFrom(TH1::Class())){
    DEBUG("using TH1");
    collectHistograms(this->GetName(), file, this->_sampleMap,this->_physics,*observable, obsName, _baseFolder, this->_paramCards);
  } else if(mode->InheritsFrom(RooHistFunc::Class()) || mode->InheritsFrom(RooParamHistFunc::Class())){
//  else if(mode->InheritsFrom(RooHistFunc::Class()) || mode->InheritsFrom(RooParamHistFunc::Class()) || mode->InheritsFrom(PiecewiseInterpolation::Class())){
    DEBUG("using RooHistFunc");
    collectRooAbsReal(this->GetName(), file, this->_sampleMap,this->_physics, obsName, this->_paramCards);
  } else if(mode->InheritsFrom(TParameter<double>::Class())){
    DEBUG("using TParameter<double>");
    collectCrosssections<double>(this->GetName(), file, this->_sampleMap,this->_physics, obsName, _baseFolder, this->_paramCards);
  } else if(mode->InheritsFrom(TParameter<float>::Class())){
    DEBUG("using TParameter<float>");
    collectCrosssections<float>(this->GetName(), file, this->_sampleMap,this->_physics, obsName, _baseFolder, this->_paramCards);
  } else if(mode->InheritsFrom(TPair::Class())){
    DEBUG("using TPair<double>");
    collectCrosssectionsTPair(this->GetName(), file, this->_sampleMap,this->_physics, obsName, _baseFolder, this->_paramCards);
  } else {
    ERROR("cannot morph objects of class '"<<mode->GetName()<<"'!");
  }
}

////////////////////////////////////////////////////////////////////////////////
/// convert the RooArgList folders into a simple vector of std::string

void RooLagrangianMorphing::RooLagrangianMorphFunc::addFolders(const RooArgList& folders)
{

  RooAbsArg* folder;
  bool foundBase = false;
  for(auto folderItr = folders.begin(); folderItr != folders.end(); ++folderItr){
    folder = (RooAbsArg*)(*folderItr);
    RooStringVar* var = dynamic_cast<RooStringVar*>(folder);
    const std::string sample(var ? var->getVal() : var->GetName());
    if(sample.size() == 0) continue;
    DEBUG("adding sample: '" << sample << "'");
    this->_folderNames.push_back(sample);
//    if(sample == this->_baseFolder){
//      foundBase = true;
//    }
  }
  if(this->_folderNames.size() > 0){
//    if(!foundBase){
//      if(this->_baseFolder.size() > 0){
//        this->_folderNames.insert(this->_folderNames.begin(),this->_baseFolder);
//      } else {
//        std::cout << "inside this loop " << _folderNames[0];
//        this->_baseFolder= _folderNames[0];
//      }
//    }
  } else {
    std::string filename = this->_config._fileName;
    TDirectory* file = openFile(filename.c_str());
    TIter next(file->GetList());
    TObject *obj = NULL;
    while ((obj = (TObject*)next())) {
      TFolder * f = dynamic_cast<TFolder*>(file->Get(obj->GetName()));
      if(!f) continue;
      std::string name(f->GetName());
      if(name.size() == 0) continue;
//      if(this->_baseFolder.size() == 0) this->_baseFolder = name;
//      if(this->_baseFolder == name){
//        this->_folderNames.insert(this->_folderNames.begin(),name);
//      }
       else {
        this->_folderNames.push_back(name);
      }
    }
    closeFile(file);
  }
}

////////////////////////////////////////////////////////////////////////////////
/// default constructor 
RooLagrangianMorphing::RooLagrangianMorphConfig::RooLagrangianMorphConfig(){}
 
////////////////////////////////////////////////////////////////////////////////
/// parameterised constructor 
RooLagrangianMorphing::RooLagrangianMorphConfig::RooLagrangianMorphConfig(const RooAbsCollection& couplings){
  extractCouplings(couplings,this->_couplings);
} 

////////////////////////////////////////////////////////////////////////////////
/// parameterised constructor 
 RooLagrangianMorphing::RooLagrangianMorphConfig::RooLagrangianMorphConfig(const RooAbsCollection& prodCouplings, const RooAbsCollection& decCouplings){
  extractCouplings(prodCouplings,this->_prodCouplings);
  extractCouplings(decCouplings,this->_decCouplings);
} 

////////////////////////////////////////////////////////////////////////////////
/// config setter for couplings
void RooLagrangianMorphing::RooLagrangianMorphConfig::setCouplings(const RooAbsCollection& couplings){
  extractCouplings(couplings,this->_couplings);
}

////////////////////////////////////////////////////////////////////////////////
/// config setter for production and decay couplings
void RooLagrangianMorphing::RooLagrangianMorphConfig::setCouplings(const RooAbsCollection& prodCouplings, const RooAbsCollection& decCouplings){
  extractCouplings(prodCouplings,this->_prodCouplings);
  extractCouplings( decCouplings,this->_decCouplings);
}

////////////////////////////////////////////////////////////////////////////////
/// config setter for input file name
void RooLagrangianMorphing::RooLagrangianMorphConfig::setFileName(const char* filename){
  this->_fileName = filename;
}

////////////////////////////////////////////////////////////////////////////////
/// config setter for name of the the observable to be morphed
void RooLagrangianMorphing::RooLagrangianMorphConfig::setObservable(const char* obsname){
  this->_obsName = obsname;
}

////////////////////////////////////////////////////////////////////////////////
/// config setter for diagrams
template <class T> 
void RooLagrangianMorphing::RooLagrangianMorphConfig::setDiagrams(const std::vector<std::vector<T> >& diagrams)
{
  for(size_t j=0; j<diagrams.size(); ++j){
    std::vector<RooListProxy*> vertices;
    for(size_t i=0; i<diagrams[j].size(); i++){
      vertices.push_back(new RooListProxy());
      vertices[i]->add(diagrams[j][i]);
    }
    this->_configDiagrams.push_back(vertices);
 }
}

////////////////////////////////////////////////////////////////////////////////
/// config setter for vertices
template <class T> 
 void RooLagrangianMorphing::RooLagrangianMorphConfig::setVertices(const std::vector<T>& vertices)
{
  std::vector<std::vector<T> > diagrams;
  diagrams.push_back(vertices);
  this->setDiagrams(diagrams);
}

template void RooLagrangianMorphing::RooLagrangianMorphConfig::setVertices<RooArgSet>(const std::vector<RooArgSet>&);
template void RooLagrangianMorphing::RooLagrangianMorphConfig::setDiagrams<RooArgSet>(const std::vector<std::vector<RooArgSet>>&);
template void RooLagrangianMorphing::RooLagrangianMorphConfig::setVertices<RooArgList>(const std::vector<RooArgList>&);
template void RooLagrangianMorphing::RooLagrangianMorphConfig::setDiagrams<RooArgList>(const std::vector<std::vector<RooArgList>>&);

////////////////////////////////////////////////////////////////////////////////
/// config setter for vertices
void RooLagrangianMorphing::RooLagrangianMorphConfig::disableInterference(const std::vector<const char*>& nonInterfering){
  // disable interference between the listed operators
  std::stringstream name;
  name << "noInteference";
  for(auto c:nonInterfering){
    name << c;
  }
 // RooListProxy* p = new RooListProxy(name.str().c_str(),name.str().c_str(),this,kTRUE,kFALSE);
  this->_nonInterfering.push_back(new RooListProxy());
  for(auto c:nonInterfering){
    p->addOwned(*(new RooStringVar(c,c,c)));
  }
}

////////////////////////////////////////////////////////////////////////////////
/// config setter for vertices
void RooLagrangianMorphing::RooLagrangianMorphConfig::disableInterferences(const std::vector<std::vector<const char*> >& nonInterfering){
  // disable interferences between the listed groups of operators
  for(size_t i=0;i<nonInterfering.size();++i){
    this->disableInterference(nonInterfering[i]);
  }
}


 RooLagrangianMorphing::RooLagrangianMorphConfig::~RooLagrangianMorphConfig()
{
  DEBUG("destructor called");
}

RooLagrangianMorphing::RooLagrangianMorphConfig::RooLagrangianMorphConfig(const RooLagrangianMorphConfig& other) :
  _obsName(other._obsName),
  _fileName(other._fileName),
  _vertices(other._vertices),
  _couplings(other._couplings),
  _prodCouplings(other._prodCouplings),
  _decCouplings(other._decCouplings),
  _nonInterfering(other._nonInterfering),
{
  DEBUG("copy constructor called");
  for(size_t j=0; j<other._configDiagrams.size(); ++j){
    std::vector<RooListProxy*> diagram;
    for(size_t i=0; i<other._configDiagrams[j].size(); ++i){
      RooListProxy* list = other._configDiagrams[j][i];
      diagram.push_back(list);
    }
    this->_configDiagrams.push_back(diagram);
  }
}

////////////////////////////////////////////////////////////////////////////////
/// protected constructor with proper arguments

RooLagrangianMorphing::RooLagrangianMorphFunc::RooLagrangianMorphFunc(const char *name, const char *title, const RooLagrangianMorphConfig& config, const char* basefolder, const RooArgList& folders, bool allowNegativeYields) :
  RooAbsReal(name,title),
  _cacheMgr(this,10,kTRUE,kTRUE),
 //_obsName(obsName),
 // _objFilter(objFilter ? objFilter : obsName),
  _baseFolder(basefolder),
  _allowNegativeYields(allowNegativeYields),
  _operators  ("operators",  "set of operators"       , this, kTRUE, kFALSE),
  _observable ("observable", "morphing observable"    , this, kTRUE, kFALSE),
  _binWidths  ("binWidths",  "set of binWidth objects", this, kTRUE, kFALSE),
  _config(config),
  _curNormSet(0)
{
  std::cout << "INSIDE ROOLAGRANGIANMORPHFUNC" << std::endl;
  DEBUG("argument constructor called: " << this);
  this->printAuthors();
  this->addFolders(folders);
  this->init();
  DEBUG("constructor completed");
}

////////////////////////////////////////////////////////////////////////////////
/// protected constructor with proper arguments

RooLagrangianMorphing::RooLagrangianMorphFunc::RooLagrangianMorphFunc(const char *name, const char *title, const RooLagrangianMorphConfig& config, const RooArgList& folders, bool allowNegativeYields) :
  RooLagrangianMorphFunc(name,title,config,"",folders,allowNegativeYields) {
  DEBUG("constructor: name,title,obsname,config,folders,allowNegativeYields");
//  this->disableInterferences(this->_nonInterfering);
  this->setup(false);
}
////////////////////////////////////////////////////////////////////////////////
/// constructor with proper arguments
RooLagrangianMorphing::RooLagrangianMorphFunc::RooLagrangianMorphFunc(const char *name, const char *title, const RooLagrangianMorphConfig& config, bool allowNegativeYields) :
  RooLagrangianMorphFunc(name,title,config,"",RooArgList(),allowNegativeYields)
{
  DEBUG("constructor: name,title,obsname,config,allowNegativeYields");
  this->setup(false);
}

RooLagrangianMorphing::RooLagrangianMorphFunc::RooLagrangianMorphFunc(const char *name, const char *title,const char* basefolder, const RooArgList& folders, bool allowNegativeYields):
  RooLagrangianMorphFunc(name,title,RooLagrangianMorphConfig(),basefolder,folders,allowNegativeYields)
{}
RooLagrangianMorphing::RooLagrangianMorphFunc::RooLagrangianMorphFunc(const char *name, const char *title, const RooArgList& folders, bool allowNegativeYields):
  RooLagrangianMorphFunc(name,title,RooLagrangianMorphConfig(),"",folders,allowNegativeYields)
{}

////////////////////////////////////////////////////////////////////////////////
/// setup this instance with the given set of operators and vertices
/// if own=true, the class will own the operatorsemplate <class Base>

void RooLagrangianMorphing::RooLagrangianMorphFunc::setup(bool own)
{
  DEBUG("setup(ops,config"<<own<<") called");
  this->_ownParameters = own;
  auto diagrams = this->_config._configDiagrams;

  if(diagrams.size() > 0){
    RooArgList operators;
    for(const std::vector<RooListProxy*> v:diagrams){
      for( const RooListProxy* t:v){
       extractOperators(*t,operators);
      }
    }

    if(own){
      this->_operators.addOwned(operators);
    } else {
      this->_operators.add(operators);
    }

  for(size_t j=0; j<diagrams.size(); ++j){
    std::vector<RooListProxy*> diagram;
    for(size_t i=0; i<diagrams[j].size(); ++i){
      std::stringstream name;
      name << "!vertex" << i;
      std::stringstream title;
      title << "set of couplings in the vertex " << i;
      diagram.push_back(new RooListProxy(name.str().c_str(),title.str().c_str(),this,kTRUE,kFALSE));
      if(own){
        diagram[i]->addOwned(*diagrams[j][i]);
      } else {
        diagram[i]->add(*diagrams[j][i]);
      }
      }
  // ownership of contents of diagrams[i][j]       
    this->_diagrams.push_back(diagram);
   }
 }


  else if(this->_config._couplings.size() > 0){
    RooArgList operators;
    std::vector<RooListProxy*> vertices;
    DEBUG("couplings provided");
    extractOperators(this->_config._couplings, operators);
    vertices.push_back(new RooListProxy("!couplings",     "set of couplings in the vertex",     this,kTRUE,kFALSE));
    if(own){
      DEBUG("adding own operators");
      this->_operators.addOwned(operators);
      vertices[0]->addOwned(this->_config._couplings);
    } else {
      DEBUG("adding non-own operators");
      this->_operators.add(operators);
      vertices[0]->add(this->_config._couplings);
    }
  this->_diagrams.push_back(vertices);
  }

  else if(this->_config._prodCouplings.size() > 0 && this->_config._decCouplings.size() > 0){
    std::vector<RooListProxy*> vertices;
    RooArgList operators;
    DEBUG("prod/dec couplings provided");
    extractOperators(this->_config._prodCouplings, operators);
    extractOperators(this->_config._decCouplings, operators);
    vertices.push_back(new RooListProxy("!production","set of couplings in the production vertex",this,kTRUE,kFALSE));
    vertices.push_back(new RooListProxy("!decay",     "set of couplings in the decay vertex",     this,kTRUE,kFALSE));
    if(own){
      DEBUG("adding own operators");
      this->_operators.addOwned(operators);
      vertices[0]->addOwned(this->_config._prodCouplings);
      vertices[1]->addOwned(this->_config._decCouplings);
    } else {
      DEBUG("adding non-own operators");
      this->_operators.add(operators);
      vertices[0]->add(this->_config._prodCouplings);
      vertices[1]->add(this->_config._decCouplings);
    }
  this->_diagrams.push_back(vertices);
  }
}

////////////////////////////////////////////////////////////////////////////////
/// (-?-)

void RooLagrangianMorphing::RooLagrangianMorphFunc::init()
{
  std::string filename = this->_config._fileName;
  TDirectory* file = openFile(filename.c_str());
  if(!file) ERROR("unable to open file '"<<filename<<"'!");
  this->readParameters(file);
  checkNameConflict(this->_paramCards,this->_operators);
  this->collectInputs(file);
  closeFile(file);
  this->addServerList(this->_physics);
  DEBUG("adding flags");
  RooRealVar* nNP0 = new RooRealVar("nNP0","nNP0",1.,0,1.);
  nNP0->setStringAttribute("NP","0");
  nNP0->setConstant(true);
  this->_flags.add(*nNP0);
  RooRealVar* nNP1 = new RooRealVar("nNP1","nNP1",1.,0,1.);
  nNP1->setStringAttribute("NP","1");
  nNP1->setConstant(true);
  this->_flags.add(*nNP1);
  RooRealVar* nNP2 = new RooRealVar("nNP2","nNP2",1.,0,1.);
  nNP2->setStringAttribute("NP","2");
  nNP2->setConstant(true);
  this->_flags.add(*nNP2);
  RooRealVar* nNP3 = new RooRealVar("nNP3","nNP3",1.,0,1.);
  nNP3->setStringAttribute("NP","3");
  nNP3->setConstant(true);
  this->_flags.add(*nNP3);
  RooRealVar* nNP4 = new RooRealVar("nNP4","nNP4",1.,0,1.);
  nNP4->setStringAttribute("NP","4");
  nNP4->setConstant(true);
  this->_flags.add(*nNP4);
}

////////////////////////////////////////////////////////////////////////////////
/// copy constructor

RooLagrangianMorphing::RooLagrangianMorphFunc::RooLagrangianMorphFunc(const RooLagrangianMorphFunc& other, const char* name) :
  RooAbsReal(other,name),
  _cacheMgr(other._cacheMgr,this),
  _scale(other._scale),
  //_objFilter(other._objFilter),
  _baseFolder(other._baseFolder),
  _allowNegativeYields(other._allowNegativeYields),
  _folderNames(other._folderNames),
  _paramCards    (other._paramCards),
  _flagValues    (other._flagValues),
  _sampleMap  (other._sampleMap),
  _physics    (other._physics.GetName(),    this,other._physics),
  _operators  (other._operators.GetName(),  this,other._operators),
  _observable (other._observable.GetName(), this,other._observable),
  _binWidths  (other._binWidths.GetName(),  this,other._binWidths),
  _flags      (other._flags.GetName(),      this,other._flags),
  _config (other._config),
  _curNormSet(0)
{
  DEBUG("copy constructor called");
  for(size_t j=0; j<other._diagrams.size(); ++j){
    std::vector<RooListProxy*> diagram;
    for(size_t i=0; i<other._diagrams[j].size(); ++i){
      RooListProxy* list = new RooListProxy(other._diagrams[j][i]->GetName(),this,*(other._diagrams[j][i]));
      diagram.push_back(list);
    }
    this->_diagrams.push_back(diagram);
  }
}

////////////////////////////////////////////////////////////////////////////////
/// set energy scale of the EFT expansion

void RooLagrangianMorphing::RooLagrangianMorphFunc::setScale(double val)
{
  this->_scale = val;
}

////////////////////////////////////////////////////////////////////////////////
/// get energy scale of the EFT expansion

double RooLagrangianMorphing::RooLagrangianMorphFunc::getScale()
{
  return this->_scale;
}

////////////////////////////////////////////////////////////////////////////////
// default constructor

RooLagrangianMorphing::RooLagrangianMorphFunc::RooLagrangianMorphFunc() :
  _operators  ("operators",  "set of operators"        , this, kTRUE, kFALSE),
  _observable("observable",  "morphing observable"     , this, kTRUE, kFALSE),
  _binWidths  ("binWidths",  "set of bin width objects", this, kTRUE, kFALSE)
{
  static int counter(0);
  DEBUG("default constructor called: " << this << " " << counter);
  counter++;
  this->printAuthors();
}

////////////////////////////////////////////////////////////////////////////////
/// default destructor

RooLagrangianMorphing::RooLagrangianMorphFunc::~RooLagrangianMorphFunc()
{
  DEBUG("destructor called");
}

////////////////////////////////////////////////////////////////////////////////
/// cloning method

TObject* RooLagrangianMorphing::RooLagrangianMorphFunc::clone(const char* newname) const
{
  return new RooLagrangianMorphFunc(*this,newname);
}

////////////////////////////////////////////////////////////////////////////////
/// print the authors information

void RooLagrangianMorphing::RooLagrangianMorphFunc::printAuthors() const
{
  std::cout << "\033[1mRooLagrangianMorph\033[0m: a RooFit class for morphing physics distributions between configurations. authors:" << std::endl;
  std::cout << "   " << "Lydia Brenner   (lbrenner@cern.ch)" << std::endl;
  std::cout << "   " << "Carsten Burgard (cburgard@cern.ch)" << std::endl;
  std::cout << "   " << "Katharina Ecker (kecker@cern.ch)" << std::endl;
  std::cout << "   " << "Adam Kaluza     (akaluza@cern.ch)" << std::endl;
  std::cout << "please feel free to contact with questions and suggestions." << std::endl;
}

////////////////////////////////////////////////////////////////////////////////
/// calculate the number of samples needed to morph a bivertex, 2-2 physics process

int RooLagrangianMorphing::countSamples(int nprod, int ndec, int nboth)
{
  FeynmanDiagram diagram;
  std::vector<bool> prod;
  std::vector<bool> dec;
  for(int i=0; i<nboth; ++i){
    prod.push_back(true);
    dec.push_back(true);
  }
  for(int i=0; i<nprod; ++i){
    prod.push_back(true);
    dec.push_back(false);
  }
  for(int i=0; i<ndec; ++i){
    prod.push_back(false);
    dec.push_back(true);
  }
  diagram.push_back(prod);
  diagram.push_back(dec);
  MorphFuncPattern morphfuncpattern;
  ::collectPolynomials(morphfuncpattern,diagram);
  return morphfuncpattern.size();
}

////////////////////////////////////////////////////////////////////////////////
/// calculate the number of samples needed to morph a certain physics process
/// usage:
///   countSamples ( { RooLagrangianMorphing::RooLagrangianMorphFunc::makeHCggfCouplings(), RooLagrangianMorphing::RooLagrangianMorphFunc::makeHCHZZCouplings() } )

int RooLagrangianMorphing::countSamples(std::vector<RooArgList*>& vertices)
{
  RooArgList operators,couplings;
  for(auto vertex: vertices){
    extractOperators(*vertex,operators);
    extractCouplings(*vertex,couplings);
  }
  FeynmanDiagram diagram;
  ::fillFeynmanDiagram<RooArgList>(diagram,vertices,couplings);
  MorphFuncPattern morphfuncpattern;
  ::collectPolynomials(morphfuncpattern,diagram);
  return morphfuncpattern.size();
}

////////////////////////////////////////////////////////////////////////////////
/// create TPair containers of the type expected by the RooLagrangianMorph

TPair* RooLagrangianMorphing::makeCrosssectionContainer(double xs, double unc)
{
  TPair* v = new TPair(new TParameter<double>("xsection",xs),new TParameter<double>("uncertainty",unc));
  return v;
}

////////////////////////////////////////////////////////////////////////////////
/// create only the weight formulas. static function for external usage.

std::map<std::string,std::string> RooLagrangianMorphing::createWeightStrings(const RooLagrangianMorphing::ParamMap& inputs, const std::vector<std::string>& couplings)
{
  return RooLagrangianMorphing::createWeightStrings(inputs,couplings);
}

////////////////////////////////////////////////////////////////////////////////
/// create only the weight formulas. static function for external usage.

std::map<std::string,std::string> RooLagrangianMorphing::createWeightStrings(const RooLagrangianMorphing::ParamMap& inputs, const std::vector<std::vector<std::string> >& vertices_str)
{
  std::vector<RooArgList*> vertices;
  RooArgList couplings;
  for(const auto& vtx:vertices_str){
    RooArgList* vertex = new RooArgList();
    for(const auto& c:vtx){
      RooRealVar* coupling = (RooRealVar*)(couplings.find(c.c_str()));
      if(!coupling){
        coupling = new RooRealVar(c.c_str(),c.c_str(),1.,0.,10.);
        couplings.add(*coupling);
      }
      vertex->add(*coupling);
    }
    vertices.push_back(vertex);
  }
  auto retval = RooLagrangianMorphing::createWeightStrings(inputs,vertices,couplings);
  for(auto v:vertices){
    delete v;
  }
  return retval;
}

////////////////////////////////////////////////////////////////////////////////
/// create only the weight formulas. static function for external usage.

std::map<std::string,std::string> RooLagrangianMorphing::createWeightStrings(const RooLagrangianMorphing::ParamMap& inputs, const std::vector<RooArgList*>& vertices, RooArgList& couplings)
{
  std::vector<RooArgList*> nonInterfering;
  RooArgList flags;
  FlagMap flagValues;
  return RooLagrangianMorphing::createWeightStrings(inputs,vertices,couplings,flagValues,flags,nonInterfering);
}

////////////////////////////////////////////////////////////////////////////////
/// create only the weight formulas. static function for external usage.
 
std::map<std::string,std::string> RooLagrangianMorphing::createWeightStrings(const RooLagrangianMorphing::ParamMap& inputs, const std::vector<RooArgList*>& vertices, RooArgList& couplings, const RooLagrangianMorphing::FlagMap& flagValues, const RooArgList& flags, const std::vector<RooArgList*>& nonInterfering)
{
  FormulaList formulas = ::createFormulas("",inputs,flagValues,{vertices},couplings,flags,nonInterfering);
  RooArgSet operators;
  extractOperators(couplings,operators);
  Matrix matrix(::buildMatrixT<Matrix>(inputs,formulas,operators,flagValues,flags));
  if(size(matrix) < 1 ){
    ERROR("input matrix is empty, please provide suitable input samples!");
  }
  Matrix inverse(::diagMatrix(size(matrix)));
  double condition __attribute__((unused)) = (double)(invertMatrix(matrix,inverse));
  auto retval = buildSampleWeightStrings(inputs,formulas,inverse);
  for(auto f:formulas){
    delete f.second;
  }
  return retval;
}

////////////////////////////////////////////////////////////////////////////////
/// create only the weight formulas. static function for external usage.

RooArgSet RooLagrangianMorphing::createWeights(const RooLagrangianMorphing::ParamMap& inputs, const std::vector<RooArgList*>& vertices, RooArgList& couplings, const RooLagrangianMorphing::FlagMap& flagValues, const RooArgList& flags, const std::vector<RooArgList*>& nonInterfering)
{
  FormulaList formulas = ::createFormulas("",inputs,flagValues,{vertices},couplings,flags,nonInterfering);
  RooArgSet operators;
  extractOperators(couplings,operators);
  Matrix matrix(::buildMatrixT<Matrix>(inputs,formulas,operators,flagValues,flags));
  if(size(matrix) < 1 ){
    ERROR("input matrix is empty, please provide suitable input samples!");
  }
  Matrix inverse(::diagMatrix(size(matrix)));
  double condition __attribute__((unused)) = (double)(invertMatrix(matrix,inverse));
  RooArgSet retval;
  ::buildSampleWeights(retval,(const char*)NULL /* name */,inputs,formulas,inverse);
  return retval;
}

////////////////////////////////////////////////////////////////////////////////
/// create only the weight formulas. static function for external usage.

RooArgSet RooLagrangianMorphing::createWeights(const RooLagrangianMorphing::ParamMap& inputs, const std::vector<RooArgList*>& vertices, RooArgList& couplings)
{
  std::vector<RooArgList*> nonInterfering;
  RooArgList flags;
  FlagMap flagValues;
  return RooLagrangianMorphing::createWeights(inputs,vertices,couplings,flagValues,flags,nonInterfering);
}

////////////////////////////////////////////////////////////////////////////////
/// find the one component that is a ParamHistFunc


RooParamHistFunc* RooLagrangianMorphing::RooLagrangianMorphFunc::getBaseTemplate(){
  RooRealSumFunc* mf = this->getFunc();
  if(!mf) ERROR("unable to retrieve morphing function");
  RooArgSet* args = mf->getComponents();
  for (auto itr = args->begin(); itr != args->end(); ++itr){
    RooProduct* prod = dynamic_cast<RooProduct*>(*itr);
      for(auto subitr = prod->components().begin(); subitr != prod->components().end(); ++subitr){
        RooParamHistFunc* p = dynamic_cast<RooParamHistFunc*>(*subitr);
      if(p){
        return p;
      }
    }
  }

  return NULL;
}

////////////////////////////////////////////////////////////////////////////////
/// return the RooProduct that is the element of the RooRealSumPdfi
///  corresponding to the given sample name

RooProduct* RooLagrangianMorphing::RooLagrangianMorphFunc::getSumElement(const char* name) const
{
  RooRealSumFunc* mf = this->getFunc();
  if(!mf) ERROR("unable to retrieve morphing function");
  RooArgSet* args = mf->getComponents();
  TString prodname (name);
  prodname.Append("_");
  prodname.Append(this->GetName());

  for(auto itr = args->begin(); itr != args->end(); ++itr){
    RooProduct* prod = dynamic_cast<RooProduct*>(*itr);
        if(!prod) continue;
    TString sname(prod->GetName());
    if(sname.CompareTo(prodname) == 0){
      return prod;
    }
  }
  return NULL;
}
////////////////////////////////////////////////////////////////////////////////
/// return the vector of sample names, used to build the morph func

const std::vector<std::string>& RooLagrangianMorphing::RooLagrangianMorphFunc::getSamples() const
{
  return this->_folderNames;
}

////////////////////////////////////////////////////////////////////////////////
/// retrieve the weight (prefactor) of a sample with the given name

RooAbsReal* RooLagrangianMorphing::RooLagrangianMorphFunc::getSampleWeight(const char* name)
{
  auto cache = this->getCache(_curNormSet);
  TString wname(name);
  wname.Prepend("w_");
  wname.Append("_");
  wname.Append(this->GetName());
  return dynamic_cast<RooAbsReal*>(cache->_weights.find(wname));
}

////////////////////////////////////////////////////////////////////////////////
/// print the current sample weights

void RooLagrangianMorphing::RooLagrangianMorphFunc::printWeights() const
{
  this->printSampleWeights();
}

////////////////////////////////////////////////////////////////////////////////
/// print the current sample weights

void RooLagrangianMorphing::RooLagrangianMorphFunc::printSampleWeights() const
{
  auto* cache = this->getCache(this->_curNormSet);
  for(const auto& sample:this->_sampleMap){
    TString weightName = TString::Format("w_%s_%s",sample.first.c_str(),this->GetName());
    RooAbsReal* weight = (RooAbsReal*)(cache->_weights.find(weightName.Data()));
    if(!weight) continue;
    std::cout << weight->GetName() << " = " << weight->GetTitle() << " = " << weight->getVal() << std::endl;
  }
}

////////////////////////////////////////////////////////////////////////////////
/// randomize the parameters a bit
/// useful to test and debug fitting

void RooLagrangianMorphing::RooLagrangianMorphFunc::randomizeParameters(double z)
{
  RooRealVar* obj;
  TRandom3 r;

  for(auto itr = _operators.begin(); itr != _operators.end(); ++itr){
    obj = dynamic_cast<RooRealVar*>(*itr);
    double val = obj->getVal();
    if(obj->isConstant()) continue;
    double variation = r.Gaus(1,z);
    obj->setVal(val*variation);
  }
}

////////////////////////////////////////////////////////////////////////////////
/// retrive the new physics objects and update the weights in the morphing function

bool RooLagrangianMorphing::RooLagrangianMorphFunc::updateCoefficients()
{
  auto cache = this->getCache(_curNormSet);

  std::string filename = this->_config._fileName;
  TDirectory* file = openFile(filename.c_str());
  if(!file){
    ERROR("unable to open file '" << filename <<"'!");
    return false;
  }
  DEBUG("reading parameter sets.");

  this->readParameters(file);

  checkNameConflict(this->_paramCards,this->_operators);
  this->collectInputs(file);

  cache->buildMatrix(this->_paramCards,this->_flagValues,this->_flags);
  this->updateSampleWeights();
  
  closeFile(file);
  return true;
}

////////////////////////////////////////////////////////////////////////////////
/// setup the morphing function with a predefined inverse matrix
/// call this function *before* any other after creating the object

bool RooLagrangianMorphing::RooLagrangianMorphFunc::useCoefficients(const TMatrixD& inverse)
{
  RooLagrangianMorphFunc::CacheElem* cache = (RooLagrangianMorphFunc::CacheElem*) _cacheMgr.getObj(0,(RooArgSet*)0);
  Matrix m = makeSuperMatrix(inverse);
  if (cache) {
    std::string filename = this->_config._fileName;
    cache->_inverse = m;
    TDirectory* file = openFile(filename.c_str());
    if(!file) ERROR("unable to open file '"<<filename<<"'!");
    DEBUG("reading parameter sets.");

    this->readParameters(file);
    checkNameConflict(this->_paramCards,this->_operators);
    this->collectInputs(file);
    
    // then, update the weights in the morphing function
    this->updateSampleWeights();

    closeFile(file);
  } else {
    cache = RooLagrangianMorphFunc::CacheElem::createCache(this,m);
    if(!cache) ERROR("unable to create cache!");
    this->_cacheMgr.setObj(0,0,cache,0) ;
  }
  return true;
}

////////////////////////////////////////////////////////////////////////////////
// setup the morphing function with a predefined inverse matrix
// call this function *before* any other after creating the object

bool RooLagrangianMorphing::RooLagrangianMorphFunc::useCoefficients(const char* filename)
{
  RooLagrangianMorphFunc::CacheElem* cache = (RooLagrangianMorphFunc::CacheElem*) _cacheMgr.getObj(0,(RooArgSet*)0);
  if (cache) {
    return false;
  }
  cache = RooLagrangianMorphFunc::CacheElem::createCache(this,readMatrixFromFileT<Matrix>(filename));
  if(!cache) ERROR("unable to create cache!");
  this->_cacheMgr.setObj(0,0,cache,0);
  return true;
}

////////////////////////////////////////////////////////////////////////////////
/// write the inverse matrix to a file

bool RooLagrangianMorphing::RooLagrangianMorphFunc::writeCoefficients(const char* filename)
{
  auto cache = this->getCache(_curNormSet);
  if(!cache) return false;
  writeMatrixToFileT(cache->_inverse,filename);
  return true;
}

////////////////////////////////////////////////////////////////////////////////
/// retrieve the cache object

typename RooLagrangianMorphing::RooLagrangianMorphFunc::CacheElem* RooLagrangianMorphing::RooLagrangianMorphFunc::getCache(const RooArgSet* /*nset*/) const
{
  RooLagrangianMorphFunc::CacheElem* cache = (RooLagrangianMorphFunc::CacheElem*) _cacheMgr.getObj(0,(RooArgSet*)0);
  if (!cache) {
    DEBUG("creating cache from getCache function for " << this);
    #ifdef _DEBUG_
   // ::printClients(this);
   // ::printServers(this);
    #endif
    
    DEBUG("current storage has size " << this->_sampleMap.size());
    cache = RooLagrangianMorphFunc::CacheElem::createCache(this);
    if(cache) this->_cacheMgr.setObj(0,0,cache,0);
    else ERROR("unable to create cache!");
  }
  return cache;
}

////////////////////////////////////////////////////////////////////////////////
/// return true if a cache object is present, false otherwise

bool RooLagrangianMorphing::RooLagrangianMorphFunc::hasCache() const
{
  return (bool)(_cacheMgr.getObj(0,(RooArgSet*)0));
}

////////////////////////////////////////////////////////////////////////////////
/// set one parameter to a specific value

void RooLagrangianMorphing::RooLagrangianMorphFunc::setParameter(const char* name, double value)
{
  RooRealVar* param = this->getParameter(name);
  if(!param){
    return;
  }
  if(value > param->getMax()) param->setMax(value);
  if(value < param->getMin()) param->setMin(value);
  param->setVal(value);
}

////////////////////////////////////////////////////////////////////////////////
/// set one flag to a specific value

void RooLagrangianMorphing::RooLagrangianMorphFunc::setFlag(const char* name, double value)
{
  RooRealVar* param = this->getFlag(name);
  if(!param){
    return;
  }
  param->setVal(value);
}

////////////////////////////////////////////////////////////////////////////////
/// set one parameter to a specific value and range

void RooLagrangianMorphing::RooLagrangianMorphFunc::setParameter(const char* name, double value, double min, double max)
{
  RooRealVar* param = this->getParameter(name);
  if(!param){
    return;
  }
  param->setMin(min);
  param->setMax(max);
  param->setVal(value);
}

////////////////////////////////////////////////////////////////////////////////
/// set one parameter to a specific value and range
void RooLagrangianMorphing::RooLagrangianMorphFunc::setParameter(const char* name, double value, double min, double max, double error)
{
  RooRealVar* param = this->getParameter(name);
  if(!param){
    return;
  }
  param->setMin(min);
  param->setMax(max);
  param->setVal(value);
  param->setError(error);
}

////////////////////////////////////////////////////////////////////////////////
/// return true if the parameter with the given name is set constant, false otherwise

bool RooLagrangianMorphing::RooLagrangianMorphFunc::isParameterConstant(const char* name) const
{
  RooRealVar* param = this->getParameter(name);
  if(param){
    return param->isConstant();
  }
  return true;
}

////////////////////////////////////////////////////////////////////////////////
/// retrieve the RooRealVar object incorporating the parameter with the given name
RooRealVar* RooLagrangianMorphing::RooLagrangianMorphFunc::getParameter(const char* name) const
{

  RooRealVar* param = dynamic_cast<RooRealVar*>(this->_operators.find(name));
  if(!param){
    return NULL;
  }
  return param;
}

////////////////////////////////////////////////////////////////////////////////
/// retrieve the RooRealVar object incorporating the flag with the given name

RooRealVar* RooLagrangianMorphing::RooLagrangianMorphFunc::getFlag(const char* name) const
{
  RooRealVar* flag = dynamic_cast<RooRealVar*>(this->_flags.find(name));
  if(!flag){
    return NULL;
  }
  return flag;
}

////////////////////////////////////////////////////////////////////////////////
/// check if a parameter of the given name is contained in the list of known parameters

bool RooLagrangianMorphing::RooLagrangianMorphFunc::hasParameter(const char* name) const
{
  RooRealVar* p = this->getParameter(name);
  if(p) return true;
  return false;
}

////////////////////////////////////////////////////////////////////////////////
/// call setConstant with the boolean argument provided on the parameter with the given name

void RooLagrangianMorphing::RooLagrangianMorphFunc::setParameterConstant(const char* name, bool constant) const
{
  RooRealVar* param = this->getParameter(name);
  if(param){
    return param->setConstant(constant);
  }
}

////////////////////////////////////////////////////////////////////////////////
/// set one parameter to a specific value

double RooLagrangianMorphing::RooLagrangianMorphFunc::getParameterValue(const char* name) const
{
  RooRealVar* param = this->getParameter(name);
  if(param){
    return param->getVal();
  }
  return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// set the morphing parameters to those supplied in the given param hist

void RooLagrangianMorphing::RooLagrangianMorphFunc::setParameters(TH1* paramhist)
{
  setParams(paramhist,this->_operators,false);
}

////////////////////////////////////////////////////////////////////////////////
/// set the morphing parameters to those supplied in the sample with the given name

void RooLagrangianMorphing::RooLagrangianMorphFunc::setParameters(const char* foldername)
{
  std::string filename = this->_config._fileName;
  TDirectory* file = openFile(filename.c_str());
  TH1* paramhist = getParamHist(file,foldername);
  setParams(paramhist,this->_operators,false);
  closeFile(file);
}

/////////////////////////////////////////////////////////////////////////////////
/// retrieve the morphing parameters associated to the sample with the given name

RooLagrangianMorphing::ParamSet RooLagrangianMorphing::RooLagrangianMorphFunc::getParameters(const char* foldername) const
{
  const std::string name(foldername);
  return _paramCards.at(name);
}

////////////////////////////////////////////////////////////////////////////////
/// set the morphing parameters to those supplied in the list with the given name

void RooLagrangianMorphing::RooLagrangianMorphFunc::setParameters(const RooArgList* list)
{
  for(auto itr = list->begin(); itr != list->end(); ++itr){
    RooRealVar* param = dynamic_cast<RooRealVar*>(*itr);
    if(!param) continue;
    this->setParameter(param->GetName(),param->getVal());
  }
}

////////////////////////////////////////////////////////////////////////////////
/// retrieve the histogram observable

RooRealVar* RooLagrangianMorphing::RooLagrangianMorphFunc::getObservable() const
{
  if(this->_observable.getSize() < 1){
    ERROR("observable not available!");
    return NULL;
  }
  return static_cast<RooRealVar*>(this->_observable.at(0));
}

////////////////////////////////////////////////////////////////////////////////
/// retrieve the histogram observable

RooRealVar* RooLagrangianMorphing::RooLagrangianMorphFunc::getBinWidth() const
{
  if(this->_binWidths.getSize() < 1){
    ERROR("bin width not available!");
    return NULL;
  }
  return static_cast<RooRealVar*>(this->_binWidths.at(0));
}

////////////////////////////////////////////////////////////////////////////////
/// retrieve a histogram output of the current morphing settings

TH1* RooLagrangianMorphing::RooLagrangianMorphFunc::createTH1(const std::string& name, RooFitResult* r)
{
  return this->createTH1(name,false,r);
}

////////////////////////////////////////////////////////////////////////////////
/// retrieve a histogram output of the current morphing settings

TH1* RooLagrangianMorphing::RooLagrangianMorphFunc::createTH1(const std::string& name, bool correlateErrors, RooFitResult* r)
{
  RooRealSumFunc* pdf = this->getFunc();
  RooRealVar* observable = this->getObservable();
  
  const int nbins = observable->getBins();
  
  TH1* hist = new TH1F(name.c_str(),name.c_str(),nbins,observable->getBinning().array());
  
  bool ownResult = !(bool)(r);
  RooArgSet* args = pdf->getComponents();
  for (int i=0; i<nbins; ++i) {
    observable->setBin(i);
    double val = 0;
    double unc2 = 0;
    double unc = 0;
    for(auto itr = args->begin(); itr != args->end(); ++itr){
      RooProduct* prod = dynamic_cast<RooProduct*>(*itr);
      if(!prod) continue;
      RooAbsArg* phys = prod->components().find(TString::Format("phys_%s",prod->GetName()));
      RooHistFunc* hf = dynamic_cast<RooHistFunc*>(phys);
      if(!hf){
        continue;
      }
      const RooDataHist& dhist = hf->dataHist();
      dhist.get(i);
      RooAbsReal* formula = dynamic_cast<RooAbsReal*>(prod->components().find(TString::Format("w_%s",prod->GetName())));
      double weight = formula->getVal();
      unc2 += dhist.weightSquared()*weight*weight;
      unc += sqrt(dhist.weightSquared())*weight;
      val += dhist.weight()*weight;
    }
    hist->SetBinContent(i+1,val);
    hist->SetBinError(i+1,correlateErrors ? unc : sqrt(unc2));
  }
  if(ownResult) delete r;
  return hist;
}

////////////////////////////////////////////////////////////////////////////////
/// count the number of formulas that correspond to the current parameter set

int RooLagrangianMorphing::RooLagrangianMorphFunc::countContributingFormulas() const{
  int nFormulas = 0;
  RooRealSumFunc* mf = this->getFunc();
  if(!mf) ERROR("unable to retrieve morphing function");
  RooArgSet* args = mf->getComponents();
  for(auto itr = args->begin(); itr != args->end(); ++itr){
    RooProduct* prod = dynamic_cast<RooProduct*>(*itr);
    if(prod->getVal() != 0){
      nFormulas++;
    }
  }
  return nFormulas;
}

////////////////////////////////////////////////////////////////////////////////
/// check if there is any morphing power provided for the given parameter
/// morphing power is provided as soon as any two samples provide different, non-zero values for this parameter

bool RooLagrangianMorphing::RooLagrangianMorphFunc::isParameterUsed(const char* paramname) const
{
  std::string pname(paramname);
  double val = 0;
  bool isUsed = false;
  for(const auto& sample : this->_paramCards){
    double thisval = sample.second.at(pname);
    if(thisval != val){
      if(val != 0) isUsed = true;
      val = thisval;
    }
  }
  return isUsed;
}

////////////////////////////////////////////////////////////////////////////////
/// check if there is any morphing power provided for the given coupling
/// morphing power is provided as soon as any two samples provide
/// different, non-zero values for this coupling

bool RooLagrangianMorphing::RooLagrangianMorphFunc::isCouplingUsed(const char* couplname) const
{
  std::string cname(couplname);
  const RooArgList* args = this->getCouplingSet();
  RooAbsReal* coupling = dynamic_cast<RooAbsReal*>(args->find(couplname));
  if(!coupling) return false;
  RooLagrangianMorphing::ParamSet params = this->getParameters();
  RooLagrangianMorphFunc* self = const_cast<RooLagrangianMorphFunc*>(this);
  double val = 0;
  bool isUsed = false;
  for(const auto& sample : this->_paramCards){
    self->setParameters(sample.second);
    double thisval = coupling->getVal();
    if(thisval != val){
      if(val != 0) isUsed = true;
      val = thisval;
    }
  }
  self->setParameters(params);
  return isUsed;
}

////////////////////////////////////////////////////////////////////////////////
/// print all the parameters and their values in the given sample to the console

void RooLagrangianMorphing::RooLagrangianMorphFunc::printParameters(const char* samplename) const
{
  for(const auto& param : this->_paramCards.at(samplename)){
    if(this->hasParameter(param.first.c_str())){
      std::cout << param.first << " = " << param.second;
      if(this->isParameterConstant(param.first.c_str())) std::cout << " (const)";
      std::cout << std::endl;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
/// print all the known samples to the console

void RooLagrangianMorphing::RooLagrangianMorphFunc::printSamples() const
{
  // print all the known samples to the console
  for(auto folder : this->_folderNames){
    std::cout << folder;
    if(folder ==  this->_baseFolder) std::cout << "*";
    std::cout << std::endl;
  }
}
////////////////////////////////////////////////////////////////////////////////
/// print the current phyiscs values

void RooLagrangianMorphing::RooLagrangianMorphFunc::printPhysics() const
{
  for(const auto& sample:this->_sampleMap){
    RooAbsArg* phys = this->_physics.at(sample.second);
    if(!phys) continue;
    phys->Print();
  }
}

////////////////////////////////////////////////////////////////////////////////
/// return the number of parameters in this morphing function

int RooLagrangianMorphing::RooLagrangianMorphFunc::nParameters() const
{
  return this->getParameterSet()->getSize();
}

////////////////////////////////////////////////////////////////////////////////
///  return the number of samples in this morphing function

int RooLagrangianMorphing::RooLagrangianMorphFunc::nSamples() const
{
  return this->_folderNames.size();
}

////////////////////////////////////////////////////////////////////////////////
/// return the number of samples in this morphing function

int RooLagrangianMorphing::RooLagrangianMorphFunc::nPolynomials() const
{
  // return the number of samples in this morphing function
  auto cache = getCache(_curNormSet);
  return cache->_formulas.size();
}


////////////////////////////////////////////////////////////////////////////////
/// print the contributing smaples and their respective weights

void RooLagrangianMorphing::RooLagrangianMorphFunc::printEvaluation() const
{
  RooRealSumFunc* mf = this->getFunc();
  if(!mf){
    std::cerr << "Error: unable to retrieve morphing function" << std::endl;
    return;
  }
  RooArgSet* args = mf->getComponents();
  for(auto itr = args->begin(); itr != args->end(); ++itr){
    RooAbsReal* formula = dynamic_cast<RooAbsReal*>(*itr);
    if(formula){
      TString name(formula->GetName());
      name.Remove(0,2);
      name.Prepend("phys_");
      if(!args->find(name.Data())){
        continue;
      }
      double val = formula->getVal();
      if(val != 0){
        std::cout << formula->GetName() << ": " << val << " = " << formula->GetTitle() << std::endl;
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
/// get the set of parameters

const RooArgList* RooLagrangianMorphing::RooLagrangianMorphFunc::getParameterSet() const
{
  return &(this->_operators);
}

////////////////////////////////////////////////////////////////////////////////
/// get the set of couplings

const RooArgList* RooLagrangianMorphing::RooLagrangianMorphFunc::getCouplingSet() const
{
  auto cache = getCache(_curNormSet);
  return &(cache->_couplings);
}

////////////////////////////////////////////////////////////////////////////////
/// retrieve a set of couplings (-?-)

RooLagrangianMorphing::ParamSet RooLagrangianMorphing::RooLagrangianMorphFunc::getCouplings() const
{
  RooLagrangianMorphing::ParamSet couplings;
  for(auto itr = this->getCouplingSet()->begin(); itr != this->getCouplingSet()->end(); ++itr){
    RooAbsReal* var = dynamic_cast<RooAbsReal*>(*itr);
    if(!var) continue;
    const std::string name(var->GetName());
    double val = var->getVal();
    couplings[name] = val;
  }
  return couplings;
}

////////////////////////////////////////////////////////////////////////////////
/// retrieve the parameter set

RooLagrangianMorphing::ParamSet RooLagrangianMorphing::RooLagrangianMorphFunc::getParameters() const
{
  return getParams(this->_operators);
}

////////////////////////////////////////////////////////////////////////////////
/// retrieve a set of couplings (-?-)

void RooLagrangianMorphing::RooLagrangianMorphFunc::setParameters(const ParamSet& params)
{
  setParams(params,this->_operators,false);
}

////////////////////////////////////////////////////////////////////////////////
/// reset all the flags

void RooLagrangianMorphing::RooLagrangianMorphFunc::resetFlags() {
  setParams(this->_flags,1.);
}

////////////////////////////////////////////////////////////////////////////////
/// pdf is self-normalized

Bool_t RooLagrangianMorphing::RooLagrangianMorphFunc::selfNormalized() const
{
  return kTRUE ;
}

////////////////////////////////////////////////////////////////////////////////
/// (currently similar to cloning the Pdf

RooWrapperPdf* RooLagrangianMorphing::RooLagrangianMorphFunc::createPdf() const
{
  auto cache = getCache(_curNormSet);
  RooRealSumFunc* func = cache->_sumFunc;
  // create a wrapper on the roorealsumfunc
  RooWrapperPdf* x = new RooWrapperPdf(TString::Format("pdf_%s",func->GetName()),TString::Format("pdf of %s",func->GetTitle()), *func);
  return x;
}
/*
////////////////////////////////////////////////////////////////////////////////
/// get a standalone clone of the pdf that does not depend on this object

RooWrapperPdf* RooLagrangianMorphing::RooLagrangianMorphFunc::clonePdf() const
{
  auto cache = getCache(_curNormSet);
  RooRealSumFunc* func = cache->_sumFunc;
  // create a wrapper on the roorealsumfunc
  RooWrapperPdf* x = new RooWrapperPdf(TString::Format("pdf_%s",func->GetName()),TString::Format("pdf of %s",func->GetTitle()), *func);
  return x;
}
*/
////////////////////////////////////////////////////////////////////////////////
/// get the func

RooRealSumFunc* RooLagrangianMorphing::RooLagrangianMorphFunc::getFunc() const
{
  auto cache = getCache(_curNormSet);
  return cache->_sumFunc;
}

////////////////////////////////////////////////////////////////////////////////
/// get a standalone clone of the func that does not depend on this object

RooRealSumFunc* RooLagrangianMorphing::RooLagrangianMorphFunc::cloneFunc() const
{
  auto cache = getCache(_curNormSet);
  RooRealSumFunc* orig = cache->_sumFunc;
  RooRealSumFunc* funcclone = new RooRealSumFunc(orig->GetName(),orig->GetTitle(),orig->funcList(),orig->coefList());
  return funcclone;
}
////////////////////////////////////////////////////////////////////////////////
/// return extended mored capabilities

RooAbsPdf::ExtendMode RooLagrangianMorphing::RooLagrangianMorphFunc::extendMode() const
{
  return this->createPdf()->extendMode();
}

////////////////////////////////////////////////////////////////////////////////
/// return expected number of events for extended likelihood calculation,
/// this is the sum of all coefficients

Double_t RooLagrangianMorphing::RooLagrangianMorphFunc::expectedEvents(const RooArgSet* nset) const
{
  return this->createPdf()->expectedEvents(nset);
}

////////////////////////////////////////////////////////////////////////////////
/// return the number of expected events for the current parameter set

Double_t RooLagrangianMorphing::RooLagrangianMorphFunc::expectedEvents() const
{
  RooArgSet set;
  set.add(*this->getObservable());
  return this->createPdf()->expectedEvents(set);
}

////////////////////////////////////////////////////////////////////////////////
/// return expected number of events for extended likelihood calculation,
/// this is the sum of all coefficients

Double_t RooLagrangianMorphing::RooLagrangianMorphFunc::expectedEvents(const RooArgSet& nset) const
{
  return createPdf()->expectedEvents(&nset) ;
}
////////////////////////////////////////////////////////////////////////////////
/// return the expected uncertainity for the current parameter set

double RooLagrangianMorphing::RooLagrangianMorphFunc::expectedUncertainty() const
{
  RooRealVar* observable = this->getObservable();
  auto cache = this->getCache(_curNormSet);
  double unc2 = 0;
  for(const auto& sample:this->_sampleMap){
    RooAbsArg* phys = this->_physics.at(sample.second);
    TString weightName = TString::Format("w_%s_%s",sample.first.c_str(),this->GetName());
    RooAbsReal* weight = (RooAbsReal*)(cache->_weights.find(weightName.Data()));
    if(!weight){
      ERROR("unable to find object "+weightName);
    }
    double newunc2 = 0;
    RooHistFunc* hf = dynamic_cast<RooHistFunc*>(phys);
    RooRealVar* rv = dynamic_cast<RooRealVar*>(phys);
    if(hf){
      const RooDataHist& hist = hf->dataHist();
      for(Int_t j=0; j<observable->getBins(); ++j){
        hist.get(j);
        newunc2 += hist.weightSquared();
      }
    } else if(rv){
      newunc2 = pow(rv->getError(),2);
    }
    double w = weight->getVal();
    unc2 += newunc2*w*w;
    // std::cout << phys->GetName() << " : " << weight->GetName() << " thisweight: " <<  w << " thisxsec2: " << newunc2 << " weight " << weight << std::endl;
  }
  return sqrt(unc2);
}

////////////////////////////////////////////////////////////////////////////////
/// print the parameters and their current values

void RooLagrangianMorphing::RooLagrangianMorphFunc::printParameters() const
{
  // print the parameters and their current values
  for(auto itr = this->_operators.begin(); itr != this->_operators.end(); ++itr){
    RooRealVar* param = dynamic_cast<RooRealVar*>(*itr);
    if(!param) continue;
    std::cout << param->GetName() << ": " << param->getVal();
    if(param->isConstant()) std::cout << " (const)";
    else {
      std::cout << " +" << param->getAsymErrorHi() << " -" << param->getAsymErrorLo();
      std::cout << " (" << param->getMin() << " - " << param->getMax() << ")";
    }
  }
    std::cout << std::endl;
}

////////////////////////////////////////////////////////////////////////////////
/// print the flags and their current values

void RooLagrangianMorphing::RooLagrangianMorphFunc::printFlags() const
{
  for(auto itr = this->_flags.begin(); itr != this->_flags.end(); ++itr){
    RooRealVar* param = dynamic_cast<RooRealVar*>(*itr);
    if(!param) continue;
    std::cout << param->GetName() << ": " << param->getVal();
    std::cout << std::endl;
  }
}

////////////////////////////////////////////////////////////////////////////////
/// print a set of couplings

void RooLagrangianMorphing::RooLagrangianMorphFunc::printCouplings() const
{
  RooLagrangianMorphing::ParamSet couplings = this->getCouplings();
  for(auto c : couplings){
    std::cout << c.first << ": " << c.second << std::endl;
  }
}

////////////////////////////////////////////////////////////////////////////////
/// retrive the lsit of bin boundaries

std::list<Double_t>* RooLagrangianMorphing::RooLagrangianMorphFunc::binBoundaries(RooAbsRealLValue& obs, Double_t xlo, Double_t xhi) const
{
  return this->getFunc()->binBoundaries(obs,xlo,xhi);
}

////////////////////////////////////////////////////////////////////////////////
/// retrive the sample Hint

std::list<Double_t>* RooLagrangianMorphing::RooLagrangianMorphFunc::plotSamplingHint(RooAbsRealLValue& obs, Double_t xlo, Double_t xhi) const
{
  return this->getFunc()->plotSamplingHint(obs,xlo,xhi);
}

////////////////////////////////////////////////////////////////////////////////
/// call getVal on the internal function

Double_t RooLagrangianMorphing::RooLagrangianMorphFunc::getValV(const RooArgSet* set) const
{
  //cout << "XX RooLagrangianMorphing::RooLagrangianMorphFunc::getValV(" << this << ") set = " << set << endl ;
  this->_curNormSet = set ;
  return RooAbsReal::getValV(set) ;
}

////////////////////////////////////////////////////////////////////////////////
/// call getVal on the internal function

Double_t RooLagrangianMorphing::RooLagrangianMorphFunc::evaluate() const
{
  RooRealSumFunc* pdf = this->getFunc();
  if(pdf) return this->_scale * pdf->getVal(_curNormSet);
  else ERROR("unable to aquire in-built pdf!");
  return 0.;
}

////////////////////////////////////////////////////////////////////////////////
/// check if this PDF is a binned distribution in the given observable

Bool_t  RooLagrangianMorphing::RooLagrangianMorphFunc::isBinnedDistribution(const RooArgSet& obs) const
{
  return this->getFunc()->isBinnedDistribution(obs);
}

////////////////////////////////////////////////////////////////////////////////
/// check if observable exists in the RooArgSet (-?-)

Bool_t RooLagrangianMorphing::RooLagrangianMorphFunc::checkObservables(const RooArgSet *nset) const
{
  return this->getFunc()->checkObservables(nset);
}

////////////////////////////////////////////////////////////////////////////////
/// Force analytical integration for the given observable

Bool_t RooLagrangianMorphing::RooLagrangianMorphFunc::forceAnalyticalInt(const RooAbsArg &arg) const
{
  return this->getFunc()->forceAnalyticalInt(arg);
}

////////////////////////////////////////////////////////////////////////////////
/// Retrieve the mat

Int_t RooLagrangianMorphing::RooLagrangianMorphFunc::getAnalyticalIntegralWN(RooArgSet &allVars, RooArgSet &numVars, const RooArgSet *normSet, const char *rangeName) const
{
  return this->getFunc()->getAnalyticalIntegralWN(allVars,numVars,normSet,rangeName);
}

////////////////////////////////////////////////////////////////////////////////
/// Retrieve the matrix of coefficients


Double_t RooLagrangianMorphing::RooLagrangianMorphFunc::analyticalIntegralWN(Int_t code, const RooArgSet *normSet, const char *rangeName) const
{
  return this->getFunc()->analyticalIntegralWN(code,normSet,rangeName);
}

////////////////////////////////////////////////////////////////////////////////
/// Retrieve the matrix of coefficients


void RooLagrangianMorphing::RooLagrangianMorphFunc::printMetaArgs(std::ostream &os) const
{
  return this->getFunc()->printMetaArgs(os);
}

////////////////////////////////////////////////////////////////////////////////
/// Retrieve the matrix of coefficients


RooAbsArg::CacheMode RooLagrangianMorphing::RooLagrangianMorphFunc::canNodeBeCached() const
{
  return this->getFunc()->canNodeBeCached();
}

////////////////////////////////////////////////////////////////////////////////
/// Retrieve the matrix of coefficients


void RooLagrangianMorphing::RooLagrangianMorphFunc::setCacheAndTrackHints(RooArgSet& arg)
{
  this->getFunc()->setCacheAndTrackHints(arg);
}

////////////////////////////////////////////////////////////////////////////////
/// Retrieve the matrix of coefficients

TMatrixD RooLagrangianMorphing::RooLagrangianMorphFunc::getMatrix() const
{
  auto cache = getCache(_curNormSet);
  if(!cache) ERROR("unable to retrieve cache!");
  return makeRootMatrix(cache->_matrix);
}

////////////////////////////////////////////////////////////////////////////////
/// Retrieve the matrix of coefficients after inversion

TMatrixD RooLagrangianMorphing::RooLagrangianMorphFunc::getInvertedMatrix() const
{
  auto cache = getCache(_curNormSet);
  if(!cache) ERROR("unable to retrieve cache!");
  return makeRootMatrix(cache->_inverse);
}

////////////////////////////////////////////////////////////////////////////////
/// Reterieve the condition of the coefficient matrix. If the condition number
/// is very large, then the matrix is ill-conditioned and is almost singular.
/// The computation of the inverse is prone to large numerical errors

double RooLagrangianMorphing::RooLagrangianMorphFunc::getCondition() const
{
  auto cache = getCache(_curNormSet);
  if(!cache) ERROR("unable to retrieve cache!");
  return cache->_condition;
}

////////////////////////////////////////////////////////////////////////////////
/// Return the linearised for of the morphing function

RooLagrangianMorphing::RooLagrangianMorphFunc* RooLagrangianMorphing::RooLagrangianMorphFunc::getLinear() const 
{
    auto cache = getCache(_curNormSet);
    RooRealSumFunc* orig = cache->_sumFunc;
   auto coeflist = orig->coefList();
   auto funclist = orig->funcList();

}