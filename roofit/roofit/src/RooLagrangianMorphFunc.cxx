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

/** \class RooLagrangianMorphFunc
    \ingroup Roofit
Class RooLagrangianMorphing is a implementation of the method of Effective
Lagrangian Morphing, described in ATL-PHYS-PUB-2015-047.
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

#include "RooAbsCollection.h"
#include "RooArgList.h"
#include "RooArgProxy.h"
#include "RooArgSet.h"
#include "RooBinning.h"
#include "RooDataHist.h"
#include "RooFormulaVar.h"
#include "RooHistFunc.h"
#include "RooLagrangianMorphFunc.h"
#include "RooLinearCombination.h"
#include "RooParamHistFunc.h"
#include "RooProduct.h"
#include "RooRealVar.h"
#include "RooStringVar.h"
#include "RooWorkspace.h"
#include "TFile.h"
#include "TFolder.h"
#include "TH1.h"
#include "TMap.h"
#include "TParameter.h"
#include "TRandom3.h"
// stl includes
#include "ROOT/RMakeUnique.hxx"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <typeinfo>

using namespace std;
ClassImp(RooLagrangianMorphFunc);

//#define _DEBUG_

///////////////////////////////////////////////////////////////////////////////
// PREPROCESSOR MAGIC /////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

// various preprocessor helpers
#define NaN std::numeric_limits<double>::quiet_NaN()

constexpr static double morphLargestWeight = 10e7;
constexpr static double morphUnityDeviation = 10e-6;

///////////////////////////////////////////////////////////////////////////////
// TEMPLATE MAGIC /////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

template <typename Test, template <typename...> class Ref>
struct is_specialization : std::false_type {};

template <template <typename...> class Ref, typename... Args>
struct is_specialization<Ref<Args...>, Ref> : std::true_type {};

///////////////////////////////////////////////////////////////////////////////
// LINEAR ALGEBRA HELPERS /////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
/// retrieve the size of a square matrix

template <class MatrixT> inline size_t size(const MatrixT &matrix);
template <> inline size_t size<TMatrixD>(const TMatrixD &mat) {
  return mat.GetNrows();
}

////////////////////////////////////////////////////////////////////////////////
/// write a matrix to a stream

template <class MatrixT>
void writeMatrixToStreamT(const MatrixT &matrix, std::ostream &stream) {
  if(!stream.good()) {
    return;
  }
  for (size_t i = 0; i < size(matrix); ++i) {
    for (size_t j = 0; j < size(matrix); ++j) {
#ifdef USE_UBLAS
      stream << std::setprecision(SuperFloatPrecision::digits10) << matrix(i, j)
             << "\t";
#else
      stream << matrix(i, j) << "\t";
#endif
    }
    stream << std::endl;
  }
}

////////////////////////////////////////////////////////////////////////////////
/// write a matrix to a text file

template <class MatrixT>
inline void writeMatrixToFileT(const MatrixT &matrix, const char *fname) {
  std::ofstream of(fname);
  if (!of.good()) {
    cerr << "unable to read file '" << fname << "'!" << std::endl;
  }
  writeMatrixToStreamT(matrix, of);
  of.close();
}

#ifdef USE_UBLAS

// boost includes
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_expression.hpp>
#include <boost/numeric/ublas/symmetric.hpp> //inc diag
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/operators.hpp>

#pragma GCC diagnostic pop

typedef boost::numeric::ublas::matrix<SuperFloat> Matrix;

////////////////////////////////////////////////////////////////////////////////
/// write a matrix

inline void printMatrix(const Matrix &mat) {
  for (size_t i = 0; i < mat.size1(); ++i) {
    for (size_t j = 0; j < mat.size2(); ++j) {
      // std::cout << std::setprecision(SuperFloatPrecision::digits10) <<
      // mat(i,j) << " ,\t";
    }
    std::cout << std::endl;
  }
}

////////////////////////////////////////////////////////////////////////////////
/// retrieve the size of a square matrix

template <> inline size_t size<Matrix>(const Matrix &matrix) {
  return matrix.size1();
}

////////////////////////////////////////////////////////////////////////////////
/// create a new diagonal matrix of size n

inline Matrix diagMatrix(size_t n) {
  return boost::numeric::ublas::identity_matrix<SuperFloat>(n);
}

////////////////////////////////////////////////////////////////////////////////
/// convert a matrix into a TMatrixD

inline TMatrixD makeRootMatrix(const Matrix &in) {
  size_t n = size(in);
  TMatrixD mat(n, n);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      mat(i, j) = double(in(i, j));
    }
  }
  return mat;
}

////////////////////////////////////////////////////////////////////////////////
/// convert a TMatrixD into a matrix

inline Matrix makeSuperMatrix(const TMatrixD &in) {
  size_t n = in.GetNrows();
  Matrix mat(n, n);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      mat(i, j) = double(in(i, j));
    }
  }
  return mat;
}

inline Matrix operator+=(const Matrix &rhs) { return add(rhs); }
inline Matrix operator*(const Matrix &m, const Matrix &otherM) {
  return prod(m, otherM);
}

////////////////////////////////////////////////////////////////////////////////
/// calculate the inverse of a matrix, returning the condition

inline SuperFloat invertMatrix(const Matrix &matrix, Matrix &inverse) {
  boost::numeric::ublas::permutation_matrix<size_t> pm(size(matrix));
  SuperFloat mnorm = norm_inf(matrix);
  Matrix lu(matrix);
  try {
    int res = lu_factorize(lu, pm);
    if (res != 0) {
      std::stringstream ss;
      ::writeMatrixToStreamT(matrix, ss);
      cxcoutP(Eval) << ss.str << std::endl;
    }
    // back-substitute to get the inverse
    lu_substitute(lu, pm, inverse);
  } catch (boost::numeric::ublas::internal_logic &error) {
    // coutE(Eval) << "boost::numberic::ublas error: matrix is not invertible!"
    // << std::endl;
  }
  SuperFloat inorm = norm_inf(inverse);
  SuperFloat condition = mnorm * inorm;
  return condition;
}

#else

#include "TDecompLU.h"
typedef TMatrixD Matrix;

////////////////////////////////////////////////////////////////////////////////
/// convert a matrix into a TMatrixD

inline TMatrixD makeRootMatrix(const Matrix &in) { return TMatrixD(in); }

////////////////////////////////////////////////////////////////////////////////
/// convert a TMatrixD into a Matrix

inline Matrix makeSuperMatrix(const TMatrixD &in) { return in; }

////////////////////////////////////////////////////////////////////////////////
/// create a new diagonal matrix of size n

inline Matrix diagMatrix(size_t n) {
  TMatrixD mat(n, n);
  mat.UnitMatrix();
  return mat;
}

////////////////////////////////////////////////////////////////////////////////
/// write a matrix

inline void printMatrix(const TMatrixD &mat) {
  writeMatrixToStreamT(mat, std::cout);
}

////////////////////////////////////////////////////////////////////////////////
// calculate the inverse of a matrix, returning the condition

inline double invertMatrix(const Matrix &matrix, Matrix &inverse) {
  TDecompLU lu(matrix);
  bool status = lu.Invert(inverse);
  // check if the matrix is invertible
  if (!status) {
    std::cerr << " matrix is not invertible!" << std::endl;
  }
  double condition = lu.GetCondition();
  const size_t n = size(inverse);
  // sanitize numeric problems
  for (size_t i = 0; i < n; ++i)
    for (size_t j = 0; j < n; ++j)
      if (fabs(inverse(i, j)) < 1e-9)
        inverse(i, j) = 0;
  return condition;
}
#endif

/////////////////////////////////////////////////////////////////////////////////
// LOCAL FUNCTIONS AND DEFINITIONS
// //////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////
/// anonymous namespace to prohibit use of these functions outside the class
/// itself
namespace {
///////////////////////////////////////////////////////////////////////////////
// HELPERS ////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

typedef std::vector<std::vector<bool>> FeynmanDiagram;
typedef std::vector<std::vector<int>> MorphFuncPattern;
typedef std::map<int, std::unique_ptr<RooAbsReal>> FormulaList;

///////////////////////////////////////////////////////////////////////////////
/// check if a std::string begins with the given character set

// inline bool begins_with(const std::string &input, const std::string &match) {
//   return input.size() >= match.size() &&
//          equal(match.begin(), match.end(), input.begin());
// }

///////////////////////////////////////////////////////////////////////////////
/// (-?-)

inline TString makeValidName(const char *input) {
  TString retval(input);
  retval.ReplaceAll("/", "_");
  retval.ReplaceAll("^", "");
  retval.ReplaceAll("*", "X");
  retval.ReplaceAll("[", "");
  retval.ReplaceAll("]", "");
  return retval;
}

//////////////////////////////////////////////////////////////////////////////
/// concatenate the names of objects in a collection to a single string

template <class List> std::string concatNames(const List &c, const char *sep) {
  std::stringstream ss;
  bool first = true;
  for (auto itr : c) {
    if (!first)
      ss << sep;
    ss << itr->GetName();
    first = false;
  }
  return ss.str();
}

///////////////////////////////////////////////////////////////////////////////
/// this is a workaround for the missing implicit conversion from
/// SuperFloat<>double

template <class A, class B> inline void assignElement(A &a, const B &b) {
  a = static_cast<A>(b);
}
///////////////////////////////////////////////////////////////////////////////
// read a matrix from a stream

template <class MatrixT>
inline MatrixT readMatrixFromStreamT(std::istream &stream) {
  std::vector<std::vector<SuperFloat>> matrix;
  std::vector<SuperFloat> line;
  while (!stream.eof()) {
    if (stream.peek() == '\n') {
      stream.get();
      stream.peek();
      continue;
    }
    SuperFloat val;
    stream >> val;
    line.push_back(val);
    while (stream.peek() == ' ' || stream.peek() == '\t') {
      stream.get();
    }
    if (stream.peek() == '\n') {
      matrix.push_back(line);
      line.clear();
    }
  }
  MatrixT retval(matrix.size(), matrix.size());
  for (size_t i = 0; i < matrix.size(); ++i) {
    if (matrix[i].size() != matrix.size()) {
      std::cerr << "matrix read from stream doesn't seem to be square!"
                << std::endl;
    }
    for (size_t j = 0; j < matrix[i].size(); ++j) {
      assignElement(retval(i, j), matrix[i][j]);
    }
  }
  return retval;
}

///////////////////////////////////////////////////////////////////////////////
/// read a matrix from a text file

template <class MatrixT> inline MatrixT readMatrixFromFileT(const char *fname) {
  std::ifstream in(fname);
  if (!in.good()) {
    std::cerr << "unable to read file '" << fname << "'!" << std::endl;
  }
  MatrixT mat = readMatrixFromStreamT<MatrixT>(in);
  in.close();
  return mat;
}

///////////////////////////////////////////////////////////////////////////////
/// convert a TH1* param hist into the corresponding ParamSet object

template <class T>
void readValues(std::map<const std::string, T> & myMap, TH1 *h_pc) {
  if (h_pc) {
    // loop over all bins of the param_card histogram
    for (int ibx = 1; ibx <= h_pc->GetNbinsX(); ++ibx) {
      // read the value of one parameter
      const std::string s_coup(h_pc->GetXaxis()->GetBinLabel(ibx));
      double coup_val = h_pc->GetBinContent(ibx);
      // add it to the map
      if (!s_coup.empty()) {
        myMap[s_coup] = T(coup_val);
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
/// retrieve a param_hist from a certain subfolder 'name' of the file

inline TH1F *getParamHist(TDirectory *file, const std::string &name,
                          const std::string &objkey = "param_card",
                          bool notFoundError = true) {
  auto f_tmp = file->Get<TFolder>(name.c_str());
  if (!f_tmp) {
    std::cerr << "unable to retrieve folder '" << name << "' from file '"
              << file->GetName() << "'!" << std::endl;
    return nullptr;
}
  // retrieve the histogram param_card which should live directly in the folder
  TH1F *h_pc = dynamic_cast<TH1F *>(f_tmp->FindObject(objkey.c_str()));
  if (h_pc) {
    return h_pc;
  }
  if (notFoundError) {
    std::cerr << "unable to retrieve " << objkey << " histogram from folder '"
              << name << "'" << std::endl;
  }
  return nullptr;
}

///////////////////////////////////////////////////////////////////////////////
/// retrieve a ParamSet from a certain subfolder 'name' of the file

template <class T>
void readValues(std::map<const std::string, T> & myMap,
           TDirectory *file, const std::string &name,
           const std::string &key = "param_card", bool notFoundError = true) {
  TH1F *h_pc = getParamHist(file, name, key, notFoundError);
  readValues(myMap, h_pc);
}

///////////////////////////////////////////////////////////////////////////////
/// retrieve the param_hists file and return a map of the parameter values
/// by providing a list of names, only the param_hists of those subfolders are
/// read leaving the list empty is interpreted as meaning 'read everything'

template <class T>
void readValues(std::map<const std::string, std::map<const std::string, T>> & inputParameters,
           TDirectory *f, const std::vector<std::string> &names,
           const std::string &key = "param_card", bool notFoundError = true) {
  inputParameters.clear();
  // if the list of names is empty, we assume that this means 'all'
  // loop over all folders in the file
  for (size_t i = 0; i < names.size(); i++) {
    const std::string name(names[i]);
    // actually read an individual param_hist
    // std::cout << "reading " << key << " '" << name << "'!" << std::endl;
    readValues(inputParameters[name], f, name, key, notFoundError);
  }

  // now the map is filled with all parameter values found for all samples
}

///////////////////////////////////////////////////////////////////////////////
/// open the file and return a file pointer

inline TDirectory *openFile(const std::string &filename) {
  if (filename.empty()) {
    return gDirectory;
  } else {
    TFile *file = TFile::Open(filename.c_str(), "READ");
    if (!file || !file->IsOpen()) {
      if (file)
        delete file;
      std::cerr << "could not open file '" << filename << "'!" << std::endl;
    }
    return file;
  }
}

///////////////////////////////////////////////////////////////////////////////
/// open the file and return a file pointer

inline void closeFile(TDirectory * d) {
  TFile *f = dynamic_cast<TFile *>(d);
  if (f) {
    f->Close();
    delete f;
    d = nullptr;
  }
}

///////////////////////////////////////////////////////////////////////////////
/// extract the operators from a single coupling
template <class T2>
inline void extractServers(const RooAbsArg &coupling, T2 &operators) {
  int nservers = 0;
  for (const auto server : coupling.servers()) {
    extractServers(*server, operators);
    nservers++;
  }
  if (nservers == 0) {
    operators.add(coupling);
  }
}

///////////////////////////////////////////////////////////////////////////////
/// extract the operators from a list of couplings

template <class T1, class T2,
          typename std::enable_if<!is_specialization<T1, std::vector>::value,
                                  T1>::type * = nullptr>
inline void extractOperators(const T1 &couplings, T2 &operators) {
  // coutD(InputArguments) << "extracting operators from
  // "<<couplings.getSize()<<" couplings" << std::endl;
  for (auto itr : couplings) {
    extractServers(*itr, operators);
  }
}

///////////////////////////////////////////////////////////////////////////////
/// extract the operators from a list of vertices

template <class T1, class T2,
          typename std::enable_if<is_specialization<T1, std::vector>::value,
                                  T1>::type * = nullptr>
inline void extractOperators(const T1 &vec, T2 &operators) {
  for (const auto &v : vec) {
    extractOperators(v, operators);
  }
}

///////////////////////////////////////////////////////////////////////////////
/// extract the couplings from a given set and copy them to a new one

template <class T1, class T2>
inline void extractCouplings(const T1 &inCouplings, T2 &outCouplings) {
  for (auto itr : inCouplings) {
    if (!outCouplings.find(itr->GetName())) {
      // coutD(InputArguments) << "adding parameter "<< obj->GetName() <<
      // std::endl;
      outCouplings.add(*itr);
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
/// find and, if necessary, create a parameter from a list

template <class T>
inline RooAbsArg &get(T &operators, const char *name, double defaultval = 0) {
  RooAbsArg *kappa = operators.find(name);
  if (kappa)
    return *kappa;
  RooRealVar *newKappa = new RooRealVar(name, name, defaultval);
  newKappa->setConstant(false);
  operators.add(*newKappa);
  return *newKappa;
}

///////////////////////////////////////////////////////////////////////////////
/// find and, if necessary, create a parameter from a list

template <class T>
inline RooAbsArg &get(T &operators, const std::string &name,
                      double defaultval = 0) {
  return get(operators, name.c_str(), defaultval);
}

////////////////////////////////////////////////////////////////////////////////
/// create a new coupling and add it to the set

template <class T>
inline void addCoupling(T &set, const TString &name, const TString &formula,
                        const RooArgList &components, bool isNP) {
  if (!set.find(name)) {
    RooFormulaVar *c = new RooFormulaVar(name, formula, components);
    c->setAttribute("NP", isNP);
    set.add(*c);
  }
}

////////////////////////////////////////////////////////////////////////////////
/// set parameter values first set all values to defaultVal (if value not
/// present in param_card then it should be 0)

inline bool setParam(RooRealVar *p, double val, bool force) {
  // coutD(InputArguments) << "setparam for "<<p->GetName()<<" to "<<val <<
  // std::endl;
  bool ok = true;
  if (val > p->getMax()) {
    if (force) {
      p->setMax(val);
    } else {
      std::cerr << ": parameter " << p->GetName() << " out of bounds: " << val
                << " > " << p->getMax() << std::endl;
      ok = false;
    }
  } else if (val < p->getMin()) {
    if (force) {
      p->setMin(val);
    } else {
      std::cerr << ": parameter " << p->GetName() << " out of bounds: " << val
                << " < " << p->getMin() << std::endl;
      ok = false;
    }
  }
  if (ok)
    p->setVal(val);
  return ok;
}

////////////////////////////////////////////////////////////////////////////////
/// set parameter values first set all values to defaultVal (if value not
/// present in param_card then it should be 0)

template <class T1, class T2> inline bool setParams(const T2 &args, T1 val) {
  for (auto itr : args) {
    RooRealVar *param = dynamic_cast<RooRealVar *>(itr);
    if (!param)
      continue;
    setParam(param, val, true);
  }
  return true;
}

////////////////////////////////////////////////////////////////////////////////
/// set parameter values first set all values to defaultVal (if value not
/// present in param_card then it should be 0)

template <class T1, class T2>
inline bool setParams(const std::map<const std::string, T1> &point,
                      const T2 &args, bool force = false, T1 defaultVal = 0) {
  bool ok = true;
  for (auto itr : args) {
    RooRealVar *param = dynamic_cast<RooRealVar *>(itr);
    if (!param || param->isConstant())
      continue;
    ok = setParam(param, defaultVal, force) && ok;
  }
  // set all parameters to the values in the param_card histogram
  for (auto paramit : point) {
    // loop over all the parameters
    const std::string param(paramit.first);
    // retrieve them from the map
    RooRealVar *p = dynamic_cast<RooRealVar *>(args.find(param.c_str()));
    if (!p)
      continue;
    // set them to their nominal value
    ok = setParam(p, paramit.second, force) && ok;
  }
  return ok;
}

////////////////////////////////////////////////////////////////////////////////
/// set parameter values first set all values to defaultVal (if value not
/// present in param_card then it should be 0)

template <class T>
inline bool setParams(TH1 *hist, const T &args, bool force = false) {
  bool ok = true;

  for (auto itr : args) {
    RooRealVar *param = dynamic_cast<RooRealVar *>(itr);
    if (!param)
      continue;
    ok = setParam(param, 0., force) && ok;
  }

  // set all parameters to the values in the param_card histogram
  TAxis *ax = hist->GetXaxis();
  for (int i = 1; i <= ax->GetNbins(); ++i) {
    // loop over all the parameters
    RooRealVar *p = dynamic_cast<RooRealVar *>(args.find(ax->GetBinLabel(i)));
    if (!p)
      continue;
    // set them to their nominal value
    ok = setParam(p, hist->GetBinContent(i), force) && ok;
  }
  return ok;
}

////////////////////////////////////////////////////////////////////////////////
/// create a set of parameters

template <class T>
inline RooLagrangianMorphFunc::ParamSet getParams(const T &parameters) {
  RooLagrangianMorphFunc::ParamSet retval;
  for (auto itr : parameters) {
    RooRealVar *param = dynamic_cast<RooRealVar *>(itr);
    if (!param)
      continue;
    retval[param->GetName()] = param->getVal();
  }
  return retval;
}

////////////////////////////////////////////////////////////////////////////////
/// collect the histograms from the input file and convert them to RooFit
/// objects

void collectHistograms(
    const char *name, TDirectory *file, std::map<std::string, int> &list_hf,
    RooArgList &physics, RooRealVar &var, const std::string &varname,
    const RooLagrangianMorphFunc::ParamMap &inputParameters) {
  // oocoutD((TObject*)0,InputArguments) << "building list of histogram
  // functions" << std::endl;
  bool binningOK = false;
  for (auto sampleit : inputParameters) {
    const std::string sample(sampleit.first);
    auto folder = file->Get<TFolder>(sample.c_str());
    if (!folder) {
      std::cerr << "Error: unable to access data from folder '" << sample
                << "'!" << std::endl;
      continue;
    }
    TH1 *hist = dynamic_cast<TH1 *>(folder->FindObject(varname.c_str()));
    if (!hist) {
      std::stringstream errstr;
      errstr << "Error: unable to retrieve histogram '" << varname
             << "' from folder '" << sample << "'. contents are:";
      TIter next(folder->GetListOfFolders()->begin());
      TFolder *f;
      while ((f = (TFolder *)next())) {
        errstr << " " << f->GetName();
      }
      std::cerr << errstr.str() << std::endl;
      return;
    }

    auto it = list_hf.find(sample);
    if (it != list_hf.end()) {
      RooHistFunc *hf = (RooHistFunc *)(physics.at(it->second));
      hf->setValueDirty();
      // commenting out To-be-resolved
      // RooDataHist* dh = &(hf->dataHist());
      // RooLagrangianMorphFunc::setDataHistogram(hist,&var,dh);
      // RooArgSet vars;
      // vars.add(var);
      // dh->importTH1(vars,*hist,1.,kFALSE);
    } else {
      if (!binningOK) {
        int n = hist->GetNbinsX();
        std::vector<double> bins;
        for (int i = 1; i < n + 1; ++i) {
          bins.push_back(hist->GetBinLowEdge(i));
        }
        bins.push_back(hist->GetBinLowEdge(n) + hist->GetBinWidth(n));
        var.setBinning(RooBinning(n, &(bins[0])));
      }

      // generate the mean value
      TString histname = makeValidName(Form("dh_%s_%s", sample.c_str(), name));
      TString funcname =
          makeValidName(Form("phys_%s_%s", sample.c_str(), name));
      RooArgSet vars;
      vars.add(var);

      RooDataHist *dh = new RooDataHist(histname, histname, vars, hist);
      // add it to the list
      RooHistFunc *hf = new RooHistFunc(funcname, funcname, var, *dh);
      int idx = physics.getSize();
      list_hf[sample] = idx;
      physics.add(*hf);
      assert(hf = static_cast<RooHistFunc *>(physics.at(idx)));
    }
    // std::cout << "found histogram " << hist->GetName() << " with integral "
    // << hist->Integral() << std::endl;
  }
}

////////////////////////////////////////////////////////////////////////////////
/// collect the RooAbsReal objects from the input directory

void collectRooAbsReal(
    const char * /*name*/, TDirectory *file,
    std::map<std::string, int> &list_hf, RooArgList &physics,
    const std::string &varname,
    const RooLagrangianMorphFunc::ParamMap &inputParameters) {
  // cxcoutP(ObjectHandling) << "building list of RooAbsReal objects" <<
  // std::endl;
  for (auto sampleit : inputParameters) {
    const std::string sample(sampleit.first);
    auto folder = file->Get<TFolder>(sample.c_str());
    if (!folder) {
      std::cerr << "Error: unable to access data from folder '" << sample
                << "'!" << std::endl;
      continue;
    }

    RooAbsReal *obj =
        dynamic_cast<RooAbsReal *>(folder->FindObject(varname.c_str()));
    if (!obj) {
      std::stringstream errstr;
      std::cerr << "Error: unable to retrieve RooAbsArg '" << varname
                << "' from folder '" << sample
                << "'. contents are:" << std::endl;
      TIter next(folder->GetListOfFolders()->begin());
      TFolder *f;
      while ((f = (TFolder *)next())) {
        errstr << " " << f->GetName();
      }
      std::cerr << errstr.str() << std::endl;
      return;
    }
    auto it = list_hf.find(sample);
    if (it == list_hf.end()) {
      int idx = physics.getSize();
      list_hf[sample] = idx;
      physics.add(*obj);
      assert(obj == physics.at(idx));
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
/// collect the TParameter objects from the input file and convert them to
/// RooFit objects

template <class T>
void collectCrosssections(
    const char *name, TDirectory *file, std::map<std::string, int> &list_xs,
    RooArgList &physics, const std::string &varname,
    const RooLagrangianMorphFunc::ParamMap &inputParameters) {
  //  cxcoutP(ObjectHandling) << "building list of histogram functions" <<
  //  std::endl;
  for (auto sampleit : inputParameters) {
    const std::string sample(sampleit.first);
    auto folder = file->Get<TFolder>(sample.c_str());
    if (!folder) {
      std::cerr << "unable to access data from folder '" << sample << "'!"
                << std::endl;
      return;
    }
    TObject *obj = folder->FindObject(varname.c_str());
    TParameter<T> *xsection = nullptr;
    TParameter<T> *error = nullptr;
    TParameter<T> *p = dynamic_cast<TParameter<T> *>(obj);
    if (p) {
      xsection = p;
    }
    TPair *pair = dynamic_cast<TPair *>(obj);
    if (pair) {
      xsection = dynamic_cast<TParameter<T> *>(pair->Key());
      error = dynamic_cast<TParameter<T> *>(pair->Value());
    }
    if (!xsection) {
      std::stringstream errstr;
      errstr << "Error: unable to retrieve cross section '" << varname
             << "' from folder '" << sample << "'. contents are:";
      TIter next(folder->GetListOfFolders()->begin());
      TFolder *f;
      while ((f = (TFolder *)next())) {
        errstr << " " << f->GetName();
      }
      std::cerr << errstr.str() << std::endl;
      return;
    }

    auto it = list_xs.find(sample.c_str());
    RooRealVar *xs;
    if (it != list_xs.end()) {
      xs = (RooRealVar *)(physics.at(it->second));
      xs->setVal(xsection->GetVal());
    } else {
      std::string objname = Form("phys_%s_%s", name, sample.c_str());
      xs = new RooRealVar(objname.c_str(), objname.c_str(), xsection->GetVal());
      xs->setConstant(true);
      int idx = physics.getSize();
      list_xs[sample] = idx;
      physics.add(*xs);
      assert(physics.at(idx) == xs);
    }
    if (error)
      xs->setError(error->GetVal());
  }
}

////////////////////////////////////////////////////////////////////////////////
/// collect the TPair<TParameter,TParameter> objects from the input file and
/// convert them to RooFit objects

void collectCrosssectionsTPair(
    const char *name, TDirectory *file, std::map<std::string, int> &list_xs,
    RooArgList &physics, const std::string &varname,
    const std::string &basefolder,
    const RooLagrangianMorphFunc::ParamMap &inputParameters) {
  auto folder = file->Get<TFolder>(basefolder.c_str());
  TPair *pair = dynamic_cast<TPair *>(folder->FindObject(varname.c_str()));
  TParameter<double> *xsec_double =
      dynamic_cast<TParameter<double> *>(pair->Key());
  if (xsec_double) {
    collectCrosssections<double>(name, file, list_xs, physics, varname,
                                 inputParameters);
  } else {
    TParameter<float> *xsec_float =
        dynamic_cast<TParameter<float> *>(pair->Key());
    if (xsec_float) {
      collectCrosssections<float>(name, file, list_xs, physics, varname,
                                  inputParameters);
    } else {
      std::cerr << "cannot morph objects of class 'TPair' if parameter is not "
                   "double or float!"
                << std::endl;
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
// FORMULA CALCULATION ////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
/// recursive function to determine polynomials

void collectPolynomialsHelper(const FeynmanDiagram &diagram,
                              MorphFuncPattern &morphfunc,
                              std::vector<int> &term, int vertexid,
                              bool first) {
  if (vertexid > 0) {
    for (size_t i = 0; i < diagram[vertexid - 1].size(); ++i) {
      if (!diagram[vertexid - 1][i])
        continue;
      std::vector<int> newterm(term);
      newterm[i]++;
      if (first) {
        ::collectPolynomialsHelper(diagram, morphfunc, newterm, vertexid,
                                   false);
      } else {
        ::collectPolynomialsHelper(diagram, morphfunc, newterm, vertexid - 1,
                                   true);
      }
    }
  } else {
    bool found = false;
    for (size_t i = 0; i < morphfunc.size(); ++i) {
      bool thisfound = true;
      for (size_t j = 0; j < morphfunc[i].size(); ++j) {
        if (morphfunc[i][j] != term[j]) {
          thisfound = false;
          break;
        }
      }
      if (thisfound) {
        found = true;
        break;
      }
    }
    if (!found) {
      morphfunc.push_back(term);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
/// calculate the morphing function pattern based on a vertex map

void collectPolynomials(MorphFuncPattern &morphfunc,
                        const FeynmanDiagram &diagram) {
  int nvtx(diagram.size());
  std::vector<int> term(diagram[0].size(), 0);

  ::collectPolynomialsHelper(diagram, morphfunc, term, nvtx, true);
}

////////////////////////////////////////////////////////////////////////////////
/// build a vertex map based on vertices and couplings appearing

template <class List>
inline void fillFeynmanDiagram(FeynmanDiagram &diagram,
                               const std::vector<List *> &vertices,
                               RooArgList &couplings) {
  const int ncouplings = couplings.getSize();
  // std::cout << "Number of couplings " << ncouplings << std::endl;
  for (auto const& vertex : vertices) {
    std::vector<bool> vertexCouplings(ncouplings, false);
    int idx = -1;
    RooAbsReal *coupling;
    for (auto citr : couplings) {
      coupling = dynamic_cast<RooAbsReal *>(citr);
      idx++;
      if (!coupling) {
        std::cerr << "encountered invalid list of couplings in vertex!"
                  << std::endl;
        return;
      }
      if (vertex->find(coupling->GetName())) {
        vertexCouplings[idx] = true;
      }
    }
    diagram.push_back(vertexCouplings);
  }
}

////////////////////////////////////////////////////////////////////////////////
/// fill the matrix of coefficients

template <class MatrixT, class T1, class T2>
inline MatrixT
buildMatrixT(const RooLagrangianMorphFunc::ParamMap &inputParameters,
             const FormulaList &formulas, const T1 &args,
             const RooLagrangianMorphFunc::FlagMap &flagValues,
             const T2 &flags) {
  const size_t dim = inputParameters.size();
  MatrixT matrix(dim, dim);
  int row = 0;
  for (auto sampleit : inputParameters) {
    const std::string sample(sampleit.first);
    // set all vars to value stored in input file
    if (!setParams<double>(sampleit.second, args, true, 0)) {
      // std::cout << "unable to set parameters for sample "<<sample<<"!" <<
      // std::endl;
    }
    auto flagit = flagValues.find(sample);
    if (flagit != flagValues.end() &&
        !setParams<int>(flagit->second, flags, true, 1)) {
      // std::cout << "unable to set parameters for sample "<<sample<<"!" <<
      // std::endl;
    }
    // loop over all the formulas
    int col = 0;
    for (auto const &formula : formulas) {
      if (!formula.second) {
        std::cerr << "Error: invalid formula encountered!" << std::endl;
      }
      matrix(row, col) = formula.second->getVal();
      // std::cout << formula.second->getVal() << " = " <<
      // formula.second->GetTitle() << " for " << sample << std::endl;
      col++;
    }
    row++;
  }
  return matrix;
}

////////////////////////////////////////////////////////////////////////////////
/// check if the matrix is square

inline void checkMatrix(const RooLagrangianMorphFunc::ParamMap &inputParameters,
                        const FormulaList &formulas) {
  if (inputParameters.size() != formulas.size()) {
    std::stringstream ss;
    ss << "matrix is not square, consistency check failed: "
       << inputParameters.size() << " samples, " << formulas.size()
       << " expressions:" << std::endl;
    ss << "formulas: " << std::endl;
    for (auto const &formula : formulas) {
      ss << formula.second->GetTitle() << std::endl;
    }
    ss << "samples: " << std::endl;
    for (auto sample : inputParameters) {
      ss << sample.first << std::endl;
    }
    std::cerr << ss.str() << std::endl;
  }
}

////////////////////////////////////////////////////////////////////////////////
/// check if the entries in the inverted matrix are sensible

inline void inverseSanity(const Matrix &matrix, const Matrix &inverse,
                          double &unityDeviation, double &largestWeight) {
  // cxcoutP(Eval) << "multiplying for sanity check" << std::endl;
  Matrix unity(inverse * matrix);

  unityDeviation = 0.;
  largestWeight = 0.;
  const size_t dim = size(unity);
  for (size_t i = 0; i < dim; ++i) {
    for (size_t j = 0; j < dim; ++j) {
      if (inverse(i, j) > largestWeight) {
        largestWeight = (double)inverse(i, j);
      }
      if (fabs(unity(i, j) - static_cast<int>(i == j)) > unityDeviation) {
        unityDeviation = fabs((double)unity(i, j)) - static_cast<int>(i == j);
      }
    }
  }
  // std::cout << "found deviation of " << unityDeviation << ", largest weight
  // is " << largestWeight << "." << std::endl;
}

////////////////////////////////////////////////////////////////////////////////
/// check for name conflicts between the input samples and an argument set
template <class List>
inline void
checkNameConflict(const RooLagrangianMorphFunc::ParamMap &inputParameters,
                  List &args) {
  for (auto sampleit : inputParameters) {
    const std::string sample(sampleit.first);
    RooAbsArg *arg = args.find(sample.c_str());
    if (arg) {
      std::cerr << "detected name conflict: cannot use sample '" << sample
                << "' - a parameter with the same name of type '"
                << arg->ClassName() << "' is present in set '" << args.GetName()
                << "'!" << std::endl;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
/// build the formulas corresponding to the given set of input files and
///  the physics process

template <class List>
inline FormulaList
buildFormulas(const char *mfname,
              const RooLagrangianMorphFunc::ParamMap &inputParameters,
              const RooLagrangianMorphFunc::FlagMap &inputFlags,
              const MorphFuncPattern &morphfunc, const RooArgList &couplings,
              const List &flags, const std::vector<List *> &nonInterfering) {
  // example vbf hww:
  //                        Operators kSM,  kHww, kAww, kHdwR,kHzz, kAzz
  // std::vector<bool> vertexProd  = {true, true, true, true, true, true };
  // std::vector<bool> vertexDecay = {true, true, true, true, false,false};
  // diagram.push_back(vertexProd);
  // diagram.push_back(vertexDecay);

  const int ncouplings = couplings.getSize();
  std::vector<bool> couplingsZero(ncouplings, true);
  std::map<TString, bool> flagsZero;

  // std::cout << "number of couplings : " << ncouplings << std::endl;
  RooArgList operators;
  extractOperators(couplings, operators);
  size_t nOps = operators.getSize();
  // std::cout << "number of operators : " << nOps << std::endl;

  for (auto sampleit : inputParameters) {
    // std::cout << "inside inputParameters loop" << std::endl;
    const std::string sample(sampleit.first);
    if (!setParams(sampleit.second, operators, true)) {
      std::cerr << "unable to set parameters for sample '" << sample << "'!"
                << std::endl;
    }

    if ((int)nOps != (operators.getSize())) {
      std::cerr << "internal error, number of operators inconsistent!"
                << std::endl;
    }

    RooAbsReal *obj0;
    int idx = 0;

    for (auto itr1 : couplings) {
      obj0 = dynamic_cast<RooAbsReal *>(itr1);
      if (obj0->getVal() != 0) {
        // std::cout << obj0->GetName() << " is non-zero for sample " << sample
        // << " (idx=" << idx << ")!" << std::endl;
        couplingsZero[idx] = false;
      } else {
        // std::cout << obj0->GetName() << " is zero for sample " << sample << "
        // (idx=" << idx << ")!" << std::endl;
      }
      idx++;
    }
  }

  for (auto itr2 : flags) {
    auto obj1 = dynamic_cast<RooAbsReal *>(itr2);
    int nZero = 0;
    int nNonZero = 0;
    for (auto sampleit : inputFlags) {
      const auto &flag = sampleit.second.find(obj1->GetName());
      if (flag != sampleit.second.end()) {
        if (flag->second == 0.)
          nZero++;
        else
          nNonZero++;
        // std::cout << "flag found " << obj1->GetName() << ", value = " <<
        // flag->second << std::endl;
      } else {
        // std::cout << "flag not found " << obj1->GetName() << std::endl;
      }
    }
    if (nZero > 0 && nNonZero == 0)
      flagsZero[obj1->GetName()] = true;
    else
      flagsZero[obj1->GetName()] = false;
  }

  //    {
  //      int idx = 0;
  //      RooAbsReal* obj2;
  //      for(auto itr = couplings.begin(); itr != couplings.end(); ++itr){
  //        obj2 = dynamic_cast<RooAbsReal*>(*itr);
  //        if(couplingsZero[idx]){
  //          //coutD(ObjectHandling) << obj2->GetName() << " is zero (idx=" <<
  //          idx << ")" << std::endl;
  //        } else {
  //          //coutD(ObjectHandling) << obj2->GetName() << " is non-zero (idx="
  //          << idx << ")" << std::endl;
  //        }
  //        idx++;
  //      }
  //    }

  FormulaList formulas;
  for (size_t i = 0; i < morphfunc.size(); ++i) {
    RooArgList ss;
    bool isZero = false;
    std::string reason;
    // check if this is a blacklisted interference term
    for (const auto &group : nonInterfering) {
      int nInterferingOperators = 0;
      for (size_t j = 0; j < morphfunc[i].size(); ++j) {
        if (morphfunc[i][j] % 2 == 0)
          continue; // even exponents are not interference terms
        if (group->find(couplings.at(j)
                            ->GetName())) { // if the coupling is part of a
                                            // "pairwise non-interfering group"
          nInterferingOperators++;
        }
      }
      if (nInterferingOperators > 1) {
        isZero = true;
        reason = "blacklisted interference term!";
      }
    }
    int nNP = 0;
    if (!isZero) {
      // prepare the term
      for (size_t j = 0; j < morphfunc[i].size(); ++j) {
        const int exponent = morphfunc[i][j];
        if (exponent == 0)
          continue;
        RooAbsReal *coupling = dynamic_cast<RooAbsReal *>(couplings.at(j));
        for (int k = 0; k < exponent; ++k) {
          ss.add(*coupling);
          if (coupling->getAttribute("NP")) {
            nNP++;
          }
        }
        std::string cname(coupling->GetName());
        if (coupling->getAttribute("LO") && exponent > 1) {
          isZero = true;
          reason = "coupling " + cname + " was listed as leading-order-only";
        }
        // mark the term as zero if any of the couplings are zero
        if (!isZero && couplingsZero[j]) {
          isZero = true;
          reason = "coupling " + cname + " is zero!";
        }
      }
    }
    // check and apply flags
    bool removedByFlag = false;

    for (auto itr : flags) {
      auto obj = dynamic_cast<RooAbsReal *>(itr);
      if (!obj)
        continue;
      TString sval(obj->getStringAttribute("NP"));
      int val = atoi(sval);
      if (val == nNP) {
        if (flagsZero.find(obj->GetName()) != flagsZero.end() &&
            flagsZero.at(obj->GetName())) {
          removedByFlag = true;
          reason = Form("flag %s is zero", obj->GetName());
        }
        ss.add(*obj);
      }
    }

    // create and add the formula
    if (!isZero && !removedByFlag) {
      // build the name
      const auto name = std::string(mfname) + "_pol" + std::to_string(i);
      //        formulas[i] = new RooProduct(name.Data(),::concatNames(ss," *
      //        ").c_str(),ss);
      formulas[i] = std::make_unique<RooProduct>(
          name.c_str(), ::concatNames(ss, " * ").c_str(), ss);
      std::cout << "creating formula " << name << ": "
                << formulas[i]->GetTitle() << std::endl;
    } else {
      // print a message and continue without doing anything
      // std::cout << "killing formula " << ::concatNames(ss," * ") << " because
      // " << reason << std::endl;
    }
  }
  return formulas;
}

////////////////////////////////////////////////////////////////////////////////
/// create the weight formulas required for the morphing

template <class T>
inline FormulaList
createFormulas(const char *name, const RooLagrangianMorphFunc::ParamMap &inputs,
               const RooLagrangianMorphFunc::FlagMap &inputFlags,
               const std::vector<std::vector<T *>> &diagrams,
               RooArgList &couplings, const T &flags,
               const std::vector<T *> &nonInterfering) {
  MorphFuncPattern morphfuncpattern;
  // std::cout << "building vertex map" << std::endl;

  for (const auto &vertices : diagrams) {
    FeynmanDiagram d;
    // std::cout << "building vertex map" << std::endl;
    ::fillFeynmanDiagram(d, vertices, couplings);
    // std::cout << "collecting polynomials for diagram of size " << d.size() <<
    // std::endl;
    ::collectPolynomials(morphfuncpattern, d);
  }
  // coutD(ObjectHandling) << "building formulas" << std::endl;
  FormulaList retval = buildFormulas(name, inputs, inputFlags, morphfuncpattern,
                                     couplings, flags, nonInterfering);
  if (retval.size() == 0) {
    // std::cout << "no formulas are non-zero, check if any if your couplings is
    // floating and missing from your param_cards!" << std::endl;
  }
  // coutD(ObjectHandling) << "checking matrix consistency" << std::endl;
  checkMatrix(inputs, retval);
  return retval;
}

////////////////////////////////////////////////////////////////////////////////
/// (-?-)
//
template <class T1>
inline void
buildSampleWeights(T1 &weights, const char *fname,
                   const RooLagrangianMorphFunc::ParamMap &inputParameters,
                   FormulaList &formulas, const Matrix &inverse) {
  int sampleidx = 0;

  for (auto sampleit : inputParameters) {
    const std::string sample(sampleit.first);
    std::stringstream title;
    TString name_full(makeValidName(sample.c_str()));
    if (fname) {
      name_full.Append("_");
      name_full.Append(fname);
      name_full.Prepend("w_");
    }

    int formulaidx = 0;
    // build the formula with the correct normalization
    RooLinearCombination *sampleformula = new RooLinearCombination(name_full.Data());
    for (auto const &formulait : formulas) {
      const SuperFloat val(inverse(formulaidx, sampleidx));
      sampleformula->add(val, formulait.second.get());
      formulaidx++;
    }
    weights.add(*sampleformula);
    sampleidx++;
  }
  // coutD(ObjectHandling) << "done building sample weights" << std::endl;
}

inline std::map<std::string, std::string> buildSampleWeightStrings(
    const RooLagrangianMorphFunc::ParamMap &inputParameters,
    const FormulaList &formulas, const Matrix &inverse) {
  int sampleidx = 0;
  std::map<std::string, std::string> weights;
  for (auto sampleit : inputParameters) {
    const std::string sample(sampleit.first);
    std::stringstream str;
    int formulaidx = 0;
    // build the formula with the correct normalization
    for (auto const &formulait : formulas) {
      double val(inverse(formulaidx, sampleidx));
      if (val != 0.) {
        if (formulaidx > 0 && val > 0)
          str << " + ";
        str << val << "*(" << formulait.second->GetTitle() << ")";
      }
      formulaidx++;
    }
    weights[sample] = str.str();
    // std::cout << str.str() << std::endl;
    sampleidx++;
  }
  return weights;
}
} // namespace

///////////////////////////////////////////////////////////////////////////////
// CacheElem magic ////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

class RooLagrangianMorphFunc::CacheElem : public RooAbsCacheElement {
public:
  std::unique_ptr<RooRealSumFunc> _sumFunc = nullptr;
  RooArgList _couplings;

  FormulaList _formulas;
  RooArgList _weights;

  Matrix _matrix;
  Matrix _inverse;
  double _condition;

  CacheElem(){};
  virtual void operModeHook(RooAbsArg::OperMode) override{};

  //////////////////////////////////////////////////////////////////////////////
  /// retrieve the list of contained args

  virtual RooArgList containedArgs(Action) override {
    RooArgList args(*_sumFunc);
    args.add(_weights);
    args.add(_couplings);
    for (auto const &it : _formulas) {
      args.add(*(it.second));
    }
    return args;
  }

  //////////////////////////////////////////////////////////////////////////////
  // default destructor

  virtual ~CacheElem() {}

  //////////////////////////////////////////////////////////////////////////////
  /// create the basic objects required for the morphing

  inline void
  createComponents(const RooLagrangianMorphFunc::ParamMap &inputParameters,
                   const RooLagrangianMorphFunc::FlagMap &inputFlags,
                   const char *funcname,
                   const std::vector<std::vector<RooListProxy *>> &diagrams,
                   const std::vector<RooListProxy *> &nonInterfering,
                   const RooListProxy &flags) {
    RooArgList operators;
    // coutD(ObjectHandling) << "collecting couplings" << std::endl;
    for (const auto &diagram : diagrams) {
      for (const auto &vertex : diagram) {
        extractCouplings(*vertex, this->_couplings);
      }
    }
    extractOperators(this->_couplings, operators);
    this->_formulas =
        ::createFormulas(funcname, inputParameters, inputFlags, diagrams,
                         this->_couplings, flags, nonInterfering);
  }

  //////////////////////////////////////////////////////////////////////////////
  /// build and invert the morphing matrix
  template <class List>
  inline void
  buildMatrix(const RooLagrangianMorphFunc::ParamMap &inputParameters,
              const RooLagrangianMorphFunc::FlagMap &inputFlags,
              const List &flags) {
    RooArgList operators;
    extractOperators(this->_couplings, operators);
    Matrix matrix(buildMatrixT<Matrix>(inputParameters, this->_formulas,
                                       operators, inputFlags, flags));
    if (size(matrix) < 1) {
      std::cerr
          << "input matrix is empty, please provide suitable input samples!"
          << std::endl;
    }
    Matrix inverse(diagMatrix(size(matrix)));

    double condition = (double)(invertMatrix(matrix, inverse));
    // coutD(Eval) << "Condition of the matrix :" << condition << std::endl;

    double unityDeviation, largestWeight;
    inverseSanity(matrix, inverse, unityDeviation, largestWeight);
    bool weightwarning(largestWeight > morphLargestWeight ? true : false);
    bool unitywarning(unityDeviation > morphUnityDeviation ? true : false);

    if (false) {
      if (unitywarning) {
        oocxcoutW((TObject *)0, Eval)
            << "Warning: The matrix inversion seems to be unstable. This can "
               "be a result to input samples that are not sufficiently "
               "different to provide any morphing power."
            << std::endl;
      } else if (weightwarning) {
        oocxcoutW((TObject *)0, Eval)
            << "Warning: Some weights are excessively large. This can be a "
               "result to input samples that are not sufficiently different to "
               "provide any morphing power."
            << std::endl;
      }
      oocxcoutW((TObject *)0, Eval) << "         Please consider the couplings "
                                       "encoded in your samples to cross-check:"
                                    << std::endl;
      for (auto sampleit : inputParameters) {
        const std::string sample(sampleit.first);
        oocxcoutW((TObject *)0, Eval) << "         " << sample << ": ";
        // set all vars to value stored in input file
        setParams(sampleit.second, operators, true);
        bool first = true;
        RooAbsReal *obj;

        for (auto itr : this->_couplings) {
          obj = dynamic_cast<RooAbsReal *>(itr);
          if (!first)
            std::cerr << ", ";
          oocxcoutW((TObject *)0, Eval)
              << obj->GetName() << "=" << obj->getVal();
          first = false;
        }
        oocxcoutW((TObject *)0, Eval) << std::endl;
      }
    }
#ifndef USE_UBLAS
    this->_matrix.ResizeTo(matrix.GetNrows(), matrix.GetNrows());
    this->_inverse.ResizeTo(matrix.GetNrows(), matrix.GetNrows());
#endif
    this->_matrix = matrix;
    this->_inverse = inverse;
    this->_condition = condition;
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// build the final morphing function

  inline void buildMorphingFunction(
      const char *name, const RooLagrangianMorphFunc::ParamMap &inputParameters,
      const std::map<std::string, int> &storage, const RooArgList &physics,
      bool allowNegativeYields, RooRealVar *observable, RooRealVar *binWidth) {
    if (!binWidth) {
      std::cerr << "invalid bin width given!" << std::endl;
      return;
    }
    if (!observable) {
      std::cerr << "invalid observable given!" << std::endl;
      return;
    }

    RooArgList operators;
    extractOperators(this->_couplings, operators);

    // retrieve the weights
    ::buildSampleWeights(this->_weights, name, inputParameters, this->_formulas,
                         this->_inverse);

    // coutD(ObjectHandling) << "creating RooProducts" << std::endl;
    // build the products of element and weight for each sample
    size_t i = 0;
    RooArgList sumElements;
    RooArgList scaleElements;
    for (auto sampleit : inputParameters) {
      // for now, we assume all the lists are nicely ordered
      TString prodname(makeValidName(sampleit.first.c_str()));

      RooAbsReal *obj =
          static_cast<RooAbsReal *>(physics.at(storage.at(prodname.Data())));

      // obj->Print();
      if (!obj) {
        std::cerr << "unable to access physics object for " << prodname
                  << std::endl;
        return;
      }

      RooAbsReal *weight = static_cast<RooAbsReal *>(this->_weights.at(i));

      // weight->Print();
      if (!weight) {
        std::cerr << "unable to access weight object for " << prodname
                  << std::endl;
        return;
      }
      prodname.Append("_");
      prodname.Append(name);
      RooArgList prodElems(*weight, *obj);

      allowNegativeYields = true;
      RooProduct *prod = new RooProduct(prodname, prodname, prodElems);
      if (!allowNegativeYields) {
        auto maxname = std::string(prodname) + "_max0";
        RooArgSet prodset(*prod);

        RooFormulaVar *max =
            new RooFormulaVar(maxname.c_str(), "max(0," + prodname + ")", prodset);
        sumElements.add(*max);
      } else {
        sumElements.add(*prod);
      }
      scaleElements.add(*(binWidth));
      i++;
    }

    // put everything together
    this->_sumFunc = make_unique<RooRealSumFunc>(
        Form("%s_morphfunc", name), name, sumElements, scaleElements);

    // coutD(ObjectHandling) << "ownership handling" << std::endl;
    if (!observable)
      std::cerr << "unable to access observable" << std::endl;
    this->_sumFunc.get()->addServer(*observable);
    if (!binWidth)
      std::cerr << "unable to access bin width" << std::endl;
    this->_sumFunc.get()->addServer(*binWidth);
    if (operators.getSize() < 1)
      std::cerr << "no operators listed" << std::endl;
    this->_sumFunc.get()->addServerList(operators);
    if (this->_weights.getSize() < 1)
      std::cerr << "unable to access weight objects" << std::endl;
    this->_sumFunc.get()->addOwnedComponents(this->_weights);
    this->_sumFunc.get()->addOwnedComponents(sumElements);
    this->_sumFunc.get()->addServerList(sumElements);
    this->_sumFunc.get()->addServerList(scaleElements);

#ifdef USE_UBLAS
    std::cout.precision(std::numeric_limits<double>::digits);
#endif

    // oocxcoutP((TObject*)0,ObjectHandling) << morphfunc->Print() << std::endl;

    // fill the this
    // this->_sumFunc = morphfunc;
  }
  //////////////////////////////////////////////////////////////////////////////
  /// create all the temporary objects required by the class

  static RooLagrangianMorphFunc::CacheElem *
  createCache(const RooLagrangianMorphFunc *func) {
    std::string obsName = func->getObservable()->GetName();
    // std::cout << "creating cache for RooLagrangianMorphFunc " << func <<
    // std::endl;
    RooLagrangianMorphFunc::ParamSet values = getParams(func->_operators);

    RooLagrangianMorphFunc::CacheElem *cache =
        new RooLagrangianMorphFunc::CacheElem();
    cache->createComponents(
        func->_config.getParamCards(), func->_config.getFlagValues(),
        func->GetName(), func->_diagrams, func->_nonInterfering, func->_flags);

    // cxcoutP(Eval) << "performing matrix operations" << std::endl;
    cache->buildMatrix(func->_config.getParamCards(),
                       func->_config.getFlagValues(), func->_flags);
    if (obsName.size() == 0) {
      std::cerr << "Matrix inversion succeeded, but no observable was "
                   "supplied. quitting..."
                << std::endl;
      return cache;
    }

    // cxcoutP(ObjectHandling) << "building morphing function" << std::endl;
    oocxcoutP((TObject *)0, ObjectHandling)
        << "observable: " << func->getObservable()->GetName() << std::endl;
    oocxcoutP((TObject *)0, ObjectHandling)
        << "binWidth: " << func->getBinWidth()->GetName() << std::endl;

    setParams(func->_flags, 1);
    cache->buildMorphingFunction(func->GetName(), func->_config.getParamCards(),
                                 func->_sampleMap, func->_physics,
                                 func->_config.IsAllowNegativeYields(),
                                 func->getObservable(), func->getBinWidth());
    setParams(values, func->_operators, true);
    setParams(func->_flags, 1);
    return cache;
  }

  //////////////////////////////////////////////////////////////////////////////
  /// create all the temporary objects required by the class
  /// function variant with precomputed inverse matrix

  static RooLagrangianMorphFunc::CacheElem *
  createCache(const RooLagrangianMorphFunc *func, const Matrix &inverse) {
    // cxcoutP(Caching) << "creating cache for RooLagrangianMorphFunc = " <<
    // func << " with matrix" << std::endl;
    RooLagrangianMorphFunc::ParamSet values = getParams(func->_operators);

    RooLagrangianMorphFunc::CacheElem *cache =
        new RooLagrangianMorphFunc::CacheElem();
    cache->createComponents(
        func->_config.getParamCards(), func->_config.getFlagValues(),
        func->GetName(), func->_diagrams, func->_nonInterfering, func->_flags);

#ifndef USE_UBLAS
    cache->_inverse.ResizeTo(inverse.GetNrows(), inverse.GetNrows());
#endif
    cache->_inverse = inverse;
    cache->_condition = NaN;

    // cxcoutP(ObjectHandling) << "building morphing function" << std::endl;
    setParams(func->_flags, 1);
    cache->buildMorphingFunction(func->GetName(), func->_config.getParamCards(),
                                 func->_sampleMap, func->_physics,
                                 func->_config.IsAllowNegativeYields(),
                                 func->getObservable(), func->getBinWidth());
    setParams(values, func->_operators, true);
    setParams(func->_flags, 1);
    return cache;
  }
};

///////////////////////////////////////////////////////////////////////////////
// Class Implementation ///////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
/// insert an object into a workspace (wrapper for RooWorkspace::import)

void RooLagrangianMorphFunc::importToWorkspace(RooWorkspace *ws,
                                               const RooAbsReal *object) {
  if (ws && object)
    ws->import(*object, RooFit::RecycleConflictNodes());
}

////////////////////////////////////////////////////////////////////////////////
/// insert an object into a workspace (wrapper for RooWorkspace::import)

void RooLagrangianMorphFunc::importToWorkspace(RooWorkspace *ws,
                                               RooAbsData *object) {
  if (ws && object)
    ws->import(*object, RooFit::RecycleConflictNodes());
}

////////////////////////////////////////////////////////////////////////////////
/// insert this object into a workspace

void RooLagrangianMorphFunc::insert(RooWorkspace *ws) {
  RooLagrangianMorphFunc::importToWorkspace(ws, this);
}

////////////////////////////////////////////////////////////////////////////////
/// write a matrix to a file

void RooLagrangianMorphFunc::writeMatrixToFile(const TMatrixD &matrix,
                                               const char *fname) {
  writeMatrixToFileT(matrix, fname);
}

////////////////////////////////////////////////////////////////////////////////
/// write a matrix to a stream

void RooLagrangianMorphFunc::writeMatrixToStream(const TMatrixD &matrix,
                                                 std::ostream &stream) {
  writeMatrixToStreamT(matrix, stream);
}

////////////////////////////////////////////////////////////////////////////////
/// read a matrix from a text file

TMatrixD RooLagrangianMorphFunc::readMatrixFromFile(const char *fname) {
  return readMatrixFromFileT<TMatrixD>(fname);
}

////////////////////////////////////////////////////////////////////////////////
/// read a matrix from a stream

TMatrixD RooLagrangianMorphFunc::readMatrixFromStream(std::istream &stream) {
  return readMatrixFromStreamT<TMatrixD>(stream);
}

////////////////////////////////////////////////////////////////////////////////
/// setup observable, recycle existing observable if defined

RooRealVar *RooLagrangianMorphFunc::setupObservable(const char *obsname,
                                                    TClass *mode,
                                                    TObject *inputExample) {
  // cxcoutP(ObjectHandling) << "setting up observable" << std::endl;
  RooRealVar *obs = nullptr;
  Bool_t obsExists(false);
  if (this->_observables.at(0) != 0) {
    obs = (RooRealVar *)this->_observables.at(0);
    obsExists = true;
  }

  if (mode && mode->InheritsFrom(RooHistFunc::Class())) {
    obs = (RooRealVar *)(dynamic_cast<RooHistFunc *>(inputExample)
                             ->getHistObsList()
                             .first());
    obsExists = true;
    this->_observables.add(*obs);
    // ClassName()
  } else if (mode && mode->InheritsFrom(RooParamHistFunc::Class())) {
    obs = (RooRealVar *)(dynamic_cast<RooParamHistFunc *>(inputExample)
                             ->paramList()
                             .first());
    obsExists = true;
    this->_observables.add(*obs);
  }

  //   Note: "found!" will be printed if s2 is a substring of s1, both s1 and s2
  //   are of type std::string. s1.find(s2)
  // obtain the observable
  if (!obsExists) {
    if (mode && mode->InheritsFrom(TH1::Class())) {
      //  cxcoutP(ObjectHandling) << "getObservable: creating new multi-bin
      //  observable object " << obsname << std::endl;
      TH1 *hist = (TH1 *)(inputExample);
      obs = new RooRealVar(obsname, obsname, hist->GetXaxis()->GetXmin(),
                           hist->GetXaxis()->GetXmax());
      obs->setBins(hist->GetNbinsX());
    } else {
      //  cxcoutP(ObjectHandling) << "getObservable: creating new single-bin
      //  observable object " << obsname << std::endl;
      obs = new RooRealVar(obsname, obsname, 0, 1);
      obs->setBins(1);
    }
    this->_observables.add(*obs);
  } else {
    // cxcoutP(ObjectHandling) << "getobservable: recycling existing observable
    // object " << this->_observables.at(0) << std::endl;
    if (strcmp(obsname, obs->GetName()) != 0) {
      //  coutW(ObjectHandling) << " name of existing observable " <<
      //  this->_observables.at(0)->GetName() << " does not match expected name
      //  " << obsname << std::endl ;
    }
  }

  TString sbw = Form("binWidth_%s", makeValidName(obs->GetName()).Data());
  RooRealVar *binWidth = new RooRealVar(sbw.Data(), sbw.Data(), 1.);
  double bw = obs->numBins() / (obs->getMax() - obs->getMin());
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

inline void RooLagrangianMorphFunc::updateSampleWeights() {
  //#ifdef USE_MULTIPRECISION_LC
  int sampleidx = 0;
  auto cache = this->getCache(_curNormSet);
  const size_t n(size(cache->_inverse));
  for (auto sampleit : this->_config.getParamCards()) {
    const std::string sample(sampleit.first);
    // build the formula with the correct normalization
    RooLinearCombination *sampleformula = dynamic_cast<RooLinearCombination *>(
        this->getSampleWeight(sample.c_str()));
    if (!sampleformula) {
      // coutE(ObjectHandling) << Form("unable to access formula for sample
      // '%s'!",sample.c_str()) << std::endl;
      return;
    }
    cxcoutP(ObjectHandling)
        << "updating formula for sample '" << sample << "'" << std::endl;
    for (size_t formulaidx = 0; formulaidx < n; ++formulaidx) {
      const SuperFloat val(cache->_inverse(formulaidx, sampleidx));
#ifdef USE_UBLAS
      if (val != val) {
#else
      if (std::isnan(val)) {
#endif
        coutE(ObjectHandling) << "refusing to propagate NaN!" << std::endl;
      }
      cxcoutP(ObjectHandling) << "   " << formulaidx << ":"
                              << sampleformula->getCoefficient(formulaidx)
                              << " -> " << val << std::endl;
      sampleformula->setCoefficient(formulaidx, val);
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

void RooLagrangianMorphFunc::Config::readParameters(TDirectory *f) {
  readValues<double>(_paramCards, f, _folderNames, "param_card", true);
  readValues<int>(_flagValues, f, _folderNames, "flags", false);
}

////////////////////////////////////////////////////////////////////////////////
/// retrieve the physics inputs

void RooLagrangianMorphFunc::collectInputs(TDirectory *file) {
  std::string obsName = this->_config.getObservableName();
  cxcoutP(InputArguments) << "initializing physics inputs from file "
                          << file->GetName() << " with object name(s) '"
                          << obsName << "'" << std::endl;
  auto foldernames = this->_config.getFolderNames();
  auto base = file->Get<TFolder>(foldernames[0].c_str());
  TObject *obj = base->FindObject(obsName.c_str());
  if (!obj) {
    std::cerr << "unable to locate object '" << obsName << "' in folder '"
              << base << "'!" << std::endl;
    return;
  }
  std::string classname = obj->ClassName();
  TClass *mode = TClass::GetClass(obj->ClassName());

  RooRealVar *observable = this->setupObservable(obsName.c_str(), mode, obj);
  if (classname.find("TH1") != std::string::npos) {
    collectHistograms(this->GetName(), file, this->_sampleMap, this->_physics,
                      *observable, obsName, this->_config.getParamCards());
  } else if (classname.find("RooHistFunc") != std::string::npos ||
             classname.find("RooParamHistFunc") != std::string::npos ||
             classname.find("PiecewiseInterpolation") != std::string::npos) {
    collectRooAbsReal(this->GetName(), file, this->_sampleMap, this->_physics,
                      obsName, this->_config.getParamCards());
  } else if (classname.find("TParameter<double>") != std::string::npos) {
    collectCrosssections<double>(this->GetName(), file, this->_sampleMap,
                                 this->_physics, obsName,
                                 this->_config.getParamCards());
  } else if (classname.find("TParameter<float>") != std::string::npos) {
    collectCrosssections<float>(this->GetName(), file, this->_sampleMap,
                                this->_physics, obsName,
                                this->_config.getParamCards());
  } else if (classname.find("TPair") != std::string::npos) {
    collectCrosssectionsTPair(this->GetName(), file, this->_sampleMap,
                              this->_physics, obsName, foldernames[0],
                              this->_config.getParamCards());
  } else {
    std::cerr << "cannot morph objects of class '" << mode->GetName() << "'!"
              << std::endl;
  }
}

////////////////////////////////////////////////////////////////////////////////
/// convert the RooArgList folders into a simple vector of std::string

void RooLagrangianMorphFunc::Config::addFolders(const RooArgList &folders) {
  for (auto const& folder : folders) {
    RooStringVar *var = dynamic_cast<RooStringVar *>(folder);
    const std::string sample(var ? var->getVal() : folder->GetName());
    if (sample.empty()) continue;
    this->_folderNames.push_back(sample);
  }

  TDirectory *file = openFile(this->getFileName());
  TIter next(file->GetList());
  TObject *obj = nullptr;
  while ((obj = (TObject *)next())) {
    auto f = file->Get<TFolder>(obj->GetName());
    if (!f)
      continue;
    std::string name(f->GetName());
    if (name.empty())
      continue;
    this->_folderNames.push_back(name);
  }
  closeFile(file);
}

////////////////////////////////////////////////////////////////////////////////
/// parameterised constructor
RooLagrangianMorphFunc::Config::Config(const RooAbsCollection &couplings) {
  extractCouplings(couplings, this->_couplings);
}

////////////////////////////////////////////////////////////////////////////////
/// parameterised constructor
RooLagrangianMorphFunc::Config::Config(const RooAbsCollection &prodCouplings,
                                       const RooAbsCollection &decCouplings) {
  extractCouplings(prodCouplings, this->_prodCouplings);
  extractCouplings(decCouplings, this->_decCouplings);
}

////////////////////////////////////////////////////////////////////////////////
/// config setter for couplings
void RooLagrangianMorphFunc::Config::setCouplings(
    const RooAbsCollection &couplings) {
  extractCouplings(couplings, this->_couplings);
}

////////////////////////////////////////////////////////////////////////////////
/// config setter for production and decay couplings
void RooLagrangianMorphFunc::Config::setCouplings(
    const RooAbsCollection &prodCouplings,
    const RooAbsCollection &decCouplings) {
  extractCouplings(prodCouplings, this->_prodCouplings);
  extractCouplings(decCouplings, this->_decCouplings);
}

////////////////////////////////////////////////////////////////////////////////
/// config setter for input file name
void RooLagrangianMorphFunc::Config::setFileName(const char *filename) {
  this->_fileName = filename;
}

////////////////////////////////////////////////////////////////////////////////
/// config setter for input folders
void RooLagrangianMorphFunc::Config::setFolders(const RooArgList &folderlist) {
  for (auto itr : folderlist)
    this->_folderlist.add(*itr);
}

////////////////////////////////////////////////////////////////////////////////
/// config setter for name of the the observable to be morphed
void RooLagrangianMorphFunc::Config::setObservableName(const char *obsname) {
  this->_obsName = obsname;
}

////////////////////////////////////////////////////////////////////////////////
/// config setter for name of the the observable to be morphed
void RooLagrangianMorphFunc::Config::allowNegativeYields(
    bool isallowNegativeYields) {
  this->_allowNegativeYields = isallowNegativeYields;
}

////////////////////////////////////////////////////////////////////////////////
/// config setter for diagrams
template <class T>
void RooLagrangianMorphFunc::Config::setDiagrams(
    const std::vector<std::vector<T>> &diagrams) {
  for (size_t j = 0; j < diagrams.size(); ++j) {
    std::vector<RooArgList *> vertices;
    for (size_t i = 0; i < diagrams[j].size(); i++) {
      vertices.push_back(new RooArgList());
      vertices[i]->add(diagrams[j][i]);
    }
    this->_configDiagrams.push_back(vertices);
  }
}

////////////////////////////////////////////////////////////////////////////////
/// config setter for vertices
template <class T>
void RooLagrangianMorphFunc::Config::setVertices(
    const std::vector<T> &vertices) {
  std::vector<std::vector<T>> diagrams;
  diagrams.push_back(vertices);
  this->setDiagrams(diagrams);
}

////////////////////////////////////////////////////////////////////////////////
/// append the parameter map with a parmeter set

//void RooLagrangianMorphFunc::Config::append(
//    RooLagrangianMorphFunc::ParamMap &map, const char *str,
//    RooLagrangianMorphFunc::ParamSet &set) {
//  map[str] = set;
//}

////////////////////////////////////////////////////////////////////////////////
/// set values to parameter set (-?-)

void RooLagrangianMorphFunc::Config::append(
    RooLagrangianMorphFunc::ParamSet &set, const char *str, double val) {
  set[str] = val;
}

template void RooLagrangianMorphFunc::Config::setVertices<RooArgSet>(
    const std::vector<RooArgSet> &);
template void RooLagrangianMorphFunc::Config::setDiagrams<RooArgSet>(
    const std::vector<std::vector<RooArgSet>> &);
template void RooLagrangianMorphFunc::Config::setVertices<RooArgList>(
    const std::vector<RooArgList> &);
template void RooLagrangianMorphFunc::Config::setDiagrams<RooArgList>(
    const std::vector<std::vector<RooArgList>> &);

/*
////////////////////////////////////////////////////////////////////////////////
/// config setter for vertices
void RooLagrangianMorphFunc::Config::disableInterference(const std::vector<const
char*>& nonInterfering){
  // disable interference between the listed operators
  std::stringstream name;
  name << "noInteference";
  for(auto c:nonInterfering){
    name << c;
  }
  RooArgList* p = new RooArgList(name.str().c_str());
  this->_nonInterfering.push_back(new RooArgList());
  for(auto c:nonInterfering){
    p->add(*(new RooStringVar(c,c,c)));
  }
}

////////////////////////////////////////////////////////////////////////////////
/// config setter for vertices
void RooLagrangianMorphFunc::Config::disableInterferences(const
std::vector<std::vector<const char*> >& nonInterfering){
  // disable interferences between the listed groups of operators
  for(size_t i=0;i<nonInterfering.size();++i){
    this->disableInterference(nonInterfering[i]);
  }
}
*/


////////////////////////////////////////////////////////////////////////////////
/// print all the parameters and their values in the given sample to the console

void RooLagrangianMorphFunc::printParameters(const char *samplename) const {
  for (const auto &param : this->_config.getParamCards().at(samplename)) {
    if (this->hasParameter(param.first.c_str())) {
      std::cout << param.first << " = " << param.second;
      if (this->isParameterConstant(param.first.c_str()))
        std::cout << " (const)";
      std::cout << std::endl;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
/// print all the known samples to the console

void RooLagrangianMorphFunc::printSamples() const {
  // print all the known samples to the console
  for (auto folder : this->_config.getFolderNames()) {
    std::cout << folder << std::endl;
  }
}

////////////////////////////////////////////////////////////////////////////////
/// print the current physics values

void RooLagrangianMorphFunc::printPhysics() const {
  for (const auto &sample : this->_sampleMap) {
    RooAbsArg *phys = this->_physics.at(sample.second);
    if (!phys)
      continue;
    phys->Print();
  }
}

////////////////////////////////////////////////////////////////////////////////
/// protected constructor with proper arguments

RooLagrangianMorphFunc::RooLagrangianMorphFunc(const char *name,
                                               const char *title,
                                               const Config &config)
    : RooAbsReal(name, title), _cacheMgr(this, 10, kTRUE, kTRUE),
      _operators("operators", "set of operators", this, kTRUE, kFALSE),
      _observables("observables", "morphing observables", this, kTRUE, kFALSE),
      _binWidths("binWidths", "set of binWidth objects", this, kTRUE, kFALSE),
      _config(config), _curNormSet(0) {
  this->_config.addFolders(_config.getFolders());
  this->init();
  this->setup(false);

  TRACE_CREATE
}

/*////////////////////////////////////////////////////////////////////////////////
/// protected constructor with proper arguments

RooLagrangianMorphFunc::RooLagrangianMorphFunc(const char *name, const char
*title, const Config& config, bool allowNegativeYields) :
  RooLagrangianMorphFunc(name,title,config,"",folders,allowNegativeYields) {
//  this->disableInterferences(this->_nonInterfering);
  this->setup(false);
  TRACE_CREATE
}
*/
////////////////////////////////////////////////////////////////////////////////
/// setup this instance with the given set of operators and vertices
/// if own=true, the class will own the operatorsemplate <class Base>

void RooLagrangianMorphFunc::setup(bool own) {
  cxcoutP(Eval) << "setup(" << own << ") called" << std::endl;
  this->_ownParameters = own;
  auto diagrams = this->_config.getDiagrams();

  if (diagrams.size() > 0) {
    RooArgList operators;
    for (auto const& v : diagrams) {
      for (const RooArgList *t : v) {
        extractOperators(*t, operators);
      }
    }

    if (own) {
      this->_operators.addOwned(operators);
    } else {
      this->_operators.add(operators);
    }

    for (size_t j = 0; j < diagrams.size(); ++j) {
      std::vector<RooListProxy *> diagram;
      for (size_t i = 0; i < diagrams[j].size(); ++i) {
        std::stringstream name;
        name << "!vertex" << i;
        std::stringstream title;
        title << "set of couplings in the vertex " << i;
        diagram.push_back(new RooListProxy(
            name.str().c_str(), title.str().c_str(), this, kTRUE, kFALSE));
        if (own) {
          diagram[i]->addOwned(*diagrams[j][i]);
        } else {
          diagram[i]->add(*diagrams[j][i]);
        }
      }
      // ownership of contents of diagrams[i][j]
      this->_diagrams.push_back(diagram);
    }
  }

  else if (this->_config.getCouplings().size() > 0) {
    RooArgList operators;
    std::vector<RooListProxy *> vertices;
    cxcoutP(InputArguments) << "couplings provided" << std::endl;
    extractOperators(this->_config.getCouplings(), operators);
    vertices.push_back(new RooListProxy(
        "!couplings", "set of couplings in the vertex", this, kTRUE, kFALSE));
    if (own) {
      cxcoutP(InputArguments) << "adding own operators" << std::endl;
      this->_operators.addOwned(operators);
      vertices[0]->addOwned(this->_config.getCouplings());
    } else {
      cxcoutP(InputArguments) << "adding non-own operators" << std::endl;
      this->_operators.add(operators);
      vertices[0]->add(this->_config.getCouplings());
    }
    this->_diagrams.push_back(vertices);
  }

  else if (this->_config.getProdCouplings().size() > 0 &&
           this->_config.getDecCouplings().size() > 0) {
    std::vector<RooListProxy *> vertices;
    RooArgList operators;
    cxcoutP(InputArguments) << "prod/dec couplings provided" << std::endl;
    extractOperators(this->_config.getProdCouplings(), operators);
    extractOperators(this->_config.getDecCouplings(), operators);
    vertices.push_back(new RooListProxy(
        "!production", "set of couplings in the production vertex", this, kTRUE,
        kFALSE));
    vertices.push_back(new RooListProxy(
        "!decay", "set of couplings in the decay vertex", this, kTRUE, kFALSE));
    if (own) {
      cxcoutP(InputArguments) << "adding own operators" << std::endl;
      this->_operators.addOwned(operators);
      vertices[0]->addOwned(this->_config.getProdCouplings());
      vertices[1]->addOwned(this->_config.getDecCouplings());
    } else {
      cxcoutP(InputArguments) << "adding non-own operators" << std::endl;
      this->_operators.add(operators);
      vertices[0]->add(this->_config.getProdCouplings());
      vertices[1]->add(this->_config.getDecCouplings());
    }
    this->_diagrams.push_back(vertices);
  }
}

////////////////////////////////////////////////////////////////////////////////
/// (-?-)

void RooLagrangianMorphFunc::init() {
  std::string filename = this->_config.getFileName();
  TDirectory *file = openFile(filename.c_str());
  if (!file) {
    coutE(InputArguments) << "unable to open file '" << filename << "'!"
                          << std::endl;
    return;
  }
  this->readParameters(file);
  checkNameConflict(this->_config.getParamCards(), this->_operators);
  this->collectInputs(file);
  closeFile(file);
  this->addServerList(this->_physics);
  cxcoutP(InputArguments) << "adding flags" << std::endl;
  RooRealVar *nNP0 = new RooRealVar("nNP0", "nNP0", 1., 0, 1.);
  nNP0->setStringAttribute("NP", "0");
  nNP0->setConstant(true);
  this->_flags.add(*nNP0);
  RooRealVar *nNP1 = new RooRealVar("nNP1", "nNP1", 1., 0, 1.);
  nNP1->setStringAttribute("NP", "1");
  nNP1->setConstant(true);
  this->_flags.add(*nNP1);
  RooRealVar *nNP2 = new RooRealVar("nNP2", "nNP2", 1., 0, 1.);
  nNP2->setStringAttribute("NP", "2");
  nNP2->setConstant(true);
  this->_flags.add(*nNP2);
  RooRealVar *nNP3 = new RooRealVar("nNP3", "nNP3", 1., 0, 1.);
  nNP3->setStringAttribute("NP", "3");
  nNP3->setConstant(true);
  this->_flags.add(*nNP3);
  RooRealVar *nNP4 = new RooRealVar("nNP4", "nNP4", 1., 0, 1.);
  nNP4->setStringAttribute("NP", "4");
  nNP4->setConstant(true);
  this->_flags.add(*nNP4);
}

////////////////////////////////////////////////////////////////////////////////
/// copy constructor

RooLagrangianMorphFunc::RooLagrangianMorphFunc(
    const RooLagrangianMorphFunc &other, const char *name)
    : RooAbsReal(other, name), _cacheMgr(other._cacheMgr, this),
      _scale(other._scale), _sampleMap(other._sampleMap),
      _physics(other._physics.GetName(), this, other._physics),
      _operators(other._operators.GetName(), this, other._operators),
      _observables(other._observables.GetName(), this, other._observables),
      _binWidths(other._binWidths.GetName(), this, other._binWidths),
      _flags(other._flags.GetName(), this, other._flags),
      _config(other._config), _curNormSet(0) {
  std::cout << "COPY CONSTRUCTOR" << std::endl;
  for (size_t j = 0; j < other._diagrams.size(); ++j) {
    std::vector<RooListProxy *> diagram;
    for (size_t i = 0; i < other._diagrams[j].size(); ++i) {
      RooListProxy *list = new RooListProxy(other._diagrams[j][i]->GetName(),
                                            this, *(other._diagrams[j][i]));
      diagram.push_back(list);
    }
    this->_diagrams.push_back(diagram);
  }
  TRACE_CREATE
}

////////////////////////////////////////////////////////////////////////////////
/// set energy scale of the EFT expansion

void RooLagrangianMorphFunc::setScale(double val) { this->_scale = val; }

////////////////////////////////////////////////////////////////////////////////
/// get energy scale of the EFT expansion

double RooLagrangianMorphFunc::getScale() { return this->_scale; }

////////////////////////////////////////////////////////////////////////////////
// default constructor

RooLagrangianMorphFunc::RooLagrangianMorphFunc()
    : _operators("operators", "set of operators", this, kTRUE, kFALSE),
      _observables("observable", "morphing observable", this, kTRUE, kFALSE),
      _binWidths("binWidths", "set of bin width objects", this, kTRUE, kFALSE) {
  static int counter(0);
  counter++;
  TRACE_CREATE
}

////////////////////////////////////////////////////////////////////////////////
/// default destructor

RooLagrangianMorphFunc::~RooLagrangianMorphFunc(){TRACE_DESTROY}

////////////////////////////////////////////////////////////////////////////////
/// cloning method

TObject *RooLagrangianMorphFunc::clone(const char *newname) const {
  return new RooLagrangianMorphFunc(*this, newname);
}

////////////////////////////////////////////////////////////////////////////////
/// print the authors information

void RooLagrangianMorphFunc::printAuthors() const {
  std::cout << "\033[1mRooLagrangianMorphFunc\033[0m: a RooFit class for "
               "morphing physics distributions between configurations. authors:"
            << std::endl;
  std::cout << "   "
            << "Lydia Brenner   (lbrenner@cern.ch)" << std::endl;
  std::cout << "   "
            << "Carsten Burgard (cburgard@cern.ch)" << std::endl;
  std::cout << "   "
            << "Katharina Ecker (kecker@cern.ch)" << std::endl;
  std::cout << "   "
            << "Adam Kaluza     (akaluza@cern.ch)" << std::endl;
  std::cout << "please feel free to contact with questions and suggestions."
            << std::endl;
}

////////////////////////////////////////////////////////////////////////////////
/// calculate the number of samples needed to morph a bivertex, 2-2 physics
/// process

int RooLagrangianMorphFunc::countSamples(int nprod, int ndec, int nboth) {
  FeynmanDiagram diagram;
  std::vector<bool> prod;
  std::vector<bool> dec;
  for (int i = 0; i < nboth; ++i) {
    prod.push_back(true);
    dec.push_back(true);
  }
  for (int i = 0; i < nprod; ++i) {
    prod.push_back(true);
    dec.push_back(false);
  }
  for (int i = 0; i < ndec; ++i) {
    prod.push_back(false);
    dec.push_back(true);
  }
  diagram.push_back(prod);
  diagram.push_back(dec);
  MorphFuncPattern morphfuncpattern;
  ::collectPolynomials(morphfuncpattern, diagram);
  return morphfuncpattern.size();
}

////////////////////////////////////////////////////////////////////////////////
/// calculate the number of samples needed to morph a certain physics process
/// usage:
///   countSamples ( { RooLagrangianMorphFunc::makeHCggfCouplings(),
///   RooLagrangianMorphFunc::makeHCHZZCouplings() } )

int RooLagrangianMorphFunc::countSamples(std::vector<RooArgList *> &vertices) {
  RooArgList operators, couplings;
  for (auto vertex : vertices) {
    extractOperators(*vertex, operators);
    extractCouplings(*vertex, couplings);
  }
  FeynmanDiagram diagram;
  ::fillFeynmanDiagram(diagram, vertices, couplings);
  MorphFuncPattern morphfuncpattern;
  ::collectPolynomials(morphfuncpattern, diagram);
  return morphfuncpattern.size();
}

////////////////////////////////////////////////////////////////////////////////
/// create TPair containers of the type expected by the RooLagrangianMorph

TPair *RooLagrangianMorphFunc::makeCrosssectionContainer(double xs,
                                                         double unc) {
  TPair *v = new TPair(new TParameter<double>("xsection", xs),
                       new TParameter<double>("uncertainty", unc));
  return v;
}

////////////////////////////////////////////////////////////////////////////////
/// create only the weight formulas. static function for external usage.

std::map<std::string, std::string> RooLagrangianMorphFunc::createWeightStrings(
    const RooLagrangianMorphFunc::ParamMap &inputs,
    const std::vector<std::vector<std::string>> &vertices_str) {
  std::stack<RooArgList> ownedVertices;
  std::vector<RooArgList *> vertices;
  RooArgList couplings;
  for (const auto &vtx : vertices_str) {
    ownedVertices.emplace();
    auto& vertex = ownedVertices.top();
    for (const auto &c : vtx) {
      RooRealVar *coupling = (RooRealVar *)(couplings.find(c.c_str()));
      if (!coupling) {
        coupling = new RooRealVar(c.c_str(), c.c_str(), 1., 0., 10.);
        couplings.add(*coupling);
      }
      vertex.add(*coupling);
    }
    vertices.push_back(&vertex);
  }
  auto retval =
      RooLagrangianMorphFunc::createWeightStrings(inputs, vertices, couplings);
  return retval;
}

////////////////////////////////////////////////////////////////////////////////
/// create only the weight formulas. static function for external usage.

std::map<std::string, std::string> RooLagrangianMorphFunc::createWeightStrings(
    const RooLagrangianMorphFunc::ParamMap &inputs,
    const std::vector<RooArgList *> &vertices, RooArgList &couplings) {
  return createWeightStrings(inputs, vertices, couplings, {}, {}, {});
}

////////////////////////////////////////////////////////////////////////////////
/// create only the weight formulas. static function for external usage.

std::map<std::string, std::string> RooLagrangianMorphFunc::createWeightStrings(
    const RooLagrangianMorphFunc::ParamMap &inputs,
    const std::vector<RooArgList *> &vertices, RooArgList &couplings,
    const RooLagrangianMorphFunc::FlagMap &flagValues, const RooArgList &flags,
    const std::vector<RooArgList *> &nonInterfering) {
  FormulaList formulas = ::createFormulas("", inputs, flagValues, {vertices},
                                          couplings, flags, nonInterfering);
  RooArgSet operators;
  extractOperators(couplings, operators);
  Matrix matrix(
      ::buildMatrixT<Matrix>(inputs, formulas, operators, flagValues, flags));
  if (size(matrix) < 1) {
    std::cerr << "input matrix is empty, please provide suitable input samples!"
              << std::endl;
  }
  Matrix inverse(::diagMatrix(size(matrix)));
  double condition __attribute__((unused)) =
      (double)(invertMatrix(matrix, inverse));
  auto retval = buildSampleWeightStrings(inputs, formulas, inverse);
  return retval;
}

////////////////////////////////////////////////////////////////////////////////
/// create only the weight formulas. static function for external usage.

RooArgSet RooLagrangianMorphFunc::createWeights(
    const RooLagrangianMorphFunc::ParamMap &inputs,
    const std::vector<RooArgList *> &vertices, RooArgList &couplings,
    const RooLagrangianMorphFunc::FlagMap &flagValues, const RooArgList &flags,
    const std::vector<RooArgList *> &nonInterfering) {
  FormulaList formulas = ::createFormulas("", inputs, flagValues, {vertices},
                                          couplings, flags, nonInterfering);
  RooArgSet operators;
  extractOperators(couplings, operators);
  Matrix matrix(
      ::buildMatrixT<Matrix>(inputs, formulas, operators, flagValues, flags));
  if (size(matrix) < 1) {
    std::cerr << "input matrix is empty, please provide suitable input samples!"
              << std::endl;
  }
  Matrix inverse(::diagMatrix(size(matrix)));
  double condition __attribute__((unused)) =
      (double)(invertMatrix(matrix, inverse));
  RooArgSet retval;
  ::buildSampleWeights(retval, (const char *)nullptr /* name */, inputs, formulas,
                       inverse);
  return retval;
}

////////////////////////////////////////////////////////////////////////////////
/// create only the weight formulas. static function for external usage.

RooArgSet RooLagrangianMorphFunc::createWeights(
    const RooLagrangianMorphFunc::ParamMap &inputs,
    const std::vector<RooArgList *> &vertices, RooArgList &couplings) {
  std::vector<RooArgList *> nonInterfering;
  RooArgList flags;
  FlagMap flagValues;
  return RooLagrangianMorphFunc::createWeights(
      inputs, vertices, couplings, flagValues, flags, nonInterfering);
}

////////////////////////////////////////////////////////////////////////////////
/// return the RooProduct that is the element of the RooRealSumPdfi
///  corresponding to the given sample name

RooProduct *RooLagrangianMorphFunc::getSumElement(const char *name) const {
  auto mf = this->getFunc();
  if (!mf) {
    coutE(Eval) << "unable to retrieve morphing function" << std::endl;
    return nullptr;
  }
  RooArgSet *args = mf->getComponents();
  TString prodname(name);
  prodname.Append("_");
  prodname.Append(this->GetName());

  for (auto itr : *args) {
    RooProduct *prod = dynamic_cast<RooProduct *>(itr);
    if (!prod)
      continue;
    TString sname(prod->GetName());
    if (sname.CompareTo(prodname) == 0) {
      return prod;
    }
  }
  return nullptr;
}
////////////////////////////////////////////////////////////////////////////////
/// return the vector of sample names, used to build the morph func

std::vector<std::string> RooLagrangianMorphFunc::getSamples() const {
  return this->_config.getFolderNames();
}

////////////////////////////////////////////////////////////////////////////////
/// retrieve the weight (prefactor) of a sample with the given name

RooAbsReal *RooLagrangianMorphFunc::getSampleWeight(const char *name) {
  auto cache = this->getCache(_curNormSet);
  auto wname = std::string("w_") + name + "_" + this->GetName();
  return dynamic_cast<RooAbsReal *>(cache->_weights.find(wname.c_str()));
}

////////////////////////////////////////////////////////////////////////////////
/// print the current sample weights

void RooLagrangianMorphFunc::printWeights() const {
  this->printSampleWeights();
}

////////////////////////////////////////////////////////////////////////////////
/// print the current sample weights

void RooLagrangianMorphFunc::printSampleWeights() const {
  auto *cache = this->getCache(this->_curNormSet);
  for (const auto &sample : this->_sampleMap) {
    auto weightName = std::string("w_") + sample.first + "_" + this->GetName();
    auto weight = static_cast<RooAbsReal *>(cache->_weights.find(weightName.c_str()));
    if (!weight)
      continue;
    std::cout << weight->GetName() << " = " << weight->GetTitle() << " = "
              << weight->getVal() << std::endl;
  }
}

////////////////////////////////////////////////////////////////////////////////
/// randomize the parameters a bit
/// useful to test and debug fitting

void RooLagrangianMorphFunc::randomizeParameters(double z) {
  RooRealVar *obj;
  TRandom3 r;

  for (auto itr : _operators) {
    obj = dynamic_cast<RooRealVar *>(itr);
    double val = obj->getVal();
    if (obj->isConstant())
      continue;
    double variation = r.Gaus(1, z);
    obj->setVal(val * variation);
  }
}

////////////////////////////////////////////////////////////////////////////////
/// retrive the new physics objects and update the weights in the morphing
/// function

bool RooLagrangianMorphFunc::updateCoefficients() {
  auto cache = this->getCache(_curNormSet);

  std::string filename = this->_config.getFileName();
  TDirectory *file = openFile(filename.c_str());
  if (!file) {
    coutE(InputArguments) << "unable to open file '" << filename << "'!"
                          << std::endl;
    return false;
  }
  cxcoutP(InputArguments) << "reading parameter sets." << std::endl;

  this->readParameters(file);

  checkNameConflict(this->_config.getParamCards(), this->_operators);
  this->collectInputs(file);

  cache->buildMatrix(this->_config.getParamCards(),
                     this->_config.getFlagValues(), this->_flags);
  this->updateSampleWeights();

  closeFile(file);
  return true;
}

////////////////////////////////////////////////////////////////////////////////
/// setup the morphing function with a predefined inverse matrix
/// call this function *before* any other after creating the object

bool RooLagrangianMorphFunc::useCoefficients(const TMatrixD &inverse) {
  auto cache = static_cast<RooLagrangianMorphFunc::CacheElem *>(_cacheMgr.getObj(0, (RooArgSet *)0));
  Matrix m = makeSuperMatrix(inverse);
  if (cache) {
    std::string filename = this->_config.getFileName();
    cache->_inverse = m;
    TDirectory *file = openFile(filename.c_str());
    if (!file) {
      coutE(InputArguments)
          << "unable to open file '" << filename << "'!" << std::endl;
      return false;
    }
    cxcoutP(InputArguments) << "reading parameter sets." << std::endl;

    this->readParameters(file);
    checkNameConflict(this->_config.getParamCards(), this->_operators);
    this->collectInputs(file);

    // then, update the weights in the morphing function
    this->updateSampleWeights();

    closeFile(file);
  } else {
    cache = RooLagrangianMorphFunc::CacheElem::createCache(this, m);
    if (!cache)
      coutE(Caching) << "unable to create cache!" << std::endl;
    this->_cacheMgr.setObj(0, 0, cache, 0);
  }
  return true;
}

////////////////////////////////////////////////////////////////////////////////
// setup the morphing function with a predefined inverse matrix
// call this function *before* any other after creating the object

bool RooLagrangianMorphFunc::useCoefficients(const char *filename) {
  auto cache = static_cast<RooLagrangianMorphFunc::CacheElem *>(_cacheMgr.getObj(0, (RooArgSet *)0));
  if (cache) {
    return false;
  }
  cache = RooLagrangianMorphFunc::CacheElem::createCache(
      this, readMatrixFromFileT<Matrix>(filename));
  if (!cache)
    coutE(Caching) << "unable to create cache!" << std::endl;
  this->_cacheMgr.setObj(0, 0, cache, 0);
  return true;
}

////////////////////////////////////////////////////////////////////////////////
/// write the inverse matrix to a file

bool RooLagrangianMorphFunc::writeCoefficients(const char *filename) {
  auto cache = this->getCache(_curNormSet);
  if (!cache)
    return false;
  writeMatrixToFileT(cache->_inverse, filename);
  return true;
}

////////////////////////////////////////////////////////////////////////////////
/// retrieve the cache object

typename RooLagrangianMorphFunc::CacheElem *
RooLagrangianMorphFunc::getCache(const RooArgSet * /*nset*/) const {
  auto cache = static_cast<RooLagrangianMorphFunc::CacheElem *>(_cacheMgr.getObj(0, (RooArgSet *)0));
  if (!cache) {
    cxcoutP(Caching) << "creating cache from getCache function for " << this
                     << std::endl;
    cxcoutP(Caching) << "current storage has size " << this->_sampleMap.size()
                     << std::endl;
    cache = RooLagrangianMorphFunc::CacheElem::createCache(this);
    if (cache)
      this->_cacheMgr.setObj(0, 0, cache, 0);
    else
      coutE(Caching) << "unable to create cache!" << std::endl;
  }
  return cache;
}

////////////////////////////////////////////////////////////////////////////////
/// return true if a cache object is present, false otherwise

bool RooLagrangianMorphFunc::hasCache() const {
  return (bool)(_cacheMgr.getObj(0, (RooArgSet *)0));
}

////////////////////////////////////////////////////////////////////////////////
/// set one parameter to a specific value

void RooLagrangianMorphFunc::setParameter(const char *name, double value) {
  RooRealVar *param = this->getParameter(name);
  if (!param) {
    return;
  }
  if (value > param->getMax())
    param->setMax(value);
  if (value < param->getMin())
    param->setMin(value);
  param->setVal(value);
}

////////////////////////////////////////////////////////////////////////////////
/// set one flag to a specific value

void RooLagrangianMorphFunc::setFlag(const char *name, double value) {
  RooRealVar *param = this->getFlag(name);
  if (!param) {
    return;
  }
  param->setVal(value);
}

////////////////////////////////////////////////////////////////////////////////
/// set one parameter to a specific value and range

void RooLagrangianMorphFunc::setParameter(const char *name, double value,
                                          double min, double max) {
  RooRealVar *param = this->getParameter(name);
  if (!param) {
    return;
  }
  param->setMin(min);
  param->setMax(max);
  param->setVal(value);
}

////////////////////////////////////////////////////////////////////////////////
/// set one parameter to a specific value and range
void RooLagrangianMorphFunc::setParameter(const char *name, double value,
                                          double min, double max,
                                          double error) {
  RooRealVar *param = this->getParameter(name);
  if (!param) {
    return;
  }
  param->setMin(min);
  param->setMax(max);
  param->setVal(value);
  param->setError(error);
}

////////////////////////////////////////////////////////////////////////////////
/// return true if the parameter with the given name is set constant, false
/// otherwise

bool RooLagrangianMorphFunc::isParameterConstant(const char *name) const {
  RooRealVar *param = this->getParameter(name);
  if (param) {
    return param->isConstant();
  }
  return true;
}

////////////////////////////////////////////////////////////////////////////////
/// retrieve the RooRealVar object incorporating the parameter with the given
/// name
RooRealVar *RooLagrangianMorphFunc::getParameter(const char *name) const {

  return dynamic_cast<RooRealVar *>(this->_operators.find(name));
}

////////////////////////////////////////////////////////////////////////////////
/// retrieve the RooRealVar object incorporating the flag with the given name

RooRealVar *RooLagrangianMorphFunc::getFlag(const char *name) const {
  return dynamic_cast<RooRealVar *>(this->_flags.find(name));
}

////////////////////////////////////////////////////////////////////////////////
/// check if a parameter of the given name is contained in the list of known
/// parameters

bool RooLagrangianMorphFunc::hasParameter(const char *name) const {
  return this->getParameter(name);
}

////////////////////////////////////////////////////////////////////////////////
/// call setConstant with the boolean argument provided on the parameter with
/// the given name

void RooLagrangianMorphFunc::setParameterConstant(const char *name,
                                                  bool constant) const {
  RooRealVar *param = this->getParameter(name);
  if (param) {
    return param->setConstant(constant);
  }
}

////////////////////////////////////////////////////////////////////////////////
/// set one parameter to a specific value

double RooLagrangianMorphFunc::getParameterValue(const char *name) const {
  RooRealVar *param = this->getParameter(name);
  if (param) {
    return param->getVal();
  }
  return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// set the morphing parameters to those supplied in the given param hist

void RooLagrangianMorphFunc::setParameters(TH1 *paramhist) {
  setParams(paramhist, this->_operators, false);
}

////////////////////////////////////////////////////////////////////////////////
/// set the morphing parameters to those supplied in the sample with the given
/// name

void RooLagrangianMorphFunc::setParameters(const char *foldername) {
  std::string filename = this->_config.getFileName();
  TDirectory *file = openFile(filename.c_str());
  TH1 *paramhist = getParamHist(file, foldername);
  setParams(paramhist, this->_operators, false);
  closeFile(file);
}

/////////////////////////////////////////////////////////////////////////////////
/// retrieve the morphing parameters associated to the sample with the given
/// name

RooLagrangianMorphFunc::ParamSet
RooLagrangianMorphFunc::getMorphParameters(const char *foldername) const {
  const std::string name(foldername);
  return this->_config.getParamCards().at(name);
}

////////////////////////////////////////////////////////////////////////////////
/// set the morphing parameters to those supplied in the list with the given
/// name

void RooLagrangianMorphFunc::setParameters(const RooArgList *list) {
  for (auto itr : *list) {
    RooRealVar *param = dynamic_cast<RooRealVar *>(itr);
    if (!param)
      continue;
    this->setParameter(param->GetName(), param->getVal());
  }
}

////////////////////////////////////////////////////////////////////////////////
/// retrieve the histogram observable

RooRealVar *RooLagrangianMorphFunc::getObservable() const {
  if (this->_observables.getSize() < 1) {
    coutE(InputArguments) << "observable not available!" << std::endl;
    return nullptr;
  }
  return static_cast<RooRealVar *>(this->_observables.at(0));
}

////////////////////////////////////////////////////////////////////////////////
/// retrieve the histogram observable

RooRealVar *RooLagrangianMorphFunc::getBinWidth() const {
  if (this->_binWidths.getSize() < 1) {
    coutE(InputArguments) << "bin width not available!" << std::endl;
    return nullptr;
  }
  return static_cast<RooRealVar *>(this->_binWidths.at(0));
}

////////////////////////////////////////////////////////////////////////////////
/// retrieve a histogram output of the current morphing settings

TH1 *RooLagrangianMorphFunc::createTH1(const std::string &name
                                       /*RooFitResult *r*/) {
  return this->createTH1(name, false /*r*/);
}

////////////////////////////////////////////////////////////////////////////////
/// retrieve a histogram output of the current morphing settings

TH1 *RooLagrangianMorphFunc::createTH1(const std::string &name,
                                       bool correlateErrors
                                       /* RooFitResult *r*/) {
  auto mf = std::make_unique<RooRealSumFunc>(*(this->getFunc()));
  RooRealVar *observable = this->getObservable();

  const int nbins = observable->getBins();

  TH1 *hist = new TH1F(name.c_str(), name.c_str(), nbins,
                       observable->getBinning().array());

  //bool ownResult = !(bool)(r);
  RooArgSet *args = mf->getComponents();
  for (int i = 0; i < nbins; ++i) {
    observable->setBin(i);
    double val = 0;
    double unc2 = 0;
    double unc = 0;
    for (auto itr : *args) {
      RooProduct *prod = dynamic_cast<RooProduct *>(itr);
      if (!prod)
        continue;
      RooAbsArg *phys =
          prod->components().find(Form("phys_%s", prod->GetName()));
      RooHistFunc *hf = dynamic_cast<RooHistFunc *>(phys);
      if (!hf) {
        continue;
      }
      const RooDataHist &dhist = hf->dataHist();
      dhist.get(i);
      RooAbsReal *formula = dynamic_cast<RooAbsReal *>(
          prod->components().find(Form("w_%s", prod->GetName())));
      double weight = formula->getVal();
      unc2 += dhist.weightSquared() * weight * weight;
      unc += sqrt(dhist.weightSquared()) * weight;
      val += dhist.weight() * weight;
    }
    hist->SetBinContent(i + 1, val);
    hist->SetBinError(i + 1, correlateErrors ? unc : sqrt(unc2));
  }
  /*if (ownResult)
    delete r;*/
  return hist;
}

////////////////////////////////////////////////////////////////////////////////
/// count the number of formulas that correspond to the current parameter set

int RooLagrangianMorphFunc::countContributingFormulas() const {
  int nFormulas = 0;
  auto mf = std::make_unique<RooRealSumFunc>(*(this->getFunc()));
  if (!mf)
    coutE(InputArguments) << "unable to retrieve morphing function"
                          << std::endl;
  RooArgSet *args = mf->getComponents();
  for (auto itr : *args) {
    RooProduct *prod = dynamic_cast<RooProduct *>(itr);
    if (prod->getVal() != 0) {
      nFormulas++;
    }
  }
  return nFormulas;
}

////////////////////////////////////////////////////////////////////////////////
/// check if there is any morphing power provided for the given parameter
/// morphing power is provided as soon as any two samples provide different,
/// non-zero values for this parameter

bool RooLagrangianMorphFunc::isParameterUsed(const char *paramname) const {
  std::string pname(paramname);
  double val = 0;
  bool isUsed = false;
  for (const auto &sample : this->_config.getParamCards()) {
    double thisval = sample.second.at(pname);
    if (thisval != val) {
      if (val != 0)
        isUsed = true;
      val = thisval;
    }
  }
  return isUsed;
}

////////////////////////////////////////////////////////////////////////////////
/// check if there is any morphing power provided for the given coupling
/// morphing power is provided as soon as any two samples provide
/// different, non-zero values for this coupling

bool RooLagrangianMorphFunc::isCouplingUsed(const char *couplname) {
  std::string cname(couplname);
  const RooArgList *args = this->getCouplingSet();
  RooAbsReal *coupling = dynamic_cast<RooAbsReal *>(args->find(couplname));
  if (!coupling)
    return false;
  RooLagrangianMorphFunc::ParamSet params = this->getMorphParameters();
  double val = 0;
  bool isUsed = false;
  for (const auto &sample : this->_config.getParamCards()) {
    this->setParameters(sample.second);
    double thisval = coupling->getVal();
    if (thisval != val) {
      if (val != 0)
        isUsed = true;
      val = thisval;
    }
  }
  this->setParameters(params);
  return isUsed;
}

////////////////////////////////////////////////////////////////////////////////
/// return the number of parameters in this morphing function

int RooLagrangianMorphFunc::nParameters() const {
  return this->getParameterSet()->getSize();
}

////////////////////////////////////////////////////////////////////////////////
/// return the number of samples in this morphing function

int RooLagrangianMorphFunc::nPolynomials() const {
  // return the number of samples in this morphing function
  auto cache = getCache(_curNormSet);
  return cache->_formulas.size();
}

////////////////////////////////////////////////////////////////////////////////
/// print the contributing samples and their respective weights

void RooLagrangianMorphFunc::printEvaluation() const {
  auto mf = std::make_unique<RooRealSumFunc>(*(this->getFunc()));
  if (!mf) {
    std::cerr << "Error: unable to retrieve morphing function" << std::endl;
    return;
  }
  RooArgSet *args = mf->getComponents();
  for (auto itr : *args) {
    RooAbsReal *formula = dynamic_cast<RooAbsReal *>(itr);
    if (formula) {
      TString name(formula->GetName());
      name.Remove(0, 2);
      name.Prepend("phys_");
      if (!args->find(name.Data())) {
        continue;
      }
      double val = formula->getVal();
      if (val != 0) {
        std::cout << formula->GetName() << ": " << val << " = "
                  << formula->GetTitle() << std::endl;
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
/// get the set of parameters

const RooArgSet *RooLagrangianMorphFunc::getParameterSet() const {
  return &(this->_operators);
}

////////////////////////////////////////////////////////////////////////////////
/// get the set of couplings

const RooArgList *RooLagrangianMorphFunc::getCouplingSet() const {
  auto cache = getCache(_curNormSet);
  return &(cache->_couplings);
}

////////////////////////////////////////////////////////////////////////////////
/// retrieve a set of couplings (-?-)

RooLagrangianMorphFunc::ParamSet RooLagrangianMorphFunc::getCouplings() const {
  RooLagrangianMorphFunc::ParamSet couplings;
  for (auto obj : *(this->getCouplingSet())) {
    RooAbsReal *var = dynamic_cast<RooAbsReal *>(obj);
    if (!var)
      continue;
    const std::string name(var->GetName());
    double val = var->getVal();
    couplings[name] = val;
  }
  return couplings;
}

////////////////////////////////////////////////////////////////////////////////
/// retrieve the parameter set

RooLagrangianMorphFunc::ParamSet RooLagrangianMorphFunc::getMorphParameters() const {
  return getParams(this->_operators);
}

////////////////////////////////////////////////////////////////////////////////
/// retrieve a set of couplings (-?-)

void RooLagrangianMorphFunc::setParameters(const ParamSet &params) {
  setParams(params, this->_operators, false);
}

////////////////////////////////////////////////////////////////////////////////
/// (currently similar to cloning the Pdf

std::unique_ptr<RooWrapperPdf> RooLagrangianMorphFunc::createPdf() const {
  auto cache = getCache(_curNormSet);
  auto func = std::make_unique<RooRealSumFunc>(*(cache->_sumFunc));

  // create a wrapper on the roorealsumfunc
  return std::make_unique<RooWrapperPdf>(Form("pdf_%s", func->GetName()),
                                         Form("pdf of %s", func->GetTitle()),
                                         *func);
}
/*
////////////////////////////////////////////////////////////////////////////////
/// get a standalone clone of the pdf that does not depend on this object

RooWrapperPdf* RooLagrangianMorphFunc::clonePdf() const
{
  auto cache = getCache(_curNormSet);
  RooRealSumFunc* func = cache->_sumFunc;
  // create a wrapper on the roorealsumfunc
  RooWrapperPdf* x = new RooWrapperPdf(Form("pdf_%s",func->GetName()),Form("pdf
of %s",func->GetTitle()), *func); return x;
}
*/
////////////////////////////////////////////////////////////////////////////////
/// get the func

RooRealSumFunc *RooLagrangianMorphFunc::getFunc() const {
  auto cache = getCache(_curNormSet);
  return cache->_sumFunc.get();
}

////////////////////////////////////////////////////////////////////////////////
/// get a standalone clone of the func that does not depend on this object
/* do we need a cloneFunc method ?
   i get segfault when i do,
   std::unique_ptr<RooRealSumFunc> morphfunc_ptjj1 = morphfunc_ptjj.cloneFunc();
   morphfunc_ptjj1.get()->Print();

std::unique_ptr<RooRealSumFunc> RooLagrangianMorphFunc::cloneFunc() const
{
  auto cache = getCache(_curNormSet);
  auto orig = cache->_sumFunc.get();
  return
std::make_unique<RooRealSumFunc>(orig->GetName(),orig->GetTitle(),orig->funcList(),orig->coefList());
}
*/
////////////////////////////////////////////////////////////////////////////////
/// return extended mored capabilities

RooAbsPdf::ExtendMode RooLagrangianMorphFunc::extendMode() const {
  return this->createPdf()->extendMode();
}

////////////////////////////////////////////////////////////////////////////////
/// return expected number of events for extended likelihood calculation,
/// this is the sum of all coefficients

Double_t RooLagrangianMorphFunc::expectedEvents(const RooArgSet *nset) const {
  return this->createPdf()->expectedEvents(nset);
}

////////////////////////////////////////////////////////////////////////////////
/// return the number of expected events for the current parameter set

Double_t RooLagrangianMorphFunc::expectedEvents() const {
  RooArgSet set;
  set.add(*this->getObservable());
  return this->createPdf()->expectedEvents(set);
}

////////////////////////////////////////////////////////////////////////////////
/// return expected number of events for extended likelihood calculation,
/// this is the sum of all coefficients

Double_t RooLagrangianMorphFunc::expectedEvents(const RooArgSet &nset) const {
  return createPdf()->expectedEvents(&nset);
}
////////////////////////////////////////////////////////////////////////////////
/// return the expected uncertainty for the current parameter set

double RooLagrangianMorphFunc::expectedUncertainty() const {
  RooRealVar *observable = this->getObservable();
  auto cache = this->getCache(_curNormSet);
  double unc2 = 0;
  for (const auto &sample : this->_sampleMap) {
    RooAbsArg *phys = this->_physics.at(sample.second);
    auto weightName = std::string("w_") + sample.first + "_" + this->GetName();
    auto weight = static_cast<RooAbsReal *>(cache->_weights.find(weightName.c_str()));
    if (!weight) {
      coutE(InputArguments)
          << "unable to find object " + weightName << std::endl;
      return 0.0;
    }
    double newunc2 = 0;
    RooHistFunc *hf = dynamic_cast<RooHistFunc *>(phys);
    RooRealVar *rv = dynamic_cast<RooRealVar *>(phys);
    if (hf) {
      const RooDataHist &hist = hf->dataHist();
      for (Int_t j = 0; j < observable->getBins(); ++j) {
        hist.get(j);
        newunc2 += hist.weightSquared();
      }
    } else if (rv) {
      newunc2 = pow(rv->getError(), 2);
    }
    double w = weight->getVal();
    unc2 += newunc2 * w * w;
    // std::cout << phys->GetName() << " : " << weight->GetName() << "
    // thisweight: " <<  w << " thisxsec2: " << newunc2 << " weight " << weight
    // << std::endl;
  }
  return sqrt(unc2);
}

////////////////////////////////////////////////////////////////////////////////
/// print the parameters and their current values

void RooLagrangianMorphFunc::printParameters() const {
  // print the parameters and their current values
  for (auto obj : this->_operators) {
    RooRealVar *param = static_cast<RooRealVar *>(obj);
    if (!param)
      continue;
    param->Print();
  }
}

////////////////////////////////////////////////////////////////////////////////
/// print the flags and their current values

void RooLagrangianMorphFunc::printFlags() const {
  for (auto flag : this->_flags) {
    RooRealVar *param = static_cast<RooRealVar *>(flag);
    if (!param)
      continue;
    param->Print();
  }
}

////////////////////////////////////////////////////////////////////////////////
/// print a set of couplings

void RooLagrangianMorphFunc::printCouplings() const {
  RooLagrangianMorphFunc::ParamSet couplings = this->getCouplings();
  for (auto c : couplings) {
    std::cout << c.first << ": " << c.second << std::endl;
  }
}

////////////////////////////////////////////////////////////////////////////////
/// retrieve the list of bin boundaries

std::list<Double_t> *
RooLagrangianMorphFunc::binBoundaries(RooAbsRealLValue &obs, Double_t xlo,
                                      Double_t xhi) const {
  return this->getFunc()->binBoundaries(obs, xlo, xhi);
}

////////////////////////////////////////////////////////////////////////////////
/// retrieve the sample Hint

std::list<Double_t> *
RooLagrangianMorphFunc::plotSamplingHint(RooAbsRealLValue &obs, Double_t xlo,
                                         Double_t xhi) const {
  return this->getFunc()->plotSamplingHint(obs, xlo, xhi);
}

////////////////////////////////////////////////////////////////////////////////
/// call getVal on the internal function

Double_t RooLagrangianMorphFunc::getValV(const RooArgSet *set) const {
  // cout << "XX RooLagrangianMorphFunc::getValV(" << this << ") set = " << set
  // << std::endl ;
  this->_curNormSet = set;
  return RooAbsReal::getValV(set);
}

////////////////////////////////////////////////////////////////////////////////
/// call getVal on the internal function

Double_t RooLagrangianMorphFunc::evaluate() const {
  // call getVal on the internal function
  RooRealSumFunc *pdf = this->getFunc();
  if (pdf)
    return this->_scale * pdf->getVal(_curNormSet);
  else
    std::cerr << "unable to acquire in-built function!" << std::endl;
  return 0.;
}

////////////////////////////////////////////////////////////////////////////////
/// check if this PDF is a binned distribution in the given observable

Bool_t
RooLagrangianMorphFunc::isBinnedDistribution(const RooArgSet &obs) const {
  return this->getFunc()->isBinnedDistribution(obs);
}

////////////////////////////////////////////////////////////////////////////////
/// check if observable exists in the RooArgSet (-?-)

Bool_t RooLagrangianMorphFunc::checkObservables(const RooArgSet *nset) const {
  return this->getFunc()->checkObservables(nset);
}

////////////////////////////////////////////////////////////////////////////////
/// Force analytical integration for the given observable

Bool_t RooLagrangianMorphFunc::forceAnalyticalInt(const RooAbsArg &arg) const {
  return this->getFunc()->forceAnalyticalInt(arg);
}

////////////////////////////////////////////////////////////////////////////////
/// Retrieve the mat

Int_t RooLagrangianMorphFunc::getAnalyticalIntegralWN(
    RooArgSet &allVars, RooArgSet &numVars, const RooArgSet *normSet,
    const char *rangeName) const {
  return this->getFunc()->getAnalyticalIntegralWN(allVars, numVars, normSet,
                                                  rangeName);
}

////////////////////////////////////////////////////////////////////////////////
/// Retrieve the matrix of coefficients

Double_t RooLagrangianMorphFunc::analyticalIntegralWN(
    Int_t code, const RooArgSet *normSet, const char *rangeName) const {
  return this->getFunc()->analyticalIntegralWN(code, normSet, rangeName);
}

////////////////////////////////////////////////////////////////////////////////
/// Retrieve the matrix of coefficients

void RooLagrangianMorphFunc::printMetaArgs(std::ostream &os) const {
  return this->getFunc()->printMetaArgs(os);
}

////////////////////////////////////////////////////////////////////////////////
/// Retrieve the matrix of coefficients

RooAbsArg::CacheMode RooLagrangianMorphFunc::canNodeBeCached() const {
  return this->getFunc()->canNodeBeCached();
}

////////////////////////////////////////////////////////////////////////////////
/// Retrieve the matrix of coefficients

void RooLagrangianMorphFunc::setCacheAndTrackHints(RooArgSet &arg) {
  this->getFunc()->setCacheAndTrackHints(arg);
}

////////////////////////////////////////////////////////////////////////////////
/// Retrieve the matrix of coefficients

TMatrixD RooLagrangianMorphFunc::getMatrix() const {
  auto cache = getCache(_curNormSet);
  if (!cache)
    coutE(Caching) << "unable to retrieve cache!" << std::endl;
  return makeRootMatrix(cache->_matrix);
}

////////////////////////////////////////////////////////////////////////////////
/// Retrieve the matrix of coefficients after inversion

TMatrixD RooLagrangianMorphFunc::getInvertedMatrix() const {
  auto cache = getCache(_curNormSet);
  if (!cache)
    coutE(Caching) << "unable to retrieve cache!" << std::endl;
  return makeRootMatrix(cache->_inverse);
}

////////////////////////////////////////////////////////////////////////////////
/// Retrieve the condition of the coefficient matrix. If the condition number
/// is very large, then the matrix is ill-conditioned and is almost singular.
/// The computation of the inverse is prone to large numerical errors

double RooLagrangianMorphFunc::getCondition() const {
  auto cache = getCache(_curNormSet);
  if (!cache)
    coutE(Caching) << "unable to retrieve cache!" << std::endl;
  return cache->_condition;
}

////////////////////////////////////////////////////////////////////////////////
/// Return the RooRatio form of products and denominators of morphing functions

std::unique_ptr<RooRatio> RooLagrangianMorphFunc::makeRatio(const char *name,
                                                            const char *title,
                                                            RooArgList &nr,
                                                            RooArgList &dr) {
  RooArgList num, denom;
  for (auto it : nr) {
    num.add(*it);
  }
  for (auto it : dr) {
    denom.add(*it);
  }
  // same for denom
  return make_unique<RooRatio>(name, title, num, denom);
}
