/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id$
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

#ifndef ROO_SIM_WS_TOOL_HH
#define ROO_SIM_WS_TOOL_HH

#include "Rtypes.h"
#include "RooWorkspace.h"
#include "RooCmdArg.h"
#include <list>
#include <map>
#include <string>

class RooAbsCategoryLValue ; 
class RooAbsCategory ;
class RooAbsArg ;
class RooAbsPdf ;
class RooCatType ;
class RooSimultaneous ;


class RooSimWSTool : public TNamed, public RooPrintable {

public:

  // Constructors, assignment etc
  RooSimWSTool(RooWorkspace& ws) ;
  virtual ~RooSimWSTool() ;

  class BuildConfig ;
  class MultiBuildConfig ;
  class SplitRule ;
  
  class ObjBuildConfig ;
  class ObjSplitRule ;
  
  RooSimultaneous* build(const char* simPdfName, const char* protoPdfName, 
			 const RooCmdArg& arg1=RooCmdArg::none(),const RooCmdArg& arg2=RooCmdArg::none(),
			 const RooCmdArg& arg3=RooCmdArg::none(),const RooCmdArg& arg4=RooCmdArg::none(),
			 const RooCmdArg& arg5=RooCmdArg::none(),const RooCmdArg& arg6=RooCmdArg::none()) ;

  RooSimultaneous* build(const char* simPdfName,BuildConfig& bc) ;
  
protected:

  RooSimWSTool(const RooSimWSTool&) ;

  ObjBuildConfig* validateConfig(BuildConfig& bc) ;
  RooSimultaneous* executeBuild(const char* simPdfName,ObjBuildConfig& obc) ;
  std::string makeSplitName(const RooArgSet& splitCatSet) ;

  RooWorkspace* _ws ;

  ClassDef(RooSimWSTool,0) // PDF Clone tool for building of simulteneous p.d.f.s
} ;



class RooSimWSTool::BuildConfig
{
 public:
  BuildConfig(const char* pdfName, SplitRule& sr) ;
  BuildConfig(const char* pdfName, const RooCmdArg& arg1=RooCmdArg::none(),const RooCmdArg& arg2=RooCmdArg::none(),
	      const RooCmdArg& arg3=RooCmdArg::none(),const RooCmdArg& arg4=RooCmdArg::none(),
	      const RooCmdArg& arg5=RooCmdArg::none(),const RooCmdArg& arg6=RooCmdArg::none()) ;

  BuildConfig(const RooArgSet& legacyBuildConfig) ;

  virtual ~BuildConfig() {} ;
  void restrictBuild(const char* catName, const char* stateList) ;

 protected:
  BuildConfig() {} ;
  friend class RooSimWSTool ;
  std::string _masterCatName ;
  std::map<std::string,SplitRule> _pdfmap ;
  std::map<std::string,std::string> _restr ;
  RooCmdArg _conflProtocol ;

  void internalAddPdf(const char* pdfName, const char* miStateList, SplitRule& sr) ;

  ClassDef(BuildConfig,0) // BuildConfig
 } ;


class RooSimWSTool::MultiBuildConfig : public RooSimWSTool::BuildConfig
{
 public:
  MultiBuildConfig(const char* masterIndexCat)  ;
  virtual ~MultiBuildConfig() {} ;
  void addPdf(const char* miStateList, const char* pdfName, SplitRule& sr) ;
  void addPdf(const char* miStateList, const char* pdfName, 
	      const RooCmdArg& arg1=RooCmdArg::none(),const RooCmdArg& arg2=RooCmdArg::none(),
	      const RooCmdArg& arg3=RooCmdArg::none(),const RooCmdArg& arg4=RooCmdArg::none(),
	      const RooCmdArg& arg5=RooCmdArg::none(),const RooCmdArg& arg6=RooCmdArg::none()) ;

 protected:
  friend class RooSimWSTool ;

  ClassDef(MultiBuildConfig,0) // MultiBuildConfig
 } ;


  
class RooSimWSTool::SplitRule : public TNamed {
public:
  SplitRule(const char* pdfName="") : TNamed(pdfName,pdfName) {} ;  
  virtual ~SplitRule() {} ;  
  void splitParameter(const char* paramList, const char* categoryList) ;
  void splitParameterConstrained(const char* paramNameList, const char* categoryNameList, const char* remainderStateName) ;
  
protected:

  friend class RooSimWSTool ;
  friend class BuildConfig ;
  friend class MultiBuildConfig ;  
  void configure(const RooCmdArg& arg1=RooCmdArg::none(),const RooCmdArg& arg2=RooCmdArg::none(),
		 const RooCmdArg& arg3=RooCmdArg::none(),const RooCmdArg& arg4=RooCmdArg::none(),
		 const RooCmdArg& arg5=RooCmdArg::none(),const RooCmdArg& arg6=RooCmdArg::none()) ;

  std::list<std::string>                                             _miStateNameList ;
  std::map<std::string, std::pair<std::list<std::string>,std::string> > _paramSplitMap  ; //<paramName,<list<splitCatSet>,remainderStateName>>
  ClassDef(SplitRule,0) // RooSimPdfSplitRule 
 } ;



class RooSimWSTool::ObjSplitRule {
public:
  ObjSplitRule() {} ;  
  virtual ~ObjSplitRule() ;  
  
protected:
  friend class RooSimWSTool ;
  friend class RooSimWSTool::ObjBuildConfig ;
  std::list<const RooCatType*>                            _miStateList ; 
  std::map<RooAbsArg*, std::pair<RooArgSet,std::string> > _paramSplitMap  ; //<paramName,<list<splitCatSet>,remainderStateName>>
  ClassDef(ObjSplitRule,0) // Object version of SplitRule 
 } ;


class RooSimWSTool::ObjBuildConfig
{
 public:
  ObjBuildConfig() {} ;
  virtual ~ObjBuildConfig() {} ;
  void print() ;

 protected:
  friend class RooSimWSTool ;
  std::map<RooAbsPdf*,ObjSplitRule> _pdfmap ;
  std::map<RooAbsCategory*,std::list<const RooCatType*> > _restr ;
  RooCategory* _masterCat ;
  RooArgSet    _usedSplitCats ;
  RooCmdArg _conflProtocol ;

  ClassDef(ObjBuildConfig,0) // Object version of BuildConfig
 } ;

#endif
