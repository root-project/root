/*
 * Project: xRooFit
 * Author:
 *   Will Buttinger, RAL 2022
 *
 * Copyright (c) 2022, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#include "Config.h"

#ifdef XROOFIT_USE_PRAGMA_ONCE
#pragma once
#endif
#if !defined(XROOFIT_XROONODE_H) || defined(XROOFIT_USE_PRAGMA_ONCE)
#ifndef XROOFIT_USE_PRAGMA_ONCE
#define XROOFIT_XROONODE_H
#endif

#include "TNamed.h"
#include <vector>
#include <functional>

class RooWorkspace;
class RooAbsReal;
class TH1;
class RooAbsLValue;
class RooArgList;
class RooAbsBinning;
class RooFitResult;
class TGraph;
class TAxis;
class TGListTreeItem;
class TGListTree;
class TVirtualPad;
class TStyle;

#include "xRooFit.h"
#include "RooLinkedList.h"
#include "RooCmdArg.h"
#include "TQObject.h"

BEGIN_XROOFIT_NAMESPACE

class xRooNode;
class xRooNLLVar;

class xRooNode : public TNamed, public std::vector<std::shared_ptr<xRooNode>> {

public:
   // functions of form value = f(orig,nom,nom_err)
   // e.g. f = ratio would be simply orig/nom
   // bool indicates if should be symmetrized
   static std::map<std::string, std::tuple<std::function<double(double, double, double)>, bool>> auxFunctions;
   static void SetAuxFunction(const char *title, const std::function<double(double, double, double)> &func,
                              bool symmetrize = false);

   // this function is here because couldn't figure out how to check a null shared_ptr in pyroot
   static inline bool isNull(const std::shared_ptr<xRooNode> &x) { return x == nullptr; }

   // name of the node needn't match the name of the component that it points to
   // the name of the node is how it is identified inside its parent
   // In c++17 for constructors with a Node2& parent, could look at using shared_from_this and if it throws except then
   // construct make_shared<Node2>(parent)
   xRooNode(const char *type, const char *name, const char *title = "");
   template <typename T>
   xRooNode(const char *name, const char *title) : TNamed(name, title), fComp(std::make_shared<T>())
   {
      if (auto x = get<TNamed>(); x) {
         x->SetNameTitle(name, title);
      }
   }
   xRooNode(const char *name = "", const std::shared_ptr<TObject> &comp = nullptr,
            const std::shared_ptr<xRooNode> &parent = nullptr);
   xRooNode(const char *name, const std::shared_ptr<TObject> &comp, const xRooNode &parent)
      : xRooNode(name, comp, std::make_shared<xRooNode>(parent))
   {
   }
   xRooNode(const char *name, const TObject &comp, const std::shared_ptr<xRooNode> &parent)
      : xRooNode(name, std::shared_ptr<TObject>(const_cast<TObject *>(&comp), [](TObject *) {}), parent)
   {
   } // needed to ensure passing a shared_ptr<Node2> for the parent doesnt become Node2(shared_ptr<Node2>) as parent
     // because of Node2(shared_ptr<TObject>) constructor
   xRooNode(const char *name, const TObject &comp, const xRooNode &parent)
      : xRooNode(name, std::shared_ptr<TObject>(const_cast<TObject *>(&comp), [](TObject *) {}), parent)
   {
   }
   xRooNode(const TObject &comp, const std::shared_ptr<xRooNode> &parent = nullptr);
   xRooNode(const TObject &comp, const xRooNode &parent) : xRooNode(comp, std::make_shared<xRooNode>(parent)) {}
   xRooNode(const std::shared_ptr<TObject> &comp, const std::shared_ptr<xRooNode> &parent = nullptr);
   template <typename T>
   xRooNode(const std::shared_ptr<T> &comp, const std::shared_ptr<xRooNode> &parent = nullptr)
      : xRooNode(std::dynamic_pointer_cast<TObject>(comp), parent)
   {
   }
   template <typename T>
   xRooNode(const std::shared_ptr<T> &comp, const xRooNode &parent)
      : xRooNode(std::dynamic_pointer_cast<TObject>(comp), std::make_shared<xRooNode>(parent))
   {
   }
   template <typename T>
   xRooNode(const std::shared_ptr<const T> &comp, const std::shared_ptr<xRooNode> &parent = nullptr)
      : xRooNode(std::dynamic_pointer_cast<TObject>(std::const_pointer_cast<T>(comp)), parent)
   {
   }
   template <typename T>
   xRooNode(const std::shared_ptr<const T> &comp, const xRooNode &parent)
      : xRooNode(std::dynamic_pointer_cast<TObject>(std::const_pointer_cast<T>(comp)),
                 std::make_shared<xRooNode>(parent))
   {
   }
   xRooNode(double value);

   virtual ~xRooNode();

   void SetName(const char *name) override; // *MENU*
   void SetTitle(const char *title) override
   {
      if (auto o = (get<TNamed>()); o)
         o->SetTitle(title);
      TNamed::SetTitle(title);
   } // *MENU*

   const char *GetNodeType() const;

   explicit operator bool() const { return strlen(GetName()) || get(); } // the 'null' Component is the empty string

   // at doesn't do an initial browse of the object, unlike [] operator
   const std::shared_ptr<xRooNode> &at(size_t idx, bool browseResult = true) const
   {
      IsFolder();
      auto &out = std::vector<std::shared_ptr<xRooNode>>::at(idx);
      if (browseResult && out)
         out->browse();
      return out;
   }
   std::shared_ptr<xRooNode> at(const std::string &name, bool browseResult = true) const;

   RooArgList argList() const;

   std::shared_ptr<xRooNode> find(const std::string &name) const;
   bool contains(const std::string &name) const; // doesn't trigger a browse of the found object, unlike find

   // most users should use these methods: will do an initial browse and will browse the returned object too
   std::shared_ptr<xRooNode> operator[](size_t idx) { return at(idx); }
   std::shared_ptr<xRooNode> operator[](const std::string &name); // will create a child node if not existing

   // needed in pyROOT to avoid it creating iterators that follow the 'get' to death
   auto begin() const -> decltype(std::vector<std::shared_ptr<xRooNode>>::begin())
   {
      return std::vector<std::shared_ptr<xRooNode>>::begin();
   }
   auto end() const -> decltype(std::vector<std::shared_ptr<xRooNode>>::end())
   {
      return std::vector<std::shared_ptr<xRooNode>>::end();
   }

   void Browse(TBrowser *b = nullptr) override; // will browse the children that aren't "null" nodes
   bool IsFolder() const override;
   const char *GetIconName() const override;
   void Inspect() const override; // *MENU*

   xRooNode &browse(); // refreshes child nodes

   std::string GetPath() const;
   void Print(Option_t *opt = "") const override; // *MENU*
   // void Reverse() {
   // std::reverse(std::vector<std::shared_ptr<Node2>>::begin(),std::vector<std::shared_ptr<Node2>>::end()); } // *MENU*

   xRooNode &operator=(const TObject &o);

   TObject *get() const { return fComp.get(); }
   template <typename T>
   T *get() const
   {
      return dynamic_cast<T *>(get());
   }

   TObject *operator->() const { return get(); }

   RooWorkspace *ws() const;
   std::shared_ptr<TObject>
   acquire(const std::shared_ptr<TObject> &arg, bool checkFactory = false, bool mustBeNew = false);
   // common pattern for 'creating' an acquired object
   template <typename T, typename... Args>
   std::shared_ptr<T> acquire(Args &&...args)
   {
      return std::dynamic_pointer_cast<T>(acquire(std::make_shared<T>(std::forward<Args>(args)...)));
   }
   template <typename T, typename... Args>
   std::shared_ptr<T> acquireNew(Args &&...args)
   {
      return std::dynamic_pointer_cast<T>(acquire(std::make_shared<T>(std::forward<Args>(args)...), false, true));
   }
   std::shared_ptr<TObject> getObject(const std::string &name, const std::string &type = "") const;
   template <typename T>
   std::shared_ptr<T> getObject(const std::string &name) const
   {
      return std::dynamic_pointer_cast<T>(getObject(name, T::Class_Name()));
   }

   xRooNode shallowCopy(const std::string &name, std::shared_ptr<xRooNode> parent = nullptr);

   std::shared_ptr<TObject> convertForAcquisition(xRooNode &acquirer, const char *opt = "") const;

   xRooNode vars() const;   // obs,globs,floats,args
   xRooNode obs() const;    // robs and globs
   xRooNode robs() const;   // just the regular obs
   xRooNode globs() const;  // just the global obs
   xRooNode pars() const;   // floats and args/consts
   xRooNode floats() const; // just floating pars
   xRooNode args() const;   // just const pars
   xRooNode consts() const { return args(); }

   xRooNode poi() const; // parameters of interest
   xRooNode np() const;  // nuisance parameters

   xRooNode components() const; // additive children
   xRooNode factors() const;    // multiplicative children
   xRooNode variations() const; // interpolated children (are bins a form of variation?)
   xRooNode coefs() const;
   xRooNode coords(bool setVals = true) const; // will move to the coords in the process if setVals=true
   xRooNode bins() const;

   xRooNode constraints() const; // pdfs other than the node's parent pdf where the deps of this node appear
   xRooNode datasets()
      const; // datasets corresponding to this pdf (parent nodes that do observable selections automatically applied)

   xRooNode Remove(const xRooNode &child);
   xRooNode
   Add(const xRooNode &child,
       Option_t *opt =
          ""); // = components()[child.GetName()]=child; although need to handle case of adding same term multiple times
   xRooNode Multiply(const xRooNode &child, Option_t *opt = ""); // = factors()[child.GetName()]=child;
   xRooNode Vary(const xRooNode &child);
   xRooNode Constrain(const xRooNode &child);

   xRooNode Combine(const xRooNode &rhs); // combine rhs with this node

   xRooNode reduced(const std::string &range = "")
      const; // return a node representing reduced version of this node, will use the SetRange to reduce if blank

   // following versions are for the menu in the GUI
   void _Add_(const char *name, const char *opt);                     // *MENU*
   xRooNode _Multiply_(const char *what) { return Multiply(what); }   // *MENU*
   void _Vary_(const char *what);                                     // *MENU*
   xRooNode _Constrain_(const char *what) { return Constrain(what); } // *MENU*

   void _ShowVars_(Bool_t set = kTRUE); // *TOGGLE* *GETTER=_IsShowVars_
   bool _IsShowVars_() const;

   void SetHidden(Bool_t set = kTRUE); // *TOGGLE* *GETTER=IsHidden
   bool IsHidden() const;

   bool SetContents(const TObject &obj)
   {
      operator=(obj);
      return true;
   }                               // populates the node's comp (creating if necessary)  from given object
   bool SetContents(double value); // uses a RooConst
   bool SetContents(double value, const char *par, double parVal = 1); // shortcut to setting a variation content
   bool SetContents(const TObject &obj, const char *par, double parVal)
   {
      variations()[TString::Format("%s=%g", par, parVal).Data()]->operator=(obj);
      return true;
   }
   bool SetBinError(int bin, double value);
   bool SetBinContent(int bin, double value, const char *par = nullptr, double parVal = 1);
   bool SetBinData(int bin, double value, const char *dataName = "obsData"); // only valid for pdf nodes

   void _SetContents_(double value);                                                     // *MENU*
   void _SetBinContent_(int bin, double value, const char *par = "", double parVal = 1); // *MENU*

   bool SetXaxis(const RooAbsBinning &binning);
   bool SetXaxis(const char *name, const char *title, int nbins, double low, double high);
   bool SetXaxis(const char *name, const char *title, int nbins, double *bins);
   bool SetXaxis(const char *title, int nbins, double low, double high)
   {
      return SetXaxis("xaxis", title, nbins, low, high);
   }
   bool SetXaxis(const char *title, int nbins, double *bins) { return SetXaxis("xaxis", title, nbins, bins); }
   bool SetXaxis(int nbins, double low, double high) { return SetXaxis("xaxis", "", nbins, low, high); }
   bool SetXaxis(int nbins, double *bins) { return SetXaxis("xaxis", "", nbins, bins); }

   std::shared_ptr<TStyle> style(TObject *initObject = nullptr) const;

   TAxis *GetXaxis() const;

   double GetBinData(int bin, const char *dataName = "obsData");
   double GetBinContent(int bin) const { return GetBinContents(bin, bin).at(0); }
   std::vector<double> GetBinContents(int binStart = 1, int binEnd = 0) const; // default will get all bins
   double GetBinError(int bin, const xRooNode &fr = "") const;
   std::vector<double> GetBinErrors(int binStart = 1, int binEnd = 0, const xRooNode &fr = "") const;
   std::pair<double, double> IntegralAndError(const xRooNode &fr = "", const char *rangeName = nullptr) const;

   // methods to access default content and error
   double GetContent() const { return GetBinContent(0); }
   double GetError() const { return GetBinError(0); }

   xRooNLLVar nll(const xRooNode &_data, std::initializer_list<RooCmdArg> nllOpts) const;
   xRooNLLVar nll(const xRooNode &_data, const RooLinkedList &nllOpts) const;
   xRooNLLVar nll(const xRooNode &_data = "") const; // uses xRooFit::createNLLOption for nllOpts

   xRooNode fitResult(const char *opt = "") const;      // todo: make this 'fitResults'
   void SetFitResult(const RooFitResult *fr = nullptr); // null means will load prefit
   void SetFitResult(const std::shared_ptr<const RooFitResult> &fr) { SetFitResult(fr.get()); }
   void SetFitResult(const xRooNode &fr);

   void _fitTo_(const char *datasetName = "", const char *constParValues = ""); // *MENU*
   void _generate_(const char *name = "", bool expected = false);               // *MENU*
   //    xRooNode fitTo(const char* datasetName) const;
   //    xRooNode fitTo(const xRooNode& _data) const;
   //    xRooNode generate(bool expected=false) const;
   //    void minosScan(const char* parName); // *MENU*
   //    void pllScan(const char* parName, int npoints=100); // *MENU*
   //    void breakdown(const char* parNames, const char* groupNames); // *MENU*

   /*
   double pll(Node2& data, const char* parName, double value, const Asymptotics::PLLType& pllType =
   Asymptotics::TwoSided) const;
   // pair is obs p_sb and p_b, vector is expected -2->+2 sigma p_sb (p_b are known by construction)
   std::pair<std::pair<double,double>,std::vector<double>>  pValue(Node2& data, const char* parName, double value,
   double alt_value, const Asymptotics::PLLType& pllType); double sigma_mu(Node2& data, const char* parName, double
   value, double alt_value) const;
*/

   void Checked(TObject *obj, bool val);
   void SetChecked(bool val = true) { Checked(this, val); }

   TGraph *BuildGraph(RooAbsLValue *v = nullptr, bool includeZeros = false, TVirtualPad *fromPad = nullptr) const;
   TH1 *BuildHistogram(RooAbsLValue *v = nullptr, bool empty = false, bool errors = false, int binStart = 1,
                       int binEnd = 0) const;
   xRooNode mainChild() const;
   void Draw(Option_t *opt = "") override; // *MENU*

   void SaveAs(const char *filename = "", Option_t *option = "") const override; // *MENU*

   TGListTreeItem *GetTreeItem(TBrowser *b) const;
   TGListTree *GetListTree(TBrowser *b) const;

   static void Interactive_PLLPlot();
   static void Interactive_Pull();
   class InteractiveObject : public TQObject {
   public:
      void Interactive_PLLPlot(TVirtualPad *pad, TObject *obj, Int_t x, Int_t y);
      ClassDef(InteractiveObject, 0)
   };
   static InteractiveObject *gIntObj;

   mutable std::shared_ptr<TObject> fComp; //!
   int fTimes = 1;      // when the same comp appears multiple times in a parent node, this is increased to reflect that
   int fBinNumber = -1; // used by 'bin' nodes (a node that refers to a specific bin of a parent)
   std::shared_ptr<xRooNode> fParent; //!
   std::string fFolder = "";          // folder to put this node in when 'organising' the parent

   void SetRange(const char *range, double low = std::numeric_limits<double>::quiet_NaN(),
                 double high = std::numeric_limits<double>::quiet_NaN()); // *MENU*
   const char *GetRange() const;
   mutable std::string fRange; //! only here so can have char* GetRange return so can return nullptr for no range set
                               //! (required for RooCategory)

   mutable std::shared_ptr<TAxis>
      fXAxis; //! appears that if was fXaxis then dialog box for SetXaxis will take as current value

   mutable bool fInterrupted = false;

   bool fAcquirer = false; // if true, when acquiring will go into objects memory rather than pass onto parent
   std::shared_ptr<xRooNode> fProvider; //! like a parent but only for use by getObject

   std::shared_ptr<xRooNode> parentPdf() const; // find first parent that is a pdf

   void sterilize();

   std::vector<std::shared_ptr<xRooNode>> fBrowsables;   // will appear in the browser tree but are not actual children
   std::function<xRooNode(xRooNode *)> fBrowseOperation; // a way to specify a custom browsing operation

   ClassDefOverride(xRooNode, 0)
};

END_XROOFIT_NAMESPACE

#endif // include guard
