//
// Created by Will Buttinger on 13/03/2021.
//

#pragma once

#include "TNamed.h"
#include <vector>
#include <functional>

class TAxis;

class RooNode;
class RooWorkspace;
class RooAbsReal;
class TH1;
class RooAbsLValue;
class RooArgList;
class RooAbsBinning;
class TGraph;
class RooFitResult;
class TGListTreeItem;
class TVirtualPad;


#include "RooLinkedList.h"
#include "RooCmdArg.h"
#include "TQObject.h"

class xRooNLLVar;

class RooNode : public TNamed, public std::vector<std::shared_ptr<RooNode>> {

public:
    // functions of form value = f(orig,nom,nom_err)
    // e.g. f = ratio would be simply orig/nom
    // bool indicates if should be symmetrized
    static std::map<std::string,std::tuple<std::function<double(double,double,double)>,bool>> auxFunctions;
    static void SetAuxFunction(const char* title, const std::function<double(double,double,double)>& func, bool symmetrize = false );

    // this function is here because couldn't figure out how to check a null shared_ptr in pyroot
    static inline bool isNull(const std::shared_ptr<RooNode>& x) {return x==nullptr; }

    // name of the node needn't match the name of the component that it points to
    // the name of the node is how it is identified inside its parent
    // In c++17 for constructors with a Node2& parent, could look at using shared_from_this and if it throws except then construct make_shared<Node2>(parent)
    RooNode(const char* type, const char* name, const char* title="");
    template<typename T> RooNode(const char* name, const char* title) : TNamed(name, title), fComp(std::make_shared<T>()) {
        if (auto x = get<TNamed>(); x) { x->SetNameTitle(name,title); }
    }
    RooNode(const char* name="", const std::shared_ptr<TObject>& comp = nullptr, const std::shared_ptr<RooNode>& parent = nullptr);
    RooNode(const char* name, const std::shared_ptr<TObject>& comp, const RooNode& parent) : RooNode(name,comp,std::make_shared<RooNode>(parent)) { }
    RooNode(const char* name, const TObject& comp, const std::shared_ptr<RooNode>& parent) : RooNode(name,std::shared_ptr<TObject>(const_cast<TObject*>(&comp),[](TObject*){}),parent) { } // needed to ensure passing a shared_ptr<Node2> for the parent doesnt become Node2(shared_ptr<Node2>) as parent because of Node2(shared_ptr<TObject>) constructor
    RooNode(const char* name, const TObject& comp, const RooNode& parent) : RooNode(name,std::shared_ptr<TObject>(const_cast<TObject*>(&comp),[](TObject*){}),parent) { }
    RooNode(const TObject& comp, const std::shared_ptr<RooNode>& parent = nullptr);
    RooNode(const TObject& comp, const RooNode& parent) : RooNode(comp,std::make_shared<RooNode>(parent)) { }
    RooNode(const std::shared_ptr<TObject>& comp, const std::shared_ptr<RooNode>& parent = nullptr);
    template<typename T> RooNode(const std::shared_ptr<T>& comp, const std::shared_ptr<RooNode>& parent = nullptr) : RooNode(std::dynamic_pointer_cast<TObject>(comp),parent) {}
    template<typename T> RooNode(const std::shared_ptr<T>& comp, const RooNode& parent) : RooNode(std::dynamic_pointer_cast<TObject>(comp),std::make_shared<RooNode>(parent)) {}
    template<typename T> RooNode(const std::shared_ptr<const T>& comp, const std::shared_ptr<RooNode>& parent = nullptr) : RooNode(std::dynamic_pointer_cast<TObject>(std::const_pointer_cast<T>(comp)),parent) {}
    template<typename T> RooNode(const std::shared_ptr<const T>& comp, const RooNode& parent) : RooNode(std::dynamic_pointer_cast<TObject>(std::const_pointer_cast<T>(comp)),std::make_shared<RooNode>(parent)) {}
    RooNode(double value);

    virtual ~RooNode();

    void SetName(const char* name) override; // *MENU*
    void SetTitle(const char* title) override { if(auto o =(get<TNamed>()); o) o->SetTitle(title); TNamed::SetTitle(title); } // *MENU*

    const char* GetNodeType() const;

    explicit operator bool() const { return strlen(GetName()) || get(); } // the 'null' Component is the empty string


    // at doesn't do an initial browse of the object, unlike [] operator
    const std::shared_ptr<RooNode>& at(ssize_t idx, bool browseResult=true) const { IsFolder(); auto& out = std::vector<std::shared_ptr<RooNode>>::at(idx); if(browseResult && out) out->browse(); return out; }
    std::shared_ptr<RooNode> at(const std::string& name,bool browseResult=true) const;

    RooArgList argList() const;

    std::shared_ptr<RooNode> find(const std::string& name) const;
    bool contains(const std::string& name) const; // doesn't trigger a browse of the found object, unlike find

    // most users should use these methods: will do an initial browse and will browse the returned object too
    std::shared_ptr<RooNode> operator[](ssize_t idx) { return at(idx); }
    std::shared_ptr<RooNode> operator[](const std::string& name); // will create a child node if not existing




    // needed in pyROOT to avoid it creating iterators that follow the 'get' to death
    auto begin() const -> decltype(std::vector<std::shared_ptr<RooNode>>::begin()) { return std::vector<std::shared_ptr<RooNode>>::begin(); }
    auto end() const  -> decltype(std::vector<std::shared_ptr<RooNode>>::end()) { return std::vector<std::shared_ptr<RooNode>>::end(); }

    void Browse(TBrowser* b = nullptr) override; // will browse the children that aren't "null" nodes
    bool IsFolder() const override;
    const char* GetIconName() const override;
    void Inspect() const override; // *MENU*

    RooNode& browse(); // refreshes child nodes

    std::string GetPath() const;
    void Print(Option_t *opt = "") const override; // *MENU*
    //void Reverse() { std::reverse(std::vector<std::shared_ptr<Node2>>::begin(),std::vector<std::shared_ptr<Node2>>::end()); } // *MENU*


    RooNode& operator=(const TObject& o);

    TObject* get() const { return fComp.get(); }
    template <typename T> T* get() const { return dynamic_cast<T*>(get()); }

    TObject* operator->() const { return get(); }

    RooWorkspace* ws() const;
    std::shared_ptr<TObject> acquire(const std::shared_ptr<TObject>& arg, bool checkFactory=false, bool mustBeNew=false);
    // common pattern for 'creating' an acquired object
    template<typename T, typename ...Args> std::shared_ptr<T> acquire(Args&& ...args) {
        return std::dynamic_pointer_cast<T>(acquire(std::make_shared<T>(std::forward<Args>(args)...)));
    }
    template<typename T, typename ...Args> std::shared_ptr<T> acquireNew(Args&& ...args) {
        return std::dynamic_pointer_cast<T>(acquire(std::make_shared<T>(std::forward<Args>(args)...),false,true));
    }
    std::shared_ptr<TObject> getObject(const std::string& name, const std::string& type="") const;
    template<typename T> std::shared_ptr<T> getObject(const std::string& name) const {
        return std::dynamic_pointer_cast<T>(getObject(name,T::Class_Name()));
    }

    RooNode shallowCopy(const std::string& name, std::shared_ptr<RooNode> parent = nullptr);

    std::shared_ptr<TObject> convertForAcquisition(RooNode& acquirer) const;

    RooNode obs() const; // robs and globs
    RooNode globs() const; // just the global obs
    RooNode robs() const; // the regular obs
    RooNode pars() const; // floats and args/consts
    RooNode floats() const; // floating pars
    RooNode deps() const; // obs,globs,floats,args
    RooNode args() const; // const pars
    RooNode vars() const; // unconst pars - DEPRECATED

    RooNode poi() const; // parameters of interest
    RooNode np() const; // nuisance parameters

    RooNode components() const; // additive children
    RooNode factors() const; // multiplicative children
    RooNode variations() const; // interpolated children (are bins a form of variation?)
    RooNode coefs() const;
    RooNode coords(bool setVals=true) const; // will move to the coords in the process if setVals=true
    RooNode bins() const;

    RooNode constraints() const; // pdfs other than the node's parent pdf where the deps of this node appear
    RooNode datasets() const; // datasets corresponding to this pdf (parent nodes that do observable selections automatically applied)

    RooNode Remove(const RooNode& child);
    RooNode Add(const RooNode& child, Option_t* opt = ""); // = components()[child.GetName()]=child; although need to handle case of adding same term multiple times
    RooNode Multiply(const RooNode& child, Option_t* opt = ""); // = factors()[child.GetName()]=child;
    RooNode Vary(const RooNode& child);
    RooNode Constrain(const RooNode& child);

    RooNode Combine(const RooNode& rhs); // combine rhs with this node

    RooNode reduced(const std::string& range = ""); // return a node representing reduced version of this node, will use the SetRange to reduce if blank

    // following versions are for the menu in the GUI
    void Add_(const char* name, const char* opt); // *MENU*
    RooNode Multiply_(const char* what) {return Multiply(what); } // *MENU*
    void Vary_(const char* what); // *MENU*
    RooNode Constrain_(const char* what) {return Constrain(what); } // *MENU*

    void SetHidden(Bool_t set = kTRUE); // *TOGGLE* *GETTER=IsHidden
    bool IsHidden() const;

    bool SetContents(const TObject& obj) { operator=(obj); return true; } // populates the node's comp (creating if necessary)  from given object
    bool SetContents(double value); // uses a RooConst
    bool SetContents(double value, const char* par, double parVal=1); // shortcut to setting a variation content
    bool SetContents(const TObject& obj, const char* par, double parVal) { variations()[TString::Format("%s=%g",par,parVal).Data()]->operator=(obj); return true; }
    bool SetBinError(int bin, double value);
    bool SetBinContent(int bin, double value, const char* par=nullptr, double parVal=1);
    bool SetBinData(int bin, double value, const char* dataName="obsData"); // only valid for pdf nodes

    void SetContents_(double value); // *MENU*
    void SetBinContent_(int bin, double value, const char* par="", double parVal=1); // *MENU*


    bool SetXaxis(const RooAbsBinning& binning);
    bool SetXaxis(const char* name, const char* title, int nbins, double low, double high);
    bool SetXaxis(const char* name, const char* title, int nbins, double* bins);
    bool SetXaxis(const char* title, int nbins, double low, double high) { return SetXaxis("xaxis",title,nbins,low,high); }
    bool SetXaxis(const char* title, int nbins, double* bins) { return SetXaxis("xaxis",title,nbins,bins); }
    bool SetXaxis(int nbins, double low, double high) { return SetXaxis("xaxis","",nbins,low,high); }
    bool SetXaxis(int nbins, double* bins) { return SetXaxis("xaxis","",nbins,bins); }

    void SetFillColor(Color_t fcolor);

    TAxis* GetXaxis() const;

    double GetBinData(int bin, const char* dataName="obsData");
    double GetBinContent(int bin) const { return GetBinContents(bin,bin).at(0); }
    std::vector<double> GetBinContents(int binStart=1, int binEnd=0) const; // default will get all bins
    double GetBinError(int bin, const RooFitResult* fr = nullptr) const;
    std::vector<double> GetBinErrors(int binStart=1, int binEnd=0, const RooFitResult* fr = nullptr) const;
    std::pair<double,double> IntegralAndError(const RooFitResult* fr = nullptr) const;

    xRooNLLVar nll(const RooNode& _data, std::initializer_list<RooCmdArg> nllOpts) const;
    xRooNLLVar nll(const RooNode& _data, const RooLinkedList& nllOpts) const;
    xRooNLLVar nll(const RooNode& _data="") const; // uses xRooFit::createNLLOption for nllOpts

    RooNode fitResult(const char* opt="") const; // todo: make this 'fitResults'
    void SetFitResult(const RooFitResult* fr = nullptr); // null means will load prefit
    void SetFitResult(const std::shared_ptr<const RooFitResult>& fr) { SetFitResult(fr.get()); }
    void SetFitResult(const RooNode& fr);

//    RooNode fitTo_(const char* datasetName) const; // *MENU*
//    RooNode fitTo(const char* datasetName) const;
//    RooNode fitTo(const RooNode& _data) const;
//    RooNode generate(bool expected=false) const;
//    void minosScan(const char* parName); // *MENU*
//    void pllScan(const char* parName, int npoints=100); // *MENU*
//    void breakdown(const char* parNames, const char* groupNames); // *MENU*

    /*
    double pll(Node2& data, const char* parName, double value, const Asymptotics::PLLType& pllType = Asymptotics::TwoSided) const;
    // pair is obs p_sb and p_b, vector is expected -2->+2 sigma p_sb (p_b are known by construction)
    std::pair<std::pair<double,double>,std::vector<double>>  pValue(Node2& data, const char* parName, double value, double alt_value, const Asymptotics::PLLType& pllType);
    double sigma_mu(Node2& data, const char* parName, double value, double alt_value) const;
*/

    void Checked(TObject* obj, bool val);
    void SetChecked(bool val=true) { Checked(this,val); }

    TGraph* BuildGraph(RooAbsLValue* v=nullptr, bool includeZeros=false, TVirtualPad* fromPad=nullptr) const;
    TH1* BuildHistogram(RooAbsLValue* v=nullptr, bool empty=false, bool errors=false, int binStart=1, int binEnd=0) const;
    RooNode mainChild() const;
    void Draw(Option_t* opt="") override; // *MENU*

    void SaveAs(const char* filename="", Option_t* option="") const override; // *MENU*

    TGListTreeItem* GetTreeItem(TBrowser* b) const;

    static void Interactive_PLLPlot();
    static void Interactive_Pull();
    class InteractiveObject : public TQObject {
      public:
        void Interactive_PLLPlot(TVirtualPad *pad, TObject *obj, Int_t x, Int_t y);
        ClassDef(InteractiveObject,0)
    };
    static InteractiveObject* gIntObj;

    mutable std::shared_ptr<TObject> fComp; //!
    int fTimes = 1; // when the same comp appears multiple times in a parent node, this is increased to reflect that
    int fBinNumber = -1; // used by 'bin' nodes (a node that refers to a specific bin of a parent)
    std::shared_ptr<RooNode> fParent; //!
    std::string fFolder = ""; // folder to put this node in when 'organising' the parent

    void SetRange(const char* range); // *MENU*
    const char* GetRange() const;
    mutable std::string fRange; //! only here so can have char* GetRange return so can return nullptr for no range set (required for RooCategory)

    mutable std::shared_ptr<TAxis> fXAxis; //! appears that if was fXaxis then dialog box for SetXaxis will take as current value

    mutable bool fInterrupted = false;

    bool fAcquirer = false; // if true, when acquiring will go into objects memory rather than pass onto parent
    std::shared_ptr<RooNode> fProvider; //! like a parent but only for use by getObject


    std::shared_ptr<RooNode> parentPdf() const; // find first parent that is a pdf

    void sterilize();

    std::vector<std::shared_ptr<RooNode>> fBrowsables; // will appear in the browser tree but are not actual children
    std::function<RooNode(RooNode*)> fBrowseOperation; // a way to specify a custom browsing operation

    ClassDefOverride(RooNode,0)

};

