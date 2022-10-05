//
// Created by Will Buttinger on 12/08/2021.
//

#pragma once

#include "xRooFit.h"

class xRooHypoSpace;

class xRooHypoPoint : public TNamed {

public:
    xRooHypoPoint() {}
    xRooHypoPoint(xRooHypoSpace* space, const RooAbsCollection& coords, const RooAbsCollection& alt_coords);
    xRooHypoSpace* fSpace = nullptr;

    const char* GetTitle() const override;

    void Draw(Option_t* opt = "") override;

    double GetObs(const char* name=nullptr) const; // if null will use space obs name
    double GetObsError(const char* name=nullptr) const;
    double GetExp(double nSigma=0,bool asymptotics=false) const; // return expected test stat value based on alt dist

    void FillNull(double value, double w=1.);
    void FillAlt(double value, double w=1.);
    void SetObs(double value, const char* name = nullptr);

    bool HasAsymptotics() const { return !std::isnan(GetSigmaMu().first); }

    size_t GetNullEntries() const { return fNull.size(); }
    size_t GetAltEntries() const { return (fAltPoint && fAltPoint!=this) ? fAltPoint->GetNullEntries() : 0; }

    double GetPCLs(double value, bool asymptotic=false) const { double out= GetPNull(value,asymptotic); if(out==0) return out; return out / GetPAlt(value,asymptotic); }
    double GetPAlt(double value, bool asymptotic=false) const;
    double GetPNull(double value, bool asymptotic=false) const;

    std::pair<double,double> GetSigmaMu() const;

    xRooFit::Asymptotics::PLLType fPllType = xRooFit::Asymptotics::Unknown;

    const RooAbsCollection* fCoords = nullptr;
    const RooAbsCollection* fPOI = nullptr; // values of the parameters of interest used for defining test stat

    xRooHypoPoint* fAltPoint = nullptr; // same pllType and fPOI (so same test statistic) but different coords

    std::vector<std::pair<double,double>> fNull;
    std::map<std::string, std::pair<double,double>> fObs; // named observed values - may include asimovData

    ClassDefOverride(xRooHypoPoint,1)
};

class RooArgList;

class xRooHypoSpace : public TNamed {

public:

    
    static double GetGraphX(const TGraph& graph, double y0=0.05, Option_t* opt="");

    xRooHypoSpace();
    xRooHypoSpace(const char* name, const char* title, const RooAbsCollection& poi);

    const RooArgList& GetAxisVars() const;

    void Draw(Option_t* opt="") override;
    void Draw(Option_t* opt, Option_t* select);

    xRooHypoPoint* GetPoint(const std::string& coords, const std::string& alt_coords="");

    // returns TGraph or TGraph2D
    TObject* Scan(const char* what, const char* select="");

    double FindLimit(const char* what);

    std::vector<xRooHypoPoint*> fPoints;
    std::string fObsName = "obs"; // name of the observed value in points
    std::string fAsimovName = "asimov"; // name of the obs value to use for asymptotics

    const RooAbsCollection* fPOI = nullptr;

    RooArgList* fAxisVars = nullptr; // as points get added to the space, this list acquires clones to identify points

    ClassDefOverride(xRooHypoSpace,1)

};
