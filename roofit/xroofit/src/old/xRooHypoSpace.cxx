
#include "xRooFit/xRooHypoSpace.h"

#include "RooStringVar.h"
#include "TPRegexp.h"
#include "RooRealVar.h"

#include "TH1D.h"
#include "TPad.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "TLegend.h"
#include "TLine.h"
#include "Math/ProbFunc.h"

#include "RooFormulaVar.h"

xRooHypoSpace::xRooHypoSpace() : TNamed(), fPOI(nullptr), fAxisVars(nullptr) { }

xRooHypoSpace::xRooHypoSpace(const char* name, const char* title, const RooAbsCollection& poi) : TNamed(name,title), fPOI(poi.snapshot()) {

}

void xRooHypoPoint::FillNull(double value, double w) {
    fNull.push_back(std::make_pair(value,w));
    return;
}

void xRooHypoPoint::SetObs(double value, const char* name) {
    fObs[ (name) ? name : fSpace->fObsName.c_str() ].first = value;
    if (fAltPoint && fAltPoint!=this) fAltPoint->SetObs(value,name);
}

void xRooHypoPoint::FillAlt(double value, double w) {
    if (!fAltPoint) throw std::runtime_error("No alt hypothesis defined for this point");
    return fAltPoint->FillNull(value,w);
}

xRooHypoPoint::xRooHypoPoint(xRooHypoSpace* space, const RooAbsCollection& coords, const RooAbsCollection& alt_coords) :
    fSpace(space), fCoords(coords.snapshot()), fPOI(fCoords->selectCommon(*fSpace->fPOI)) {

    if (!alt_coords.empty()) {
        // create an alt point too
        fAltPoint = new xRooHypoPoint(fSpace,alt_coords,RooArgSet());

        // guess pll type based on coords
        if(fPOI->size()==1 && fAltPoint->fPOI->size()==1) {
            auto v = dynamic_cast<RooRealVar*>(fPOI->first())->getVal();
            auto va = dynamic_cast<RooRealVar*>(fAltPoint->fPOI->first())->getVal();
            if (v > va) fPllType = xRooFit::Asymptotics::OneSidedPositive;
            else if (v < va) fPllType = xRooFit::Asymptotics::Uncapped;
        }
        // give alt point the same poi values
        delete fAltPoint->fPOI;
        fAltPoint->fPOI = fPOI->snapshot();
        fAltPoint->fPllType = fPllType;
        fAltPoint->fAltPoint = fAltPoint;
    }
}



double xRooHypoPoint::GetObs(const char* name) const {
    std::string sName( (name) ? name : fSpace->fObsName.c_str() );
    auto itr = fObs.find(sName);
    if (itr==fObs.end()) return std::numeric_limits<double>::quiet_NaN();
    return itr->second.first;
}

double xRooHypoPoint::GetObsError(const char* name) const {
    std::string sName( (name) ? name : fSpace->fObsName.c_str() );
    auto itr = fObs.find(sName);
    if (itr==fObs.end()) return std::numeric_limits<double>::quiet_NaN();
    return itr->second.second;
}

double xRooHypoPoint::GetExp(double nSigma,bool asymptotics) const {
    if (!fAltPoint) return std::numeric_limits<double>::quiet_NaN();
    if (fAltPoint != this) return fAltPoint->GetExp(nSigma,asymptotics);
    if (asymptotics) {
        if (fPllType == xRooFit::Asymptotics::Unknown) return  std::numeric_limits<double>::quiet_NaN();
        auto sigma_mu = GetSigmaMu();
        if (std::isnan(sigma_mu.first)) return std::numeric_limits<double>::quiet_NaN();
        auto v =dynamic_cast<RooRealVar*>(fPOI->first());
        if (!v) return std::numeric_limits<double>::quiet_NaN();
        auto vp = dynamic_cast<RooRealVar*>(fCoords->find(*v));
        if (!vp) return  std::numeric_limits<double>::quiet_NaN();
        return xRooFit::Asymptotics::k(fPllType,ROOT::Math::gaussian_cdf(nSigma),v->getVal(),vp->getVal(),sigma_mu.first,v->getMin(),v->getMax());
    }
    std::sort(fAltPoint->fNull.begin(),fAltPoint->fNull.end());
    return fAltPoint->fNull.at( fAltPoint->fNull.size()*ROOT::Math::gaussian_cdf_c(nSigma) ).first;

}

std::pair<double,double> xRooHypoPoint::GetSigmaMu() const {
    auto asi = GetObs(fSpace->fAsimovName.c_str());
    if (std::isnan(asi) || asi<=0 || fPOI->size() != 1 || !fAltPoint) return std::make_pair(std::numeric_limits<double>::quiet_NaN(),0);

    auto v = dynamic_cast<RooRealVar*>(fPOI->first());
    auto va = dynamic_cast<RooRealVar*>(fAltPoint->fCoords->find(*fPOI->first()));

    if (!v || !va) return std::make_pair(std::numeric_limits<double>::quiet_NaN(),0);
    return std::make_pair( abs(v->getVal() - va->getVal())/sqrt(asi), 0.5*GetObsError(fSpace->fAsimovName.c_str())*abs(v->getVal() - va->getVal())/(asi*sqrt(asi)));

}

double xRooHypoPoint::GetPAlt(double value, bool asymptotics) const {
    if (!fAltPoint) return std::numeric_limits<double>::quiet_NaN();
    return fAltPoint->GetPNull(value,asymptotics);
}

double xRooHypoPoint::GetPNull(double value, bool asymptotics) const {
    if (std::isnan(value)) { value = GetObs(); }
    if (std::isnan(value)) return std::numeric_limits<double>::quiet_NaN();
    if (asymptotics) {
        if (fPllType == xRooFit::Asymptotics::Unknown) return  std::numeric_limits<double>::quiet_NaN();
        auto sigma_mu = GetSigmaMu();
        if (std::isnan(sigma_mu.first)) return std::numeric_limits<double>::quiet_NaN();
        auto v =dynamic_cast<RooRealVar*>(fPOI->first());
        if (!v) return std::numeric_limits<double>::quiet_NaN();
        auto vp = dynamic_cast<RooRealVar*>(fCoords->find(*v));
        if (!vp) return  std::numeric_limits<double>::quiet_NaN();
        return xRooFit::Asymptotics::PValue(fPllType,value,v->getVal(),vp->getVal(),sigma_mu.first,v->getMin(),v->getMax());
    }

    double d=0,n=0;
    for(auto& p : fNull) {
        if (!std::isnan(p.second)) {
            d += p.second;
            if (p.first >= value) n += p.second;
        }
    }
    return n/d;
}

const char* xRooHypoPoint::GetTitle() const {
    if (!fSpace) return TNamed::GetTitle();
    TString s;
    for(auto f : fSpace->GetAxisVars()) {
        if(auto v = dynamic_cast<RooRealVar*>(fCoords->find(*f)); v) {
            if (s!="") s += ", ";
            s += TString::Format("%s = %g",f->GetTitle(),v->getVal());
        }
    }
    const_cast<xRooHypoPoint*>(this)->SetTitle(s);
    return TNamed::GetTitle();
}

const RooArgList& xRooHypoSpace::GetAxisVars() const {
    return *fAxisVars;
}

void xRooHypoPoint::Draw(Option_t* opt) {

    TString sOpt(opt);
    sOpt.ToLower();
    bool hasSame = sOpt.Contains("same"); sOpt.ReplaceAll("same","");
    bool isAlt = sOpt.Contains("isalt"); sOpt.ReplaceAll("isalt","");
    isAlt = (fAltPoint==this);

    TVirtualPad *pad = gPad;

    TH1* hAxis = nullptr;

    auto clearPad = []() {
        gPad->Clear();
        if (gPad->GetNumber()==0) {
            gPad->SetBottomMargin(gStyle->GetPadBottomMargin());
            gPad->SetTopMargin(gStyle->GetPadTopMargin());
            gPad->SetLeftMargin(gStyle->GetPadLeftMargin());
            gPad->SetRightMargin(gStyle->GetPadRightMargin());
        }
    };

    if (!hasSame || !pad) {
        if (!pad) {
            TCanvas::MakeDefCanvas();
            pad = gPad;
        }
        clearPad();
    } else {
        // get the histogram representing the axes
        hAxis = dynamic_cast<TH1*>(pad->GetPrimitive("axis"));
        if (!hAxis) {
            for(auto o : *pad->GetListOfPrimitives()) {
                if (hAxis = dynamic_cast<TH1*>(o); hAxis) break;
            }
        }
    }


    // get min and max values
    double _min = std::numeric_limits<double>::quiet_NaN();
    double _max = -std::numeric_limits<double>::quiet_NaN();

    for(auto& p : fNull) {
        if (p.second==0) continue;
        _min = std::min(p.first,_min);
        _max = std::max(p.first,_max);
    }

    // if has alt point include that range too
    if (fAltPoint) {
        for(auto& p : fAltPoint->fNull) {
            if (p.second==0) continue;
            _min = std::min(p.first,_min);
            _max = std::max(p.first,_max);
        }
    }

    auto obs = GetObs();
    if (!std::isnan(obs)) {
        _min = std::min(obs-abs(obs)*0.1,_min);
        _max = std::max(obs+abs(obs)*0.1,_max);
    }

    auto asi = GetObs(fSpace->fAsimovName.c_str());
    if (!std::isnan(asi) && asi>0 && fPOI->size()==1 && fPllType != xRooFit::Asymptotics::Unknown && fAltPoint) {
        // can calculate asymptotic distributions,
        _min = std::min(asi-abs(asi),_min);
        _max = std::max(asi+abs(asi),_max);
    }

    auto h = new TH1D((isAlt) ? "alt" : "null","",100,_min,_max + (_max-_min)*0.01);
    h->SetDirectory(0);
    size_t nBadOrZero=0;
    for(auto& p : fNull) {
        double w = std::isnan(p.second) ? 0 : p.second;
        if (w==0) nBadOrZero++;
        h->Fill(p.first,w);
    }
    if(h->GetEntries()>0) h->Scale( 1./h->Integral(0,h->GetNbinsX()+1));
    TString title;
    if (fAltPoint) {
        // add POI values to identify hypos
        for(auto p : *fPOI) {
            if (auto v = dynamic_cast<RooRealVar*>(p)) {
                if (auto v2 = dynamic_cast<RooRealVar*>(fAltPoint->fCoords->find(*v)); v2 && v2->getVal()!=v->getVal()) {
                    // found point that differs in poi and altpoint value, so print my coords value for this
                    title += TString::Format("%s' = %g, ",v->GetTitle(),dynamic_cast<RooRealVar*>(fCoords->find(*v))->getVal());
                }
            }
        }
    }
    title += TString::Format("N_{toys}=%lu",fNull.size());
    if (nBadOrZero > 0) title += TString::Format(" (N_{bad/0}=%lu)",nBadOrZero);

    if(fPllType == xRooFit::Asymptotics::OneSidedPositive) {
        auto v = dynamic_cast<RooRealVar*>(fPOI->first());
        if (v && v->getMin()==0) title += TString::Format(";#tilde{q}_{%s=%g}",v->GetTitle(),v->getVal());
        else if(v) title += TString::Format(";q_{%s=%g}",v->GetTitle(),v->getVal());
        else title += ";q";
    } else if(fPllType == xRooFit::Asymptotics::TwoSided) {
        auto v = dynamic_cast<RooRealVar*>(fPOI->first());
        if (v && v->getMin()==0) title += TString::Format(";#tilde{t}_{%s=%g}",v->GetTitle(),v->getVal());
        else if(v) title += TString::Format(";t_{%s=%g}",v->GetTitle(),v->getVal());
        else title += ";t";
    } else if(fPllType == xRooFit::Asymptotics::OneSidedNegative) {
        auto v = dynamic_cast<RooRealVar*>(fPOI->first());
        if (v && v->getMin()==0) title += TString::Format(";#tilde{r}_{%s=%g}",v->GetTitle(),v->getVal());
        else if(v) title += TString::Format(";r_{%s=%g}",v->GetTitle(),v->getVal());
        else title += ";r";
    } else if(fPllType == xRooFit::Asymptotics::Uncapped) {
        auto v = dynamic_cast<RooRealVar*>(fPOI->first());
        if (v && v->getMin()==0) title += TString::Format(";#tilde{s}_{%s=%g}",v->GetTitle(),v->getVal());
        else if(v) title += TString::Format(";s_{%s=%g}",v->GetTitle(),v->getVal());
        else title += ";s";
    } else {
        title += ";Test Statistic";
    }
    title += TString::Format(";Probability Density");
    h->SetTitle(title);
    h->SetLineColor(isAlt ? kRed : kBlue); h->SetLineWidth(2);
    h->SetMarkerSize(0);
    h->SetBit(kCanDelete);



    TLegend* l = nullptr;
    TString htitle = h->GetTitle();
    if (!hasSame) {
        gPad->SetLogy();
        h->SetMinimum(1e-3);
        h->SetTitle(GetTitle());
        h->Draw("histe");//h->Draw("axis"); cant use axis option if want title drawn
        hAxis=h;
        l = new TLegend(0.4,0.7,1.-gPad->GetRightMargin(),1.-gPad->GetTopMargin());
        l->SetName("legend");
        l->SetFillStyle(0);l->SetBorderSize(0);
        l->SetBit(kCanDelete);
        l->Draw();
    } else {
        for(auto o : *gPad->GetListOfPrimitives()) {
            l = dynamic_cast<TLegend*>(o);
            if (l) break;
        }
    }

    if (hasSame) {
        if(h->GetEntries()>0) h->Draw( "histesame" );
        else h->Draw("axissame");// for unknown reason if second histogram empty it still draws with two weird bars???
    }

    if(l) { l->AddEntry(h,htitle,"l"); }



    if (!std::isnan(asi) && asi>0 && fPOI->size()==1 && fPllType != xRooFit::Asymptotics::Unknown && fAltPoint) {
        auto _my_mu = dynamic_cast<RooRealVar*>(fCoords->find(*fPOI->first()))->getVal();
        auto _test_mu = dynamic_cast<RooRealVar*>(fPOI->first());
        double sigma_mu = _test_mu->getVal() - dynamic_cast<RooRealVar*>(fAltPoint->fCoords->find(*fPOI->first()))->getVal();
        sigma_mu = abs(sigma_mu)/sqrt(asi);

        auto hh = (TH1*)h->Clone("null_asymptotic");
        hh->SetLineStyle(2);
        hh->Reset();
        for(int i=1;i<=hh->GetNbinsX();i++) {
            hh->SetBinContent(i, xRooFit::Asymptotics::PValue(fPllType,hh->GetBinLowEdge(i),_test_mu->getVal(),_my_mu,sigma_mu,_test_mu->getMin(),_test_mu->getMax()) -
                    xRooFit::Asymptotics::PValue(fPllType,hh->GetBinLowEdge(i+1),_test_mu->getVal(),_my_mu,sigma_mu,_test_mu->getMin(),_test_mu->getMax()));
        }
        hh->Draw("lsame");

    }

    if (fAltPoint && fAltPoint!=this) fAltPoint->Draw("same");

    // draw observed points
    if (fAltPoint != this) {
        TLine ll; ll.SetLineStyle(2);
        for(auto p : fObs) {
            auto tl = ll.DrawLine(p.second.first,hAxis->GetMinimum(),p.second.first,0.1);
            auto label = TString::Format("%s = %.4f",p.first.c_str(),p.second.first);
            if (p.second.second) label += TString::Format(" +/- %.4f",p.second.second);
            double pNull = GetPNull(p.second.first);
            double pAlt = GetPAlt(p.second.first);

            double pNullA = GetPNull(p.second.first,true);
            double pAltA = GetPAlt(p.second.first,true);

            l->AddEntry(tl,label,"l");
            label="";
            if (!std::isnan(pNull) || !std::isnan(pAlt)) {
                double pCLs = GetPCLs(p.second.first);
                label += " p_{toy}=(";
                label += (std::isnan(pNull)) ? "-" : TString::Format("%g",pNull);
                label += (std::isnan(pAlt)) ? ",-" : TString::Format(",%g",pAlt);
                label += (std::isnan(pCLs)) ? ",-)" : TString::Format(",%g)",pCLs);
            }
            if (label.Length()>0) l->AddEntry("",label,"");
            label="";
            if (!std::isnan(pNullA) || !std::isnan(pAltA)) {
                double pCLs = GetPCLs(p.second.first,true);
                label += " p_{asymp}=(";
                label += (std::isnan(pNullA)) ? "-" : TString::Format("%g",pNullA);
                label += (std::isnan(pAltA)) ? ",-" : TString::Format(",%g",pAltA);
                label += (std::isnan(pCLs)) ? ",-)" : TString::Format(",%g)",pCLs);
            }
            if (label.Length()>0) l->AddEntry("",label,"");

        }
    }





}

RooArgSet stringToSet(const std::string& coords, const RooAbsCollection* ref) {
    TStringToken pattern(coords,",");
    RooArgSet c;
    while(pattern.NextToken()) {
        TString s= pattern.Data();
        auto i = s.Index("=");
        if (i==-1) throw std::runtime_error("coords must contain value");
        TString cName = s(0,i);
        bool isPoi=false;
        if (cName.EndsWith("*")) {
            cName = cName(0,cName.Length()-1);
            isPoi=true;
        }
        TString cVal = s(i+1,s.Length());
        if (cVal.IsFloat()) {
            if (ref && ref->find(cName)) {
                c.addClone(*ref->find(cName));
                c.setRealValue(cName,cVal.Atof());
            } else {
                c.addClone(RooRealVar(cName, cName, cVal.Atof()));
            }
        } else {
            c.addClone(RooStringVar(cName,cName,cVal));
        }
        c[cName].setAttribute("poi",true);
    }
    return c;
}

xRooHypoPoint* xRooHypoSpace::GetPoint(const std::string& coords, const std::string& alt_coords) {

    auto out = new xRooHypoPoint(this,stringToSet(coords,fPOI),stringToSet(alt_coords,fPOI));

    if (!fAxisVars) fAxisVars = new RooArgList;

    for(auto p : fPoints) {
        for(auto a : *out->fCoords) {
            if(fAxisVars->find(*a)) continue;
            auto v = dynamic_cast<RooRealVar*>(a);
            if (!v) continue;
            auto v2 = dynamic_cast<RooRealVar*>(p->fCoords->find(*a)); // allow not to exist??
            if (!v2 || v->getVal()!=v2->getVal()) {
                fAxisVars->addClone(*v);
            }
        }
    }
    fPoints.push_back(out);
    return out;

}

void xRooHypoSpace::Draw(Option_t* opt, Option_t* select) {
    TString sOpt(opt);

    bool hasSame = sOpt.Contains("same");
    TVirtualPad* pad = gPad;
    if (!pad && hasSame) return;

    if (!pad) {
        TCanvas::MakeDefCanvas();
        pad = gPad;
    }
    if (!hasSame) pad->Clear();

    if(sOpt.Contains("dist")) {
        // draw distributions for selected points
        std::vector<xRooHypoPoint*> selPoints;
        for(auto p : fPoints) {
            RooArgList l; l.add(*p->fCoords);
            if(!select || strlen(select)==0 || RooFormulaVar("select",select,l,false).getVal()) {
                selPoints.push_back(p);
            }
        }
        if (selPoints.empty()) {
            Warning("Draw","No points selected");
            return;
        }
        if (selPoints.size()>1) ((TPad*)(gPad))->DivideSquare(selPoints.size());
        int i=0;
        for(auto p : selPoints) {
            i++;
            if(selPoints.size()>1) pad->cd(i);
            p->Draw();
        }
        pad->cd();
        pad->Modified();
        return;
    }

    // default to draw pvalues
    // get all the graphs first so can decide if can draw toys and asymptotics or just one
    std::vector<std::string> graphStr = {"","exp-2","exp-1","exp0","exp1","exp2"};
    std::vector<TGraph*> toy_graphs;
    std::vector<TGraph*> asymp_graphs;
    for(auto s : graphStr) {
        toy_graphs.push_back(dynamic_cast<TGraph*>(Scan((s+" toys").c_str())));
        asymp_graphs.push_back(dynamic_cast<TGraph*>(Scan(s.c_str())));
    }

    auto& graphs = toy_graphs;
    auto drawBand = [&](int i1, int i2) {
        if (graphs[i1] && graphs[i2] && graphs[i1]->GetN() && graphs[i2]->GetN()) {
            // can draw band, put graphs together as a band and shade
            graphs[i1]->Sort(TGraph::CompareX, false);
            TList l;
            l.Add(graphs[i1]);
            graphs[i2]->Merge(&l);
            l.Clear();
            delete graphs[i1];
            graphs[i2]->SetBit(kCanDelete);
            graphs[i2]->SetFillColor((i2==5) ? kYellow : kGreen);
            TString opt = "F";
            if (gPad->GetListOfPrimitives()->IsEmpty()) opt += "A";
            graphs[i2]->SetName((i2==5) ? "exp2" : "exp1");
            graphs[i2]->GetHistogram()->SetYTitle("p-value");
            graphs[i2]->Draw(opt);
        }
    };

    drawBand(1,5); drawBand(2,4);
    TString lOpt = "L";
    if (gPad->GetListOfPrimitives()->IsEmpty()) lOpt += "A";
    if(graphs[3] && graphs[3]->GetN()) {graphs[3]->SetBit(kCanDelete);graphs[3]->Draw(lOpt);}
    else if(graphs[3]) { delete graphs[3]; }
    lOpt = "L";
    if (gPad->GetListOfPrimitives()->IsEmpty()) lOpt += "A";
    if(graphs[0] && graphs[0]->GetN()) {graphs[0]->SetBit(kCanDelete);graphs[0]->Draw(lOpt);}
    else if(graphs[0]) { delete graphs[0]; }

    gPad->RedrawAxis();


}

void xRooHypoSpace::Draw(Option_t* opt) {
    Draw(opt,"");
}

#include "Math/BrentRootFinder.h"
#include "Math/WrappedFunction.h"
#include "Math/Functor.h"
#include "TGraph2D.h"
#include "TGraphAsymmErrors.h"

double xRooHypoSpace::GetGraphX(const TGraph& graph, double y0, Option_t* opt) {
    if (graph.GetN()<2) return std::numeric_limits<double>::quiet_NaN();
    // todo: allow to control if looking for upcrossing (lower limit) or downcrossing (upper limit)
    TString sOpt(opt);
    sOpt.ToLower();
    bool doSpline = (sOpt.Contains("s"));
    auto func = [&](double x) {
        return (doSpline) ? graph.Eval(x, nullptr, "S") - y0 : graph.Eval(x) - y0;
    };
    ROOT::Math::Functor1D f1d(func);

    ROOT::Math::BrentRootFinder brf;
    brf.SetFunction(f1d,graph.GetPointX(0),graph.GetPointX(graph.GetN()-1));
    brf.SetNpx(TMath::Max(graph.GetN()*2,100) );
    bool ret = brf.Solve(100, 1.E-16, 1.E-6);
    if (!ret) return std::numeric_limits<double>::quiet_NaN();
    return brf.Root();
}

double xRooHypoSpace::FindLimit(const char* what) {
    std::unique_ptr<TGraph> gr(dynamic_cast<TGraph*>(Scan(what)));
    if (!gr) return std::numeric_limits<double>::quiet_NaN();
    TString sWhat(what);
    return GetGraphX(*gr,0.05, sWhat.Contains("spline") ? "S" : "");
}

TObject* xRooHypoSpace::Scan(const char* what, const char* select) {

    TString sWhat(what);

    TStringToken pattern(sWhat,":");
    std::vector<TString> parts;
    while(pattern.NextToken()) {
        parts.push_back( pattern.Data() );
    }

    if (parts.empty()) parts.push_back("");

    if (parts.size()==1) {
        parts.push_back( (fAxisVars && !fAxisVars->empty() ) ? fAxisVars->at(0)->GetName() : "0" );
    }

    sWhat = parts.at(0);
    sWhat += " ";
    sWhat.ToLower();

    double nSigma = std::numeric_limits<double>::quiet_NaN();
    if (auto i = sWhat.Index("exp"); i != -1) {
        nSigma = TString(sWhat(i+3, sWhat.Index(" ",i)-(i+3))).Atof();
    }

    // scan through points building graph of p-values
    TNamed* out;
    if(parts.size()>2) {
        out = new TGraph2D;
    } else {
        out = new TGraph;
    }
    TString xTitle = parts.at(1);
    if (fAxisVars) {
        for(auto a : *fAxisVars) { xTitle.ReplaceAll(a->GetName(),a->GetTitle()); }
    }
    TString title = sWhat;
    if (select && strlen(select)>0) {title += "[["; title += select; title += "]]";}
    out->SetTitle(TString::Format("%s;%s",title.Data(),xTitle.Data()));


    auto valGetter = [&](xRooHypoPoint* p, double ts, bool doAsymp) {
        if (sWhat.Contains("ts ")) {
            return ts;
        } else if (sWhat.Contains("null ")) {
            return p->GetPNull(ts,doAsymp);
        } else if (sWhat.Contains("alt ")) {
            return p->GetPAlt(ts,doAsymp);
        } else {
            return p->GetPCLs(ts,doAsymp);
        }
    };

    for(auto p : fPoints) {
        RooArgList l; l.add(*p->fCoords);
        if (select && strlen(select)>0) {
            if (RooFormulaVar("select",select,l,false).getVal()==0) continue;
        }

        double x = RooFormulaVar("x",parts.at(1),l,false).getVal();
        double y = (parts.size()>2) ? RooFormulaVar("y",parts.at(2),l,false).getVal() : 0;


        bool doAsymp = !sWhat.Contains("toy");

        double ts = std::isnan(nSigma) ? p->GetObs() : p->GetExp(nSigma,doAsymp);

        double v = valGetter(p,ts,doAsymp);

        if (!std::isnan(v) && !std::isinf(v)) {
            if(parts.size()>2) dynamic_cast<TGraph2D*>(out)->SetPoint(dynamic_cast<TGraph2D*>(out)->GetN(),x,y,v);
            else dynamic_cast<TGraph*>(out)->SetPoint(dynamic_cast<TGraph*>(out)->GetN(),x,v);
        }

    }

    if(auto g = dynamic_cast<TGraph*>(out); g) {
        g->Sort();
        if(!std::isnan(nSigma)) g->SetLineStyle(2);
    }

    return out;
}