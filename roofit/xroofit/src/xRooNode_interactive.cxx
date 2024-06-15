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

// Interactive methods of xRooNode

#include "xRooFit/xRooNode.h"

#include "RooArgList.h"
#include "RooArgSet.h"

#include "TCanvas.h"
#include "TVirtualX.h"
#include "TH1F.h"
#include "TStyle.h"
#include "TGraphAsymmErrors.h"
#include "TMultiGraph.h"
#include "TSystem.h"

BEGIN_XROOFIT_NAMESPACE

void xRooNode::Interactive_Pull()
{
   static bool doRestore = false;
   auto select = dynamic_cast<TGraph *>(gPad->GetSelected());
   // if (!select) return;
   int event = gPad->GetEvent();
   if (event == 1) {
      doRestore = false;
      // if this is one of the 'black' points, color it temporarily
   } else if (event == 11 || doRestore) {
      if (!select || doRestore) {
         // now need to assemble a snapshot corresponding to the current variation
         auto _h = static_cast<TGraph *>(gPad->GetPrimitive("nominal"))->GetHistogram();
         for (int i = 1; i <= _h->GetNbinsX(); i++) {
            std::string parName = _h->GetXaxis()->GetBinLabel(i);
            // ensure point is back .. sometimes editing mode allows point to drift
            auto _gr = static_cast<TGraph *>(static_cast<TMultiGraph *>(gPad->GetPrimitive("editables"))
                                                ->GetListOfGraphs()
                                                ->FindObject(parName.c_str()));
            _gr->SetPoint(0, i - 1, static_cast<TGraph *>(gPad->GetPrimitive("nominal"))->GetPointY(i - 1));
         }
         gPad->GetMother()->GetMother()->cd();
         doRestore = false;
         return;
      }
      doRestore = true;
      // mouse up event, if this was an original point it needs snapping back
      // double _y = select->GetPointY(0);
      TString _name = select->GetName();
      TString _varyName = "";
      if (_name.Contains(";")) {
         _varyName = TString(_name(_name.Index(";") + 1, _name.Length()));
         _name = _name(0, _name.Index(";"));
      }
      auto _h = static_cast<TH1 *>(gPad->GetPrimitive("nominal")->FindObject("scales"));
      if (!_h)
         return;
      for (int i = 1; i <= _h->GetNbinsX(); i++) {
         if (_name == _h->GetXaxis()->GetBinLabel(i)) {
            auto _gr = static_cast<TGraph *>(gPad->GetPrimitive("nominal"));
            if (_varyName == "") {
               int vNum = 1;
               TGraphAsymmErrors *newPoint = dynamic_cast<TGraphAsymmErrors *>(
                  gPad->GetPrimitive(TString::Format("%s;variation %d", select->GetName(), vNum)));
               while (newPoint && newPoint->GetN() > 0) {
                  vNum++;
                  newPoint = static_cast<TGraphAsymmErrors *>(
                     gPad->GetPrimitive(TString::Format("%s;variation %d", select->GetName(), vNum)));
               }
               _varyName = TString::Format("variation %d", vNum);
               if (!newPoint) {
                  newPoint = static_cast<TGraphAsymmErrors *>(
                     select->Clone(TString::Format("%s;%s", select->GetName(), _varyName.Data())));
               }
               newPoint->SetPointX(0, _gr->GetPointX(i - 1));
               newPoint->SetMarkerColor(860 + (vNum - 1) * 20);
               newPoint->SetLineColor(newPoint->GetMarkerColor());
               newPoint->SetPointEYlow(0, 0);
               newPoint->SetPointEYhigh(0, 0); // remove errors because currently meaningless!
               newPoint->Draw("z0p");
               select->SetPoint(0, _gr->GetPointX(i - 1), _gr->GetPointY(i - 1));
               select = newPoint;
            } else {
               select->SetPointX(0, _gr->GetPointX(i - 1));
            }
            static_cast<TGraph *>(
               static_cast<TMultiGraph *>(gPad->GetPrimitive("editables"))->GetListOfGraphs()->FindObject(_name))
               ->SetPoint(0, i - 1, _gr->GetPointY(i - 1));
            break;
         }
      }

      // then do an overlay update
      auto _node = dynamic_cast<xRooNode *>(gPad->GetPrimitive("node"));
      if (!_node)
         return;
      RooArgSet _pars(_node->pars().argList());
      std::unique_ptr<RooArgSet> snap(_pars.snapshot());

      // now need to assemble a snapshot corresponding to the current variation
      for (int i = 1; i <= _h->GetNbinsX(); i++) {
         std::string parName = _h->GetXaxis()->GetBinLabel(i);
         // ensure point is back .. sometimes editing mode allows point to drift
         // dynamic_cast<TGraph*>(gPad->GetPrimitive(parName.c_str()))->SetPoint(0,i-1,dynamic_cast<TGraph*>(gPad->GetPrimitive("nominal"))->GetPointY(i-1));
         if (auto g =
                dynamic_cast<TGraph *>(gPad->GetPrimitive(TString::Format("%s;%s", parName.c_str(), _varyName.Data())));
             g && g->GetN() > 0) {
            double _val =
               g->GetPointY(0) * _h->GetBinError(i) + _h->GetBinContent(i); // error is scale, content is offset
            _pars.setRealValue(parName.c_str(), _val);
            g->SetTitle(TString::Format("%s=%g", parName.c_str(), _val));
         } else {
            _pars.setRealValue(parName.c_str(), static_cast<TGraph *>(gPad->GetPrimitive("nominal"))->GetPointY(i - 1) *
                                                      _h->GetBinError(i) +
                                                   _h->GetBinContent(i));
         }
      }
      TAttLine bak = *gStyle;
      TAttFill bak2 = *gStyle;
      gStyle->SetLineStyle(5);
      gStyle->SetLineWidth(2);
      gStyle->SetLineColor(select->GetMarkerColor());
      auto _tmpPad = gPad;
      gPad->GetMother()->GetMother()->cd(1);
      _node->Draw(TString::Format("same overlay%s", _varyName.Data()));
      // TODO: find the drawn variation and set its title equal to a _pars value string
      static_cast<TAttLine &>(*gStyle) = bak;
      static_cast<TAttFill &>(*gStyle) = bak2;
      _pars = *snap;
      _tmpPad->GetCanvas()->cd();
      gPad->GetCanvas()->Paint();
      gPad->GetCanvas()->Update();
   }
}

void xRooNode::Interactive_PLLPlot()
{

   // TObject *select = gPad->GetSelected();
   // if(!select) return;
   // if (!select->InheritsFrom(TGraph::Class())) {gPad->SetUniqueID(0); return;}
   // gPad->GetCanvas()->FeedbackMode(true);

   auto _pull_pad = gPad->GetPad(1);
   auto _hidden_pad = gPad->GetPad(2);

   if (!_pull_pad || strcmp(_pull_pad->GetName(), "pulls") != 0)
      return;
   if (!_hidden_pad)
      return;

   // erase old position and draw a line at current position
   // int pxold = gPad->GetUniqueID();
   int px = gPad->GetEventX();
   // int py = gPad->GetEventY();
   // int pymin = gPad->YtoAbsPixel(gPad->GetUymin());
   // int pymax = gPad->YtoAbsPixel(gPad->GetUymax());
   // if(pxold) gVirtualX->DrawLine(pxold,pymin,pxold,pymax);
   // gVirtualX->DrawLine(px,pymin,px,pymax);
   gPad->SetUniqueID(px);
   float upx = gPad->AbsPixeltoX(px);
   float x = gPad->PadtoX(upx);

   // find which graph in the hidden pad best reflects current x value
   TObject *foundGraph = nullptr;
   for (auto g : *_hidden_pad->GetListOfPrimitives()) {
      double midpoint = TString(g->GetName()).Atof();
      if (!foundGraph) {
         foundGraph = g;
      } else {
         midpoint = (midpoint + TString(foundGraph->GetName()).Atof()) / 2.;
      }
      if (midpoint >= x)
         break;
      foundGraph = g;
   }
   if (foundGraph) {
      auto _x = TString(foundGraph->GetName()).Atof();
      if (auto line = dynamic_cast<TGraph *>(gPad->GetListOfPrimitives()->FindObject("markerLine")); line) {
         line->SetPointX(0, _x);
         line->SetPointX(1, _x);
      } else {
         line = new TGraph;
         line->SetLineStyle(2);
         line->SetName("markerLine");
         line->SetBit(kCanDelete);
         line->SetPoint(0, _x, -100);
         line->SetPoint(1, _x, 100);
         line->Draw("Lsame");
      }
      gPad->Modified();
      gPad->Update();
      for (auto o : *_pull_pad->GetListOfPrimitives()) {
         if (!o->InheritsFrom("TGraph"))
            continue;
         if (_hidden_pad->GetListOfPrimitives()->FindObject(o) || TString(o->GetName()).EndsWith("_pull")) {
            // todo: might need to delete the "_pull" plot if removing for first time
            _pull_pad->GetListOfPrimitives()->Remove(o);
            break;
         }
      }
      auto tmp = gPad;
      _pull_pad->cd();
      foundGraph->Draw("pz0 same");
      tmp->cd();
      _pull_pad->Modified();
      _pull_pad->Update();
   }
}

void xRooNode::InteractiveObject::Interactive_PLLPlot(TVirtualPad *pad, TObject *obj, Int_t x, Int_t /*y*/)
{

   if (auto g = dynamic_cast<TGraph *>(obj); g && pad && pad->GetMother() && pad->GetNumber() == 1) {
      auto frPad = pad->GetMother()->GetPad(2);
      if (frPad) {
         if (!g->IsHighlight()) {
            x = -1;
         } else if (x >= 0) {
            x += 1;
         }
         // x is the point index
         TVirtualPad *_pad = frPad->GetPad(x);
         auto selPad = dynamic_cast<TVirtualPad *>(frPad->GetPrimitive("selected"));
         if (_pad && selPad) {
            auto prim = selPad->GetListOfPrimitives();
            prim->Remove(prim->At(0));
            prim->Add(_pad);

            //                for (auto p: *pad->GetListOfPrimitives()) {
            //                    if (auto _p = dynamic_cast<TPad *>(p)) {
            //                        _p->Modified();
            //                    }
            //                }
            //                pad->Modified();
            //                pad->Update();
            selPad->Modified();
            selPad->Update();
            gSystem->ProcessEvents();
         }
      }
   }
}

END_XROOFIT_NAMESPACE
