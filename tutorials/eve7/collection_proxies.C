/// \file
/// \ingroup tutorial_eve7
///  This example display collection of ??? in web browser
///
/// \macro_code
///


#include "ROOT/REveDataTable.hxx"
#include "ROOT/REveDataSimpleProxyBuilderTemplate.hxx"
#include "ROOT/REveManager.hxx"
#include "ROOT/REveScalableStraightLineSet.hxx"
#include "ROOT/REveViewContext.hxx"
#include <ROOT/REveGeoShape.hxx>
#include <ROOT/REveJetCone.hxx>
#include <ROOT/REvePointSet.hxx>
#include <ROOT/REveProjectionBases.hxx>
#include <ROOT/REveProjectionManager.hxx>
#include <ROOT/REveScene.hxx>
#include <ROOT/REveTableProxyBuilder.hxx>
#include <ROOT/REveTableInfo.hxx>
#include <ROOT/REveTrack.hxx>
#include <ROOT/REveTrackPropagator.hxx>
#include <ROOT/REveViewer.hxx>
#include <ROOT/REveViewContext.hxx>
#include <ROOT/REveBoxSet.hxx>
#include <ROOT/REveSelection.hxx>
#include <ROOT/REveCalo.hxx>

#include "TGeoTube.h"
#include "TROOT.h"
#include "TList.h"
#include "TParticle.h"
#include "TRandom.h"
#include "TApplication.h"
#include "TFile.h"
#include "TH2F.h"
#include <iostream>


const Double_t kR_min = 300;
const Double_t kR_max = 299;
const Double_t kZ_d   = 300;

ROOT::Experimental::REveManager *eveMng = nullptr;
ROOT::Experimental::REveProjectionManager* g_projMng = nullptr;
using namespace ROOT::Experimental;

//==============================================================================
//============== EMULATE FRAMEWORK CLASSES =====================================
//==============================================================================


// a demo class, can be provided from experiment framework
class Jet : public TParticle
{
public:
   float fEtaSize{0};
   float fPhiSize{0};

   float GetEtaSize() const { return fEtaSize; }
   float GetPhiSize() const { return fPhiSize; }
   void  SetEtaSize(float iEtaSize) { fEtaSize = iEtaSize; }
   void  SetPhiSize(float iPhiSize) { fPhiSize = iPhiSize; }

  Jet(Int_t pdg, Int_t status, Int_t mother1, Int_t mother2, Int_t daughter1, Int_t daughter2,
        Double_t px, Double_t py, Double_t pz, Double_t etot) :
    TParticle(pdg, status, mother1, mother2, daughter1, daughter2, px, py, pz, etot,  0, 0, 0, 0)
  {}

   ClassDef(Jet, 1);
};

class RecHit : public TObject
{
public:
   float fX{0};
   float fY{0};
   float fZ{0};
   float fPt{0};

   RecHit(float pt, float x, float y, float z): fPt(pt), fX(x), fY(y), fZ(z) {}
   ClassDef(RecHit, 1);
};


class Event
{
public:
   int eventId{0};
   int N_tracks{0};
   int N_jets{0};   
   std::vector<TList*> fListData;
   
   REveCaloDataHist* fCaloData{nullptr};
   
   Event()
   {
      TFile::SetCacheFileDir(".");
       const char* histFile =
      "http://amraktad.web.cern.ch/amraktad/cms_calo_hist.root";
      auto hf = TFile::Open(histFile, "CACHEREAD");
      auto ecalHist = (TH2F*)hf->Get("ecalLego");
      auto hcalHist = (TH2F*)hf->Get("hcalLego");
      fCaloData = new REveCaloDataHist();
      fCaloData->AddHistogram(ecalHist);
      fCaloData->RefSliceInfo(0).Setup("ECAL", 0.f, kBlue);
      fCaloData->AddHistogram(hcalHist);
      fCaloData->RefSliceInfo(1).Setup("HCAL", 0.1, kRed);
      eveMng->GetEventScene()->AddElement(fCaloData);
   }

   void MakeJets(int N)
   {
      TRandom &r = *gRandom;
      r.SetSeed(0);
      TList* list = new TList();
      list->SetName("Jets");
      for (int i = 1; i <= N; ++i)
      {
         double pt  = r.Uniform(0.5, 10);
         double eta = r.Uniform(-2.55, 2.55);
         double phi = r.Uniform(-TMath::Pi(), TMath::Pi());

         double px = pt * std::cos(phi);
         double py = pt * std::sin(phi);
         double pz = pt * (1. / (std::tan(2*std::atan(std::exp(-eta)))));

         auto jet = new Jet(0, 0, 0, 0, 0, 0, px, py, pz, std::sqrt(px*px + py*py + pz*pz + 80*80));
         jet->SetEtaSize(r.Uniform(0.02, 0.2));
         jet->SetPhiSize(r.Uniform(0.01, 0.3));
         list->Add(jet);
      }
      fListData.push_back(list);
   }

   void MakeParticles(int N)
   {
      TRandom &r = *gRandom;
      r.SetSeed(0);
      TList* list = new TList();
      list->SetName("Tracks");
      for (int i = 1; i <= N; ++i)
      {
         double pt  = r.Uniform(0.5, 10);
         double eta = r.Uniform(-2.55, 2.55);
         double phi = r.Uniform(0, TMath::TwoPi());

         double px = pt * std::cos(phi);
         double py = pt * std::sin(phi);
         double pz = pt * (1. / (std::tan(2*std::atan(std::exp(-eta)))));

         // printf("Event::MakeParticles %2d: pt=%.2f, eta=%.2f, phi=%.2f\n", i, pt, eta, phi);
         auto particle = new TParticle(0, 0, 0, 0, 0, 0,
                                       px, py, pz, std::sqrt(px*px + py*py + pz*pz + 80*80),
                                       0, 0, 0, 0 );

         int pdg = 11 * (r.Integer(2) > 0 ? 1 : -1);
         particle->SetPdgCode(pdg);

         list->Add(particle);
      }
      fListData.push_back(list);
   }

   void MakeRecHits(int N)
   {
      TRandom &r = *gRandom;
      r.SetSeed(0);
      TList* list = new TList();
      list->SetName("RecHits");

      for (int i = 1; i <= N; ++i)
      {
         float pt = r.Uniform(0.5, 10);
         float x =  r.Uniform(-200, 200);
         float y =  r.Uniform(-200, 200);
         float z =  r.Uniform(-500, 500);
         auto rechit = new RecHit(pt, x, y, z);
         list->Add(rechit);
      }
      fListData.push_back(list);
   }

   void Clear()
   {
      for (auto &l : fListData)
         delete l;
      fListData.clear();
   }

   void Create()
   {
      Clear();
      MakeJets(4);
      MakeParticles(100);
      MakeRecHits(20);

      // refill calo data from jet list
      TList* jlist = fListData[0];
      auto  ecalHist = fCaloData->GetHist(0);      
      auto  hcalHist = fCaloData->GetHist(1);
      ecalHist->Reset();
      hcalHist->Reset();
      for (int i = 0; i <= jlist->GetLast(); ++i) {
         const Jet* j = (Jet*)jlist->At(i);
         float offX = j->Eta();
         float offY = j->Phi() > TMath::Pi() ? j->Phi() -  TMath::TwoPi() :  j->Phi();
         for (int k=0; k<100; ++k) {
            double x, y, v;
            x = gRandom->Uniform(-j->GetEtaSize(), j->GetEtaSize());
            y = gRandom->Uniform(-j->GetPhiSize(),j->GetPhiSize());
            v = j->Pt();
            ecalHist->Fill(offX + x, offY + y, v + gRandom->Uniform(2,3));
            hcalHist->Fill(offX + x, offY + y, v + gRandom->Uniform(1,2));
         }
      }
      fCaloData->DataChanged();
      eventId++;
   }
};


//==============================================================================
//== PROXY BUILDERS ============================================================
//==============================================================================

class JetProxyBuilder: public REveDataSimpleProxyBuilderTemplate<Jet>
{
   bool HaveSingleProduct() const override { return false; }

   using REveDataSimpleProxyBuilderTemplate<Jet>::BuildViewType;

   void BuildViewType(const Jet& dj, int idx, REveElement* iItemHolder,
                      std::string viewType, const REveViewContext* context) override
   {
      auto jet = new REveJetCone();
      jet->SetCylinder(context->GetMaxR(), context->GetMaxZ());
      jet->AddEllipticCone(dj.Eta(), dj.Phi(), dj.GetEtaSize(), dj.GetPhiSize());
      SetupAddElement(jet, iItemHolder, true);
      jet->SetLineColor(jet->GetMainColor());

      float  size  = 50.f * dj.Pt(); // values are saved in scale
      double theta = dj.Theta();
      // printf("%s jet theta =  %f, phi = %f \n",  iItemHolder->GetCName(), theta, dj.Phi());
      double phi = dj.Phi();


      if (viewType == "Projected" )
      {
         static const float_t offr = 6;
         float r_ecal = context->GetMaxR() + offr;
         float z_ecal = context->GetMaxZ() + offr;

         float  transAngle = abs(atan(r_ecal/z_ecal));
         double r = 0;
         bool debug = false;
         if (theta < transAngle || 3.14-theta < transAngle)
         {
            z_ecal = context->GetMaxZ() + offr/transAngle;
            r = z_ecal/fabs(cos(theta));
         }
         else
         {
            debug = true;
            r = r_ecal/sin(theta);
         }

         REveVector p1(0, (phi<TMath::Pi() ? r*fabs(sin(theta)) : -r*fabs(sin(theta))), r*cos(theta));
         REveVector p2(0, (phi<TMath::Pi() ? (r+size)*fabs(sin(theta)) : -(r+size)*fabs(sin(theta))), (r+size)*cos(theta));

         auto marker = new REveScalableStraightLineSet("jetline");
         marker->SetScaleCenter(p1.fX, p1.fY, p1.fZ);
         marker->AddLine(p1, p2);
         marker->SetLineWidth(4);
         if (debug)
             marker->AddMarker(0, 0.9);

         SetupAddElement(marker, iItemHolder, true);
         marker->SetName(Form("line %s %d", Collection()->GetCName(), idx));
      }
   }


   using REveDataProxyBuilderBase::LocalModelChanges;

   void LocalModelChanges(int idx, REveElement* el, const REveViewContext* ctx) override
   {
      // printf("LocalModelChanges jet %s ( %s )\n", el->GetCName(), el->FirstChild()->GetCName());
      REveJetCone* cone = dynamic_cast<REveJetCone*>(el->FirstChild());
      cone->SetLineColor(cone->GetMainColor());
   }
};


class TParticleProxyBuilder : public REveDataSimpleProxyBuilderTemplate<TParticle>
{
   using REveDataSimpleProxyBuilderTemplate<TParticle>::Build;

   void Build(const TParticle& p, int idx, REveElement* iItemHolder, const REveViewContext* context) override
   {
      const TParticle *x = &p;
      auto track = new REveTrack((TParticle*)(x), 1, context->GetPropagator());
      track->MakeTrack();
      SetupAddElement(track, iItemHolder, true);
   }
};

class RecHitProxyBuilder: public REveDataProxyBuilderBase
{
private:
   void buildBoxSet(REveBoxSet* boxset) {
      auto collection = Collection();
      boxset->SetMainColor(collection->GetMainColor());
      boxset->Reset(REveBoxSet::kBT_FreeBox, true, collection->GetNItems());
      TRandom r(0);
#define RND_BOX(x) (Float_t)r.Uniform(-(x), (x))
      for (int h = 0; h < collection->GetNItems(); ++h)
      {
         RecHit* hit = (RecHit*)collection->GetDataPtr(h);
         const REveDataItem* item = Collection()->GetDataItem(h);

         if (!item->GetVisible())
           continue;
         Float_t x = hit->fX;
         Float_t y = hit->fY;
         Float_t z = hit->fZ;
         Float_t a = hit->fPt;
         Float_t d = 0.05;
         Float_t verts[24] = {
                              x - a + RND_BOX(d), y - a + RND_BOX(d), z - a + RND_BOX(d),
                              x - a + RND_BOX(d), y + a + RND_BOX(d), z - a + RND_BOX(d),
                              x + a + RND_BOX(d), y + a + RND_BOX(d), z - a + RND_BOX(d),
                              x + a + RND_BOX(d), y - a + RND_BOX(d), z - a + RND_BOX(d),
                              x - a + RND_BOX(d), y - a + RND_BOX(d), z + a + RND_BOX(d),
                              x - a + RND_BOX(d), y + a + RND_BOX(d), z + a + RND_BOX(d),
                              x + a + RND_BOX(d), y + a + RND_BOX(d), z + a + RND_BOX(d),
                              x + a + RND_BOX(d), y - a + RND_BOX(d), z + a + RND_BOX(d) };
         boxset->AddBox(verts);
         boxset->DigitId(h);
         boxset->DigitColor(item->GetVisible() ? collection->GetMainColor() : 0); // set color on the last one
      }
      boxset->GetPlex()->Refit();
      boxset->StampObjProps();
   }

public:
   using REveDataProxyBuilderBase::Build;
   void Build(const REveDataCollection* collection, REveElement* product, const REveViewContext*)override
   {
      // printf("-------------------------FBOXSET proxy builder %d \n",  collection->GetNItems());
      auto boxset = new REveBoxSet();
      boxset->SetName(collection->GetCName());
      boxset->SetAlwaysSecSelect(1);
      boxset->SetDetIdsAsSecondaryIndices(true);
      boxset->SetSelectionMaster(((REveDataCollection*)collection)->GetItemList());
      buildBoxSet(boxset);
      product->AddElement(boxset);
   }

   using REveDataProxyBuilderBase::FillImpliedSelected;
   void FillImpliedSelected(REveElement::Set_t& impSet, Product* p) override
   {
      // printf("RecHit fill implioed ----------------- !!!%zu\n", Collection()->GetItemList()->RefSelectedSet().size());
      impSet.insert(p->m_elements->FirstChild());
   }

   using REveDataProxyBuilderBase::ModelChanges;
   void ModelChanges(const REveDataCollection::Ids_t& ids, Product* product) override
   {
      // We know there is only one element in this product
      //  printf("RecHitProxyBuilder::model changes %zu\n", ids.size());
      buildBoxSet((REveBoxSet*)product->m_elements->FirstChild());
   }
};


//==============================================================================
//== COLLECTION MANGER  ================================================================
//==============================================================================

class CollectionManager
{
private:
   Event                    *fEvent{nullptr};

   std::vector<REveScene *>  m_scenes;
   REveViewContext          *m_viewContext {nullptr};

   std::vector<REveDataProxyBuilderBase *> m_builders;

   REveScene *m_collections    {nullptr};
   bool       m_inEventLoading {false};

public:
   CollectionManager(Event* event) : fEvent(event)
   {
      //view context
      float r = 300;
      float z = 300;
      auto prop = new REveTrackPropagator();
      prop->SetMagFieldObj(new REveMagFieldDuo(350, 3.5, -2.0));
      prop->SetMaxR(r);
      prop->SetMaxZ(z);
      prop->SetMaxOrbs(6);
      prop->IncRefCount();

      m_viewContext = new REveViewContext();
      m_viewContext->SetBarrel(r, z);
      m_viewContext->SetTrackPropagator(prop);

      // table specs
      auto tableInfo = new REveTableViewInfo();

      tableInfo->table("TParticle").
         column("pt",  1, "i.Pt()").
         column("eta", 3, "i.Eta()").
         column("phi", 3, "i.Phi()");

      tableInfo->table("Jet").
         column("eta",     1, "i.Eta()").
         column("phi",     1, "i.Phi()").
         column("etasize", 2, "i.GetEtaSize()").
         column("phisize", 2, "i.GetPhiSize()");

      tableInfo->table("RecHit").
         column("pt",     1, "i.fPt");

      m_viewContext->SetTableViewInfo(tableInfo);

      for (auto &c : eveMng->GetScenes()->RefChildren()) {
         if (c != eveMng->GetGlobalScene() && strncmp(c->GetCName(), "Geometry", 8) )
         {
            m_scenes.push_back((REveScene*)c);
         }
         if (!strncmp(c->GetCName(),"Table", 5))
         c->AddElement(m_viewContext->GetTableViewInfo());

      }

      m_collections = eveMng->SpawnNewScene("Collections", "Collections");
   }

   void SetDataItemsFromEvent(REveDataCollection* collection)
   {
      for (auto &l : fEvent->fListData) {
         TIter next(l);
         if (collection->GetName() == std::string(l->GetName()))
         {
            collection->ClearItems();

            for (int i = 0; i <= l->GetLast(); ++i)
            {
               std::string cname = collection->GetName();
               auto len = cname.size();
               char end = cname[len-1];
               if (end == 's') {
                  cname = cname.substr(0, len-1);
               }
               TString pname(Form("%s %2d",  cname.c_str(), i));
               collection->AddItem(l->At(i), pname.Data(), "");
            }
         }
         collection->ApplyFilter();
       }
   }

   void LoadEvent()
   {
      m_inEventLoading = true;

      for (auto &el: m_collections->RefChildren())
      {
         auto c = dynamic_cast<REveDataCollection *>(el);
         SetDataItemsFromEvent(c);
      }

      for (auto proxy : m_builders)
      {
         proxy->Build();
      }

      fEvent->fCaloData->DataChanged();
      m_inEventLoading = false;
   }

   void addCollection(REveDataCollection* collection, bool showInTable)
   {
      m_collections->AddElement(collection);

      // load data
      SetDataItemsFromEvent(collection);

      // create builder from classname
      REveDataProxyBuilderBase* glBuilder = 0;
      char* cmd = Form("*((REveDataProxyBuilderBase**) 0x%lx) = new %sProxyBuilder()", (unsigned long)&glBuilder, collection->GetItemClass()->GetName());
      gROOT->ProcessLine(cmd);

      glBuilder->SetCollection(collection);
      glBuilder->SetHaveAWindow(true);
      for (auto scene : m_scenes)
      {
         REveElement *product = glBuilder->CreateProduct(scene->GetTitle(), m_viewContext);

         if (strncmp(scene->GetCName(), "Tables", 5) == 0) continue;

         if (!strncmp(scene->GetCTitle(), "Projected", 8))
         {
            g_projMng->ImportElements(product, scene);
         }
         else
         {
            scene->AddElement(product);
         }
      }
      m_builders.push_back(glBuilder);
      glBuilder->Build();

      // Tables
      auto tableBuilder = new REveTableProxyBuilder();
      tableBuilder->SetHaveAWindow(true);
      tableBuilder->SetCollection(collection);
      REveElement* tablep = tableBuilder->CreateProduct("table-type", m_viewContext);
      auto tableMng =  m_viewContext->GetTableViewInfo();
      if (showInTable)
      {
         tableMng->SetDisplayedCollection(collection->GetElementId());
      }

      for (auto s : m_scenes)
      {
         if (strncmp(s->GetCTitle(), "Table", 5) == 0)
         {
            s->AddElement(tablep);
            tableBuilder->Build(collection, tablep, m_viewContext );
         }
      }
      tableMng->AddDelegate([=]() { tableBuilder->ConfigChanged(); });
      m_builders.push_back(tableBuilder);


      // set tooltip expression for items
      auto tableEntries =  tableMng->RefTableEntries(collection->GetItemClass()->GetName());
      int N  = TMath::Min(int(tableEntries.size()), 3);
      for (int t = 0; t < N; t++) {
         auto te = tableEntries[t];
         collection->GetItemList()->AddTooltipExpression(te.fName, te.fExpression);
      }

      collection->GetItemList()->SetItemsChangeDelegate([&] (REveDataItemList* collection, const REveDataCollection::Ids_t& ids)
                                    {
                                       this->ModelChanged( collection, ids );
                                    });
      collection->GetItemList()->SetFillImpliedSelectedDelegate([&] (REveDataItemList* collection, REveElement::Set_t& impSelSet)
                                    {
                                       this->FillImpliedSelected( collection,  impSelSet);
                                    });
   }

   void finishViewCreate()
   {
      auto mngTable = m_viewContext->GetTableViewInfo();
      if (mngTable)
      {
         for (auto &el : m_collections->RefChildren())
         {
            if (el->GetName() == "Tracks")
               mngTable->SetDisplayedCollection(el->GetElementId());
         }
      }
   }


   void ModelChanged(REveDataItemList* itemList, const REveDataCollection::Ids_t& ids)
   {
      if (m_inEventLoading) return;

      for (auto proxy : m_builders)
      {
         if (proxy->Collection()->GetItemList() == itemList)
         {
            // printf("Model changes check proxy %s: \n", proxy->Type().c_str());
            proxy->ModelChanges(ids);
         }
      }
   }

   void FillImpliedSelected(REveDataItemList* itemList, REveElement::Set_t& impSelSet)
   {
      if (m_inEventLoading) return;

      for (auto proxy : m_builders)
      {
         if (proxy->Collection()->GetItemList() == itemList)
         {
            proxy->FillImpliedSelected(impSelSet);
         }
      }
   }

};


//==============================================================================
//== Event Manager =============================================================
//==============================================================================

class EventManager : public REveElement
{
private:
   Event* fEvent;
   CollectionManager* fCMng;

public:
   EventManager(Event* e, CollectionManager* m): fEvent(e), fCMng(m) {}

   virtual ~EventManager() {}

   virtual void NextEvent()
   {
      eveMng->DisableRedraw();
      eveMng->GetSelection()->ClearSelection();
      eveMng->GetHighlight()->ClearSelection();
      fEvent->Create();
      fCMng->LoadEvent();
      eveMng->EnableRedraw();
   }
};


//==============================================================================
//== main() ====================================================================
//==============================================================================

void collection_proxies(bool proj=true)
{
   eveMng = REveManager::Create();
   auto event = new Event();
   event->Create();

   // create scenes and views
   REveScene* rhoZEventScene = nullptr;
   
   auto b1 = new REveGeoShape("Barrel 1");
   b1->SetShape(new TGeoTube(kR_min, kR_max, kZ_d));
   b1->SetMainColor(kCyan);
   eveMng->GetGlobalScene()->AddElement(b1);
  
   rhoZEventScene = eveMng->SpawnNewScene("RhoZ Scene","Projected");
   g_projMng = new REveProjectionManager(REveProjection::kPT_RhoZ);
   g_projMng->SetImportEmpty(true);

   auto rhoZView = eveMng->SpawnNewViewer("RhoZ View");
   rhoZView->AddScene(rhoZEventScene);
   auto pgeoScene = eveMng->SpawnNewScene("Geometry projected");
   rhoZView->AddScene(pgeoScene);
   g_projMng->ImportElements(b1, pgeoScene);

   auto tableScene = eveMng->SpawnNewScene ("Tables", "Tables");
   auto tableView  = eveMng->SpawnNewViewer("Table",  "Table View");
   tableView->AddScene(tableScene);

   // create event data from list
   auto collectionMng = new CollectionManager(event);

   REveDataCollection* trackCollection = new REveDataCollection("Tracks");
   trackCollection->SetItemClass(TParticle::Class());
   trackCollection->SetMainColor(kGreen);
   trackCollection->SetFilterExpr("i.Pt() > 4.1 && std::abs(i.Eta()) < 1");
   collectionMng->addCollection(trackCollection, true);

   REveDataCollection* jetCollection = new REveDataCollection("Jets");
   jetCollection->SetItemClass(Jet::Class());
   jetCollection->SetMainColor(kYellow);
   collectionMng->addCollection(jetCollection, false);

   REveDataCollection* hitCollection = new REveDataCollection("RecHits");
   hitCollection->SetItemClass(RecHit::Class());
   hitCollection->SetMainColor(kOrange + 7);
   hitCollection->SetFilterExpr("i.fPt > 5");
   collectionMng->addCollection(hitCollection, false);

   // add calorimeters
   auto calo3d = new REveCalo3D(event->fCaloData);
   calo3d->SetBarrelRadius(kR_max);
   calo3d->SetEndCapPos(kZ_d);
   calo3d->SetMaxTowerH(300);
   eveMng->GetEventScene()->AddElement(calo3d);
   REveCalo2D* calo2d = (REveCalo2D*) g_projMng->ImportElements(calo3d, rhoZEventScene);


   // event navigation
   auto eventMng = new EventManager(event, collectionMng);
   eventMng->SetName("EventManager");
   eveMng->GetWorld()->AddElement(eventMng);

   eveMng->GetWorld()->AddCommand("NextEvent", "sap-icon://step", eventMng, "NextEvent()");

   eveMng->Show();
}
