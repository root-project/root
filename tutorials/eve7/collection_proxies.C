/// \file
/// \ingroup tutorial_eve7
///  This example display collection of ??? in web browser
///
/// \macro_code
///


#include "ROOT/REveDataClasses.hxx"
//#include "ROOT/REveDataProxyBuilderBase.hxx"
//#include "ROOT/REveDataSimpleProxyBuilder.hxx"
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

#include "TGeoTube.h"
#include "TList.h"
#include "TParticle.h"
#include "TRandom.h"
#include "TApplication.h"


namespace REX = ROOT::Experimental;

bool gRhoZView = false;

REX::REveManager *eveMng = nullptr;


//==============================================================================
//============== EMULATE FRAMEWORK CLASSES =====================================
//==============================================================================

using namespace ROOT::Experimental;

// a demo class, can be provided from experiment framework
class XYJet : public TParticle
{
private:
   float m_etaSize{0};
   float m_phiSize{0};

public:
   float GetEtaSize() const { return m_etaSize; }
   float GetPhiSize() const { return m_phiSize; }
   void  SetEtaSize(float iEtaSize) { m_etaSize = iEtaSize; }
   void  SetPhiSize(float iPhiSize) { m_phiSize = iPhiSize; }

  XYJet(Int_t pdg, Int_t status, Int_t mother1, Int_t mother2, Int_t daughter1, Int_t daughter2,
        Double_t px, Double_t py, Double_t pz, Double_t etot) :
    TParticle(pdg, status, mother1, mother2, daughter1, daughter2, px, py, pz, etot,  0, 0, 0, 0)
  {}

   ClassDef(XYJet, 1);
};

class Event
{
public:
   int eventId{0};
   int N_tracks{0};
   int N_jets{0};

   Event() = default;

   void MakeJets(int N)
   {
      TRandom &r = *gRandom;
      r.SetSeed(0);
      TList* list = new TList();
      list->SetName("XYJets");
      for (int i = 1; i <= N; ++i)
      {
         double pt  = r.Uniform(0.5, 10);
         double eta = r.Uniform(-2.55, 2.55);
         double phi = r.Uniform(-TMath::Pi(), TMath::Pi());

         double px = pt * std::cos(phi);
         double py = pt * std::sin(phi);
         double pz = pt * (1. / (std::tan(2*std::atan(std::exp(-eta)))));

         auto jet = new XYJet(0, 0, 0, 0, 0, 0, px, py, pz, std::sqrt(px*px + py*py + pz*pz + 80*80));
         jet->SetEtaSize(r.Uniform(0.02, 0.2));
         jet->SetPhiSize(r.Uniform(0.01, 0.3));
         list->Add(jet);
      }
      m_data.push_back(list);
   }

   void MakeParticles(int N)
   {
      TRandom &r = *gRandom;
      r.SetSeed(0);
      TList* list = new TList();
      list->SetName("XYTracks");
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
      m_data.push_back(list);
   }

   std::vector<TList*> m_data;

   void Clear()
   {
      for (auto &l : m_data)
         delete l;
      m_data.clear();
   }

   void Create()
   {
      Clear();
      MakeJets(4);
      MakeParticles(100);
      eventId++;
   }
};


//==============================================================================
//== PROXY BUILDERS ============================================================
//==============================================================================

class XYJetProxyBuilder: public REveDataSimpleProxyBuilderTemplate<XYJet>
{
   bool HaveSingleProduct() const override { return false; }

   using REveDataSimpleProxyBuilderTemplate<XYJet>::BuildViewType;

   void BuildViewType(const XYJet& dj, int idx, REveElement* iItemHolder,
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
      printf("LocalModelChanges jet %s ( %s )\n", el->GetCName(), el->FirstChild()->GetCName());
      REveJetCone* cone = dynamic_cast<REveJetCone*>(el->FirstChild());
      cone->SetLineColor(cone->GetMainColor());
   }
};


class TrackProxyBuilder : public REveDataSimpleProxyBuilderTemplate<TParticle>
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


//==============================================================================
//== XY MANGER  ================================================================
//==============================================================================

class XYManager
{
private:
   Event                    *m_event{nullptr};

   std::vector<REveScene *>  m_scenes;
   REveViewContext          *m_viewContext {nullptr};
   REveProjectionManager    *m_mngRhoZ     {nullptr};

   std::vector<REveDataProxyBuilderBase *> m_builders;

   REveScene *m_collections    {nullptr};
   bool       m_inEventLoading {false};

public:
   XYManager(Event* event) : m_event(event)
   {
      //view context
      float r = 300;
      float z = 300;
      auto prop = new REveTrackPropagator();
      prop->SetMagFieldObj(new REveMagFieldDuo(350, -3.5, 2.0));
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

      tableInfo->table("XYJet").
         column("eta",     1, "i.Eta()").
         column("phi",     1, "i.Phi()").
         column("etasize", 2, "i.GetEtaSize()").
         column("phisize", 2, "i.GetPhiSize()");

      m_viewContext->SetTableViewInfo(tableInfo);

      createScenesAndViews();
   }

   void createScenesAndViews()
   {
      // collections
      m_collections = eveMng->SpawnNewScene("Collections", "Collections");

      // 3D
      m_scenes.push_back(eveMng->GetEventScene());

      // Geometry
      auto b1 = new REveGeoShape("Barrel 1");
      float dr = 3;
      b1->SetShape(new TGeoTube(m_viewContext->GetMaxR() , m_viewContext->GetMaxR() + dr, m_viewContext->GetMaxZ()));
      b1->SetMainColor(kCyan);
      eveMng->GetGlobalScene()->AddElement(b1);

      // RhoZ
      if (gRhoZView)
      {
         auto rhoZEventScene = eveMng->SpawnNewScene("RhoZ Scene","Projected");
         m_mngRhoZ = new REveProjectionManager(REveProjection::kPT_RhoZ);
         m_mngRhoZ->SetImportEmpty(true);
         auto rhoZView = eveMng->SpawnNewViewer("RhoZ View", "");
         rhoZView->AddScene(rhoZEventScene);
         m_scenes.push_back(rhoZEventScene);

         auto pgeoScene = eveMng->SpawnNewScene("Projection Geometry","xxx");
         m_mngRhoZ->ImportElements(b1,pgeoScene );
         rhoZView->AddScene(pgeoScene);
      }

      // Table
      if (1)
      {
         auto tableScene = eveMng->SpawnNewScene ("Tables", "Tables");
         auto tableView  = eveMng->SpawnNewViewer("Table",  "Table View");
         tableView->AddScene(tableScene);
         tableScene->AddElement(m_viewContext->GetTableViewInfo());
         m_scenes.push_back(tableScene);
      }
   }

   // this should be handeled with framefor plugins
   REveDataProxyBuilderBase* makeGLBuilderForType(TClass* c)
   {
      std::string cn = c->GetName();
      if (cn == "XYJet") {
         return new XYJetProxyBuilder();
      }
      else
      {
         return new TrackProxyBuilder();
      }
   }

   void LoadCurrentEvent(REveDataCollection* collection)
   {
      for (auto &l : m_event->m_data) {
         TIter next(l);
         if (collection->GetName() == std::string(l->GetName()))
         {
            collection->ClearItems();
            collection->DestroyElements();

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

   void NextEvent()
   {
      m_inEventLoading = true;

      for (auto &el: m_collections->RefChildren())
      {
         auto c = dynamic_cast<REveDataCollection *>(el);
         LoadCurrentEvent(c);
      }

      for (auto proxy : m_builders)
      {
         proxy->Build();
      }

      m_inEventLoading = false;
   }

   void addCollection(REveDataCollection* collection, bool showInTable)
   {
      m_collections->AddElement(collection);

      // load data
      LoadCurrentEvent(collection);

      // GL view types
      auto glBuilder = makeGLBuilderForType(collection->GetItemClass());
      glBuilder->SetCollection(collection);
      glBuilder->SetHaveAWindow(true);
      for (auto scene : m_scenes)
      {
         REveElement *product = glBuilder->CreateProduct(scene->GetTitle(), m_viewContext);

         if (strncmp(scene->GetCTitle(), "Table", 5) == 0) continue;

         if (!strncmp(scene->GetCTitle(), "Projected", 8))
         {
            m_mngRhoZ->ImportElements(product, scene);
         }
         else
         {
            scene->AddElement(product);
         }
      }
      m_builders.push_back(glBuilder);
      glBuilder->Build();

      // Table view types
      auto tableBuilder = new REveTableProxyBuilder();
      tableBuilder->SetHaveAWindow(true);
      tableBuilder->SetCollection(collection);
      REveElement* tablep = tableBuilder->CreateProduct("table-type", m_viewContext);
      auto tableMng =  m_viewContext->GetTableViewInfo();
      if (showInTable)
      {
         tableMng->SetDisplayedCollection(collection->GetElementId());
      }
      tableMng->AddDelegate([=]() { tableBuilder->ConfigChanged(); });
      for (REveScene* scene : m_scenes)
      {
         if (strncmp(scene->GetCTitle(), "Table", 5) == 0)
         {
            scene->AddElement(tablep);
            tableBuilder->Build(collection, tablep, m_viewContext );
         }
      }
      m_builders.push_back(tableBuilder);

      collection->SetHandlerFunc([&] (REveDataCollection* collection)
                                 {
                                    this->CollectionChanged( collection );
                                 });

      collection->SetHandlerFuncIds([&] (REveDataCollection* collection, const REveDataCollection::Ids_t& ids)
                                    {
                                       this->ModelChanged( collection, ids );
                                    });
   }

   void finishViewCreate()
   {
      auto mngTable = m_viewContext->GetTableViewInfo();
      if (mngTable)
      {
         for (auto &el : m_collections->RefChildren())
         {
            if (el->GetName() == "XYTracks")
               mngTable->SetDisplayedCollection(el->GetElementId());
         }
      }
   }

   void CollectionChanged(REveDataCollection* collection)
   {
      printf("collection changes not implemented %s!\n", collection->GetCName());
   }

   void ModelChanged(REveDataCollection* collection, const REveDataCollection::Ids_t& ids)
   {
      if (m_inEventLoading) return;

      for (auto proxy : m_builders)
      {
         if (proxy->Collection() == collection)
         {
            // printf("Model changes check proxy %s: \n", proxy->Type().c_str());
            proxy->ModelChanges(ids);
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
   Event* m_event;
   XYManager* m_xymng;

public:
   EventManager(Event* e, XYManager* m): m_event(e), m_xymng(m) {}

   virtual ~EventManager() {}

   virtual void NextEvent()
   {
      m_event->Create();
      m_xymng->NextEvent();
   }

   virtual void QuitRoot()
   {
      printf("Quit ROOT\n");
      if (gApplication) gApplication->Terminate();
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

   gRhoZView = true;

   // debug settings
   auto xyManager = new XYManager(event);

   if (1)
   {
      REveDataCollection* trackCollection = new REveDataCollection("XYTracks");
      trackCollection->SetItemClass(TParticle::Class());
      trackCollection->SetMainColor(kGreen);
      trackCollection->SetFilterExpr("i.Pt() > 4.1 && std::abs(i.Eta()) < 1");
      xyManager->addCollection(trackCollection, true);
   }

   if (1)
   {
      REveDataCollection* jetCollection = new REveDataCollection("XYJets");
      jetCollection->SetItemClass(XYJet::Class());
      jetCollection->SetMainColor(kRed);
      xyManager->addCollection(jetCollection, false);
   }

   auto eventMng = new EventManager(event, xyManager);
   eventMng->SetName("EventManager");
   eveMng->GetWorld()->AddElement(eventMng);

   eveMng->GetWorld()->AddCommand("QuitRoot",  "sap-icon://log",  eventMng, "QuitRoot()");
   eveMng->GetWorld()->AddCommand("NextEvent", "sap-icon://step", eventMng, "NextEvent()");

   eveMng->Show();
}
