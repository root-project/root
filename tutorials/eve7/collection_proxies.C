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


// a demo class, can be provided from experiment framework
class XYJet : public TParticle
{
private:
   float m_etaSize{0};
   float m_phiSize{0};

public:
   float GetEtaSize() const { return m_etaSize; }
   float GetPhiSize() const { return m_phiSize; }
   void SetEtaSize(float iEtaSize) { m_etaSize = iEtaSize; }
   void SetPhiSize(float iPhiSize) { m_phiSize = iPhiSize; }
   XYJet(Int_t pdg, Int_t status, Int_t mother1, Int_t mother2, Int_t daughter1, Int_t daughter2, Double_t px, Double_t py, Double_t pz, Double_t etot):
    TParticle(pdg, status, mother1, mother2, daughter1, daughter2, px, py, pz, etot,  0, 0, 0, 0) {}

   ClassDef(XYJet, 1);
};

class Event {
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

         printf("Event::MakeParticles %2d: pt=%.2f, eta=%.2f, phi=%.2f\n", i, pt, eta, phi);

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

   void Clear() {
      for (auto &l : m_data)
         delete l;
      m_data.clear();
   }

   void Create() {
      Clear();
      MakeJets(4);
      MakeParticles(10);
      eventId++;
   }
};

//==============================================================================
//============ PROXY BUILDERS  ================================================
//==============================================================================
class XYJetProxyBuilder: public REX::REveDataSimpleProxyBuilderTemplate<XYJet>
{
   virtual bool HaveSingleProduct() const { return false; }

   using REveDataSimpleProxyBuilderTemplate<XYJet>::BuildViewType;
   virtual void BuildViewType(const XYJet& dj, unsigned int /*idx*/, REX::REveElement* iItemHolder, std::string viewType, const REX::REveViewContext* context)
   {
      auto jet = new REX::REveJetCone();
      jet->SetCylinder(context->GetMaxR(), context->GetMaxZ());
      jet->AddEllipticCone(dj.Eta(), dj.Phi(), dj.GetEtaSize(), dj.GetPhiSize());
      SetupAddElement(jet, iItemHolder, true);

      REX::REveVector p1;
      REX::REveVector p2;

      float size = 50.f * dj.Pt(); // values are saved in scale
      double theta = dj.Theta();
      // printf("%s jet theta =  %f, phi = %f \n",  iItemHolder->GetCName(), theta, dj.Phi());
      double phi = dj.Phi();


      if (viewType == "Projected" )
      {
         static const float_t offr = 6;
         float r_ecal = context->GetMaxR() + offr;
         float z_ecal = context->GetMaxZ() + offr;

         float transAngle = abs(atan(r_ecal/z_ecal));
         double r(0);
         bool debug = false;
         if ( theta < transAngle || 3.14-theta < transAngle)
         {
            z_ecal = context->GetMaxZ() + offr/transAngle;
            r = z_ecal/fabs(cos(theta));
         }
         else
         {
            debug = 3;
            r = r_ecal/sin(theta);
         }

         p1.Set( 0., (phi<TMath::Pi() ? r*fabs(sin(theta)) : -r*fabs(sin(theta))), r*cos(theta));
         p2.Set( 0., (phi<TMath::Pi() ? (r+size)*fabs(sin(theta)) : -(r+size)*fabs(sin(theta))), (r+size)*cos(theta) );


         auto marker = new REX::REveScalableStraightLineSet("jetline");
         marker->SetScaleCenter(p1.fX, p1.fY, p1.fZ);
         marker->AddLine(p1, p2);

         marker->SetLineWidth(4);
         if (debug)
             marker->AddMarker(0, 0.9);

         SetupAddElement(marker, iItemHolder, true);
      }

      jet->SetName(Form("element %s", iItemHolder->GetName().c_str()));
   }
};

class TrackProxyBuilder : public REX::REveDataSimpleProxyBuilderTemplate<TParticle>
{
   using REveDataSimpleProxyBuilderTemplate<TParticle>::Build;
   virtual void Build(const TParticle& p, unsigned int /*idx*/, REX::REveElement* iItemHolder, const REX::REveViewContext* context)
   {
      const TParticle *x = &p;
      // printf("==============  BUILD track %s (pt=%f, eta=%f) \n", iItemHolder->GetCName(), p.Pt(), p.Eta());
      auto track = new REX::REveTrack((TParticle*)(x), 1, context->GetPropagator());
      track->MakeTrack();
      SetupAddElement(track, iItemHolder, true);
      // iItemHolder->AddElement(track);
      track->SetName(Form("element %s id=%d", iItemHolder->GetCName(), track->GetElementId()));
   }
};


//==============================================================================
//==============================================================================
// ================= XY MANGER  ================================================
//==============================================================================
//==============================================================================
class XYManager
{
private:
   Event *m_event{nullptr};

   std::vector<REX::REveScene *> m_scenes;
   REX::REveViewContext *m_viewContext{nullptr};
   REX::REveProjectionManager *m_mngRhoZ{nullptr};

   std::vector<REX::REveDataProxyBuilderBase *> m_builders;
   REX::REveScene *m_collections{nullptr};
   bool m_inEventLoading{false};

public:
   XYManager(Event* event): m_event(event)
   {
      //view context
      float r = 300;
      float z = 300;
      auto prop = new REX::REveTrackPropagator();
      prop->SetMagFieldObj(new REX::REveMagFieldDuo(350, -3.5, 2.0));
      prop->SetMaxR(r);
      prop->SetMaxZ(z);
      prop->SetMaxOrbs(6);
      prop->IncRefCount();

      m_viewContext = new REX::REveViewContext();
      m_viewContext->SetBarrel(r, z);
      m_viewContext->SetTrackPropagator(prop);

      // table specs
      auto tableInfo = new REX::REveTableViewInfo();
      tableInfo->table("XYTracks").
         column("pt", 1, "Pt").
         column("eta", 3, "Eta").
         column("phi", 3, "Phi");

      tableInfo->table("XYJets").
         column("eta", 1, "Eta").
         column("phi", 1, "Phi").
         column("etasize", 2, "GetEtaSize").
         column("phisize", 2, "GetPhiSize");

      m_viewContext->SetTableViewInfo(tableInfo);

      createScenesAndViews();
   }

   void createScenesAndViews()
   {
      // collections
      m_collections = eveMng->SpawnNewScene("Collections","Collections");

      // 3D
      m_scenes.push_back(eveMng->GetEventScene());

      // Geometry
      auto b1 = new REX::REveGeoShape("Barrel 1");
      float dr = 3;
      b1->SetShape(new TGeoTube(m_viewContext->GetMaxR() , m_viewContext->GetMaxR() + dr, m_viewContext->GetMaxZ()));
      b1->SetMainColor(kCyan);
      eveMng->GetGlobalScene()->AddElement(b1);

      // RhoZ
      if (gRhoZView) {
         auto rhoZEventScene = eveMng->SpawnNewScene("RhoZ Scene","Projected");
         m_mngRhoZ = new REX::REveProjectionManager(REX::REveProjection::kPT_RhoZ);
         m_mngRhoZ->SetImportEmpty(true);
         auto rhoZView = eveMng->SpawnNewViewer("RhoZ View", "");
         rhoZView->AddScene(rhoZEventScene);
         m_scenes.push_back(rhoZEventScene);

         auto pgeoScene = eveMng->SpawnNewScene("Projection Geometry","xxx");
         m_mngRhoZ->ImportElements(b1,pgeoScene );
         rhoZView->AddScene(pgeoScene);
      }

      // Table
      if (1) {
         auto tableScene  = eveMng->SpawnNewScene("Tables", "Tables");
         auto tableView = eveMng->SpawnNewViewer("Table", "Table View");
         tableView->AddScene(tableScene);
         tableScene->AddElement(m_viewContext->GetTableViewInfo());
         m_scenes.push_back(tableScene);
      }

   }

   // this should be handeled with framefor plugins
   REX::REveDataProxyBuilderBase*  makeGLBuilderForType(TClass* c)
   {
      std::string cn = c->GetName();
      // printf("proxy builder for type %s\n", c->GetName());
      if (cn == "XYJet") {
         return new XYJetProxyBuilder();
      }
      else
      {
         return new TrackProxyBuilder();
      }
   }

   void LoadCurrentEvent(REX::REveDataCollection* collection)
   {
      //  printf("load current event \n");
      for (auto &l : m_event->m_data) {
         TIter next(l);
         if (collection->GetName() == std::string(l->GetName()))
         {
            // printf("collection for list %s %s\n", collection->GetCName(), l->GetName());
            collection->ClearItems();
            collection->DestroyElements();

            for (int i = 0; i <= l->GetLast(); ++i)
            {
               TString pname; pname.Form("item %2d", i);
               collection->AddItem(l->At(i), pname.Data(), "");
            }
         }
         //collections->RefChildren())
       }
   }

   void NextEvent()
   {
      m_inEventLoading = true;
      for (auto &el: m_collections->RefChildren())
      {
         auto c = dynamic_cast<REX::REveDataCollection *>(el);
         LoadCurrentEvent(c);
         c->ApplyFilter();
      }

      for (auto proxy : m_builders) {
         proxy->Build();
      }
      m_inEventLoading = false;
   }

   void addCollection(REX::REveDataCollection* collection, bool makeTable)
   {
      // load data
      LoadCurrentEvent(collection);

      // GL view types
      auto glBuilder = makeGLBuilderForType(collection->GetItemClass());
      glBuilder->SetCollection(collection);
      glBuilder->SetHaveAWindow(true);
      for (auto scene : m_scenes) {
         REX::REveElement *product = glBuilder->CreateProduct(scene->GetTitle(), m_viewContext);
         if (strncmp(scene->GetCTitle(), "Table", 5) == 0) continue;
         if (!strncmp(scene->GetCTitle(), "Projected", 8)) {
            m_mngRhoZ->ImportElements(product, scene);
         }
         else {
            scene->AddElement(product);
         }
      }
      m_builders.push_back(glBuilder);
      glBuilder->Build();

      if (makeTable) {
         // Table view types      {
         auto tableBuilder = new REX::REveTableProxyBuilder();
         tableBuilder->SetHaveAWindow(true);
         tableBuilder->SetCollection(collection);
         REX::REveElement* tablep = tableBuilder->CreateProduct("table-type", m_viewContext);

         auto tableMng =  m_viewContext->GetTableViewInfo();
         tableMng->SetDisplayedCollection(collection->GetElementId());
         tableMng->AddDelegate([=](REX::ElementId_t elId) { tableBuilder->DisplayedCollectionChanged(elId); });

         for (REX::REveScene* scene : m_scenes) {
            if (strncmp(scene->GetCTitle(), "Table", 5) == 0) {
               scene->AddElement(tablep);
               tableBuilder->Build(collection, tablep, m_viewContext );
            }
         }

         m_builders.push_back(tableBuilder);
      }

      m_collections->AddElement(collection);
      collection->SetHandlerFunc([&] (REX::REveDataCollection* collection) { this->CollectionChanged( collection ); });
      collection->SetHandlerFuncIds([&] (REX::REveDataCollection* collection, const REX::REveDataCollection::Ids_t& ids) { this->ModelChanged( collection, ids ); });
   }

   void finishViewCreate()
   {
      auto mngTable = m_viewContext->GetTableViewInfo();
      if (mngTable) {
         for (auto &el : m_collections->RefChildren())
         {
            if (el->GetName() == "XYTracks")
               mngTable->SetDisplayedCollection(el->GetElementId());
         }
      }
   }

   void CollectionChanged(REX::REveDataCollection* collection) {
      printf("collection changes not implemented %s!\n", collection->GetCName());
   }

   void ModelChanged(REX::REveDataCollection* collection, const REX::REveDataCollection::Ids_t& ids) {
      if (m_inEventLoading) return;

      for (auto proxy : m_builders) {
         if (proxy->Collection() == collection) {
            // printf("Model changes check proxy %s: \n", proxy->Type().c_str());
            proxy->ModelChanges(ids);
         }
      }
   }
};


//==============================================================================

class EventManager : public REX::REveElement
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



void collection_proxies(bool proj=true)
{
   eveMng = REX::REveManager::Create();

   auto event = new Event();
   event->Create();

   gRhoZView = true;

   // debug settings
   auto xyManager = new XYManager(event);

   if (1) {
      REX::REveDataCollection* trackCollection = new REX::REveDataCollection("XYTracks");
      trackCollection->SetItemClass(TParticle::Class());
      trackCollection->SetMainColor(kGreen);
      //trackCollection->SetFilterExpr("i.Pt() > 0.1 && std::abs(i.Eta()) < 1");
      xyManager->addCollection(trackCollection, true);
   }

   if (1) {
      REX::REveDataCollection* jetCollection = new REX::REveDataCollection("XYJets");
      jetCollection->SetItemClass(XYJet::Class());
      jetCollection->SetMainColor(kRed);
      xyManager->addCollection(jetCollection, false);
   }

   auto eventMng = new EventManager(event, xyManager);
   eventMng->SetName("EventManager");
   eveMng->GetWorld()->AddElement(eventMng);

   eveMng->GetWorld()->AddCommand("QuitRoot", "sap-icon://log", eventMng, "QuitRoot()");
   eveMng->GetWorld()->AddCommand("NextEvent", "sap-icon://step", eventMng, "NextEvent()");

   eveMng->Show();
}
