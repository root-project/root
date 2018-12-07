#ifndef ROOT7_REveDataProxyBuilderBase
#define ROOT7_REveDataProxyBuilderBase

#include <ROOT/REveElement.hxx>
#include <ROOT/REveCompound.hxx>
#include <ROOT/REveDataClasses.hxx>

namespace ROOT {
namespace Experimental {

//class REveDataInteractionList;
class REveViewContext;
class REveTrackPropagator;

class REveDataProxyBuilderBase
{
public:
   struct Product
   {
      std::string              m_viewType;
      const REveViewContext   *m_viewContext;
      REveCompound            *m_elements;

      Product(std::string viewType, const REveViewContext* c);
      virtual ~Product();
   };

   // ---------- const member functions ---------------------

   const REveViewContext&    Context()    const;
   const REveDataCollection* Collection() const { return m_collection;  }

   // ---------- constructor/destructor  ---------------------

   REveDataProxyBuilderBase(std::string type);
   virtual ~REveDataProxyBuilderBase() {}

   void SetCollection(REveDataCollection*);
   //   virtual void SetInteractionList(REveDataInteractionList*, const std::string&);

   virtual void CollectionBeingDestroyed(const REveDataCollection*);

   void Build();
   // virtual void Build(REveElement* product);

   REveElement* CreateProduct(std::string viewType, const REveViewContext*);
   //  void removePerViewProduct(const REveViewContext* vc);

   void ModelChanges(const REveDataCollection::Ids_t&);
   void CollectionChanged(const REveDataCollection*);

   void SetupElement(REveElement* el, bool color = true) const;
   void SetupAddElement(REveElement* el, REveElement* parent,  bool set_color = true) const;

   bool GetHaveAWindow() const { return m_haveWindow; }
   void SetHaveAWindow(bool);

   std::string Type() const { return m_type; }

   // const member functions
   virtual bool HaveSingleProduct() const { return true; }

protected:
   // Override this if visibility changes can cause (re)-creation of proxies.
   // Returns true if new proxies were created.
   virtual bool VisibilityModelChanges(int idx, REveElement*, const REveViewContext*);

   virtual void Build(const REveDataCollection* iItem, REveElement* product, const REveViewContext*);
   virtual void BuildViewType(const REveDataCollection* iItem, REveElement* product, std::string viewType, const REveViewContext*);

   virtual void ModelChanges(const REveDataCollection::Ids_t&, Product*);

  // utility
   REveCompound* CreateCompound(bool set_color=true, bool propagate_color_to_all_children=false) const;
   virtual void Clean();
   virtual void CleanLocal();

   typedef std::vector<Product*>::iterator Product_it;
   std::vector<Product*> m_products;

private:
   std::string           m_type;

   const REveDataCollection*   m_collection;

   float                 m_layer;
   //   REveDataInteractionList*  m_interactionList;
   bool                  m_haveWindow;
   bool                  m_modelsChanged;


   ClassDef(REveDataProxyBuilderBase, 0);

};


} // namespace Experimental
} // namespace ROOT
#endif
