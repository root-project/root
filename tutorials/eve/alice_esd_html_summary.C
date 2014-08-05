// @(#)root/eve:$Id$

// Html table and event summary for alice_esd.C

//==============================================================================

class HtmlObjTable : public TObject
{
public:                     // make them public for shorter code

   TString   fName;
   Int_t     fNValues;      // number of values
   Int_t     fNFields;      // number of fields
   TArrayF  *fValues;
   TString  *fLabels;
   Bool_t    fExpand;

   TString   fHtml;         // HTML output code

   void Build();
   void BuildTitle();
   void BuildLabels();
   void BuildTable();

public:
   HtmlObjTable(const char *name, Int_t nfields, Int_t nvals, Bool_t exp=kTRUE);
   virtual ~HtmlObjTable();

   void     SetLabel(Int_t col, const char *label) { fLabels[col] = label; }
   void     SetValue(Int_t col, Int_t row, Float_t val) { fValues[col].SetAt(val, row); }
   TString  Html() const { return fHtml; }

   ClassDef(HtmlObjTable, 0);
};

//==============================================================================

class HtmlSummary
{
public:                           // make them public for shorter code
   Int_t           fNTables;
   TOrdCollection *fObjTables;    // ->array of object tables
   TString         fHtml;         // output HTML string
   TString         fTitle;        // page title
   TString         fHeader;       // HTML header
   TString         fFooter;       // HTML footer

   void     MakeHeader();
   void     MakeFooter();

public:
   HtmlSummary(const char *title);
   virtual ~HtmlSummary();

   HtmlObjTable  *AddTable(const char *name, Int_t nfields, Int_t nvals,
                           Bool_t exp=kTRUE, Option_t *opt="");
   HtmlObjTable  *GetTable(Int_t at) const { return (HtmlObjTable *)fObjTables->At(at); }
   void           Build();
   void           Clear(Option_t *option="");
   void           Reset(Option_t *option="");
   TString        Html() const { return fHtml; }

   ClassDef(HtmlSummary, 0);
};

//==============================================================================

HtmlSummary *fgHtmlSummary = 0;
TGHtml      *fgHtml        = 0;

//==============================================================================

//______________________________________________________________________________
HtmlObjTable::HtmlObjTable(const char *name, Int_t nfields, Int_t nvals, Bool_t exp) :
   fName(name), fNValues(nvals), fNFields(nfields), fExpand(exp)
{
   // Constructor.

   fValues = new TArrayF[fNFields];
   for (int i=0;i<fNFields;i++)
      fValues[i].Set(nvals);
   fLabels = new TString[fNFields];
}

//______________________________________________________________________________
HtmlObjTable::~HtmlObjTable()
{
   // Destructor.

   delete [] fValues;
   delete [] fLabels;
}

//______________________________________________________________________________
void HtmlObjTable::Build()
{
   // Build HTML code.

   fHtml = "<table width=100% border=1 cellspacing=0 cellpadding=0 bgcolor=f0f0f0> ",

   BuildTitle();
   if (fExpand && (fNFields > 0) && (fNValues > 0)) {
      BuildLabels();
      BuildTable();
   }

   fHtml += "</table>";
}

//______________________________________________________________________________
void HtmlObjTable::BuildTitle()
{
   // Build table title.

   fHtml += "<tr><td colspan=";
   fHtml += Form("%d>", fNFields+1);
   fHtml += "<table width=100% border=0 cellspacing=2 cellpadding=0 bgcolor=6e6ea0>";
   fHtml += "<tr><td align=left>";
   fHtml += "<font face=Verdana size=3 color=ffffff><b><i>";
   fHtml += fName;
   fHtml += "</i></b></font></td>";
   fHtml += "<td>";
   fHtml += "<td align=right> ";
   fHtml += "<font face=Verdana size=3 color=ffffff><b><i>";
   fHtml += Form("Size = %d", fNValues);
   fHtml += "</i></b></font></td></tr>";
   fHtml += "</table>";
   fHtml += "</td></tr>";
}

//______________________________________________________________________________
void HtmlObjTable::BuildLabels()
{
   // Build table labels.

   Int_t i;
   fHtml += "<tr bgcolor=c0c0ff>";
   fHtml += "<th> </th>"; // for the check boxes
   for (i=0;i<fNFields;i++) {
      fHtml += "<th> ";
      fHtml += fLabels[i];
      fHtml += " </th>"; // for the check boxes
   }
   fHtml += "</tr>";
}

//______________________________________________________________________________
void HtmlObjTable::BuildTable()
{
   // Build part of table with values.

   for (int i = 0; i < fNValues; i++) {
      if (i%2)
         fHtml += "<tr bgcolor=e0e0ff>";
      else
         fHtml += "<tr bgcolor=ffffff>";

      TString name = fName;
      name.ReplaceAll(" ", "_");
      // checkboxes
      fHtml += "<td bgcolor=d0d0ff align=\"center\">";
      fHtml += "<input type=\"checkbox\" name=\"";
      fHtml += name;
      fHtml += Form("[%d]\">",i);
      fHtml += "</td>";

      for (int j = 0; j < fNFields; j++) {
         fHtml += "<td width=";
         fHtml += Form("%d%%", 100/fNFields);
         fHtml += " align=\"center\"";
         fHtml += ">";
         fHtml += Form("%1.4f", fValues[j][i]);
         fHtml += "</td>";
      }
      fHtml += "</tr> ";
   }
}

//______________________________________________________________________________
HtmlSummary::HtmlSummary(const char *title) : fNTables(0), fTitle(title)
{
   // Constructor.

   fObjTables = new TOrdCollection();
}

//______________________________________________________________________________
HtmlSummary::~HtmlSummary()
{
   // Destructor.

   Reset();
}

//______________________________________________________________________________
HtmlObjTable *HtmlSummary::AddTable(const char *name, Int_t nfields, Int_t nvals,
                                    Bool_t exp, Option_t *option)
{
   // Add a new table in our list of tables.

   TString opt = option;
   opt.ToLower();
   HtmlObjTable *table = new HtmlObjTable(name, nfields, nvals, exp);
   fNTables++;
   if (opt.Contains("first"))
      fObjTables->AddFirst(table);
   else
      fObjTables->Add(table);
   return table;
}

//______________________________________________________________________________
void HtmlSummary::Clear(Option_t *option)
{
   // Clear the table list.

   if (option && option[0] == 'D')
      fObjTables->Delete(option);
   else
      fObjTables->Clear(option);
   fNTables = 0;
}

//______________________________________________________________________________
void HtmlSummary::Reset(Option_t *)
{
   // Reset (delete) the table list;

   delete fObjTables; fObjTables = 0;
   fNTables = 0;
}

//______________________________________________________________________________
void HtmlSummary::Build()
{
   // Build the summary.

   MakeHeader();
   for (int i=0;i<fNTables;i++) {
      GetTable(i)->Build();
      fHtml += GetTable(i)->Html();
   }
   MakeFooter();
}

//______________________________________________________________________________
void HtmlSummary::MakeHeader()
{
   // Make HTML header.

   fHeader  = "<html><head><title>";
   fHeader += fTitle;
   fHeader += "</title></head><body>";
   fHeader += "<center><h2><font color=#2222ee><i>";
   fHeader += fTitle;
   fHeader += "</i></font></h2></center>";
   fHtml    = fHeader;
}

//______________________________________________________________________________
void HtmlSummary::MakeFooter()
{
   // Make HTML footer.

   fFooter  = "<br><p><br><center><strong><font size=2 color=#2222ee>";
   fFooter += "Example of using Html widget to display tabular data";
   fFooter += "<br>";
   fFooter += "(c) 2007-2010 Bertrand Bellenot";
   fFooter += "</font></strong></center></body></html>";
   fHtml   += fFooter;
}

//==============================================================================

//______________________________________________________________________________
void update_html_summary()
{
   // Update summary of current event.

   TEveElement::List_i i;
   TEveElement::List_i j;
   Int_t k;
   TEveElement *el;
   HtmlObjTable *table;
   TEveEventManager *mgr = gEve ? gEve->GetCurrentEvent() : 0;
   if (mgr) {
      fgHtmlSummary->Clear("D");
      for (i=mgr->BeginChildren(); i!=mgr->EndChildren(); ++i) {
         el = ((TEveElement*)(*i));
         if (el->IsA() == TEvePointSet::Class()) {
            TEvePointSet *ps = (TEvePointSet *)el;
            TString ename  = ps->GetElementName();
            TString etitle = ps->GetElementTitle();
            if (ename.First('\'') != kNPOS)
               ename.Remove(ename.First('\''));
            etitle.Remove(0, 2);
            Int_t nel = atoi(etitle.Data());
            table = fgHtmlSummary->AddTable(ename, 0, nel);
         }
         else if (el->IsA() == TEveTrackList::Class()) {
            TEveTrackList *tracks = (TEveTrackList *)el;
            TString ename  = tracks->GetElementName();
            if (ename.First('\'') != kNPOS)
               ename.Remove(ename.First('\''));
            table = fgHtmlSummary->AddTable(ename.Data(), 5,
                     tracks->NumChildren(), kTRUE, "first");
            table->SetLabel(0, "Momentum");
            table->SetLabel(1, "P_t");
            table->SetLabel(2, "Phi");
            table->SetLabel(3, "Theta");
            table->SetLabel(4, "Eta");
            k=0;
            for (j=tracks->BeginChildren(); j!=tracks->EndChildren(); ++j) {
               Float_t p     = ((TEveTrack*)(*j))->GetMomentum().Mag();
               table->SetValue(0, k, p);
               Float_t pt    = ((TEveTrack*)(*j))->GetMomentum().Perp();
               table->SetValue(1, k, pt);
               Float_t phi   = ((TEveTrack*)(*j))->GetMomentum().Phi();
               table->SetValue(2, k, phi);
               Float_t theta = ((TEveTrack*)(*j))->GetMomentum().Theta();
               table->SetValue(3, k, theta);
               Float_t eta   = ((TEveTrack*)(*j))->GetMomentum().Eta();
               table->SetValue(4, k, eta);
               ++k;
            }
         }
      }
      fgHtmlSummary->Build();
      fgHtml->Clear();
      fgHtml->ParseText((char*)fgHtmlSummary->Html().Data());
      fgHtml->Layout();
   }
}
