// @(#)root/treeplayer:$Id$
// Author: Axel Naumann, 2011-09-21

/*************************************************************************
 * Copyright (C) 1995-2013, Rene Brun and Fons Rademakers and al.        *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TTreeReader.h"

#include "TChain.h"
#include "TDirectory.h"
#include "TTreeReaderValue.h"

////////////////////////////////////////////////////////////////////////////////
/*BEGIN_HTML
 TTreeReader is a simple, robust and fast interface to read values from a TTree,
 TChain or TNtuple. It uses TTreeReaderValue&lt;T&gt; and
 TTreeReaderArray&lt;T&gt; to access the data. Example code can be found in
 tutorials/tree/hsimpleReader.C and tutorials/trees/h1analysisTreeReader.h and
 tutorials/trees/h1analysisTreeReader.C for a TSelector.
 Roottest contains an
 <a href="http://root.cern.ch/gitweb?p=roottest.git;a=tree;f=root/tree/reader;hb=HEAD">example</a>
 showing the full power.
 Usually it is as simple as this:
<div class="code">
<table><tr><td>
<pre class="_listing">
<span class="cpp">#include &lt;<a href="TFile.h">TFile.h</a>&gt;</span>
<span class="cpp">#include &lt;<a href="TTreeReader.h">TTreeReader.h</a>&gt;</span>
<span class="cpp">#include &lt;<a href="TTreeReaderValue.h">TTreeReaderValue.h</a>&gt;</span>
<span class="cpp">#include &lt;<a href="TTreeReaderArray.h">TTreeReaderArray.h</a>&gt;</span>
&nbsp;
<span class="cpp">#include "TriggerInfo.h"</span>
<span class="cpp">#include "Muon.h"</span>
<span class="cpp">#include "Tau.h"</span>
&nbsp;
<span class="cpp">#include &lt;vector&gt;</span>
<span class="cpp">#include &lt;iostream&gt;</span>
&nbsp;
<span class="keyword">bool</span> CheckValue(<a href="ROOT__TTreeReaderValueBase.html">ROOT::TTreeReaderValueBase</a>* value) {
   <span class="keyword">if</span> (value-&gt;GetSetupStatus() &lt; 0) {
      std::cerr &lt;&lt; <span class="string">"Error "</span> &lt;&lt; value-&gt;GetSetupStatus()
                &lt;&lt; <span class="string">"setting up reader for "</span> &lt;&lt; value-&gt;GetBranchName() &lt;&lt; '\n';
      <span class="keyword">return</span> <span class="keyword">false</span>;
   }
   <span class="keyword">return</span> <span class="keyword">true</span>;
}
&nbsp;
&nbsp;
<span class="comment">// Analyze the tree <span class="string">"MyTree"</span> in the file passed into the function.</span>
<span class="comment">// Returns false in case of errors.</span>
<span class="keyword">bool</span> analyze(<a href="TFile.html">TFile</a>* file) {
   <span class="comment">// Create a <a href="TTreeReader.html">TTreeReader</a> named <span class="string">"MyTree"</span> from the given <a href="TDirectory.html">TDirectory</a>.</span>
   <span class="comment">// The <a href="TTreeReader.html">TTreeReader</a> gives access to the <a href="TTree.html">TTree</a> to the TTreeReaderValue and</span>
   <span class="comment">// TTreeReaderArray objects. It knows the current entry number and knows</span>
   <span class="comment">// how to iterate through the <a href="TTree.html">TTree</a>.</span>
   <a href="TTreeReader.html">TTreeReader</a> reader(<span class="string">"MyTree"</span>, file);
&nbsp;
   <span class="comment">// <a href="TObject.html#TObject:Read" title="Int_t TObject::Read(const char* name)">Read</a> a single <a href="ListOfTypes.html#float">float</a> value in each tree entries:</span>
   TTreeReaderValue&lt;<span class="keyword">float</span>&gt; weight(reader, <span class="string">"event.weight"</span>);
   <span class="keyword">if</span> (!CheckValue(weight)) <span class="keyword">return</span> <span class="keyword">false</span>;
&nbsp;
   <span class="comment">// <a href="TObject.html#TObject:Read" title="Int_t TObject::Read(const char* name)">Read</a> a TriggerInfo object from the tree entries:</span>
   TTreeReaderValue&lt;TriggerInfo&gt; triggerInfo(reader, <span class="string">"triggerInfo"</span>);
   <span class="keyword">if</span> (!CheckValue(triggerInfo)) <span class="keyword">return</span> <span class="keyword">false</span>;
&nbsp;
   <span class="comment">// <a href="TObject.html#TObject:Read" title="Int_t TObject::Read(const char* name)">Read</a> a vector of Muon objects from the tree entries:</span>
   TTreeReaderValue&lt;std::vector&lt;Muon&gt;&gt; muons(reader, <span class="string">"muons"</span>);
   <span class="keyword">if</span> (!CheckValue(muons)) <span class="keyword">return</span> <span class="keyword">false</span>;
&nbsp;
   <span class="comment">// <a href="TObject.html#TObject:Read" title="Int_t TObject::Read(const char* name)">Read</a> the pT for all jets in the tree entry:</span>
   TTreeReaderArray&lt;<span class="keyword">double</span>&gt; jetPt(reader, <span class="string">"jets.pT"</span>);
   <span class="keyword">if</span> (!CheckValue(jetPt)) <span class="keyword">return</span> <span class="keyword">false</span>;
&nbsp;
   <span class="comment">// <a href="TObject.html#TObject:Read" title="Int_t TObject::Read(const char* name)">Read</a> the taus in the tree entry:</span>
   TTreeReaderArray&lt;Tau&gt; taus(reader, <span class="string">"taus"</span>);
   <span class="keyword">if</span> (!CheckValue(taus)) <span class="keyword">return</span> <span class="keyword">false</span>;
&nbsp;
&nbsp;
   <span class="comment">// Now iterate through the <a href="TTree.html">TTree</a> entries and fill a histogram.</span>
&nbsp;
   <a href="TH1.html">TH1</a>* hist = <span class="keyword">new</span> <a href="TH1F.html">TH1F</a>(<span class="string">"hist"</span>, <span class="string">"TTreeReader example histogram"</span>, 10, 0., 100.);
&nbsp;
   <span class="keyword">while</span> (reader.Next()) {
&nbsp;
      <span class="keyword">if</span> (reader.GetEntryStatus() == <a href="TTreeReader.html#TTreeReader:kEntryValid" title="TTreeReader::EEntryStatus TTreeReader::kEntryValid">kEntryValid</a>) {
         std::cout &lt;&lt; <span class="string">"Loaded entry "</span> &lt;&lt; reader.GetCurrentEntry() &lt;&lt; '\n';
      } <span class="keyword">else</span> {
         <span class="keyword">switch</span> (reader.GetEntryStatus()) {
         <a href="TTreeReader.html#TTreeReader:kEntryValid" title="TTreeReader::EEntryStatus TTreeReader::kEntryValid">kEntryValid</a>:
            <span class="comment">// Handled above.</span>
            <span class="keyword">break</span>;
         <a href="TTreeReader.html#TTreeReader:kEntryNotLoaded" title="TTreeReader::EEntryStatus TTreeReader::kEntryNotLoaded">kEntryNotLoaded</a>:
            std::cerr &lt;&lt; <span class="string">"Error: TTreeReader has not loaded any data yet!\n"</span>;
            <span class="keyword">break</span>;
         <a href="TTreeReader.html#TTreeReader:kEntryNoTree" title="TTreeReader::EEntryStatus TTreeReader::kEntryNoTree">kEntryNoTree</a>:
            std::cerr &lt;&lt; <span class="string">"Error: TTreeReader cannot find a tree names \"MyTree\"!\n"</span>;
            <span class="keyword">break</span>;
         <a href="TTreeReader.html#TTreeReader:kEntryNotFound" title="TTreeReader::EEntryStatus TTreeReader::kEntryNotFound">kEntryNotFound</a>:
            <span class="comment">// Can't really happen as <a href="TTreeReader.html">TTreeReader</a>::<a href="TTreeReader.html#TTreeReader:Next" title="Bool_t TTreeReader::Next()">Next</a>() knows when to stop.</span>
            std::cerr &lt;&lt; <span class="string">"Error: The entry number doe not exist\n"</span>;
            <span class="keyword">break</span>;
         <a href="TTreeReader.html#TTreeReader:kEntryChainSetupError" title="TTreeReader::EEntryStatus TTreeReader::kEntryChainSetupError">kEntryChainSetupError</a>:
            std::cerr &lt;&lt; <span class="string">"Error: TTreeReader cannot access a chain element, e.g. file without the tree\n"</span>;
            <span class="keyword">break</span>;
         <a href="TTreeReader.html#TTreeReader:kEntryChainFileError" title="TTreeReader::EEntryStatus TTreeReader::kEntryChainFileError">kEntryChainFileError</a>:
            std::cerr &lt;&lt; <span class="string">"Error: TTreeReader cannot open a chain element, e.g. missing file\n"</span>;
            <span class="keyword">break</span>;
         <a href="TTreeReader.html#TTreeReader:kEntryDictionaryError" title="TTreeReader::EEntryStatus TTreeReader::kEntryDictionaryError">kEntryDictionaryError</a>:
            std::cerr &lt;&lt; <span class="string">"Error: TTreeReader cannot find the dictionary for some data\n"</span>;
            <span class="keyword">break</span>;
         }
         <span class="keyword">return</span> <span class="keyword">false</span>;
      }
&nbsp;
      <span class="comment">// Access the TriggerInfo object as if it's a pointer.</span>
      <span class="keyword">if</span> (!triggerInfo-&gt;hasMuonL1())
         <span class="keyword">continue</span>;
&nbsp;
      <span class="comment">// Ditto for the vector&lt;Muon&gt;.</span>
      <span class="keyword">if</span> (!muons-&gt;size())
         <span class="keyword">continue</span>;
&nbsp;
      <span class="comment">// Access the jetPt as an array, whether the <a href="TTree.html">TTree</a> stores this as</span>
      <span class="comment">// a std::vector, std::list, <a href="TClonesArray.html">TClonesArray</a> or Jet* C-style array, with</span>
      <span class="comment">// fixed or variable array size.</span>
      <span class="keyword">if</span> (jetPt.<a href="TCollection.html#TCollection:GetSize" title="Int_t TCollection::GetSize() const">GetSize</a>() &lt; 2 || jetPt[0] &lt; 100)
         <span class="keyword">continue</span>;
&nbsp;
      <span class="comment">// Access the array of taus.</span>
      <span class="keyword">if</span> (!taus.<a href="TObjArray.html#TObjArray:IsEmpty" title="Bool_t TObjArray::IsEmpty() const">IsEmpty</a>()) {
         <span class="keyword">float</span> currentWeight = *weight;
         <span class="keyword">for</span> (<span class="keyword">int</span> iTau = 0, nTau = taus.<a href="TCollection.html#TCollection:GetSize" title="Int_t TCollection::GetSize() const">GetSize</a>(); iTau &lt; nTau; ++iTau) {
            <span class="comment">// Access a <a href="ListOfTypes.html#float">float</a> value - need to dereference as TTreeReaderValue</span>
            <span class="comment">// behaves like an iterator</span>
            hist-&gt;Fill(taus[iTau].eta(), currentWeight);
         }
      }
   }
}
</pre></td></tr></table></div>

<div class="clear">
</div>

&nbsp;

<p>A simpler analysis example - the one from the tutorials - can be found below, in
the Source tab: it histograms a function of the px and py branches:</p>
END_HTML

BEGIN_MACRO(source)
../../../tutorials/tree/hsimpleReader.C
END_MACRO */
////////////////////////////////////////////////////////////////////////////////

ClassImp(TTreeReader)

//______________________________________________________________________________
TTreeReader::TTreeReader(TTree* tree):
   fTree(tree),
   fDirectory(0),
   fEntryStatus(kEntryNotLoaded),
   fDirector(0)
{
   // Access data from tree.
   Initialize();
}

//______________________________________________________________________________
TTreeReader::TTreeReader(const char* keyname, TDirectory* dir /*= NULL*/):
   fTree(0),
   fDirectory(dir),
   fEntryStatus(kEntryNotLoaded),
   fDirector(0)
{
   // Access data from the tree called keyname in the directory (e.g. TFile)
   // dir, or the current directory if dir is NULL. If keyname cannot be
   // found, or if it is not a TTree, IsZombie() will return true.
   if (!fDirectory) fDirectory = gDirectory;
   fDirectory->GetObject(keyname, fTree);
   Initialize();
}

//______________________________________________________________________________
TTreeReader::~TTreeReader()
{
   // Tell all value readers that the tree reader does not exist anymore.
   for (std::deque<ROOT::TTreeReaderValueBase*>::const_iterator
           i = fValues.begin(), e = fValues.end(); i != e; ++i) {
      (*i)->MarkTreeReaderUnavailable();
   }
   delete fDirector;
   fProxies.SetOwner();
}

//______________________________________________________________________________
void TTreeReader::Initialize()
{
   // Initialization of the director.
   if (!fTree) {
      MakeZombie();
      fEntryStatus = kEntryNoTree;
   } else {
      fDirector = new ROOT::TBranchProxyDirector(fTree, -1);
   }
}

//______________________________________________________________________________
Long64_t TTreeReader::GetCurrentEntry() const {
   //Returns the index of the current entry being read

   if (!fDirector) return 0;
   Long64_t currentTreeEntry = fDirector->GetReadEntry();
   if (fTree->IsA() == TChain::Class() && currentTreeEntry >= 0) {
      return ((TChain*)fTree)->GetChainEntryNumber(currentTreeEntry);
   }
   return currentTreeEntry;
}

//______________________________________________________________________________
TTreeReader::EEntryStatus TTreeReader::SetEntryBase(Long64_t entry, Bool_t local)
{
   // Load an entry into the tree, return the status of the read.
   // For chains, entry is the global (i.e. not tree-local) entry number.

   if (!fTree) {
      fEntryStatus = kEntryNoTree;
      return fEntryStatus;
   }

   TTree* prevTree = fDirector->GetTree();

   int loadResult;
   if (!local){
      Int_t treeNumInChain = fTree->GetTreeNumber();

      loadResult = fTree->LoadTree(entry);

      if (loadResult == -2) {
         fEntryStatus = kEntryNotFound;
         return fEntryStatus;
      }

      Int_t currentTreeNumInChain = fTree->GetTreeNumber();
      if (treeNumInChain != currentTreeNumInChain) {
            fDirector->SetTree(fTree->GetTree());
      }
   }
   else {
      loadResult = entry;
   }
   if (!prevTree || fDirector->GetReadEntry() == -1) {
      // Tell readers we now have a tree
      for (std::deque<ROOT::TTreeReaderValueBase*>::const_iterator
              i = fValues.begin(); i != fValues.end(); ++i) { // Iterator end changes when parameterized arrays are read
         (*i)->CreateProxy();

         if (!(*i)->GetProxy()){
            fEntryStatus = kEntryDictionaryError;
            return fEntryStatus;
         }
      }
   }
   fDirector->SetReadEntry(loadResult);
   fEntryStatus = kEntryValid;
   return fEntryStatus;
}

//______________________________________________________________________________
void TTreeReader::SetTree(TTree* tree)
{
   // Set (or update) the which tree to reader from. tree can be
   // a TTree or a TChain.
   fTree = tree;
   if (fTree) {
      ResetBit(kZombie);
      if (fTree->InheritsFrom(TChain::Class())) {
         SetBit(kBitIsChain);
      }
   }

   if (!fDirector) {
      Initialize();
   }
   else {
      fDirector->SetTree(fTree);
      fDirector->SetReadEntry(-1);
   }
}

//______________________________________________________________________________
void TTreeReader::RegisterValueReader(ROOT::TTreeReaderValueBase* reader)
{
   // Add a value reader for this tree.
   fValues.push_back(reader);
}

//______________________________________________________________________________
void TTreeReader::DeregisterValueReader(ROOT::TTreeReaderValueBase* reader)
{
   // Remove a value reader for this tree.
   std::deque<ROOT::TTreeReaderValueBase*>::iterator iReader
      = std::find(fValues.begin(), fValues.end(), reader);
   if (iReader == fValues.end()) {
      Error("DeregisterValueReader", "Cannot find reader of type %s for branch %s", reader->GetDerivedTypeName(), reader->fBranchName.Data());
      return;
   }
   fValues.erase(iReader);
}
