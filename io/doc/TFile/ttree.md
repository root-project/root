\page ttree Streamer information for TTree related classes


Here is the streamer information for TTree related classes in release 3.02.06:
(For the explanation of the meaning of the type, see "fType" in \ref streamerinfo.)

<div style="background-color: lightgrey; font-size: 0.9vw;"><pre>
----------------------------------------------------
StreamerInfo for class: TTree, version=6
  BASE          TNamed          offset=  0 type=67 The basis for a named object (name, title)
  BASE          TAttLine        offset=  0 type= 0 Line attributes
  BASE          TAttFill        offset=  0 type= 0 Fill area attributes
  BASE          TAttMarker      offset=  0 type= 0 Marker attributes
  Stat_t        fEntries        offset=  0 type= 8 Number of entries
  Stat_t        fTotBytes       offset=  0 type= 8 Total number of bytes in all branches before compression
  Stat_t        fZipBytes       offset=  0 type= 8 Total number of bytes in all branches after compression
  Stat_t        fSavedBytes     offset=  0 type= 8 Number of autosaved bytes
  Int_t         fTimerInterval  offset=  0 type= 3 Timer interval in milliseconds
  Int_t         fScanField      offset=  0 type= 3 Number of runs before prompting in Scan
  Int_t         fUpdate         offset=  0 type= 3 Update frequency for EntryLoop
  Int_t         fMaxEntryLoop   offset=  0 type= 3 Maximum number of entries to process
  Int_t         fMaxVirtualSize offset=  0 type= 3 Maximum total size of buffers kept in memory
  Int_t         fAutoSave       offset=  0 type= 3 Autosave tree when fAutoSave bytes produced
  Int_t         fEstimate       offset=  0 type= 3 Number of entries to estimate histogram limits
  TObjArray     fBranches       offset=  0 type=61 List of Branches
  TObjArray     fLeaves         offset=  0 type=61 Direct pointers to individual branch leaves
  TArrayD       fIndexValues    offset=  0 type=62 Sorted index values
  TArrayI       fIndex          offset=  0 type=62 Index of sorted values
  TList*        fFriends        offset=  0 type=64 pointer to list of friend elements

StreamerInfo for class: TAttLine, version=1
  Color_t       fLineColor      offset=  0 type= 2 line color
  Style_t       fLineStyle      offset=  0 type= 2 line style
  Width_t       fLineWidth      offset=  0 type= 2 line width

StreamerInfo for class: TAttFill, version=1
  Color_t       fFillColor      offset=  0 type= 2 fill area color
  Style_t       fFillStyle      offset=  0 type= 2 fill area style

StreamerInfo for class: TAttMarker, version=1
  Color_t       fMarkerColor    offset=  0 type= 2 Marker color index
  Style_t       fMarkerStyle    offset=  0 type= 2 Marker style
  Size_t        fMarkerSize     offset=  0 type= 5 Marker size

StreamerInfo for class: TBranch, version=7
  BASE          TNamed          offset=  0 type=67 The basis for a named object (name, title)
  Int_t         fCompress       offset=  0 type= 3 (=1 branch is compressed, 0 otherwise)
  Int_t         fBasketSize     offset=  0 type= 3 Initial Size of  Basket Buffer
  Int_t         fEntryOffsetLen offset=  0 type= 3 Initial Length of fEntryOffset table in the basket buffers
  Int_t         fWriteBasket    offset=  0 type= 3 Last basket number written
  Int_t         fEntryNumber    offset=  0 type= 3 Current entry number (last one filled in this branch)
  Int_t         fOffset         offset=  0 type= 3 Offset of this branch
  Int_t         fMaxBaskets     offset=  0 type= 6 Maximum number of Baskets so far
  Int_t         fSplitLevel     offset=  0 type= 3 Branch split level
  Stat_t        fEntries        offset=  0 type= 8 Number of entries
  Stat_t        fTotBytes       offset=  0 type= 8 Total number of bytes in all leaves before compression
  Stat_t        fZipBytes       offset=  0 type= 8 Total number of bytes in all leaves after compression
  TObjArray     fBranches       offset=  0 type=61 -> List of Branches of this branch
  TObjArray     fLeaves         offset=  0 type=61 -> List of leaves of this branch
  TObjArray     fBaskets        offset=  0 type=61 -> List of baskets of this branch
  Int_t*        fBasketBytes    offset=  0 type=43 [fMaxBaskets] Length of baskets on file
  Int_t*        fBasketEntry    offset=  0 type=43 [fMaxBaskets] Table of first entry in eack basket
  Seek_t*       fBasketSeek     offset=  0 type=43 [fMaxBaskets] Addresses of baskets on file
  TString       fFileName       offset=  0 type=65 Name of file where buffers are stored ("" if in same file as Tree header)

StreamerInfo for class: TBranchElement, version=7
  BASE          TBranch         offset=  0 type= 0 Branch descriptor
  TString       fClassName      offset=  0 type=65 Class name of referenced object
  TString       fParentName     offset=  0 type=65 Name of parent class
  TString       fClonesName     offset=  0 type=65 Name of class in TClonesArray (if any)
  Int_t         fClassVersion   offset=  0 type= 3 Version number of class
  Int_t         fID             offset=  0 type= 3 element serial number in fInfo
  Int_t         fType           offset=  0 type= 3 branch type
  Int_t         fStreamerType   offset=  0 type= 3 branch streamer type
  Int_t         fMaximum        offset=  0 type= 3 Maximum entries for a TClonesArray or variable array
  TBranchElement*fBranchCount    offset=  0 type=64 pointer to primary branchcount branch
  TBranchElement*fBranchCount2   offset=  0 type=64 pointer to secondary branchcount branch

StreamerInfo for class: TLeaf, version=2
  BASE          TNamed          offset=  0 type=67 The basis for a named object (name, title)
  Int_t         fLen            offset=  0 type= 3 Number of fixed length elements
  Int_t         fLenType        offset=  0 type= 3 Number of bytes for this data type
  Int_t         fOffset         offset=  0 type= 3 Offset in ClonesArray object (if one)
  Bool_t        fIsRange        offset=  0 type=11 (=kTRUE if leaf has a range, kFALSE otherwise)
  Bool_t        fIsUnsigned     offset=  0 type=11 (=kTRUE if unsigned, kFALSE otherwise)
  TLeaf*        fLeafCount      offset=  0 type=64 Pointer to Leaf count if variable length

StreamerInfo for class: TLeafElement, version=1
  BASE          TLeaf           offset=  0 type= 0 Leaf: description of a Branch data type
  Int_t         fID             offset=  0 type= 3 element serial number in fInfo
  Int_t         fType           offset=  0 type= 3 leaf type
</pre></div>
