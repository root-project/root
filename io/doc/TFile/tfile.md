\page tfile Format of the root (first) directory record

Format of the root (first) directory record in release 6.22.06.  It is never compressed.

This directory record differs from subdirectories (see \ref tdirectory) in the additional
Name and Title at the beginning of the DATA (after the TKey).

If the SeekKeys or SeekPdir in the TKey are located past the 32 bit file limit (> 2000000000),
then these fields will be 8 instead of 4 bytes and 1000 is added to the TKey Version.

If the SeekDir, SeekParent, or SeekKeys in the TDirectory header are past the 32 bit file limit,
then these fields will be 8 instead of 4 bytes and 1000 is added to the TDirectory Version.

<div style="background-color: lightgrey; font-size: 0.9vw;"><pre>
 ----------TKey---------------
  byte 0->3           Nbytes    = Number of bytes compressed record (TKey+data)          TKey::fNbytes
       4->5           Version   = TKey class version identifier                          TKey::fVersion
       6->9           ObjLen    = Number of bytes of uncompressed data                   TKey::fObjLen
      10->13          Datime    = Date and time when record was written to file          TKey::fDatime
                       | (year-1995)<<26|month<<22|day<<17|hour<<12|minute<<6|second
      14->15          KeyLen    = Number of bytes in key structure (TKey)                TKey::fKeyLen
      16->17          Cycle     = Cycle of key                                           TKey::fCycle
      18->21 [18->25] SeekKey   = Byte offset of record itself (consistency check) (64)  TKey::fSeekKey
      22->25 [26->33] SeekPdir  = Byte offset of parent directory record (0)             TKey::fSeekPdir
      26->26 [34->34] lname     = Number of bytes in the class name (5)                  TKey::fClassName
      27->.. [35->..] ClassName = Object Class Name ("TFile")                            TKey::fClassName
       0->0  lname     = Number of bytes in the object name                              TNamed::fName
       1->..          Name      = lName bytes with the name of the object <file name>    TNamed::fName
       0->0           lTitle    = Number of bytes in the object title                    TNamed::fTitle
       1->..          Title     = lTitle bytes with the title of the object <file title> TNamed::fTitle
 --------DATA-----------------
       0->0           lname     = Number of bytes in the TFile name                      TNamed::fName
       1->.. Name      = lName bytes with the name of the TFile <file name>              TNamed::fName
       0->0           lTitle    = Number of bytes in the TFile title                     TNamed::fTitle
       1->..          Title     = lTitle bytes with the title of the TFile <file title>  TNamed::fTitle
       0->1           Version   = TDirectory class version identifier                    TDirectory::Class_Version()
       2->5           DatimeC   = Date and time when directory was created               TDirectory::fDatimeC
                       | (year-1995)<<26|month<<22|day<<17|hour<<12|minute<<6|second
       6->9           DatimeM   = Date and time when directory was last modified         TDirectory::fDatimeM
                       | (year-1995)<<26|month<<22|day<<17|hour<<12|minute<<6|second
      10->13          NbytesKeys= Number of bytes in the associated KeysList record      TDirectory::fNbyteskeys
      14->17          NbytesName= Number of bytes in TKey+TNamed at creation             TDirectory::fNbytesName
      18->21 [18->25] SeekDir   = Byte offset of directory record in file (64)           TDirectory::fSeekDir
      22->25 [26->33] SeekParent= Byte offset of parent directory record in file (0)     TDirectory::fSeekParent
      26->29 [34->41] SeekKeys  = Byte offset of associated KeysList record in file      TDirectory::fSeekKeys
</pre></div>
