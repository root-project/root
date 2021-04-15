\page tdirectory Format of a TDirectory record

Format of a TDirectory record in release 6.22.06.  It is never compressed.

 If the SeekKeys or SeekPdir in the TKey are located past the 32 bit file limit (> 2000000000),
 then these fields will be 8 instead of 4 bytes and 1000 is added to the TKey Version.

 If the SeekDir, SeekParent, or SeekKeys in the TDirectory header are past the 32 bit file limit,
 then these fields will be 8 instead of 4 bytes and 1000 is added to the TDirectory Version.

<div style="background-color: lightgrey; font-size: small;"><pre>
 ----------TKey--------------
  byte 0->3           Nbytes    = Number of bytes in compressed record (Tkey+data)            TKey::fNbytes
       4->5           Version   = TKey class version identifier                               TKey::fVersion
       6->9           ObjLen    = Number of bytes of uncompressed data                        TKey::fObjLen
      10->13          Datime    = Date and time when record was written to file               TKey::fDatime
                                | (year-1995)<<26|month<<22|day<<17|hour<<12|minute<<6|second
      14->15          KeyLen    = Number of bytes in key structure (TKey)                     TKey::fKeyLen
      16->17          Cycle     = Cycle of key                                                TKey::fCycle
      18->21 [18->25] SeekKey   = Byte offset of record itself (consistency check)            TKey::fSeekKey
      22->25 [26->33] SeekPdir  = Byte offset of parent directory record                      TKey::fSeekPdir
      26->26 [33->33] lname     = Number of bytes in the class name (10)                      TKey::fClassName
      27->.. [34->..] ClassName = Object Class Name ("TDirectory")                            TKey::fClassName
       0->0           lname     = Number of bytes in the object name                          TNamed::fName
       1->..          Name      = lName bytes with the name of the object <directory name>    TNamed::fName
       0->0           lTitle    = Number of bytes in the object title                         TNamed::fTitle
       1->..          Title     = lTitle bytes with the title of the object <directory title> TNamed::fTitle
 --------DATA----------------
       0->1           Version   = TDirectory class version identifier                         TDirectory::Class_Version()
       2->5           DatimeC   = Date and time when directory was created                    TDirectory::fDatimeC
                                | (year-1995)<<26|month<<22|day<<17|hour<<12|minute<<6|second
       6->9           DatimeM   = Date and time when directory was last modified              TDirectory::fDatimeM
                                | (year-1995)<<26|month<<22|day<<17|hour<<12|minute<<6|second
      10->13          NbytesKeys= Number of bytes in the associated KeysList record           TDirectory::fNbyteskeys
      14->17          NbytesName= Number of bytes in TKey+TNamed at creation                  TDirectory::fNbytesName
      18->21 [18->25] SeekDir   = Byte offset of directory record in file                     TDirectory::fSeekDir
      22->25 [26->33] SeekParent= Byte offset of parent directory record in file              TDirectory::fSeekParent
      26->29 [34->41] SeekKeys  = Byte offset of associated KeysList record in file           TDirectory::fSeekKeys
      30->31 [42->43] UUID vers = TUUID class version identifier                              TUUID::Class_Version()
      32->47 [44->59] UUID      = Universally Unique Identifier                               TUUID::fTimeLow through fNode[6]
      48->59          Extra space to allow SeekKeys to become 64 bit without moving this header
</pre></div>

Format of a TDirectory record in release 3.02.06.  It is never compressed.

<div style="background-color: lightgrey; font-size: small;"><pre>
 ----------TKey--------------
  byte 0->3  Nbytes    = Number of bytes in compressed record (Tkey+data)            TKey::fNbytes
       4->5  Version   = TKey class version identifier                               TKey::fVersion
       6->9  ObjLen    = Number of bytes of uncompressed data                        TKey::fObjLen
      10->13 Datime    = Date and time when record was written to file               TKey::fDatime
                       | (year-1995)<<26|month<<22|day<<17|hour<<12|minute<<6|second
      14->15 KeyLen    = Number of bytes in key structure (TKey)                     TKey::fKeyLen
      16->17 Cycle     = Cycle of key                                                TKey::fCycle
      18->21 SeekKey   = Byte offset of record itself (consistency check)            TKey::fSeekKey
      22->25 SeekPdir  = Byte offset of parent directory record                      TKey::fSeekPdir
      26->26 lname     = Number of bytes in the class name (10)                      TKey::fClassName
      27->.. ClassName = Object Class Name ("TDirectory")                            TKey::fClassName
       0->0  lname     = Number of bytes in the object name                          TNamed::fName
       1->.. Name      = lName bytes with the name of the object <directory name>    TNamed::fName
       0->0  lTitle    = Number of bytes in the object title                         TNamed::fTitle
       1->.. Title     = lTitle bytes with the title of the object <directory title> TNamed::fTitle
 --------DATA----------------
       0->0  Modified  = True if directory has been modified                         TDirectory::fModified
       1->1  Writable = True if directory is writable                                TDirectory::fWriteable
       2->5  DatimeC   = Date and time when directory was created                    TDirectory::fDatimeC
                       | (year-1995)<<26|month<<22|day<<17|hour<<12|minute<<6|second
       6->9  DatimeM   = Date and time when directory was last modified              TDirectory::fDatimeM
                       | (year-1995)<<26|month<<22|day<<17|hour<<12|minute<<6|second
      10->13 NbytesKeys= Number of bytes in the associated KeysList record           TDirectory::fNbyteskeys
      14->17 NbytesName= Number of bytes in TKey+TNamed at creation                  TDirectory::fNbytesName
      18->21 SeekDir   = Byte offset of directory record in file                     TDirectory::fSeekDir
      22->25 SeekParent= Byte offset of parent directory record in file              TDirectory::fSeekParent
      26->29 SeekKeys  = Byte offset of associated KeysList record in file           TDirectory::fSeekKeys
</pre></div>
