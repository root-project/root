\page header File header format

 Here is the file header format as of release 6.22.06.  It is never compressed.

 If END, SeekFree, or SeekInfo are located past the 32 bit file limit (> 2000000000)
 then these fields will be 8 instead of 4 bytes and 1000000 is added to the file format version.

<div style="background-color: lightgrey; font-size: 0.9vw;"><pre>
 -----------------------------------
 byte  0->3  "root"               = Identifies this file as a ROOT file.
       4->7  Version              = File format version                         TFile::fVersion
                                  |  (10000*major+100*minor+cycle (e.g. 62206 for 6.22.06))
       8->11          BEGIN       = Byte offset of first data record (100)      TFile::fBEGIN
      12->15 [12->19] END         = Pointer to first free word at the EOF       TFile::fEND
                                  | (will be == to file size in bytes)
      16->19 [20->27] SeekFree    = Byte offset of FreeSegments record          TFile::fSeekFree
      20->23 [28->31] NbytesFree  = Number of bytes in FreeSegments record      TFile::fNBytesFree
      24->27 [32->35] nfree       = Number of free data records
      28->31 [36->39] NbytesName  = Number of bytes in TKey+TNamed for TFile at creation TDirectory::fNbytesName
      32->32 [40->40] Units       = Number of bytes for file pointers (4)       TFile::fUnits
      33->36 [41->44] Compress    = Zip compression level (i.e. 0-9)            TFile::fCompress
      37->40 [45->52] SeekInfo    = Byte offset of StreamerInfo record          TFile::fSeekInfo
      41->44 [53->56] NbytesInfo  = Number of bytes in StreamerInfo record      TFile::fNbytesInfo
      45->46 [57->58] UUID vers   = TUUID class version identifier              TUUID::Class_Version()
      47->62 [59->74] UUID        = Universally Unique Identifier               TUUID::fTimeLow through fNode[6]
      63->99 [75->99] Extra space to allow END, SeekFree, or SeekInfo to become 64 bit without moving this header
</pre></div>

 Here is the file header format as of release 3.02.06.  It is never compressed.

<div style="background-color: lightgrey; font-size: 0.9vw;"><pre>
 -----------------------------------
 byte  0->3  "root"      = Identifies this file as a ROOT file.
       4->7  Version     = File format version                        TFile::fVersion
                         |  (10000*major+100*minor+cycle (e.g. 30203 for 3.2.3))
       8->11 BEGIN       = Byte offset of first data record (64)      TFile::fBEGIN
      12->15 END         = Pointer to first free word at the EOF      TFile::fEND
                         | (will be == to file size in bytes)
      16->19 SeekFree    = Byte offset of FreeSegments record         TFile::fSeekFree
      20->23 NbytesFree  = Number of bytes in FreeSegments record     TFile::fNBytesFree
      24->27 nfree       = Number of free data records
      28->31 NbytesName  = Number of bytes in TKey+TNamed for TFile at creation TDirectory::fNbytesName
      32->32 Units       = Number of bytes for file pointers (4)      TFile::fUnits
      33->36 Compress    = Zip compression level (i.e. 0-9)           TFile::fCompress
      37->40 SeekInfo    = Byte offset of StreamerInfo record         TFile::fSeekInfo
      41->44 NbytesInfo  = Number of bytes in StreamerInfo record     TFile::fNbytesInfo
      45->63             = Unused??
</pre></div>