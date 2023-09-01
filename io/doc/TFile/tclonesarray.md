\page tclonesarray Format of the DATA for a TClonesArray object

Here is the format (release 3.02.06)  of the DATA for a TClonesArray object in a ROOTIO file.

<div style="background-color: lightgrey; font-size: 0.9vw;"><pre>
      0->3  ByteCount = Number of remaining bytes in TClonesArray object (uncompressed)
                      |   OR'd with kByteCountMask (0x40000000)
      4->.. ClassInfo = Information about TClonesArray class
                      | If this is the first occurrence of a TClonesArray object in the record
                      |  4->7  -1        = New class tag (constant kNewClassTag = 0xffffffff)
                      |  8->17 Classname = Object Class Name "TClonesArray" (null terminated)
                      | Otherwise
                      |  4->7 clIdx      = Byte offset of new class tag in record, plus 2.
                      | OR'd with kClassMask (0x80000000)
      0->3  ByteCount = Number of remaining bytes in TClonesArray object (uncompressed)
                      |   OR'd with kByteCountMask (0x40000000)
      4->5  Version   = Version of TClonesArray Class
      6->15           = TObject object (a base class of TClonesArray) (see \ref tobject).
                      |   Would be two bytes longer (6->17) if object were referenced.
     16->.. fName     = Number of bytes in name of TClonesArray object, followed by the
                      |   name itself.  (TCollection::fName).  This name will be the
                      |   class name of the cloned object, appended with an 's'
                      |   (e.g. "TXxxs")
      0->..           = Number of bytes in name and version of the cloned class, followed
                      |   by the name and version themselves (e.g. "TXxx;1")
      0->3  nObjects  = Number of objects in clones array.
      4->7  fLowerBound= Lower bound of clones array.
      8->.. objects   = Sequentially, objects in the clones array.  However, the data
                      |   ordering depends on whether or not kBypassStreamer (0x1000) is
                      |   set in TObject::fBits.   By default, it is set.  If it is not set,
                      |   the objects are streamed sequentially using the streamer of the
                      |   cloned class (e.g. TXxx::Streamer()).
                      |
                      |   If it is set, the cloned class is split into its base classes and
                      |   persistent data members, and those streamers are used.  So, if the
                      |   base classes and persistent data members of class TXxx are TXxxbase,
                      |   TXxxdata0, TXxxdata1, etc.,  all the TXxxbase data from the entire
                      |   clones array is streamed first, followed by all the TXxxdata0 data,
                      |   etc.  This breakdown is not recursive, in that the member objects
                      |   are not again split.
 -End TClonesArray object
</pre></div>