\page trefarray Format of the DATA for a TRefArray object

Here is the format of the DATA for a TRefArray object in Release 3.02.06.

<div style="background-color: lightgrey; font-size: small;"><pre>
--------
       0->3  ByteCount = Number of remaining bytes in TRefArray object (uncompressed)
                       |   OR'd with kByteCountMask (0x40000000)
       4->5  Version   = Version of TRefArray Class
       6->15           = TObject object (Base class of TRefArray) (see \ref tobject).
                       |   Will be two bytes longer (6->17) if TRefArray object is
                       |   itself referenced (unlikely).
      16->.. fName     = Number of bytes in name of TRefArray object, followed by the
                       |   name itself.  (TCollection::fName). Currently, TRefArrays
                       |   are not named, so this is a single byte containing 0.
       0->3  nObjects  | Number of object references (fUIDs) in this TRefArray.
       4->7  fLowerBound= Lower bound of array.  Typically 0.
       8->9  pidf  = An identifier of the TProcessID record for the process that wrote the
                       | referenced objects. This identifier is an unsigned short.  The relevant
                       | record has a name that is the string "ProcessID" concatenated with the
                       | ASCII decimal representation of "pidf" (no leading zeros).
                       | 0 is a valid pidf.
      10->.. fUIDs     = Sequentially, object Unique ID's.
                       | Each Unique ID is a four byte unsigned integer.
                       | If non-zero, it matches the Unique ID in the referenced
                       | object.  If zero, it is an unused element in the array.
                       | The fUIDs are written out only up to the last used element,
                       | so the last fUID will always be non-zero.
</pre></div>
