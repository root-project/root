\page tref Format of the DATA for a TRef object

Here is the format of the DATA for a TRef object in Release 3.02.06.

<div style="background-color: lightgrey; font-size: small;"><pre>
--------
  0->1  Version   = Version of TObject Class (base class of TRef)
  2->5  fUniqueID = Unique ID of referenced object.  Typically, every referenced
                       | object has an ID that is a positive integer set to a counter
        | of the number of referenced objects in the file, beginning at 1.
                       | fUniqueID in the TRef object matches fUniqueID in the
                       | referenced object.
  6->9  fBits     = A 32 bit mask containing status bits for the TRef object.
                       | The bits relevant to ROOTIO are:
        | 0x00000008 - Other objects may need to be deleted when this one is.
        | 0x00000010 - Object is referenced by pointer to persistent object.
        | 0x01000000 - Object is on Heap.
        | 0x02000000 - Object has not been deleted.
 10->11 pidf  = An identifier of the TProcessID record for the process that wrote the
                       | referenced object. This identifier is an unsigned short.  The relevant
                       | record has a name that is the string "ProcessID" concatenated with the
                       | ASCII decimal representation of "pidf" (no leading zeros).
                       | 0 is a valid pidf.
-------
</pre></div>