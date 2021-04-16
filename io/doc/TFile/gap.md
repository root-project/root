\page gap Free segment in middle of file

  A gap (free segment in middle of file) has the following format.

<div style="background-color: lightgrey; font-size: small;"><pre>
 ------------------------
  byte 0->3  Nbytes    = Negative of number of bytes in gap
       4->.. irrelevant
</pre></div>
