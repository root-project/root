// @(#)root/mathcore:$Id$
// Author: Fernando Hueso-González   04/08/2021

/**

\class TRandomBinary

@ingroup Random

This class contains a generator of Pseudo Random Binary Sequences (<a
href="https://en.wikipedia.org/wiki/Pseudorandom_binary_sequence">PRBS</a>).
It should NOT be used for general-purpose random number generation or any
statistical study, see ::TRandom2 class instead.

The goal is to generate binary bit sequences with the same algorithm as the ones usually implemented
in electronic chips, so that the theoretically expected ones can be compared with the acquired sequences.

The main ingredients of a PRBS generator are a monic polynomial of maximum degree \f$n\f$, with coefficients
either 0 or 1, and a <a href="https://www.nayuki.io/page/galois-linear-feedback-shift-register">Galois</a>
linear-feedback shift register with a non-zero seed. When the monic polynomial exponents are chosen appropriately,
the period of the resulting bit sequence (0s and 1s) yields \f$2^n - 1\f$.

Other implementations can be found here:

- https://gist.github.com/mattbierner/d6d989bf26a7e54e7135
- https://root.cern/doc/master/civetweb_8c_source.html#l06030
- https://cryptography.fandom.com/wiki/Linear_feedback_shift_register
- https://www3.advantest.com/documents/11348/33b24c8a-c8cb-40b8-a2a7-37515ba4abc8
- https://www.reddit.com/r/askscience/comments/63a10q/for_prbs3_with_clock_input_on_each_gate_how_can/
- https://es.mathworks.com/help/serdes/ref/prbs.html
- https://metacpan.org/pod/Math::PRBS

*/

#include "TRandomBinary.h"

ClassImp(TRandomBinary);
