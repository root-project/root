The VDT mathematical library
============================

What is VDT?
------------

**VDT is a library of mathematical functions**, implemented in double and single
precision. The implementation is **fast** and with the aid of modern compilers
(e.g. gcc 4.7) vectorisable.

VDT exploits also Pade polynomials. A lot of ideas were inspired by the **cephes
math library** (by Stephen L. Moshier, moshier@na-net.ornl.gov) as well as
portions of actual code. The Cephes library can be found here:
http://www.netlib.org/cephes

Implemented functions
 * log
 * exp
 * sincos
 * sin
 * cos
 * tan
 * asin
 * acos
 * atan
 * atan2
 * inverse sqrt
 * inverse (faster than division, based on isqrt)


Copyright Danilo Piparo, Vincenzo Innocente, Thomas Hauth (CERN) 2012-14

VDT is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser Public License for more details.

You should have received a copy of the GNU Lesser Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

