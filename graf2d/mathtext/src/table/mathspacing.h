// mathtext - A TeX/LaTeX compatible rendering library. Copyright (C)
// 2008-2012 Yue Shi Lai <ylai@users.sourceforge.net>
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public License
// as published by the Free Software Foundation; either version 2.1 of
// the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
// 02110-1301 USA

// See also Knuth (1986), p. 170.
static const int nvr = INT_MIN;
static const int spacing_table[64] = {
	// right ->  Ord    Op   Bin   Rel  Open Close Punct Inner
	/* Ord   */    0,    1,   -2,   -3,    0,    0,    0,   -1,
	/* Op    */    1,    1,  nvr,   -3,    0,    0,    0,   -1,
	/* Bin   */   -2,   -2,  nvr,  nvr,   -2,  nvr,  nvr,   -2,
	/* Rel   */   -3,   -3,  nvr,    0,   -2,  nvr,  nvr,   -2,
	/* Open  */    0,    0,  nvr,    0,    0,    0,    0,    0,
	/* Close */    0,    1,   -2,   -3,    0,    0,    0,   -1,
	/* Punct */   -1,   -1,  nvr,   -1,   -1,   -1,   -1,   -1,
	/* Inner */   -1,    1,   -2,   -3,   -1,    0,   -1,   -1
};
