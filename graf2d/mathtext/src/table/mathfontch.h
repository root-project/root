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

static const unsigned long nfont_change = 8;
static const char *
font_change_control_sequence[nfont_change] = {
	"\\mathbb", "\\mathbf", "\\mathfr", "\\mathit", "\\mathrm",
	"\\mathscr", "\\mathsf", "\\mathtt"
};
static const unsigned int font_change_family[nfont_change] = {
	math_symbol_t::FAMILY_MATH_BLACKBOARD_BOLD,
	math_symbol_t::FAMILY_BOLD,
	math_symbol_t::FAMILY_MATH_FRAKTUR_REGULAR,
	math_symbol_t::FAMILY_ITALIC,
	math_symbol_t::FAMILY_REGULAR,
	math_symbol_t::FAMILY_MATH_SCRIPT_ITALIC,
	math_symbol_t::FAMILY_MATH_SANS_SERIF_REGULAR,
	math_symbol_t::FAMILY_MATH_MONOSPACE
};
