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

// See also Knuth (1986), p. 447

/////////////////////////////////////////////////////////////////////
// TeX Font Parameters

// Following variables are derived from MathTime Professional II's
// mtpro2.sty
const float math_text_renderer_t::script_ratio = 0.7;
const float math_text_renderer_t::script_script_ratio = 0.55;
// Following variables are derived from plain.tex
const float math_text_renderer_t::thin_mu_skip = 3.0F;
const float math_text_renderer_t::med_mu_skip = 4.0F;
const float math_text_renderer_t::thick_mu_skip = 5.0F;
const float math_text_renderer_t::delimiter_factor = 901.0F;
const float math_text_renderer_t::delimiter_shortfall = 5.0F;

/////////////////////////////////////////////////////////////////////
// (Explicit) TFM Font Parameters

// Following variables are derived from MathTime Professional II's
// mt2syt.tfm
const float math_text_renderer_t::num_1 = 0.701F;
const float math_text_renderer_t::num_2 = 0.404F;
const float math_text_renderer_t::num_3 = 0.454F;
const float math_text_renderer_t::denom_1 = 0.710F;
const float math_text_renderer_t::denom_2 = 0.355F;
const float math_text_renderer_t::sup_1 = 0.413F;
const float math_text_renderer_t::sup_2 = 0.363F;
const float math_text_renderer_t::sup_3 = 0.289F;
const float math_text_renderer_t::sub_1 = 0.150F;
const float math_text_renderer_t::sub_2 = 0.269F;
const float math_text_renderer_t::sup_drop = 0.386F;
const float math_text_renderer_t::sub_drop = 0.050F;
const float math_text_renderer_t::delim_1 = 2.390F;
const float math_text_renderer_t::delim_2 = 1.010F;
// Following variables are derived from the STIX fonts. Axis
// height is defined by the horizontal axis of the minus sign
// (U+2212), the default rule thickness is the thickness of the em
// dash (U+2014).
const float math_text_renderer_t::axis_height = 0.253F;
const float math_text_renderer_t::default_rule_thickness = 0.049F;
// Following variables are derived from MathTime Professional II's
// mt2exa.tfm
const float math_text_renderer_t::big_op_spacing_1 = 0.116667F;
const float math_text_renderer_t::big_op_spacing_2 = 0.177777F;
const float math_text_renderer_t::big_op_spacing_3 = 0.211111F;
const float math_text_renderer_t::big_op_spacing_4 = 0.6F;
const float math_text_renderer_t::big_op_spacing_5 = 0.111111F;

/////////////////////////////////////////////////////////////////////
// Implicit TFM Font Parameters
const float math_text_renderer_t::radical_rule_thickness = 0.054F;
const float math_text_renderer_t::large_operator_display_scale = 1.4F;

/////////////////////////////////////////////////////////////////////
// Text Mode Parameters
const float math_text_renderer_t::baselineskip_factor = 1.2F;
