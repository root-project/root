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

#include <cmath>
#include <iostream>
#include <string>
#include "mathtext.h"

/////////////////////////////////////////////////////////////////////

namespace mathtext {

	/////////////////////////////////////////////////////////////////

	void math_text_t::
	tree_view_prefix(const std::vector<bool> &branch,
					 const bool final) const
	{
		if(branch.size() > 0) {
			std::cerr << ' ';
			for(std::vector<bool>::const_iterator iterator =
					branch.begin();
				iterator != branch.end(); iterator++) {
				if(*iterator) {
					if(iterator + 1 == branch.end()) {
						if(final)
							std::cerr << "\342\224\224\342\224\200 ";
						else
							std::cerr << "\342\224\234\342\224\200 ";
					}
					else
						std::cerr << "\342\224\202   ";
				}
				else
					if(iterator + 1 == branch.end())
						std::cerr << "   ";
					else
						std::cerr << "    ";
			}
		}
	}

	void math_text_t::tree_view(const field_t &field,
								std::vector<bool> &branch,
								const bool final) const
	{
		switch(field._type) {
		case field_t::TYPE_MATH_SYMBOL:
			tree_view_prefix(branch, true);
			std::cerr << "<math_symbol code=\""
					  << field._math_symbol._code
					  << "\" glyph="
					  << (unsigned int)field._math_symbol._glyph
					  << "/>" << std::endl;
			break;
		case field_t::TYPE_BOX:
			tree_view_prefix(branch, true);
			std::cerr << "<box/>" << std::endl;
			break;
		case field_t::TYPE_MATH_LIST:
			if(field._math_list.empty()) {
				tree_view_prefix(branch, true);
				std::cerr << "<empty/>" << std::endl;
			}
			else {
				tree_view_prefix(branch, false);
				std::cerr << "<math_list>" << std::endl;

				std::vector<bool> branch_copy = branch;

				branch_copy.back() = !final;
				for(std::vector<item_t>::const_iterator iterator =
						field._math_list.begin();
					iterator != field._math_list.end(); iterator++) {
					branch_copy.back() = !final;
					branch_copy.push_back(true);
					tree_view(*iterator, branch_copy,
							  iterator + 1 == field._math_list.end());
					branch_copy.pop_back();
				}
				branch_copy.back() = !final;
				tree_view_prefix(branch_copy, true);
				std::cerr << "</math_list>" << std::endl;
			}
			break;
		default:
			tree_view_prefix(branch, true);
			std::cerr << "<err_field/>" << std::endl;
		}
	}

	void math_text_t::tree_view(const item_t &item,
								std::vector<bool> &branch,
								const bool final) const
	{
		std::vector<bool> branch_copy = branch;

		switch(item._type) {
		case item_t::TYPE_ATOM:
			tree_view_prefix(branch, final);
			std::cerr << "<atom>" << std::endl;
			branch_copy.back() = !final;
			branch_copy.push_back(true);
			tree_view(item._atom, branch_copy, final);
			branch_copy.pop_back();
			tree_view_prefix(branch_copy, final);
			std::cerr << "</atom>" << std::endl;
			break;
		case item_t::TYPE_BOUNDARY:
			tree_view_prefix(branch, final);
			std::cerr << "<boundary>" << std::endl;
			branch_copy.back() = !final;
			branch_copy.push_back(true);
			tree_view(item._atom, branch_copy, final);
			branch_copy.pop_back();
			tree_view_prefix(branch_copy, final);
			std::cerr << "</boundary>" << std::endl;
			break;
		case item_t::TYPE_GENERALIZED_FRACTION:
			tree_view_prefix(branch, final);
			std::cerr << "<generalized_fraction/>" << std::endl;
			break;
		default:
			tree_view_prefix(branch, final);
			std::cerr << "<err_item/>" << std::endl;
		}
	}

	void math_text_t::tree_view(const atom_t &atom,
								std::vector<bool> &branch,
								const bool final) const
	{
		tree_view_prefix(branch, false);
		std::cerr << "<type>";
		switch(atom._type) {
		case atom_t::TYPE_ORD:
			std::cerr << "Ord";
			break;
		case atom_t::TYPE_OP:
			std::cerr << "Op";
			break;
		case atom_t::TYPE_BIN:
			std::cerr << "Bin";
			break;
		case atom_t::TYPE_REL:
			std::cerr << "Rel";
			break;
		case atom_t::TYPE_OPEN:
			std::cerr << "Open";
			break;
		case atom_t::TYPE_CLOSE:
			std::cerr << "Close";
			break;
		case atom_t::TYPE_PUNCT:
			std::cerr << "Punct";
			break;
		case atom_t::TYPE_INNER:
			std::cerr << "Inner";
			break;
		case atom_t::TYPE_OVER:
			std::cerr << "Over";
			break;
		case atom_t::TYPE_UNDER:
			std::cerr << "Under";
			break;
		case atom_t::TYPE_ACC:
			std::cerr << "Acc";
			break;
		case atom_t::TYPE_RAD:
			std::cerr << "Rad";
			break;
		case atom_t::TYPE_VCENT:
			std::cerr << "Vcent";
			break;
		default:
			std::cerr << "??" << atom._type;
			break;
		}
		std::cerr << "</type>" << std::endl;

		std::vector<bool> branch_copy = branch;

		if(!atom._nucleus.empty()) {
			const bool way_final = atom._superscript.empty() &&
				atom._subscript.empty();

			tree_view_prefix(branch, way_final);
			std::cerr << "<nucleus>" << std::endl;
			branch_copy.back() = !way_final;
			branch_copy.push_back(true);
			tree_view(atom._nucleus, branch_copy, final);
			branch_copy.pop_back();
			tree_view_prefix(branch_copy, way_final);
			std::cerr << "</nucleus>" << std::endl;
		}
		if(!atom._superscript.empty()) {
			const bool way_final = atom._subscript.empty();

			tree_view_prefix(branch, way_final);
			std::cerr << "<superscript>" << std::endl;
			branch_copy.back() = !way_final;
			branch_copy.push_back(true);
			tree_view(atom._superscript, branch_copy, final);
			branch_copy.pop_back();
			tree_view_prefix(branch_copy, way_final);
			std::cerr << "</superscript>" << std::endl;
		}
		if(!atom._subscript.empty()) {
			tree_view_prefix(branch, true);
			std::cerr << "<subscript>" << std::endl;
			branch_copy.back() = false;
			branch_copy.push_back(true);
			tree_view(atom._subscript, branch_copy, final);
			branch_copy.pop_back();
			tree_view_prefix(branch_copy, true);
			std::cerr << "</subscript>" << std::endl;
		}
	}

}
