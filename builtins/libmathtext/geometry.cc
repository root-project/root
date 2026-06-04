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
#include <sstream>
#include <mathtext/geometry.h>

namespace mathtext {

	point_t::operator std::string(void) const
	{
		std::stringstream stream;

		stream << '(' << _x[0] << ", " << _x[1] << ')';

		return stream.str();
	}

	const affine_transform_t affine_transform_t::identity =
		affine_transform_t(1.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F);

	affine_transform_t affine_transform_t::
	translate(const float tx, const float ty)
	{
		return affine_transform_t(1.0F, 0.0F, 0.0F, 1.0F, tx, ty);
	}

	affine_transform_t affine_transform_t::
	scale(const float sx, const float sy)
	{
		return affine_transform_t(sx, 0.0F, 0.0F, sy, 0.0F, 0.0F);
	}

	affine_transform_t affine_transform_t::rotate(const float angle)
	{
		float sin_angle;
		float cos_angle;

		sincosf(angle, &sin_angle, &cos_angle);

		return affine_transform_t(cos_angle, sin_angle,
								  -sin_angle, cos_angle, 0, 0);
	}

	affine_transform_t::operator std::string(void) const
	{
		std::stringstream stream;

		stream << '(' << _a[0] << ", " << _a[1] << ", 0)" << std::endl;
		stream << '(' << _a[2] << ", " << _a[3] << ", 0)" << std::endl;
		stream << '(' << _a[4] << ", " << _a[5] << ", 1)";

		return stream.str();
	}

}
