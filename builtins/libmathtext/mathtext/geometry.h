// -*- mode: c++; -*-

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

#ifndef MATHTEXT_GEOMETRY_H_
#define MATHTEXT_GEOMETRY_H_

#include <cmath>
#include <vector>
#include <algorithm>

namespace mathtext {

	/**
	 * 2D point (and vector)
	 */
	class point_t {
	private:
		float _x[2];
	public:
		inline point_t(void)
		{
		}
		inline point_t(const point_t &point)
		{
			_x[0] = point._x[0];
			_x[1] = point._x[1];
		}
		inline point_t(const float x0, const float y0)
		{
			_x[0] = x0;
			_x[1] = y0;
		}
		inline const float *x(void) const
		{
			return _x;
		}
		inline float *x(void)
		{
			return _x;
		}
		inline float operator[](const int n) const
		{
			return _x[n];
		}
		inline float &operator[](const int n)
		{
			return _x[n];
		}
		inline point_t operator+(const point_t &point) const
		{
			return point_t(_x[0] + point._x[0],
						   _x[1] + point._x[1]);
		}
		inline point_t operator-(const point_t &point) const
		{
			return point_t(_x[0] - point._x[0],
						   _x[1] - point._x[1]);
		}
		inline point_t operator+=(const point_t &point)
		{
			_x[0] += point._x[0];
			_x[1] += point._x[1];

			return *this;
		}
		inline point_t operator-=(const point_t &point)
		{
			_x[0] -= point._x[0];
			_x[1] -= point._x[1];

			return *this;
		}
		inline point_t operator*(const float scale) const
		{
			return point_t(_x[0] * scale, _x[1] * scale);
		}
		inline point_t operator/(const float scale) const
		{
			return point_t(_x[0] / scale, _x[1] / scale);
		}
		inline point_t operator*=(const float scale)
		{
			_x[0] *= scale;
			_x[1] *= scale;

			return *this;
		}
		inline float dot(const point_t &point) const
		{
			return _x[0] * point._x[0] + _x[1] * point._x[1];
		}
		inline float cross(const point_t &point) const
		{
			return _x[0] * point._x[1] - _x[1] * point._x[0];
		}
		inline float norm_square(void) const
		{
			return _x[0] * _x[0] + _x[1] * _x[1];
		}
		inline float norm(void) const
		{
			return sqrtf(norm_square());
		}
		inline point_t unit_vector(void) const
		{
			return *this / norm();
		}
		inline point_t rotate_cw(void) const
		{
			return point_t(_x[1], -_x[0]);
		}
		inline point_t rotate_ccw(void) const
		{
			return point_t(-_x[1], _x[0]);
		}
		inline bool operator==(const point_t &point) const
		{
			return _x[0] == point._x[0] && _x[1] == point._x[1];
		}
		inline bool operator!=(const point_t &point) const
		{
			return _x[0] != point._x[0] || _x[1] != point._x[1];
		}
		friend point_t operator*(const float scale,
								 const point_t &point);
		operator std::string(void) const;
	};

	inline point_t operator*(const float scale, const point_t &point)
	{
		return point_t(scale * point._x[0], scale * point._x[1]);
	}

	/**
	 * General 2D affine transform of the Adobe Imaging Model
	 *
	 * @see Adobe Systems, Inc., PostScript Language Reference Manual
	 * (Addison-Wesley, Reading, MA, 1999), section 4.3.3, pp.
	 * 187-189.
	 * @see Adobe Systems, Inc., PDF Reference, 6th Edition, Version
	 * 1.7 (Adobe Systems, Inc., San Jose, CA, 2006), section
	 * 4.2.2-4.2.3, pp. 204-209.
	 */
	class affine_transform_t {
	private:
		float _a[6];
	public:
		static const affine_transform_t identity;
		static const affine_transform_t flip_y;
		static affine_transform_t
		translate(const float tx, const float ty);
		static affine_transform_t
		scale(const float sx, const float sy);
		static affine_transform_t rotate(const float angle);
		inline affine_transform_t(const float a, const float b,
								  const float c, const float d,
								  const float tx, const float ty)
		{
			_a[0] = a;
			_a[1] = b;
			_a[2] = c;
			_a[3] = d;
			_a[4] = tx;
			_a[5] = ty;
		}
		inline const float *a(void) const
		{
			return _a;
		}
		inline float *a(void)
		{
			return _a;
		}
		inline float operator[](const int n) const
		{
			return _a[n];
		}
		inline float &operator[](const int n)
		{
			return _a[n];
		}
		inline affine_transform_t linear(void) const
		{
			return affine_transform_t(_a[0], _a[1], _a[2], _a[3],
									  0.0F, 0.0F);
		}
		inline affine_transform_t translate(void) const
		{
			return affine_transform_t(1.0F, 0.0F, 0.0F, 1.0F,
									  _a[4], _a[5]);
		}
		inline affine_transform_t operator*(const float s) const
		{
			return affine_transform_t(_a[0] * s, _a[1] * s,
									  _a[2] * s, _a[3] * s,
									  _a[4] * s, _a[5] * s);
		}
		inline affine_transform_t operator/(const float s) const
		{
			return affine_transform_t(_a[0] / s, _a[1] / s,
									  _a[2] / s, _a[3] / s,
									  _a[4] / s, _a[5] / s);
		}
		inline point_t operator*(const point_t x) const
		{
			return point_t(_a[0] * x[0] + _a[2] * x[1] + _a[4],
						   _a[1] * x[0] + _a[3] * x[1] + _a[5]);
		}
		inline affine_transform_t
		operator*(const affine_transform_t b) const
		{
			return affine_transform_t(
				_a[0] * b._a[0] + _a[1] * b._a[2],
				_a[0] * b._a[1] + _a[1] * b._a[3],
				_a[2] * b._a[0] + _a[3] * b._a[2],
				_a[2] * b._a[1] + _a[3] * b._a[3],
				_a[4] * b._a[0] + _a[5] * b._a[2] + b._a[4],
				_a[4] * b._a[1] + _a[5] * b._a[3] + b._a[5]);
		}
		inline float determinant(void) const
		{
			return _a[0] * _a[3] - _a[1] * _a[2];
		}
		inline affine_transform_t inverse(void) const
		{
			return affine_transform_t(
				_a[3], -_a[1], -_a[2], _a[0],
				_a[2] * _a[5] - _a[3] * _a[4],
				_a[1] * _a[4] - _a[0] * _a[5]) *
				(1.0F / determinant());
		}
		operator std::string(void) const;
	};

	class bounding_box_t {
	private:
		point_t _lower_left;
		point_t _upper_right;
		float _advance;
		float _italic_correction;
	public:
		inline bounding_box_t(void)
		{
		}
		inline bounding_box_t(const point_t lower_left,
							  const point_t upper_right,
							  const float advance,
							  const float italic_correction)
			: _lower_left(lower_left), _upper_right(upper_right),
			  _advance(advance), _italic_correction(italic_correction)
		{
		}
		inline bounding_box_t(const float left, const float bottom,
							  const float right, const float top,
							  const float advance,
							  const float italic_correction)
			: _advance(advance), _italic_correction(italic_correction)
		{
			_lower_left[0] = left;
			_lower_left[1] = bottom;
			_upper_right[0] = right;
			_upper_right[1] = top;
		}
		inline point_t lower_left(void) const
		{
			return _lower_left;
		}
		inline point_t &lower_left(void)
		{
			return _lower_left;
		}
		inline point_t upper_right(void) const
		{
			return _upper_right;
		}
		inline point_t &upper_right(void)
		{
			return _upper_right;
		}
		inline float left(void) const
		{
			return _lower_left[0];
		}
		inline float &left(void)
		{
			return _lower_left[0];
		}
		inline float top(void) const
		{
			return _upper_right[1];
		}
		inline float &top(void)
		{
			return _upper_right[1];
		}
		inline float right(void) const
		{
			return _upper_right[0];
		}
		inline float &right(void)
		{
			return _upper_right[0];
		}
		inline float bottom(void) const
		{
			return _lower_left[1];
		}
		inline float &bottom(void)
		{
			return _lower_left[1];
		}
		inline float advance(void) const
		{
			return _advance;
		}
		inline float &advance(void)
		{
			return _advance;
		}
		inline float italic_correction(void) const
		{
			return _italic_correction;
		}
		inline float &italic_correction(void)
		{
			return _italic_correction;
		}
		inline float width(void) const
		{
			return _upper_right[0] - _lower_left[0];
		}
		inline float height(void) const
		{
			return _upper_right[1] - _lower_left[1];
		}
		inline float horizontal_center(void) const
		{
			return 0.5F * (_lower_left[0] + _upper_right[0]);
		}
		inline float vertical_center(void) const
		{
			return 0.5F * (_lower_left[1] + _upper_right[1]);
		}
		inline float ascent(void) const
		{
			return _upper_right[1];
		}
		inline float descent(void) const
		{
			return -_lower_left[1];
		}
		inline bounding_box_t
		merge(const bounding_box_t &bounding_box) const
		{
			bounding_box_t ret;

			ret._lower_left[0] =
				std::min(_lower_left[0],
						 bounding_box._lower_left[0]);
			ret._lower_left[1] =
				std::min(_lower_left[1],
						 bounding_box._lower_left[1]);
			if (bounding_box._upper_right[0] > _upper_right[0]) {
				ret._upper_right[0] = bounding_box._upper_right[0];
				ret._italic_correction =
					bounding_box._italic_correction;
			}
			else {
				ret._upper_right[0] = _upper_right[0];
				ret._italic_correction = _italic_correction;
			}
			ret._upper_right[1] =
				std::max(_upper_right[1],
						 bounding_box._upper_right[1]);
			ret._advance =
				std::max(_upper_right[0] + _advance,
						 bounding_box._upper_right[0] +
						 bounding_box._advance) -
				ret._upper_right[0];

			return ret;
		}
		inline bounding_box_t operator+(const point_t &point) const
		{
			return bounding_box_t(_lower_left + point,
								  _upper_right + point,
								  _advance + point[0],
								  _italic_correction);
		}
		inline bounding_box_t operator-(const point_t &point) const
		{
			return bounding_box_t(_lower_left - point,
								  _upper_right - point,
								  _advance - point[0],
								  _italic_correction);
		}
		inline bounding_box_t operator+=(const point_t &point)
		{
			_lower_left += point;
			_upper_right += point;
			_advance += point[0];

			return *this;
		}
		inline bounding_box_t operator-=(const point_t &point)
		{
			_lower_left -= point;
			_upper_right -= point;
			_advance -= point[0];

			return *this;
		}
		friend bounding_box_t
		operator+(const point_t &, const bounding_box_t &);
		friend bounding_box_t
		operator-(const point_t &, const bounding_box_t &);
		friend bounding_box_t
		operator*(const affine_transform_t &,
				  const bounding_box_t &);
	};

	inline bounding_box_t
	operator+(const point_t &point,
			  const bounding_box_t &bounding_box)
	{
		return bounding_box_t(point + bounding_box._lower_left,
							  point + bounding_box._upper_right,
							  point[0] + bounding_box._advance,
							  bounding_box._italic_correction);
	}

	inline bounding_box_t
	operator-(const point_t &point,
			  const bounding_box_t &bounding_box)
	{
		return bounding_box_t(point - bounding_box._lower_left,
							  point - bounding_box._upper_right,
							  point[0] + bounding_box._advance,
							  bounding_box._italic_correction);
	}

	inline bounding_box_t
	operator*(const affine_transform_t &transform,
			  const bounding_box_t &bounding_box)
	{
		return bounding_box_t(
			transform * bounding_box._lower_left,
			transform * bounding_box._upper_right,
			(transform * point_t(bounding_box._advance, 0))[0],
			(transform * point_t(
				bounding_box._italic_correction, 0))[0]);
	}

}

#endif // MATHTEXT_GEOMETRY_H_
