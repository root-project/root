/*  This file is part of the Vc library. {{{

    Copyright (C) 2013 Matthias Kretz <kretz@kde.org>

    Vc is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    Vc is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with Vc.  If not, see <http://www.gnu.org/licenses/>.

}}}*/

#ifndef VC_COMMON_OPERAND_H
#define VC_COMMON_OPERAND_H

namespace ROOT {
namespace Vc
{
template<typename Parent> class Operand
{
    public:
        Parent *parent() { return static_cast<Parent *>(this); }
        const Parent *parent() const { return static_cast<const Parent *>(this); }

    private:
};

enum BinaryOperation {
    AddOp,
    SubOp,
    MulOp,
    DivOp
};

template<typename Result, typename Left, typename Right> class BinaryOperation : public Operand<BinaryOperation<Result, Left, Right> >
{
    Left m_left;
    Right m_right;
    public:
#if VC_CXX11
        Vc_ALWAYS_INLINE BinaryOperation(Left &&l, Right &&r)
#endif
        operator Result()
};

} // namespace Vc
} // namespace ROOT

#endif // VC_COMMON_OPERAND_H
