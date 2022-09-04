""" cppyy extensions for numba
"""

import cppyy
import cppyy.types as cpp_types
import cppyy.reflex as cpp_refl

import numba
import numba.extending as nb_ext
import numba.core.cgutils as nb_cgu
import numba.core.datamodel.models as nb_models
import numba.core.imputils as nb_iutils
import numba.core.registry as nb_reg
import numba.core.types as nb_types
import numba.core.typing as nb_typing

from llvmlite import ir
from llvmlite.llvmpy.core import Type as irType


ir_voidptr = irType.pointer(irType.int(8))  # by convention

cppyy_addressof_ptr = None


_numba2cpp = {
    nb_types.void            : 'void',
    nb_types.long_           : 'long',
    nb_types.int64           : 'int64_t',
    nb_types.float64         : 'double',
}

def numba2cpp(val):
    if hasattr(val, 'literal_type'):
        val = val.literal_type
    return _numba2cpp[val]

_cpp2numba = {
    'void'                   : nb_types.void,
    'long'                   : nb_types.long_,
    'int64_t'                : nb_types.int64,
    'double'                 : nb_types.float64,
}

def cpp2numba(val):
    if type(val) != str:
        return numba.typeof(val)
    return _cpp2numba[val]

_cpp2ir = {
    'int64_t'                : irType.int(64),
}

def cpp2ir(val):
    return _cpp2ir[val]


#
# C++ function pointer -> Numba
#
class CppFunctionNumbaType(nb_types.Callable):
    targetdescr = nb_reg.cpu_target
    requires_gil = False

    def __init__(self, func):
        super(CppFunctionNumbaType, self).__init__('CppFunction(%s)' % str(func))

        self.sig = None
        self._func = func

        self._signatures = list()
        self._impl_keys = dict()

    def is_precise(self):
        return True          # by definition

    def get_call_type(self, context, args, kwds):
        try:
            return self._impl_keys[args].sig
        except KeyError:
            pass

        ol = CppFunctionNumbaType(self._func.__overload__(*(numba2cpp(x) for x in args)))

        ol.sig = nb_typing.Signature(
            return_type=cpp2numba(ol._func.__cpp_reflex__(cpp_refl.RETURN_TYPE)),
            args=args,
            recvr=None)  # this pointer

        self._impl_keys[args] = ol

        @nb_iutils.lower_builtin(ol, *args)
        def lower_external_call(context, builder, sig, args,
                ty=nb_types.ExternalFunctionPointer(ol.sig, ol.get_pointer), pyval=self._func):
            ptrty = context.get_function_pointer_type(ty)
            ptrval = context.add_dynamic_addr(
                builder, ty.get_pointer(pyval), info=str(pyval))
            fptr = builder.bitcast(ptrval, ptrty)
            return context.call_function_pointer(builder, fptr, args)

        return ol.sig

    def get_call_signatures(self):
        return list(self._signatures), False

    def get_impl_key(self, sig):
        return self._impl_keys[sig.args]

    def get_pointer(self, func):
        if func is None: func = self._func
        ol = func.__overload__(*(numba2cpp(x) for x in self.sig.args))
        address = cppyy.addressof(ol)
        if not address:
            raise RuntimeError("unresolved address for %s" % str(ol))
        return address

    @property
    def key(self):
        return self._func


@nb_ext.typeof_impl.register(cpp_types.Function)
def typeof_function(val, c):
    return CppFunctionNumbaType(val)

@nb_ext.typeof_impl.register(cpp_types.Template)
def typeof_template(val, c):
    if hasattr(val, '__overload__'):
        return CppFunctionNumbaType(val)
    raise RuntimeError("only function templates supported")

@nb_ext.register_model(CppFunctionNumbaType)
class CppFunctionModel(nb_models.PrimitiveModel):
    def __init__(self, dmm, fe_type):
      # the function pointer of this overload can not be exactly typed, but
      # only the storage size is relevant, so simply use a void*
        be_type = ir.PointerType(dmm.lookup(nb_types.void).get_value_type())
        super(CppFunctionModel, self).__init__(dmm, fe_type, be_type)

@nb_iutils.lower_constant(CppFunctionNumbaType)
def constant_function_pointer(context, builder, ty, pyval):
    # TODO: needs to exist for the proper flow, but why? The lowering of the
    # actual overloads is handled dynamically.
    return


#
# C++ class -> Numba
#
class CppClassNumbaType(CppFunctionNumbaType):
    def __init__(self, scope):
        super(CppClassNumbaType, self).__init__(scope.__init__)
        self.name = 'CppClass(%s)' % scope.__cpp_name__    # overrides value in Type
        self._scope = scope

    def get_call_type(self, context, args, kwds):
        sig = super(CppClassNumbaType, self).get_call_type(context, args, kwds)
        self.sig = sig
        return sig

    def is_precise(self):
        return True

    @property
    def key(self):
        return self._scope


class_numbatypes = dict()

@nb_ext.typeof_impl.register(cpp_types.Scope)
def typeof_scope(val, c):
    if 'namespace' in repr(val):
        return nb_types.Module(val)

    global class_numbatypes

    try:
        cnt = class_numbatypes[val]
    except KeyError:
        class ImplClassType(CppClassNumbaType):
            pass

        cnt = ImplClassType(val)
        class_numbatypes[val] = cnt

        fields = list()
        for key, value in val.__dict__.items():
            if type(value) == cpp_types.DataMember:
                fields.append((key, value))

        nb_ftypes = list(); ir_ftypes = dict(); offsets = list()
        for f, d in fields:
          # declare field to Numba
            nb_ext.make_attribute_wrapper(ImplClassType, f, f)

          # collect field type information for Numba and the IR builder
            ct = d.__cpp_reflex__(cpp_refl.TYPE)
            nb_ftypes.append((f, cpp2numba(ct)))
            ir_ftypes[f] = cpp2ir(ct)

          # collect field offsets
            offsets.append((f, d.__cpp_reflex__(cpp_refl.OFFSET)))

      # TODO: this refresh is needed b/c the scope type is registered as a
      # callable after the tracing started; no idea of the side-effects ...
        nb_reg.cpu_target.typing_context.refresh()

      # create a model description for Numba
        @nb_ext.register_model(ImplClassType)
        class ImplClassModel(nb_models.StructModel):
            def __init__(self, dmm, fe_type):
                members = nb_ftypes
                nb_models.StructModel.__init__(self, dmm, fe_type, members)

      # Python proxy unwrapping for arguments into the Numba trace
        @nb_ext.unbox(ImplClassType)
        def unbox_instance(typ, obj, c):
            global cppyy_addressof_ptr
            if cppyy_addressof_ptr is None:
                # TODO: loading this for retrieving addresses of objects only; is a bit overkill
                cppyy.include("CPyCppyy/API.h")
                fp1 = cppyy.addressof(cppyy.gbl.CPyCppyy.Instance_AsVoidPtr)
                ptrty = irType.pointer(irType.function(ir_voidptr, [ir_voidptr]))
                ptrval = c.context.add_dynamic_addr(c.builder, fp1, info='Instance_AsVoidPtr')
                cppyy_addressof_ptr = c.builder.bitcast(ptrval, ptrty)

            pobj = c.context.call_function_pointer(c.builder, cppyy_addressof_ptr, [obj])

            # TODO: the use of create_struct_proxy is (too) slow and probably unnecessary:w
            d = nb_cgu.create_struct_proxy(typ)(c.context, c.builder)
            for f, o in offsets:
                pf = c.builder.bitcast(pobj, irType.pointer(ir_ftypes[f]))
                setattr(d, f, c.builder.load(pf))

            # TODO: after typing, cppyy_addressof_ptr can not fail, so error checking is
            # only done b/c of create_struct_proxy(typ), which should go
            return nb_ext.NativeValue(d._getvalue(), is_error=c.pyapi.c_api_error())

    return cnt


#
# C++ instance -> Numba
#
@nb_ext.typeof_impl.register(cpp_types.Instance)
def typeof_instance(val, c):
    return typeof_scope(type(val), c)

