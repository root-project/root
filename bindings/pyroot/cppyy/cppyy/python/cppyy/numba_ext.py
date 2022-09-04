""" cppyy extensions for numba
"""

import cppyy
import cppyy.types

import numba
import numba.extending as nb_ext
import numba.core.datamodel.models as nb_models
import numba.core.imputils as nb_iutils
import numba.core.types as nb_types
import numba.core.registry as nb_reg

from llvmlite import ir


numba2cpp = {
    nb_types.long_           : 'long',
    nb_types.int64           : 'int64_t',
    nb_types.float64         : 'double',
}

cpp2numba = {
    'long'                   : nb_types.long_,
    'int64_t'                : nb_types.int64,
    'double'                 : nb_types.float64,
}


class CppyyFunctionNumbaType(nb_types.Callable):
    targetdescr = nb_reg.cpu_target
    requires_gil = False

    def __init__(self, func):
        super(CppyyFunctionNumbaType, self).__init__(str(func))

        self.sig = None
        self._func = func

        self._signatures = list()
        self._impl_keys = dict()

    def is_precise(self):
        return True          # by definition

    def get_call_type(self, context, args, kwds):
        ol = CppyyFunctionNumbaType(self._func.__overload__(*(numba2cpp[x] for x in args)))

        ol.sig = numba.core.typing.Signature(
            return_type=cpp2numba[ol._func.__cpp_rettype__],
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
        ol = func.__overload__(*(numba2cpp[x] for x in self.sig.args))
        address = cppyy.addressof(ol)
        if not address:
            raise RuntimeError("unresolved address for %s" % str(ol))
        return address

    @property
    def key(self):
        return self._func


@nb_ext.register_model(CppyyFunctionNumbaType)
class CppyyFunctionModel(nb_models.PrimitiveModel):
    def __init__(self, dmm, fe_type):
      # the function pointer of this overload can not be exactly typed, but
      # only the storage size is relevant, so simply use a void*
        be_type = ir.PointerType(dmm.lookup(nb_types.void).get_value_type())
        super(CppyyFunctionModel, self).__init__(dmm, fe_type, be_type)

@nb_iutils.lower_constant(CppyyFunctionNumbaType)
def constant_function_pointer(context, builder, ty, pyval):
    # TODO: needs to exist for the proper flow, but why? The lowering of the
    # actual overloads is handled dynamically.
    return

@nb_ext.typeof_impl.register(cppyy.types.Scope)
def typeof_scope(val, c):
    if 'namespace' in repr(val):
        return numba.types.Module(val)
    return CppyyScopeNumbaType(val)

@nb_ext.typeof_impl.register(cppyy.types.Function)
def typeof_function(val, c):
    return CppyyFunctionNumbaType(val)

@nb_ext.typeof_impl.register(cppyy.types.Template)
def typeof_template(val, c):
    if hasattr(val, '__overload__'):
        return CppyyFunctionNumbaType(val)
    raise RuntimeError("only function templates supported")