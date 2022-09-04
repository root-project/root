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
import numba.core.typing.templates as nb_tmpl
import numba.core.types as nb_types
import numba.core.typing as nb_typing

from llvmlite import ir
from llvmlite.llvmpy.core import Type as irType


# setuptools entry point for Numba
def _init_extension():
    pass


ir_voidptr  = irType.pointer(irType.int(8))  # by convention
ir_charptr  = ir_voidptr
ir_intptr_t = irType.int(cppyy.sizeof('void*')*8)

cppyy_addressof_ptr = None

_numba2cpp = {
    nb_types.void            : 'void',
    nb_types.voidptr         : 'void*',
    nb_types.int_            : 'int',
    nb_types.int32           : 'int32_t',
    nb_types.int64           : 'int64_t',
    nb_types.long_           : 'long',
    nb_types.float32         : 'float',
    nb_types.float64         : 'double',
}

def numba2cpp(val):
    if hasattr(val, 'literal_type'):
        val = val.literal_type
    return _numba2cpp[val]

_cpp2numba = {
    'void'                   : nb_types.void,
    'void*'                  : nb_types.voidptr,
    'int'                    : nb_types.intc,
    'int32_t'                : nb_types.int32,
    'int64_t'                : nb_types.int64,
    'long'                   : nb_types.long_,
    'float'                  : nb_types.float32,
    'double'                 : nb_types.float64,
}

def cpp2numba(val):
    if type(val) != str:
        return numba.typeof(val)
    return _cpp2numba[val]

_cpp2ir = {
    'int'                    : irType.int(nb_types.intc.bitwidth),
    'int32_t'                : irType.int(32),
    'int64_t'                : irType.int(64),
    'float'                  : irType.float(),
    'double'                 : irType.double(),
}

def cpp2ir(val):
    return _cpp2ir[val]


#
# C++ function pointer -> Numba
#
class CppFunctionNumbaType(nb_types.Callable):
    targetdescr = nb_reg.cpu_target
    requires_gil = False

    def __init__(self, func, is_method=False):
        super(CppFunctionNumbaType, self).__init__('CppFunction(%s)' % str(func))

        self.sig = None
        self._func = func
        self._is_method = is_method

        self._signatures = list()
        self._impl_keys = dict()

    def is_precise(self):
        return True          # by definition

    def get_call_type(self, context, args, kwds):
        try:
            return self._impl_keys[args].sig
        except KeyError:
            pass

        ol = CppFunctionNumbaType(self._func.__overload__(*(numba2cpp(x) for x in args)), self._is_method)

        if self._is_method:
            args = (nb_types.voidptr, *args)

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
            if hasattr(context, 'cppyy_currentcall_this'):
                args = [context.cppyy_currentcall_this]+args
                del context.cppyy_currentcall_this
            return context.call_function_pointer(builder, fptr, args)

        return ol.sig

    def get_call_signatures(self):
        return list(self._signatures), False

    def get_impl_key(self, sig):
        return self._impl_keys[sig.args]

    def get_pointer(self, func):
        if func is None: func = self._func
        ol = func.__overload__(*(numba2cpp(x) for x in self.sig.args[int(self._is_method):]))
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
    # actual overload is handled dynamically.
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

@nb_tmpl.infer_getattr
class CppClassAttribute(nb_tmpl.AttributeTemplate):
     key = CppClassNumbaType

     def generic_resolve(self, typ, attr):
         try:
             f = getattr(typ._scope, attr)
             if type(f) == cpp_types.Function:
                 return CppFunctionNumbaType(f, is_method=True)
         except AttributeError:
             pass

@nb_iutils.lower_getattr_generic(CppClassNumbaType)
def cppclass_getattr_impl(context, builder, typ, val, attr):
    # TODO: the following relies on the fact that numba will first lower the
    # field access, then immediately lower the call; and that the `val` loads
    # the struct representing the C++ object. Neither need be stable.
    context.cppyy_currentcall_this = builder.bitcast(val.operands[0], ir_voidptr)
    return context.cppyy_currentcall_this


scope_numbatypes = dict()

@nb_ext.typeof_impl.register(cpp_types.Scope)
def typeof_scope(val, c):
    global scope_numbatypes

    try:
        cnt = scope_numbatypes[val]
    except KeyError:
        if val.__cpp_reflex__(cpp_refl.IS_NAMESPACE):
            cnt = nb_types.Module(val)
            scope_numbatypes[val] = cnt
            return cnt

        class ImplClassType(CppClassNumbaType):
            pass

        cnt = ImplClassType(val)
        scope_numbatypes[val] = cnt

      # declare data members to Numba
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
                # TODO: loading the CPyCppyy API just for Instance_AsVoidPtr is a bit overkill
                cppyy.include("CPyCppyy/API.h")
                cppyy_addressof_ptr = cppyy.addressof(cppyy.gbl.CPyCppyy.Instance_AsVoidPtr)
            ptrty = irType.pointer(irType.function(ir_voidptr, [ir_voidptr]))
            ptrval = c.context.add_dynamic_addr(c.builder, cppyy_addressof_ptr, info='Instance_AsVoidPtr')
            fp = c.builder.bitcast(ptrval, ptrty)

            pobj = c.context.call_function_pointer(c.builder, fp, [obj])

            # TODO: the use of create_struct_proxy is (too) slow and probably unnecessary
            d = nb_cgu.create_struct_proxy(typ)(c.context, c.builder)
            basep = c.builder.bitcast(pobj, ir_charptr)
            for f, o in offsets:
                pfc = c.builder.gep(basep, [ir.Constant(ir_intptr_t, o)])
                pf = c.builder.bitcast(pfc, irType.pointer(ir_ftypes[f]))
                setattr(d, f, c.builder.load(pf))

            return nb_ext.NativeValue(d._getvalue(), is_error=None, cleanup=None)

    return cnt


#
# C++ instance -> Numba
#
@nb_ext.typeof_impl.register(cpp_types.Instance)
def typeof_instance(val, c):
    global scope_numbatypes

    try:
        return scope_numbatypes[type(val)]
    except KeyError:
        pass

    return typeof_scope(type(val), c)
