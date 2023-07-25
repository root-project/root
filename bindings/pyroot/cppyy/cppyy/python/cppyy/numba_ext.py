""" cppyy extensions for numba
"""

import cppyy
import cppyy.types as cpp_types
import cppyy.reflex as cpp_refl

import numba
import numba.extending as nb_ext
import numba.core.cgutils as nb_cgu
import numba.core.datamodel as nb_dm
import numba.core.imputils as nb_iutils
import numba.core.registry as nb_reg
import numba.core.typing.templates as nb_tmpl
import numba.core.types as nb_types
import numba.core.typing as nb_typing

from llvmlite import ir


# setuptools entry point for Numba
def _init_extension():
    pass


class Qualified:
    default   = 0
    value     = 1

ir_voidptr  = ir.PointerType(ir.IntType(8))  # by convention
ir_byteptr  = ir_voidptr
ir_intptr_t = ir.IntType(cppyy.sizeof('void*')*8)     # use MACHINE_BITS?

# special case access to unboxing/boxing APIs
cppyy_as_voidptr   = cppyy.addressof('Instance_AsVoidPtr')
cppyy_from_voidptr = cppyy.addressof('Instance_FromVoidPtr')


_cpp2numba = {
    'void'                   : nb_types.void,
    'void*'                  : nb_types.voidptr,
    'int8_t'                 : nb_types.int8,
    'uint8_t'                : nb_types.uint8,
    'short'                  : nb_types.short,
    'unsigned short'         : nb_types.ushort,
    'internal_enum_type_t'   : nb_types.intc,
    'int'                    : nb_types.intc,
    'unsigned int'           : nb_types.uintc,
    'int32_t'                : nb_types.int32,
    'uint32_t'               : nb_types.uint32,
    'int64_t'                : nb_types.int64,
    'uint64_t'               : nb_types.uint64,
    'long'                   : nb_types.long_,
    'unsigned long'          : nb_types.ulong,
    'Long64_t'               : nb_types.longlong,   # Note: placed above long long as the last value is used in numba2cpp
    'long long'              : nb_types.longlong,   # this value will be used in numba2cpp
    'unsigned long long'     : nb_types.ulonglong,
    'float'                  : nb_types.float32,
    'long double'            : nb_types.float64,    # Note: see Long64_t
    'double'                 : nb_types.float64,    # this value will be used in numba2cpp
}

def cpp2numba(val):
    if type(val) != str:
        # TODO: distinguish ptr/ref/byval
        return typeof_scope(val, nb_typing.typeof.Purpose.argument, Qualified.value)
    return _cpp2numba[val]

_numba2cpp = dict()
for key, value in _cpp2numba.items():
    _numba2cpp[value] = key

def numba2cpp(val):
    if hasattr(val, 'literal_type'):
        val = val.literal_type
    return _numba2cpp[val]

# TODO: looks like Numba treats unsigned types as signed when lowering,
# which seems to work as they're just reinterpret_casts
_cpp2ir = {
    'char*'                  : ir_byteptr,
    'int8_t'                 : ir.IntType(8),
    'uint8_t'                : ir.IntType(8),
    'short'                  : ir.IntType(nb_types.short.bitwidth),
    'unsigned short'         : ir.IntType(nb_types.ushort.bitwidth),
    'internal_enum_type_t'   : ir.IntType(nb_types.intc.bitwidth),
    'int'                    : ir.IntType(nb_types.intc.bitwidth),
    'unsigned int'           : ir.IntType(nb_types.uintc.bitwidth),
    'int32_t'                : ir.IntType(32),
    'uint32_t'               : ir.IntType(32),
    'int64_t'                : ir.IntType(64),
    'uint64_t'               : ir.IntType(64),
    'long'                   : ir.IntType(nb_types.long_.bitwidth),
    'unsigned long'          : ir.IntType(nb_types.ulong.bitwidth),
    'long long'              : ir.IntType(nb_types.longlong.bitwidth),
    'unsigned long long'     : ir.IntType(nb_types.ulonglong.bitwidth),
    'float'                  : ir.FloatType(),
    'double'                 : ir.DoubleType(),
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

        ol = CppFunctionNumbaType(self._func.__overload__(tuple(numba2cpp(x) for x in args)), self._is_method)

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
        ol = func.__overload__(tuple(numba2cpp(x) for x in self.sig.args[int(self._is_method):]))
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
class CppFunctionModel(nb_dm.models.PrimitiveModel):
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
# C++ method / data member -> Numba
#
class CppDataMemberInfo(object):
    __slots__ = ['f_name', 'f_offset', 'f_nbtype', 'f_irtype']

    def __init__(self, name, offset, cpptype):
        self.f_name   = name
        self.f_offset = offset
        self.f_nbtype = cpp2numba(cpptype)
        self.f_irtype = cpp2ir(cpptype)


#
# C++ class -> Numba
#
class CppClassNumbaType(CppFunctionNumbaType):
    def __init__(self, scope, qualifier):
        super(CppClassNumbaType, self).__init__(scope.__init__)
        self.name = 'CppClass(%s)' % scope.__cpp_name__    # overrides value in Type
        self._scope     = scope
        self._qualifier = qualifier

    def get_scope(self):
        return self._scope

    def get_qualifier(self):
        return self._qualifier

    def get_call_type(self, context, args, kwds):
        sig = super(CppClassNumbaType, self).get_call_type(context, args, kwds)
        self.sig = sig
        return sig

    def is_precise(self):
        return True

    @property
    def key(self):
        return (self._scope, self._qualifier)

@nb_tmpl.infer_getattr
class CppClassFieldResolver(nb_tmpl.AttributeTemplate):
    key = CppClassNumbaType

    def generic_resolve(self, typ, attr):
        ft = typ.__dict__.get(attr, None)
        if ft is not None:
            return ft

        try:
            f = getattr(typ._scope, attr)
            if type(f) == cpp_types.Function:
                ft = CppFunctionNumbaType(f, is_method=True)
        except AttributeError:
            pass

        try:
            f = typ._scope.__dict__[attr]
            if type(f) == cpp_types.DataMember:
                ct = f.__cpp_reflex__(cpp_refl.TYPE)
                ft = cpp2numba(ct)
        except AttributeError:
            pass

        if ft is not None:
            typ.__dict__[attr] = ft

        return ft

@nb_iutils.lower_getattr_generic(CppClassNumbaType)
def cppclass_getattr_impl(context, builder, typ, val, attr):
    # TODO: the following relies on the fact that numba will first lower the
    # field access, then immediately lower the call; and that the `val` loads
    # the struct representing the C++ object. Neither need be stable.
    if attr in typ._scope.__dict__ and type(typ._scope.__dict__[attr]) == cpp_types.DataMember:
        dm = typ._scope.__dict__[attr]
        ct = dm.__cpp_reflex__(cpp_refl.TYPE)
        offset = dm.__cpp_reflex__(cpp_refl.OFFSET)

        q = typ.get_qualifier()
        if q == Qualified.default:
            llval = builder.bitcast(val, ir_byteptr)
            pfc = builder.gep(llval, [ir.Constant(ir_intptr_t, offset)])
            pf = builder.bitcast(pfc, ir.PointerType(cpp2ir(ct)))
            return builder.load(pf)

        elif q == Qualified.value:
            # TODO: access members of by value returns
            model = nb_dm.default_manager.lookup(typ)
            return model.get(builder, val, attr)

        else:
            assert not "unknown qualified type"

        # TODO: easier with inttoptr and ptrtoint (cgutils.pointer_add)?
        llval = builder.bitcast(val, ir_byteptr)
        pfc = builder.gep(llval, [ir.Constant(ir_intptr_t, offset)])
        pf = builder.bitcast(pfc, ir.PointerType(cpp2ir(ct)))
        return builder.load(pf)

  # assume this is a method
    q = typ.get_qualifier()
    if q == Qualified.default:
        context.cppyy_currentcall_this = builder.bitcast(val, ir_voidptr)

    elif q == Qualified.value:
        # TODO: take address of by value returns
        context.cppyy_currentcall_this = None

    else:
        assert not "unknown qualified type"

    return context.cppyy_currentcall_this


scope_numbatypes = (dict(), dict())

@nb_ext.typeof_impl.register(cpp_types.Scope)
def typeof_scope(val, c, q = Qualified.default):
    global scope_numbatypes

    try:
        return scope_numbatypes[q][val]
    except KeyError:
        pass

    if val.__cpp_reflex__(cpp_refl.IS_NAMESPACE):
        cnt = nb_types.Module(val)
        scope_numbatypes[Qualified.default][val] = cnt
        return cnt

    class ImplClassType(CppClassNumbaType):
        pass

    cnt = ImplClassType(val, q)
    scope_numbatypes[q][val] = cnt

  # declare data members to Numba
    data_members = list()
    for name, field in val.__dict__.items():
        if type(field) == cpp_types.DataMember:
            data_members.append(CppDataMemberInfo(
                name, field.__cpp_reflex__(cpp_refl.OFFSET), field.__cpp_reflex__(cpp_refl.TYPE))
            )

  # TODO: this refresh is needed b/c the scope type is registered as a
  # callable after the tracing started; no idea of the side-effects ...
    nb_reg.cpu_target.typing_context.refresh()

  # create a model description for Numba
    if q == Qualified.default:
        @nb_ext.register_model(ImplClassType)
        class ImplClassModel(nb_dm.models.StructModel):
            def __init__(self, dmm, fe_type):
                self._data_members = data_members

              # TODO: eventually we need not derive from StructModel
                members = [(dmi.f_name, dmi.f_nbtype) for dmi in data_members]
                nb_dm.models.StructModel.__init__(self, dmm, fe_type, members)

          # proxies are always accessed by pointer, which are not composites
            def traverse(self, builder):
                return []

            def traverse_models(self):
                return []

            def traverse_types(self):
                return [self._fe_type]      # from StructModel

          # data: representation used when storing into containers (e.g. arrays).
            # TODO ...

          # value: representation inside function body. Maybe stored in stack.
          #        The representation here are flexible.
            def get_value_type(self):
              # the C++ object, b/c through a proxy, is always accessed by pointer; it is represented
              # as a pointer to POD to allow indexing by Numba for data member type checking, but the
              # address offsetting for loading data member values is independent (see get(), below),
              # so the exact layout need not match a POD
                return ir.PointerType(super(ImplClassModel, self).get_value_type())

          # argument: representation used for function argument. Needs to be builtin type,
          #           but unlike other Numba composites, C++ proxies are no flattened.
            def get_argument_type(self):
                return self.get_value_type()

            def as_argument(self, builder, value):
                return value

            def from_argument(self, builder, value):
                return value

          # return: representation used for return argument.
            # TODO ...

          # access to public data members
            def get(self, builder, val, pos):
                """Get a field at the given position/field name"""
                if isinstance(pos, str):
                    pos = self.get_field_position(pos)
                dmi = self._data_members[pos]
                llval = builder.bitcast(val, ir_byteptr)
                pfc = builder.gep(llval, [ir.Constant(ir_intptr_t, dmi.f_offset)])
                pf = builder.bitcast(pfc, ir.PointerType(dmi.f_irtype))
                return builder.load(pf)

    elif q == Qualified.value:
        @nb_ext.register_model(ImplClassType)
        class ImplClassModel(nb_dm.models.StructModel):
            def __init__(self, dmm, fe_type):
                members = [(dmi.f_name, dmi.f_nbtype) for dmi in data_members]
                nb_dm.models.StructModel.__init__(self, dmm, fe_type, members)

    else:
        assert not "unknown qualified type"

  # Python proxy unwrapping for arguments into the Numba trace
    @nb_ext.unbox(ImplClassType)
    def unbox_instance(typ, obj, c):
        global cppyy_as_voidptr

        ptrty = ir.PointerType(ir.FunctionType(ir_voidptr, [ir_voidptr]))
        ptrval = c.context.add_dynamic_addr(c.builder, cppyy_as_voidptr, info='Instance_AsVoidPtr')
        fp = c.builder.bitcast(ptrval, ptrty)

        vptr = c.context.call_function_pointer(c.builder, fp, [obj])
        model = nb_dm.default_manager.lookup(typ)
        pobj = c.builder.bitcast(vptr, model.get_argument_type())

        return nb_ext.NativeValue(pobj, is_error=None, cleanup=None)

  # C++ object to Python proxy wrapping for returns from Numba trace
    @nb_ext.box(ImplClassType)
    def box_instance(typ, val, c):
        assert not "requires object model and passing of intact object, not memberwise copy"
        global cppyy_from_voidptr

        ir_pyobj = c.context.get_argument_type(nb_types.pyobject)
        ir_int   = cpp2ir('int')

        ptrty = ir.PointerType(ir.FunctionType(ir_pyobj, [ir_voidptr, cpp2ir('char*'), ir_int]))
        ptrval = c.context.add_dynamic_addr(c.builder, cppyy_from_voidptr, info='Instance_FromVoidPtr')
        fp = c.builder.bitcast(ptrval, ptrty)

        module = c.builder.basic_block.function.module
        clname = c.context.insert_const_string(module, typ._scope.__cpp_name__)

        NULL = c.context.get_constant_null(nb_types.voidptr)     # TODO: get the real thing
        return c.context.call_function_pointer(c.builder, fp, [NULL, clname, ir_int(0)])

    return cnt


#
# C++ instance -> Numba
#
@nb_ext.typeof_impl.register(cpp_types.Instance)
def typeof_instance(val, c):
    global scope_numbatypes

    try:
        return scope_numbatypes[Qualified.default][type(val)]
    except KeyError:
        pass

    return typeof_scope(type(val), c, Qualified.default)
