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
from numba.extending import make_attribute_wrapper
import re

# setuptools entry point for Numba
def _init_extension():
    pass


class Qualified:
    default   = 0
    value     = 1
    instance  = 2

ir_byte     = ir.IntType(8)
ir_voidptr  = ir.PointerType(ir_byte)                 # by convention
ir_byteptr  = ir_voidptr                              # for clarity
ir_intptr_t = ir.IntType(cppyy.sizeof('void*')*8)

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
    'int'                    : nb_types.intc,
    'unsigned int'           : nb_types.uintc,
    'int32_t'                : nb_types.int32,
    'uint32_t'               : nb_types.uint32,
    'int64_t'                : nb_types.int64,
    'uint64_t'               : nb_types.uint64,
    'long'                   : nb_types.long_,
    'unsigned long'          : nb_types.ulong,
    'long long'              : nb_types.longlong,
    'unsigned long long'     : nb_types.ulonglong,
    'float'                  : nb_types.float32,
    'double'                 : nb_types.float64,
    'char'                   : nb_types.char,
    'unsigned char'          : nb_types.uchar,
    'char*'                  : nb_types.unicode_type
}

def resolve_std_vector(val):
    return re.match(r'std::vector<(.+?)>', val).group(1)

def resolve_const_types(val):
    return re.match(r'const\s+(.+)\s*\*', val).group(1)

def cpp2numba(val):
    if not isinstance(val, str):
        # TODO: distinguish ptr/ref/byval
        # TODO: Only metaclasses/proxies end up here since
        #  ref cases makes the RETURN_TYPE from reflex a string
        return typeof_scope(val, nb_typing.typeof.Purpose.argument, Qualified.value)
    elif val.startswith("std::vector"):
        type_arr = getattr(numba, str(cpp2numba(resolve_std_vector(val))))[:]
        return type_arr
    elif val[-1] == '*' or val[-1] == '&':
        if val.startswith('const'):
            return nb_types.CPointer(cpp2numba(resolve_const_types(val)))
        return nb_types.CPointer(_cpp2numba[val[:-1]])
    return _cpp2numba[val]

_numba2cpp = dict()
for key, value in _cpp2numba.items():
    _numba2cpp[value] = key
# prefer "int" in the case of intc over "int32_t"
_numba2cpp[nb_types.intc] = 'int'

def numba2cpp(val):
    if hasattr(val, 'literal_type'):
        val = val.literal_type
        if val == nb_types.int64:      # Python int
            # TODO: this is only necessary until "best matching" is in place
            val = nb_types.intc        # more likely match candidate
    elif isinstance(val, numba.types.CPointer):
        return _numba2cpp[val.dtype]
    elif isinstance(val, numba.types.RawPointer):
        return _numba2cpp[nb_types.voidptr]
    elif isinstance(val, numba.types.Array):
        return "std::vector<" + _numba2cpp[val.dtype] + ">"
    elif isinstance(val, CppClassNumbaType):
        return val._scope.__cpp_name__
    else:
        try:
            return _numba2cpp[val]
        except:
            raise RuntimeError("Type mapping failed from Numba to C++ for ", val)

def numba_arg_convertor(args):
    args_cpp = []
    for arg in list(args):
        # If the user explicitly passes an argument using numba CPointer, the regex match is used
        # to detect the pass by reference since the dispatcher always returns typeref[val*]
        match = re.search(r"typeref\[(.*?)\*\]", str(arg))
        if match:
            literal_val = match.group(1)
            arg_type = numba.typeof(eval(literal_val))
            args_cpp.append(to_ref(numba2cpp(arg_type)))
        else:
            args_cpp.append(numba2cpp(arg))
    return tuple(args_cpp)

def to_ref(type_list):
    ref_list = []
    for l in type_list:
        ref_list.append(l + '&')
    return ref_list

# TODO: looks like Numba treats unsigned types as signed when lowering,
# which seems to work as they're just reinterpret_casts
_cpp2ir = {
    'char*'                  : ir_byteptr,
    'int8_t'                 : ir.IntType(8),
    'uint8_t'                : ir.IntType(8),
    'short'                  : ir.IntType(nb_types.short.bitwidth),
    'unsigned short'         : ir.IntType(nb_types.ushort.bitwidth),
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
    try:
        return _cpp2ir[val]
    except KeyError:
        if val.startswith("std::vector"):
            ## TODO should be possible to obtain the vector length from the CPPDataMember val
            type_arr = ir.VectorType(cpp2ir(resolve_std_vector(val)), 3)
            return type_arr
        elif val != "char*" and val[-1] == "*":
            if val.startswith('const'):
                return ir.PointerType(cpp2ir(resolve_const_types(val)))
            type_2 = _cpp2ir[val[:-1]]
            return ir.PointerType(type_2)

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
        self._arg_set_matched = tuple()
        self.ret_type = None

    def is_precise(self):
        return True          # by definition

    def get_call_type(self, context, args, kwds):
        try:
            return self._impl_keys[args].sig
        except KeyError:
            pass

        ol = CppFunctionNumbaType(
                self._func.__overload__(numba_arg_convertor(args)), self._is_method)

        thistype = None
        if self._is_method:
            thistype = nb_types.voidptr

        self.ret_type = cpp2numba(ol._func.__cpp_reflex__(cpp_refl.RETURN_TYPE))
        ol.sig = nb_typing.Signature(
            return_type=self.ret_type,
            args=args,
            recvr=thistype)

        extsig = ol.sig
        if self._is_method:
            self.ret_type = ol.sig.return_type
            args = (nb_types.voidptr, *args)
            extsig = nb_typing.Signature(
                return_type=ol.sig.return_type, args=args, recvr=None)

        self._impl_keys[args] = ol
        self._arg_set_matched = numba_arg_convertor(args)


        @nb_iutils.lower_builtin(ol, *args)
        def lower_external_call(context, builder, sig, args,
                ty=nb_types.ExternalFunctionPointer(extsig, ol.get_pointer),
                pyval=self._func, is_method=self._is_method):
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

    # TODO: Remove the redundancy of __overload__ matching and use this function
    # to only obtain the address given the matched overload
    def get_pointer(self, func):
        if func is None:
            func = self._func

        ol = func.__overload__(numba_arg_convertor(self.sig.args))

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
        addr = None
        cppinstance_val = None
        if qualifier == Qualified.instance:
            addr = cppyy.addressof(scope)
            cppinstance_val = scope
            scope = type(scope)
            qualifier = Qualified.default
        super(CppClassNumbaType, self).__init__(scope.__init__)
        self.name = 'CppClass(%s)' % scope.__cpp_name__    # overrides value in Type
        self._scope     = scope
        self._qualifier = qualifier
        self._cppinstanceval = cppinstance_val
        self._addr = addr

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
            if isinstance(f, cpp_types.Function):
                ft = CppFunctionNumbaType(f, is_method=True)
        except AttributeError:
            pass

        try:
            f = typ._scope.__dict__[attr]
            if isinstance(f, cpp_types.DataMember):
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
    if attr in typ._scope.__dict__ and isinstance(typ._scope.__dict__[attr], cpp_types.DataMember):
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
        return builder.bitcast(val, ir_voidptr)

    elif q == Qualified.value:
        return None

    assert not "unknown qualified type"
    return None


class ImplAggregateValueModel(nb_dm.models.StructModel):
    def get(self, builder, val, pos):
        """Get a field at the given position/field name"""

        if isinstance(pos, str):
            pos = self.get_field_position(pos)

      # Use the offsets for direct addressing, rather than getting the elements
      # from the struct type.
        dmi = self._data_members[pos]

        stack = nb_cgu.alloca_once(builder, self.get_data_type())
        builder.store(val, stack)

        llval = builder.bitcast(stack, ir_byteptr)
        pfc = builder.gep(llval, [ir.Constant(ir_intptr_t, dmi.f_offset)])
        pf = builder.bitcast(pfc, ir.PointerType(dmi.f_irtype))

        return builder.load(pf)

class ImplClassValueModel(ImplAggregateValueModel):
  # TODO : Should the address have to be passed here and stored in meminfo
  # value: representation inside function body. Maybe stored in stack.
  #        The representation here are flexible.
    def get_value_type(self):
        return self.get_data_type()

  # data: representation used when storing into containers (e.g. arrays).
    def get_data_type(self):
      # The struct model relies on data being a POD, but for C++ objects, there
      # can be hidden data (e.g. vtable, thunks, or simply private members), and
      # the alignment of Cling and Numba also need not be the same. Therefore, the
      # struct is split in a series of byte members to get the total size right
      # and to allow addressing at the correct offsets.
        if self._data_type is None:
            self._data_type = \
                    ir.LiteralStructType([ir_byte for i in range(self._sizeof)], packed=True)
        return self._data_type

  # return: representation used for return argument.
    def get_return_type(self):
        return self.get_data_type()


scope_numbatypes = (dict(), dict())

@nb_ext.typeof_impl.register(cpp_types.Scope)
def typeof_scope(val, c, q = Qualified.default):
    is_instance = False
    cppinstance_val = None
    if q == Qualified.instance:
        cppinstance_val = val
        val = type(val)
        q = Qualified.default
        is_instance = True

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

    if is_instance:
        cnt = ImplClassType(cppinstance_val, Qualified.instance)
    else:
        cnt = ImplClassType(val, q)

    scope_numbatypes[q][val] = cnt

  # declare data members to Numba
    data_members = list()
    member_methods = dict()

    for name, field in val.__dict__.items():
        if isinstance(field, cpp_types.DataMember):
            data_members.append(CppDataMemberInfo(
                name, field.__cpp_reflex__(cpp_refl.OFFSET), field.__cpp_reflex__(cpp_refl.TYPE))
            )
        elif isinstance(field, cpp_types.Function):
            member_methods[name] = field.__cpp_reflex__(cpp_refl.RETURN_TYPE)

  # TODO: this refresh is needed b/c the scope type is registered as a
  # callable after the tracing started; no idea of the side-effects ...
    nb_reg.cpu_target.typing_context.refresh()

  # create a model description for Numba
    if q == Qualified.default:
        @nb_ext.register_model(ImplClassType)
        class ImplClassModel(nb_dm.models.StructModel):
            def __init__(self, dmm, fe_type):
                self._data_members = data_members
                self._member_methods = member_methods

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
              # the C++ object, b/c through a proxy, is always accessed by pointer; it is
              # represented as a pointer to POD to allow indexing by Numba for data member
              # type checking, but the address offsetting for loading data member values is
              # independent (see get(), below), so the exact layout need not match a POD

              # TODO: this doesn't work for real PODs, b/c those are unpacked into their elements
              # and passed through registers
                return ir.PointerType(super(ImplClassModel, self).get_value_type())

          # argument: representation used for function argument. Needs to be builtin type,
          #           but unlike other Numba composites, C++ proxies are not flattened.
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
        if val.__cpp_reflex__(cpp_refl.IS_AGGREGATE):
            @nb_ext.register_model(ImplClassType)
            class ImplClassModel(ImplAggregateValueModel):
                pass
        else:
            @nb_ext.register_model(ImplClassType)
            class ImplClassModel(ImplClassValueModel):
                pass

        def init(self, dmm, fe_type, sz = cppyy.sizeof(val)):
            self._data_members = data_members
            self._member_methods = member_methods
            self._sizeof = sz

          # TODO: this code exists purely to be able to use the indexing and hierarchy
          # of the base class StructModel, which isn't much of a reason
            members = [(dmi.f_name, dmi.f_nbtype) for dmi in data_members]
            nb_dm.models.StructModel.__init__(self, dmm, fe_type, members)

        ImplClassModel.__init__ = init

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

    def make_implclass(context, builder, typ, **kwargs):
        return nb_cgu.create_struct_proxy(typ)(context, builder, **kwargs)

  # C++ object to Python proxy wrapping for returns from Numba trace
    @nb_ext.box(ImplClassType)
    def box_instance(typ, val, c):
        assert not "requires object model and passing of intact object, not memberwise copy"

        global cppyy_from_voidptr

        if isinstance(val, ir.Constant):
            if val.constant == ir.Undefined:
                assert not "Value passed to instance boxing is undefined"
                return NULL

        implclass = make_implclass(c.context, c.builder, typ)
        classobj = c.pyapi.unserialize(c.pyapi.serialize_object(cpp_types.Instance))

        box_list = []

        model = implclass._datamodel
        cfr = CppClassFieldResolver(c.context)

        for i in typ._scope.__dict__:
            if isinstance(cfr.generic_resolve(typ, i), nb_types.Type):
                box_list.append(c.box(cfr.generic_resolve(typ, i), getattr(implclass, i)))

        box_res = c.pyapi.call_function_objargs(
            classobj, tuple(box_list)
        )
        # Required for nopython mode, numba nrt requres each member box call to decref
        # since it steals the reference
        for i in box_list:
            c.pyapi.decref(i)

        return box_res

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
    # Pass the val itself to obtain Cling address of the CPPInstance for reference to C++ objects
    return typeof_scope(val, c, Qualified.instance)
