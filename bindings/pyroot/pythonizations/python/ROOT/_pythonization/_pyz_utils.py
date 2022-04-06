# Author: Enric Tejedor CERN  4/2022

################################################################################
# Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

import abc
import sys


class MethodTemplateGetter(object):
    '''
    Instances of this class can be injected in class proxies to replace method
    templates that we want to pythonize. Similarly to what `partialmethod`
    does, this class implements `__get__` to return a wrapper object of a
    method that is bound to an instance of the pythonized class. Such object is
    both callable and subscriptable.

    Attributes:
        _original_method (cppyy TemplateProxy): original cppyy method template
            being pythonized.
        _wrapper_class (subclass of MethodTemplateWrapper): class that wraps a
            pythonized method template.
        _extra_args (tuple): extra arguments to be forwarded to
            `_wrapper_class`'s __init__ method, to be used by the wrapper object
            when receiving a call.
    '''

    def __init__(self, original_method, wrapper_class, *extra_args):
        '''
        Initializes the getter object of a method template.
        Saves the original implementation of the method template to be replaced
        (i.e. the one provided by cppyy) so that it can be later used in the
        implementation of the pythonization by the wrapper object.

        Args:
            original_method (cppyy TemplateProxy): original cppyy method
                template being pythonized.
            wrapper_class (subclass of MethodTemplateWrapper): class that wraps
                a pythonized method template.
            extra_args (tuple, optional): extra arguments to be forwarded to
                `wrapper_class`'s __init__ method, to be used by the wrapper
                object when receiving a call.
        '''
        self._original_method = original_method
        self._wrapper_class = wrapper_class
        self._extra_args = extra_args

    def __get__(self, instance, instance_type=None):
        '''
        Creates and returns a wrapper object for a method template. The type of
        the wrapper is a subclass of MethodTemplateWrapper.
        By implementing `__get__`, we obtain a handle to the instance of the
        pythonized class on which the application accessed the method template.
        That allows us to get an original implementation of the method template
        that is bound to that instance, and pass such implementation along to
        the wrapper object for later use.

        Args:
            instance (class instance): instance of the pythonized class on
                which the application accessed the method template.
            instance_type (class type): type of the instance.

        Returns:
            instance of MethodTemplateWrapper subclass: contains a handle to
                the original implementation of the method template that is
                bound to `instance` and, possibly, some extra arguments to be
                used when receiving a call.
        '''
        bound_method = self._original_method.__get__(instance, instance_type)
        return self._wrapper_class(bound_method, *self._extra_args)


# Needed below to define an abstract base class
if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta('ABC', (), {})

class MethodTemplateWrapper(ABC):
    '''
    Abstract base class that defines some common logic to properly pythonize
    method templates. More precisely, it provides an implementation of
    `__getitem__` that makes wrappers subscriptable and allows them to capture
    template arguments.
    Subclasses of this class must redefine `__call__` with the actual
    pythonization of the method template.

    Attributes:
        _original_method (cppyy TemplateProxy): original implementation of the
            method template that is bound to the instance on which the template
            was accessed.
        _extra_args (tuple): extra arguments to be used when receiving a call.
    '''

    def __init__(self, original_method, *extra_args):
        '''
        Constructor of a wrapper object for a method template.

        Args:
            original_method (cppyy TemplateProxy): original implementation of
                the method template that is bound to the instance on which the
                template was accessed.
            extra_args (tuple): extra arguments to be used when receiving a
                call.
        '''
        self._original_method = original_method
        self._extra_args = extra_args

    def __getitem__(self, template_args):
        '''
        Captures the template arguments used to instantiate the method template.

        Args:
            template_args (tuple): template arguments.

        Returns:
            instance of MethodTemplateWrapper subclass: a new wrapper instance
                is returned, with an original method on which the template
                arguments have been applied.
        '''
        return self.__class__(self._original_method[template_args],
                              *self._extra_args)

    @abc.abstractmethod
    def __call__(self, *args):
        '''
        Abstract method to be implemented by subclasses with the actual
        pythonization of the method template.
        '''
        pass

