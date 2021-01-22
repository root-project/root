from abc import ABCMeta, abstractmethod
import functools

# Abstract class declaration
# This ensures compatibility between Python 2 and 3 versions, since in
# Python 2 there is no ABC class
ABC = ABCMeta('ABC', (object,), {})


class Backend(ABC):
    """
    Base class for RDataFrame backends. Subclasses of this class need to
    implement the 'execute' method.

    Attributes:
        supported_operations (list): List of operations supported by the
            backend.
        initialization (function): Store user's initialization method, if
            defined.
    """
    supported_operations = [
        'Define',
        'Filter',
        'Histo1D',
        'Histo2D',
        'Histo3D',
        'Profile1D',
        'Profile2D',
        'Profile3D',
        'Count',
        'Min',
        'Max',
        'Mean',
        'Sum',
        'Fill',
        'Report',
        'Range',
        'Take',
        'Snapshot',
        'Foreach',
        'Reduce',
        'Aggregate',
        'Graph',
        'AsNumpy'
    ]

    initialization = staticmethod(lambda: None)

    def __init__(self, config={}):
        """
        Creates a new instance of the desired implementation of :obj:`Backend`.

        Args:
            config (dict): The config object for the required backend. The
                default value is an empty Python dictionary: :obj:`{}`.
        """
        self.config = config

    @classmethod
    def register_initialization(cls, fun, *args, **kwargs):
        """
        Convert the initialization function and its arguments into a callable
        without arguments. This callable is saved on the backend parent class.
        Therefore, changes on the runtime backend do not require users to set
        the initialization function again.

        Args:
            fun (function): Function to be executed.

            *args (list): Variable length argument list used to execute the
                function.

            **kwargs (dict): Keyword arguments used to execute the function.
        """
        cls.initialization = functools.partial(fun, *args, **kwargs)
        fun(*args, **kwargs)

    def check_supported(self, operation_name):
        """
        Checks if a given operation is supported
        by the given backend.

        Args:
            operation_name (str): Name of the operation to be checked.

        Raises:
            Exception: This happens when `operation_name` doesn't exist
            the `supported_operations` instance attribute.
        """
        if operation_name not in self.supported_operations:
            raise Exception(
                "The current backend doesn't support \"{}\" !"
                .format(operation_name)
            )

    @abstractmethod
    def execute(self, generator):
        """
        Subclasses must define how to run the RDataFrame graph on a given
        environment.
        """
        pass
