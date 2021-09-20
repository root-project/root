# @author Enric Tejedor
# @date 2021-07

################################################################################
# Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

import os
from collections import namedtuple

import ROOT
RunGraphs = ROOT.RDF.RunGraphs

class CppWorkflow(object):
    '''
    Class that encapsulates the generation of the code of an RDataFrame workflow
    in C++, together with its compilation into a shared library and execution.

    This class is used by worker processes to execute in C++ the RDataFrame
    graph they receive. This is done for the sake of performance, since running
    the graph from Python necessarily relies on jitted code, which is less
    optimized and thus slower than a shared library compiled with ACLiC.

    Attributes:
        action_res_types (dict): result types for each supported action.

        cached_wfs (set): ids of workflow functions that have been already
            compiled and loaded by the current process. Used to prevent
            recompilation of already executed workflow functions.

        FUNCTION_NAME (string): name of the function that encapsulates the
            RDataFrame graph creation

        FUNCTION_NAMESPACE (string): namespace of the function that
            encapsulates the RDataFrame graph creation

        graph_nodes (list): statements that add nodes to the graph.

        includes (list): include statements needed by the workflow.

        lambdas (list): lambda functions used by the workflow.

        lambda_id (int): counter used to generate ids for each defined lambda
            function.

        node_id (int): counter used to generate ids for each graph node.

        py_actions (list): list that contains PyActionData pairs
            `(res_id,operation)`, where `res_id` is the Python action result
            index in the vector of workflow results and `operation` is the
            Operation object corresponding to the action.

        res_ptr_id (int): counter used to generate ids for each result generated
            by graph actions.

        snapshots (list): list that contains SnapshotData pairs
            `(res_id,filename)`, where `res_id` is the Snapshot result index in
            the vector of workflow results and `filename` is its modified output
            file name.
    '''

    FUNCTION_NAME = 'GenerateGraph'
    FUNCTION_NAMESPACE = 'DistRDF_Internal'

    cached_wfs = set()

    SnapshotData = namedtuple('SnapshotData', ['res_id', 'filename'])
    PyActionData = namedtuple('PyActionData', ['res_id', 'operation'])

    class ListWithLifeline(list):
        '''
        List-like helper class that is constructed from the elements stored in
        an std::vector. Moreover, it stores a reference to the std::tuple that
        contains the vector, so that the life of the tuple is tied to that
        of the new list.

        Attributes:
            v (std::vector): vector used to construct the list.
            t (std::tuple): tuple that contains `v`. It must not be destroyed
                before the list is.
        '''
        def __init__(self, v, t):
            super(CppWorkflow.ListWithLifeline, self).__init__()
            for elem in v:
                self.append(elem)
            self.t = t  # set lifeline for tuple

    def __init__(self, head_node, range_id):
        '''
        Generates the C++ code of an RDF workflow that corresponds to the
        received graph and data range.

        Args:
            head_node (Node): head node of a graph that represents an RDF
                workflow.
            range_id (int): id of the data range to be processed by this
                workflow. Needed to assign a name to a partial Snapshot output
                file.
        '''

        self.includes = '''
#include "ROOT/RDataFrame.hxx"
#include "ROOT/RResultHandle.hxx"
#include "ROOT/RDFHelpers.hxx"

#include <vector>
#include <tuple>
'''

        self.lambdas = ''
        self.lambda_id = 0

        self.graph_nodes = ''
        self.node_id = 1  # 0 is the head node we receive

        self.res_ptr_id = 0

        self.snapshots = []

        self.py_actions = []

        # Generate the C++ workflow.
        # Recurse over children nodes of received graph head node
        head_node_id = 0
        for child_node in head_node.children:
            self._explore_graph(child_node, range_id, head_node_id)

    def _explore_graph(self, node, range_id, parent_id):
        """
        Recursively traverses the graph nodes in DFS order and, for each of
        them, adds a new node to the C++ workflow.

        Args:
            node (Node): object that contains the information to add the
                corresponding node to the C++ workflow.
            range_id (int): id of the current range. Needed to assign a name to a
                partial Snapshot output file.
            parent_id (int): id of the parent node in the C++ workflow.
        """
        node_id = self.add_node(node.operation, range_id, parent_id)

        for child_node in node.children:
            self._explore_graph(child_node, range_id, node_id)

    def add_include(self, header):
        '''
        Adds a new include statement.

        Args:
            header (str): header to be included.
        '''

        self.includes += '\n#include "{}"'.format(header)

    def add_lambda(self, lambda_code):
        '''
        Adds a new lambda to be defined, which is needed by some operation
        in the workflow.

        Args:
            lambda_code (str): code of the lambda to be defined.
        '''

        self.lambdas += '\n  auto rdf_lambda{n} = {code};' \
                        .format(n=lambda_id, code=lambda_code)
        self.lambda_id += 1

    def add_node(self, operation, range_id, parent_id):
        '''
        Adds the corresponding statement to add a new node in the RDataFrame
        graph. What the statement returns depends on the type of the
        operation:
        - Transformation: the statement produces a new dataset node.
        - Action: the statement returns an RResultPtr, which is added to a
        vector of results to be returned at the end of the workflow
        generation function. The type of the elements of the vector is
        ROOT::RDF::RResultHandle to do type erasure since there could be
        multiple booked actions with results of different types (e.g. integers,
        TH1D, etc.).

        Args:
            operation (Operation): object representing the operation to be added
            to the graph.

            range_id (int): id of the current range. Needed to assign a name to
            a partial Snapshot output file.

            parent_id (int): id of the parent node in this workflow. Used to
                connect the new node to be added with its parent.

        Returns:
            int: identifier of the node that was added to the graph
            corresponding to its DFS order, if the operation is a
            transformation, or None, if the operation is an action.
        '''

        # Operations that need special treatment
        if operation.name == 'AsNumpy':
            self._handle_asnumpy(operation, parent_id)
            return None # nothing else to do

        if operation.name == 'Snapshot':
            self._handle_snapshot(operation, range_id) # this modifies operation.args

        # Generate the code of the call that creates the new node
        op_call = 'rdf{n}.{op}{templ_args}({args});' \
                  .format(n=parent_id, op=operation.name, \
                          templ_args=self._get_template_args(operation), \
                          args=self._get_call_args(operation))

        if operation.is_transformation():
            new_node_id = self.node_id
            self.graph_nodes += '\n  auto rdf{n} = {call}' \
                                .format(n=new_node_id, call=op_call)
            self.node_id += 1
            return new_node_id

        # Else it's an action or instant action
        self.graph_nodes += '\n  auto res_ptr{n} = {call}' \
                            .format(n=self.res_ptr_id, call=op_call)

        # The result is stored in the vector of results to be returned
        self.graph_nodes += '\n  result_handles.emplace_back(res_ptr{});' \
                            .format(self.res_ptr_id)

        # The result type is stored in the vector of result types to be
        # returned
        self.graph_nodes += \
            '\n  auto c{} = TClass::GetClass(typeid(res_ptr{}));' \
            .format(self.res_ptr_id, self.res_ptr_id)
        self.graph_nodes += \
            '\n  result_types.emplace_back(c{}->GetName());' \
            .format(self.res_ptr_id)

        self.res_ptr_id += 1

    def _handle_snapshot(self, operation, range_id):
        '''
        Does two extra settings needed for Snapshot nodes:
        - Modifies the output file name to be of the form `filename_rangeid`,
        since it is a partial snapshot for a given range.
        - Stores the index of the returned vector<RResultHandle> in which the
        result of this Snapshot is stored, together with the modified file
        path.

        Args:
            operation (Operation): object representing the operation to be added
                to the graph.

            range_id (int): id of the current range. Needed to assign a name to
                a partial Snapshot output file.
        '''

        # Modify file name
        filename = operation.args[1].partition('.root')[0]
        path_with_range = '{filename}_{rangeid}.root' \
                          .format(filename=filename, rangeid=range_id)
        operation.args[1] = path_with_range

        # Store Snapshot result index -> path
        self.snapshots.append(
            CppWorkflow.SnapshotData(self.res_ptr_id, path_with_range))

    def _handle_asnumpy(self, operation, parent_id):
        '''
        Since AsNumpy is a Python-only action, it can't be included in the
        C++ workflow built by this class. Therefore, this function takes care
        of saving the RDF node, generated in C++, on which an AsNumpy action
        should be applied from Python.
        This function also stores the index of the AsNumpy result in the final
        list of results, where the index depends on the DFS traversal of the
        computation graph.

        Args:
            operation (Operation): object representing the AsNumpy operation
            
            parent_id (int): id of the parent node in this workflow. Used to
                save that node in the vector of output nodes.
        '''

        self.py_actions.append(CppWorkflow.PyActionData(self.res_ptr_id, operation))
        self.graph_nodes += \
            '\n  output_nodes.push_back(ROOT::RDF::AsRNode(rdf{}));' \
            .format(parent_id)
        self.res_ptr_id += 1

    def _get_template_args(self, operation):
        '''
        Gets the template arguments with which to generate the call to a given
        operation.

        Args:
            operation (Operation): object representing the operation whose
                template arguments need to be returned.

        Returns:
            string: template arguments for this operation.
        '''

        # TODO: generate templated operations when possible, e.g. Max<double>

        return ''

    def _get_call_args(self, operation):
        '''
        Gets the arguments with which to generate the call to a given operation.

        Args:
            operation (Operation): object representing the operation whose
                call arguments need to be returned.

        Returns:
            string: call arguments for this operation.
        '''

        # TODO
        # - Do a more thorough type conversion
        # - Use RDF helper functions to convert jitted strings to lambdas

        args = ""

        # Argument type conversion
        for narg, arg in enumerate(operation.args):
            if (narg > 0):
                args += ', '

            if isinstance(arg, str):
                args += '"{}"'.format(arg)
            elif isinstance(arg, tuple):
                args += '{'
                for nelem, elem in enumerate(arg):
                    if nelem > 0:
                        args += ','
                    if isinstance(elem, str):
                        args += '"{}"'.format(elem)
                    else:
                        args += '{}'.format(elem)
                args += '}'

        # Make Snapshot lazy
        # TODO: Do a proper processing of the args (user might have specified
        # her own options object)
        if operation.name == 'Snapshot':
            args += ', "", lazy_options'

        return args

    def execute(self, rdf):
        '''
        Compiles the workflow generation code and executes it.

        Args:
            rdf (ROOT::RDF::RNode): object that represents the dataset on
                which to execute the workflow.

        Returns:
            tuple: the first element is the list of results of the actions in
                the C++ workflow, the second element is the list of result types
                corresponding to those actions.
        '''

        function_id = self._compile()
        return self._run_function(rdf, function_id)

    def _compile(self):
        '''
        Generates the workflow code C++ file and compiles it with ACLiC
        into a shared library. The library is also loaded as part of the
        `TSystem::CompileMacro` call.

        The name of the generated C++ file contains both a hash of its
        code and the ID of the process that created it. This is done to
        prevent clashes between multiple (non-sandboxed) worker processes
        that try to write to the same file concurrently.

        A class-level cache keeps track of the workflows that have been already
        compiled to prevent unncessary recompilation (e.g. when a worker
        process runs multiple times the same workflow).

        Returns:
            string: the id of the function to be executed. Such id is appended
                to CppWorkflow.FUNCTION_NAME to prevent name clashes (a worker
                process might compile and load multiple workflow functions).
                A function id is mostly the hash of its code.
        '''

        # TODO: Make this function thread-safe? To support Dask threaded
        # workers

        code = self._get_code()
        code_hash = hash(code)
        # A hash can be a negative value, replace '-' with '_' so it can be
        # part of a function name
        function_id = str(code_hash).replace('-', '_', 1)
        if code_hash in CppWorkflow.cached_wfs:
            # We already compiled and loaded a workflow function with this
            # code. Return the id of that function
            return function_id

        # We are trying to run this workflow for the first time in this
        # process. First dump the code in a file with the right function name
        cpp_file_name = 'rdfworkflow_{code_hash}_{pid}.cpp' \
                        .format(code_hash=code_hash, pid=os.getpid())
        code = code.replace(CppWorkflow.FUNCTION_NAME, CppWorkflow.FUNCTION_NAME + function_id, 1)
        with open(cpp_file_name, 'w') as f:
            f.write(code)

        # Now compile and load the code
        if not ROOT.gSystem.CompileMacro(cpp_file_name, 'O'):
            raise RuntimeError(
                'Error compiling the RDataFrame workflow file: {}' \
                .format(cpp_file_name))

        # Let the cache know there is a new workflow
        CppWorkflow.cached_wfs.add(code_hash)

        return function_id

    def _run_function(self, rdf, function_id):
        '''
        Runs the workflow generation function.

        Args:
            rdf (ROOT::RDF::RNode): object that represents the dataset on
                which to execute the workflow.
            function_id (string): identifier of the workflow function to be
                executed.

        Returns:
            tuple: the first element is the list of results of the actions in
                the C++ workflow, the second element is the list of result types
                corresponding to those actions.
        '''

        ns = getattr(ROOT, CppWorkflow.FUNCTION_NAMESPACE)
        func = getattr(ns, CppWorkflow.FUNCTION_NAME + function_id)

        # Run the workflow generator function
        vectors = func(rdf)
        v_results, v_res_types, v_nodes = vectors

        # Convert the vector of results into a list so that we can mix
        # different types in it.
        # The std::tuple `vectors` is passed as parameter to tie its life with
        # that of the new list
        results = self.ListWithLifeline(v_results, vectors)

        # Strip out the ROOT::RDF::RResultPtr<> part of the type
        def get_result_type(s):
            s = str(s)
            pos = s.find('<')
            if pos == -1:
                raise RuntimeError(
                    'Error parsing the result types of RDataFrame workflow')
            return s[pos+1:-1].strip()

        res_types = [ get_result_type(elem) for elem in v_res_types ]

        # Add Python-only actions on their corresponding nodes
        for (i, operation), n in zip(self.py_actions, v_nodes):
            operation.kwargs['lazy'] = True  # make it lazy
            res = getattr(n, operation.name)(*operation.args, **operation.kwargs)
            results.insert(i, res)
            res_types.insert(i, None) # placeholder

        if v_results:
            # We trigger the event loop here, so make sure we release the GIL
            old_rg = RunGraphs.__release_gil__
            RunGraphs.__release_gil__ = True
            RunGraphs(v_results)
            RunGraphs.__release_gil__ = old_rg

        # Replace the RResultHandle of each Snapshot by its modified output
        # path, since the latter is what we actually need in the reducer
        for i, path in self.snapshots:
            results[i] = [path]
            res_types[i] = None # placeholder

        # Replace the future-like result of every Python-only action (e.g.
        # AsNumpyResult) by its actual value
        for i, operation in self.py_actions:
            results[i] = results[i].GetValue()

        return results, res_types

    def __repr__(self):
        '''
        Generates a string representation for this C++ workflow.

        Returns:
            string: code of this C++ workflow.
        '''

        return self._get_code()

    def _get_code(self):
        '''
        Composes the workflow generation code from the different attributes
        of this class. The resulting code contains a function that will be
        called to generate the RDataFrame graph. Such function returns a tuple
        of three elements:
        1. A vector of results of the graph actions.
        2. A vector with the result types of those actions.
        3. A vector of RDF nodes that will be used in Python to invoke
        Python-only actions on them (e.g. `AsNumpy`).
        '''

        code = '''
{includes}

namespace {namespace} {{

using CppWorkflowResult = std::tuple<std::vector<ROOT::RDF::RResultHandle>,
                          std::vector<std::string>,
                          std::vector<ROOT::RDF::RNode>>;

CppWorkflowResult {func_name}(ROOT::RDF::RNode &rdf0)
{{
  std::vector<ROOT::RDF::RResultHandle> result_handles;
  std::vector<std::string> result_types;
  std::vector<ROOT::RDF::RNode> output_nodes;

  // To make Snapshots lazy
  ROOT::RDF::RSnapshotOptions lazy_options;
  lazy_options.fLazy = true;

{lambdas}

{nodes}

  return {{ result_handles, result_types, output_nodes }};
}}

}}
'''.format(func_name=CppWorkflow.FUNCTION_NAME,
           namespace=CppWorkflow.FUNCTION_NAMESPACE,
           includes=self.includes, lambdas=self.lambdas, nodes=self.graph_nodes)

        return code

