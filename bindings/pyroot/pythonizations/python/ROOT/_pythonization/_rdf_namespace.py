# Author: Vincenzo Eduardo Padulano CERN 10/2024

################################################################################
# Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

"""
This module contains utilities to help in the organization of the RDataFrame
namespace and the interaction between the C++ and Python functionalities
"""


def _create_distributed_module(parent):
    """
    Create the ROOT.RDF.Experimental.Distributed python module.

    This module will be injected into the ROOT.RDF namespace.

    Arguments:
        parent: The ROOT.RDF namespace. Needed to define __package__.

    Returns:
        types.ModuleType: The ROOT.RDF.Experimental.Distributed submodule.
    """
    import DistRDF

    return DistRDF.create_distributed_module(parent)


def _rungraphs(distrdf_rungraphs, rdf_rungraphs):
    """
    Create a callable that correctly dispatches either to the local or
    distributed version of RunGraphs.
    """

    def rungraphs(handles):
        # Caveat: we should not call `hasattr` on the result pointer, since
        # this will implicitly trigger the connected computation graph
        if len(handles) > 0 and "DistRDF" in str(type(handles[0])):
            return distrdf_rungraphs(handles)
        else:
            return rdf_rungraphs(handles)

    return rungraphs


def _variationsfor(distrdf_variationsfor, rdf_variationsfor):
    """
    Create a callable that correctly dispatches either to the local or
    distributed version of VariationsFor.
    """

    def variationsfor(resptr):
        # Caveat: we should not call `hasattr` on the result pointer, since
        # this will implicitly trigger the connected computation graph
        if "DistRDF" in str(type(resptr)):
            return distrdf_variationsfor(resptr)
        else:
            # Help local VariationsFor with the type of the value held by the result pointer
            inner_type = type(resptr).__name__
            inner_type = inner_type[
                inner_type.index("<") + 1: inner_type.rindex(">")]
            return rdf_variationsfor[inner_type](resptr)

    return variationsfor


def _rdataframe(local_rdf, distributed_rdf):
    """
    Create a callable that correctly dispatches either to the local or
    distributed RDataFrame constructor, depending on whether the "executor"
    keyword argument is absent or not.
    """

    def rdataframe(*args, **kwargs):
        import ROOT
        from libROOTPythonizations import PyObjRefCounterAsStdAny

        if kwargs.get("executor", None) is not None:
            rdf = distributed_rdf(*args, **kwargs)
            rnode = ROOT.RDF.AsRNode(rdf._headnode.rdf_node)
        else:
            rdf = local_rdf(*args, **kwargs)
            rnode = ROOT.RDF.AsRNode(rdf)

        if args and isinstance(args[0], ROOT.TTree):
            ROOT.Internal.RDF.SetTTreeLifeline(
                rnode, PyObjRefCounterAsStdAny(args[0]))

        return rdf

    return rdataframe
