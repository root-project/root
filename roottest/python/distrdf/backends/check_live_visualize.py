import os, threading, tempfile
from functools import partial
import pytest

from DistRDF import LiveVisualize

import ROOT

# Callback functions
def set_marker(graph):
    graph.SetMarkerStyle(5)

def set_marker_two_args(graph, marker):
    graph.SetMarkerStyle(marker)

def set_line_color(graph):
    graph.SetLineColor(ROOT.kRed)

def delete_graph(graph):
    graph.Delete()

def print_num_data_points(graph, data_points_file):
    num_points = graph.GetN()
    with threading.Lock():
        with open(data_points_file, "a") as file:
            file.write(f"{num_points}\n")

def check_data_points_increase(filename, num_entries):
    with open(filename, "r") as file:
        num_points = [int(line) for line in file]
        return all(curr > prev for prev, curr in zip(num_points, num_points[1:])) and num_points[-1] == num_entries


class TestLiveVisualizationCallback:
    """
    Tests the functionning of LiveVisualize with callback functions.
    """
    def test_data_points_increase(self, payload):
        """
        Tests that the number of data points in conitnuously increasing 
        at each step of the live visualization indicating that 
        the merging is working correctly.
        """
        connection, backend = payload
        if backend != "dask":
            # This feature is only available with the Dask backend
            return
        num_entries = 50
        d = ROOT.RDataFrame(num_entries, executor=connection)
        graph = d.Define("x", "rdfentry_").Define("y", "x*x").Graph("x", "y")

        # Create a temp file to store the number of data points
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file_path = temp_file.name

        # Apply the LiveVisualize function to the graph and pass a callback function
        # to continuously write the number of data points in a file
        LiveVisualize({graph: partial(print_num_data_points, data_points_file=temp_file_path)})
        
        graph.Draw("APL")

        # Assert that callback functions have been applied correctly
        assert check_data_points_increase(temp_file_path, num_entries) == True

        # Remove the created temp file
        temp_file.close()
        os.unlink(temp_file_path)
        
    def test_multiple_callbacks(self, payload):
        """
        Tests that the callback functions are correctly 
        setting multiple graphs' attributes.
        """
        connection, backend = payload
        if backend != "dask":
            # This feature is only available with the Dask backend
            return
        d = ROOT.RDataFrame(100, executor=connection)
        dd = d.Define("x", "rdfentry_").Define("y1", "x*x").Define("y2", "x+10")
        graph1 = dd.Graph("x", "y1")
        graph2 = dd.Graph("x", "y2")

        # Apply the LiveVisualize function to the graph and pass styling functions
        LiveVisualize({graph1: set_marker, graph2: set_line_color})
        
        # Assert that the callback functions have been applied correctly
        assert graph1.GetMarkerStyle() == 5 and graph2.GetLineColor() == 632

    def test_partial_callback(self, payload):
        """
        Tests that that we can use functools.partial to pass more
        than 1 argument to the callback functions.
        """
        connection, backend = payload
        if backend != "dask":
            # This feature is only available with the Dask backend
            return
        d = ROOT.RDataFrame(100, executor=connection)
        graph = d.Define("x", "rdfentry_").Define("y1", "x*x").Define("y2", "x+10").Graph("x", "y1")

        # Use functools.partial to pass a callback that takes 2 arguments 
        # to LiveVisualize making sure only 1 remains unfilled
        LiveVisualize([graph], partial(set_marker_two_args, marker=10))
        
        # Assert that the callback functions have been applied correctly
        assert graph.GetMarkerStyle() == 10
    

class TestInvalidArgument:
    """
    Tests handling of invalid arguments in LiveVisualize.
    """
    def test_plot_argument(self, payload):
        """
        Tests passing invalid plot objects.
        """
        connection, backend = payload
        if backend != "dask":
            # This feature is only available with the Dask backend
            return
        d = ROOT.RDataFrame(100, executor=connection)
        df = d.Define("x", "rdfentry_").Define("y", "x*x")
        graph = df.Graph("x", "y")
        
        with pytest.raises(AttributeError):
            # Pass None as the first argument
            LiveVisualize([None], set_marker)
        
        with pytest.raises(TypeError):
            # Pass a non-iterable as the first argument
            LiveVisualize(graph)  

    def test_callback_argument(self, payload):
        """
        Tests passing invalid callback functions.
        """
        connection, backend = payload
        if backend != "dask":
            # This feature is only available with the Dask backend
            return
        d = ROOT.RDataFrame(100, executor=connection)
        graph = d.Define("x", "rdfentry_").Define("y", "x*x").Graph("x", "y")

        with pytest.warns(UserWarning, match="The provided callback is not callable. Skipping callback."):
            # Pass a non-callable as callback
            LiveVisualize({graph: graph})

        with pytest.warns(UserWarning, match="The provided callback function contains blocked actions. Skipping callback."):
            # Pass a callback with a blocked action
            LiveVisualize({graph: delete_graph})

        with pytest.warns(UserWarning, match="The callback function should have exactly one parameter to fill. Skipping callback."):
            # Pass a callback with more than 1 argument to fill
            LiveVisualize({graph: set_marker_two_args})


class TestUnsupportedCall:
    """
    Tests handling of unsupported cases for LiveVisualize.
    """
    def test_call_with_two_rdataframes(self, payload):
        """
        Tests passing plots from different RDataframes.
        """
        connection, backend = payload
        if backend != "dask":
            # This feature is only available with the Dask backend
            return
        d1 = ROOT.RDataFrame(100, executor=connection)
        d2 = ROOT.RDataFrame(50, executor=connection)

        graph1 = d1.Define("x1", "rdfentry_").Define("y1", "x1*x1").Graph("x1", "y1")
        graph2 = d2.Define("x2", "rdfentry_").Define("y2", "x2+10").Graph("x2", "y2")

        with pytest.raises(ValueError):
            # Pass 2 graphs from different RDataFRames
            LiveVisualize([graph1, graph2])

    def test_unsupported_operation(self, payload):
        """
        Tests passing an unsupported operation to LiveVisualize.
        """
        connection, backend = payload
        if backend != "dask":
            # This feature is only available with the Dask backend
            return
        d = ROOT.RDataFrame(100, executor=connection)
        df = d.Define("x", "rdfentry_")
        sum = df.Sum("x")
        
        with pytest.raises(ValueError):
            # Pass an unsupported operation as the first argument
            LiveVisualize([sum], set_marker)

    def test_call_after_computation(self, payload):
        """
        Tests calling LiveVisualize after triggering the computation graph.
        """
        connection, backend = payload
        if backend != "dask":
            # This feature is only available with the Dask backend
            return
        d = ROOT.RDataFrame(100, executor=connection)
        graph = d.Define("x", "rdfentry_").Define("y", "x*x").Graph("x", "y")

        # Trigger the computation graph
        graph.Draw()

        with pytest.warns(UserWarning, match="LiveVisualize should be called before triggering the computation graph. Skipping live visualization."):
            # Call LiveVisualize after triggering the computation graph
            LiveVisualize([graph])

if __name__ == "__main__":
    pytest.main(args=[__file__])
