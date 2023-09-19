#  @author Silia Taider
#  @date 2023-08

################################################################################
# Copyright (C) 1995-2023, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################
from __future__ import annotations

from typing import Dict, Optional, Callable, Any
import ast
import inspect
import warnings
from functools import singledispatch, partial


@singledispatch   
def LiveVisualize(drawable_callback_dict: Dict[Any, Optional[Callable]], 
                  global_callback: Optional[Callable] = None) -> None:
    """
    Enables real-time data representation for the given drawable objects.
    The objects are drawn and updated every time a partial result returns from distributed execution.

    Args:
        drawable_callback_dict (dict): A dictionary where keys are drawable objects 
        and values are optional corresponding callback functions. 

        global_callback (function): An optional global callback function that 
            is applied to all drawable objects.

    Raises:
        ValueError: If a passed drawable object is not valid.
    """
    # Check if the objects already have a value (the computation graph has already been triggered)
    if any(obj.proxied_node.value for obj in drawable_callback_dict):
        warnings.warn("LiveVisualize should be called before triggering the computation graph. Skipping live visualization.")
        return

    # Check if all drawables share the same headnode
    unique_headnodes = list({obj.proxied_node.get_head() for obj in drawable_callback_dict})
    if len(unique_headnodes) > 1:
        raise ValueError("Live visualization is not supported for operations belonging to different RDataFrame graphs.")
    
    global_callback_validated = process_callback(global_callback)
    drawable_id_callback_dict = {
        # Key: node_id of the drawable object's proxied_node
        # Value: List of validated callback functions for the drawable object
        obj.proxied_node.node_id: [process_callback(callback), global_callback_validated]
        for obj, callback in drawable_callback_dict.items()
        # Filter: Only include valid drawable objects
        if is_valid_drawable(obj)
    }

    unique_headnodes[0].drawables_dict = drawable_id_callback_dict


@LiveVisualize.register(list)
@LiveVisualize.register(tuple)
def _1(drawables, callback: Optional[Callable] = None) -> None:
    """
    Wrapper function to facilitate calling LiveVisualize with a list or a tuple of drawable objects.

    Args:
        drawables (list | tuple): Drawable objects to visualize.
		
        callback (function): An optional callback function to be applied to the drawable objects.

    Notes:
        This function constructs a dictionary of drawable objects and their associated callback functions,
        and then calls the main LiveVisualize function with the constructed dictionary.
    """
    if callback is None:
        drawable_callback_dict = {obj: None for obj in drawables}
    else:
        drawable_callback_dict = {obj: callback for obj in drawables}

    LiveVisualize(drawable_callback_dict)


def process_callback(callback: Callable) -> Callable:
    """
    Process and validate a callback function.

    Args:		
        callback: The callback function to be validated.

    Returns:
        validated_callback: The validated callback function, or None if not valid.
    """
    if callback is None:
        return None
		
    elif not callable(callback):
        warnings.warn("The provided callback is not callable. Skipping callback.")
        return None

    elif not has_correct_argument_count(callback):
        warnings.warn("The callback function should have exactly one parameter to fill. Skipping callback.")
        return None
    
    elif not is_callback_safe(callback):
        warnings.warn("The provided callback function contains blocked actions. Skipping callback.")
        return None    
        
    return callback


def has_correct_argument_count(callback: Callable) -> bool:
    """
    Checks if the provided callback function has exactly one unfilled argument.

    Args:
        callback (Callable): The callback function to check.

    Returns:
        bool: True if the callback function has exactly one unfilled argument,
              False otherwise.
    """
    # Get the values of the functions parameters
    unfilled_parameters = [param for param in inspect.signature(callback).parameters.values() if param.default == param.empty]
    if len(unfilled_parameters) != 1:
            return False
    
    return True


def is_callback_safe(callback: Callable) -> bool:
    """
    Checks if the provided callback function is safe for live visualization, 
        (does not contain blocked actions).

    Args:
        callback (function): The callback function to check.

    Returns:
        bool: True if the callback function is safe, 
            False otherwise.
    """
    # Parse the callback function's source code
    if isinstance(callback, partial):
            callback = callback.func
    callback_source_ast = ast.parse(inspect.getsource(callback))
    for node in ast.walk(callback_source_ast):
        if is_action_blocked(node):
            return False
        
    return True


def is_action_blocked(node: ast.AST) -> bool:
    """
    Checks if the given Abstract Syntax Tree (AST) node corresponds to a blocked action.

    Args:
        node (ast.AST): The AST node to check.

    Returns:
        bool: True if the AST node corresponds to a blocked action,
              False otherwise.
    """
    BLOCKED_ACTIONS = [ "Add", "AddBinContent", "BufferFill", "Build",  "ClearUnderflowAndOverflow", 
                       "Delete", "Divide", "DoFillN", "Fill", "FillN",  "FillRandom", "LabelsDeflate", 
                       "Merge",  "Multiply", "Rebin", "Rebuild", "RecursiveRemove", "Reset", "Scale", 
                       "SetBinContent", "SetBinError", "SetBins", "SetBinsLength",  "SetBuffer", 
                       "SetCellContent", "SetCellError", "SetContent", "SetDirectory", "SetEntries", 
                       "SetError", "SetMaximum", "SetMinimum",  "Smooth", "TransformHisto", 
                       "UpdateBinContent"]

    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in BLOCKED_ACTIONS:
                return True
            
    return False

	
def is_valid_drawable(obj: Any) -> bool:
    """
    Checks if the object is a valid drawable object for live visualization.

    Args:
        obj: The object to be checked.

    Returns:
        bool: True if the object is a valid drawable object for live visualization 
            according to the ALLOWED_OPERATIONS list , False otherwise.
    """
    ALLOWED_OPERATIONS = ["Histo1D", "Histo2D", "Histo3D",
                          "Graph", "GraphAsymmErrors",
                          "Profile1D", "Profile2D"]
    
    if hasattr(obj, "proxied_node") and hasattr(obj.proxied_node, "operation") and hasattr(obj.proxied_node.operation, "name"):
        if obj.proxied_node.operation.name in ALLOWED_OPERATIONS:
            return True
    
    raise ValueError(f"Allowed operations are: {ALLOWED_OPERATIONS}. Skipping live visualization.")
   