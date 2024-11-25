import cppyy

def ann_adapt(node: 'FPTA::Node&') -> cppyy.gbl.FPTA.EventId:
    return cppyy.gbl.FPTA.EventId(node.fData)

def ann_ref_mod(node: 'FPTA::Node&') -> cppyy.gbl.FPTA.EventId:
    ev_id = cppyy.gbl.FPTA.EventId(node.fData)
    node.fData = 81
    return ev_id
