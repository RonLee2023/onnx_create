import json
import os
import onnx
import onnx.numpy_helper
from onnx import helper 
from onnx import TensorProto 
import numpy as np
import onnx.numpy_helper

#shape ,dtype info of the node's input and output  
class LayerNodeInfo:
    def __init__(self, name, json_info:json) -> None:
        self.from_json(name, json_info)

    def from_json(self, name, json_info:json):
        self._name = name
        self._dtype = json_info["dtype"]
        self._shape = json_info["shape"]

    @property
    def name(self):
        return self._name
    @property
    def dtype(self):
        return self._dtype
    @property
    def shape(self):
        return self._shape

class LayerNode:
    def __init__(self, node_name, op_type, dtype, shape) -> None:
        self._name = node_name
        #if not input output node, _tensor_value_info is the input of the LayerNode
        self._tensor_value_info = None
        self._node = None
        self._is_ionode = False
        if op_type in ["net_inputs", "net_outputs"]:
            self._tensor_value_info = helper.make_tensor_value_info(node_name, self.str2tensorprototype(dtype), shape) 
            self._is_ionode = True
        else:
            self._node = helper.make_node(op_type, [], [],name=node_name)
            self._is_ionode = False
        #input output 's type and shape 
        self._input_infos = []
        self._output_infos = []
        #input output 's name 
        self._inputs = []
        self._outputs = []
    @property
    def name(self):
        return self._name
    @property
    def onnx_node(self):
        if self._tensor_value_info !=None:
            return self._tensor_value_info
        return self._node
    @property
    def tensor_value_info(self):
        return self._tensor_value_info

    @property
    def is_ionode(self):
        return self._is_ionode
    
    @property
    def input_infos(self):
        return self._input_infos
    @property
    def output_infos(self):
        return self._output_infos
    @property
    def inputs(self):
        return self._inputs
    @property
    def outputs(self):
        return self._outputs

    @property
    def input_info(self, idx):
        return self.input_infos[idx]
    @property
    def output_info(self, idx):
        return self.output_infos[idx]
    
    def get_input_idx_by_name(self, name:str, ):
        return self.inputs.index(name)
    
    def get_output_idx_by_name(self, name:str, ):
        return self.outputs.index(name)

    def str2tensorprototype(self, str_type):
        dict={
            "TensorProto.FLOAT":TensorProto.FLOAT,
             "TensorProto.INT8":TensorProto.INT8,
            "TensorProto.UINT8":TensorProto.UINT8,
            "TensorProto.UINT16":TensorProto.UINT16,
            "TensorProto.INT16":TensorProto.INT16,
            "TensorProto.INT32":TensorProto.INT32,
            "TensorProto.INT64":TensorProto.INT64,
            "TensorProto.FLOAT16":TensorProto.FLOAT16,
            "TensorProto.DOUBLE":TensorProto.DOUBLE,
            "TensorProto.UINT32":TensorProto.UINT32,
            "TensorProto.UINT64":TensorProto.UINT64
        }
        if str_type in dict :
            return dict[str_type]
        else:
            raise ValueError("TensorProto type error")
        
    def add_weights(self, weights_json):
        if weights_json == None:
            return
        
        weights_nps = []
        d_dict={
        "TensorProto.FLOAT32":np.float32,
        "TensorProto.FLOAT16":np.float16,
        "TensorProto.FLOAT":np.float32,
        "TensorProto.INT8":np.int8,
        "TensorProto.UINT8":np.uint8,
        }
        for w_json in weights_json:
            key = w_json["dtype"]
            r_min = w_json["rand_min"]
            r_max = w_json["rand_max"]
            if r_min != r_max:
                input = np.random.random(w_json["shape"]) * r_max + r_min
            else:
                input = np.ones(w_json["shape"]) * r_min
            weight_np = input.astype(d_dict[key])
            const_node_weight = onnx.numpy_helper.from_array(
                weight_np,
                w_json["name"])
            self._node.input.append(const_node_weight.name)
            weights_nps.append(const_node_weight)
        return weights_nps

    def set_attribute(self, attrs_json):
        for key_name in attrs_json:
            new_attr=onnx.helper.make_attribute(key_name, attrs_json[key_name])
            self._node.attribute.append(new_attr)



