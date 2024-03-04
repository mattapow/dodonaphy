import torch
import json

class TensorEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            dic = {
                "type": "torch.Tensor",
                "values": obj.tolist(),
                "dtype": str(obj.dtype),
                "requires_grad": obj.requires_grad
            }
            return dic
        return json.JSONEncoder.default(self, obj)


class TensorDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, dic):
        if "type" in dic and dic["type"] == "torch.Tensor":
            del dic["type"]
            if "dtype" in dic:
                dic["dtype"] = getattr(torch, dic["dtype"].split(".")[-1])
            values = dic["values"]
            del dic["values"]
            tensor = torch.tensor(values, **dic)
            return tensor
        return dic