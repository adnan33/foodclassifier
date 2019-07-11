# Usage: fix_keras_model.py old_model.h5 new_model.h5
import h5py
import shutil
import json
import sys
from keras.layers import Concatenate, Dense, LSTM, Input, concatenate
input_model_path = r"F:\model4b.10-0.68.hdf5"
output_model_path = r"F:\model4b.10-0.68_new.hdf5"
shutil.copyfile(input_model_path, output_model_path)

with h5py.File(output_model_path, "r+") as out_h5:
  v = out_h5.attrs.get("model_config").decode("utf-8")
  config = json.loads(v)
  for i, l in enumerate(config["config"]["layers"]):
    print(l)
    if(l["class_name"]=="Merge"):
        print(l["inbound_nodes"][0])
        l=concatenate(l["inbound_nodes"][0])
    dtype = l["config"].pop("input_dtype", None)
    if dtype is not None:
      l["config"]["dtype"] = dtype

  new_config_str = json.dumps(config).encode("utf-8")
  out_h5.attrs.modify("model_config", new_config_str)

# # Check that it worked.
# from keras.models import load_model
# load_model(output_model_path)