import json
import transformers
from convert import convert_file
# From  https://github.com/huggingface/safetensors/blob/b947b59079a6197d7930dfb535818ac4896113e8/bindings/python/convert.py#L84
def get_discard_names(config_filename):
    try:
        import json
        import transformers
        with open(config_filename, "r") as f:
            config = json.load(f)
        architecture = config["architectures"][0]
        class_ = getattr(transformers, architecture)
        # Name for this varible depends on transformers version.
        discard_names = getattr(class_, "_tied_weights_keys", [])
    except Exception:
        discard_names = []
    return discard_names

config = "ckpts/models--LiheYoung--depth_anything_vits14/config.json"
pt_filename = "ckpts/models--LiheYoung--depth_anything_vits14/pytorch_model.bin"
sf_filename = "ckpts/models--LiheYoung--depth_anything_vits14/model.safetensors"

discard_names = get_discard_names(config)
convert_file(pt_filename , sf_filename, discard_names)