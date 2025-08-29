import sys
import json
import inspect
import importlib
from dataclasses import asdict
from argparse import ArgumentParser
from huggingface_hub import hf_hub_download
from src.profiler import profile_model, profile_device

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-p", "--profile", dest="ret", type=str,
                        help="Select return type: Device information vs Model information")
    parser.add_argument("-m", "--model", dest="model", type=str,
                        help="mlx_lm model to be traced and profiled.")
    parser.add_argument("-r", "--repo", dest="repo_id", type=str,
                        help="Hugging Face repository ID to load the config from (e.g., 'Qwen/Qwen3-4B-MLX-8bit')")
    parser.add_argument("-o", "--output", dest="output_path", type=str,
                        help="Path to directory where we create the output file.")
    parser.add_argument("-b" , "--batch", dest="B", type=int, help="Batch size")
    parser.add_argument("-s" , "--sequence", dest="L", type=int, help="Sequence length")
    args = parser.parse_args() 
    
    if args.ret == "device":
        if args.output_path is None:
            raise ValueError("No output path given. Aborting")
            sys.exit(-1)
        with open(args.output_path, "w") as f:
            ret = profile_device()
            if ret is not None:
                f.write(ret.json())
            else:
                raise RuntimeError("Unable to profile device.")
                sys.exit(-1)
        sys.exit(1)
    elif args.ret == "model":
        if args.model is None:
            raise ValueError("'model' value is needed to profile a model.")
        try:
            module = importlib.import_module(f"mlx_lm.models.{args.model}")
        except ImportError as e:
            raise RuntimeError(f"Model {args.model} not found in mlx_lm registry.")
            sys.exit(-1)
        Model = getattr(module, "Model")
        ModelArgs = getattr(module, "ModelArgs")
        assert inspect.isclass(Model)
        assert inspect.isclass(ModelArgs)

        if Model is None or ModelArgs is None:
            raise RuntimeError(f"Could not import symbols 'Model' and 'ModelArgs' from mlx_lm.models.{args.model}")
        # Load config from Hugging Face Hub
        if args.repo_id:
            try:
                config_path = hf_hub_download(repo_id=args.repo_id, filename="config.json")
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
            except Exception as e:
                raise RuntimeError(f"Unable to download config from Hugging Face Hub: {e}")
                sys.exit(-1)
        else:
            raise ValueError("Repository ID (--repo) is required to load model config from Hugging Face Hub.")
            sys.exit(-1)
        
        try:
            # Filter config_dict to only include parameters accepted by ModelArgs
            import inspect
            modelargs_params = inspect.signature(ModelArgs.__init__).parameters
            # Skip 'self' parameter
            valid_params = [p for p in modelargs_params if p != 'self']
            filtered_config = {k: v for k, v in config_dict.items() if k in valid_params}
            
            
            config_obj = ModelArgs(**filtered_config)
            obj = Model(config_obj)
        except Exception as e:
            raise RuntimeError(f"Unable to instantiate model object from config: {e}")
            sys.exit(-1)

        if args.output_path is None:
            raise ValueError("No output path give.")
            sys.exit(-1)
        with open(args.output_path, "w") as f:
            ret = profile_model(obj, config_obj, int(args.B), int(args.L))
            f.write(json.dumps(asdict(ret)))
        sys.exit(1)
        
    else:
        raise ValueError("Unknown 'return' argument. We can only handle 'model' or 'device'.") 
        sys.exit(-1)
