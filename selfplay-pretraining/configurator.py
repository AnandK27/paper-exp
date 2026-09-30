"""
nanoGPT's "poor man's configurator". Usage from a script:
    exec(open('configurator.py').read())
then
    python selfplay.py config/selfplay_1m.py --batch_size=32 --compile=False
"""
import sys
from ast import literal_eval

for arg in sys.argv[1:]:
    if "=" not in arg:
        assert not arg.startswith("--")
        config_file = arg
        print(f"Overriding config with {config_file}:")
        with open(config_file) as f:
            print(f.read())
        exec(open(config_file).read())
    else:
        assert arg.startswith("--")
        key, val = arg.split("=", 1)
        key = key[2:]
        if key in globals():
            try:
                attempt = literal_eval(val)
            except (SyntaxError, ValueError):
                attempt = val
            if globals()[key] is not None and attempt is not None:
                assert type(attempt) == type(globals()[key]), f"type mismatch for {key}"
            print(f"Overriding: {key} = {attempt}")
            globals()[key] = attempt
        else:
            raise ValueError(f"Unknown config key: {key}")
