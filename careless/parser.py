import argparse
from os.path import exists
from pathlib import Path

class EnvironmentSettingsMixin(argparse.ArgumentParser):
    """
    Defers scientific-library initialization until the input kind is known.
    """
    def parse_args(self, *args, **kwargs):
        # NumPy and Torch are deliberately imported later by run_careless.
        # Raw DIALS input must load the cctbx native extensions first.
        return super().parse_args(*args, **kwargs)

class CustomParser(EnvironmentSettingsMixin):
    """
    A custom ArgumentParser with parse_args overloaded in order to 
     - Set tensorflow environment variables
     - Detect conflicting arguments and raise an informative error
    """
    def _validate_input_files(self, parser):
        if parser.type == 'devices':
            return
        parser.input_kind = 'files'
        directories = [Path(name) for name in parser.reflection_files if Path(name).is_dir()]
        if directories:
            if len(parser.reflection_files) != 1:
                self.error(
                    "A prepared dataset or raw DIALS directory must be the only "
                    "reflection input"
                )
            directory = directories[0]
            manifest = (directory / "manifest.json").is_file()
            complete = (directory / "COMPLETE").is_file()
            if manifest or complete:
                if not (manifest and complete):
                    self.error(f"Prepared reflection dataset {directory} is incomplete")
                if parser.type != 'mono':
                    self.error("Careless tensor caches are supported by mono only")
                parser.input_kind = 'prepared'
            elif parser.type == 'mono' and any(directory.glob(parser.dials_expt_glob)):
                parser.input_kind = 'dials'
            else:
                pattern = getattr(parser, 'dials_expt_glob', '**/*.expt')
                self.error(
                    f"Reflection directory {directory} is neither a completed Careless "
                    f"tensor cache nor a raw DIALS directory matching {pattern!r}"
                )
        for inFN in parser.reflection_files:
            if not exists(inFN):
                self.error(f"Unmerged reflection file {inFN} does not exist")
            elif Path(inFN).is_dir():
                continue
            elif inFN.endswith(".mtz") or inFN.endswith(".stream"):
                continue
            self.error(
                f"Could not determine filetype for reflection file, {inFN}." 
                 "Expected an '.mtz' or '.stream' file, or a completed "
                 "Careless prepared-dataset directory."
                )
        if getattr(parser, 'save_tensors', None) is not None and parser.input_kind != 'dials':
            self.error("--save-tensors is only valid with a raw DIALS input directory")
        if getattr(parser, 'tensor_shards', None) is not None and parser.save_tensors is None:
            self.error("--tensor-shards requires --save-tensors")
        for name in ('dials_max_files', 'ray_workers_per_node', 'ray_block_mib', 'tensor_shards'):
            value = getattr(parser, name, None)
            if value is not None and value < 1:
                self.error(f"--{name.replace('_', '-')} must be positive")

    def parse_args(self, *args, **kwargs):
        parser = super().parse_args(*args, **kwargs)
        self._validate_input_files(parser)
        return parser

import re
import textwrap
class CustomFormatter(argparse.HelpFormatter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._whitespace_matcher=re.compile("\n(?!\n)")

    def _fill_text(self, text, width, indent):
        #First replace single newlines with a space
        text = re.sub(r'(?!>\n)\n(?!\n)', '', text)
        return textwrap.fill(text, width, initial_indent=indent, subsequent_indent=indent, replace_whitespace=False, drop_whitespace=False)

description = """
Scale and merge crystallographic data by \n\n\n approximate inference.
"""
parser = CustomParser(description=description, formatter_class=CustomFormatter)

# Add --version argument
import careless
parser.add_argument("--version", action="version", version=f"careless {careless.__version__}")

subs = parser.add_subparsers(title="Experiment Type", required=True, dest="type")
mono_sub = subs.add_parser("mono", help="Process monochromatic diffraction data.", formatter_class=CustomFormatter)
poly_sub = subs.add_parser("poly", help="Process polychromatic, 'Laue', diffraction data.", formatter_class=CustomFormatter)
devices_sub = subs.add_parser("devices", help="Print available physical devices", formatter_class=CustomFormatter)

from careless.args import required,poly,groups,dials

for args,kwargs in required.args_and_kwargs:
    mono_sub.add_argument(*args, **kwargs)
    poly_sub.add_argument(*args, **kwargs)

for args,kwargs in poly.args_and_kwargs:
    poly_sub.add_argument(*args, **kwargs)

dials_group = mono_sub.add_argument_group(dials.name, dials.description)
for args,kwargs in dials.args_and_kwargs:
    dials_group.add_argument(*args, **kwargs)

for group in groups:
    if group.name is not None and group.description is not None:
        mono_group = mono_sub.add_argument_group(group.name, group.description)
        poly_group = poly_sub.add_argument_group(group.name, group.description)
    elif group.name is not None:
        mono_group = mono_sub.add_argument_group(group.name)
        poly_group = poly_sub.add_argument_group(group.name)
    else:
        mono_group = mono_sub
        poly_group = poly_sub
    for args,kwargs in group.args_and_kwargs:
        mono_group.add_argument(*args, **kwargs)
        poly_group.add_argument(*args, **kwargs)

# Test needs environment settings options
from careless.args import tf_options
for args,kwargs in tf_options.args_and_kwargs:
    devices_sub.add_argument(*args, **kwargs)
