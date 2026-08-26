"""Generate headers omitted from the cctbx-base wheel.

These are normally produced by libtbx while building cctbx from source. DIALS
and dxtbx need them only while compiling their own native extensions.
"""

from pathlib import Path

import boost_adaptbx
import cctbx
import scitbx
import smtbx
from cctbx.source_generators import flex_fwd_h as cctbx_flex_fwd
from scitbx.source_generators import flex_fwd_h as scitbx_flex_fwd
from scitbx.source_generators.array_family import generate_all
from smtbx.source_generators import flex_fwd_h as smtbx_flex_fwd


scitbx_root = Path(scitbx.__file__).parent
generate_all.refresh(scitbx_root / "array_family")

targets = (
    (scitbx_flex_fwd, scitbx_root / "array_family" / "boost_python"),
    (cctbx_flex_fwd, Path(cctbx.__file__).parent / "boost_python"),
    (smtbx_flex_fwd, Path(smtbx.__file__).parent / "boost_python"),
)
for generator, target in targets:
    target.mkdir(parents=True, exist_ok=True)
    generator.run(target)

# boost_adaptbx normally probes this during its build. The container target is
# linux/amd64, where size_t and unsigned long are the same C++ type.
(Path(boost_adaptbx.__file__).parent / "type_id_eq.h").write_text(
    "// Generated for the linux/amd64 container target.\n"
    "#ifndef BOOST_ADAPTBX_TYPE_ID_EQ_H\n"
    "#define BOOST_ADAPTBX_TYPE_ID_EQ_H\n"
    "#define BOOST_ADAPTBX_TYPE_ID_SIZE_T_EQ_UNSIGNED_LONG\n"
    "#endif\n",
    encoding="utf-8",
)
