name = None
description = None

args_and_kwargs = (
    # Required args
    (("metadata_keys", ),  {
        "help":"Metadata keys for scaling. This is expected to be a comma delimitted string. "
               "You may use `rs.mtzdump` to check the available metadata column names in mtz files. "
               "Careless always provides the special metadata keys, "
               "'dHKL,Hobs,Kobs,Lobs,image_id'. "
               "For stream files, careless does not support arbitrary metadata columns but rather "
               "provides the metadata keys, "
               "'BATCH,s1x,s1y,s1z,ewald_offset,angular_ewald_offset', "
               "These correspond to image number, scattered beam wavevectors (x,y,z), and ewald offsets (inverse angstroms, degrees).",
        "type":str, 
    }),

    (("reflection_files", ), { 
        "metavar":"reflections.{mtz,stream,prepared}|dials_directory",
        "help":"MTZ or stream file(s), a completed Careless tensor-cache "
               "directory, or (for mono only) one directory containing paired "
               "DIALS .expt/.refl files. "
               "If you are supplying stream files, you must also use the --spacegroups option to supply the symmetry for merging. "
               "See the metadata_keys param for more info about stream file usage. "
               "careless poly does not currently support .stream files.",
        "type":str, 
        "nargs":'+',
    }),

    (("output_base", ), {
        "metavar":"out", 
        "help":"Output filename base.", 
        "type":str,
    }),
)
