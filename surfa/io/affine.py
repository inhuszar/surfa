import numpy as np

from surfa.io import fsio
from surfa.io import protocol
from surfa.io import check_file_readability
from surfa.transform import Affine


def load_affine(filename, fmt=None, **kwargs):
    """
    Load an `Affine` from file.

    Parameters
    ----------
    filename : str
        File path to read.
    fmt : str, optional
        Forced file format. If None (default), file format is extrapolated
        from extension.
    **kwargs
        Extra keyword arguments passed to the format-specific loader, such as
        the `source` and `target` geometries of a register.dat file.

    Returns
    -------
    Affine
        Loaded affine. 
    """
    check_file_readability(filename)

    if fmt is None:
        iop = protocol.find_protocol_by_extension(affine_io_protocols, filename)
        if iop is None:
            raise ValueError(f'cannot determine file format from extension for {filename}')
    else:
        iop = protocol.find_protocol_by_name(affine_io_protocols, fmt)
        if iop is None:
            raise ValueError(f'unknown file format {fmt}')

    return iop().load(filename, **kwargs)


def save_affine(aff, filename, fmt=None):
    """
    Save a `Affine` object to file.

    Parameters
    ----------
    aff : Affine
        Object to write.
    filename: str
        Destination file path.
    fmt : str
        Forced file format. If None (default), file format is extrapolated
        from extension.
    """
    if fmt is None:
        iop = protocol.find_protocol_by_extension(affine_io_protocols, filename)
        if iop is None:
            raise ValueError(f'cannot determine file format from extension for {filename}')
    else:
        iop = protocol.find_protocol_by_name(affine_io_protocols, fmt)
        if iop is None:
            raise ValueError(f'unknown file format {fmt}')
        filename = iop.enforce_extension(filename)

    iop().save(aff, filename)


class LinearTransformArrayIO(protocol.IOProtocol):
    """
    Affine IO protocol for LTA files.
    """

    name = 'lta'
    extensions = ('.lta',)

    def load(self, filename):
        """
        Read affine from an LTA file.

        Parameters
        ----------
        filename : str
            File path to read.

        Returns
        -------
        Affine
            Affine object loaded from file.
        """
        with open(filename, 'r') as file:
            lines = [line.rstrip() for line in file]
            lines = [line for line in lines if line and not line.startswith('#')]

        # determine the coordinate space
        space_id = int(lines[0].split()[2])
        space = {0: 'vox',
                 1: 'world',
                 3: 'surf'}.get(space_id, None)
        if space is None:
            raise ValueError(f'unknown affine LTA type ID: {space_id}')

        # read in the actual matrix data
        matrix = np.asarray([line.split() for line in lines[5:9]], dtype=np.float64)

        # read in source and target geometry (if valid)
        source = fsio.image_geometry_from_string('\n'.join(lines[10:18]))
        target = fsio.image_geometry_from_string('\n'.join(lines[19:27]))
        if source is None and target is None:
            space = None

        return Affine(matrix, source=source, target=target, space=space)

    def save(self, aff, filename):
        """
        Write affine to an LTA file.

        Parameters
        ----------
        aff : Affine
            Array to save.
        filename : str
            Target file path.
        """
        with open(filename, 'w') as file:

            # determine LTA coordinate space
            if aff.space is None or aff.space == 'vox':
                file.write('type      = 0 # LINEAR_VOX_TO_VOX\n')
            elif aff.space == 'world':
                file.write('type      = 1 # LINEAR_RAS_TO_RAS\n')
            elif aff.space == 'surf':
                file.write('type      = 3 # LINEAR_SURF_TO_SURF\n')
            else:
                raise NotImplementedError(f'cannot write coordinate space {aff.space} to LTA - this is a '
                                           'bug, not a user error')

            # this is all useless legacy information
            file.write('nxforms   = 1\n')
            file.write('mean      = 0.0000 0.0000 0.0000\n')
            file.write('sigma     = 1.0000\n')
            file.write('1 4 4\n')

            # write the actual matrix data
            file.write('%.15e %.15e %.15e %.15e\n' % tuple(aff.matrix[0]))
            file.write('%.15e %.15e %.15e %.15e\n' % tuple(aff.matrix[1]))
            file.write('%.15e %.15e %.15e %.15e\n' % tuple(aff.matrix[2]))
            file.write('%.15e %.15e %.15e %.15e\n' % tuple(aff.matrix[3]))

            # write source geometry (if any)
            file.write('src volume info\n')
            file.write(fsio.image_geometry_to_string(aff.source))

            # write target geometry (if any)
            file.write('dst volume info\n')
            file.write(fsio.image_geometry_to_string(aff.target))
            
            
class MNITransformIO(protocol.IOProtocol):
    """
    Affine IO protocol for MNI .xfm transform files
    (\"MNI Transform File\" as used by FreeSurfer's ltaMNIwrite/ltaMNIread).
    """

    name = "xfm"
    extensions = (".xfm",)

    def load(self, filename):
        """
        Read affine from an MNI .xfm file.

        This is a RAS-to-RAS (world-to-world) 3D transform.
        Source/target geometries are not stored in the file,
        so the returned Affine has source/target = None and space = 'world'.
        """
        with open(filename, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        if not lines[0].lower().startswith("mni transform file"):
            raise ValueError(f"{filename} does not look like an MNI .xfm file")

        # Find the "Linear_Transform" block
        lt_idx = None
        for i, line in enumerate(lines):
            if line.lower().startswith("linear_transform"):
                lt_idx = i + 1
                break

        if lt_idx is None:
            raise ValueError(
                f"could not find 'Linear_Transform' block in {filename}")

        rows = []
        for r in range(3):
            line = lines[lt_idx + r]
            # strip trailing ';' if present
            line = line.replace(";", "")
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(
                    f"expected 4 numbers on row {r+1} "
                    f"of Linear_Transform in {filename}, "
                    f"got: {line!r}"
                )
            rows.append([float(x) for x in parts[:4]])

        mat = np.eye(4, dtype=np.float64)
        mat[:3, :4] = np.asarray(rows, dtype=np.float64)

        # By definition this is a world/RAS-to-world/RAS transform
        return Affine(mat, space="world")

    def save(self, aff, filename):
        """
        Write affine to an MNI .xfm file.

        If aff.space is 'world' or 'surf', the matrix is written as-is
        (LINEAR_RAS_TO_RAS case).

        If aff.space is None or 'vox', the matrix is interpreted as
        voxel-to-voxel (LINEAR_VOX_TO_VOX) and converted to a RAS-to-RAS
        transform using the source/target geometries, exactly as in
        ltaMNIwrite() in transform.cpp.
        """
        if aff.ndim != 3:
            raise NotImplementedError("MNI .xfm only supports 3D transforms")

        # Interpret 'space' like the LTA writer does
        space = aff.space or "vox"

        # Decide which matrix to write: either directly m_L (RAS2RAS),
        # or converted from VOX2VOX into RAS2RAS.
        if space in ("world", "surf"):
            ras_to_ras = np.asarray(aff.matrix, dtype=np.float64)
        elif space == "vox":
            if aff.source is None or aff.target is None:
                raise ValueError(
                    "cannot write voxel-space affine to MNI .xfm without "
                    "both source and target geometries"
                )

            # C code:
            #   voxFromRAS  = extract_r_to_i(&lt->src);        // world->vox
            #   tmp         = m_L * voxFromRAS;
            #   rasFromVoxel= extract_i_to_r(&lt->dst);       // vox->world
            #   rasToRAS    = rasFromVoxel * tmp;
            #
            # Here:
            #   src.world2vox.matrix ~ extract_r_to_i(src)
            #   dst.vox2world.matrix ~ extract_i_to_r(dst)
            m_L = np.asarray(aff.matrix, dtype=np.float64)
            vox_from_ras = np.asarray(
                aff.source.world2vox.matrix, dtype=np.float64)
            ras_from_vox = np.asarray(
                aff.target.vox2world.matrix, dtype=np.float64)

            tmp = m_L @ vox_from_ras
            ras_to_ras = ras_from_vox @ tmp
        else:
            raise NotImplementedError(
                f"cannot write affine with space={aff.space!r} to MNI .xfm"
            )

        # The C code only writes the top 3 rows.
        # Use filenames if they exist on geometries; otherwise leave blank.
        src_name = getattr(getattr(aff, "source", None), "filename", "") or ""
        dst_name = getattr(getattr(aff, "target", None), "filename", "") or ""

        with open(filename, "w") as fp:
            fp.write("MNI Transform File\n")
            fp.write(
                f"%Generated by surfa.Affine.save "
                f"src {src_name} dst {dst_name}\n"
            )
            fp.write("\n")
            fp.write("Transform_Type = Linear;\n")
            fp.write("Linear_Transform =\n")

            for row in range(3):
                vals = ras_to_ras[row, :4]
                line = " ".join(f"{v:13.8f}" for v in vals)
                if row == 2:
                    line += " ;"
                fp.write(line + "\n")


class RegisterDatIO(protocol.IOProtocol):
    """
    Affine IO protocol for FreeSurfer register.dat files, as written by
    tkregister2, bbregister and lta_convert --outreg.

    The file holds a subject name, the in-plane and between-plane resolutions
    (mm), an intensity scale, a 4x4 matrix R and an optional float2int token
    ('tkregister', 'floor' or 'round'). R maps the tkregister (surface) RAS of
    the target (reference, anatomical) volume to the tkregister RAS of the
    movable volume, that is MovRAS = R @ TargRAS (see `tkregister2 --help`).
    This is the reverse of surfa's source-to-target convention, so the loaded
    `Affine` holds the inverse of R, with source = movable and target =
    reference, matching the LTA that `lta_convert --inreg` writes.

    The extension '.dat' is generic. It is claimed only among the affine IO
    protocols (`load_affine`, `Affine.save`), and the loader raises an error
    if the file does not parse as a register.dat file.
    """

    name = 'regdat'
    extensions = ('.dat', '.reg')
    float2int_methods = ('tkregister', 'floor', 'round')

    def load(self, filename, source=None, target=None):
        """
        Read affine from a register.dat file.

        The file stores no image geometry, so the transform can only be given
        in tkregister space. Pass the movable and reference geometries to
        convert it to scanner RAS with `aff.convert(space='world')`.

        Parameters
        ----------
        filename : str
            File path to read.
        source : castable ImageGeometry, optional
            Movable (source) image geometry.
        target : castable ImageGeometry, optional
            Reference (target) image geometry.

        Returns
        -------
        Affine
            Affine object loaded from file, in 'surf' space. The header fields
            are kept in `metadata` under the keys 'subject', 'inplaneres',
            'betplaneres', 'intensity' and 'float2int' (None if absent).
        """
        # tokenise like regio_read_register(), which reads with fscanf
        with open(filename, 'r') as file:
            tokens = file.read().split()

        if len(tokens) < 20:
            raise ValueError(f'{filename} does not look like a register.dat file')
        try:
            inplaneres, betplaneres, intensity = (float(t) for t in tokens[1:4])
            matrix = np.asarray(tokens[4:20], dtype=np.float64).reshape(4, 4)
        except ValueError:
            raise ValueError(f'{filename} does not look like a register.dat file')

        float2int = tokens[20] if len(tokens) > 20 else None
        if float2int is not None and float2int not in self.float2int_methods:
            raise ValueError(f'unknown float2int method {float2int!r} in {filename}')

        metadata = {
            'subject': tokens[0],
            'inplaneres': inplaneres,
            'betplaneres': betplaneres,
            'intensity': intensity,
            'float2int': float2int,
        }

        # the file maps target to source, surfa maps source to target
        return Affine(np.linalg.inv(matrix), source=source, target=target,
                      space='surf', metadata=metadata)

    def save(self, aff, filename):
        """
        Write affine to a register.dat file.

        An affine not in 'surf' space is converted to it, which requires both
        source and target geometries. Header fields are taken from
        `aff.metadata` when present, otherwise subject 'subject-unknown', the
        source voxel size, intensity 0.1 and float2int 'round', as
        lta_convert writes them.

        Parameters
        ----------
        aff : Affine
            Array to save.
        filename : str
            Target file path.
        """
        if aff.ndim != 3:
            raise NotImplementedError('register.dat only supports 3D transforms')

        # interpret 'space' like the LTA writer does
        space = aff.space or 'vox'
        if space != 'surf':
            if aff.source is None or aff.target is None:
                raise ValueError(f'cannot write affine in {space} space to register.dat '
                                 'without both source and target geometries')
            aff = Affine(aff.matrix, source=aff.source, target=aff.target, space=space,
                         metadata=aff.metadata).convert(space='surf')

        # the file maps target to source, surfa maps source to target
        matrix = np.linalg.inv(aff.matrix)

        voxsize = aff.source.voxsize if aff.source is not None else (1.0, 1.0, 1.0)
        meta = aff.metadata
        subject = meta.get('subject', 'subject-unknown')
        inplaneres = meta.get('inplaneres', voxsize[0])
        betplaneres = meta.get('betplaneres', voxsize[2])
        intensity = meta.get('intensity', 0.1)
        float2int = meta.get('float2int', 'round')
        if float2int is not None and float2int not in self.float2int_methods:
            raise ValueError(f'unknown float2int method {float2int!r}')

        # same layout as regio_write_register()
        with open(filename, 'w') as file:
            file.write(f'{subject}\n')
            file.write('%f\n' % inplaneres)
            file.write('%f\n' % betplaneres)
            file.write('%f\n' % intensity)
            file.write('%.15e %.15e %.15e %.15e\n' % tuple(matrix[0]))
            file.write('%.15e %.15e %.15e %.15e\n' % tuple(matrix[1]))
            file.write('%.15e %.15e %.15e %.15e\n' % tuple(matrix[2]))
            file.write('0 0 0 1\n')
            if float2int is not None:
                file.write(f'{float2int}\n')


# enabled affine IO protocol classes
affine_io_protocols = [
    LinearTransformArrayIO,
    MNITransformIO,
    RegisterDatIO,
]
