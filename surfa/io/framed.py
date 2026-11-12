import warnings
import gzip
import numpy as np

# NumPy compatibility: prefer the numpy boolean scalar dtype object.
# `np.bool_` exists in both NumPy 1.x and 2.x; keep a fallback for safety.
if hasattr(np, 'bool_'):
    _np_bool_dtype = np.bool_  # NumPy 1.x
else:
    _np_bool_dtype = np.bool   # NumPy 2.x

from surfa import Volume
from surfa import Slice
from surfa import Overlay
from surfa import Warp
from surfa import TimeSeries
from surfa.core.array import pad_vector_length
from surfa.core.framed import FramedArray
from surfa.core.framed import FramedArrayIntents
from surfa.image.framed import FramedImage
from surfa.io import fsio
from surfa.io import protocol
from surfa.io.utils import read_int
from surfa.io.utils import write_int
from surfa.io.utils import read_bytes
from surfa.io.utils import write_bytes
from surfa.io.utils import read_geom
from surfa.io.utils import write_geom
from surfa.io.utils import check_file_readability
from surfa.transform.geometry import ImageGeometry
from surfa.transform.geometry import image_geometry_equal


def load_volume(filename, fmt=None):
    """
    Load an image `Volume` from a 3D array file.

    Parameters
    ----------
    filename : str or Path
        File path to read.
    fmt : str, optional
        Explicit file format. If None, we extrapolate from the file extension.

    Returns
    -------
    Volume
        Loaded volume.
    """
    return load_framed_array(filename=filename, atype=Volume, fmt=fmt)


def load_slice(filename, fmt=None):
    """
    Load an image `Slice` from a 2D array file.

    Parameters
    ----------
    filename : str or Path
        File path to read.
    fmt : str, optional
        Explicit file format. If None, we extrapolate from the file extension.

    Returns
    -------
    Slice
        Loaded slice.
    """
    return load_framed_array(filename=filename, atype=Slice, fmt=fmt)


def load_overlay(filename, fmt=None):
    """
    Load a surface `Overlay` from a 1D array file.

    Parameters
    ----------
    filename : str or Path
        File path to read.
    fmt : str, optional
        Explicit file format. If None, we extrapolate from the file extension.

    Returns
    -------
    Overlay
        Loaded overlay.
    """
    return load_framed_array(filename=filename, atype=Overlay, fmt=fmt)


def load_warp(filename, fmt=None):
    """
    Load an image `Warp` from a 3D or 4D array file, or from a FreeSurfer m3z morph,
    which loads in abs_crs format (see `FreeSurferM3ZIO`).

    Parameters
    ----------
    filename : str
        File path to read.
    fmt : str, optional
        Explicit file format. If None, we extrapolate from the file extension.

    Returns
    -------
    Warp
        Loaded warp.
    """
    return load_framed_array(filename=filename, atype=Warp, fmt=fmt)


def load_framed_array(filename, atype, fmt=None):
    """
    Generic loader for `FramedArray` objects.

    Parameters
    ----------
    filename : str or Path
        File path to read.
    atype : class
        Particular FramedArray subclass to read into.
    fmt : str, optional
        Explicit file format. If None, we extrapolate from the file extension.

    Returns
    -------
    FramedArray
        Loaded framed array.
    """
    check_file_readability(filename)

    if fmt is None:
        iop = protocol.find_protocol_by_extension(array_io_protocols, filename)
        if iop is None:
            if atype is Overlay:
                # some freesurfer overlays do not have file extensions (another bizarre convention),
                # so let's fallback to the 'curv' format here
                iop = FreeSurferCurveIO
            else:
                raise ValueError(f'cannot determine file format from extension for {filename}')
    else:
        iop = protocol.find_protocol_by_name(array_io_protocols, fmt)
        if iop is None:
            raise ValueError(f'unknown file format {fmt}')

    return iop().load(filename, atype)


# optional parameter to specify FramedArray intent, default is MRI data
def save_framed_array(arr, filename, fmt=None, intent=FramedArrayIntents.mri):
    """
    Save a `FramedArray` object to file.

    Parameters
    ----------
    arr : FramedArray
        Object to write.
    filename: str or Path
        Destination file path.
    fmt : str
        Forced file format. If None (default), file format is extrapolated
        from extension.
    """
    if fmt is None:
        iop = protocol.find_protocol_by_extension(array_io_protocols, filename)
        if iop is None:
            raise ValueError(f'cannot determine file format from extension for {filename}')
    else:
        iop = protocol.find_protocol_by_name(array_io_protocols, fmt)
        if iop is None:
            raise ValueError(f'unknown file format {fmt}')
        if iop.name != 'curv':
            filename = iop.enforce_extension(filename)

    # pass intent if iop() is an instance of MGHArrayIO or NiftiArrayIO
    if (isinstance(iop(), MGHArrayIO) or isinstance(iop(), NiftiArrayIO)):
       iop().save(arr, filename, intent=intent)
    else:
       iop().save(arr, filename)


def framed_array_from_4d(atype, data):
    """
    Squeeze and cast a 4D data array (as read by NIFTI and MGH files) to a FramedArray.

    Parameters
    ----------
    atype : class
        Particular FramedArray subclass to cast to.
    data : ndarray
        Array data.

    Returns
    -------
    FramedArray
        Squeezed framed array.
    """
    # this code is a bit ugly - it does the job but should probably be cleaned up
    if atype == Volume or atype == TimeSeries:
        return atype(data)
    if atype == Warp:
        if data.ndim == 4 and data.shape[-1] == 2:
            data = data.squeeze(-2)
        elif (data.ndim == 5):
            # this must be NIFTI_INTENT_DISPVECT, which has shape [5, c, r, s, 1, 3]
            data = data.squeeze(-2)
        return atype(data)
    # slice
    if data.ndim == 3:
        data = np.expand_dims(data, -1)
    data = data.squeeze(-2)
    if atype == Slice:
        return atype(data)
    # overlay
    data = data.squeeze(-2)
    return atype(data)


class MGHArrayIO(protocol.IOProtocol):
    """
    Array IO protocol for MGH and compressed MGZ files.
    """

    name = 'mgh'
    extensions = ('.mgz', 'mgh', '.mgh.gz')

    def dtype_from_id(self, id):
        """
        Convert a FreeSurfer datatype ID to a numpy datatype.

        Parameters
        ----------
        id : int
            FreeSurfer datatype ID.

        Returns
        -------
        np.dtype
            Converted numpy datatype.
        """
        mgh_types = {
            0:  '>u1',  # uchar
            1:  '>i4',  # int32
            2:  '>i8',  # int64
            3:  '>f4',  # float
            4:  '>i2',  # short
            6:  '>f4',  # tensor
            7:  '>c8',  # complex64
            10: '>u2',  # ushort
        }
        dtype = mgh_types.get(id)
        if dtype is None:
            raise NotImplementedError(f'unsupported MGH data type ID: {id}')
        return np.dtype(dtype)

    def load(self, filename, atype):
        """
        Read array from an MGH/MGZ file.

        Parameters
        ----------
        filename : str or Path
            File path to read.
        atype : class
            FramedArray subclass to load.

        Returns
        -------
        FramedArray
            Array object loaded from file.
        """

        # check if the file is gzipped
        fopen = gzip.open if str(filename).lower().endswith('gz') else open
        with fopen(filename, 'rb') as file:

            # read version number, retrieve intent
            intent = read_bytes(file, '>i4', 1) >> 8 & 0xffff

            # read shape and type info
            shape = read_bytes(file, '>u4', 4)
            dtype_id = read_bytes(file, '>u4')
            dof = read_bytes(file, '>u4')

            # read geometry
            geom_params = {}
            unused_header_space = 254
            valid_geometry = bool(read_bytes(file, '>u2'))

            # ignore geometry if flagged as invalid
            if valid_geometry:
                geom_params = dict(
                    voxsize=read_bytes(file, '>f4', 3),
                    rotation=read_bytes(file, '>f4', 9).reshape((3, 3), order='F'),
                    center=read_bytes(file, '>f4', 3),
                )
                unused_header_space -= 60

            # skip empty header space
            file.read(unused_header_space)

            # read data buffer (MGH files store data in fortran order)
            dtype = self.dtype_from_id(dtype_id)
            data = read_bytes(file, dtype, int(np.prod(shape))).reshape(shape, order='F')

            # init array
            arr = framed_array_from_4d(atype, data)

            # read scan parameters
            # these are not required, so first let's make sure we're not at EOF
            scan_params = {}
            fbytes = file.read(np.dtype('>f4').itemsize)
            if fbytes:
                scan_params['tr'] = np.frombuffer(fbytes, dtype='>f4')
                scan_params['fa'] = read_bytes(file, dtype='>f4')
                scan_params['te'] = read_bytes(file, dtype='>f4')
                scan_params['ti'] = read_bytes(file, dtype='>f4')

                # next parameter is the image FOV, which is not directly
                # used so let's ignore it (unsure why it's in there at all).
                # it's also not required in the freesurfer definition, so we'll
                # use the read() function directly in case end-of-file is reached
                file.read(np.dtype('>f4').itemsize)

            # update image-specific information
            if isinstance(arr, FramedImage):
                arr.geom.update(**geom_params)
                arr.metadata.update(scan_params)
                arr.metadata['intent'] = intent

            # read metadata tags
            while True:
                tag, length = fsio.read_tag(file)
                if tag is None:
                    break

                # command history
                elif tag == fsio.tags.history:
                    history = file.read(length).decode('utf-8').rstrip('\x00')
                    if arr.metadata.get('history'):
                        arr.metadata['history'].append(history)
                    else:
                        arr.metadata['history'] = [history]

                # embedded lookup table
                elif tag == fsio.tags.old_colortable:
                    arr.labels = fsio.read_binary_lookup_table(file)

                # phase encode direction
                elif tag == fsio.tags.pedir:
                    pedir = file.read(length).decode('utf-8').rstrip('\x00')
                    if pedir != 'UNKNOWN':
                        arr.metadata['phase-encode-direction'] = pedir

                # field strength
                elif tag == fsio.tags.fieldstrength:
                    arr.metadata['field-strength'] = read_bytes(file, dtype='>f4')

                # gcamorph src & trg geoms (mgz warp)
                elif tag == fsio.tags.gcamorph_geom:
                    arr.source, valid, fname = read_geom(file)
                    arr.metadata['source-valid'] = valid
                    arr.metadata['source-fname'] = fname

                    arr.target, valid, fname = read_geom(file)
                    arr.metadata['target-valid'] = valid
                    arr.metadata['target-fname'] = fname

                # gcamorph src & trg geoms (mgz warp)
                elif tag == fsio.tags.gcamorph_geom_plusshear:
                    arr.source, valid, fname = read_geom(file, shearless=False)
                    arr.metadata['source-valid'] = valid
                    arr.metadata['source-fname'] = fname

                    arr.target, valid, fname = read_geom(file, shearless=False)
                    arr.metadata['target-valid'] = valid
                    arr.metadata['target-fname'] = fname

                # gcamorph meta (mgz warp: int int float)
                elif tag == fsio.tags.gcamorph_meta:
                    arr.format = read_bytes(file, dtype='>i4')
                    arr.metadata['spacing'] = read_bytes(file, dtype='>i4')
                    arr.metadata['exp_k'] = read_bytes(file, dtype='>f4')

                # skip everything else
                else:
                    file.read(length)

        return arr

    # optional parameter to specify FramedArray intent, default is MRI data
    def save(self, arr, filename, intent=FramedArrayIntents.mri):
        """
        Write array to a MGH/MGZ file.

        Parameters
        ----------
        arr : FramedArray
            Array to save.
        filename : str or Path
            Target file path.
        """

        # determine whether to write compressed data
        if str(filename).lower().endswith('gz'):
            fopen = lambda f: gzip.open(f, 'wb', compresslevel=6)
        else:
            fopen = lambda f: open(f, 'wb')

        with fopen(filename) as file:

            # before we map dtypes to MGZ-supported types, smartly convert int64 to int32
            if arr.dtype == np.int64:
                if arr.max() > np.iinfo(np.int32).max or arr.min() < np.iinfo(np.int32).min:
                    raise ValueError('MGH files only support int32 datatypes, but array cannot be ',
                                     'casted since its values exceed the int32 integer limits')
                arr = arr.astype(np.int32)

            # determine supported dtype to save as (order here is very important)
            type_map = {
                _np_bool_dtype: 0,
                np.uint8: 0,
                np.int32: 1,
                np.floating: 3,
                np.int16: 4,
                np.uint16: 10,
                np.complex64: 7,
            }
            dtype_id = next((i for dt, i in type_map.items() if np.issubdtype(arr.dtype, dt)), None)
            if dtype_id is None:
                raise ValueError(f'writing dtype {arr.dtype.name} to MGH format is not supported')

            # shape must always be a length-4 vector, so let's pad with ones
            shape = np.ones(4, dtype=np.int64)
            shape[:arr.basedim] = arr.baseshape
            shape[-1] = arr.nframes

            # begin writing header
            intent = arr.metadata.get('intent', intent)
            version = ((intent & 0xffff) << 8) | 1  # encode intent in version
            write_bytes(file, version, '>u4')  # version
            write_bytes(file, shape, '>u4')  # shape
            write_bytes(file, dtype_id, '>u4')  # MGH data type
            write_bytes(file, 1, '>u4')  # DOF

            # include geometry only if necessary
            unused_header_space = 254
            is_image = isinstance(arr, FramedImage)
            write_bytes(file, is_image, '>u2')
            if is_image:
                # the mgz file type cannot store shear parameters
                voxsize, rotation, center = arr.geom.shearless_components()
                write_bytes(file, voxsize, '>f4')
                write_bytes(file, np.ravel(rotation, order='F'), '>f4')
                write_bytes(file, center, '>f4')
                unused_header_space -= 60

            # fill empty header space
            file.write(bytearray(unused_header_space))

            # write array data
            write_bytes(file, np.ravel(arr.data, order='F'), self.dtype_from_id(dtype_id))

            # write scan parameters
            write_bytes(file, arr.metadata.get('tr', 0.0), '>f4')
            write_bytes(file, arr.metadata.get('fa', 0.0), '>f4')
            write_bytes(file, arr.metadata.get('te', 0.0), '>f4')
            write_bytes(file, arr.metadata.get('ti', 0.0), '>f4')

            # compute FOV (freesurfer doesn't actually read this information though)
            volsize = pad_vector_length(arr.baseshape, 3, 1)
            fov = max(arr.geom.voxsize * volsize) if is_image else arr.shape[0]
            write_bytes(file, fov, '>f4')

            # write lookup table tag
            if arr.labels is not None:
                fsio.write_tag(file, fsio.tags.old_colortable)
                fsio.write_binary_lookup_table(file, arr.labels)

            # phase encode direction
            pedir = arr.metadata.get('phase-encode-direction', 'UNKNOWN')
            fsio.write_tag(file, fsio.tags.pedir, len(pedir))
            file.write(pedir.encode('utf-8'))

            # field strength
            fsio.write_tag(file, fsio.tags.fieldstrength, 4)
            write_bytes(file, arr.metadata.get('field-strength', 0.0), '>f4')

            # gcamorph geom and gcamorph meta for mgz warp
            # output both fsio.tags.gcamorph_geom and fsio.tags.gcamorph_geom_plusshear
            if intent == FramedArrayIntents.warpmap:
                # gcamorph src & trg geoms (mgz warp)
                # fsio.tags.gcamorph_geom
                fsio.write_tag(file, fsio.tags.gcamorph_geom)
                write_geom(file,
                           geom=arr.source,
                           valid=arr.metadata.get('source-valid', True),
                           fname=arr.metadata.get('source-fname', ''))
                write_geom(file,
                           geom=arr.target,
                           valid=arr.metadata.get('target-valid', True),
                           fname=arr.metadata.get('target-fname', ''))

                # fsio.tags.gcamorph_geom_plusshear
                # gcamorph_geom_plusshear has a length, datalength needs to be consistent with write_geom()
                datalength = 1200
                fsio.write_tag(file, fsio.tags.gcamorph_geom_plusshear, datalength)
                write_geom(file,
                           geom=arr.source,
                           valid=arr.metadata.get('source-valid', True),
                           fname=arr.metadata.get('source-fname', ''),
                           shearless=False)
                write_geom(file,
                           geom=arr.target,
                           valid=arr.metadata.get('target-valid', True),
                           fname=arr.metadata.get('target-fname', ''),
                           shearless=False)

                # gcamorph meta (mgz warp: int int float)
                fsio.write_tag(file, fsio.tags.gcamorph_meta, 12)
                write_bytes(file, arr.format, dtype='>i4')
                write_bytes(file, arr.metadata.get('spacing', 1), dtype='>i4')
                write_bytes(file, arr.metadata.get('exp_k', 0.0), dtype='>f4')

            # write history tags
            for hist in arr.metadata.get('history', []):
                fsio.write_tag(file, fsio.tags.history, len(hist))
                file.write(hist.encode('utf-8'))


class NiftiArrayIO(protocol.IOProtocol):
    """
    Array IO protocol for nifti files.
    """
    name = 'nifti'
    extensions = ('.nii.gz', '.nii')

    def __init__(self):
        try:
            import nibabel as nib
        except ImportError:
            raise ImportError('the `nibabel` python package must be installed for nifti IO')
        self.nib = nib

        # define space and time units
        self.units_to_code = {
            'unknown': 0,
            'm':       1,
            'mm':      2,
            'um':      3,
            'sec':     8,
            'msec':    16,
            'usec':    24,
            'hz':      32,
            'ppm':     40,
            'rads':    48,
        }
        self.code_to_units = {v: k for k, v in self.units_to_code.items()}

    def load(self, filename, atype):
        """
        Read array from a nifiti file.

        Parameters
        ----------
        filename : str or Path
            File path read.
        atype : class
            FramedArray subclass to load.

        Returns
        -------
        FramedArray
            Array object loaded from file.
        """
        nii = self.nib.load(filename)
        data = np.asanyarray(nii.dataobj)
        arr = framed_array_from_4d(atype, data)
        if isinstance(arr, FramedImage):
            if (atype == Warp):
                """
                assert (nii.header['intent_code'] == self.nib.nifti1.intent_codes['NIFTI_INTENT_DISPVECT']), \
                    f"To load {filename} as Warp, it must have intent code 'NIFTI_INTENT_DISPVECT' ({self.nib.nifti1.intent_codes['NIFTI_INTENT_DISPVECT']})"
                """
                if (nii.header['intent_code'] == self.nib.nifti1.intent_codes['NIFTI_INTENT_DISPVECT']):
                    # the displacement field vector is in ras space
                    # if the .nii.gz has FS nifti1 header extension, the format will also be updated in FSNifti1Extension.update_framedimage()
                    arr.format = Warp.Format.disp_ras

            voxsize = nii.header['pixdim'][1:4]
            arr.geom.update(vox2world=nii.affine, voxsize=voxsize)
            arr.metadata['qform_code'] = int(nii.header['qform_code'])
            arr.metadata['sform_code'] = int(nii.header['sform_code'])
            # temporal unit
            time_units_code = nii.header['xyzt_units'] & 56
            arr.metadata['frame_units'] = self.code_to_units.get(time_units_code, 0)
            arr.metadata['frame_dim'] = nii.header['pixdim'][4]
            # spatial unit: always convert to mm and assume unknown is also mm
            spatial_units_code = nii.header['xyzt_units'] & 7
            if spatial_units_code == self.units_to_code['m']:
                arr.geom.voxsize = arr.geom.voxsize * 1000
            elif spatial_units_code == self.units_to_code['um']:
                arr.geom.voxsize = arr.geom.voxsize * 0.001

            # freesurfer saves tr as msec internally
            time_units_factor = 0.0
            if (arr.metadata['frame_units'] == 'sec'):
                time_units_factor = 1000.0
            elif (arr.metadata['frame_units'] == 'msec'):
                time_units_factor = 1.0
            elif (arr.metadata['frame_units'] == 'usec'):
                time_units_factor = 0.001

            arr.metadata['tr'] = arr.metadata['frame_dim'] * time_units_factor

            # handle nifti1 header extension
            niiextsions = nii.header.extensions
            if (not niiextsions):
                return arr
            
            # try to find freesurfer nifti1 header extension
            index = -1
            ecodes = niiextsions.get_codes()            
            for ind, ecode in enumerate(ecodes):
                if (ecode == 14):
                    index = ind
                    break

            if (index < 0):
                return arr

            # found freesurfer nifti1 header extension
            import io
            from surfa.io.fsnifti1extension import FSNifti1Extension

            # retrieve freesurfer header extension data in bytes
            fsext = nii.header.extensions[index]
            esize = fsext.get_sizeondisk()
            fsext_bytedata = fsext.get_content()

            # create file-like object from bytes data
            bytes_reader = io.BytesIO(fsext_bytedata)

            # read freesurfer header extension data into FSNifti1Extension.Content object
            fsextreader = FSNifti1Extension()
            fsextcontent = fsextreader.read(bytes_reader, esize)

            # update arr metadata
            fsextcontent.update_framedimage(arr)
                
        return arr

    def save(self, arr, filename, intent=FramedArrayIntents.mri):
        """
        Write array to a nifti file.

        Parameters
        ----------
        arr : FramedArray
            Array to save.
        filename : str or Path
            Target file path.
        """
        is_image = isinstance(arr, FramedImage)

        intent = arr.metadata.get('intent', intent)
        if (intent == FramedArrayIntents.warpmap):
            assert (isinstance(arr, Warp)), "arr needs to be a Warp object"
            arr = arr.convert(format=Warp.Format.disp_ras)
            shape = np.ones(5, dtype=np.int64)
            shape[:arr.basedim] = arr.baseshape
            shape[-2] = 1
            shape[-1] = arr.nframes
        else:
            # shape must be padded, so let's pad with 4 ones then chop down to 3 dimensions if needed
            shape = np.ones(4, dtype=np.int64)
            shape[:arr.basedim] = arr.baseshape
            shape[-1] = arr.nframes
            if arr.nframes == 1:
                shape = shape[:-1]

        # convert to a valid output type (for now this is only bool but there are probably more)
        type_map = {
            _np_bool_dtype: np.uint8,
        }
        dtype_id = next((i for dt, i in type_map.items() if np.issubdtype(arr.dtype, dt)), None)
        data = arr.data if dtype_id is None else arr.data.astype(dtype_id)

        # make image object and complete header data
        nii = self.nib.Nifti1Image(data.reshape(shape), np.eye(4))
        if (intent == FramedArrayIntents.warpmap):
           nii.header.set_intent(self.nib.nifti1.intent_codes['NIFTI_INTENT_DISPVECT']) 

        # initialize spatial and temporal spacing
        nii.header['pixdim'][:] = 1
        nii.header['pixdim'][4] = arr.metadata.get('frame_dim', 1)

        tr = arr.metadata.get('tr')
        if (tr is not None):
            nii.header['pixdim'][4] = tr / 1000.0

        # for now we pretty much have to enforce spatial units of mm
        # and if frame units isn't specified, fallback to seconds
        spatial_units_code = self.units_to_code['mm']
        frame_units_code = self.units_to_code['sec']
        # check if frame units is set in metadata
        frame_units = arr.metadata.get('frame_units')
        if frame_units is not None:
            metadata_code = self.units_to_code.get(frame_units)
            if metadata_code is None:
                warnings.warn(f'unknown frame units \'{frame_units}\', using seconds instead')
            else:
                frame_units_code = metadata_code

        nii.header['xyzt_units'] = np.asarray(spatial_units_code, dtype=np.uint8) | \
                                   np.asarray(frame_units_code, dtype=np.uint8)

        # geometry-specific header data
        if is_image:
            nii.set_sform(arr.geom.vox2world.matrix, arr.metadata.get('sform_code', 1))
            nii.set_qform(arr.geom.vox2world.matrix, arr.metadata.get('qform_code', 1))
            nii.header['pixdim'][1:4] = arr.geom.voxsize.astype(np.float32)

            # add freesurfer nifti1 header extension
            import io
            from surfa.io.fsnifti1extension import FSNifti1Extension

            # create FSNifti1Extension.Content object from arr
            fsextcontent = FSNifti1Extension.Content(arr)
            
            # create file-like object
            bytes_writer = io.BytesIO()

            # write freesurfer header extension data to buffer
            fsext = FSNifti1Extension()
            esize = fsext.write(bytes_writer, fsextcontent)
            
            # convert FSNifti1Extension.Content to bytes
            fsext_bytesdata = bytes_writer.getvalue()
            niiext = self.nib.nifti1.Nifti1Extension(14, fsext_bytesdata)
            nii.header.extensions.append(niiext)

        # write
        self.nib.save(nii, filename)


class FreeSurferM3ZIO(protocol.IOProtocol):
    """
    Warp IO protocol for FreeSurfer GCA morph files (.m3z, gzip-compressed, and the
    uncompressed .m3d), as written by `GCAMwrite()` and read by `GCAMread()` in
    FreeSurfer's utils/gcamorph.cpp (`__m3zWrite()` and `__m3zRead()`).

    File layout, all values big-endian:

        float   version (1.0, the only version FreeSurfer defines)
        int     width, height, depth (node grid), spacing (atlas voxels per node)
        float   exp_k
        per node, x outermost and z innermost:
            float origx, origy, origz   original (pre-morph) node position
            float x, y, z               current node position in source (image) voxels
            int   xn, yn, zn            node index
        tags, each an int ID without a data length:
            TAG_GCAMORPH_GEOM (10)      source (image) then target (atlas) VOL_GEOM
            TAG_GCAMORPH_TYPE (11)      int, GCAM_VOX (2) or GCAM_RAS (1)
            TAG_GCAMORPH_LABELS (12)    int label per node, in node order
            TAG_MGH_XFORM (31)          optional linear transform, written by
                                        `znzWriteMatrix()` as an int tag (0), a 64-bit
                                        length and a 1600-byte text buffer; older
                                        files hold `MatrixAsciiWriteInto()` text
                                        instead, which is read but not written

    Node (i, j, k) sits at target voxel (i, j, k) * spacing, and its position (x, y, z)
    is the absolute source voxel coordinate it maps to. An m3z file is therefore a
    `Warp.Format.abs_crs` warp. For spacing 1, the warp's target geometry is the atlas
    geometry. For a coarser node grid, the target geometry is the node grid
    (atlas vox2ras scaled by the spacing), and the atlas geometry and spacing are kept
    in the metadata keys 'm3z-atlas' and 'm3z-spacing' so that saving writes them back.

    On save, the warp is converted to abs_crs. The original node positions, the node
    labels and the linear transform are kept in the metadata keys 'm3z-orig',
    'm3z-labels' and 'm3z-affine' when present. Without them, the original positions are
    written as node indices and the labels as zeros, as FreeSurfer does when it converts
    an mgz warp to m3z, and no linear transform is written.
    """

    name = 'm3z'
    extensions = ('.m3z', '.m3d')

    version = 1.0
    gcam_ras = 1
    gcam_vox = 2
    matrix_strlen = 4 * 4 * 100

    # per-node record of __m3zWrite()
    node_dtype = np.dtype([('orig', '>f4', 3), ('pos', '>f4', 3), ('index', '>i4', 3)])

    def _open(self, filename, mode):
        """
        Open an m3z (gzip) or m3d (plain) file, as FreeSurfer decides by extension.
        """
        if str(filename).lower().endswith('.m3z'):
            return gzip.open(filename, mode)
        return open(filename, mode)

    def load(self, filename, atype):
        """
        Read a warp from a FreeSurfer m3z file.

        Parameters
        ----------
        filename : str or Path
            File path to read.
        atype : class
            FramedArray subclass to load. Must be `Warp`.

        Returns
        -------
        Warp
            Warp in abs_crs format, indexed by target voxel.
        """
        if atype is not Warp:
            raise ValueError(f'{filename} is an m3z morph and can only be loaded as a Warp')

        with self._open(filename, 'rb') as file:

            version = read_bytes(file, '>f4')
            if version != self.version:
                raise ValueError(f'unsupported m3z version {version} in {filename}')

            shape = tuple(int(n) for n in read_bytes(file, '>i4', 3))
            spacing = int(read_bytes(file, '>i4'))
            exp_k = float(read_bytes(file, '>f4'))

            count = int(np.prod(shape))
            buffer = file.read(count * self.node_dtype.itemsize)
            if len(buffer) != count * self.node_dtype.itemsize:
                raise ValueError(f'{filename} ends before its {shape} node grid')
            nodes = np.frombuffer(buffer, dtype=self.node_dtype).reshape(shape)

            source = atlas = None
            source_valid = target_valid = True
            source_fname = target_fname = ''
            gcam_type = self.gcam_vox
            labels = None
            affine = None

            # tag loop of __m3zRead(): every tag is lengthless except the matrix
            # that follows TAG_MGH_XFORM, which carries its own tag and length
            while True:
                fbytes = file.read(4)
                if len(fbytes) < 4:
                    break
                tag = int(np.frombuffer(fbytes, dtype='>i4')[0])
                if tag == 0:
                    break
                elif tag == fsio.tags.gcamorph_geom:
                    # file names are C strings: keep what precedes the first null
                    source, source_valid, source_fname = read_geom(file)
                    atlas, target_valid, target_fname = read_geom(file)
                    source_fname = source_fname.split('\x00', 1)[0]
                    target_fname = target_fname.split('\x00', 1)[0]
                elif tag == fsio.tags.gcamorph_type:
                    gcam_type = int(read_bytes(file, '>i4'))
                elif tag == fsio.tags.gcamorph_labels:
                    labels = read_bytes(file, '>i4', count).reshape(shape)
                elif tag == fsio.tags.xform:
                    head = file.read(4)
                    if head == bytes(4):
                        # znzWriteMatrix(): tag 0, 64-bit length, 'Matrix' and 16 values
                        length = int(read_bytes(file, '>i8'))
                        text = file.read(length).decode('utf-8').split('\x00', 1)[0]
                        values = text.split()[1:17]
                    else:
                        # older files: MatrixAsciiWriteInto() text, 'type rows cols'
                        # and 16 values, up to the end of the file
                        text = (head + file.read()).decode('utf-8').split()
                        values = text[3:19]
                    affine = np.asarray(values, dtype=np.float64).reshape(4, 4)
                else:
                    raise ValueError(f'unknown tag {tag} in m3z file {filename}')

        if gcam_type != self.gcam_vox:
            raise NotImplementedError('only m3z morphs with voxel node positions '
                                      f'(GCAM_VOX) are supported, {filename} has type '
                                      f'{gcam_type}')

        # target geometry: the atlas itself or, for a coarser grid, the node grid
        metadata = {}
        if atlas is None:
            target = None
        elif spacing == 1 and tuple(atlas.shape) == shape:
            target = atlas
        else:
            scale = np.diag([spacing, spacing, spacing, 1.0])
            target = ImageGeometry(shape=shape, vox2world=atlas.vox2world.matrix @ scale)
            metadata['m3z-atlas'] = atlas
            metadata['m3z-spacing'] = spacing

        data = np.array(nodes['pos'], dtype=np.float32)
        warp = Warp(data, source=source, target=target, format=Warp.Format.abs_crs)

        warp.metadata.update(metadata)
        warp.metadata['source-valid'] = source_valid
        warp.metadata['source-fname'] = source_fname
        warp.metadata['target-valid'] = target_valid
        warp.metadata['target-fname'] = target_fname
        warp.metadata['exp_k'] = exp_k
        warp.metadata['m3z-orig'] = np.array(nodes['orig'], dtype=np.float32)
        if labels is not None and np.any(labels):
            warp.metadata['m3z-labels'] = labels.astype(np.int32)
        if affine is not None:
            warp.metadata['m3z-affine'] = affine
        return warp

    def save(self, arr, filename):
        """
        Write a warp to a FreeSurfer m3z file.

        Parameters
        ----------
        arr : Warp
            Warp to save, in any format. It is converted to abs_crs.
        filename : str or Path
            Target file path.
        """
        if not isinstance(arr, Warp):
            raise ValueError('only a Warp can be saved as an m3z morph')
        if arr.basedim != 3 or arr.nframes != 3:
            raise ValueError(f'm3z morphs must be 3D warps, got shape {arr.shape}')
        if arr.source is None:
            raise ValueError('saving an m3z morph requires a source geometry')

        shape = tuple(arr.baseshape)
        meta = arr.metadata
        data = arr.convert(Warp.Format.abs_crs, copy=False).data

        # restore the atlas and spacing of a coarse node grid, but only if the
        # target geometry is still the node grid derived from them
        atlas, spacing = arr.target, 1
        coarse_atlas = meta.get('m3z-atlas')
        coarse_spacing = int(meta.get('m3z-spacing', 1))
        if coarse_atlas is not None and coarse_spacing > 1:
            scale = np.diag([coarse_spacing, coarse_spacing, coarse_spacing, 1.0])
            node_grid = ImageGeometry(shape=shape,
                                      vox2world=coarse_atlas.vox2world.matrix @ scale)
            if image_geometry_equal(node_grid, arr.target, tol=1e-4):
                atlas, spacing = coarse_atlas, coarse_spacing

        nodes = np.empty(shape, dtype=self.node_dtype)
        index = np.stack(np.meshgrid(*(np.arange(n) for n in shape), indexing='ij'), -1)
        orig = meta.get('m3z-orig')
        if orig is None or np.shape(orig) != (*shape, 3):
            orig = index
        nodes['orig'] = orig
        nodes['pos'] = data
        nodes['index'] = index

        labels = meta.get('m3z-labels')
        if labels is None or np.shape(labels) != shape:
            labels = np.zeros(shape, dtype=np.int32)

        with self._open(filename, 'wb') as file:
            write_bytes(file, self.version, '>f4')
            write_bytes(file, shape, '>i4')
            write_bytes(file, spacing, '>i4')
            write_bytes(file, meta.get('exp_k', 0.0), '>f4')
            file.write(nodes.tobytes())

            write_int(file, fsio.tags.gcamorph_geom)
            write_geom(file,
                       geom=arr.source,
                       valid=meta.get('source-valid', True),
                       fname=meta.get('source-fname', ''))
            write_geom(file,
                       geom=atlas,
                       valid=meta.get('target-valid', True),
                       fname=meta.get('target-fname', ''))

            write_int(file, fsio.tags.gcamorph_type)
            write_int(file, self.gcam_vox)

            write_int(file, fsio.tags.gcamorph_labels)
            write_bytes(file, np.ravel(labels), '>i4')

            affine = meta.get('m3z-affine')
            if affine is not None:
                affine = np.asarray(affine, dtype=np.float64).reshape(4, 4)
                text = 'Matrix ' + ' '.join('%10f' % v for v in affine.ravel())
                write_int(file, fsio.tags.xform)
                write_int(file, 0)
                write_bytes(file, self.matrix_strlen, '>i8')
                file.write(text.encode('utf-8').ljust(self.matrix_strlen, b'\x00'))


class FreeSurferAnnotationIO(protocol.IOProtocol):
    """
    Array IO protocol for 1D mesh annotation files.
    """
    name = 'annot'
    extensions = '.annot'

    def labels_to_mapping(self, labels):
        """
        The annotation file format saves each vertex label value as a
        bit-manipulated int32 value that represents an RGB. But, the label
        lookup table is embedded in the file, so it's kind of a pointless
        format that could be simply replaced by an MGH file with embedded
        labels, like any other volumetric segmentaton. This function builds
        a mapping to convert label RGBs to a lookup of bit-manipulated int32
        values. Using this mapping, we can convert between a classic integer
        segmentation and annotation-style values.
        """
        rgb = np.array([elt.color[:3].astype(np.int32) for elt in labels.values()])
        idx = np.array(list(labels.keys()))
        mapping = np.zeros(idx.max() + 1, dtype=np.int32)
        mapping[idx] = (rgb[:, 2] << 16) + (rgb[:, 1] << 8) + rgb[:, 0]
        return mapping

    def load(self, filename, atype):
        """
        Read overlay from an annot file.

        Parameters
        ----------
        filename : str or Path
            File path read.
        atype : class
            FramedArray subclass to load. When reading annot files, this
            must be Overlay.

        Returns
        -------
        Overlay
            Array object loaded from file.
        """
        if atype is not Overlay:
            raise ValueError('annotation files can only be loaded as 1D overlays')

        with open(filename, 'rb') as file:

            nvertices = read_bytes(file, '>i4')
            data = np.zeros(nvertices, dtype=np.int32)

            value_map = read_bytes(file, '>i4', nvertices * 2)
            vnos = value_map[0::2]
            vals = value_map[1::2]
            data[vnos] = vals

            tag, length = fsio.read_tag(file)
            if tag is None or tag != fsio.tags.old_colortable:
                raise ValueError('annotation file does not have embedded label lookup data')
            labels = fsio.read_binary_lookup_table(file)

        # cache the zero value annotations (unknown labels)
        unknown_mask = data == 0

        # convert annotation values to corresponding label values
        mapping = self.labels_to_mapping(labels)
        ds = np.argsort(mapping)
        pos = np.searchsorted(mapping[ds], data)
        index = np.take(ds, pos, mode='clip')
        mask = mapping[index] != data
        data = np.ma.array(index, mask=mask)

        # all of the unknown labels should be converted to -1
        data[unknown_mask] = -1

        return Overlay(data, labels=labels)

    def save(self, arr, filename):
        """
        Write overlay to an annot file.

        Parameters
        ----------
        arr : Overlay
            Array to save.
        filename : str or Path
            Target file path.
        """
        if not isinstance(arr, Overlay):
            raise ValueError(f'can only save 1D overlays as annotations, but got array type {type(arr)}')

        if not np.issubdtype(arr.dtype, np.integer):
            raise ValueError(f'annotations must have integer dtype, but overlay has dtype {arr.dtype}')

        if arr.nframes > 1:
            raise ValueError(f'annotations must only have 1 frame, but overlay has {arr.nframes} frames')

        if arr.labels is None:
            raise ValueError('overlay must have label lookup if saving as annotation')

        unknown_mask = arr.data < 0

        # make sure all indices exist in the label lookup
        cleaned = arr.data[np.logical_not(unknown_mask)]
        found = np.in1d(cleaned, list(arr.labels.keys()))
        if not np.all(found):
            missing = list(np.unique(cleaned[found == False]))
            raise ValueError('cannot save overlay as annotation because it contains the following values '
                            f'that do not exist in its label lookup: {missing}')

        cleaned = arr.data.copy()
        cleaned[unknown_mask] = 0
        colors = self.labels_to_mapping(arr.labels)[arr.data]
        colors[unknown_mask] = 0

        with open(filename, 'bw') as file:
            # write the total number of vertices covered by the overlay
            nvertices = arr.shape[0]
            write_bytes(file, nvertices, '>i4')

            # write the data as sequences of (vertex number, color) for every 'vertex'
            # in the annotation, where color is 
            annot = np.zeros(nvertices * 2, dtype=np.int32)
            annot[0::2] = np.arange(nvertices, dtype=np.int32)
            annot[1::2] = colors
            write_bytes(file, annot, '>i4')

            # include the label lookup information
            fsio.write_tag(file, fsio.tags.old_colortable)
            fsio.write_binary_lookup_table(file, arr.labels)


class FreeSurferCurveIO(protocol.IOProtocol):
    """
    Array IO protocol for 1D FS curv files. This is another silly file format that
    could very well just be replaced by MGH files.
    """
    name = 'curv'
    extensions = ()

    def load(self, filename, atype):
        """
        Read overlay from a curv file.

        Parameters
        ----------
        filename : str or Path
            File path read.
        atype : class
            FramedArray subclass to load. When reading curv files, this
            must be Overlay.

        Returns
        -------
        Overlay
            Array object loaded from file.
        """
        if atype is not Overlay:
            raise ValueError('curve files can only be loaded as 1D overlays')
        with open(filename, 'rb') as file:
            magic = read_int(file, size=3)
            nvertices = read_bytes(file, '>i4')
            read_bytes(file, '>i4')
            read_bytes(file, '>i4')
            data = read_bytes(file, '>f4', nvertices)
        return Overlay(data)

    def save(self, arr, filename):
        """
        Write overlay to a curv file.

        Parameters
        ----------
        arr : Overlay
            Array to save.
        filename : str or Path
            Target file path.
        """
        if arr.nframes > 1:
            raise ValueError(f'curv files must only have 1 frame, but overlay has {arr.nframes} frames')

        with open(filename, 'bw') as file:
            write_int(file, -1, size=3)
            write_bytes(file, arr.shape[0], '>i4')
            write_bytes(file, 0, '>i4')
            write_bytes(file, 1, '>i4')
            write_bytes(file, arr.data, '>f4')


class ImageSliceIO(protocol.IOProtocol):
    """
    Generic array IO protocol for common image formats.
    """

    def __init__(self):
        try:
            from PIL import Image
        except ImportError:
            raise ImportError(f'the `pillow` python package must be installed for {self.name} IO')
        self.Image = Image

    def save(self, arr, filename):
        self.Image.fromarray(arr.data).save(filename)

    def load(self, filename, atype):
        image = np.asarray(self.Image.open(filename))
        return Slice(image)


class JPEGArrayIO(ImageSliceIO):
    name = 'jpeg'
    extensions = ('.jpeg', '.jpg')


class PNGArrayIO(ImageSliceIO):
    name = 'png'
    extensions = '.png'


class TIFFArrayIO(ImageSliceIO):
    name = 'tiff'
    extensions = ('.tif', '.tiff')


# enabled array IO protocol classes
array_io_protocols = [
    MGHArrayIO,
    NiftiArrayIO,
    FreeSurferM3ZIO,
    FreeSurferAnnotationIO,
    FreeSurferCurveIO,
    JPEGArrayIO,
    PNGArrayIO,
    TIFFArrayIO,
]
