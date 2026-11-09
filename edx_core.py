"""EDX spectrum-image processing core (no GUI).

* FEI TIA reader (.emi + _N.ser): spectrum image, metadata, survey image
* EDAX Genesis / TEAM reader (.spd map, .lsd line scan, .spc spectrum; .ipr/.bmp electron image)
* electron images for the overlay (TIFF / BMP / PNG / JPEG; FEI, Zeiss, EDAX calibration)
* background: SNIP, polynomial in user windows
* line model: Gaussian families (K/L/M) with energy-dependent FWHM and Si escape peaks
* spectrum fit: bounded weighted least squares + refinement of calibration/resolution
* automatic element identification (iterative fit of residual peaks)
* per-pixel MLLS (peak deconvolution) and window maps; hot-pixel removal (after DM 'Point Blemish')
* PCA: preprocessing (dead channels, energy drift, bad pixels), then Poisson-normalised SVD
  (eigenvalues, loadings, scores), noise level from the Marchenko-Pastur edge
* Richardson-Lucy spectral deconvolution
"""
import glob
import os
import re
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, replace

import numpy as np
from scipy import linalg, ndimage, optimize, sparse
from scipy.signal import find_peaks
from scipy.special import betainc, erfc, erfcx, gammaincc, ndtr, ndtri

from edx_xray_lines import ELEMENTS, ESCAPE_E, ESCAPE_FRAC, SI_KA, SI_K_EDGE

E_MN_KA = 5.8987          # keV, reference line for detector resolution
EPS_SI = 0.00364          # keV per electron-hole pair in Si
FWHM2SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))

GREEK = {'a': 'α', 'b': 'β', 'g': 'γ', 'z': 'ζ', 'n': 'η', 'l': 'ℓ'}

# never proposed by automatic identification (no stable isotopes / gases that do not stay in a TEM sample)
AUTO_EXCLUDED = {'He', 'Ne', 'Kr', 'Xe', 'Rn', 'Tc', 'Pm', 'Po', 'At', 'Fr', 'Ra', 'Ac', 'Pa', 'Np', 'Pu', 'Am'}
# less frequent elements get a lower prior in automatic identification
RARE = {'Sc', 'Br', 'Rb', 'I', 'Cs', 'Pr', 'Sm', 'Eu', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
        'Re', 'Os', 'Tl', 'Th', 'U', 'Be', 'B', 'Li', 'Tc', 'Rh', 'Ir'}


def line_label(sym):
    """'Ka1' -> 'Kα1', 'Lb2,15' -> 'Lβ2,15'."""
    if len(sym) < 2:
        return sym
    return sym[0] + GREEK.get(sym[1], sym[1]) + sym[2:]


# ----------------------------------------------------------------------------------------------
# TIA reader
# ----------------------------------------------------------------------------------------------
SER_DTYPES = {1: '<u1', 2: '<u2', 3: '<u4', 4: '<i1', 5: '<i2', 6: '<i4', 7: '<f4', 8: '<f8'}


def read_ser(path):
    """Read a TIA .ser file. Returns dict(data, dims, cal, kind)."""
    mm = np.memmap(path, dtype=np.uint8, mode='r')
    hdr = mm[:30].tobytes()
    byte_order, series_id, version = struct.unpack('<3h', hdr[:6])
    if byte_order != 0x4949 or series_id != 0x0197:
        raise ValueError(f'{os.path.basename(path)}: not a TIA .ser file')
    data_type, tag_type, n_total, n_valid = struct.unpack('<4i', hdr[6:22])
    pos = 22
    if version >= 0x220:
        off_array, = struct.unpack('<q', mm[pos:pos + 8].tobytes())
        pos += 8
        itype = '<i8'
    else:
        off_array, = struct.unpack('<i', mm[pos:pos + 4].tobytes())
        pos += 4
        itype = '<i4'
    n_dims, = struct.unpack('<i', mm[pos:pos + 4].tobytes())
    pos += 4
    dims = []
    for _ in range(n_dims):
        size, = struct.unpack('<i', mm[pos:pos + 4].tobytes())
        c_off, c_delta = struct.unpack('<2d', mm[pos + 4:pos + 20].tobytes())
        c_el, n_desc = struct.unpack('<2i', mm[pos + 20:pos + 28].tobytes())
        pos += 28
        desc = mm[pos:pos + n_desc].tobytes().decode('latin-1')
        pos += n_desc
        n_unit, = struct.unpack('<i', mm[pos:pos + 4].tobytes())
        pos += 4
        units = mm[pos:pos + n_unit].tobytes().decode('latin-1')
        pos += n_unit
        dims.append(dict(size=size, offset=c_off, delta=c_delta, element=c_el, desc=desc, units=units))

    isz = np.dtype(itype).itemsize
    offsets = np.frombuffer(mm[off_array:off_array + n_total * isz].tobytes(), dtype=itype)[:n_valid]
    first = int(offsets[0])
    head = mm[first:first + 64].tobytes()
    if data_type == 0x4120:      # 1D arrays (spectra)
        c_off, c_delta, c_el, dt, length = struct.unpack('<2dihi', head[:26])
        hdr_len, item_shape = 26, (length,)
        cal = dict(offset=c_off, delta=c_delta, element=c_el)
    elif data_type == 0x4122:    # 2D arrays (images)
        cx0, cdx, cxe, cy0, cdy, cye, dt, sx, sy = struct.unpack('<2di2dih2i', head[:50])
        hdr_len, item_shape = 50, (sy, sx)
        cal = dict(x0=cx0, dx=cdx, y0=cy0, dy=cdy)
    else:
        raise ValueError(f'unsupported TIA data type 0x{data_type:x}')
    dtype = np.dtype(SER_DTYPES[dt])
    n_items = int(np.prod(item_shape))

    out = np.zeros((n_total, n_items), dtype=dtype)
    steps = np.diff(offsets)
    if n_valid > 1 and np.all(steps == steps[0]) and steps[0] >= hdr_len + n_items * dtype.itemsize:
        view = np.ndarray(shape=(n_valid, n_items), dtype=dtype, buffer=mm, offset=first + hdr_len,
                          strides=(int(steps[0]), dtype.itemsize))
        out[:n_valid] = view
    else:
        for k, o in enumerate(offsets):
            o = int(o) + hdr_len
            out[k] = np.frombuffer(mm[o:o + n_items * dtype.itemsize].tobytes(), dtype=dtype)
    del mm
    spatial = tuple(d['size'] for d in dims[::-1])     # first dimension is the fast (x) one
    data = out.reshape(spatial + item_shape)
    kind = 'spectra' if data_type == 0x4120 else 'images'
    return dict(data=data, dims=dims, cal=cal, kind=kind)


def _xml_text(node, path, default=None):
    el = node.find(path)
    return el.text if el is not None and el.text is not None else default


def read_emi(path):
    """Metadata and (if found) the survey image stored in a TIA .emi file."""
    raw = open(path, 'rb').read()
    meta, survey = {}, None
    i = raw.find(b'<ObjectInfo>')
    j = raw.find(b'</ObjectInfo>', i)
    if i >= 0 and j > 0:
        try:
            root = ET.fromstring(raw[i:j + 13].decode('latin-1'))
            v = _xml_text(root, 'ExperimentalConditions/MicroscopeConditions/AcceleratingVoltage')
            if v:
                meta['Voltage (kV)'] = float(v) / 1000.0
            for d in root.iter('Data'):
                lab, val, unit = d.findtext('Label'), d.findtext('Value'), d.findtext('Unit') or ''
                if lab and val is not None:
                    meta[lab] = (val.strip() + ' ' + unit).strip()
            for tag in ('AcquireDate', 'AcquireInfo/DwellTimePath', 'AcquireInfo/FrameTime', 'Manufacturer'):
                val = _xml_text(root, tag)
                if val:
                    meta[tag.split('/')[-1]] = val
        except ET.ParseError:
            pass
    try:
        survey = _find_survey(raw, i if i > 0 else len(raw))
    except Exception:
        survey = None
    return meta, survey


def _find_survey(raw, limit):
    """Heuristic search for the 'Spectrum image Scanning' reference image (uint16) in the .emi."""
    for mk in re.finditer(b'Spectrum image Scanning', raw[:limit]):
        k = mk.start()
        for p in range(k, min(k + 400, limit - 8)):
            w, h = struct.unpack('<2i', raw[p:p + 8])
            if not (32 <= w <= 8192 and 32 <= h <= 8192):
                continue
            end = p + 8 + w * h * 2
            if end > limit:
                continue
            cal = {}
            for m in re.finditer(rb'A\x00(.)\x01(.{8})', raw[end:end + 160], re.S):
                cal.setdefault(m.group(1)[0], struct.unpack('<d', m.group(2))[0])
            if 0x00 not in cal or 0x20 not in cal:
                continue
            img = np.frombuffer(raw[p + 8:end], dtype='<u2').reshape(h, w).astype(np.float32)
            return dict(image=img, x0=cal.get(0x00), dx=cal.get(0x20), y0=cal.get(0x10), dy=cal.get(0x30))
    return None


@dataclass
class SpectrumImage:
    data: np.ndarray            # (ny, nx, n_channels), raw counts
    energy: np.ndarray          # keV, channel centres
    pixel_size: float = 0.0     # m
    origin: tuple = (0.0, 0.0)  # (x, y) of pixel (0, 0), m
    step: tuple = (0.0, 0.0)    # signed (dx, dy) per pixel, m
    meta: dict = field(default_factory=dict)
    survey: dict = None
    haadf: np.ndarray = None    # image on the SI pixel grid (HAADF in TIA, electron image in EDAX)
    name: str = ''
    path: str = ''
    image_name: str = 'HAADF'   # what `haadf` is
    fwhm_mn: float = None       # keV, detector resolution at Mn Ka stored in the file (EDAX)
    image_full: np.ndarray = None   # `haadf` at the resolution of the image file (same field of view)
    dispersion: float = 1.0     # variance / mean of the counts: 1 = Poisson (see count_dispersion)

    @property
    def shape(self):
        return self.data.shape[:2]

    @property
    def disp(self):
        return float(self.energy[1] - self.energy[0])

    @property
    def voltage(self):
        return float(self.meta.get('Voltage (kV)', 300.0))

    def sum_spectrum(self):
        return self.data.sum(axis=(0, 1), dtype=np.float64)

    def roi_spectrum(self, y0, y1, x0, x1):
        return self.data[y0:y1, x0:x1].sum(axis=(0, 1), dtype=np.float64)

    def total_image(self, e_min=0.0):
        i0 = int(np.searchsorted(self.energy, e_min))
        return self.data[..., i0:].sum(axis=-1, dtype=np.float64).astype(np.float32)

    def si_rect_m(self):
        """SI field of view (xmin, ymin, xmax, ymax) in m, for drawing on the survey image."""
        ny, nx = self.shape
        xs = [self.origin[0], self.origin[0] + (nx - 1) * self.step[0]]
        ys = [self.origin[1], self.origin[1] + (ny - 1) * self.step[1]]
        return min(xs), min(ys), max(xs), max(ys)


def count_dispersion(data, chunk=4096):
    """Variance / mean of the counts of a channel in a pixel (1 for Poisson counts), from the sparse
    channels (< 0.01 count per pixel on average, else < 0.03), where a count is a single event:
    sum(x^2 - lambda^2) / sum(x). EDAX Genesis / TEAM store dead-time corrected counts - a part of
    the events is recorded twice (2 counts in one channel of one pixel far more often than Poisson
    allows) - the variance is then ~1.4 x the counts. Values below 1.05 are taken as 1."""
    cube = data.reshape(-1, data.shape[-1])
    npix = cube.shape[0]
    b = np.zeros(cube.shape[1])
    for s in range(0, npix, chunk):
        b += cube[s:s + chunk].sum(axis=0, dtype=np.float64)
    mu = b / npix
    for lim in (0.01, 0.03):
        j = np.flatnonzero((mu > 0) & (mu < lim))
        if b[j].sum() >= 2000:
            break
    else:
        return 1.0      # too few counts in sparse channels to tell
    s2 = 0.0
    for s in range(0, npix, chunk):
        x = cube[s:s + chunk, j].astype(np.int64)
        s2 += float((x * x).sum())
    d = (s2 - npix * float((mu[j] ** 2).sum())) / float(b[j].sum())
    return 1.0 if d < 1.05 else float(min(d, 4.0))


def load_tia(path):
    """Load a TIA spectrum image from the .emi or any of its _N.ser files."""
    path = os.path.abspath(path)
    low = path.lower()
    if low.endswith('.emi'):
        stem, emi = path[:-4], path
    elif low.endswith('.ser'):
        m = re.match(r'(.*)_(\d+)\.ser$', path, re.I)
        stem = m.group(1) if m else path[:-4]
        emi = stem + '.emi'
    else:
        raise ValueError('ожидается файл .emi или .ser (FEI TIA)')
    sers = sorted(glob.glob(glob.escape(stem) + '_*.ser'))
    if low.endswith('.ser') and path not in sers:
        sers.insert(0, path)
    if not sers:
        raise FileNotFoundError(f'не найдено файлов {os.path.basename(stem)}_N.ser рядом с .emi')

    si, others = None, []
    for s in sers:
        r = read_ser(s)
        if r['kind'] == 'spectra' and r['data'].shape[-1] > 16 and si is None:
            si = r
        else:
            others.append(r)
    if si is None:
        raise ValueError('в серии нет спектров (не спектральное изображение)')

    data = si['data']
    if data.ndim == 2:          # line scan or single spectrum -> (1, n, E)
        data = data[None]
    elif data.ndim == 1:
        data = data[None, None]
    cal = si['cal']
    n_ch = data.shape[-1]
    energy = (cal['offset'] + (np.arange(n_ch) - cal['element']) * cal['delta']) / 1000.0  # eV -> keV

    dims = si['dims']
    dx = dims[0]['delta'] if dims else 0.0
    dy = dims[1]['delta'] if len(dims) > 1 else dx
    x0 = dims[0]['offset'] - dims[0]['element'] * dx if dims else 0.0
    y0 = dims[1]['offset'] - dims[1]['element'] * dy if len(dims) > 1 else 0.0

    haadf = None
    for r in others:
        d = r['data']
        if r['kind'] == 'spectra' and d.shape[-1] == 1:
            d = d[..., 0]
        if d.shape == data.shape[:2]:
            haadf = d.astype(np.float32)
            break

    meta, survey = (read_emi(emi) if os.path.exists(emi) else ({}, None))
    meta['Pixels'] = f'{data.shape[1]} x {data.shape[0]}'
    meta['Channels'] = f'{n_ch} x {cal["delta"]:.2f} eV'
    return SpectrumImage(data=data, energy=energy, pixel_size=abs(dx), origin=(x0, y0), step=(dx, dy),
                         meta=meta, survey=survey, haadf=haadf, name=os.path.basename(stem), path=path)


# ----------------------------------------------------------------------------------------------
# EDAX Genesis / TEAM reader
# Byte layouts from the EDAX specifications 'SpcMap-spd', 'SPECTRUM V70' and 'ImageIPR'
# (as used by RosettaSciIO / HyperSpy).
# ----------------------------------------------------------------------------------------------
SPD_HEADER = np.dtype([('tag', 'S16'), ('version', '<i4'), ('nSpectra', '<i4'), ('nPoints', '<i4'),
                       ('nLines', '<i4'), ('nChannels', '<i4'), ('countBytes', '<i4'), ('dataOffset', '<i4'),
                       ('nFrames', '<i4'), ('fName', 'S120')])
EDAX_DETECTORS = {1: 'Si(Li)', 2: 'Si(Li) UTW', 3: 'Si(Li) Super UTW', 30: 'Apollo 10 SDD', 31: 'Apollo XV SDD',
                  32: 'Apollo 10+ SDD', 40: 'Apollo 40 SDD', 50: 'Apollo X SDD', 51: 'Apollo XP SDD',
                  52: 'Apollo XL SDD', 53: 'Apollo XL-XRF SDD', 60: 'Apollo XLT-LS SDD', 61: 'Apollo XLT-NW SDD',
                  62: 'Apollo XLT-SUTW SDD'}


def read_spc(path):
    """EDAX .spc spectrum (format V0.61-0.70): counts and the header values used here.
    energy of channel i = e_start + i * ev_per_ch / 1000 (keV)."""
    b = open(path, 'rb').read()
    name = os.path.basename(path)
    if len(b) < 3840:
        raise ValueError(f'{name}: файл .spc слишком короткий')

    def f4(o):
        return float(struct.unpack_from('<f', b, o)[0])

    start, = struct.unpack_from('<i', b, 28)
    n, = struct.unpack_from('<h', b, 32)
    ev, = struct.unpack_from('<i', b, 384)
    if not (0 < n <= 4096 and 0 < ev < 1000 and 0 < start and start + 4 * n <= len(b)):
        raise ValueError(f'{name}: не похоже на спектр EDAX .spc')
    year, = struct.unpack_from('<h', b, 16)
    day, mon, minute, hour = b[18], b[19], b[20], b[21]
    date = (f'{year}-{mon:02d}-{day:02d} {hour:02d}:{minute:02d}'
            if 1990 < year < 2200 and 1 <= mon <= 12 and 1 <= day <= 31 else '')
    z2sym = {v[0]: k for k, v in ELEMENTS.items()}
    n_el, = struct.unpack_from('<h', b, 638)
    zs = struct.unpack_from('<48H', b, 640)[:max(0, min(n_el, 48))]
    kv, fwhm = f4(532), f4(472)
    return dict(counts=np.frombuffer(b, '<u4', n, start).copy(), ev_per_ch=int(ev), e_start=f4(448),
                kv=kv if 0 < kv < 1000 else None, live_time=f4(456), tilt=f4(460), takeoff=f4(464),
                fwhm_mn=fwhm if 40 < fwhm < 400 else None, elevation=f4(512), azimuth=f4(508),
                detector=struct.unpack_from('<I', b, 476)[0], date=date,
                elements=sorted({z2sym[z] for z in zs if z in z2sym}, key=lambda s: ELEMENTS[s][0]))


def read_ipr(path):
    """EDAX .ipr description of an image: µm per pixel, magnification, high tension."""
    b = open(path, 'rb').read()
    if len(b) < 72:
        raise ValueError(f'{os.path.basename(path)}: файл .ipr слишком короткий')
    acc, = struct.unpack_from('<H', b, 52)
    mag, = struct.unpack_from('<I', b, 58)
    mpp_x, mpp_y = struct.unpack_from('<2f', b, 64)
    return dict(kv=acc / 10.0, mag=mag, mpp_x=mpp_x, mpp_y=mpp_y)


def _read_bitmap(path):
    """Grey-scale image (float32) from a .bmp; None if it cannot be read."""
    try:
        return _read_raster(path)
    except Exception:
        return None


def load_edax(path):
    """Load an EDAX Genesis / TEAM spectrum image (.spd), line scan (.lsd) or spectrum (.spc).

    The .spc of the same name gives the energy calibration, high tension and detector data (without
    it 10 eV/channel from 0 is assumed). The electron image named in the .spd header (.bmp) and its
    calibration (.ipr) are used when they lie next to the .spd. An .spc opened next to an .spd of
    the same name is the sum spectrum of that map: the map is opened."""
    path = os.path.abspath(path)
    stem, ext = os.path.splitext(path)
    ext = ext.lower()
    if ext == '.spc':
        for e in ('.spd', '.lsd'):
            if os.path.exists(stem + e):
                path, ext = stem + e, e
                break
    if ext not in ('.spd', '.lsd', '.spc'):
        raise ValueError('ожидается файл .spd, .lsd или .spc (EDAX Genesis / TEAM)')
    spc = read_spc(stem + '.spc') if os.path.exists(stem + '.spc') else None
    meta = {'Software': 'EDAX Genesis / TEAM'}
    survey = haadf = image_full = ipr = None
    image_name = 'HAADF'
    px = px_y = 0.0

    if ext == '.spc':
        data = spc['counts'][None, None]
    else:
        h = np.fromfile(path, dtype=SPD_HEADER, count=1)
        if not len(h) or not h['tag'][0].startswith(b'MAPSPECTRA_DATA'):
            raise ValueError(f'{os.path.basename(path)}: не файл EDAX {ext} (нет метки MAPSPECTRA_DATA)')
        h = h[0]
        nx, ny, nch = int(h['nPoints']), int(h['nLines']), int(h['nChannels'])
        dt = {1: '<u1', 2: '<u2', 4: '<u4'}.get(int(h['countBytes']))
        if dt is None or nx < 1 or ny < 1 or nch < 2:
            raise ValueError(f'{os.path.basename(path)}: неверный заголовок ({nx}×{ny}×{nch}, '
                             f'{int(h["countBytes"])} байт на отсчёт)')
        n = nx * ny * nch
        data = np.fromfile(path, dtype=dt, count=n, offset=int(h['dataOffset']))
        if data.size < n:
            raise ValueError(f'{os.path.basename(path)}: файл обрезан ({data.size} из {n} отсчётов)')
        data = data.reshape(ny, nx, nch)
        meta['Frames'] = str(int(h['nFrames']))
        if ext == '.lsd':
            data = data.reshape(1, nx * ny, nch)[:, :max(1, min(int(h['nSpectra']), nx * ny))]
        else:
            # electron image acquired with the map and its calibration
            img = h['fName'].split(b'\0')[0].decode('latin-1').strip()
            bases = ([os.path.splitext(os.path.basename(img))[0]] if img else []) + [os.path.basename(stem) + '_Img']
            folder, bmp = os.path.dirname(path), None
            for base in bases:
                p = os.path.join(folder, base)
                if ipr is None and os.path.exists(p + '.ipr'):
                    try:
                        ipr = read_ipr(p + '.ipr')
                    except (ValueError, struct.error):
                        ipr = None
                for ext_img in ('.bmp', '.tif', '.tiff'):     # TEAM saves the image named .bmp as .tif
                    if bmp is None and os.path.exists(p + ext_img):
                        bmp = _read_bitmap(p + ext_img)
                        if bmp is not None:
                            image_name = ('Electron image'
                                          + (f' ({base.rsplit("_", 1)[1]})' if '_' in base else ''))
            ipx = ipy = None
            if ipr and ipr['mpp_x'] > 0:
                ipx = ipr['mpp_x'] * 1e-6
                ipy = ipr['mpp_y'] * 1e-6 if ipr['mpp_y'] > 0 else ipx
                k = bmp.shape[1] / nx if bmp is not None else 1.0    # image pixels per map pixel
                px, px_y = ipx * k, ipy * k
                if ipr['mag']:
                    meta['Magnification'] = f'{ipr["mag"]}x'
            if bmp is not None:
                survey = dict(image=bmp, x0=0.0, dx=ipx, y0=0.0, dy=ipy)
                if abs(bmp.shape[1] / bmp.shape[0] / (nx / ny) - 1) <= IMAGE_ASPECT_TOL:   # covers the map field
                    image_full, haadf = bmp, resize_image(bmp, (ny, nx))
            elif img:
                meta['missing_image'] = img         # named in the .spd, not found (.bmp / .tif)

    n_ch = data.shape[-1]
    if spc is not None:
        ev, e0 = spc['ev_per_ch'], spc['e_start']
        kv = spc['kv'] or (ipr['kv'] if ipr and ipr['kv'] > 0 else None)
        if spc['date']:
            meta['AcquireDate'] = spc['date']
        if spc['live_time'] > 0:
            meta['Live time'] = f'{spc["live_time"]:.1f} s'
            meta['live_time_s'] = float(spc['live_time'])
        det = EDAX_DETECTORS.get(spc['detector'], f'type {spc["detector"]}')
        meta['EDS detector'] = det + (f', FWHM(Mn Kα) {spc["fwhm_mn"]:.0f} eV' if spc['fwhm_mn'] else '')
        meta['Take-off angle'] = f'{spc["takeoff"]:.1f}°'
        if spc['elements']:
            meta['Elements (EDAX)'] = ', '.join(spc['elements'])
    else:
        ev, e0 = 10, 0.0
        kv = ipr['kv'] if ipr and ipr['kv'] > 0 else None
        meta['no_spc'] = os.path.basename(stem) + '.spc'      # calibration 10 eV/channel from 0 assumed
    if kv:
        meta['Voltage (kV)'] = float(kv)
        meta['High tension'] = f'{kv:g} kV'
    meta['Pixels'] = f'{data.shape[1]} x {data.shape[0]}'
    meta['Channels'] = f'{n_ch} x {ev:.2f} eV'
    energy = e0 + np.arange(n_ch) * ev / 1000.0
    return SpectrumImage(data=data, energy=energy, pixel_size=px, step=(px, px_y or px), meta=meta,
                         survey=survey, haadf=haadf, name=os.path.basename(stem), path=path,
                         image_name=image_name, image_full=image_full,
                         fwhm_mn=spc['fwhm_mn'] / 1000.0 if spc is not None and spc['fwhm_mn'] else None)


def load_si(path):
    """Spectrum image from any supported file: FEI TIA (.emi/.ser) or EDAX (.spd/.lsd/.spc)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.spd', '.lsd', '.spc'):
        si = load_edax(path)
    elif ext in ('.emi', '.ser'):
        si = load_tia(path)
    else:
        raise ValueError('неизвестный формат: ожидается .emi / .ser (FEI TIA) или .spd / .lsd / .spc (EDAX)')
    si.dispersion = count_dispersion(si.data)
    return si


# ----------------------------------------------------------------------------------------------
# Electron images (SEM / STEM) loaded separately, for the overlay of the maps
# ----------------------------------------------------------------------------------------------
IMAGE_FILTER = 'Изображения (*.tif *.tiff *.bmp *.png *.jpg *.jpeg)'
IMAGE_ASPECT_TOL = 0.03     # relative difference of width/height up to which an image covers the map field
_LENGTH_UNITS = {'pm': 1e-12, 'nm': 1e-9, 'um': 1e-6, 'µm': 1e-6, 'μm': 1e-6, 'mm': 1e-3, 'm': 1.0}


def _grey(a):
    """2-D grey-scale float32 image from a grey, RGB(A) or channel-first array."""
    a = np.asarray(a)
    if a.ndim == 3:
        a = a[..., :3].mean(axis=-1) if a.shape[-1] in (3, 4) else (a[:3].mean(axis=0) if a.shape[0] in (3, 4)
                                                                    else a[0])
    if a.ndim != 2 or min(a.shape) < 2:
        raise ValueError(f'не изображение (размер {a.shape})')
    return a.astype(np.float32)


def _read_raster(path):
    """Grey-scale float32 image from TIFF (tifffile, any bit depth) or BMP / PNG / JPEG (Pillow)."""
    if path.lower().endswith(('.tif', '.tiff')):
        try:
            import tifffile
            return _grey(tifffile.imread(path, key=0))
        except Exception:
            pass        # compression / layout tifffile cannot read: try Pillow
    from PIL import Image
    with Image.open(path) as im:
        if im.mode in ('P', 'PA', '1', 'LA', 'CMYK', 'YCbCr', 'HSV', 'LAB'):
            im = im.convert('RGB')
        return _grey(np.asarray(im))


def _sem_tiff_meta(path):
    """Pixel size (m) and scanned area (width, height; an SEM data bar lies below it) from the metadata
    of an FEI / Thermo Fisher or Zeiss SEM TIFF."""
    out = {}
    try:
        import tifffile
        with tifffile.TiffFile(path) as tf:
            fei, zeiss = tf.fei_metadata, tf.sem_metadata
    except Exception:
        return out
    try:
        if fei:
            pw = float(fei.get('Scan', {}).get('PixelWidth') or 0)
            res = fei.get('Image', {})
            rx, ry = int(res.get('ResolutionX') or 0), int(res.get('ResolutionY') or 0)
            if pw > 0:
                out.update(px=pw, calib='метаданные FEI')
            if rx > 0 and ry > 0:
                out['scan'] = (rx, ry)
        elif zeiss:
            v = zeiss.get('ap_image_pixel_size') or zeiss.get('ap_pixel_size')
            if v and len(v) >= 3 and isinstance(v[1], (int, float)) and v[2] in _LENGTH_UNITS and v[1] > 0:
                out.update(px=v[1] * _LENGTH_UNITS[v[2]], calib='метаданные Zeiss')
    except (TypeError, ValueError, AttributeError):
        pass
    return out


def read_image(path):
    """Electron image from a TIFF / BMP / PNG / JPEG file: dict(image=float32 (h, w) grey scale,
    px=pixel size in m (if known), calib=its source, scan=(w, h) of the scanned area if the file
    holds more (SEM data bar)). Calibration from FEI / Thermo Fisher and Zeiss SEM TIFF metadata
    or an EDAX .ipr of the same name."""
    out = dict(image=_read_raster(path))
    if path.lower().endswith(('.tif', '.tiff')):
        out.update(_sem_tiff_meta(path))
    ipr = os.path.splitext(path)[0] + '.ipr'
    if 'px' not in out and os.path.exists(ipr):
        try:
            r = read_ipr(ipr)
            if r['mpp_x'] > 0:
                out.update(px=r['mpp_x'] * 1e-6, calib='EDAX .ipr')
        except (ValueError, struct.error):
            pass
    return out


def resize_image(a, shape):
    """Image resampled to shape (rows, cols) over the same field of view (float32): area average
    when reduced, bilinear interpolation when enlarged."""
    a = np.ascontiguousarray(a, dtype=np.float32)
    h, w = int(shape[0]), int(shape[1])
    if a.shape == (h, w):
        return a
    from PIL import Image
    filt = Image.BOX if h <= a.shape[0] and w <= a.shape[1] else Image.BILINEAR
    return np.asarray(Image.fromarray(a).resize((w, h), filt), dtype=np.float32)


# ----------------------------------------------------------------------------------------------
# Detector response and line families
# ----------------------------------------------------------------------------------------------
@dataclass
class Detector:
    fwhm_mn: float = 0.130   # keV, FWHM at Mn Ka
    fano: float = 0.114
    spread: float = 0.0      # relative gain spread (e.g. between the 4 Super-X detectors)
    offset: float = 0.0      # keV, energy calibration correction  E_true = E*(1+gain) + offset
    gain: float = 0.0
    tail: float = 0.0        # fraction of every line in the low-energy tail (incomplete charge collection)
    tail_width: float = 1.0  # tail slope, in units of FWHM(E)
    escape: bool = True
    shifts: dict = field(default_factory=dict)   # keV, position corrections of single families {'Nb M': -0.017}
    thr: float = 0.0         # keV, low-energy cut-off of the recorded spectrum (pulse-processor threshold); 0 = none
    thr_width: float = 0.01  # keV, width of the cut-off (Gaussian edge)
    # line shapes refined by the NLLS fit, per family key: width factor and intensity factors of
    # the line groups relative to the tabulated weights {'Fe L': {'Lb': 0.8, 'Ll': 1.4}}
    widths: dict = field(default_factory=dict)
    ratios: dict = field(default_factory=dict)

    def fwhm(self, e):
        """FWHM^2 = FWHM_Mn^2 + 8ln2*F*eps*(E - E_Mn) + (2.355*spread)^2*(E^2 - E_Mn^2)."""
        e = np.asarray(e, dtype=float)
        f2 = (self.fwhm_mn ** 2 + 8 * np.log(2) * self.fano * EPS_SI * (e - E_MN_KA)
              + (self.spread / FWHM2SIGMA) ** 2 * (e ** 2 - E_MN_KA ** 2))
        return np.sqrt(np.maximum(f2, 0.02 ** 2))

    def sigma(self, e):
        return self.fwhm(e) * FWHM2SIGMA

    def true_energy(self, e):
        return np.asarray(e) * (1.0 + self.gain) + self.offset

    def threshold(self, e):
        """Recorded fraction of the counts at the recorded energy e (low-energy cut-off)."""
        e = np.asarray(e, dtype=float)
        if self.thr <= 0:
            return np.ones_like(e)
        return ndtr((e - self.thr) / max(self.thr_width, 1e-4))

    def measured_energy(self, e, key=None):
        """Position on the recorded energy axis of a line with tabulated energy e of family `key`."""
        return (np.asarray(e) + self.shifts.get(key, 0.0) - self.offset) / (1.0 + self.gain)


def escape_fraction(e):
    e = np.asarray(e, dtype=float)
    f = np.interp(e, ESCAPE_E, ESCAPE_FRAC, left=0.0, right=0.0)
    return np.where(e > SI_K_EDGE, f, 0.0)


@dataclass(frozen=True)
class Family:
    element: str
    family: str             # 'K', 'L', 'M'

    @property
    def key(self):
        return f'{self.element} {self.family}'

    @property
    def lines(self):
        return ELEMENTS[self.element][3][self.family]

    @property
    def main(self):
        """(name, energy, weight) of the strongest line."""
        return max(self.lines, key=lambda t: t[2])

    @property
    def edge(self):
        return ELEMENTS[self.element][2][self.family]


# lightest element for which a family is used in fits (EDS practice: weaker series are unreliable);
# for Y..Ba the M family is the M-zeta line (0.13-0.8 keV), clearly visible e.g. for Nb at 0.17 keV
FAMILY_MIN_Z = {'K': 1, 'L': 21, 'M': 39}
# K and M families with the main line below this energy get an individual position correction when
# the calibration is refined: incomplete charge collection and the non-linearity of the energy scale
# move such peaks by up to a few tens of eV (Nb M-zeta, C K, O K)
LOW_E_LINES = 0.6       # keV
SHIFT_PRIOR = 0.010     # keV, prior width of these corrections
SHIFT_MAX = 0.040       # keV
# the M family of Y..Ba is much weaker than its L family (Nb: M/L ~ 0.1); its counts are limited to
# this fraction of the L counts, so that it cannot take over a light-element line (Rh M vs C K)
M_TO_L_MAX = 0.5
# L and M families with the main line above this energy (Nb L, Pt M, I L...: heavy elements) stay out of the
# detector refinement during the identification (fit_spectrum refine_skip_lm)
HEAVY_LM_E = 1.5        # keV
HEAVY_LM_SHARE = 0.1    # ... when they carry at least this share of the counts in the lines
# in maps, families with the main line below this energy may be tied to the main family of the element
TIE_LOW_E = 1.0         # keV
# NLLS line-shape fit: Gaussian priors (shift in keV, width and group ratios as natural logs) and bounds
NLLS_PRIOR = dict(shift=0.010, width=0.15, ratio=0.4)
NLLS_BOUNDS = dict(shift=(-0.05, 0.05), width=(np.log(0.6), np.log(1.8)), ratio=(np.log(0.2), np.log(5.0)))
NLLS_SCALE = dict(shift=0.003, width=0.05, ratio=0.1)


def element_families(el, e_lo, e_hi, e0_kev=300.0):
    """Families of `el` whose main line lies in [e_lo, e_hi] and that can be excited."""
    if el not in ELEMENTS:
        return []
    out = []
    for fam in ('K', 'L', 'M'):
        if fam not in ELEMENTS[el][3] or ELEMENTS[el][0] < FAMILY_MIN_Z[fam]:
            continue
        f = Family(el, fam)
        if e_lo <= f.main[1] <= e_hi and f.edge < e0_kev:
            out.append(f)
    return out


def lines_near(e, tol, min_weight=0.02, e0_kev=300.0):
    """All lines within tol of energy e: list of (dE, element, family, line, E, weight)."""
    res = []
    for el, (z, name, edges, fams) in ELEMENTS.items():
        for fam, lines in fams.items():
            if edges.get(fam, 0) >= e0_kev:
                continue
            for sym, le, w in lines:
                if w >= min_weight and abs(le - e) <= tol:
                    res.append((le - e, el, fam, sym, le, w))
    res.sort(key=lambda t: (abs(t[0]) / max(t[5], 0.05) ** 0.25))
    return res


def line_group(name):
    """Group of a line inside its family: 'Ka1' -> 'Ka', 'Lb2,15' -> 'Lb', 'Mz' -> 'Mz'."""
    return name[:2]


def peak_matrix(energy, families, det, disp):
    """Design matrix (n_channels, n_families); each column integrates to ~1 over all lines
    so the fitted coefficient equals the total number of counts in the family."""
    et = det.true_energy(energy)
    half = 0.5 * disp * (1.0 + det.gain)
    lo, hi = et - half, et + half
    A = np.zeros((len(energy), len(families)))
    for j, f in enumerate(families):
        sh = det.shifts.get(f.key, 0.0)
        kw = det.widths.get(f.key, 1.0)
        rat = det.ratios.get(f.key, {})
        lines = [(n, e, w * rat.get(line_group(n), 1.0)) for n, e, w in f.lines]
        wsum = sum(t[2] for t in lines)
        for _, le, w in lines:
            le = le + sh
            w = w / wsum
            pk = [(le, w, kw * det.sigma(le))]
            if det.escape and le > SI_K_EDGE:
                fe = float(escape_fraction(le))
                if fe > 1e-5:
                    pk = [(le, w * (1 - fe), kw * det.sigma(le)),
                          (le - SI_KA, w * fe, kw * det.sigma(max(le - SI_KA, 0.1)))]
            for c, ww, s in pk:
                i1 = np.searchsorted(et, c + 6 * s)
                g = 1.0 - det.tail
                if g > 0:
                    i0 = np.searchsorted(et, c - 6 * s)
                    if i1 > i0:
                        A[i0:i1, j] += g * ww * (ndtr((hi[i0:i1] - c) / s) - ndtr((lo[i0:i1] - c) / s))
                if det.tail > 0:
                    beta = det.tail_width * s / FWHM2SIGMA
                    i0 = np.searchsorted(et, c - 12 * beta - 6 * s)
                    if i1 > i0:
                        A[i0:i1, j] += det.tail * ww * _tail(et[i0:i1], c, s, beta) * disp * (1.0 + det.gain)
    if det.thr > 0:
        A *= det.threshold(energy)[:, None]
    return A


def _tail(e, c, s, beta):
    """Unit-area low-energy exponential tail convolved with the Gaussian (Hypermet / EMG)."""
    u = e - c
    x = u / (np.sqrt(2) * s) + s / (np.sqrt(2) * beta)
    out = np.empty_like(u)
    neg = x < 0
    a = u[neg] / beta + s ** 2 / (2 * beta ** 2)
    out[neg] = np.exp(a) * erfc(x[neg])
    out[~neg] = np.exp(-u[~neg] ** 2 / (2 * s ** 2)) * erfcx(x[~neg])
    return out / (2 * beta)


def bg_basis(energy, shape, n_terms, noise=None):
    """Background basis: shape(E) * Legendre_k(x), k < n_terms, plus (optionally) the decaying
    tail of the detector noise peak noise=(centre, sigma). shape may hold several columns
    (backgrounds of different phases), each is multiplied by the polynomial."""
    x = 2 * (energy - energy[0]) / max(energy[-1] - energy[0], 1e-9) - 1
    P = np.polynomial.legendre.legvander(x, max(n_terms - 1, 0))
    if shape is not None:
        s = np.asarray(shape, dtype=float)
        s = s / np.maximum(np.mean(np.abs(s), axis=0), 1e-12)
        P = np.hstack([P * c[:, None] for c in s.T]) if s.ndim == 2 else P * s[:, None]
    if noise is not None:
        c0, s0 = noise
        col = np.exp(-np.maximum(energy - c0, 0) / (1.5 * s0))
        if col.max() > 1e-3:
            P = np.hstack([P, col[:, None]])
    return P


PILEUP_LOW_E = 0.6     # keV: pulses below ~this (C, N, O K) escape the fast pile-up rejector more often


def pileup_shape(energy, y, e_min=0.1, normalize=True, partner_max=None):
    """Shape of the pile-up (sum) peaks of a spectrum: its autoconvolution (two events counted as one
    at E1 + E2; C+O at 0.80 keV, O+Na at 1.57 keV, Ti Ka + C K at 4.79 keV...), from the channels above
    e_min (above the noise peak / threshold), on the energy axis, maximum 1 (normalize). partner_max:
    only the sums with a partner below this energy (light-element pulses, which the fast channel of the
    pulse processor often does not see and so does not reject). The amount is fitted: the counts of the
    sum peaks are tau / T x the autoconvolution (tau: pulse-pair resolving time, T: live time; ~1 % of
    the counts at 20 kcps)."""
    energy = np.asarray(energy, dtype=float)
    s = np.where(energy > e_min, np.maximum(np.asarray(y, dtype=float), 0.0), 0.0)
    p = s if partner_max is None else np.where(energy < partner_max, s, 0.0)
    n = 2 * len(s) - 1
    nfft = 1 << (n - 1).bit_length()
    c = np.fft.irfft(np.fft.rfft(s, nfft) * np.fft.rfft(p, nfft), nfft)[:n]
    out = np.interp(energy, 2 * energy[0] + np.arange(n) * (energy[1] - energy[0]), np.maximum(c, 0.0))
    return out / out.max() if normalize and out.max() > 0 else out


def _with_pileup(B0, pileup, sel):
    """Background basis with the pile-up shape(s) (n_channels or n_channels x k) as its first column(s)
    (fitted >= 0 together with the families); (basis, number of pile-up columns)."""
    if pileup is None:
        return B0, 0
    P = np.asarray(pileup, dtype=float)
    P = (P[:, None] if P.ndim == 1 else P)[sel]
    P = P[:, P.max(axis=0) > 0]
    if not P.shape[1]:
        return B0, 0
    return np.hstack([P / P.max(axis=0), B0]), P.shape[1]


def recorded_edge(energy, y):
    """Index of the first fully recorded channel. The pulse processor discards the counts below a
    threshold (~0.1 keV), above it the spectrum rises within a few channels; 0 if nothing is cut."""
    y = np.asarray(y, float)
    nz = np.flatnonzero(y > 0)
    if len(nz) == 0 or nz[0] == 0:
        return 0
    i = int(nz[0])
    seg = y[i:int(np.searchsorted(energy, energy[i] + 0.1)) + 1]
    return i + int(np.argmax(seg >= 0.5 * seg.max()))


def estimate_threshold(energy, y):
    """Starting values (thr, width) in keV of the low-energy cut-off, from the rising edge of a
    spectrum whose lowest channels are empty; None if nothing is cut."""
    i = recorded_edge(energy, y)
    if i == 0:
        return None
    return float(energy[i]) - 0.01, 0.01


def detect_noise_peak(energy, y, e_max=0.25):
    """Detector noise / zero peak below e_max keV: (centre, sigma) or None.

    The noise peak is centred at 0 keV; when the channels below the threshold are discarded only
    its decaying tail is left, starting at the first recorded channel. A maximum well above the
    threshold with the counts rising towards it is an X-ray line (e.g. Nb M-zeta at 0.17 keV)."""
    m = (energy > 0.0) & (energy < e_max)
    if m.sum() < 5:
        return None
    ym = np.asarray(y, float)[m]
    i = int(np.argmax(ym))
    ref = np.median(np.asarray(y, float)[(energy > 0.3) & (energy < 0.6)]) if np.any(energy > 0.6) else 0
    if ym[i] < 2 * max(ref, 1):
        return None
    e = energy[m]
    half = ym[i] / 2
    j = i
    while j + 1 < len(ym) and ym[j] > half:
        j += 1
    s0 = max((e[j] - e[i]) / 1.1774, energy[1] - energy[0])
    k = int(np.argmax(ym >= 0.1 * ym[i]))      # start of the rising edge
    if e[i] > 3 * s0 and e[i] - e[k] > 2.5 * (energy[1] - energy[0]):
        return None
    return float(e[i]), float(s0)


# ----------------------------------------------------------------------------------------------
# Background
# ----------------------------------------------------------------------------------------------
def snip_background(energy, y, det, width=2.0, smooth=True, lls=True):
    """SNIP background (Ryan 1988 / Morhac 1997) with energy-dependent clipping window
    (width x FWHM(E)) and LLS transform."""
    raw = np.asarray(y, dtype=float)
    y = ndimage.gaussian_filter1d(raw, 1.0) if smooth else raw.copy()
    i0 = recorded_edge(energy, raw)
    if i0 > 0:
        # nothing is recorded below the threshold: continue the spectrum there with the lowest level
        # just above it, otherwise a peak at the edge (Nb M) pulls the background down to zero
        y[:i0] = y[i0:np.searchsorted(energy, energy[i0] + 0.3)].min()
    v = np.log(np.log(np.sqrt(np.maximum(y, 0) + 1) + 1) + 1) if lls else y.copy()
    disp = energy[1] - energy[0]
    w = np.maximum((width * det.fwhm(np.maximum(energy, 0.1)) / disp).astype(int), 1)
    n = len(v)
    idx = np.arange(n)
    for p in range(1, int(w.max()) + 1):
        m = (w >= p) & (idx >= p) & (idx < n - p)
        ii = idx[m]
        a = 0.5 * (v[ii - p] + v[ii + p])
        v[ii] = np.minimum(v[ii], a)
    if lls:
        v = (np.exp(np.exp(v) - 1) - 1) ** 2 - 1
    v = np.maximum(v, 0)
    v[:i0] = np.minimum(v[:i0], raw[:i0])
    return v


def minima_background(energy, y, smooth=5, ncycles=20):
    """Background through the minima of the spectrum (DM script 'Autobackground SI from minima').

    Nodes are the local minima of the spectrum smoothed with a Gaussian kernel of `smooth`
    channels (sigma = smooth/5), the node value is the raw count there. In each of `ncycles`
    passes every node higher than both neighbouring nodes is replaced by their mean; the
    background is the polyline through the nodes. Unlike the DM script the first and the last
    nodes are kept (the script loses one end node per pass), zero-count minima are nodes too and
    the background is continued at a constant level beyond the end nodes."""
    raw = np.asarray(y, dtype=float)
    n = len(raw)
    i0 = recorded_edge(energy, raw)
    s = ndimage.gaussian_filter1d(raw, smooth / 5, truncate=2.5) if smooth > 1 else raw
    k = np.arange(max(i0, 1), n - 1)
    k = k[(s[k] < s[k - 1]) & (s[k] < s[k + 1])]
    if len(k) < 2:
        raise ValueError('в спектре не найдено минимумов для фона')
    v = raw[k]
    for _ in range(ncycles):
        mid = v[1:-1]
        v[1:-1] = np.where((mid > v[:-2]) & (mid > v[2:]), 0.5 * (v[:-2] + v[2:]), mid)
    bg = np.interp(np.arange(n), k, v)
    bg[:i0] = np.minimum(bg[:i0], raw[:i0])
    return bg


def poly_background(energy, y, windows, order=3):
    """Weighted polynomial fit of y inside background windows [(e1, e2), ...]."""
    y = np.asarray(y, dtype=float)
    m = np.zeros(len(energy), bool)
    for a, b in windows:
        m |= (energy >= min(a, b)) & (energy <= max(a, b))
    if m.sum() <= order + 1:
        raise ValueError('слишком мало каналов в окнах фона')
    x = 2 * (energy - energy[m].min()) / max(np.ptp(energy[m]), 1e-9) - 1
    w = 1 / np.sqrt(np.maximum(y[m], 1))
    c = np.polynomial.legendre.legfit(x[m], y[m], order, w=w)
    return np.maximum(np.polynomial.legendre.legval(x, c), 0)


def parse_windows(text):
    """'0.9-1.1, 3-5.2' -> [(0.9, 1.1), (3.0, 5.2)]."""
    out = []
    for part in re.split(r'[;,]', text):
        m = re.match(r'\s*([\d.]+)\s*[-:]\s*([\d.]+)\s*$', part)
        if m:
            out.append((float(m.group(1)), float(m.group(2))))
    return out


# ----------------------------------------------------------------------------------------------
# Spectrum fitting
# ----------------------------------------------------------------------------------------------
@dataclass
class FitResult:
    families: list
    coef: np.ndarray        # net counts per family
    err: np.ndarray         # 1 sigma
    bg_coef: np.ndarray
    det: Detector
    erange: tuple
    sel: np.ndarray         # channel mask of the fit range
    model: np.ndarray       # full energy axis
    background: np.ndarray
    components: np.ndarray  # (n_families, n_channels)
    chi2_red: float
    pileup: np.ndarray = None   # sum peaks of the model (full energy axis), None if not modelled

    def table(self):
        rows = []
        for f, c, e in zip(self.families, self.coef, self.err):
            name, en, _ = f.main
            rows.append(dict(element=f.element, family=f.family, line=line_label(name), energy=en,
                             counts=c, err=e, sigma=c / e if e > 0 else 0.0))
        return rows


def _solve(M, y, var, n_peaks, nonneg=True, ub=None):
    """Weighted least squares; the first n_peaks coefficients >= 0 (nonneg); ub: upper bounds (or None)."""
    w = 1.0 / np.sqrt(var)
    Mw = M * w[:, None]
    yw = y * w
    if nonneg and n_peaks:
        lb = np.r_[np.zeros(n_peaks), np.full(M.shape[1] - n_peaks, -np.inf)]
        res = optimize.lsq_linear(Mw, yw, bounds=(lb, np.inf if ub is None else ub), method='bvls')
        c = res.x
    else:
        c = np.linalg.lstsq(Mw, yw, rcond=None)[0]
    r = yw - Mw @ c
    return c, float(r @ r), Mw


def _tie_families(families):
    """(j_M, j_L) index pairs: the M family of an element with Z < 57 and its L family."""
    idx = {f.key: j for j, f in enumerate(families)}
    return [(j, idx[f'{f.element} L']) for j, f in enumerate(families)
            if f.family == 'M' and ELEMENTS[f.element][0] < 57 and f'{f.element} L' in idx]


def _merge_ratio(M, pairs):
    """Design matrix with column j added to column k at the ratio r, for pairs {j: (k, r)}."""
    if not pairs:
        return M
    M = M.copy()
    for j, (k, r) in pairs.items():
        M[:, k] += r * M[:, j]
        M[:, j] = 0.0
    return M


def _merge_ties(M, active):
    """Design matrix with the M column of each active pair added to its L column at M_TO_L_MAX."""
    return _merge_ratio(M, {jm: (jl, M_TO_L_MAX) for jm, jl in active})


def _solve_tied(M, y, var, n_peaks, ties, nonneg=True, ub=None):
    """_solve with c_M <= M_TO_L_MAX * c_L for the tied pairs (active set: a violated pair is refitted
    with the ratio at its maximum). Returns c, chi2, Mw (merged), active pairs."""
    active = []
    for _ in range(3):
        c, chi2, Mw = _solve(_merge_ties(M, active), y, var, n_peaks, nonneg, ub)
        for jm, jl in active:
            c[jm] = M_TO_L_MAX * c[jl]
        new = [t for t in ties if t not in active and c[t[0]] > M_TO_L_MAX * max(c[t[1]], 0.0) + 1e-9]
        if not new:
            break
        active += new
    return c, chi2, Mw, active


def _refine_low(E, Y, var, families, det, B0, disp, refine_thr=False):
    """Low-energy refinement: individual position corrections of the families below LOW_E_LINES
    keV (Gaussian prior of width SHIFT_PRIOR) and, optionally, the low-energy cut-off (thr,
    thr_width), which shapes the same peaks (Nb M, C K). The other columns are computed once.
    B0: background basis without the cut-off. L families (Ti..Cr L) keep the common calibration:
    with their several lines of uncertain ratios a free shift lets them slide onto O K / N K."""
    idx = [j for j, f in enumerate(families) if f.main[1] < LOW_E_LINES and f.family != 'L']
    if not idx and not refine_thr:
        return det
    keys = [families[j].key for j in idx]
    low = [families[j] for j in idx]
    ns = len(idx)
    nf = len(families)
    ties = _tie_families(families)
    A = peak_matrix(E, families, replace(det, thr=0.0), disp)
    w = 1.0 / np.sqrt(var)
    p0 = list(np.clip([det.shifts.get(k, 0.0) for k in keys], -0.99 * SHIFT_MAX, 0.99 * SHIFT_MAX))
    lo, hi, xs = [-SHIFT_MAX] * ns, [SHIFT_MAX] * ns, [0.005] * ns
    if refine_thr:
        p0 += [np.clip(det.thr, 0.021, 0.39), np.clip(det.thr_width, 0.0021, 0.049)]
        lo += [0.02, 0.002]
        hi += [0.4, 0.05]
        xs += [0.003, 0.002]

    def unpack(p):
        d = replace(det, shifts={**det.shifts, **{k: float(v) for k, v in zip(keys, p[:ns])}})
        return replace(d, thr=float(p[ns]), thr_width=float(p[ns + 1])) if refine_thr else d

    def resid(p):
        d = unpack(p)
        if ns:
            A[:, idx] = peak_matrix(E, low, replace(d, thr=0.0), disp)
        T = d.threshold(E)[:, None]
        c, _, Mw, _ = _solve_tied(np.hstack([A * T, B0 * T]), Y, var, nf, ties)
        return np.r_[Y * w - Mw @ c, np.asarray(p[:ns]) / SHIFT_PRIOR]

    r = optimize.least_squares(resid, np.array(p0), bounds=(lo, hi), x_scale=xs, diff_step=1e-3,
                               max_nfev=80, loss='soft_l1', f_scale=3.0)
    return unpack(r.x)


def free_groups(family, others=(), det=None, min_share=0.03):
    """Line groups of a family whose intensity relative to the main group may be refined
    (K-beta, L-beta, L-gamma, L-l, M-beta...): all but the main one carrying >= min_share and not
    lying within 1 FWHM of a strong line (>= 20 % of its family) of another family in `others`:
    such a ratio would just trade counts between the elements (Pt M-beta under Nb L-alpha)."""
    groups, top = {}, {}
    for n, e, w in family.lines:
        g = line_group(n)
        groups[g] = groups.get(g, 0.0) + w
        if w > top.get(g, (0, 0))[1]:
            top[g] = (e, w)
    main = line_group(family.main[0])
    total = sum(groups.values())
    det = det or Detector()
    out = []
    for g, w in groups.items():
        if g == main or w < min_share * total:
            continue
        e = top[g][0]
        fw = float(det.fwhm(max(e, 0.1)))
        if any(o != family and abs(le - e) < fw and lw >= 0.2 for o in others for _, le, lw in o.lines):
            continue
        out.append(g)
    return out


def strip_nlls(det):
    """Detector without the NLLS line shapes (keeps the low-energy K/M shifts of the calibration)."""
    keep = {}
    for k, v in det.shifts.items():
        el, fam = k.split()
        if el in ELEMENTS and fam != 'L' and fam in ELEMENTS[el][3] and Family(el, fam).main[1] < LOW_E_LINES:
            keep[k] = v
    return replace(det, shifts=keep, widths={}, ratios={})


def _refine_nlls(E, Y, var, families, det, B, disp, free=('shift', 'width', 'ratio'), min_sigma=10.0, bounds=None,
                 max_nfev=30):
    """Non-linear least squares of the line shapes. Every family detected above min_sigma gets
    its own position, width factor and intensity ratios of its line groups (Kb/Ka, Lb/La...),
    with Gaussian priors (NLLS_PRIOR) that keep poorly determined parameters at the detector-model
    values. Amplitudes and background are solved linearly at every step (variable projection);
    only the column of the family whose parameter changes is recomputed."""
    nf = len(families)
    A = peak_matrix(E, families, det, disp)
    c, _, Mw = _solve(np.hstack([A, B]), Y, var, nf, True)
    err = np.sqrt(np.maximum(np.diag(np.linalg.pinv(Mw.T @ Mw))[:nf], 0))
    # families at the zero bound of the non-negative fit stay out (an unconstrained negative
    # amplitude would let the neighbouring shapes adapt to it)
    keep = np.r_[c[:nf] > 0, np.ones(B.shape[1], bool)]
    present = [f for j, f in enumerate(families) if c[j] > 0]
    params = []                     # (family index, kind, group)
    for j, f in enumerate(families):
        if err[j] <= 0 or c[j] < min_sigma * err[j]:
            continue
        params += [(j, kind, None) for kind in ('shift', 'width') if kind in free]
        if 'ratio' in free:
            params += [(j, 'ratio', g) for g in free_groups(f, present, det)]
    if not params:
        return det
    p0, lo, hi, xs, sig = [], [], [], [], []
    for j, kind, g in params:
        key = families[j].key
        v = {'shift': det.shifts.get(key, 0.0), 'width': np.log(det.widths.get(key, 1.0)),
             'ratio': np.log(det.ratios.get(key, {}).get(g, 1.0))}[kind]
        a, b = (bounds or {}).get(kind, NLLS_BOUNDS[kind])
        p0.append(float(np.clip(v, a + 1e-3 * (b - a), b - 1e-3 * (b - a))))
        lo.append(a)
        hi.append(b)
        xs.append(NLLS_SCALE[kind])
        sig.append(NLLS_PRIOR[kind])
    sig = np.array(sig)
    by_fam = {}
    for i, (j, _, _) in enumerate(params):
        by_fam.setdefault(j, []).append(i)

    def shapes(p, fams):
        """Detector with the NLLS shapes of the given families taken from p."""
        shifts, widths, ratios = dict(det.shifts), dict(det.widths), {k: dict(v) for k, v in det.ratios.items()}
        for j in fams:
            key = families[j].key
            for i in by_fam[j]:
                _, kind, g = params[i]
                if kind == 'shift':
                    shifts[key] = float(p[i])
                elif kind == 'width':
                    widths[key] = float(np.exp(p[i]))
                else:
                    ratios.setdefault(key, {})[g] = float(np.exp(p[i]))
        return replace(det, shifts=shifts, widths=widths, ratios=ratios)

    cache = {}
    M = np.hstack([A, B])
    w = 1.0 / np.sqrt(var)

    def resid(p):
        for j, idx in by_fam.items():
            key = tuple(p[i] for i in idx)
            if cache.get(j, (None,))[0] != key:
                cache[j] = (key, peak_matrix(E, [families[j]], shapes(p, [j]), disp)[:, 0])
            M[:, j] = cache[j][1]
        cc, _, Mw_ = _solve(M[:, keep], Y, var, int(keep[:nf].sum()), nonneg=False)
        return np.r_[Y * w - Mw_ @ cc, np.asarray(p) / sig]

    r = optimize.least_squares(resid, np.array(p0), bounds=(lo, hi), x_scale=xs, diff_step=1e-3,
                               max_nfev=max_nfev)
    return shapes(r.x, list(by_fam))


def fit_spectrum(energy, y, families, det, bg_shape, n_bg=3, erange=(0.2, 20.0),
                 refine_calib=False, refine_res=False, refine_tail=False, nonneg=True, noise=None, nlls=(),
                 dispersion=1.0, pileup=None, nlls_min_sigma=10.0, nlls_bounds=None, refine_skip_lm=False):
    """Weighted least-squares fit of a spectrum with fixed-shape line families + background
    (bg_shape x Legendre polynomial). Optionally refines energy calibration (incl. the positions
    of the families below LOW_E_LINES keV), resolution (FWHM_Mn, Fano, gain spread) and the
    low-energy tail with a robust non-linear fit. nlls: line-shape parameters of the significant
    families refined by NLLS afterwards, any of ('shift', 'width', 'ratio'). dispersion: variance /
    counts (1 = Poisson; EDAX dead-time corrected counts ~1.4), scales the errors. pileup: shape of
    the sum peaks (pileup_shape) - a non-negative model term besides the background.
    refine_skip_lm: if the model has L or M families of heavy elements (main line >= HEAVY_LM_E), scale,
    resolution and tail are refined only on the channels of the K lines above LOW_E_LINES, away from the
    heavy lines; without such a K family the detector stays as given. Heavy
    L / M families overlap their neighbours (Pt M / Au M, 72 eV apart) and have uncertain ratios: with only
    Au in the model a refinement widened the peaks to 158 eV to cover Pt + Au, and moved the scale."""
    y = np.asarray(y, dtype=float)
    disp = energy[1] - energy[0]
    sel = (energy >= erange[0]) & (energy <= erange[1])
    E, Y = energy[sel], y[sel]
    shape = None if bg_shape is None else np.maximum(np.asarray(bg_shape)[sel], 1e-6 * max(np.max(bg_shape), 1))
    B0, npu = _with_pileup(bg_basis(E, shape, n_bg, noise), pileup, sel)
    B = B0 * det.threshold(E)[:, None]
    nf = len(families)
    var0 = dispersion * np.maximum(Y, 1.0)

    # The gain of the energy scale needs two anchors: main lines above LOW_E_LINES at least 1 keV apart
    # (with only Ti found, the gain moved Ti L onto O K, Ti K stayed); else only the offset is refined.
    # In a high-statistics spectrum (SEM) the lines
    # below, which get their own position corrections (_refine_low), stay out of this step: their incomplete
    # charge collection with millions of counts would pull the scale (C, O shifted Na, P by ~10 eV)
    heavy = [f for f in families if f.family in ('L', 'M') and f.main[1] >= HEAVY_LM_E] if refine_skip_lm else []
    if heavy:       # only if they carry a good part of the line counts (not a trace, or a candidate on trial)
        c0 = _solve(np.hstack([peak_matrix(E, families, det, disp), B]), Y, var0, nf + npu, True)[0][:nf]
        hc = sum(c0[j] for j, f in enumerate(families) if f in heavy)
        if hc < HEAVY_LM_SHARE * max(float(np.sum(np.maximum(c0, 0))), 1.0):
            heavy = []
    # with heavy families: only the K families above LOW_E_LINES (one element, known ratios; the L families
    # of Fe..Zn at 0.7-1 keV have uncertain ratios too - Zn L moved the scale of a Pt + Zn + Mo spectrum)
    light = [f for f in families if f.family == 'K' and f.main[1] >= LOW_E_LINES] if heavy else families
    anchors = sorted(f.main[1] for f in light if f.main[1] >= LOW_E_LINES)
    two_anchors = len(anchors) >= 2 and anchors[-1] - anchors[0] >= 1.0
    upper = (E >= LOW_E_LINES if anchors and high_statistics(Y) and np.sum(E >= LOW_E_LINES) > 10
             else np.ones(len(E), bool))
    res_free = ['fwhm_mn', 'fano', 'spread']
    if heavy:
        # only the channels of the other families' lines, away from the strong heavy lines (also no peaks of
        # elements not yet in the model, which would pull the scale - Mo K at 17.4 keV in a Pt + Pd + Zn spectrum)
        near = np.zeros(len(E), bool)
        for f in light:
            for _, le, w in f.lines:
                if w >= 0.05:
                    near |= np.abs(E - le) < 1.5 * float(det.fwhm(le))
        for f in heavy:
            for _, le, w in f.lines:
                if w >= 0.1:
                    near &= np.abs(E - le) >= float(det.fwhm(le))
        upper &= near
        if not anchors:         # nothing left to refine on
            upper[:] = False
        elif not two_anchors:   # one energy: the width there, not its energy dependence
            res_free = ['fwhm_mn']
    lims = dict(offset=(-0.1, 0.1), gain=(-0.03, 0.03), fwhm_mn=(0.06, 0.30), fano=(0.10, 0.13),
                spread=(0.0, 0.01), tail=(0.0, 0.25), tail_width=(0.3, 6.0))
    scale = dict(offset=0.005, gain=0.002, fwhm_mn=0.005, fano=0.01, spread=0.001, tail=0.01, tail_width=0.2)
    start = dict(tail=0.02, spread=0.001)

    def refine(det, free, mask):
        """Robust non-linear fit of the detector parameters `free` on the channels `mask`."""
        p0 = []
        for k in free:
            lo_k, hi_k = lims[k]
            v = getattr(det, k) or start.get(k, 0.0)
            p0.append(np.clip(v, lo_k + 1e-3 * (hi_k - lo_k), hi_k - 1e-3 * (hi_k - lo_k)))

        def resid(p):
            d = replace(det, **dict(zip(free, p)))
            M = np.hstack([peak_matrix(E, families, d, disp), B0 * d.threshold(E)[:, None]])
            c, _, Mw = _solve(M, Y, var0, nf, nonneg=False)
            return (Y / np.sqrt(var0) - Mw @ c)[mask]

        # robust loss: peaks of elements that are not (yet) in the model must not bias the detector
        r = optimize.least_squares(resid, np.array(p0), bounds=([lims[k][0] for k in free], [lims[k][1] for k in free]),
                                   x_scale=[scale[k] for k in free], diff_step=1e-3, max_nfev=100, loss='soft_l1',
                                   f_scale=3.0)
        return replace(det, **dict(zip(free, r.x)))

    # scale, resolution and tail from the lines above LOW_E_LINES (one tail for all lines cannot follow both
    # the soft C, N, O K peaks and e.g. Ti K; the low-energy lines get their own positions in _refine_low)
    free = ((['offset', 'gain'] if two_anchors else ['offset']) if refine_calib else []) \
        + (res_free if refine_res else []) + (['tail', 'tail_width'] if refine_tail else [])
    if free and nf and upper.sum() > 10:
        det = refine(det, free, upper)
    if refine_calib and nf:
        cut = det.thr > 0 and erange[0] < det.thr + 5 * det.thr_width     # the range reaches the cut-off
        det = _refine_low(E, Y, var0, families, det, B0, disp, refine_thr=cut)
        B = B0 * det.threshold(E)[:, None]
    if nlls and nf:
        M = np.hstack([peak_matrix(E, families, det, disp), B])
        c = _solve(M, Y, var0, nf, True)[0]
        det = _refine_nlls(E, Y, dispersion * np.maximum(M @ c, 1.0), families, det, B, disp, free=tuple(nlls),
                           min_sigma=nlls_min_sigma, bounds=nlls_bounds)

    M = np.hstack([peak_matrix(E, families, det, disp), B])
    ties = _tie_families(families)
    var = var0
    for _ in range(2):   # re-weight with the model (less bias at low counts)
        c, chi2, Mw, active = _solve_tied(M, Y, var, nf + npu, ties, nonneg)
        var = dispersion * np.maximum(M @ c, 1.0)
    c, chi2, Mw, active = _solve_tied(M, Y, var, nf + npu, ties, nonneg)
    try:
        cov = np.linalg.pinv(Mw.T @ Mw)
        err = np.sqrt(np.maximum(np.diag(cov), 0))
    except np.linalg.LinAlgError:
        err = np.full(M.shape[1], np.nan)
    for jm, jl in active:
        err[jm] = M_TO_L_MAX * err[jl]

    n = len(energy)
    model = np.zeros(n)
    background = np.zeros(n)
    piled = np.zeros(n)
    comps = np.zeros((nf, n))
    model[sel] = M @ c
    background[sel] = B[:, npu:] @ c[nf + npu:]
    piled[sel] = B[:, :npu] @ c[nf:nf + npu]
    for j in range(nf):
        comps[j, sel] = M[:, j] * c[j]
    dof = max(sel.sum() - M.shape[1], 1)
    return FitResult(families=list(families), coef=c[:nf], err=err[:nf], bg_coef=c[nf + npu:], det=det,
                     erange=tuple(erange), sel=sel, model=model, background=background,
                     components=comps, chi2_red=chi2 / dof, pileup=piled if npu else None)


# ----------------------------------------------------------------------------------------------
# Automatic element identification
# ----------------------------------------------------------------------------------------------
def _window_sums(values, energy, det, k=0.6):
    """Sum of values within +-k*FWHM(E) around every channel (cumulative sums)."""
    disp = energy[1] - energy[0]
    h = np.maximum(np.ceil(k * det.fwhm(np.maximum(energy, 0.1)) / disp).astype(int), 1)
    idx = np.arange(len(energy))
    cs = np.r_[0.0, np.cumsum(values)]
    lo = np.clip(idx - h, 0, len(energy))
    hi = np.clip(idx + h + 1, 0, len(energy))
    return cs[hi] - cs[lo]


def _families_consistent(fams, c, err, matched, e0_kev, threshold, ratio=0.3):
    """The family that matched the peak must be significant by itself, and the element's other
    well detected families (main line 1.5..20 keV, overvoltage > 2.5) must carry at least
    `ratio` of its counts (e.g. Lu M at 1.58 keV requires Lu L at 7.66 keV)."""
    d = {f.family: (f, cc, ee) for f, cc, ee in zip(fams, c, err)}
    if matched not in d:
        return False
    _, cm, em = d[matched]
    if cm < threshold * em:
        return False
    for fam, (f, cc, ee) in d.items():
        if fam == matched or not (1.5 <= f.main[1] <= 20.0) or e0_kev < 2.5 * f.edge:
            continue
        expected = ratio * cm
        if expected > 2 * ee and cc < expected:
            return False
    return True


PILEUP_MAX = 0.1    # a sum peak holds at most this fraction of the counts of its weaker parent family
ID_FAMILY_RATIO = 0.3   # other well detected families of a candidate carry >= this share of its matched family
ID_REFINE_SKIP_LM = True    # detector refinements during the identification without heavy L / M families (fit_spectrum)
ID_MIN_OVERVOLTAGE = 1.5    # E0 / E_edge of a line by which an element may be identified
ID_SHIFT_MIN_SIGMA = 100.0  # families this significant get their own position in the identification fits
ID_SHIFT_MAX = 0.020       # keV, bound of that position correction (not enough to take over a neighbour)
ID_SHAPE_PARAMS = ('shift', 'width')    # line-shape parameters of those families refined in the identification fits
# their bounds: enough for the small shape errors, not to take over a neighbouring line (C K vs N K)
ID_SHAPE_BOUNDS = dict(shift=(-ID_SHIFT_MAX, ID_SHIFT_MAX))
# light elements tested once more on the complete model (high statistics; K lines hidden under L / M lines)
ID_LIGHT_CHECK = ('B', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'K', 'Ca')
ID_SHOULDER_FRAC = 0.02    # high statistics: a candidate on the shoulder of a found line needs this share of its counts
# counts in the strongest channel above which the model systematics (line positions, sum peaks) exceed the
# noise (SEM spectra with ~1e7 counts; TEM spectrum images ~1e6): sum peaks and line positions are fitted
ID_HIGH_COUNTS = 1e5


def high_statistics(y):
    """The spectrum has so many counts that sum peaks and small line shifts matter (ID_HIGH_COUNTS)."""
    return float(np.max(y)) >= ID_HIGH_COUNTS
PILEUP_TAU = 2e-6   # s, upper bound of the pulse-pair resolving time of an SDD (~0.3-1 us)


def _pileup_peaks(fit_result, n_strong, live_time=None):
    """Possible sum peaks of the n strongest fitted families: (E_i + E_j of the main lines, largest
    plausible counts). An event of line i piles up with one of line j with the probability tau x the
    rate of j, so the sum peak holds ~tau c_i c_j / T counts (twice that for i != j; T = live time,
    tau <= PILEUP_TAU), and never more than PILEUP_MAX of its weaker parent (Na Ka at 1.04 keV next to
    a strong O Ka is not the O+O sum peak, Al Ka at 1.49 keV not the Na Ka + Ti L sum peak)."""
    fams, c = fit_result['fams'], fit_result['c']
    if not fams:
        return []
    order = [i for i in np.argsort(c)[::-1][:n_strong] if c[i] > 0]
    out = []
    for a, i in enumerate(order):
        for j in order[a:]:
            n_max = PILEUP_MAX * min(c[i], c[j])
            if live_time:
                n_max = min(n_max, (1 if i == j else 2) * PILEUP_TAU * c[i] * c[j] / live_time)
            out.append((fams[i].main[1] + fams[j].main[1], n_max))
    return out


@dataclass
class IdResult:
    elements: list          # identified element symbols, in order of discovery
    details: list           # dicts: element, energy (peak), line, line_energy, dchi2, sigma
    unidentified: list      # peak energies without a convincing candidate
    det: Detector = None    # detector refined during identification


def auto_identify(energy, y, det, bg, erange=(0.2, 20.0), e0_kev=300.0, threshold=5.0,
                  start=(), exclude=(), systematic=0.10, max_elements=25, refine=True, noise=None,
                  log=None, dispersion=1.0, live_time=None, pileup=None, shapes=True):
    """Iterative identification: take the most significant residual peak, test every element
    that has a line there by refitting the whole spectrum, keep the best one, repeat.

    * a candidate must explain the peak itself (not only gain chi2 elsewhere);
    * if its matched family is strong, its other families in the well detected range
      (1..25 keV) must be present as well (e.g. Zr L at 2.04 keV requires Zr K at 15.8 keV);
    * the variance contains a systematic term (systematic * peak model)^2 so that small
      shape imperfections of strong lines do not create phantom elements;
    * residual peaks at E1 + E2 of the strongest lines are treated as pile-up (sum) peaks, if they
      are weak enough for that (see _pileup_peaks; live_time: s, of the spectrum, if known);
    * a peak is matched only by a line excited with the overvoltage E0 / E_edge >= ID_MIN_OVERVOLTAGE,
      weaker excited families stay out of the model (near E0 the background is uncertain, such a line
      would be far stronger in another family); an element with only one usable family whose confirming
      family is not excited needs twice the threshold;
    * high statistics (high_statistics: SEM spectra with ~1e7 counts), where the model systematics exceed
      the noise: pileup=True with live_time - all sum peaks of the lines found so far (autoconvolution
      of the model, resolving time <= PILEUP_TAU) are fitted; shapes=True - the detector refinements also
      fit the positions of the strongest families (>= ID_SHIFT_MIN_SIGMA, within ID_SHIFT_MAX), so that
      their small misfits do not look like other elements; the returned detector keeps the line shapes
      it was given (only the calibration of the low-energy lines is updated);
    * dispersion: variance / counts (1 = Poisson, EDAX ~1.4) for the significance of the peaks;
    * rare elements need a twice larger chi2 gain;
    * after the first elements are found, energy calibration and resolution are refined;
    * finally, elements whose families are all below the threshold are dropped."""
    y = np.asarray(y, dtype=float)
    disp = energy[1] - energy[0]
    sel = (energy >= erange[0]) & (energy <= erange[1])
    E, Y = energy[sel], y[sel]
    shape = np.maximum(np.asarray(bg)[sel], 1e-6 * max(np.max(bg), 1))
    high = high_statistics(Y)
    Bbg = bg_basis(E, shape, 2, noise)
    # sum peaks (high statistics, live time known): autoconvolution of the current model - only sums of the
    # lines found so far - with at most PILEUP_TAU of resolving time
    use_pile = bool(pileup) and high and bool(live_time)
    pile = dict(shape=None, ub=None)

    def update_pileup(model):
        """Sum peaks of the model (all pairs of its counts)."""
        if use_pile:
            full = np.zeros(len(energy))
            full[sel] = model
            raw = pileup_shape(energy, full, erange[0], normalize=False)
            if raw[sel].max() > 0:
                pile.update(shape=raw, ub=PILEUP_TAU / live_time * float(raw[sel].max()))
    excl = set(exclude) | AUTO_EXCLUDED
    cache = {}
    det_in = det
    nlls = ID_SHAPE_PARAMS if shapes and high else ()

    def excited(f):
        return not f.edge or e0_kev / f.edge >= ID_MIN_OVERVOLTAGE

    def single_unconfirmed(el):
        """Only one family of el can be used, another one exists but is not excited enough to confirm it
        (E0 / E_edge < ID_MIN_OVERVOLTAGE, or the edge above E0: Mo, Ag, Sn L in SEM at 15 kV)."""
        usable = [f for f in element_families(el, erange[0], erange[1], e0_kev) if excited(f)]
        others = [Family(el, fam) for fam in ('K', 'L', 'M') if fam in ELEMENTS[el][3]
                  and ELEMENTS[el][0] >= FAMILY_MIN_Z[fam] and Family(el, fam) not in usable]
        return len(usable) == 1 and any(not excited(f) for f in others)

    def strictness(el, r=None):
        """Threshold factor: an element with one usable family that cannot be confirmed by another one needs
        2x the threshold; in a high-statistics spectrum (model systematics > noise) 3x, and 3x also if an
        excited family of it is not seen at all (Cu L at 0.93 keV without Cu K at 8.05 keV)."""
        if single_unconfirmed(el):
            return 3.0 if high else 2.0
        return 3.0 if high and r is not None and missing_families(el, r) else 1.0

    def on_shoulder(e, counts, r):
        """A candidate peak at e within 2 FWHM of a found line with > 1 / ID_SHOULDER_FRAC times its counts:
        in a high-statistics spectrum it cannot be told from a small shape error of that line."""
        fw = float(det.fwhm(max(e, 0.1)))
        return any(abs(f.main[1] - e) < 2 * fw and counts < ID_SHOULDER_FRAC * c
                   for f, c in zip(r['fams'], r['c']) if f.element in chosen)

    def cols(el):
        if el not in cache:
            # families hardly excited (E0 / E_edge < ID_MIN_OVERVOLTAGE, lines just below E0 where the
            # background is uncertain) stay out of the identification model
            fams = [f for f in element_families(el, erange[0], erange[1], e0_kev) if excited(f)]
            cache[el] = (fams, peak_matrix(E, fams, det, disp) if fams else np.zeros((len(E), 0)))
        return cache[el]

    def fit(els, var, force=None):
        """Linear fit of the families of els. force=(el, family): the other well detected families of el
        (those _families_consistent checks) are held at >= ID_FAMILY_RATIO x the counts of that family."""
        fams, mats = [], []
        for el in els:
            f, a = cols(el)
            fams += f
            mats.append(a)
        A = np.hstack(mats) if mats else np.zeros((len(E), 0))
        Af, linked = A, []
        if force:
            jm = next(j for j, f in enumerate(fams) if (f.element, f.family) == force)
            linked = [j for j, f in enumerate(fams) if f.element == force[0] and j != jm
                      and 1.5 <= f.main[1] <= 20.0 and e0_kev >= 2.5 * f.edge]
            Af = A.copy()
            for j in linked:            # c_j = ratio x c_m + (free excess >= 0)
                Af[:, jm] += ID_FAMILY_RATIO * A[:, j]
        B0, npu = _with_pileup(Bbg, pile['shape'], sel)
        M = np.hstack([Af, B0 * det.threshold(E)[:, None]])    # det is refined on the way
        nf = A.shape[1]
        ub = np.r_[np.full(nf, np.inf), pile['ub'], np.full(Bbg.shape[1], np.inf)] if npu else None
        c, chi2, Mw, active = _solve_tied(M, Y, var, nf + npu, _tie_families(fams), ub=ub)  # sum peaks >= 0
        cov = np.linalg.pinv(Mw.T @ Mw)
        err = np.sqrt(np.maximum(np.diag(cov)[:nf], 1e-12))
        for jm_, jl in active:
            err[jm_] = M_TO_L_MAX * err[jl]
        model = M @ c
        c = c[:nf].copy()
        for j in linked:
            c[j] += ID_FAMILY_RATIO * c[jm]
        return dict(fams=fams, A=A, c=c, chi2=chi2, peaks=A @ c, model=model, err=err,
                    r2=(Y - model) ** 2 / var)

    def variance(els):      # counts x dispersion (1 = Poisson) + systematic part of the line model
        r = fit(els, dispersion * np.maximum(Y, 1.0))
        return dispersion * np.maximum(r['model'], 1.0) + (systematic * r['peaks']) ** 2

    def window(i):
        h = max(int(np.ceil(0.6 * float(det.fwhm(max(E[i], 0.1))) / disp)), 1)
        return slice(max(i - h, 0), i + h + 1)

    def is_sum_peak(e, resid, sums):
        """The residual peak at e is weak enough to be a sum peak of the strongest lines (see _pileup_peaks)."""
        net = resid[window(int(np.argmin(np.abs(E - e))))].sum()
        # wide in energy: incomplete summation of the pulses puts sum peaks also below E_i + E_j
        return any(abs(e - es) < 1.5 * float(det.fwhm(es)) and net <= n_max for es, n_max in sums)

    chosen = [el for el in start if el in ELEMENTS]
    details, rejected = [], []
    piled = []      # peaks taken for sum peaks: only until the next element, the parents' counts change
    n_refined = 0
    for _ in range(4 * max_elements):
        if len(chosen) >= max_elements:
            break
        if refine and chosen and n_refined < 3 and len(chosen) > n_refined:
            fams = [f for el in chosen for f in cols(el)[0]]
            if fams:
                det = fit_spectrum(energy, y, fams, det, bg, n_bg=2, erange=erange, noise=noise,
                                   refine_calib=True, refine_res=True, refine_tail=True, dispersion=dispersion,
                                   pileup=pile['shape'], nlls=nlls, nlls_min_sigma=ID_SHIFT_MIN_SIGMA, refine_skip_lm=ID_REFINE_SKIP_LM,
                                   nlls_bounds=ID_SHAPE_BOUNDS).det
                cache.clear()
            n_refined = len(chosen)
        var = variance(chosen)
        cur = fit(chosen, var)
        if use_pile and chosen:         # sum peaks of the lines found so far, then refit
            update_pileup(cur['model'])
            var = variance(chosen)
            cur = fit(chosen, var)
        resid = Y - cur['model']
        sums = [] if use_pile else _pileup_peaks(cur, 3, live_time)     # (modelled sum peaks: no guessing)
        # (with only O in the model its counts absorb C, Na...: the bound of the O+O sum peak is then far
        # too large - a peak taken for a sum peak stays aside only while the current fit allows it)
        piled = [e for e in piled if is_sum_peak(e, resid, sums)]
        sig = _window_sums(resid, E, det) / np.sqrt(np.maximum(_window_sums(var, E, det), 1.0))
        fw = det.fwhm(np.maximum(E, 0.1))
        sig[(E < E[0] + 0.25 * fw) | (E > E[-1] - fw)] = 0
        sig[E > e0_kev / ID_MIN_OVERVOLTAGE] = 0     # no line excited well enough there (edge > line energy)
        for er in rejected + piled:
            sig[np.abs(E - er) < 0.7 * det.fwhm(er)] = 0
        if sig.max() < threshold:
            break
        ip = int(np.argmax(sig))
        ep = float(E[ip])
        win = window(ip)
        net_peak = resid[win].sum()
        near = np.abs(E - ep) < 2.5 * float(det.fwhm(max(ep, 0.1)))
        # low-energy lines appear up to a few tens of eV below their tabulated energy
        tol = 0.5 * float(det.fwhm(ep)) + 0.015 + (0.03 if ep < LOW_E_LINES else 0.0)
        # at the energy of a possible sum peak an element is taken only if its other lines explain residuals
        # elsewhere too (Au L at 9.71 keV = C K + Pt L-alpha, 11.44 keV ~ Pt M + Pt L; Na K alone at the
        # O + O sum energy stays a sum peak)
        sum_like = is_sum_peak(ep, resid, sums)
        cands = {}
        for de, el, fam, sym, le, w in lines_near(ep, tol, min_weight=0.15, e0_kev=e0_kev):
            if el in chosen or el in excl or Family(el, fam) not in cols(el)[0]:     # (also: not excited)
                continue
            cands.setdefault(el, (fam, sym, le))
        best = None
        base = len(cur['fams'])
        for el, (fam, sym, le) in cands.items():
            r = fit(chosen + [el], var)
            if not _families_consistent(r['fams'][base:], r['c'][base:], r['err'][base:], fam, e0_kev, threshold):
                # another family of el is weaker than expected - absent, or its counts taken by an element
                # found before whose lines overlap (Pt M under Au M, the L lines of both resolved): refit
                # with the expected ratio held; the data must not contradict it (chi2 increase < threshold^2)
                rf = fit(chosen + [el], var, force=(el, fam))
                if rf['chi2'] - r['chi2'] > threshold ** 2 or not _families_consistent(
                        rf['fams'][base:], rf['c'][base:], rf['err'][base:], fam, e0_kev, threshold):
                    continue
                r = rf
            gain = cur['chi2'] - r['chi2']
            # the candidate is ranked by how well it explains this peak (local chi2 gain);
            # the global gain is only used as the acceptance threshold
            local = cur['r2'][near].sum() - r['r2'][near].sum()
            contrib = (r['A'][:, base:] @ r['c'][base:])[win].sum()
            if contrib < 0.4 * net_peak:
                continue
            if high and on_shoulder(ep, float(np.sum(r['c'][base:])), r):
                continue
            prior = 0.5 if el in RARE else 1.0
            if single_unconfirmed(el):
                # one family, the confirming one not excited: a higher threshold, and a candidate whose
                # families can all be checked is preferred (Mg K rather than As L at 1.25-1.28 keV at 15 kV)
                prior *= 0.5
                if gain * prior < (strictness(el) * threshold) ** 2:
                    continue
            if best is None or local * prior > best[0]:
                outside = cur['r2'][~near].sum() - r['r2'][~near].sum()
                best = (local * prior, el, fam, sym, le, gain * prior, outside * prior)
        if sum_like and (best is None or best[5] < threshold ** 2 or best[6] < threshold ** 2):
            piled.append(ep)
            if log:
                log(f'пик {ep:.3f} кэВ: вероятно, пик суммирования (pile-up)')
            continue
        if best is None or best[5] < threshold ** 2:
            rejected.append(ep)
            if log:
                log(f'пик {ep:.3f} кэВ: нет подходящего элемента')
            continue
        score, el, fam, sym, le, gain, _ = best
        chosen.append(el)
        main = Family(el, fam).main
        details.append(dict(element=el, family=fam, energy=ep, line=f'{el} {line_label(main[0])}',
                            line_energy=main[1], dchi2=gain, sigma=float(sig[ip])))
        if log:
            log(f'{ep:.3f} кэВ -> {el} {line_label(sym)} ({le:.3f} кэВ), Δχ²={gain:.0f}')

    # prune elements that are not significant in the final fit (after a last detector refinement)
    if refine and chosen:
        fams = [f for el in chosen for f in cols(el)[0]]
        det = fit_spectrum(energy, y, fams, det, bg, n_bg=2, erange=erange, noise=noise,
                           refine_calib=True, refine_res=True, refine_tail=True, dispersion=dispersion,
                           pileup=pile['shape'], nlls=nlls, nlls_min_sigma=ID_SHIFT_MIN_SIGMA, refine_skip_lm=ID_REFINE_SKIP_LM,
                                   nlls_bounds=ID_SHAPE_BOUNDS).det
        cache.clear()
    # high statistics: a light element whose peak a found family took over during the search (N K at
    # 0.392 keV under Ti Ll at 0.395 keV, once Ti is found its free L intensity absorbs N) is tested once
    # more against the complete model
    if high and chosen:
        for el in ID_LIGHT_CHECK:
            if el in chosen or el in excl or not cols(el)[0]:
                continue
            var = variance(chosen)
            cur, r = fit(chosen, var), fit(chosen + [el], var)
            base = len(cur['fams'])
            sg = max((c / e for c, e in zip(r['c'][base:], r['err'][base:])), default=0.0)
            if cur['chi2'] - r['chi2'] >= threshold ** 2 and sg >= threshold \
                    and not on_shoulder(r['fams'][base].main[1], float(np.sum(r['c'][base:])), r):
                chosen.append(el)
                main = r['fams'][base].main
                details.append(dict(element=el, family=r['fams'][base].family, energy=main[1],
                                    line=f'{el} {line_label(main[0])}', line_energy=main[1],
                                    dchi2=cur['chi2'] - r['chi2'], sigma=sg))
                if log:
                    log(f'{el}: найден при проверке лёгких элементов на полной модели ({sg:.0f}σ)')

    def missing_families(el, r):
        """Families of el in the well detected range (1.5-20 keV) that are not seen (< 2 sigma) in the fit r."""
        fs = [(f, c / e) for f, c, e in zip(r['fams'], r['c'], r['err']) if f.element == el]
        top = max((sg for _, sg in fs), default=0.0)
        return [f.key for f, sg in fs if sg < 2 and 1.5 <= f.main[1] <= 20.0 and sg < top]

    def required(el, r):
        return strictness(el, r) * threshold

    while chosen:
        r = fit(chosen, variance(chosen))
        best_sig = {}
        for f, c, e in zip(r['fams'], r['c'], r['err']):
            best_sig[f.element] = max(best_sig.get(f.element, 0.0), c / e)
        weakest = min(chosen, key=lambda el: best_sig.get(el, 0.0) / required(el, r))
        if best_sig.get(weakest, 0.0) >= required(weakest, r):
            break
        chosen.remove(weakest)
        details = [d for d in details if d['element'] != weakest]
        if log:
            why = ('' if required(weakest, r) == threshold else ' (вторая серия не возбуждается)'
                   if single_unconfirmed(weakest) else f' (нет серии {", ".join(missing_families(weakest, r))})')
            log(f'{weakest} удалён: {best_sig.get(weakest, 0.0):.1f}σ < {required(weakest, r):g}σ{why}')
    # confidence: strong main family and no well-detected family missing -> 'надёжно'
    if chosen:
        r = fit(chosen, variance(chosen))
        per_el = {}
        for f, c, e in zip(r['fams'], r['c'], r['err']):
            per_el.setdefault(f.element, []).append((f, c / e))
        for d in details:
            fs = per_el.get(d['element'], [])
            top = max((sg for _, sg in fs), default=0.0)
            missing = [f.key for f, sg in fs if sg < 2 and 1.5 <= f.main[1] <= 20.0 and sg < top]
            d['sigma'] = top
            d['confidence'] = 'надёжно' if top >= 2 * threshold and not missing else 'возможно'
            d['missing'] = missing
    if shapes:      # calibration (incl. the low-energy line positions) and resolution as refined, shapes as given
        det = replace(det, shifts={**det_in.shifts, **strip_nlls(det).shifts}, widths=dict(det_in.widths),
                      ratios=dict(det_in.ratios))
    return IdResult(elements=chosen, details=details, unidentified=rejected, det=det)


# ----------------------------------------------------------------------------------------------
# Maps
# ----------------------------------------------------------------------------------------------
@dataclass
class MapResult:
    families: list
    maps: np.ndarray        # (n_families, ny, nx) net counts
    method: str
    bg_maps: np.ndarray = None


def map_filter(energy, disp, families, det, bg_shape, n_bg, erange, weight_spectrum, noise=None, tie_low=True):
    """The linear per-pixel least squares of compute_maps: (sel, P, pairs, A, B). The coefficients of a
    pixel spectrum y (families, then the background terms) are P @ y[sel]; then coef[j] = r * coef[k]
    for the tied families j: (k, r) in pairs (in this order). A, B: peak and background design matrices
    on the channels sel. weight_spectrum: the summed spectrum (or its model) - weights and family ratios."""
    sel = np.where((energy >= erange[0]) & (energy <= erange[1]))[0]
    E = energy[sel]
    shape = None if bg_shape is None else np.maximum(np.asarray(bg_shape)[sel], 1e-6 * max(np.max(bg_shape), 1))
    A = peak_matrix(E, families, det, disp)
    B = bg_basis(E, shape, n_bg, noise) * det.threshold(E)[:, None]
    M = np.hstack([A, B])
    nf = len(families)
    # ratios between families from the summed spectrum: M families of Z < 57 limited by their
    # L family (active limits only) and, with tie_low, the low-energy families
    ys = np.asarray(weight_spectrum, float)[sel]
    c, _, _, active = _solve_tied(M, ys, np.maximum(ys, 1.0), nf, _tie_families(families))
    pairs = {jm: (jl, M_TO_L_MAX) for jm, jl in active}
    if tie_low:
        for j, f in enumerate(families):
            mains = [i for i, g in enumerate(families)
                     if g.element == f.element and g.main[1] >= TIE_LOW_E and c[i] > 0]
            if f.main[1] < TIE_LOW_E and mains:
                k = max(mains, key=lambda i: c[i])
                pairs[j] = (k, max(c[j], 0.0) / c[k])
    M = _merge_ratio(M, pairs)
    # weights: 1 / the expected counts of a pixel (the summed spectrum / n_pix, at least 1 / n_pix); the
    # common factor n_pix drops out of the solution
    w = 1.0 / np.maximum(ndimage.gaussian_filter1d(ys, 2.0), 1.0)
    G = M.T @ (M * w[:, None])
    P = np.linalg.solve(G + 1e-12 * np.trace(G) * np.eye(len(G)), (M * w[:, None]).T)   # (k, n_sel)
    return sel, P, pairs, A, B


def compute_maps(si, families, det, bg_shape, n_bg=3, erange=(0.2, 20.0), method='mlls',
                 weight_spectrum=None, noise=None, tie_low=True, chunk=4096, progress=None):
    """Per-pixel maps.

    mlls        : weighted linear least squares with fixed peak shapes -> overlapping lines
                  are separated (deconvolved); background = sum-spectrum background shape x
                  Legendre polynomial (n_bg terms).
    window_net  : counts in +-0.6 FWHM around the main line minus the fitted background and the
                  fitted contributions of the other lines (e.g. the Nb M tail in the C K window).
    window      : gross counts in the same window.
    bg_shape may hold several columns (backgrounds of phases, see phase_backgrounds).

    tie_low: a family with the main line below TIE_LOW_E keV (Nb M, Fe L, Cr L...) follows the
    strongest family of its element above that energy, at the ratio found in the summed spectrum.
    Its per-pixel counts are then not traded against overlapping light-element lines (C K, O K)."""
    energy = si.energy
    if weight_spectrum is None:
        weight_spectrum = si.sum_spectrum()
    sel, P, pairs, A, B = map_filter(energy, si.disp, families, det, bg_shape, n_bg, erange, weight_spectrum,
                                     noise, tie_low)
    E = energy[sel]
    nf = len(families)
    ny, nx = si.shape
    npix = ny * nx

    cube = si.data.reshape(npix, -1)
    i0, i1 = sel[0], sel[-1] + 1
    coef = np.empty((npix, P.shape[0]), dtype=np.float32)
    for s in range(0, npix, chunk):
        d = cube[s:s + chunk, i0:i1].astype(np.float32)
        coef[s:s + chunk] = d @ P.T.astype(np.float32)
        if progress:
            progress(min(1.0, (s + chunk) / npix))
    for j, (k, r) in pairs.items():
        coef[:, j] = r * coef[:, k]
    bgc = coef[:, nf:]

    if method == 'mlls':
        maps = coef[:, :nf].T
    else:
        maps = np.empty((nf, npix), dtype=np.float32)
        for j, f in enumerate(families):
            le = f.main[1] + det.shifts.get(f.key, 0.0)
            et = det.true_energy(E)
            h = 0.6 * float(det.fwhm(le)) * det.widths.get(f.key, 1.0)
            win = (et >= le - h) & (et <= le + h)
            if not win.any():
                maps[j] = 0
                continue
            idx = np.where(win)[0] + i0
            gross = cube[:, idx].sum(axis=1, dtype=np.float64).astype(np.float32)
            if method == 'window':
                maps[j] = gross
            else:
                # minus the fitted background and the fitted tails of the other lines in the window
                others = A[win].sum(axis=0).astype(np.float32)
                others[j] = 0.0
                maps[j] = gross - bgc @ B[win].sum(axis=0).astype(np.float32) - coef[:, :nf] @ others
    return MapResult(families=list(families), maps=maps.reshape(nf, ny, nx).astype(np.float32),
                     method=method, bg_maps=bgc.T.reshape(-1, ny, nx))


MAP_BG_SHARE = 0.5      # share of the contrast between the phases that changes with the background model: map flagged
MAP_PHASE_SIGMA = 3.0   # contrast between the phases (in σ) below which a map has no significant distribution
MAP_STRUCTURE = 5.0     # neighbour covariance of a map (in σ of white noise) above which it has spatial structure


def phase_spectra(si, labels, chunk=4096):
    """Summed spectrum of every phase (n_phases, n_channels) and the numbers of their pixels."""
    flat = si.data.reshape(-1, si.data.shape[-1])
    lab = np.asarray(labels).ravel()
    n = int(lab.max()) + 1
    Y = np.zeros((n, flat.shape[1]))
    for s in range(0, len(lab), chunk):
        part, lp = flat[s:s + chunk], lab[s:s + chunk]
        for c in np.unique(lp):
            Y[c] += part[lp == c].sum(axis=0, dtype=np.float64)
    return Y, np.bincount(lab, minlength=n)


@dataclass
class MapCheck:
    """How well the data fix the distribution of a family map between the phases (map_background_check)."""
    models: list            # names of the background models, the first one is the model of the maps
    values: np.ndarray      # (n_models, n_phases) mean counts per pixel of every phase under every model
    err: np.ndarray         # (n_phases,) statistical errors (1σ) of values[0]
    npix: np.ndarray        # (n_phases,) pixels of every phase
    signif: float           # contrast between the phases (largest - smallest mean of values[0]) in σ
    share: float            # largest share of that contrast changed by another background model (inf: lost / inverted)
    worst: str              # the model that changes it most ('' if none does significantly)


def map_background_check(si, families, det, labels, models, erange, weight_spectrum, noise=None, tie_low=True,
                         spectra=None):
    """Does the distribution of each family map between the phases (labels) depend on the background
    model? The summed spectra of the phases are fitted by the per-pixel least squares of compute_maps
    (map_filter, method 'mlls') with every model of models = [(name, bg_shape, n_bg), ...], the first one
    being the model of the maps. The pattern of the phase means under another model is compared with that
    of the maps up to scale and offset (the display of a map stretches its own range, and tied families
    only change their ratio): the residual of the affine fit beyond 3σ of the difference, relative to the
    contrast between the phases. A model that changes no phase mean beyond 3σ counts as no change.
    Statistical errors: counts x the count dispersion. A weak line under which the background is not fixed
    by the data shows the error of the background model - spread like the main phases - rather than the
    element (share >= MAP_BG_SHARE). spectra: phase_spectra(si, labels) if already computed.
    Returns a MapCheck per family (None: an empty map, or fewer than 2 phases)."""
    Y, npx = spectra if spectra is not None else phase_spectra(si, labels)
    keep = npx > 0
    Y, npx = Y[keep], npx[keep].astype(float)
    if len(npx) < 2:
        return [None] * len(families)
    disp_c = float(getattr(si, 'dispersion', 1.0) or 1.0)
    nf = len(families)
    F = []
    for _, shape, n_bg in models:
        sel, P, pairs, _, _ = map_filter(si.energy, si.disp, families, det, shape, n_bg, erange, weight_spectrum,
                                         noise, tie_low)
        Pf = P[:nf].copy()          # effective filter of every family, the tied ones included
        for j, (k, r) in pairs.items():
            Pf[j] = r * Pf[k]
        F.append(Pf)
    Ys = Y[:, sel]

    def err(p):                     # statistical errors of the phase means given by the filter p
        return np.sqrt(np.maximum(Ys @ (p * p), 0.0) * disp_c) / npx

    names = [m[0] for m in models]
    out = []
    for j in range(nf):
        P0 = F[0][j]
        if not np.any(P0):
            out.append(None)
            continue
        vals = np.stack([Ys @ Pf[j] / npx for Pf in F])
        d0, e0 = vals[0], err(P0)
        hi, lo = int(np.argmax(d0)), int(np.argmin(d0))
        contrast = float(d0[hi] - d0[lo])
        signif = contrast / max(float(np.hypot(e0[hi], e0[lo])), 1e-30)
        X = np.stack([np.ones_like(d0), d0], 1)
        share, worst = 0.0, ''
        for name, Pf, dm in zip(names[1:], F[1:], vals[1:]):
            Pm = Pf[j]
            if np.all(np.abs(dm - d0) <= 3 * err(Pm - P0)):
                continue
            a, b = np.linalg.lstsq(X, dm, rcond=None)[0]
            if b <= 0 or contrast <= 0:
                s = np.inf
            else:
                s = float(np.max(np.maximum(np.abs(dm - a - b * d0) - 3 * err(Pm - b * P0), 0.0))) / (b * contrast)
            if s > share:
                share, worst = s, name
        out.append(MapCheck(models=names, values=vals, err=e0, npix=npx, signif=float(signif), share=share,
                            worst=worst))
    return out


def map_structure(m):
    """Spatial structure of a map: covariance of neighbouring pixels in σ of that of white noise
    (independent pixels, ~N(0, 1)); 0 for a map too small."""
    m = np.asarray(m, dtype=np.float64)
    if m.ndim != 2 or min(m.shape) < 2 or m.var() <= 0:
        return 0.0
    return _neighbour_cov(m) / (m.var() / np.sqrt(2 * m.size))


def smooth_map(m, mode='none', value=1.0):
    if mode == 'gauss' and value > 0:
        return ndimage.gaussian_filter(m, value)
    if mode == 'bin' and int(value) > 1:
        k = int(value)
        ny, nx = m.shape[0] // k * k, m.shape[1] // k * k
        return m[:ny, :nx].reshape(ny // k, k, nx // k, k).sum(axis=(1, 3))
    return m


BLEMISH_WINDOW = 15     # pixels, window of the local noise level of the hot-pixel removal


def _fill_from_neighbours(out, bad):
    """Bad pixels filled in place from the edge of each bad region inwards with the mean of their
    good 3x3 neighbours (as the DM script 'Point Blemish')."""
    good, ones = ~bad, np.ones((3, 3))
    while not good.all():
        s = ndimage.convolve(np.where(good, out, 0.0), ones, mode='constant')
        c = ndimage.convolve(good.astype(np.float64), ones, mode='constant')
        fill = ~good & (c > 0)
        if not fill.any():
            break
        out[fill] = s[fill] / c[fill]
        good |= fill


def remove_blemishes(m, nsigma=5.0, area=False, window=BLEMISH_WINDOW, passes=3):
    """Hot and cold pixels of a map replaced from their neighbours, after the DM script 'Point Blemish'.

    A pixel is bad if its Laplacian (8 x centre - 8 neighbours) differs from the mean Laplacian by more
    than nsigma standard deviations; area=True: also if its value differs from the mean of the map by
    more than nsigma sigma (spots of several pixels, but also small bright particles; first pass only).
    Bad pixels are filled from the edge of each bad region inwards with the mean of their good 3x3
    neighbours. window=None: mean and sigma of the Laplacian over the whole map, one pass (exactly the
    DM script). Default: within window x window pixels, repeated while new bad pixels are found
    (up to `passes`): the noise of X-ray maps grows with the counts (Poisson), with one sigma for the
    whole map the noise of bright phases is taken for blemishes; a local sigma also keeps one very hot
    pixel from hiding weaker ones elsewhere and leaves the neighbours of a hot pixel alone.
    Returns (map, number of replaced pixels)."""
    m = np.asarray(m, dtype=np.float64)
    if m.ndim != 2 or min(m.shape) < 3:
        return m, 0
    kern = -np.ones((3, 3))
    kern[1, 1] = 8.0
    out, replaced = m.copy(), np.zeros(m.shape, bool)
    for k in range(passes if window else 1):
        lap = ndimage.convolve(out, kern, mode='nearest')      # map border: no false edge
        if window:
            w = int(min(window, *m.shape))
            mu = ndimage.uniform_filter(lap, w, mode='nearest')
            sd = np.sqrt(np.maximum(ndimage.uniform_filter(lap * lap, w, mode='nearest') - mu * mu, 0.0))
        else:
            mu, sd = lap.mean(), lap.std()
        bad = np.abs(lap - mu) > nsigma * sd
        if area and k == 0:
            bad |= np.abs(out - out.mean()) > nsigma * out.std()
        if not bad.any() or bad.all():
            break
        replaced |= bad
        _fill_from_neighbours(out, bad)
    n = int(replaced.sum())
    return (out, n) if n else (m, 0)


# ----------------------------------------------------------------------------------------------
# Preprocessing for the PCA: dead channels, energy drift, bad pixels
# ----------------------------------------------------------------------------------------------
PRE_CHANNEL_SIGMA = 8.0     # significance of a dead / hot channel of the summed spectrum
PRE_PIXEL_SIGMA = 5.0       # significance of a bad pixel (total counts against its neighbours)
PRE_DRIFT_BLOCKS = 16       # blocks of rows (acquisition time) of the energy-drift estimate
PRE_DRIFT_MIN = 0.2         # channels: smallest drift that is corrected
PRE_DRIFT_P = 1e-3          # probability of the scatter of the block shifts without a drift, below it: drift
PRE_DRIFT_AGREE = 4.0       # chi2/dof of the peaks around their block shift above which they disagree


@dataclass
class Preprocessing:
    """What preprocess() found and changed."""
    dead_channels: np.ndarray   # channels replaced in every pixel by the interpolation between their neighbours
    empty_channels: int         # channels of the energy range without a single count (left out of the PCA)
    drift: dict                 # energy_drift() result or None (too few rows / no usable peaks / not tested)
    drift_applied: bool
    bad_pixels: np.ndarray      # (ny, nx) bool, pixels replaced by a good neighbour
    pixel_z: np.ndarray         # (ny, nx) deviation of the total counts from the neighbours, in noise units
    changed: bool               # the data were copied and changed


def find_dead_channels(energy, y, det, erange=None, nsigma=PRE_CHANNEL_SIGMA, dispersion=1.0, max_n=50):
    """Dead / hot channels of the summed spectrum y (stuck or missing ADC codes, single-channel spikes):
    channels that differ from the mean of their two neighbours more than an X-ray line can. A line of
    height A and width s channels (detector resolution) differs from that mean by at most A / (2 s^2);
    allowed: A / s^2 (A = highest count within +-2 s, the channel itself excluded) + nsigma x the noise.
    The worst channel is replaced by the mean of its neighbours and the test repeated (a dead channel also
    lifts the deviation of its neighbours). The rising edge at the pulse-processor threshold is not tested.
    Returns the sorted channel indices."""
    y = np.asarray(y, dtype=np.float64).copy()
    n = len(y)
    disp = float(energy[1] - energy[0])
    lo, hi = 1, n - 2
    if erange is not None:
        lo = max(lo, int(np.searchsorted(energy, erange[0])))
        hi = min(hi, int(np.searchsorted(energy, erange[1], side='right')) - 1)
    lo = max(lo, recorded_edge(energy, y) + 3)
    if hi - lo < 2:
        return np.zeros(0, int)
    j = np.arange(lo, hi + 1)
    s = det.sigma(np.maximum(energy[j], 0.1)) / disp
    half = np.maximum(np.ceil(2 * s).astype(int), 1)
    found = []
    for _ in range(max_n):
        A = np.empty(len(j))
        for h in np.unique(half):
            fp = np.ones(2 * h + 1, bool)
            fp[h] = False
            m = half == h
            A[m] = ndimage.maximum_filter(y, footprint=fp, mode='nearest')[j[m]]
        nb = 0.5 * (y[j - 1] + y[j + 1])
        noise = np.sqrt(np.maximum(y[j] + 0.5 * nb, 1.0) * dispersion)
        r = np.abs(y[j] - nb) / (A / s ** 2 + nsigma * noise)
        k = int(np.argmax(r))
        if r[k] <= 1.0:
            break
        found.append(int(j[k]))
        y[j[k]] = nb[k]
    return np.array(sorted(set(found)), int)


def _replace_channels(cube, chans, seed=0):
    """Channels `chans` of every spectrum replaced (in place) by the linear interpolation between the
    nearest good channels; integer data by stochastic rounding (unbiased, the data stay counts)."""
    bad = set(int(c) for c in chans)
    n = cube.shape[-1]
    rng = np.random.default_rng(seed)
    for j in sorted(bad):
        lo, hi = j - 1, j + 1
        while lo in bad:
            lo -= 1
        while hi in bad:
            hi += 1
        lo, hi = max(lo, 0), min(hi, n - 1)
        t = (j - lo) / max(hi - lo, 1)
        v = (1 - t) * cube[..., lo].astype(np.float64) + t * cube[..., hi].astype(np.float64)
        if np.issubdtype(cube.dtype, np.integer):
            v = np.floor(v + rng.random(v.shape))
        cube[..., j] = v.astype(cube.dtype)


def _shift_channels(x, d):
    """Spectra x (..., n) sampled at the channels j + d (a feature at channel j + d moves to j): linear
    interpolation, zero beyond the ends."""
    n = x.shape[-1]
    k = int(np.floor(d))
    f = float(d - k)
    out = np.zeros_like(x)
    for w, sh in ((1.0 - f, k), (f, k + 1)):
        lo, hi = max(0, -sh), min(n, n - sh)
        if w > 0 and hi > lo:
            out[..., lo:hi] += w * x[..., lo + sh:hi + sh]
    return out


def energy_drift(si, det, erange=None, y=None, n_blocks=PRE_DRIFT_BLOCKS, n_peaks=6):
    """Shift of the energy scale during the acquisition. The rows of the scan (acquisition time) are taken
    in n_blocks blocks; around each of the strongest peaks of the summed spectrum y (window +-0.8 FWHM) the
    summed spectrum of a block is fitted as c r(E - d) + a + b E ~ c r - c d r' + a + b E (r = y; weighted
    least squares, own c, a, b per peak: the composition changes from block to block); the shift d of a
    block is the weighted mean over the peaks. A per-pixel shift cannot be measured (~100 counts per pixel).
    Returns dict(rows: block centre rows, shift / err: keV relative to the weighted mean of the blocks (a
    constant shift is a matter of the calibration), peaks: keV, p: probability of the scatter of the shifts
    without a drift (chi^2), agree: chi^2/dof of the peaks around the shift of their block (>> 1: the peaks
    disagree - composition, not drift)), or None (fewer than 6 rows / no usable peak)."""
    ny = si.shape[0]
    nb = min(n_blocks, ny // 2)
    if nb < 3:
        return None
    E, disp = si.energy, si.disp
    D = float(getattr(si, 'dispersion', 1.0) or 1.0)
    y = si.sum_spectrum() if y is None else np.asarray(y, dtype=np.float64)
    bg = snip_background(E, y, det)
    net = ndimage.gaussian_filter1d(y - bg, 1.0)
    ok = E > 0.25 if erange is None else (E >= max(erange[0], 0.25)) & (E <= erange[1])
    pk, _ = find_peaks(np.where(ok, net, 0.0), height=20 * np.sqrt(np.maximum(bg, 1.0) * D))
    windows = []
    for p in sorted(pk, key=lambda p: -net[p]):         # strongest first, windows must not overlap
        hw = max(int(round(0.8 * float(det.fwhm(E[p])) / disp)), 3)
        if p - hw < 1 or p + hw >= len(E) - 1 or any(abs(p - q) <= hw + w for q, w in windows):
            continue
        windows.append((p, hw))
        if len(windows) == n_peaks:
            break
    if not windows:
        return None
    idx = np.concatenate([np.arange(p - hw, p + hw + 1) for p, hw in windows])
    edges = np.linspace(0, ny, nb + 1).astype(int)
    blocks = np.stack([si.data[a:b][..., idx].sum(axis=(0, 1), dtype=np.float64)
                       for a, b in zip(edges[:-1], edges[1:])])
    grad = np.gradient(y)
    d = np.full((nb, len(windows)), np.nan)
    v = np.full((nb, len(windows)), np.inf)
    o = 0
    for k, (p, hw) in enumerate(windows):
        j = np.arange(p - hw, p + hw + 1)
        r = y[j]
        X = np.stack([r, grad[j], np.ones(len(j)), (j - p) / hw], axis=1)
        for b in range(nb):
            z = blocks[b, o:o + len(j)]
            w = 1.0 / np.sqrt(np.maximum(r * z.sum() / max(r.sum(), 1e-12), 1.0) * D)
            Xw = X * w[:, None]
            try:
                cov = np.linalg.inv(Xw.T @ Xw)
            except np.linalg.LinAlgError:
                continue
            c, e = (cov @ (Xw.T @ (z * w)))[:2]
            if c <= 5 * np.sqrt(cov[0, 0]):
                continue
            dd = -e / c
            var = (cov[1, 1] + dd * dd * cov[0, 0] + 2 * dd * cov[0, 1]) / (c * c)
            if abs(dd) < 3 and var > 0:
                d[b, k], v[b, k] = dd, var
        o += len(j)
    wts = np.where(np.isfinite(d), 1.0 / v, 0.0)
    good = wts.sum(axis=1) > 0
    if good.sum() < 3:
        return None
    shift = np.where(good, (wts * np.nan_to_num(d)).sum(axis=1) / np.maximum(wts.sum(axis=1), 1e-300), np.nan)
    err = np.where(good, 1.0 / np.sqrt(np.maximum(wts.sum(axis=1), 1e-300)), np.inf)
    within = np.nansum(wts * (np.nan_to_num(d) - np.nan_to_num(shift)[:, None]) ** 2)
    dof_w = int((wts > 0).sum() - good.sum())
    mean = float(np.sum(shift[good] / err[good] ** 2) / np.sum(1.0 / err[good] ** 2))
    shift = shift - mean
    chi2 = float(np.sum((shift[good] / err[good]) ** 2))
    return dict(rows=0.5 * (edges[:-1] + edges[1:] - 1), shift=shift * disp, err=err * disp,
                peaks=tuple(float(E[p]) for p, _ in windows), p=float(gammaincc((good.sum() - 1) / 2, chi2 / 2)),
                agree=float(within / dof_w) if dof_w > 0 else 1.0)


def find_bad_pixels(total, nsigma=PRE_PIXEL_SIGMA, dispersion=1.0):
    """Hot and dead pixels (e.g. extra dwell at the start of a scan line, a lost pixel): total counts x that
    even the second brightest (second darkest) of the 8 neighbours, n counts, cannot explain. Exact test of
    equal Poisson rates (counts / D): x | x + n ~ Binomial(x + n, 1/2); bad if the one-sided probability is
    below the normal tail of nsigma. Real structure has a neighbour of similar brightness - a line, an edge,
    a particle of 2 x 2 pixels - and is left alone; single pixels and pairs unlike all other neighbours are
    bad. Returns (bad mask, significance in normal sigma: + hot, - dead); maps smaller than 3 x 3: none."""
    t = np.asarray(total, dtype=np.float64)
    if t.ndim != 2 or min(t.shape) < 3:
        return np.zeros(t.shape, bool), np.zeros(t.shape)
    x = np.maximum(t, 0.0) / dispersion
    fp = np.ones((3, 3), bool)
    fp[1, 1] = False
    hi = ndimage.rank_filter(x, 6, footprint=fp, mode='mirror')    # second brightest of the 8
    lo = ndimage.rank_filter(x, 1, footprint=fp, mode='mirror')    # second darkest
    p_hot = betainc(np.maximum(x, 1e-9), hi + 1.0, 0.5)             # P(X >= x)
    p_dead = betainc(np.maximum(lo, 1e-9), x + 1.0, 0.5)            # P(X <= x)
    z = np.where(x > hi, -ndtri(np.clip(p_hot, 1e-300, 0.5)),
                 np.where(x < lo, ndtri(np.clip(p_dead, 1e-300, 0.5)), 0.0))
    return np.abs(z) > nsigma, z


def _replace_pixels(cube, bad, total):
    """Spectra of the bad pixels replaced (in place) by the spectrum of the good 3x3 neighbour with the
    median total counts, from the edge of every bad region inwards. Unlike a mean of the neighbours the
    data stay counts (integers, Poisson noise)."""
    good, total = ~bad, np.array(total, dtype=np.float64)
    ny, nx = bad.shape
    while not good.all():
        todo = []
        for y, x in zip(*np.nonzero(~good)):
            ys, xs = np.mgrid[max(y - 1, 0):min(y + 2, ny), max(x - 1, 0):min(x + 2, nx)]
            ok = good[ys, xs]
            if ok.any():
                cy, cx = ys[ok], xs[ok]
                k = np.argsort(total[cy, cx])[(len(cy) - 1) // 2]
                todo.append((y, x, cy[k], cx[k]))
        if not todo:
            break
        for y, x, sy, sx in todo:
            cube[y, x] = cube[sy, sx]
            total[y, x] = total[sy, sx]
            good[y, x] = True


def preprocess(si, det, erange=None, channels=True, drift=True, pixels=True, pixel_sigma=PRE_PIXEL_SIGMA):
    """Spectrum image prepared for the PCA (raw cube -> preprocessing -> decomposition):
    1. dead / hot channels (find_dead_channels, summed spectrum) - replaced in every pixel by the
       interpolation between the neighbouring good channels;
    2. energy drift along the acquisition (energy_drift, blocks of rows) - corrected row by row if it is
       significant (probability < PRE_DRIFT_P), larger than PRE_DRIFT_MIN channels and the peaks agree;
    3. bad pixels (find_bad_pixels, total counts within erange) - replaced by a good neighbour.
    Channels of the range without a single count are counted (pca_decompose leaves them out).
    Returns (spectrum image, Preprocessing); the data are copied only if something changes (float32 after
    a drift correction, else the original type: counts stay counts)."""
    E = si.energy
    D = float(getattr(si, 'dispersion', 1.0) or 1.0)
    ny, nx, nch = si.data.shape
    i0, i1 = (0, nch) if erange is None else (int(np.searchsorted(E, erange[0])),
                                              int(np.searchsorted(E, erange[1], side='right')))
    data = None         # copy of the data once something changes
    y = si.sum_spectrum()

    dead = find_dead_channels(E, y, det, erange, dispersion=D) if channels else np.zeros(0, int)
    if len(dead):
        data = si.data.copy()
        _replace_channels(data, dead)
        y[dead] = data[..., dead].sum(axis=(0, 1), dtype=np.float64)

    dr = energy_drift(si if data is None else replace(si, data=data), det, erange, y=y) if drift else None
    applied = False
    if dr is not None:
        dch = np.nan_to_num(dr['shift'] / si.disp)
        if dr['p'] < PRE_DRIFT_P and np.abs(dch).max() > PRE_DRIFT_MIN and dr['agree'] < PRE_DRIFT_AGREE:
            data = (si.data if data is None else data).astype(np.float32)
            per_row = np.interp(np.arange(ny), dr['rows'][np.isfinite(dr['shift'])], dch[np.isfinite(dr['shift'])])
            for r in range(ny):
                data[r] = _shift_channels(data[r], per_row[r])
            applied = True
            y = data.sum(axis=(0, 1), dtype=np.float64)

    src = si.data if data is None else data
    total = src[..., i0:i1].sum(axis=-1, dtype=np.float64)
    bad, z = (find_bad_pixels(total, pixel_sigma, D) if pixels
              else (np.zeros((ny, nx), bool), np.zeros((ny, nx))))
    if bad.any() and not bad.all():
        if data is None:
            data = si.data.copy()
        _replace_pixels(data, bad, total)
    else:
        bad = np.zeros((ny, nx), bool)
    rep = Preprocessing(dead_channels=dead, empty_channels=int((y[i0:i1] <= 0).sum()), drift=dr,
                        drift_applied=applied, bad_pixels=bad, pixel_z=z, changed=data is not None)
    return (si if data is None else replace(si, data=data)), rep


# ----------------------------------------------------------------------------------------------
# PCA denoising
# ----------------------------------------------------------------------------------------------
@dataclass
class PCAResult:
    eigvals: np.ndarray     # leading eigenvalues of Xw^T Xw (descending)
    total: float            # sum of all eigenvalues (= |Xw|^2)
    factors: np.ndarray     # (n, n_sel) loadings (spectral components), weighted space, orthonormal
    scores: np.ndarray      # (n, n_pix) pixel scores, weighted space
    sel: np.ndarray         # indices of the channels used
    row_w: np.ndarray       # (n_pix,) pixel weights  (Xw = X / row_w / col_w)
    col_w: np.ndarray       # (n_sel,) channel weights
    noise_edge: float       # largest eigenvalue expected from pure noise (Marchenko-Pastur)
    autocorr: np.ndarray    # (n,) correlation of the score maps between neighbouring pixels
    suggested: int          # number of leading components up to the last one that carries signal
    poisson: bool
    shape: tuple            # (ny, nx)
    signal: np.ndarray = None   # (n,) bool, component classified as signal (noise ones may lie in between)
    binning: int = 1            # b x b pixel blocks of the decomposition (scores: full resolution, projections)

    def spectrum(self, i):
        """Component i as a spectrum on the selected channels (count scale)."""
        return self.factors[i] * self.col_w

    def score_map(self, i):
        return (self.scores[i] * self.row_w).reshape(self.shape)

    def explained(self, k):
        """Fraction of |Xw|^2 described by the first k components."""
        return float(self.eigvals[:k].sum() / self.total) if self.total > 0 else 0.0


PCA_ROW_FLOOR = 0.25    # lowest pixel weight of the Poisson normalisation, x the median counts per pixel
PCA_SPIKE_MAX = 0.5     # largest energy of a single normalised count, x the noise edge (sets the channel floor)
PCA_MIN_SPREAD = 3.0    # fewest channels (participation ratio) of a signal component


def _autocorr(m):
    """Mean correlation of horizontally and vertically neighbouring pixels (0 for white noise)."""
    out = []
    for a, b in ((m[:, :-1], m[:, 1:]), (m[:-1], m[1:])):
        if a.size > 2 and a.std() > 0 and b.std() > 0:
            out.append(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    return float(np.mean(out)) if out else 0.0


def _top_eigh(C, k):
    """The k largest eigenvalues (descending) and eigenvectors of the symmetric matrix C. Lanczos (ARPACK)
    when k is small next to the size of C (the tiles of the local PCA: 24 of ~2000, ~4x faster than the
    full tridiagonal reduction, same values to 1e-15), LAPACK otherwise."""
    n = len(C)
    if k * 30 < n:
        from scipy.sparse.linalg import ArpackNoConvergence, eigsh
        try:
            w, V = eigsh(C, k=k, which='LA', v0=np.ones(n))
            o = np.argsort(w)[::-1]
            return w[o], V[:, o]
        except ArpackNoConvergence:
            pass
    w, V = linalg.eigh(C, subset_by_index=[n - k, n - 1])
    return w[::-1], V[:, ::-1]


def bin_pixels(data, b, sel, max_elements=5_000_000):
    """Spectrum image (ny, nx, n_ch) summed over b x b pixel blocks on the channels sel (pixels of an
    incomplete block at the right / bottom edge left out): float32 (ny // b * nx // b, len(sel))."""
    ny, nx, _ = data.shape
    nyb, nxb, m = ny // b, nx // b, len(sel)
    out = np.empty((nyb, nxb, m), np.float32)
    step = max(1, max_elements // (b * b * nxb * m))     # binned rows per chunk
    for r in range(0, nyb, step):
        r1 = min(r + step, nyb)
        d = data[r * b:r1 * b, :nxb * b][..., sel]
        out[r:r1] = d.reshape(r1 - r, b, nxb, b, m).sum(axis=(1, 3), dtype=np.float32)
    return out.reshape(nyb * nxb, m)


def pca_decompose(si, erange=None, poisson=True, n_keep=48, n_eig=150, chunk=4096, binning=1):
    """PCA (SVD without centring) of the spectrum image on the channels within erange; si is the
    preprocessed spectrum image (preprocess). Result: eigenvalues, loadings (factors, spectra) and
    scores (maps) of the components.

    poisson=True: the data are divided by the square root of the rank-1 estimate of the expected
    counts, sqrt(a_i b_j / N) (Keenan & Kotula 2004), so the Poisson noise becomes ~white with the
    variance si.dispersion (1 for Poisson counts). Pure noise then gives eigenvalues below the
    Marchenko-Pastur edge dispersion x (sqrt(n_pix) + sqrt(n_ch))^2; a component is taken as signal if
    its eigenvalue is above the edge and its score map is spatially correlated, or the eigenvalue is
    far (3x) above the edge, and it is not a spike localised in 1-2 channels / a few pixels. Without
    the normalisation only the spatial correlation of the score maps is used.

    binning b > 1 (as the spatial binning of MSA for DigitalMicrograph, HREM Research): the
    decomposition - eigenvalues, loadings, noise edge, classification of the components - is done on
    the data summed over b x b pixels. The noise edge drops with the number of pixels while the
    eigenvalue of a spatially extended component does not, so weak extended components rise above the
    noise. The scores are then the projections of the
    full-resolution spectra on the loadings: the maps keep the full resolution (a feature smaller
    than a block is detected worse). Eigenvalues, explained fractions and autocorr refer to the
    binned data. Maps with fewer than 2 blocks along a side are not binned."""
    ny, nx, nch = si.data.shape
    npix = ny * nx
    cube = si.data.reshape(npix, nch)
    E = si.energy
    inr = np.ones(nch, bool) if erange is None else (E >= erange[0]) & (E <= erange[1])
    b = np.zeros(nch)
    for s in range(0, npix, chunk):
        b += cube[s:s + chunk].sum(axis=0, dtype=np.float64)
    sel = np.flatnonzero(inr & (b > 0))
    m = len(sel)
    if m < 2 or npix < 2:
        raise ValueError('слишком мало данных для PCA')
    a_full = np.concatenate([cube[s:s + chunk, sel].sum(axis=1, dtype=np.float64) for s in range(0, npix, chunk)])
    binning = int(binning) if binning and int(binning) > 1 and min(ny, nx) // int(binning) >= 2 else 1
    if binning > 1:     # the decomposition data: b x b blocks (channels sel only)
        dshape = (ny // binning, nx // binning)
        data = bin_pixels(si.data, binning, sel)
        a = data.sum(axis=1, dtype=np.float64)
    else:
        dshape, data, a = (ny, nx), None, a_full
    npd = len(a)        # pixels of the decomposition

    def src(s):
        return data[s:s + chunk] if data is not None else cube[s:s + chunk, sel]

    # noise variance after the Poisson normalisation: 1, or the count dispersion (EDAX: ~1.4)
    disp_c = float(getattr(si, 'dispersion', 1.0) or 1.0) if poisson else 1.0
    edge = disp_c * (np.sqrt(npd) + np.sqrt(m)) ** 2
    if poisson:
        # pixel weights floored at PCA_ROW_FLOOR x the median counts: in nearly empty pixels (vacuum,
        # pores; 1-2 counts) a single count in a nearly empty channel would get a huge weight and form
        # its own spurious 'component' above the noise edge. The floor only down-weights such pixels
        # in the decomposition, their reconstruction (projection) does not depend on the pixel weight.
        row_w = np.sqrt(np.maximum(a, max(PCA_ROW_FLOOR * float(np.median(a)), 1.0)))
        # channel weights likewise: an event (1 count, 2 if recorded twice) in a channel with a few counts
        # in the whole map (near E0, beyond the last line) normalised by sqrt(a_i b_j / N) has the energy
        # x^2 N / (a_i b_j); the floor keeps it below PCA_SPIKE_MAX x the noise edge in a pixel at the floor
        n_tot = max(a.sum(), 1.0)
        x_max = 2.0 if disp_c > 1.0 else 1.0
        b_floor = x_max ** 2 * n_tot / (PCA_SPIKE_MAX * edge * float(row_w.min()) ** 2)
        bd = data.sum(axis=0, dtype=np.float64) if data is not None else b[sel]   # channel sums of the same pixels
        col_w = np.sqrt(np.maximum(bd, b_floor) / n_tot)
    else:
        row_w, col_w = np.ones(npd), np.ones(m)

    def block(s):
        return (src(s) / row_w[s:s + chunk, None] / col_w[None, :]).astype(np.float32)

    n_eig = min(n_eig, m, npd)
    n_keep = min(n_keep, n_eig)
    if m <= npd:        # Gram matrix in channel space
        C = np.zeros((m, m))
        for s in range(0, npd, chunk):
            x = block(s)
            C += x.T @ x
        total = float(np.trace(C))
        w, V = _top_eigh(C, n_eig)
        V = V[:, :n_keep]
        scores = np.concatenate([block(s) @ V.astype(np.float32) for s in range(0, npd, chunk)]).T
    else:               # few pixels (line scan): Gram matrix in pixel space
        X = np.concatenate([block(s) for s in range(0, npd, chunk)]).astype(np.float64)
        K = X @ X.T
        total = float(np.trace(K))
        w, U = linalg.eigh(K, subset_by_index=[npd - n_eig, npd - 1])
        w, U = w[::-1], U[:, ::-1][:, :n_keep]
        sv = np.sqrt(np.maximum(w[:n_keep], 1e-30))
        V = (X.T @ U) / sv
        scores = (U * sv).T
    w = np.maximum(w, 0.0)
    factors = V.T.astype(np.float64)
    scores = scores.astype(np.float32)
    for i in range(n_keep):      # sign convention: the strongest channel of a component is positive
        if factors[i][np.argmax(np.abs(factors[i]))] < 0:
            factors[i] *= -1
            scores[i] *= -1
    ac = np.array([_autocorr((scores[i] * row_w).reshape(dshape)) for i in range(n_keep)])
    # channels and pixels a component spreads over (participation ratios): an X-ray line is several
    # channels wide; a component in 1-2 channels and a few pixels is a noise spike (a few counts in a
    # nearly empty channel)
    spread = 1.0 / np.maximum(np.sum(factors ** 4, axis=1), 1e-30)
    s2 = scores * scores
    pix_spread = (s2.sum(axis=1, dtype=np.float64) ** 2
                  / np.maximum((s2 * s2).sum(axis=1, dtype=np.float64), 1e-300))
    local = (spread < PCA_MIN_SPREAD) | (pix_spread < min(20.0, 0.05 * npd))
    k = 0
    if poisson:
        # sparse data (tens of counts per pixel over ~1000 channels) lift pure-noise eigenvalues up to
        # ~2x above the edge (factors = spikes in the near-empty channels, maps = isolated pixels):
        # a component is signal if its map is spatially coherent or it is far above the edge, and it is
        # not a localised spike; k = the last signal component
        sig = (w[:n_keep] > edge) & ((ac > 0.1) | (w[:n_keep] > 3 * edge)) & ~local
        k = int(np.flatnonzero(sig)[-1]) + 1 if sig.any() else 0
    else:
        # the raw Poisson noise is not white, the Marchenko-Pastur edge does not apply: signal components
        # are those with spatially coherent maps, the noise level is the first incoherent one
        sig = ac > 0.2
        while k < n_keep and ac[k] > 0.2:
            k += 1
        edge = float(w[min(k, len(w) - 1)])
    if not sig.any():
        sig[0] = True
    if binning > 1:
        # scores of the full-resolution pixels: projections of their spectra on the loadings. The pixel
        # weights cancel in the score maps and the reconstruction; Poisson ones as without binning
        row_w = (np.sqrt(np.maximum(a_full, max(PCA_ROW_FLOOR * float(np.median(a_full)), 1.0))) if poisson
                 else np.ones(npix))
        F = factors.T.astype(np.float32)
        scores = np.concatenate([(cube[s:s + chunk, sel] / row_w[s:s + chunk, None] / col_w[None, :])
                                 .astype(np.float32) @ F for s in range(0, npix, chunk)]).T
    return PCAResult(eigvals=w, total=total, factors=factors, scores=scores, sel=sel, row_w=row_w,
                     col_w=col_w, noise_edge=float(edge), autocorr=ac, suggested=max(k, 1), poisson=poisson,
                     shape=(ny, nx), signal=sig, binning=binning)


def pca_components(pca, k, skip_noise=True):
    """Indices of the components of a model 'first k components', without the ones classified as
    noise if skip_noise (component 1, the mean spectrum, is always kept)."""
    k = int(np.clip(k, 1, len(pca.factors)))
    sig = pca.signal if pca.signal is not None else np.ones(len(pca.factors), bool)
    return tuple(i for i in range(k) if i == 0 or not skip_noise or sig[i])


def phase_labels(pca, n_phases=None, smooth=2.0, min_frac=0.01):
    """Pixels grouped into phases (int map ny x nx): k-means on the spatially smoothed ratios of
    the PCA scores s_i / s_1 (i < k, k = n_phases or the number of signal components, >= 2).
    Clusters smaller than min_frac of the pixels are merged into the nearest one."""
    from scipy.cluster.vq import kmeans2
    n_sig = int(pca.signal.sum()) if pca.signal is not None else pca.suggested
    k = int(np.clip(n_phases or n_sig, 2, min(6, len(pca.factors))))
    ny, nx = pca.shape
    idx = pca_components(pca, len(pca.factors))[:k]        # signal components first, noise ones skipped
    idx = idx + tuple(i for i in range(len(pca.factors)) if i not in idx)[:k - len(idx)]
    S = np.stack([ndimage.gaussian_filter(pca.score_map(i), smooth) for i in idx], -1).reshape(-1, k)
    X = S[:, 1:] / np.where(np.abs(S[:, :1]) > 0, S[:, :1], 1.0)
    X = (X - X.mean(0)) / np.maximum(X.std(0), 1e-12)
    cent, lab = kmeans2(X, k, seed=1, minit='++')
    while True:
        sizes = np.bincount(lab, minlength=len(cent))
        live = np.flatnonzero(sizes > 0)
        small = [c for c in live if sizes[c] < min_frac * len(lab)]
        if not small or len(live) <= 1:
            break
        c = small[0]
        others = [o for o in live if o != c]
        lab[lab == c] = others[int(np.argmin(np.linalg.norm(cent[others] - cent[c], axis=1)))]
    _, lab = np.unique(lab, return_inverse=True)
    return lab.reshape(ny, nx)


def phase_backgrounds(si, labels, det, width=2.0, estimator=None, spectra=None):
    """Background of the summed spectrum of every phase: (n_channels, n_phases), SNIP or
    estimator(energy, y). Used as background shapes of the per-pixel fit, the background of a
    pixel is their mixture. spectra: phase_spectra(si, labels) if already computed."""
    est = estimator or (lambda e, y: snip_background(e, y, det, width))
    Y = (spectra if spectra is not None else phase_spectra(si, labels))[0]
    return np.stack([est(si.energy, y) for y in Y], axis=-1)


PCA_MAP_LOST = 0.3      # share of the structure of a map lost / added by PCA above which its PCA map is flagged


def _neighbour_cov(m):
    m = m - m.mean()
    return float(np.mean([np.mean(m[:, :-1] * m[:, 1:]), np.mean(m[:-1] * m[1:])]))


def pca_map_check(raw, pca):
    """Share of the spatial structure of an element map that the PCA reconstruction misses or adds:
    covariance of neighbouring pixels of (raw - pca) / that of the map of the loaded data. The pixel
    noise is independent between pixels and drops out of the neighbour covariance: ~0 - the PCA map
    holds the structure of the raw map; ~1 - none of it (a weak element whose own distribution does not
    form a component: its PCA map is a combination of the main phases). If the raw map has no
    significant structure, relative to its detection limit (>= 1: the PCA map shows structure the data
    do not support). None if neither map has significant structure (or the map is too small)."""
    raw, pca = np.asarray(raw, dtype=np.float64), np.asarray(pca, dtype=np.float64)
    if raw.ndim != 2 or min(raw.shape) < 2 or raw.shape != pca.shape:
        return None
    limit = 5 * raw.var() / np.sqrt(2 * raw.size)       # 5 sigma of the neighbour covariance of white noise
    c_raw = _neighbour_cov(raw)
    if c_raw <= limit and _neighbour_cov(pca) <= limit:
        return None
    return max(_neighbour_cov(raw - pca) / max(c_raw, limit), 0.0)


def pca_reconstruct(si, pca, comps, chunk=4096):
    """Spectrum image rebuilt from the components `comps`: an int k (the first k) or a sequence of
    component indices (float32; may contain small negative values). Channels outside the
    decomposition range are copied unchanged."""
    ny, nx, nch = si.data.shape
    npix = ny * nx
    out = si.data.astype(np.float32)
    flat = out.reshape(npix, nch)
    if np.isscalar(comps):
        idx = list(range(int(np.clip(comps, 1, len(pca.factors)))))
    else:
        idx = [i for i in comps if 0 <= i < len(pca.factors)] or [0]
    F = (pca.factors[idx] * pca.col_w).astype(np.float32)
    S = pca.scores[idx] * pca.row_w.astype(np.float32)
    for s in range(0, npix, chunk):
        part = flat[s:s + chunk]
        part[:, pca.sel] = S[:, s:s + chunk].T @ F
    return out


LOCAL_PCA_TILE = 64     # px, default side of the tiles of the spatially local PCA
LOCAL_PCA_KEEP = 24     # components kept per tile
LOCAL_PCA_NEW = 0.5     # share of a tile component outside the global model above which it is added to the tile


@dataclass
class LocalPCA:
    """Spatially local PCA (local_pca_decompose): the PCA of every tile. The signal components of a tile are
    its first `suggested`, without the noise ones if skip_noise (model)."""
    tile: int
    shape: tuple            # (ny, nx) of the map
    tiles: list             # (y0, x0, h, w, PCAResult) per tile
    skip_noise: bool = True

    def model(self, p):
        return pca_components(p, p.suggested, self.skip_noise)

    def n_comps(self):
        return np.array([len(self.model(t[4])) for t in self.tiles])

    def tile_map(self, values):
        """Per-tile values as a map, blended over the overlapping tiles as the reconstruction."""
        acc, wsum = np.zeros(self.shape), np.zeros(self.shape)
        for (y0, x0, h, w, _), v in zip(self.tiles, values):
            wt = _tile_weights(h, w)
            acc[y0:y0 + h, x0:x0 + w] += wt * v
            wsum[y0:y0 + h, x0:x0 + w] += wt
        return acc / np.maximum(wsum, 1e-12)


def _tile_starts(n, t):
    """Starts of tiles of size t covering 0 ... n with a half-tile step; the last one ends at n."""
    if n <= t:
        return [0]
    s = list(range(0, n - t + 1, max(t // 2, 1)))
    if s[-1] != n - t:
        s.append(n - t)
    return s


def _tile_weights(h, w):
    """Blending weights of a tile: sin^2 across each side. Tiles shifted by half a side add up to a
    constant weight, so the reconstruction changes smoothly from tile to tile (no seams); the weights are
    > 0 everywhere, the edges of the map are covered by one tile."""
    wy = np.sin(np.pi * (np.arange(h) + 0.5) / h) ** 2
    wx = np.sin(np.pi * (np.arange(w) + 0.5) / w) ** 2
    return wy[:, None] * wx[None, :]


def local_pca_decompose(si, erange=None, tile=LOCAL_PCA_TILE, poisson=True, binning=1, skip_noise=True,
                        n_keep=LOCAL_PCA_KEEP, progress=None):
    """Spatially local PCA (as the Spatial Local wPCA of MSA for DigitalMicrograph, HREM Research): the map
    is cut into tiles of tile x tile pixels overlapping by half a tile, every tile is decomposed by
    pca_decompose on its own (its signal components: noise edge and spatial coherence of the tile). A
    minor phase confined to a part of the map (segregation at an interface, a few particles) carries a
    larger share of the variance of its tile than of the whole map and can form a component there, while
    in the global PCA it stays below the noise edge and its spectrum is rebuilt from the main phases.
    The tiles complement the global model (local_pca_reconstruct) rather than replace it: a weak phase
    spread over the whole map is found by the global PCA but may stay below the noise edge of every tile
    (on 01593top a pure tile-by-tile model kept 32-81 % of the three weaker global components).
    progress(fraction) is called after every tile."""
    ny, nx = si.data.shape[:2]
    ys, xs = _tile_starts(ny, tile), _tile_starts(nx, tile)
    tiles = []
    n = len(ys) * len(xs)
    for y0 in ys:
        for x0 in xs:
            h, w = min(tile, ny), min(tile, nx)
            p = pca_decompose(replace(si, data=si.data[y0:y0 + h, x0:x0 + w]), erange, poisson=poisson,
                              n_keep=n_keep, n_eig=n_keep, binning=binning)
            tiles.append((y0, x0, h, w, p))
            if progress:
                progress(len(tiles) / n)
    return LocalPCA(tile=tile, shape=(ny, nx), tiles=tiles, skip_noise=skip_noise)


def _local_basis(pca, comps, p, model):
    """Model of a tile: orthonormal basis (columns; channels pca.sel, weighted space of the global PCA) of
    the global components comps plus the directions of the tile's signal components (p.factors[model]) that
    lie outside their span: the singular vectors of the part outside the span with singular value
    > LOCAL_PCA_NEW. A tile component that only remixes the main phases lies (almost) inside the global
    span and adds nothing. Returns (basis, number of added directions)."""
    G = pca.factors[list(comps)].T
    keep = np.isin(p.sel, pca.sel)
    pos = np.searchsorted(pca.sel, p.sel[keep])
    T = np.zeros((len(pca.sel), len(model)))
    T[pos] = (p.factors[list(model)][:, keep] * p.col_w[keep]).T / pca.col_w[pos, None]   # counts -> global weights
    T /= np.maximum(np.linalg.norm(T, axis=0), 1e-30)
    U, s, _ = np.linalg.svd(T - G @ (G.T @ T), full_matrices=False)
    new = U[:, s > LOCAL_PCA_NEW]
    return np.hstack([G, new]), new.shape[1]


def local_pca_added(lp, pca, comps):
    """Number of local directions added to the global model comps in every tile (_local_basis)."""
    return np.array([_local_basis(pca, comps, t[4], lp.model(t[4]))[1] for t in lp.tiles])


def local_pca_reconstruct(si, lp, pca, comps):
    """Spectrum image rebuilt by the global PCA pca (components comps) complemented tile by tile by the
    local PCA lp: the spectra of a tile are projected on its model (_local_basis), the overlapping tiles
    blended with _tile_weights (float32). Without local directions this is exactly pca_reconstruct(si,
    pca, comps). Channels outside the decomposition range are unchanged."""
    nch = si.data.shape[-1]
    out = np.zeros(si.data.shape, np.float32)
    wsum = np.zeros(lp.shape, np.float32)
    cw = pca.col_w.astype(np.float32)
    for y0, x0, h, w, p in lp.tiles:
        B = _local_basis(pca, comps, p, lp.model(p))[0].astype(np.float32)
        rec = si.data[y0:y0 + h, x0:x0 + w].reshape(-1, nch).astype(np.float32)
        rec[:, pca.sel] = ((rec[:, pca.sel] / cw) @ B) @ B.T * cw
        wt = _tile_weights(h, w).astype(np.float32)
        out[y0:y0 + h, x0:x0 + w] += wt[..., None] * rec.reshape(h, w, nch)
        wsum[y0:y0 + h, x0:x0 + w] += wt
    out /= wsum[..., None]
    return out


# ----------------------------------------------------------------------------------------------
# Richardson-Lucy spectral deconvolution
# ----------------------------------------------------------------------------------------------
def response_matrix(energy, det):
    """Sparse detector response R (n x n): column j = detector response to a line at E_j."""
    n = len(energy)
    disp = energy[1] - energy[0]
    et = det.true_energy(energy)
    sig = det.sigma(np.maximum(et, 0.05))
    kmax = int(np.ceil(5 * sig.max() / disp)) + 1
    rows, cols, vals = [], [], []
    j = np.arange(n)
    for k in range(-kmax, kmax + 1):
        i = j + k
        ok = (i >= 0) & (i < n)
        jj, ii = j[ok], i[ok]
        s = sig[jj]
        v = ndtr((et[ii] + disp / 2 - et[jj]) / s) - ndtr((et[ii] - disp / 2 - et[jj]) / s)
        rows.append(ii)
        cols.append(jj)
        vals.append(v)
    R = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    return R


def richardson_lucy(energy, y, det, background=None, n_iter=30):
    """Resolution enhancement of a spectrum (background removed first, result >= 0)."""
    d = np.asarray(y, float) - (0 if background is None else background)
    d = np.maximum(d, 0)
    R = response_matrix(energy, det)
    RT = R.T.tocsr()
    norm = np.maximum(RT @ np.ones(len(d)), 1e-12)
    x = np.maximum(d, 1e-6).copy()
    for _ in range(n_iter):
        est = np.maximum(R @ x, 1e-12)
        x *= (RT @ (d / est)) / norm
    return x
