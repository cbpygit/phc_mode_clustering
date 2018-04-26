# -*- coding: utf-8 -*-
import os
import hashlib
import logging
import requests
import sys
import time
from urllib import urlretrieve

import click

# Download path of the raw data
DOI = 'https://dx.doi.org/10.5442/ND000002'

# Full and reduced checksums for file verification
CHECKSUMS_FULL = {
    'parameters_and_results.h5':
        '9d5bd9642128641dfe63644b514219261bf4570ffccbbec3c68ec517a0040b63',
    'field_data_E_TE.h5':
        'fd3664d82d86e7612837730460372e74432a2d043bf7b3bd105f6a6abd122dfa',
    'field_data_H_TE.h5':
        '4598f20a3356635cd926f2b0f4cb6e6dbef6d63fd672bed8cf1d76ef252ededd',
    'field_data_E_TM.h5':
        '6604419a9b90e4945e9b2b399a72c93703d9496b36282e1e746934d0120ab659',
    'field_data_H_TM.h5':
        'b23d2d43fc4432013b1ade848dc67bbfec8291895803d1b5a7397b27e3e9e830'}

CHECKSUMS_REDUCED = {
    'parameters_and_results.h5':
        'f678e9efbe8dccef11afac2e928e1c9b2f07ab85c45539ce4fffbcbb8c1497c4',
    'field_data_E_TE.h5':
        '4c567e2e911716cb3f353296e571c08e449a3d65bcbaa4e9ac174c6f9aa3fa05',
    'field_data_H_TE.h5':
        '0cb3e0330b429095ea944cc0b3cef21069c9014d6c30cf65ce9898864d9370ad',
    'field_data_E_TM.h5':
        '4abb219d29cd6878d5cc5a1ac9f0349c16e075954a4bf1f53ebe90429b534102',
    'field_data_H_TM.h5':
        '0ba02ab71d5db1d8d253ba45aa30d1e61ed405782f2aec52749d52f00b8248c9'}

# Additional constants
DATA_DOWNLOAD_FILENAMES = CHECKSUMS_FULL.keys()
MAX_BLOCKS = 10000

# Path to the root folder
project_dir = os.path.normpath(os.path.join(os.path.dirname(__file__),
                                            os.pardir,
                                            os.pardir))


def sha256_checksum(filename, block_size=65536, max_blocks=None):
    """Returns the SHA256 checksum for the file with `filename`."""
    sha256 = hashlib.sha256()
    i_block = 0
    with open(filename, 'rb') as f:
        for block in iter(lambda: f.read(block_size), b''):
            sha256.update(block)
            if max_blocks is not None:
                i_block += 1
                if i_block >= max_blocks:
                    break
    
    # Include file size into the reduced hash
    if max_blocks is not None:
        sha256.update(str(os.path.getsize(filename)))
    return sha256.hexdigest()


def verify_file(filename, known_checksum, max_blocks=None):
    """Compares the SHA256 checksum for the file with `filename` to
    `known_checksum`.
    
    """
    return sha256_checksum(filename, max_blocks=max_blocks) == known_checksum


def resolve_doi(doi):
    """Returns the URL to which a DOI redirects."""
    response = requests.get(doi)
    return response.history[-1].headers['Location']


def _urlretrieve_reporthook(count, block_size, total_size):
    """Callback function for `urlretrieve` that print progress information.
    If `tqdm` is installed, a `tqdm` progress bar will be used with automatic
    remaining time calculation. Otherwise reduced info is printed to
    stdout.

    Based on a snippet of Shichao An's Blog:
        https://blog.shichao.io/

    """
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    progress_size_MB = progress_size / 1048576
    speed = int(progress_size / (1024 * duration))
    percent = min(int(count*block_size*100 / total_size), 100)
    if _TQDM_INSTANCE is None:
        sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                         (percent, progress_size_MB, speed, duration))
        sys.stdout.flush()
    else:
        _TQDM_INSTANCE.n = percent
        od = {'size': '{0:d}MB'.format(progress_size_MB),
              'speed': '{} KB/s'.format(speed)}
        _TQDM_INSTANCE.set_postfix(od)


def _get_tqdm_if_available(**kwargs):
    """Returns a `tqdm` progress bar instance if this module is installed,
    otherwise returns `None`. `kwargs` are passed to the `tqdm.__init__`
    method."""
    try:
        from tqdm import tqdm
        return tqdm(**kwargs)
    except ImportError:
        return None


def download_file_to(url, filepath):
    """Downloads a file from `url` to `filepath`, displaying progress and
    duration using the `_urlretrieve_reporthook` function."""
    global _TQDM_INSTANCE
    tqdm_desc = 'Progress for {}'.format(os.path.basename(filepath))
    _TQDM_INSTANCE = _get_tqdm_if_available(total=100,
                                            desc=tqdm_desc)
    urlretrieve(url, filepath, _urlretrieve_reporthook)
    if not _TQDM_INSTANCE is None:
        _TQDM_INSTANCE.close()
    del _TQDM_INSTANCE


def download_data():
    """Downloads the data from the hard-coded sources."""
    logger = logging.getLogger(__name__)
    url_base = resolve_doi(DOI)
    success = True
    for fn in DATA_DOWNLOAD_FILENAMES:
        url = url_base + '/' + fn
        target = os.path.join(project_dir, 'data', 'raw', fn)
        if os.path.isfile(target):
            logger.info('\tskipping download of {} as the target file ({}) '.
                        format(url, target) + ' already exists')
        else:
            logger.info('\tpreparing download for file {}'.format(url))
            if int(requests.get(url).status_code) == 404:
                logger.warn('\tunable to download: response was error 404!')
                success = False
            else:
                logger.info('\tdownloading to {}'.format(target))
                download_file_to(url, target)
    return success


@click.command()
@click.option('--full_checksum', default=False, is_flag=True,
              help='Use full SHA256 checksums instead of reduced ones.')
@click.option('--print_checksums', default=False, is_flag=True,
              help='Print-out the actual checksums instead of checking.')
def main(full_checksum, print_checksums):
    """Downloads and verifies the raw data needed for the processing scripts
    and saves it to ../raw.
    
    """
    logger = logging.getLogger(__name__)
    logger.info('Downloading the raw data...')
    success = download_data()
    if success:
        logger.info('Download successful.')
    else:
        logger.critical('Terminating as some downloads failed. See previous ' +
                        'warnings for details.')

    logger.info('Verifying the downloaded files...')
    max_blocks = None if full_checksum else MAX_BLOCKS
    refs = CHECKSUMS_FULL if full_checksum else CHECKSUMS_REDUCED
    for f in DATA_DOWNLOAD_FILENAMES:
        fn = os.path.join(project_dir, 'data', 'raw', f)
        if print_checksums:
            type_ =['Reduced', 'Full'][int(full_checksum)]
            cs = sha256_checksum(fn, max_blocks=max_blocks)
            logger.info('{} checksum for "{}": {}'.format(type_, f, cs))
        else:
            match = verify_file(fn, refs[f], max_blocks)
            if not match:
                raise ValueError('Checksum mismatch for file "{}". Stopping.'.
                                 format(fn))
    if not print_checksums:
        logger.info('Verification successful.')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
