# -*- coding: utf-8 -*-
import os
import hashlib
import logging
import requests
from urllib import urlretrieve

import click
from dotenv import find_dotenv, load_dotenv

# Download path of the raw data
DOI = 'https://dx.doi.org/10.5442/ND000001'
DATA_DOWNLOAD_FILENAMES = ['parameters_and_results.h5', 'field_data.h5']

CHECKSUMS_FULL = {
    'parameters_and_results.h5':
        '9d5bd9642128641dfe63644b514219261bf4570ffccbbec3c68ec517a0040b63',
    'field_data.h5':
        '7eb68ea68792ac69fd8f428049fb5398f37e6116f6b97a5c037d570afd1af623'}
CHECKSUMS_REDUCED = {
    'parameters_and_results.h5':
        'f678e9efbe8dccef11afac2e928e1c9b2f07ab85c45539ce4fffbcbb8c1497c4',
    'field_data.h5':
        '32884d902dce79b93b6e8eb5a3d71149c056db5ec20036c386f3016feae8dc61'}
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
                urlretrieve(url, target)
    return success


@click.command()
@click.option('--full_checksum', default=True, type=click.BOOL,
              help='Use full SHA256 checksums instead of reduced ones.')
@click.option('--print_checksums', default=True, type=click.BOOL,
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
                raise ValueError('Checksum mismatch for file {}! Stopping.'.
                                 format(fn))
    logger.info('Verification successful.')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
