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
        'af6d90a9a93cb470742d30d85b19dc4f7fc41df3e15241d6b9dd31b1f53654b8',
    'field_data.h5':
        '7eb68ea68792ac69fd8f428049fb5398f37e6116f6b97a5c037d570afd1af623'}
CHECKSUMS_REDUCED = {
    'parameters_and_results.h5':
        '5d68122d67bb2cddf7deab66ff5e8a4a25ceda9412396cb5fe3e552e956c5e63',
    'field_data.h5':
        'e66f04c911d9a7f279faee7680748bd4db0fa319222dbebb1f120485a93d8286'}
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
            if not max_blocks is None:
                i_block += 1
                if i_block >= max_blocks:
                    break
    
    # Include file size into the reduced hash
    if not max_blocks is None:
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
@click.option('--full_checksum', default=False, type=click.BOOL,
              help='Use full SHA256 checksums instead of reduced ones.')
def main(full_checksum):
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
