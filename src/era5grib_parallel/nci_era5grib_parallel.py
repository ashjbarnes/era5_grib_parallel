# Copyright 2024 ACCESS-NRI (https://www.access-nri.org.au/)
# See the top-level COPYRIGHT.txt file for details.
#
# SPDX-License-Identifier: Apache-2.0
#
# Created by: Chermelle Engel <Chermelle.Engel@anu.edu.au>


"""
Use CDO commands to set up era5 grib files for use by the nesting suite

The nesting suite expects grib files to be named like
    AINITIAL="$ROSE_DATA/era5grib/ec_grib_${FDATE}.t+000"
where FDATE is in YYYYmmddHHMM format

"""

from pathlib import Path
import argparse
from multiprocessing import Pool, TimeoutError
import os
from datetime import timedelta
import pandas

from era5grib_parallel import cdo_era5grib

def create_grib(START,outdir,region=None):

    """
    Function that creates one single GRIB file per date-time from the ERA5 archive

    Parameters
    ----------
    START : string
            The requested date to be repackaged in %Y-%m-%dT%H:%M:%S format 
    outdir : Path
            The path for the output file to be written to
    region : tuple of float, optional
            (lon1, lon2, lat1, lat2) bounding box to subset with CDO before
            conversion. Default is None (no subsetting).

    Returns
    -------
    int
        The process id
    """

    # Create the grib file from the netcdf archive
    cdo_era5grib.repackage_grib(START, outdir, region=region)

    return os.getpid()


def parse_region(value):
    """
    Parse a '--region' command-line argument of the form 'lon1,lon2,lat1,lat2'
    into a tuple of four floats for cdo's sellonlatbox.

    Parameters
    ----------
    value : string
            Comma-separated 'lon1,lon2,lat1,lat2'

    Returns
    -------
    tuple of float
        (lon1, lon2, lat1, lat2)
    """

    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "--region expects four comma-separated values: lon1,lon2,lat1,lat2"
        )
    try:
        return tuple(float(p) for p in parts)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "--region values must be numeric: lon1,lon2,lat1,lat2"
        )


def main():
    """
    The main function that creates a worker pool and generates single GRIB files 
    for requested date/times in parallel.

    Parameters
    ----------
    None.  The arguments are given via the command-line

    Returns
    -------
    None.  The GRIB file are written to the output directory
    """


    # Parse the command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--start', required=True, type=pandas.to_datetime)
    parser.add_argument('--count', default=1, type=int)
    parser.add_argument('--freq', default=60*60, type=lambda x: int(x))
    parser.add_argument('--region', default=None, type=parse_region,
                         help="Optional lon1,lon2,lat1,lat2 bounding box to subset "
                              "with CDO's sellonlatbox before conversion. "
                              "Default: no subsetting (global domain).")
    parser.add_argument('--tasks', default=4, type=int,
                         help="Number of worker processes to run in parallel. "
                              "Default: 4.")
    args = parser.parse_args() 

    # Create a list of the requested date/times
    all_dates = []
    sd = args.start
    start_date = sd.strftime("%Y%m%d%H%M")
    for i in range(args.count):
        cd = sd + timedelta(seconds=i*args.freq)
        cd_string = cd.strftime("%Y-%m-%dT%H:%M:%S")
        all_dates.append(cd_string) 
    print(all_dates)

    # Farm out the date/times to args.tasks worker processes at a time until done.
    with Pool(processes=args.tasks) as pool: 

        # Select args.tasks dates to work on, then create the GRIB files in parallel
        for offset in range(0, len(all_dates)+1, args.tasks): 

            subset_dates = []

            for i in range(offset, offset + args.tasks):
                try:
                    subset_dates.append(all_dates[i])
                except:
                    pass

            # launching multiple evaluations asynchronously *may* use more processes
            multiple_results = [pool.apply_async(create_grib, (dt,args.output,args.region,)) for dt in subset_dates]
            for res in multiple_results:
                try:
                    res.get(timeout=600)
                except TimeoutError as e:
                    print(e)

        print("For the moment, the pool remains available for more work")

    # exiting the 'with'-block has stopped the pool
    print("Now the pool is closed an no longer available")

if __name__ == '__main__':
    main()

