# Copyright 2024 ACCESS-NRI (https://www.access-nri.org.au/)
# See the top-level COPYRIGHT.txt file for details.
#
# SPDX-License-Identifier: Apache-2.0
#
# Created by: Chermelle Engel <Chermelle.Engel@anu.edu.au>
# Modified in 2026 by: Ashley Barnes <ashley.barnes@monash.edu>

from pathlib import Path
from dask.distributed import Client
from datetime import datetime
from datetime import timedelta
import xarray as xr
import dask
import subprocess
import argparse
import pandas


def replacement_era5grib(starttime, endtime,outdir,region=(None,None,None,None),tasks = 52):
    client = Client(n_workers = tasks)

    era5_vars = {
        "skt" : {"stype" : "single", "code" : 235},
        "sp" : {"stype" : "single", "code" : 134},
        "ci" : {"stype" : "single", "code" : 31},
        "sst" : {"stype" : "single", "code" : 34},
        "sd" : {"stype" : "single", "code" : 141},
        "stl1" : {"stype" : "single", "code" : 139},
        "stl2" : {"stype" : "single", "code" : 170},
        "stl3" : {"stype" : "single", "code" : 183},
        "stl4" : {"stype" : "single", "code" : 236},
        "swvl1" : {"stype" : "single", "code" : 39},
        "swvl2" : {"stype" : "single", "code" : 40},
        "swvl3" : {"stype" : "single", "code" : 41},
        "swvl4" : {"stype" : "single", "code" : 42},
        "u" : {"stype" : "pressure", "code" : 131},
        "v" : {"stype" : "pressure", "code" : 132},
        "t" : {"stype" : "pressure", "code" : 130},
        "q" : {"stype" : "pressure", "code" : 133},
        "lsm" : {"stype" : "single", "code" : 172},
        "z" : {"stype" : "single", "code" : 129}
        }


    ERADIR = "/g/data/rt52/era5/"


    era5_months = [starttime.month]
    era5_years = [starttime.year]

    # If the time period straddles more than one month, add this month to read
    if (
        (starttime.month + 1 == endtime.month and starttime.year == endtime.year) or
        (starttime.month == endtime.month + 11 and starttime.year + 1 == endtime.year) # Account for December -> Jan
    ):
        era5_months.append(endtime.month)
        era5_years.append(endtime.year)

        # Raise an error otherwise. Making boundary conditions with more than a months worth of data would 
        # result in incredibly huge files anyway - not something we intend to support.
    elif starttime.month == endtime.month and starttime.year == endtime.year:
        pass

    else:
        raise ValueError(f"Starttime of {starttime} can only be a maximum of 1 month ahead of endtime {endtime}.")

    # Get a list of all files we'll read across every month and variable

    era5_files = []
    for month,year in  zip(era5_months,era5_years):
        for var in era5_vars:
            stype, eccode = era5_vars[var]["stype"], era5_vars[var]["code"]
            filedir = ERADIR + f"/{stype}-levels/reanalysis/{var}/{year}/"
            yyyymm = "%4.4d%2.2d"%(year, month)
            era5_files.append(next(Path(filedir).glob(f"*_{yyyymm}*")))



    # Load all files and subset in memory to just your time and region
    alldata = xr.open_mfdataset(era5_files).sel(longitude = slice(region[0],region[1]),latitude = slice(region[3],region[2]), time = slice(starttime,endtime)).rename({"siconc":"ci"}).load()

    # Do some metadata stuff here
    for var in era5_vars:
        alldata[var].attrs["code"] = era5_vars[var]["code"]
        alldata[var].attrs["table"] = 128

    # 

    ## Save all of the individual raw netcdf files before grib conversion. 

    # Function to call in parallel with dask
    def write_single_file(ds,outdir):
        time_string = datetime.strftime(
        ds.time[0].values.astype('datetime64[s]').astype(object), # Converts to a datetime object so we can format string
        '%Y%m%d%H%M.t+000'
        )
        ds.to_netcdf(Path(outdir) / f"raw_{time_string}.nc")
        return time_string


    time_strings = []

    write_nc_tasks = []
    for i in range(len(alldata.time)):
        write_nc_tasks.append(
            write_single_file(alldata.isel(time = [i]),outdir)
        )

    time_strings = dask.compute(write_nc_tasks)[0]

    ## Perform all the grib conversion in parallel on each file
    cdo_command_tasks = []
    for t in time_strings:
        cdo_command_tasks.append(f"cdo -L --eccodes -f grb1 copy {str(outdir)}/raw_{t}.nc {str(outdir)}/ec_grib_{t}")

    # Function to call in parallel with dask
    def run_cdo(command):
        subprocess.run(
            command,
            shell=True
        )

    dask.compute(
        [dask.delayed(run_cdo)(command) for command in cdo_command_tasks]
    )
    return 


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
    parser.add_argument('--output', required=True, type=Path,help = "output path")
    parser.add_argument('--start', required=True, type=pandas.to_datetime,help="Date of first output file")
    parser.add_argument('--count', default=1, type=int,
                        help="Number of files to create, one every `freq`")
    parser.add_argument('--freq', default=60*60, type=lambda x: int(x),help="How often to save grib files in seconds. Default: 3600 (hourly)")
    parser.add_argument('--region', default=None, type=str,
                         help="Optional lon1,lon2,lat1,lat2 bounding box to subset "
                              "with CDO's sellonlatbox before conversion. "
                              "Default: no subsetting (global domain).")
    parser.add_argument('--tasks', default=4, type=int,
                         help="Number of worker processes to run in parallel. "
                              "Default: 4.")
    args = parser.parse_args() 

    # Parse the lat/lon region
    if args.region:
        parts = [p.strip() for p in args.region.split(",")]
        if len(parts) != 4:
            raise argparse.ArgumentTypeError(
                "--region expects four comma-separated values: lon1,lon2,lat1,lat2"
            )
        try:
            region = tuple(float(p) for p in parts)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "--region values must be numeric: lon1,lon2,lat1,lat2"
            )
    else:
        # Default case just slices Nones down the track in xarray
        region = (None,None,None,None)


    # Create a list of the requested date/times
    starttime = args.start
    endtime = starttime + timedelta(seconds = args.count * args.freq)

    # Pass to the era5 -> grib function
    replacement_era5grib(starttime, endtime,args.output,region,args.tasks)
    return

if __name__ == '__main__':
    main()

