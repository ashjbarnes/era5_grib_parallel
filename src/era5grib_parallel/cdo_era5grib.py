# Copyright 2024 ACCESS-NRI (https://www.access-nri.org.au/)
# See the top-level COPYRIGHT.txt file for details.
#
# SPDX-License-Identifier: Apache-2.0
#
# Created by: Chermelle Engel <Chermelle.Engel@anu.edu.au>

import os
from datetime import datetime
from random import random

# Base directory of the ERA5 archive on NCI
ERADIR = "/g/data/rt52/era5/"

def fields():
    """
    Short function to return a list of lines with one line per field to be packaged (in order).
    Each line contains the ERA field name, whether the field is a single or pressure-level field 
    and the field number (in the ECMWF table)

    Parameters
    ----------
    None

    Returns
    -------
    list of string
        List of comma-delimited strings
    """
    
    lines = [
        "skt, single, 235",
        "sp, single, 134",
        "ci, single, 31",
        "sst, single, 34",
        "sd, single, 141",
        "stl1, single, 139",
        "stl2, single, 170",
        "stl3, single, 183",
        "stl4, single, 236",
        "swvl1, single, 39",
        "swvl2, single, 40",
        "swvl3, single, 41",
        "swvl4, single, 42",
        "u, pressure, 131",
        "v, pressure, 132",
        "t, pressure, 130",
        "q, pressure, 133",
        "lsm, single, 172",
        "z, single, 129"]
    return lines

def repackage_grib(dt_string,outdir,region=None):
    """
    For a single date/time, this function reads through the ERA5 netcdf archive
    to produce a GRIB file for UM driving model reconfiguration.

    Parameters
    ----------
    dt_string : string
            The requested date to be repackaged in %Y-%m-%dT%H:%M:%S format 
    outdir : Path
            The path for the output file to be written to
            requested date to be repackaged in %Y-%m-%dT%H:%M:%S format 
    region : tuple of float, optional
            (lon1, lon2, lat1, lat2) bounding box to subset with CDO's
            sellonlatbox before conversion. Default is None, which keeps
            the full (unsubset) global domain.

    Returns
    -------
    None.  The GRIB file is left in the outdir directory.
    """

    r = "%10.10f"%random()
    r = r.replace("0.", "")

    # Build the optional region-subsetting CDO operator. Left empty when no
    # region is requested so the generated commands are byte-identical to
    # the previous behaviour.
    if region is not None:
        lon1, lon2, lat1, lat2 = region
        region_op = "-sellonlatbox,%s,%s,%s,%s " % (lon1, lon2, lat1, lat2)
    else:
        region_op = ""
    
    dt = datetime.strptime(dt_string, "%Y-%m-%dT%H:%M:%S")
    compact_dt_string = datetime.strftime(dt, "%Y%m%d%H%M.t+000")
    
    os.chdir(outdir)
    
    merge_files = []

    lines = fields()

    for line in lines:
        var, stype, eccode = line.replace(' ', '').split(',')
        eccode = int(eccode)
    
        filedir = ERADIR + "/%s-levels/reanalysis/%s/%4.4d/"%(stype, var, dt.year)

        files = os.listdir(filedir)
        yyyymm = "%4.4d%2.2d"%(dt.year, dt.month)
        fname = [f for f in files if yyyymm in f][0]

        # Select out the specific date/time (and optionally the region) from
        # the netcdf archive file
        outfname = var + "_" + compact_dt_string + "_" + r + ".nc"
        cmd = "cdo -L --eccodes seldate," + dt_string + " " + region_op + filedir + "/" + fname + " " + outfname
        os.system(cmd)

        # Change the name of the variable if sea-ice
        if var == "ci":
          os.system("cdo -L --eccodes chname,siconc,ci " + outfname + " " + outfname + ".1")
          os.system("mv " + outfname + ".1 " + outfname)


        # Add the eccode and table information
        cmd = "cdo -L -setattribute," + var + "@code=%d"%eccode + " -setattribute," + var + "@table=128 " + outfname + " " + outfname+".1"
        os.system(cmd)
        os.system("mv " + outfname + ".1 "+outfname)

        merge_files.append(outfname)
    
    outfname = "ec_grib_" + compact_dt_string + ".nc"
    if os.path.exists(outfname):
            os.remove(outfname)
    if os.path.exists(outfname.replace('.nc', '')):
            os.remove(outfname.replace('.nc', ''))

    # merge the data into a single netcdf file  
    cmd = "cdo -L --eccodes merge " + " ".join(merge_files) + " " + outfname
    os.system(cmd)

    # convert the netcdf file into grib1 format
    os.system("cdo -L --eccodes -f grb1 copy " + outfname + " " + outfname.replace('.nc', ''))

    # Remove all the small files
    for fname in merge_files:
        if os.path.exists(fname):
            os.remove(fname)

    if os.path.exists(outfname):
            os.remove(outfname)

