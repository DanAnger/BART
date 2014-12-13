#! /usr/bin/env python

# ****************************** START LICENSE *******************************
# Bayesian Atmospheric Radiative Transfer (BART), a code to infer
# properties of planetary atmospheres based on observed spectroscopic
# information.
# 
# This project was completed with the support of the NASA Planetary
# Atmospheres Program, grant NNX12AI69G, held by Principal Investigator
# Joseph Harrington. Principal developers included graduate students
# Patricio E. Cubillos and Jasmina Blecic, programmer Madison Stemm, and
# undergraduates M. Oliver Bowman and Andrew S. D. Foster.  The included
# 'transit' radiative transfer code is based on an earlier program of
# the same name written by Patricio Rojo (Univ. de Chile, Santiago) when
# he was a graduate student at Cornell University under Joseph
# Harrington.  Statistical advice came from Thomas J. Loredo and Nate
# B. Lust.
# 
# Copyright (C) 2014 University of Central Florida.  All rights reserved.
# 
# This is a test version only, and may not be redistributed to any third
# party.  Please refer such requests to us.  This program is distributed
# in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.
# 
# Our intent is to release this software under an open-source,
# reproducible-research license, once the code is mature and the first
# research paper describing the code has been accepted for publication
# in a peer-reviewed journal.  We are committed to development in the
# open, and have posted this code on github.com so that others can test
# it and give us feedback.  However, until its first publication and
# first stable release, we do not permit others to redistribute the code
# in either original or modified form, nor to publish work based in
# whole or in part on the output of this code.  By downloading, running,
# or modifying this code, you agree to these conditions.  We do
# encourage sharing any modifications with us and discussing them
# openly.
# 
# We welcome your feedback, but do not guarantee support.  Please send
# feedback or inquiries to:
# 
# Joseph Harrington <jh@physics.ucf.edu>
# Patricio Cubillos <pcubillos@fulbrightmail.org>
# Jasmina Blecic <jasmina@physics.ucf.edu>
# 
# or alternatively,
# 
# Joseph Harrington, Patricio Cubillos, and Jasmina Blecic
# UCF PSB 441
# 4111 Libra Drive
# Orlando, FL 32816-2385
# USA
# 
# Thank you for testing BART!
# ******************************* END LICENSE *******************************

import sys, os
import argparse, ConfigParser
import numpy as np
from mpi4py import MPI

BARTdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(BARTdir + "/../modules/MCcubed/src/")
import mcutils as mu

import makeatm as mat
import PT as pt
import wine   as w
import reader as rd


def main(comm):
  """
  This is a hacked version of MC3's func.py.
  This function directly call's the modeling function for the BART project.

  Modification History:
  ---------------------
  2014-04-19  patricio  Initial implementation.  pcubillos@fulbrightmail.org
  2014-06-25  patricio  Added support for inner-MPI loop.
  """
  # Parse arguments:
  cparser = argparse.ArgumentParser(description=__doc__, add_help=False,
                         formatter_class=argparse.RawDescriptionHelpFormatter)
  # Add config file option:
  cparser.add_argument("-c", "--config_file", 
                       help="Configuration file", metavar="FILE")
  # Remaining_argv contains all other command-line-arguments:
  args, remaining_argv = cparser.parse_known_args()

  # Get parameters from configuration file:
  cfile = args.config_file
  if cfile:
    config = ConfigParser.SafeConfigParser()
    config.read([cfile])
    defaults = dict(config.items("MCMC"))
  else:
    defaults = {}
  parser = argparse.ArgumentParser(parents=[cparser])
  parser.add_argument("--func",      dest="func",      type=mu.parray, 
                                     action="store",  default=None)
  parser.add_argument("--indparams", dest="indparams", type=mu.parray, 
                                     action="store",   default=[])
  parser.add_argument("--params",    dest="params",    type=mu.parray,
                                     action="store",   default=None,
                      help="Model fitting parameter [default: %(default)s]")
  parser.add_argument("--molfit",    dest="molfit",    type=mu.parray,
                                     action="store",   default=None,
                      help="Molecule fit [default: %(default)s]")
  parser.add_argument("--quiet",             action="store_true",
                      help="Set verbosity level to minimum",
                      dest="quiet")
  # Input-Converter Options:
  group = parser.add_argument_group("Input Converter Options")
  group.add_argument("--atmospheric_file",  action="store",
                     help="Atmospheric file [default: %(default)s]",
                     dest="atmfile", type=str,    default=None)
  # transit Options:
  group = parser.add_argument_group("transit Options")
  group.add_argument("--config",  action="store",
                     help="transit configuration file [default: %(default)s]",
                     dest="config", type=str,    default=None)
  # Output-Converter Options:
  group = parser.add_argument_group("Output Converter Options")
  group.add_argument("--filter",                 action="store",
                     help="Waveband filter name [default: %(default)s]",
                     dest="filter",   type=mu.parray, default=None)
  group.add_argument("--tep_name",          action="store",
                     help="A TEP file [default: %(default)s]",
                     dest="tep_name", type=str,    default=None)
  group.add_argument("--kurucz_file",           action="store",
                     help="Stellar Kurucz file [default: %(default)s]",
                     dest="kurucz",   type=str,       default=None)
  group.add_argument("--solutiontype",                    action="store",
                     help="Solution geometry [default: %(default)s]",
                     dest="solutiontype", type=str,       default="transit",
                     choices=('transit', 'eclipse'))

  parser.set_defaults(**defaults)
  args2, unknown = parser.parse_known_args(remaining_argv)

  # Add path to func:
  # if len(args2.func) == 3:
  #   sys.path.append(args2.func[2])

  # Quiet all threads except rank 0:
  rank = comm.Get_rank()
  verb = rank == 0

  # Get (Broadcast) the number of parameters and iterations from MPI:
  array1 = np.zeros(2, np.int)
  mu.comm_bcast(comm, array1)
  npars, niter = array1

  mu.msg(verb, "BFUN FLAG 40  ***")
  # :::::::  Initialize the Input converter ::::::::::::::::::::::::::
  atmfile = args2.atmfile
  molfit  = args2.molfit
  params  = args2.params

  # Number of parameters:
  nfree   = len(params)                # Total number of free parameters
  nmolfit = len(molfit)                # Number of molecular free parameters
  nPT     = len(params) - len(molfit)  # Number of PT free parameters

  mu.msg(verb, "ICON FLAG 50")
  # Read atmospheric file to get data arrays:
  species, pressure, temp, abundances = mat.readatm(atmfile)
  nlayers  = len(pressure)   # Number of atmospheric layers
  nspecies = len(species)    # Number of species in the atmosphere
  mu.msg(verb, "There are {:d} layers and {:d} species.".format(nlayers,
                                                                nspecies))
  # Find index for Hydrogen and Helium:
  species = np.asarray(species)
  iH2     = np.where(species=="H2")[0]
  iHe     = np.where(species=="He")[0]
  # Get H2/He abundance ratio:
  ratio = (abundances[:,iH2] / abundances[:,iHe]).squeeze()
  # Find indices for the metals:
  imetals = np.where((species != "He") & (species != "H2"))[0]
  # Index of molecular abundances being modified:
  imol = np.zeros(nmolfit, dtype='i')
  for i in np.arange(nmolfit):
    imol[i] = np.where(np.asarray(species) == molfit[i])[0]

  # Send nlayers + nspecies to master:
  #mu.comm_gather(comm, np.array([nlayers, nspecies], dtype='i'), MPI.INT)
  mu.msg(verb, "ICON FLAG 55")

  # Determine inversion or non-inversion (by looking at parameter P2):
  MadhuPT = params[3] != -1

  # Allocate arrays for receiving and sending data to master:
  freepars = np.zeros(nfree,                 dtype='d')
  profiles = np.zeros((nspecies+1, nlayers), dtype='d')

  mu.msg(verb, "ICON FLAG 62: i-molecs {}".format(imol))

  # Store abundance profiles:
  for i in np.arange(nspecies):
    profiles[i+1] = abundances[:, i]

  # :::::::  Spawn transit code  :::::::::::::::::::::::::::::::::::::
  # Silence all threads except rank 0:
  if verb == 0:
    rargs = ["--quiet"]
  else:
    rargs = []

  # transit executable:
  tfunc = BARTdir + "/../modules/transit/transit/MPItransit"
  # transit configuration file:
  config = args2.config
  transitcfile = os.getcwd() + "/" + config

  # Spawn transit MPI communicator:
  transitcomm = mu.comm_spawn(tfunc, 1, transitcfile, rargs=rargs,
                              path=args2.func[2])

  # Get the number of spectral samples from transit:
  arrsize = np.zeros(1, dtype="i")
  mu.comm_gather(transitcomm, arrsize)
  nwave   = arrsize[0]

  # Send the input array size to transit (temperature and abundance profiles):
  nprofile = nlayers * (nspecies + 1)
  mu.comm_bcast(transitcomm, np.array([nprofile, niter], dtype="i"), MPI.INT)

  # Get wavenumber array from transit:
  specwn = np.zeros(nwave, dtype="d")
  mu.comm_gather(transitcomm, specwn)

  # :::::::  Output Converter  :::::::::::::::::::::::::::::::::::::::
  ffile    = args2.filter        # Filter files
  tepfile  = args2.tep_name      # TEP file
  kurucz   = args2.kurucz        # Kurucz file
  solution = args2.solutiontype  # Solution type

  # Some constants:
  # http://nssdc.gsfc.nasa.gov/planetary/factsheet/jupiterfact.html
  # http://nssdc.gsfc.nasa.gov/planetary/factsheet/sunfact.html
  Rjup =  71492.0 * 1e3 # m
  Rsun = 696000.0 * 1e3 # m

  # Extract necessary values from the TEP file:
  tep = rd.File(tepfile)
  # Stellar temperature in K.
  tstar = float(tep.getvalue('Ts')[0])
  # Log10(stellar gravity)
  gstar = float(tep.getvalue('loggstar')[0])
  # Planet-to-star radius ratio:
  rprs  = float(tep.getvalue('Rp')[0])*Rjup/(float(tep.getvalue('Rs')[0])*Rsun)
  mu.msg(verb, "OCON FLAG 10: {}, {}, {}".format(tstar, gstar, rprs))

  nfilters = len(ffile)  # Number of filters:

  # FINDME: Separate filter/stellar interpolation?
  # Get stellar model:
  starfl, starwn, tmodel, gmodel = w.readkurucz(kurucz, tstar, gstar)
  # Read and resample the filters:
  nifilter  = [] # Normalized interpolated filter
  istarfl   = [] # interpolated stellar flux
  wnindices = [] # wavenumber indices used in interpolation
  mu.msg(verb, "OCON FLAG 66: Prepare!")
  for i in np.arange(nfilters):
    # Read filter:
    filtwaven, filttransm = w.readfilter(ffile[i])
    # Check that filter boundaries lie within the spectrum wn range:
    if filtwaven[0] < specwn[0] or filtwaven[-1] > specwn[-1]:
      mu.exit(message="Wavenumber array ({:.2f} - {:.2f} cm-1) does not "
              "cover the filter[{:d}] wavenumber range ({:.2f} - {:.2f} "
              "cm-1).".format(specwn[0], specwn[-1], i, filtwaven[0],
                                                        filtwaven[-1]))

    # Resample filter and stellar spectrum:
    nifilt, strfl, wnind = w.resample(specwn, filtwaven, filttransm,
                                              starwn,    starfl)
    mu.msg(verb, "OCON FLAG 67: mean star flux: %.3e"%np.mean(strfl))
    nifilter.append(nifilt)
    istarfl.append(strfl)
    wnindices.append(wnind)

  # Allocate arrays for receiving and sending data to master:
  spectrum = np.zeros(nwave,    dtype='d')
  bandflux = np.zeros(nfilters, dtype='d')

  # Allocate array to receive parameters from MPI:
  params = np.zeros(npars, np.double)

  # ::::::  Main MCMC Loop  ::::::::::::::::::::::::::::::::::::::::::
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

  while niter >= 0:
    # Receive parameters from MCMC:
    mu.msg(verb, "ICON FLAG 70: Start iteration")
    mu.comm_scatter(comm, params)
    mu.msg(verb, "ICON FLAG 71: incon pars: {}".format(params))

    # Input converter calculate the profiles:
    try:
      profiles[0] = pt.PT_generator(pressure, params[0:nPT], MadhuPT)
    except ValueError:
      mu.msg(verb, 'Input parameters give non-physical profile.')
      # FINDME: what to do here?

    # Scale abundance profiles:
    for i in np.arange(nmolfit):
      m = imol[i]
      # Use variable as the log10:
      profiles[m+1] = (abundances[:, m] * 10.0**params[nPT+i])
    # Update H2, He abundances so sum(abundances) = 1.0 in each layer:
    q = 1.0 - np.sum(profiles[imetals+1], axis=0)
    profiles[iH2+1] = ratio * q / (1.0 + ratio)
    profiles[iHe+1] =         q / (1.0 + ratio)

    # transit calculates the model spectrum:
    mu.comm_scatter(transitcomm, profiles.flatten(), MPI.DOUBLE)
    mu.msg(verb, "BART FLAG 81: sent data to transit")
    # Gather (receive) spectrum from transit:
    mu.comm_gather(transitcomm, spectrum)

    # Output converter band-integrate the spectrum:
    mu.msg(verb, "OCON FLAG 91: receive spectum")
    # Calculate the band-integrated intensity per filter:
    for i in np.arange(nfilters):
      if   solution == "eclipse":
        fluxrat = (spectrum[wnindices[i]]/istarfl[i]) * rprs*rprs
        bandflux[i] = w.bandintegrate(fluxrat, specwn,
                                      nifilter[i], wnindices[i])
      elif solution == "transit":
        bandflux[i] = w.bandintegrate(spectrum[wnindices[i]], specwn,
                                      nifilter[i], wnindices[i])

    # Send resutls back to MCMC:
    mu.msg(verb, "OCON FLAG 95: Flux band integrated")
    mu.comm_gather(comm, bandflux, MPI.DOUBLE)
    mu.msg(verb, "OCON FLAG 97: Sent results back to MCMC")
    niter -= 1

  # Close the transit communicators:
  transitcomm.Barrier()
  transitcomm.Disconnect()

  mu.msg(verb, "FUNC FLAG 99: func out")
  # Close communications and disconnect:
  mu.exit(comm)


if __name__ == "__main__":
  # Open communications with the master:
  comm = MPI.Comm.Get_parent()
  main(comm)