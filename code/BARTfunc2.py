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
# Copyright (C) 2015 University of Central Florida.  All rights reserved.
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
import scipy.constants as sc
#from mpi4py import MPI

import makeatm as mat
import PT as pt
import wine   as w
import reader as rd

BARTdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(BARTdir + "/../modules/MCcubed/src/")
import mcutils as mu

sys.path.append(BARTdir + "/../modules/transit/transit/python")
import imp

class BARTclass():
  def __init__(self,config):
    self.config = config
  def my_init(self):
    config = self.config
    """
    This is a class to define all the oporations surrounding 
    Modification History:
    ---------------------
    5-1-15 natelust@linux.com	Initial adaptation of this class from
                                older bartfunc.py
    """
    #import transit module only on class initialization so it happens
    #in the correct thread
    trm = __import__("transit_module")
    self.trm = trm
    #trm = imp.load_dynamic('_transit_module',BARTdir + "/../modules/transit/transit/python/_transit_module.so")

    # Some constants:
    # http://nssdc.gsfc.nasa.gov/planetary/factsheet/jupiterfact.html
    # http://nssdc.gsfc.nasa.gov/planetary/factsheet/sunfact.html
    Mjup =   1898.3 * 1e24 # m
    Rjup =  71492.0 * 1e3  # m
    Rsun = 696000.0 * 1e3  # m
    sys.argv = ["BARTFunc","-c",config]
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
      config.optionxform = str
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
                        help="Model-fitting parameters [default: %(default)s]")
    parser.add_argument("--molfit",    dest="molfit",    type=mu.parray,
                                       action="store",   default=None,
                        help="Molecules fit [default: %(default)s]")
    parser.add_argument("--Tmin",      dest="Tmin",      type=float,
                        action="store",  default=400.0,
                        help="Lower Temperature boundary [default: %(default)s]")
    parser.add_argument("--Tmax",      dest="Tmax",      type=float,
                        action="store",  default=3000.0,
                        help="Higher Temperature boundary [default: %(default)s]")
    parser.add_argument("--quiet",             action="store_true",
                        help="Set verbosity level to minimum",
                        dest="quiet")
    # Input-Converter Options:
    group = parser.add_argument_group("Input Converter Options")
    group.add_argument("--atmospheric_file",  action="store",
                       help="Atmospheric file [default: %(default)s]",
                       dest="atmfile", type=str,    default=None)
    group.add_argument("--PTtype",            action="store",
                       help="PT profile type.",
                       dest="PTtype",  type=str,    default="none")
                       #choices=('line', 'madhu'))
    group.add_argument("--tint",              action="store",
                       help="Internal temperature of the planet [default: "
                       "%(default)s].",
                       dest="tint",    type=float,  default=100.0)
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
    group.add_argument("--solution",                    action="store",
                       help="Solution geometry [default: %(default)s]",
                       dest="solution", type=str,       default="None",
                       choices=('transit', 'eclipse'))

    parser.set_defaults(**defaults)
    args2, unknown = parser.parse_known_args(remaining_argv)

    # Quiet all threads except rank 0:
    #rank = comm.Get_rank()
    #verb = rank == 0

    # Get (Broadcast) the number of parameters and iterations from MPI:
    #array1 = np.zeros(2, np.int)
    #mu.comm_bcast(comm, array1)
    #npars, niter = array1

    # :::::::  Initialize the Input converter ::::::::::::::::::::::::::
    atmfile = args2.atmfile
    molfit  = args2.molfit
    PTtype  = args2.PTtype
    params  = args2.params
    tepfile = args2.tep_name
    tint    = args2.tint
    Tmin    = args2.Tmin
    Tmax    = args2.Tmax

    # Extract necessary values from the TEP file:
    tep = rd.File(tepfile)
    # Stellar temperature in K:
    tstar = float(tep.getvalue('Ts')[0])
    # Stellar radius (in meters):
    rstar = float(tep.getvalue('Rs')[0]) * Rsun
    # Semi-major axis (in meters):
    sma   = float(tep.getvalue( 'a')[0]) * sc.au
    # Planetary radius (in meters):
    rplanet = float(tep.getvalue('Rp')[0]) * Rjup
    # Planetary mass (in kg):
    mplanet = float(tep.getvalue('Mp')[0]) * Mjup

    # Number of parameters:
    nfree   = len(params)                # Total number of free parameters
    nmolfit = len(molfit)                # Number of molecular free parameters
    nPT     = len(params) - len(molfit)  # Number of PT free parameters

    # Read atmospheric file to get data arrays:
    species, pressure, temp, abundances = mat.readatm(atmfile)
    # Reverse pressure order (for PT to work):
    pressure = pressure[::-1]
    nlayers  = len(pressure)   # Number of atmospheric layers
    nspecies = len(species)    # Number of species in the atmosphere
    #mu.msg(verb, "There are {:d} layers and {:d} species.".format(nlayers,
                                                                  #nspecies))
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
    #mu.msg(verb, "ICON FLAG 55")

    # Pressure-Temperature profile:
    PTargs = [PTtype]
    if PTtype == "line":
      # Planetary surface gravity (in cm s-2):
      gplanet = 100.0 * sc.G * mplanet / rplanet**2
      # Additional PT arguments:
      PTargs += [rstar, tstar, tint, sma, gplanet]

    # Allocate arrays for receiving and sending data to master:
    freepars = np.zeros(nfree,                 dtype='d')
    profiles = np.zeros((nspecies+1, nlayers), dtype='d')

    #mu.msg(verb, "ICON FLAG 62: i-molecs {}".format(imol))

    # Store abundance profiles:
    for i in np.arange(nspecies):
      profiles[i+1] = abundances[:, i]

    # :::::::  Spawn transit code  :::::::::::::::::::::::::::::::::::::
    # # transit configuration file:
    transitcfile = args2.config
 
    # FINDME: Find a way to set verb to the transit subprocesses.
    # Silence all threads except rank 0:
    # if verb == 0:
    #   rargs = ["--quiet"]
    # else:
    #   rargs = []

    # Initialize the transit python module:
    transit_args = ["transit", "-c", transitcfile]
    trm.transit_init(len(transit_args), transit_args)

    # Get wavenumber array from transit:
    nwave  = trm.get_no_samples()
    specwn = trm.get_waveno_arr(nwave)

    # :::::::  Output Converter  :::::::::::::::::::::::::::::::::::::::
    ffile    = args2.filter    # Filter files
    kurucz   = args2.kurucz    # Kurucz file
    solution = args2.solution  # Solution type

    # Log10(stellar gravity)
    gstar = float(tep.getvalue('loggstar')[0])
    # Planet-to-star radius ratio:
    rprs  = float(tep.getvalue('Rp')[0])*Rjup/(float(tep.getvalue('Rs')[0])*Rsun)
    #mu.msg(verb, "OCON FLAG 10: {}, {}, {}".format(tstar, gstar, rprs))

    nfilters = len(ffile)  # Number of filters:

    # FINDME: Separate filter/stellar interpolation?
    # Get stellar model:
    starfl, starwn, tmodel, gmodel = w.readkurucz(kurucz, tstar, gstar)
    # Read and resample the filters:
    nifilter  = [] # Normalized interpolated filter
    istarfl   = [] # interpolated stellar flux
    wnindices = [] # wavenumber indices used in interpolation
    #mu.msg(verb, "OCON FLAG 66: Prepare!")
    for i in np.arange(nfilters):
      # Read filter:
      filtwaven, filttransm = w.readfilter(ffile[i])
      # Check that filter boundaries lie within the spectrum wn range:
      #if filtwaven[0] < specwn[0] or filtwaven[-1] > specwn[-1]:
        #mu.exit(message="Wavenumber array ({:.2f} - {:.2f} cm-1) does not "
                #"cover the filter[{:d}] wavenumber range ({:.2f} - {:.2f} "
                #"cm-1).".format(specwn[0], specwn[-1], i, filtwaven[0],
                                                          #filtwaven[-1]))

      # Resample filter and stellar spectrum:
      nifilt, strfl, wnind = w.resample(specwn, filtwaven, filttransm,
                                                starwn,    starfl)
      #mu.msg(verb, "OCON FLAG 67: mean star flux: %.3e"%np.mean(strfl))
      nifilter.append(nifilt)
      istarfl.append(strfl)
      wnindices.append(wnind)
      
      #copy variables needed to self so they are accessable from the
      #run function
      self.pressure   = pressure
      self.nPT        = nPT
      self.PTargs     = PTargs
      self.Tmin       = Tmin
      self.Tmax       = Tmax
      self.nmolfit    = nmolfit
      self.imol       = imol
      self.abundances = abundances
      self.imetals    = imetals
      self.ratio      = ratio
      self.iH2        = iH2
      self.iHe        = iHe
      self.nwave      = nwave
      self.nfilters   = nfilters
      self.wnindices  = wnindices
      self.istarfl    = istarfl
      self.rprs       = rprs
      self.specwn     = specwn
      self.profiles   = profiles
      self.trm        = trm
      self.solution   = solution
      self.nifilter   = nifilter

  def run(self,params,indep,ex):
    #variables indep and ex are not used for this function, but are defined
    #to make it compatiable with existing demczs code.

    #Copy the parameters out of self to local variables, this may be slower
    #than refferencing them in line, but it is a quick hack to get things
    #working, someone may want to experament in cleaning this up later.
    pressure   =  self.pressure  
    nPT        =  self.nPT       
    PTargs     =  self.PTargs    
    Tmin       =  self.Tmin      
    Tmax       =  self.Tmax      
    nmolfit    =  self.nmolfit   
    imol       =  self.imol      
    abundances =  self.abundances
    imetals    =  self.imetals   
    ratio      =  self.ratio     
    iH2        =  self.iH2       
    iHe        =  self.iHe       
    nwave      =  self.nwave     
    nfilters   =  self.nfilters  
    wnindices  =  self.wnindices 
    istarfl    =  self.istarfl   
    rprs       =  self.rprs      
    specwn     =  self.specwn   
    profiles   = self.profiles 
    trm        = self.trm
    solution   = self.solution
    nifilter   = self.nifilter

    # Input converter calculate the profiles:
    try:
      profiles[0] = pt.PT_generator(pressure, params[0:nPT], PTargs)[::-1]
    except ValueError:
      print('Input parameters give non-physical profile.')
      # FINDME: what to do here?
    if np.sum(np.isnan(profiles[0])) >=1:
      return -9999999999*np.ones(nfilters)

    # If the temperature goes out of bounds:
    #print(params)
    #print(profiles[0])
    if np.any(profiles[0] < Tmin) or np.any(profiles[0] > Tmax):
      print("Out of bounds")
      return -999999999*np.ones(nfilters)

    # Scale abundance profiles:
    for i in np.arange(nmolfit):
      m = imol[i]
      # Use variable as the log10:
      profiles[m+1] = abundances[:, m] * 10.0**params[nPT+i]
    # Update H2, He abundances so sum(abundances) = 1.0 in each layer:
    q = 1.0 - np.sum(profiles[imetals+1], axis=0)
    profiles[iH2+1] = ratio * q / (1.0 + ratio)
    profiles[iHe+1] =         q / (1.0 + ratio)
    # print("qH2O: {}, Qmetals: {}, QH2: {}  p: {}".format(params[nPT],
    #                               q[50], profiles[iH2+1,50], profiles[:,50]))

    # Let transit calculate the model spectrum:
    #mu.msg(verb, "FLAG 81: sent data to transit")
    spectrum = trm.run_transit(profiles.flatten(),nwave)

    # Output converter band-integrate the spectrum:
    #mu.msg(verb, "FLAG 91: receive spectum")
    # Calculate the band-integrated intensity per filter:
    bandflux = np.zeros(nfilters)
    for i in np.arange(nfilters):
      if   solution == "eclipse":
        fluxrat = (spectrum[wnindices[i]]/istarfl[i]) * rprs*rprs
        bandflux[i] = w.bandintegrate(fluxrat, specwn,
                                      nifilter[i], wnindices[i])
      elif solution == "transit":
        bandflux[i] = w.bandintegrate(spectrum[wnindices[i]], specwn,
                                      nifilter[i], wnindices[i])

    # Send resutls back to MCMC:
    return bandflux

  # ::::::  End main Loop  :::::::::::::::::::::::::::::::::::::::::::
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


def bart_chi(model,data,errors,ex):
  return np.sum(((model-data)/errors)**2)

def bart_constraint(par,bound):
  chi = 0
  for i,p in enumerate(par):
    if p>=bound[i][0] and p<=bound[i][1]:
      chi += 0
    else:
      chi += 1e99
  return chi

#if __name__ == "__main__":
  # Open communications with the master:
  #comm = MPI.Comm.Get_parent()
  #main(comm)
