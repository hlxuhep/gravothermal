import numpy as np

# Natural_Units
# 1. SI-prefixes
yotta = 1.0e24
zetta = 1.0e21
exa   = 1.0e18
peta  = 1.0e15
tera  = 1.0e12
giga  = 1.0e9
mega  = 1.0e6
kilo  = 1.0e3
hecto = 1.0e2
deca  = 1.0e1
deci  = 1.0e-1
centi = 1.0e-2
milli = 1.0e-3
micro = 1.0e-6
nano  = 1.0e-9
pico  = 1.0e-12
femto = 1.0e-15
atto  = 1.0e-18
zepto = 1.0e-21
yocto = 1.0e-24

# 2. Units
# 2.1 Angles
deg	= np.pi / 180.0
arcmin = deg / 60.0
arcsec = arcmin / 60.0

# 2.2 Energy
GeV = 1.0
meV = 1.0e-12 * GeV
eV	 = 1.0e-9 * GeV
keV = 1.0e-6 * GeV
MeV = 1.0e-3 * GeV
TeV = 1.0e3 * GeV
PeV = 1.0e6 * GeV
Rydberg = 13.605693009 * eV

# 2.3 Mass
gram  = 5.60958884493318e23 * GeV
kg	   = 1.0e3 * gram
tonne = 1.0e3 * kg
lbs   = 0.453592 * kg
AMU   = 0.9314940954 * GeV

# 2.4 Length
cm			 = 5.067730214314311e13 / GeV
mm			 = 1.0e-1 * cm
meter		 = 1.0e2 * cm
km			 = 1.0e3 * meter
fm			 = 1.0e-15 * meter
inch		 = 2.54 * cm
foot		 = 12.0 * inch
yard		 = 3.0 * foot
mile		 = 1609.344 * meter
Angstrom	 = 1.0e-10 * meter
Bohr_Radius = 5.291772083e-11 * meter


# 2.5 Area
barn	 = 1.0e-24 * cm * cm
pb		 = pico * barn
acre	 = 4046.86 * meter * meter
hectare = 1.0e4 * meter * meter

# 2.6 Time
sec	= 299792458.0 * meter
ms		= milli * sec
ns		= nano * sec
minute = 60.0 * sec
hr		= 60 * minute
day	= 24 * hr
week	= 7 * day
year	= 365.25 * day

Joule	 = kg * np.power(meter / sec, 2)
erg	 = gram * np.power(cm / sec, 2)
cal	 = 4.184 * Joule

# 2.7 Frequency
Hz = 1.0 / sec

# 2.8 Force
Newton = kg * meter / sec / sec
dyne	= 1.0e-5 * Newton

# 2.9 Power
Watt = Joule / sec

# 2.10 Pressure
Pa	   = Newton / meter / meter
hPa   = hecto * Pa
kPa   = kilo * Pa
bar   = 1.0e5 * Pa
barye = dyne / cm / cm

# 2.11 Temperature
Kelvin = 8.6173303e-14 * GeV

# 2.12 Electric Charge
Elementary_Charge = 0.30282212
Coulomb		   = Elementary_Charge / (1.602176565e-19)

# 2.13 Voltage
Volt = Joule / Coulomb

# 2.14 Electric current
Ampere = Coulomb / sec

# 2.15 Electrical capacitance
Farad = Coulomb / Volt

# 2.16 Magnetic induction/ flux density
Tesla = (Newton * sec) / (Coulomb * meter)
Gauss = 1.0e-4 * Tesla

# 2.17 Magnetic flux
Weber = Tesla * meter * meter

# 2.20 Electrical resistance
Ohm	 = Volt / Ampere
Siemens = 1.0 / Ohm

# 2.21 Amount
mole = 6.02214076e23

# 3. Physical constants
# 3.1 Hadron masses
mProton  = 938.2720813 * MeV
mNeutron = 939.5654133 * MeV
mNucleon = 0.932 * GeV

# 3.2 Quark masses
mUp	  = 2.3 * MeV
mDown	  = 4.8 * MeV
mCharm	  = 1.275 * GeV
mStrange = 95 * MeV
mTop	  = 173.210 * GeV
mBottom  = 4.180 * GeV

# 3.3 Lepton masses
mElectron = 0.5109989461 * MeV
mMuon	   = 105.6583745 * MeV
mTau	   = 1776.86 * MeV

# 3.4 Boson masses
mZ		= 91.1876 * GeV
mW		= 80.379 * GeV
mHiggs = 125.18 * GeV

# 3.5 Coupling constants
aEM			 = 1.0 / 137.035999139
mPlanck		 = 1.22091e19 * GeV
mPlanck_reduced = mPlanck / np.sqrt(8.0 * np.pi)
G_Newton		 = 1.0 / mPlanck / mPlanck
G_Fermi		 = 1.1663787e-5 / GeV / GeV

# 3.6 Energy scales
Higgs_VeV = np.power((np.sqrt(2) * G_Fermi), -0.5)
QCD_scale = 218 * MeV

# 4. Astronomical parameters
# 4.1 Masses
mEarth = 5.9724e24 * kg
mSun	= 1.98848e30 * kg

# 4.2 Distances
rEarth = 6371 * km
rSun	= 6.957E8 * meter
AU		= 149597870700 * meter
pc		= 3.08567758e16 * meter
kpc	= kilo * pc
Mpc	= mega * pc
ly		= 365.25 * day

# 5. Useful functions
# 5.1 Reduced mass
def Reduced_Mass(ma, mb):
    return (ma * mb) / (ma + mb) 