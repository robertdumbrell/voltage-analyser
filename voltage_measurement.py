import numpy as np
import matplotlib.pyplot as plt
from semiconductor.recombination.intrinsic import Radiative
from semiconductor.material.bandgap_narrowing import BandGapNarrowing
import scipy.constants as C
from scipy.optimize import minimize_scalar

# self.Vt = C.k * self.Temp / C.e
# raw_names = ['t', 'gen', 'Voc', 'PL']
# raw = np.genfromtxt(fname, skip_header=1, delimiter='\t', names=raw_names)
#
# T = 300
# V_T = C.k * T / C.e
# W = 0.02
# N_dop = 1e15
#
# Fs = 1e20
# Ai = 1e16
# f_Sun = 0.8
# ni_eff = 9.71e9
# B = 4.73e-15
#
# nxc_Voc = 0.5 * (np.sqrt(N_dop**2 + 4 * ni_eff**2 * np.exp(raw['Voc'] / V_T)) - N_dop)
# nxc_PL = 0.5 * (np.sqrt(N_dop**2 + 4 * Ai * raw['PL'] / B) - N_dop)

def implied_J(Jsc, suns):
    return Jsc * (1 - suns)

class Voltage_Measurement:
    names = ['t', 'ref', 'Voc', 'PL']
    def __init__(self, f, Na=1e16, Nd=1, W=0.02, binn=1, bkgnd_corr=0.95, bkgnd_loc='end', T=300, crop_start=0, crop_end=1):
        self.f = f
        self.Na = Na
        self.Nd = Nd
        self.Ndop = np.abs(Na-Nd)
        self.W = W
        self.T = T
        self.Ai = 1e16
        self.Fs = 1e20
        self.R = 0.
        self.binn = binn
        self.bkgnd_corr = bkgnd_corr
        self.bkgnd_loc = bkgnd_loc
        self.crop_start = crop_start
        self.crop_end = crop_end
        self.raw = np.genfromtxt(self.f, delimiter='\t', skip_header=1, names=Voltage_Measurement.names)
        # self.data = self.data()
        # self._read_data()

    def data(self, **kwargs):
        """
        Function that conditions the raw data and returns the processed data.
        Processing order: background correction -> binning -> cropping
        """

        # do bagkground correction for reference and PL signal
        corr_index = int(self.raw.shape[0]*self.bkgnd_corr)

        if self.bkgnd_loc == 'start':
            self.raw['ref'] -= np.mean(self.raw['ref'][:corr_index])
            self.raw['PL'] -= np.mean(self.raw['PL'][:corr_index])
        elif self.bkgnd_loc == 'end':
            self.raw['ref'] -= np.mean(self.raw['ref'][corr_index:])
            self.raw['PL'] -= np.mean(self.raw['PL'][corr_index:])

        data = self.crop_data(self.binn_data(self.raw, self.binn), self.crop_start, self.crop_end)
        return data

    def binn_data(self, data, binn):
        """
        Bin a one dimensional structured numpy array along its axis.
        """
        if binn == 1:
            return data

        binned = np.zeros(data.shape[0] // binn, dtype=data.dtype)

        for name in data.dtype.names:
            for i in range(data.shape[0] // binn):
                binned[name][i] = np.mean(
                    data[name][i * binn:(i + 1) * binn], axis=0)

        return binned

    def crop_data(self, data, start, end):
        start_index = int(data.shape[0] * start)
        end_index = int(data.shape[0] * end)
        return data[start_index:end_index]

    def nxc_from_Voc(self, filt=slice(None)):

        Voc = self.data()['Voc'][filt]

        def nxc_Voc(x):
            return (np.sqrt(self.Ndop**2 + 4 * self.ni_eff(nxc=x)**2 * np.exp(Voc/self.V_T())) - self.Ndop) / 2

        nxc_guess = (np.sqrt(self.Ndop**2 + self.ni_eff(nxc=np.ones_like(Voc), constant=1e10)**2 * np.exp(Voc/self.V_T())) - self.Ndop) / 2
        nxc = self.find_iteratively(nxc_guess, nxc_Voc)
        return nxc

    def nxc_from_PL(self, filt=slice(None)):

        PL = self.data()['PL'][filt]

        def nxc_PL(x):
            return (np.sqrt(self.Ndop**2 + 4 * self.Ai * PL / self.B_rad(nxc=x)) - self.Ndop) / 2

        guess_nxc = (np.sqrt(self.Ndop**2 + 4 * self.Ai * PL / self.B_rad(nxc=np.ones_like(PL), constant=True)) - self.Ndop) / 2

        nxc = self.find_iteratively(guess_nxc, nxc_PL, verbose=False)
        return nxc

    def iVoc_from_nxc(self, nxc):
        iVoc = self.V_T() * np.log(nxc * (self.Ndop + nxc) / self.ni_eff(nxc=nxc)**2)
        return iVoc

    def tau_eff(self, nxc, method='general'):
        if method == 'general':
            tau = nxc / self.gen_net(self.dndt(nxc, self.data()['t']))
        elif method == 'steady-state':
            tau = nxc / self.gen_av()
        elif method == 'transient':
            tau = -1 * nxc / self.dndt(nxc, self.data()['t'])
        else:
            raise ValueError('tau_eff analysis method not recognized.')

        return tau

    def local_ideality_factor(self, Voc, G):
        # print (Voc.dtype)
        return 1 / (self.V_T()*np.gradient(np.log(G), Voc, edge_order=2))

    def find_iteratively(self, guess, fn, e=0.00001, verbose=False):
        diff = 1
        count = 0
        x_i = guess
        while np.mean(diff) > e:
            x_j = fn(x_i)
            diff = np.abs((x_j - x_i) / x_i)
            x_i = x_j
            if verbose:
                print(count, np.mean(diff) )
            count += 1
            if count > 100:
                raise ConvergenceError
        return x_i

    def dndt(self, n, t):
        return np.gradient(n, t, edge_order=2)

    def gen_av(self, ):
        ref = self.data()['ref']
        return self.Fs * ref * (1 - self.R) / self.W

    def gen_net(self, dndt, suns=True):

        return self.gen_av() - dndt

    def V_T(self):
        return C.k * self.T / C.e

    def ni_eff(self, nxc, constant=False):
        if constant == False:
            author = 'Schenk_1988fer'
            ni = 9.65e9
            ni_eff = ni*BandGapNarrowing(nxc=nxc,Nd=self.Nd,Na=self.Na,temp=self.T,author=author).ni_multiplier()
        else:
            ni_eff = np.ones_like(nxc) * constant

        return ni_eff

    def B_rad(self, nxc, constant=False):
        if constant == False:
            rad = Radiative(nxc=nxc, temp=self.T, Na=self.Na, Nd=self.Nd, author='Altermatt_2005')
            B = rad.get_B(nxc=nxc)
        elif constant == True:
            B = np.ones_like(nxc) * 4.73e-15
        return B

    def _SSdiffs_Ai(self, Ai, filt):
        """
        Returns the sum of squared differences between the Voc and iVoc curves
        within a range given by the filter. Ai is taken as an input paramter.
        Strictly for Ai optimization routines. Watch out for accidentally changing
        Ai when you don't mean to.
        """
        self.Ai = Ai
        return np.sum(np.power((self.data()['Voc'][filt] - self.iVoc_from_nxc(nxc=self.nxc_from_PL(filt=filt)))/self.data()['Voc'][filt], 2))


    def find_Ai(self, filt, verbose=False):
        """
        Find optimal Ai by matching the Voc and iVoc curves within a given range.
        Sets self.Ai.
        Note: Ai is assumed between 1e14 and 1e20
        """

        # Local function with single input Ai to be used with minimize_scaler.
        def SS_fn(x):
            return self._SSdiffs_Ai(x, filt)

        result = minimize_scalar(SS_fn, bounds=(1e14, 1e20), method='bounded')
        self.Ai = result.x

        if verbose:
            print(result.x, result.success)

        return result

    def filter_data(self, data, field, data_range):
        """
        Returns a boolean array where values of the data within the data_range
        are True and values outside the data_range are False.

        data = a numpy array
        field = A string corresponding to a field of a structured array to do the
                filtering on. Can be an empty slice if data is not a structured
                array.
        data_range = a tuple of form (min_val, max_val)
        """

        filt = (data[field] >= data_range[0]) * (data[field] <= data_range[1])
        return filt

if __name__ == '__main__':
    path = r'C:\Users\Robert\Dropbox\PhD\Code\voltage-analyser\B50-W22-5pc-2_S6_R4_Flash_34cmsample_52cmtable_10av_SiPDRefOD3.Raw Data.dat'
    meas = Voltage_Measurement(f=path, Nd=1e15,Na=1, W=0.02, binn=5)
    # plt.plot(meas.data()['t'], meas.data()['ref'], '.', label='none')
    # # meas.crop_end=0.8
    # # meas.crop_start=0.1
    # # meas.binn=10
    # meas.bkgnd_loc='start'
    # meas.bkgnd_corr=0.2
    # plt.plot(meas.data()['t'], meas.data()['ref'], '.', label='bkgnd start')
    # # plt.plot(meas.data['t'], meas.data['ref'], '.', label='5')
    # # d = meas.binn_data(meas.data, 2)
    # # d_crop = meas.crop_data(d, 0.1, 0.75)
    # # plt.plot(d['t'], d['ref'], '.', label='2')
    # # plt.plot(d_crop['t'], d_crop['ref'], '.', label='2_cropped')
    # plt.legend()
    # plt.semilogy()
    # plt.show()

    # x = np.linspace(0,10)
    # y = x**2
    # y_dash = np.gradient(y, x[1]-x[0], edge_order=2)
    # y_dashdash = np.gradient(y_dash, x[1]-x[0], edge_order=2)
    # plt.plot(x, y, '.', label='y')
    # plt.plot(x, y_dash, label='dydx')
    # plt.plot(x, y_dashdash, label='d2ydx2')
    # plt.show()

    # B tests
    # nxc = np.logspace(10,18)
    # rad = Radiative(nxc=nxc, temp=300, doping=1e14, author='Altermatt_2005')
    # # print(rad.author)
    # plt.plot(nxc, rad.get_B(nxc=nxc, ))
    # plt.plot(nxc, rad.get_B(nxc=nxc, Nd=1e15))
    # plt.plot(nxc, rad.get_B(nxc=nxc, Nd=1e17))
    # plt.semilogx()
    # plt.show()

    # #ni_eff and B_rad tests
    # meas.nxc_from_Voc()
    # meas.Ai=3.2e18
    # plt.plot(meas.data()['t'],meas.nxc_from_PL(), label='PL')
    # plt.plot(meas.data()['t'],meas.nxc_from_Voc(), label='Voc')
    # plt.semilogy()
    # plt.legend()

    a = np.logspace(0,3)



    plt.show()
