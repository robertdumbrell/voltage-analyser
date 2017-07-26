import numpy as np
import matplotlib.pyplot as plt
import semiconductor.recombination
import scipy.constants as C

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

class Voltage_Measurement:
    names = ['t', 'ref', 'Voc', 'PL']
    def __init__(self, f, Ndop=1e15, W=0.02, binn=1, bkgnd_corr=0.95, bkgnd_loc='end', T=300, crop_start=0, crop_end=1):
        self.f = f
        self.Ndop = Ndop
        self.W = W
        self.T = T
        self.Ai = 1e16
        self.Fs = 1e20
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

    def nxc_from_Voc(self, Voc):
        nxc = (np.sqrt(self.Ndop**2 + self.ni_eff()**2 * np.exp(Voc/self.V_T())) - self.Ndop) / 2
        return nxc

    def nxc_from_PL(self, PL):
        nxc = (np.sqrt(self.Ndop**2 + 4 * self.Ai * PL / self.B_rad) - self.Ndop) / 2
        return nxc

    def iVoc_from_nxc(self, nxc):
        iVoc = self.V_T() * np.log(nxc * (self.Ndop + nxc) / self.ni_eff()**2)
        return iVoc

    def dndt(self, n, t):
        return np.gradient(n, t[1]-t[0])

    def gen_av(self, ref):
        return self.Fs * ref

    def gen_net(self, gen_av, dndt):
        return gen_av - dndt

    def V_T(self):
        return C.Boltzmann * self.T / C.e

    def ni_eff(self):
        return 1e10

    def B_rad(self):
        return 4.73E-15





if __name__ == '__main__':
    # path = r'C:\Users\Robert\Dropbox\PhD\Code\voltage-analyser\B50-W22-5pc-2_S6_R4_Flash_34cmsample_52cmtable_10av_SiPDRefOD3.Raw Data.dat'
    # meas = Voltage_Measurement(f=path, Ndop=1e15, W=0.02, binn=5)
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

    x = np.linspace(0,10)
    y = x**2
    y_dash = np.gradient(y, x[1]-x[0], edge_order=2)
    y_dashdash = np.gradient(y_dash, x[1]-x[0], edge_order=2)
    plt.plot(x, y, '.', label='y')
    plt.plot(x, y_dash, label='dydx')
    plt.plot(x, y_dashdash, label='d2ydx2')
    plt.show()
