import voltage_measurement as vm
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as C
from glob import glob
import os

debug_plots = False
plot_SunsVoc = True
plot_nxc = False

path = r'C:\Users\Robert\Google Drive\Ziv, Kyung, Robert Shared Folder\Robert\Experiments\RD016 - EPFL Lateral transport\20170712 EPFL FrontO2 Exp' + '/'
cell_data_lookup = r'C:\Users\Robert\Google Drive\Ziv, Kyung, Robert Shared Folder\Robert\Experiments\RD016 - EPFL Lateral transport\Front O2 - one_sun_stats.csv'

SS_names = ['Name', 'Wafer', 'Cell', 'Eff', 'Voc', 'Jsc', 'FF', 'Vmp', 'Jmp', 'Pmp', 'Rsc', 'Roc']
SS_dat = np.genfromtxt(cell_data_lookup, skip_header=2, delimiter=',', names=SS_names, dtype=None)

# print(SS_dat['Name'][:][:-2])
# print(SS_dat.dtype)

W = 0.024
Nd = 1.6e15

directories = ['B50-W19 10pc O2 (BL)', 'B50-W20 20pc O2', 'B50-W21 50pc O2', 'B50-W22 5pc O2']
wafer_names = ['B50-W19', 'B50-W20', 'B50-W21', 'B50-W22']
# directories = ['B50-W20 20pc O2']
# wafer_names = ['B50-W20']

for directory, wafer_name in zip(directories, wafer_names):

    for cell_num in range(1,6):
        fHI = path + directory + '/' + wafer_name + '-' + str(cell_num) + '_S6_R4_SiPDRefOD4_10av_52cmtotable_35cmtosample.Raw Data.dat'
        fLI = path + directory + '/' + wafer_name + '-' + str(cell_num) + '_S7_R5_SiPDRefOD4_ND124_10av_52cmtotable_35cmtosample.Raw Data.dat'

        mHI = vm.Voltage_Measurement(fHI, Na=1, Nd=Nd, W=W, binn=75, crop_end=0.65, crop_start=0.04)
        mLI = vm.Voltage_Measurement(fLI, Na=1, Nd=Nd, W=W, binn=75, crop_end=0.65, crop_start=0.07)

        # find the steady-state one sun IV data from the lookup file
        for i, name in enumerate(SS_dat['Name']):
            if wafer_name.encode() + b'_' + str(cell_num).encode() in name:
                ss_Jsc = SS_dat['Jsc'][i] / 1000
                ss_Voc = SS_dat['Voc'][i] / 1000
                # print(name, ss_Jsc, ss_Voc)

        # find the index of the data point that matches the steady-state Voc closest
        onesun_i = np.nanargmin(np.abs(mHI.data()['Voc'] - ss_Voc))

        # Find Fs using the one sun Jsc and dndt from Voc using the high injection
        # measurement
        Fs_Voc = (ss_Jsc / C.e / mHI.W + mHI.dndt(mHI.nxc_from_Voc(), mHI.data()['t'])[onesun_i]) / mHI.data()['ref'][onesun_i]
        mHI.Fs = Fs_Voc
        mLI.Fs = Fs_Voc / 10

        # Convert generation rates to units of suns using dndt from Voc
        suns_net_Voc = mHI.gen_net(dndt=mHI.dndt(mHI.nxc_from_Voc(), mHI.data()['t'])) / mHI.gen_net(dndt=mHI.dndt(mHI.nxc_from_Voc(), mHI.data()['t']))[onesun_i]
        suns_net_Voc_low = mLI.gen_net(dndt=mLI.dndt(mLI.nxc_from_Voc(), mLI.data()['t'])) / mHI.gen_net(dndt=mHI.dndt(mHI.nxc_from_Voc(), mHI.data()['t']))[onesun_i]

        # Auto find Ai from the low injection data. The Suns from the Voc data
        # is only used for creating the filter. The minimization is done comparing
        # Voc and iVoc data.
        mLI.find_Ai(filt=mLI.filter_data(suns_net_Voc_low, slice(None), data_range=(5e-3, 5e-2)))
        mHI.Ai = mLI.Ai * 10

        # Find Fs using the one sun Jsc and dndt from PL using the high injection
        # measurement
        # Fs_PL = (ss_Jsc / C.e / mHI.W + mHI.dndt(mHI.nxc_from_PL(), mHI.data()['t'])[onesun_i]) / mHI.data()['ref'][onesun_i]
        # mHI.Fs = Fs_PL
        # mLI.Fs = Fs_PL / 10

        # Convert generation rates to units of suns using dndt from PL
        suns_net_PL = mHI.gen_net(dndt=mHI.dndt(mHI.nxc_from_PL(), mHI.data()['t'])) / mHI.gen_net(dndt=mHI.dndt(mHI.nxc_from_PL(), mHI.data()['t']))[onesun_i]
        suns_net_PL_low = mLI.gen_net(dndt=mLI.dndt(mLI.nxc_from_PL(), mLI.data()['t'])) / mHI.gen_net(dndt=mHI.dndt(mHI.nxc_from_PL(), mHI.data()['t']))[onesun_i]

        if debug_plots:
            fig_raw, (ax_raw, ax_raw2) = plt.subplots(2, 1, sharex=True)
            fig_raw.canvas.set_window_title(str(cell_num))
            for name in mHI.data().dtype.names:
                if not name is 't':
                    ax_raw.plot(mHI.raw['t'], mHI.raw[name], '.', markersize=5, color='black', label='raw '+name)
                    ax_raw.plot(mHI.data()['t'], mHI.data()[name],'.', label=name)


                    ax_raw2.plot(mLI.raw['t'], mLI.raw[name], '.', markersize=5,color='black', label='raw '+name)
                    ax_raw2.plot(mLI.data()['t'], mLI.data()[name],'.', label=name)
            ax_raw.semilogy()
            ax_raw2.legend(loc=0)
            ax_raw2.semilogy()

            fig1, ax1 = plt.subplots()
            fig1.canvas.set_window_title(str(cell_num))
            ax1.plot(mHI.data()['Voc'],  mHI.data()['ref'], 'b.', label = 'Voc v V_gen' )
            ax1.plot(mLI.data()['Voc'],  mLI.data()['ref']/10, 'c.', label = 'Voc v V_gen - LI' )
            ax1.set_xlabel('Voc')
            ax1.semilogy()
            ax1.legend(loc='lower right')
            ax2 = ax1.twiny()
            ax2.plot(np.log(mHI.data()['PL']), mHI.data()['ref'],'r.', label = 'log(PL) v V_gen')
            ax2.plot(np.log(mLI.data()['PL']/10), mLI.data()['ref']/10,'m.', label = 'log(PL) v V_gen LI')
            ax2.set_xlabel('log(PL)')
            ax2.legend(loc='upper left')

            fig2, ax3 = plt.subplots()
            fig2.canvas.set_window_title(str(cell_num))
            ax3.plot(mHI.data()['PL'], np.exp(mHI.data()['Voc']/mHI.V_T()), '.', label='HI - PL vs exp(Voc/V_T)')
            ax3.plot(mLI.data()['PL']/10, np.exp(mLI.data()['Voc']/mLI.V_T()),'.', label='LI - PL vs exp(Voc/V_T)')
            ax3.loglog()
            ax3.legend()

        if plot_SunsVoc:
            # plot Suns-Voc
            # plt.figure('Voc')
            # plt.plot(m.data()['Voc'], m.gen_net(dndt=m.dndt(m.nxc_from_Voc(), m.data()['t'])), '.', label='gen_net')
            # plt.plot(mHI.data()['Voc'], m.gen_net(dndt=m.dndt(m.nxc_from_Voc(), m.data()['t'])), '.', label='gen_net')
            # plt.plot(m.data()['Voc'], m.gen_av(), '.', label='gen_av')
            # plt.legend()
            # plt.semilogy()
            #
            # plt.figure('PL')
            # plt.plot(m.iVoc_from_nxc(m.nxc_from_PL()), m.gen_net(dndt=m.dndt(m.nxc_from_PL(), m.data()['t'])), '.', label='gen_net')
            # plt.plot(m.iVoc_from_nxc(m.nxc_from_PL()), m.gen_av(), '.', label='gen_av')
            # plt.legend()
            # plt.semilogy()

            fig, ax = plt.subplots()
            fig.canvas.set_window_title(wafer_name + '-' + str(cell_num))
            # ax2 = ax.twinx()
            # ax2.plot(m.data()['Voc'], m.gen_net(dndt=m.dndt(m.nxc_from_Voc(), m.data()['t'])), 'c.',  label='Suns-Voc')
            ax.plot(mHI.data()['Voc'],suns_net_Voc, 'b.',  label='Suns-Voc')
            # m.Fs = (ss_Jsc / C.e / W + m.dndt(m.nxc_from_PL(), m.data()['t'])[onesun_i]) / m.data()['ref'][onesun_i]
            # ax2.plot(m.iVoc_from_nxc(m.nxc_from_PL()), m.gen_net(dndt=m.dndt(m.nxc_from_PL(), m.data()['t'])), 'm.', label='Suns-iVoc')
            print(mHI.Ai)
            ax.plot(mHI.iVoc_from_nxc(mHI.nxc_from_PL()), suns_net_PL, 'r.', label='Suns-iVoc')
            ax.plot(mLI.data()['Voc'], suns_net_Voc_low, 'c.',label='Suns-Voc_low')
            print(mLI.Ai)
            ax.plot(mLI.iVoc_from_nxc(mLI.nxc_from_PL()), suns_net_PL_low, 'm.', label='Suns-iVoc_low')
            ax.legend(loc='upper left')
            ax.set_ylabel('Illumination intensity (suns)')
            ax.set_xlabel('Voltage/Implied voltage (V)')
            ax.set_ylim(1e-3,1e2)
            ax.set_xlim(0.4,0.85)
            ax.semilogy()
            # ax2.legend(loc = 'lower right')
            # ax2.set_ylabel('gen rate (cm-3)')
            # ax2.semilogy()

        # if plot_nxc:
        #     plt.figure()
        #     plt.plot(m.data()['t'], m.nxc_from_Voc(), label='from Voc')
        #
        #     plt.plot(m.data()['t'], m.nxc_from_PL(), label='from PL')
        #     plt.legend()
        #     plt.semilogy()
        #
        #     # plt.figure()
        #     # plt.plot(m.data()['Voc'], m.iVoc_from_nxc(nxc=m.nxc_from_Voc()), '.')
        #     # plt.plot(m.data()['Voc'], m.data()['Voc'], '-')



if debug_plots or plot_SunsVoc or plot_nxc:
    plt.show()
