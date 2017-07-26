import voltage_measurement as vm
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as C

plot_raw = True
plot_SunsVoc = True
plot_nxc = False

path = r'H:\DATA\20170712 EPFL FrontO2 Exp\B50-W22 5pc O2'+'/'
f = 'B50-W22-1_S6_R4_SiPDRefOD4_10av_52cmtotable_35cmtosample.Raw Data.dat'
f2 = 'B50-W22-1_S7_R5_SiPDRefOD4_ND124_10av_52cmtotable_35cmtosample.Raw Data.dat'

W = 0.024
Nd = 1.6e15
m = vm.Voltage_Measurement(path+f, Na=1, Nd=Nd, W=W, binn=50, crop_end=0.5, crop_start=0.04)
m2 = vm.Voltage_Measurement(path+f2, Na=1, Nd=Nd, W=W, binn=50, crop_end=0.60, crop_start=0.07)

if plot_raw:
    plt.figure()
    for name in m.data().dtype.names:
        if not name is 't':
            plt.subplot(1,2,1)
            plt.plot(m.raw['t'], m.raw[name], '.', color='black', label='raw '+name)
            plt.plot(m.data()['t'], m.data()[name],'.', label=name)
            plt.semilogy()
            plt.subplot(1,2,2)
            plt.plot(m2.raw['t'], m2.raw[name], '.', color='black', label='raw '+name)
            plt.plot(m2.data()['t'], m2.data()[name],'.', label=name)

    plt.legend(loc=0)
    plt.semilogy()

    fig, ax = plt.subplots()
    ax.plot(m.data()['Voc'],  m.data()['ref'], 'b.', label = 'Voc v V_gen' )
    ax.plot(m2.data()['Voc'],  m2.data()['ref']/10, 'c.', label = 'Voc v V_gen - LI' )
    ax.set_xlabel('Voc')
    ax.semilogy()
    ax.legend(loc='lower right')
    ax2 = ax.twiny()
    ax2.plot(np.log(m.data()['PL']), m.data()['ref'],'r.', label = 'log(PL) v V_gen')
    ax2.plot(np.log(m2.data()['PL']/10), m2.data()['ref']/10,'m.', label = 'log(PL) v V_gen LI')
    ax2.set_xlabel('log(PL)')
    ax2.legend(loc='upper left')

    plt.figure()
    plt.plot(m.data()['PL'], np.exp(m.data()['Voc']/m.V_T()), '.', label='HI - PL vs exp(Voc/V_T)')
    plt.plot(m2.data()['PL']/10, np.exp(m2.data()['Voc']/m2.V_T()),'.', label='LI - PL vs exp(Voc/V_T)')
    plt.loglog()
    plt.legend()

ss_Jsc = 0.03718
ss_Voc = 0.7148

# find the index of the data point that matches the steady-state Voc closest
onesun_i = np.nanargmin(np.abs(m.data()['Voc'] - ss_Voc))

# Find Fs from Voc
m.Fs = (ss_Jsc / C.e / W + m.dndt(m.nxc_from_Voc(), m.data()['t'])[onesun_i]) / m.data()['ref'][onesun_i]
m.Ai = 1.267e19
suns_net_Voc = m.gen_net(dndt=m.dndt(m.nxc_from_Voc(), m.data()['t'])) / m.gen_net(dndt=m.dndt(m.nxc_from_Voc(), m.data()['t']))[onesun_i]
# m.Fs = (ss_Jsc / C.e / W + m.dndt(m.nxc_from_PL(), m.data()['t'])[onesun_i]) / m.data()['ref'][onesun_i]
suns_net_PL = m.gen_net(dndt=m.dndt(m.nxc_from_PL(), m.data()['t'])) / m.gen_net(dndt=m.dndt(m.nxc_from_PL(), m.data()['t']))[onesun_i]

m2.Fs = m.Fs/10
# m2.Ai = m.Ai/10
suns_net_Voc_low = m2.gen_net(dndt=m2.dndt(m2.nxc_from_Voc(), m2.data()['t'])) / m.gen_net(dndt=m.dndt(m.nxc_from_Voc(), m.data()['t']))[onesun_i]
m2.find_Ai(filt=m2.filter_data(suns_net_Voc_low, slice(None), data_range=(5e-3, 5e-2)))
# print(suns_net_Voc_low[m2.filter_data(suns_net_Voc_low, slice(None), data_range=(5e-3, 1e-2))])
suns_net_PL_low = m2.gen_net(dndt=m2.dndt(m2.nxc_from_PL(), m2.data()['t'])) / m.gen_net(dndt=m.dndt(m.nxc_from_PL(), m.data()['t']))[onesun_i]

if plot_SunsVoc:
    # plot Suns-Voc
    # plt.figure('Voc')
    # plt.plot(m.data()['Voc'], m.gen_net(dndt=m.dndt(m.nxc_from_Voc(), m.data()['t'])), '.', label='gen_net')
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
    # ax2 = ax.twinx()
    # ax2.plot(m.data()['Voc'], m.gen_net(dndt=m.dndt(m.nxc_from_Voc(), m.data()['t'])), 'c.',  label='Suns-Voc')
    ax.plot(m.data()['Voc'],suns_net_Voc, 'b.',  label='Suns-Voc')
    # m.Fs = (ss_Jsc / C.e / W + m.dndt(m.nxc_from_PL(), m.data()['t'])[onesun_i]) / m.data()['ref'][onesun_i]
    # ax2.plot(m.iVoc_from_nxc(m.nxc_from_PL()), m.gen_net(dndt=m.dndt(m.nxc_from_PL(), m.data()['t'])), 'm.', label='Suns-iVoc')
    ax.plot(m.iVoc_from_nxc(m.nxc_from_PL()), suns_net_PL, 'r.', label='Suns-iVoc')
    ax.plot(m2.data()['Voc'], suns_net_Voc_low, 'c.',label='Suns-Voc_low')
    print(m2.Ai)
    ax.plot(m2.iVoc_from_nxc(m2.nxc_from_PL()), suns_net_PL_low, 'm.', label='Suns-iVoc_low')
    ax.legend(loc='upper left')
    ax.set_ylabel('suns')
    ax.semilogy()
    # ax2.legend(loc = 'lower right')
    # ax2.set_ylabel('gen rate (cm-3)')
    # ax2.semilogy()

if plot_nxc:
    plt.figure()
    plt.plot(m.data()['t'], m.nxc_from_Voc(), label='from Voc')

    plt.plot(m.data()['t'], m.nxc_from_PL(), label='from PL')
    plt.legend()
    plt.semilogy()

    # plt.figure()
    # plt.plot(m.data()['Voc'], m.iVoc_from_nxc(nxc=m.nxc_from_Voc()), '.')
    # plt.plot(m.data()['Voc'], m.data()['Voc'], '-')



if plot_raw or plot_SunsVoc or plot_nxc:
    plt.show()
