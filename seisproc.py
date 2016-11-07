import numpy as np
import scipy
import obspy
import math

class seisproc:
# A series function for SAC seismic data processing

    def readpz(fname):
    # This function read SAC polezero files and return polezero to a dictionary.
        f = open(fname, "r")
        flines = f.readlines()
        i=0
        for line in flines:
            if "ZEROS" in line:
                zeroindex = i
            elif "POLES" in line:
                poleindex = i
            elif "CONSTANT" in line:
                consindex = i
            i = i+1
        pz = {}

        # Get zeros
        nzero = int(flines[zeroindex].split()[1])
        zeros = []
        for i in range(1,nzero+1):
            templst = flines[zeroindex+1].split()
            temp = complex(float(templst[0]),float(templst[1]))
            zeros.append(temp)
        pz["zeros"] = zeros

        # Get poles
        npole = int(flines[poleindex].split()[1])
        poles = []
        for i in range(1,npole+1):
            templst = flines[poleindex+1].split()
            temp = complex(float(templst[0]),float(templst[1]))
            poles.append(temp)
        pz["poles"] = poles

        # Get gain and sensitivity, set gain = 1.0, senstivity = CONSTANT
        pz["gain"] = 1.0
        pz["sensitivity"] = float(flines[consindex].split()[1])

        return pz


    def spec(tr,start_time=0,end_time=0):
    # Calculate spectrum of a time series, return amplitude an phase
        delta=round(tr.stats['delta'],6)
        fq=round(1.0/delta)

        if start_time == 0 and end_time ==0:
            signal=tr.data
        else:
            signal=tr.data[start_time * fq : end_time * fq]
        freq_signal=np.fft.rfft(signal)
        freqs=np.fft.rfftfreq(len(signal),d=delta)
        Ampart=abs(freq_signal)
        Phpart=np.angle(freq_signal)

        spec_data={}
        spec_data['Freq']=freqs
        spec_data['Amp']=Ampart
        spec_data['Pha']=Phpart

        return spec_data


    def whiten(tr,minf,maxf):
    # Spectral whitening
        delta=round(tr.stats['delta'],6)
        freq_s=np.fft.rfft(tr.data)
        freqs=np.fft.rfftfreq(len(tr.data),d=delta)

        Phpart=np.angle(freq_s,deg=False)

        alpha=0.5
        N=(maxf-minf)/((freqs[1]-freqs[0])*(1-alpha))
        window=scipy.signal.tukey(N,alpha=alpha)
        Ampart=np.array([window[1] for x in range(len(Phpart))])
        n1=minf/(freqs[1]-freqs[0])
        n2=alpha*(N-1)/2
        if n1 > n2:
            w_end=min((len(Ampart)-n1+n2),len(window))
            Ampart[(n1-n2):(w_end+n1-n2)]=window[0:w_end]
        else:
            w_end=min(len(Ampart),(len(window)-n2+n1))
            Ampart[0:w_end]=window[(n2-n1):(w_end+n2-n1)]

        freq_s_n=Ampart*np.cos(Phpart) + Ampart*np.sin(Phpart)*1j
        time_s=tr.copy()
        time_s.data=np.fft.irfft(freq_s_n)

        return time_s


    def rotate(trN,trE,theta):
    # rotate horizontal components of seismogram, theta is counterclockwise rotate angle.
        rad=math.radians(theta)

        N_y=trN.copy()
        E_x=trE.copy()
        N_y.data=trN.data*math.cos(rad)-trE.data*math.sin(rad)
        E_x.data=trN.data*math.sin(rad)+trE.data*math.cos(rad)

        return N_y,E_x

