import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
import control
import os, datetime
rand = np.random.rand
import matplotlib.ticker
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

__version__ = "1.1.1"

def assign_poles_to_bins(max_poles=5, bins=5):
    """Assign poles to frequency bins.  Each bin is assigned
    0,1, or 2 poles until max_poles is reached."""
    poles = [0]*bins
    
    for i in range(bins):
        num_poles = np.sum(poles)
        max_remaining = max_poles-num_poles
        if max_remaining <= 0:
            break
        elif max_remaining == 1:
            # the current bin can have 0 or 1 poles
            r = rand()
            if r > 0.5:
                poles[i] = 1
        elif max_remaining > 1:
            # current bin can have 0, 1, or 2 poles
            r = rand()
            if r > 0.7:
                poles[i] = 2
            elif r > 0.3:
                poles[i] = 1
        
    return poles



def assign_zeros_to_bins(poles, max_zeros=None):
    n = np.sum(poles)
    m = n-1
    if (max_zeros is None) or (max_zeros>m):
        max_zeros = m
    bins = len(poles)
    zeros = [0]*bins
    
    for i, p_i in enumerate(poles):
        num_zeros = np.sum(zeros)
        max_remaining = max_zeros-num_zeros
        if p_i > 0:
            # no zeros in this bin 
            continue
        if max_remaining <= 0:
            break
        elif max_remaining == 1:
            # the current bin can have 0 or 1 poles
            r = rand()
            if r > 0.5:
                zeros[i] = 1
        elif max_remaining > 1:
            # current bin can have 0, 1, or 2 poles
            r = rand()
            if r > 0.7:
                zeros[i] = 2
            elif r > 0.3:
                zeros[i] = 1
        
    return zeros
    

def random_log_freq(low_exponent):
    """Generate a random frequency on the range 10**low_exponent -
    10**(low_exponent+1)"""
    high_exponent = low_exponent + 1
    mid_exp = (low_exponent+high_exponent)/2
    act_exp = mid_exp + 0.7*(rand()-0.5)
    freq = 10**act_exp
    return freq


def pole_bins_to_den(poles):
    # powers of 10 corresponding to each frequency bin
    exponents = [None, -2, -1, 0, 1]
    
    if poles[0] == 1:
        G = control.TransferFunction(1,[1,0])
    elif poles[0] == 2:
        G = control.TransferFunction(1,[1,0,0])
    else:
        G = 1
    
    for p_i, exp_i in zip(poles[1:], exponents[1:]):
        if p_i == 0:
            # skip
            continue
        freq_i = random_log_freq(exp_i)
        w_i = 2.0*np.pi*freq_i
        
        if p_i == 1:
            G_i = control.TransferFunction(1,[1,w_i])
        elif p_i == 2:
            z_i = 0.8*rand()
            G_i = control.TransferFunction(1,[1,2*z_i*w_i,w_i**2])
            
        G *= G_i
        
    return np.squeeze(G.den)




def zero_bins_to_num(zeros):
    exponents = [None, -2, -1, 0, 1]# powers of 10 corresponding to each frequency bin
    
    if zeros[0] == 1:
        G = control.TransferFunction([1,0],1)
    elif zeros[0] == 2:
        G = control.TransferFunction([1,0,0],1)
    else:
        G = 1
    
    for z_i, exp_i in zip(zeros[1:], exponents[1:]):
        if z_i == 0:
            # skip
            continue
        freq_i = random_log_freq(exp_i)
        w_i = 2.0*np.pi*freq_i
        
        if z_i == 1:
            G_i = control.TransferFunction([1,w_i],1)
        elif z_i == 2:
            z_i = 0.8*rand()
            G_i = control.TransferFunction([1,2*z_i*w_i,w_i**2],1)
            
        G *= G_i
        
    if G == 1:
        # This is the default value if zeros is a list of 
        # all zeros: [0,0,0,...,0]
        return G
    else:
        return np.squeeze(G.num)




def random_Bode_TF(max_poles=5, max_zeros=None):
    plist = assign_poles_to_bins(max_poles=max_poles)
    while not np.any(plist):
        # We will not allow a TF that has no poles
        plist = assign_poles_to_bins(max_poles=max_poles)
    
    zlist = assign_zeros_to_bins(plist, max_zeros=max_zeros)
    den = pole_bins_to_den(plist)
    num = zero_bins_to_num(zlist)
    G = control.TransferFunction(num,den)
    return G





def set_log_ticks(ax,nullx=False):
    locmaj = matplotlib.ticker.LogLocator(base=10,numticks=12) 
    ax.xaxis.set_major_locator(locmaj)
    if nullx:
        ax.xaxis.set_major_formatter(matplotlib.ticker.NullFormatter())

    mysubs = np.arange(0.1,0.99,0.1)
    locmin = matplotlib.ticker.LogLocator(base=10.0,subs=mysubs,numticks=12)
    ax.xaxis.set_minor_locator(locmin)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())





def mygrid(ax):
    ax.grid(1, which="both",ls=":", color='0.75')





def set_db_ticks(ax, db):
    dbmin = db.min()
    dbmax = db.max()
    # aim for less than 6 ticks in muliples of 10, 20, 40 , ...
    myspan = dbmax-dbmin
    maxticks = 6
    
    ticklist = [10,20,40,60,80]
    
    N = None
    
    for tick in ticklist:
        if myspan/tick < maxticks:
            N = tick
            break

    if N is None:
        N = 100
        
    majorLocator = MultipleLocator(N)
    majorFormatter = FormatStrFormatter('%d')

    ax.yaxis.set_major_locator(majorLocator)
    ax.yaxis.set_major_formatter(majorFormatter)





def set_phase_ticks(ax, phase):
    phmin = phase.min()
    phmax = phase.max()
    # if 4 or 5 multiples of 45 is enough, use 45 as the base
    mul45 = (phmax-phmin)/45
    if mul45 < 6:
        N = 45
    elif mul45 < 12:
        N = 90
    else:
        N = 180
    majorLocator = MultipleLocator(N)
    majorFormatter = FormatStrFormatter('%d')

    ax.yaxis.set_major_locator(majorLocator)
    ax.yaxis.set_major_formatter(majorFormatter)





def fix_phase(phase):
    """Double integrators seem to cause the phase to start at +180, 
    which is not right.  Since the code generates only strictly proper
    TFs with relative degree of at least 1, there should always be at 
    least one more pole than zero and the high frequency phase should 
    always be negative.  If the phase ends above 0, shift it by -360 
    degrees
    """
    if phase[-1] > 0:
        phase -= 360
    return phase



def calc_mag_and_phase(G, f):
    w = 2.0*np.pi*f
    s = 1.0j*w
    Gjw = G(s)
    db = 20.0*np.log10(abs(Gjw))
    phase_rad = np.angle(Gjw)
    phase_rad = np.unwrap(phase_rad)
    phase = phase_rad*180.0/np.pi
    phase = fix_phase(phase)
    return db, phase


def plot_bode(f, db, phase):
    plt.figure()
    plt.subplot(211)
    plt.semilogx(f,db)
    plt.ylabel('dB Mag.')
    ax = plt.gca()
    set_log_ticks(ax,nullx=True)
    set_db_ticks(ax, db)
    mygrid(ax)
    plt.subplot(212)
    plt.semilogx(f,phase)
    plt.ylabel('Phase (deg.)')
    plt.xlabel('Freq. (Hz)')
    ax = plt.gca()
    set_log_ticks(ax)
    set_phase_ticks(ax, phase)
    mygrid(ax)
    


def plot_bode_for_TF(G, f=None):
    if f is None:
        f = np.logspace(-4,3,1000)
    db, phase = calc_mag_and_phase(G,f)
    plot_bode(f, db, phase)




def steady_state_fixed_sine(G,f,input_amp=1.0):
    w = 2*np.pi*f
    s = 1.0j*w
    Gjw = G(s)
    m = abs(Gjw)*input_amp
    phi = np.angle(Gjw)
    y_ss = '%0.4g sin(2*pi*%0.4g %+0.4g)' % (m,f,phi)
    return y_ss



def preserve_G(G):
    """Print out the code to recreate a specific TF"""
    ns = np.squeeze(G.num)
    ds = np.squeeze(G.den)
    numstr = 'num = %s' % np.array2string(ns,separator=',')
    denstr = 'den = %s' % np.array2string(ds,separator=',')
    Gstr = 'G = control.TransferFunction(num,den)'
    
    print(numstr)
    print(denstr)
    print(Gstr)


def get_csv_filename(basename="bode_id"):
    base2 = basename + '_%0.3i'
    # find unused file name/number
    for i in range(1,1000):
        pat = base2 % i
        pat += '*.csv'
        if not os.path.exists(pat):
            break

    base_out = basename + '_%0.3i' % i
    fmt = '_%m_%d_%Y_%I_%M%P'
    now = datetime.datetime.now()
    time_stamp = now.strftime(fmt) 
    fn = base_out + time_stamp + '.csv'
    return fn
    

def save_bode_to_csv(G, f=None, basename="bode_id"):
    if f is None:
        f = np.logspace(-4,3,1000)
    db, phase = calc_mag_and_phase(G,f)
    data = np.column_stack([f,db,phase])
    header = '#Freq. (Hz.), dB Mag., Phase (deg.)'
    # need safe file name
    fn = get_csv_filename(basename)
    np.savetxt(fn, data,  delimiter=',', header=header)
    return fn


# need a load from csv feature
def plot_bode_from_csv(fn):
    data = np.loadtxt(fn, delimiter=',')
    f = data[:,0]
    db = data[:,1]
    phase = datea[:,2]
    plot_bode(f, db, phase)
                      
