import numpy
from scipy.misc import logsumexp

def like_ring(z1, z2, fs1, fs2, dmax=1., snr_thresh=6.):
    """
    Project the complex SNR from a network of detectors to obtain the circularly
    polarized SNR for left/right circular signals for an array of points
    :param z1: complex snr in first detector
    :param z2: complex snr in second detector
    :param f_sig1: sensitivity of first detector: sigma * (F+ + i Fx)
    :param f_sig1: sensitivity of second detector: sigma * (F+ + i Fx)
    """
    norm = numpy.sqrt(numpy.absolute(fs1 * fs1.conjugate() + fs2 * fs2.conjugate()))
    fp_fc = abs((fs1.conjugate() *fs2 ).imag)
    
    cos_fac = norm**2/fp_fc

    snr = {}
    snr["coh"] = numpy.linalg.norm(numpy.array([z1, z2]))
    snr["left"] = numpy.absolute((z1 * fs1.conjugate() + z2 * fs2.conjugate()) / norm)
    snr["right"] = numpy.absolute((z1 * fs1 + z2 * fs2) / norm)
    loglike = {}

    for hand in ["left", "right"]:
        d = norm/snr[hand]
        cos_width = numpy.minimum(cos_fac / snr[hand] ** 0.5, 0.5)
        ll = snr[hand]**2/2 + 3 * numpy.log(d) - 3 * numpy.log(dmax) - \
            2 * numpy.log(snr[hand]) + numpy.log(cos_width)
        ll[snr[hand] < snr_thresh] = 0.
        loglike[hand] = logsumexp(ll)

    calc_coh = ((snr["coh"] ** 2 - snr["right"] ** 2) > 1.) & \
         ((snr["coh"] ** 2 - snr["left"] ** 2) > 1.)
    if sum(calc_coh):
        d,cosi = calc_d_cosi(z1, z2, fs1[calc_coh], fs2[calc_coh])
        ll = snr["coh"]**2/2 + numpy.log(32) + 7 * numpy.log(d) - 3 * numpy.log(dmax) \
            - 2 * numpy.log(fp_fc[calc_coh]) - 3 * numpy.log(1 - cosi ** 2)
        loglike["coh"] = logsumexp(ll)
    else:
        loglike["coh"] = 0
    like = logsumexp(loglike.values()) - numpy.log(len(fs2))
    return like


def calc_d_cosi(z1, z2, fs1, fs2):
    # First calculate the F-stat A parameters:
    A1iA3 = (fs2.imag * z1 - fs1.imag * z2) / (fs1.conjugate() * fs2).imag
    A2iA4 = (fs2.real * z1 - fs1.real * z2) / (fs1 * fs2.conjugate()).imag

    # these variables are what they say [ (a_plus +/- a_cross)^2 ]
    ap_plus_ac = numpy.absolute(A1iA3 - 1j * A2iA4)
    ap_minus_ac = numpy.absolute(A1iA3 + 1j * A2iA4)
    a_plus = 0.5 * (ap_plus_ac + ap_minus_ac)
    a_cross = 0.5 * (ap_plus_ac - ap_minus_ac)
    amp = a_plus + numpy.sqrt(a_plus ** 2 - a_cross ** 2)
    cosi = a_cross / amp
    d = 1. / amp
    return d, cosi
