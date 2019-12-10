
from argparse import ArgumentParser

parser = ArgumentParser("tilt filt")
parser.add_argument("--minsnr", default=1.1, type=float)
parser.add_argument("--vmax", default=None, type=float)
parser.add_argument("--vmin", default=None, type=float)
args = parser.parse_args()

import numpy as np
import h5py
import pylab as plt
from IPython import embed

from scitbx.matrix import sqr
from dxtbx.model.experiment_list import ExperimentList, ExperimentListFactory
from dials.array_family import flex
from dials.algorithms.shoebox import MaskCode
from dials.model.data import Shoebox
from dials.algorithms.indexing import assign_indices


def is_outlier(points, thresh=3.5):
    """http://stackoverflow.com/a/22357811/2077270"""
    if len(points.shape) == 1:
        points = points[:, None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median) ** 2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


def refls_to_q(refls, detector, beam, update_table=False):

    orig_vecs = {}
    fs_vecs = {}
    ss_vecs = {}
    u_pids = set([r['panel'] for r in refls])
    for pid in u_pids:
        orig_vecs[pid] = np.array(detector[pid].get_origin())
        fs_vecs[pid] = np.array(detector[pid].get_fast_axis())
        ss_vecs[pid] = np.array(detector[pid].get_slow_axis())

    s1_vecs = []
    q_vecs = []
    for r in refls:
        pid = r['panel']
        i_fs, i_ss, _ = r['xyzobs.px.value']
        panel = detector[pid]
        orig = orig_vecs[pid]
        fs = fs_vecs[pid]
        ss = ss_vecs[pid]

        fs_pixsize, ss_pixsize = panel.get_pixel_size()
        s1 = orig + i_fs*fs*fs_pixsize + i_ss*ss*ss_pixsize  # scattering vector
        s1 = s1 / np.linalg.norm(s1) / beam.get_wavelength()
        s1_vecs.append(s1)
        q_vecs.append(s1-beam.get_s0())

    if update_table:
        refls['s1'] = flex.vec3_double(tuple(map(tuple,s1_vecs)))
        refls['rlp'] = flex.vec3_double(tuple(map(tuple,q_vecs)))

    return np.vstack(q_vecs)


def refls_to_hkl(refls, detector, beam, crystal,
                 update_table=False, returnQ=False):
    """
    convert pixel panel reflections to miller index data

    :param refls:  reflecton table for a panel or a tuple of (x,y)
    :param detector:  dxtbx detector model
    :param beam:  dxtbx beam model
    :param crystal: dxtbx crystal model
    :param update_table: whether to update the refltable
    :param returnQ: whether to return intermediately computed q vectors
    :return: if as_numpy two Nx3 numpy arrays are returned
        (one for fractional and one for whole HKL)
        else dictionary of hkl_i (nearest) and hkl (fractional)
    """
    q_vecs = refls_to_q(refls, detector, beam, update_table=update_table)
    Ai = sqr(crystal.get_A()).inverse()
    Ai = Ai.as_numpy_array()
    HKL = np.dot(Ai, q_vecs.T)
    HKLi = map(lambda h: np.ceil(h-0.5).astype(int), HKL)
    if update_table:
        refls['miller_index'] = flex.miller_index(len(refls),(0,0,0))
        mil_idx = flex.vec3_int(tuple(map(tuple, np.vstack(HKLi).T)))
        for i in range(len(refls)):
            refls['miller_index'][i] = mil_idx[i]
    if returnQ:
        return np.vstack(HKL).T, np.vstack(HKLi).T, q_vecs
    else:
        return np.vstack(HKL).T, np.vstack(HKLi).T


def tilt_fit(bbox_expanse, photon_gain, sigma_rdout, zinger_zscore,
             exper, predicted_refls, minsnr=None, plot=False, **kwargs):

    refls.centroid_px_to_mm(exper.detector)
    refls.map_centroids_to_reciprocal_space(exper.detector, exper.beam)
    refls['id'] = flex.int(len(refls), -1)
    refls['imageset_id'] = flex.int(len(refls), 0)
    ss_dim, fs_dim = imgs[0].shape
    n_refl = len(predicted_refls)
    integrations = []
    variances = []
    coeffs = []
    new_shoeboxes = []
    for i_ref, ref in enumerate(predicted_refls):
        i1_, i2_, j1_, j2_, _, _ = ref['bbox']  # bbox of prediction
        # expand bbox a bit to include more bg pix
        # trim bbox if its along edge of detector
        i1 = max(i1_ - bbox_expanse, 0)
        i2 = min(i2_ + bbox_expanse, fs_dim)
        j1 = max(j1_ - bbox_expanse, 0)
        j2 = min(j2_ + bbox_expanse, ss_dim)

        # which detector panel am I on ?
        i_panel = ref['panel']


        # get the iamge and mask
        shoebox_img = imgs[i_panel][j1:j2, i1:i2] / photon_gain  # NOTE: gain is imortant here!
        dials_mask = np.zeros(shoebox_img.shape).astype(np.int32)

        # initially all pixels are valie
        dials_mask += MaskCode.Valid
        shoebox_mask = is_bg_pix[i_panel, j1:j2, i1:i2]

        dials_mask[shoebox_mask] = dials_mask[shoebox_mask] + MaskCode.Background

        new_shoebox = Shoebox((i1, i2, j1, j2, 0, 1))
        new_shoebox.allocate()
        new_shoebox.data = flex.float(shoebox_img[None,])

        # get coordinates arrays of the image
        Y, X = np.indices(shoebox_img.shape)

        # determine if any more outliers are present in background pixels
        img1d = shoebox_img.ravel()
        mask1d = shoebox_mask.ravel()  # mask specifies which pixels are bg
        # out1d specifies which bg pixels are outliers (zingers)
        out1d = np.zeros(mask1d.shape, bool)
        out1d[mask1d] = is_outlier(img1d[mask1d].ravel(), zinger_zscore)
        out2d = out1d.reshape(shoebox_img.shape)

        # these are points we fit to: both zingers and original mask
        fit_sel = np.logical_and(~out2d, shoebox_mask)  # fit plane to these points, no outliers, no masked
        # update the dials mask...
        dials_mask[fit_sel] = dials_mask[fit_sel] + MaskCode.BackgroundUsed

        # fast scan pixels, slow scan pixels, pixel values (corrected for gain)
        fast, slow, rho_bg = X[fit_sel], Y[fit_sel], shoebox_img[fit_sel]

        # do the fit of the background plane
        A = np.array([fast, slow, np.ones_like(fast)]).T

        # weights matrix:
        W = np.diag(1 / (sigma_rdout ** 2 + rho_bg))
        AWA = np.dot(A.T, np.dot(W, A))
        AWA_inv = np.linalg.inv(AWA)
        AtW = np.dot(A.T, W)
        a, b, c = np.dot(np.dot(AWA_inv, AtW), rho_bg)
        coeffs.append((a, b, c))

        # fit of the tilt plane background
        X1d = np.ravel(X)
        Y1d = np.ravel(Y)
        background = (X1d * a + Y1d * b + c).reshape(shoebox_img.shape)
        new_shoebox.background = flex.float(background[None, ])

        # vector of residuals
        r = rho_bg - np.dot(A, (a, b, c))
        Nbg = len(rho_bg)
        Nparam = 3
        r_fact = np.dot(r.T, np.dot(W, r)) / (Nbg - Nparam)
        var_covar = AWA_inv * r_fact
        abc_var = var_covar[0][0], var_covar[1][1], var_covar[2][2]

        # place the strong spot mask in the expanded shoebox
        peak_mask = ref['shoebox'].mask.as_numpy_array()[0] == 5
        peak_mask_expanded = np.zeros_like(shoebox_mask)
        Nj, Ni = j2_ - j1_, i2_ - i1_
        jstart = j1_ - j1
        istart = i1_ - i1
        peak_mask_expanded[jstart:jstart+Nj, istart: istart+Ni] = peak_mask

        # update the dials mask
        dials_mask[peak_mask_expanded] = dials_mask[peak_mask_expanded] + MaskCode.Foreground

        p = X[peak_mask_expanded]  # fast scan coords
        q = Y[peak_mask_expanded]  # slow scan coords
        rho_peak = shoebox_img[peak_mask_expanded]  # pixel values

        Isum = np.sum(rho_peak - a*p - b*q - c)  # summed spot intensity

        var_rho_peak = sigma_rdout ** 2 + rho_peak  # include readout noise in the variance
        Ns = len(rho_peak)  # number of integrated peak pixels

        # variance propagated from tilt plane constants
        var_a_term = abc_var[0] * ((np.sum(p))**2)
        var_b_term = abc_var[1] * ((np.sum(q))**2)
        var_c_term = abc_var[2] * (Ns**2)
        # total variance of the spot
        var_Isum = np.sum(var_rho_peak) + var_a_term + var_b_term + var_c_term

        integrations.append(Isum)
        variances.append(var_Isum)
        new_shoebox.mask = flex.int(dials_mask[None,])
        new_shoeboxes.append(new_shoebox)

        if i_ref % 50 == 0:
            print("Integrated refls %d / %d" % (i_ref+1, n_refl))

    integrations = np.array(integrations)
    variances = np.array(variances)
    refls["intensity.sum.value.Leslie99"] = flex.double(integrations)
    refls["intensity.sum.variance.Leslie99"] = flex.double(variances)
    refls['shoebox'] = flex.shoebox(new_shoeboxes)

    # assign miller indices to the spots
    #_=refls_to_hkl(refls, detector=exper.detector, beam=exper.beam,
    #             crystal=exper.crystal, update_table=True)
    El = ExperimentList()
    El.append(exper)
    idx_assign = assign_indices.AssignIndicesGlobal(tolerance=0.333)
    idx_assign(refls, El)

    if plot:
        assert args.minsnr is not None
        all_xx = []
        all_yy = []
        for i_panel in range(len(imgs)):
            node = exper.detector[i_panel]
            pix = node.get_pixel_size()[0]
            fs = np.array(node.get_fast_axis())
            ss = np.array(node.get_slow_axis())
            o = np.array(node.get_origin()) / pix

            Nfs, Nss = node.get_image_size()
            Y, X = np.indices((Nss, Nfs))
            xx = fs[0] * X + ss[0] * Y + o[0]
            yy = fs[1] * X + ss[1] * Y + o[1]
            all_xx.append(xx)
            all_yy.append(yy)
        all_xx = np.array(all_xx)
        all_yy = np.array(all_yy)

        binsX = np.arange(int(np.min(all_xx)), int(np.max(all_xx)), 1)
        binsY = np.arange(int(np.min(all_yy)), int(np.max(all_yy)), 1)
        out = np.histogram2d(all_xx.ravel(), all_yy.ravel(), bins=(binsX, binsY), weights=imgs.ravel())


        r_sz = 5
        r_color = 'Limegreen'

        rej_sz = 5
        rej_color = 'Darkorange'
        rej_patches = []
        r_patches = []
        snr = integrations / np.sqrt(variances)
        for i_r in range(len(predicted_refls)):
            ref = predicted_refls[i_r]
            node = exper.detector[ref['panel']]
            pixsize = node.get_pixel_size()[0]
            xpix, ypix, _ = ref['xyzobs.px.value']

            xlab, ylab, _ = np.array(node.get_pixel_lab_coord((xpix, ypix)))/pixsize

            if snr[i_r] >= minsnr:
                r_rect = plt.Rectangle(xy=(ylab - r_sz, xlab - r_sz), width=2 * r_sz,
                                       height=2 * r_sz, ec=r_color, fc='None')
                r_patches.append(r_rect)

            else:
                rect = plt.Rectangle(xy=(ylab-rej_sz, xlab-rej_sz), width=2*rej_sz,
                                 height=2*rej_sz, ec=rej_color, fc='None')
                rej_patches.append(rect)

        plt.clf()
        plt.imshow(out[0], extent=(out[2][0], out[2][-1], out[1][-1], out[1][0]), **kwargs)
        coll = plt.mpl.collections.PatchCollection(rej_patches, match_original=True)
        r_coll = plt.mpl.collections.PatchCollection(r_patches, match_original=True)
        plt.gca().add_collection(coll)
        plt.gca().add_collection(r_coll)
        plt.suptitle("Green: SNR >= %.2f, Orange:SNR < %.2f" % (minsnr, minsnr))
        plt.show()

    return refls, coeffs, integrations, variances


if __name__ == "__main__":
    expand_fact = 7
    GAIN = 28
    sigma_readout = 3
    outlier_Z = np.inf
    data = h5py.File("img_and_spotmask.h5py", "r")
    is_bg_pix = data["is_background_pixel"][()]  # boolean tells if pixel is bg
    imgs = data["panel_imgs"][()]
    El = ExperimentListFactory.from_json_file('idx-dalinar_rank0_data0_fluence0.h5_refined.expt', check_format=False)
    refls = flex.reflection_table.from_file("R")
    # first step is to enlarge the shoebox region

    #for r in refls:
    #    sb.deallocate()
    experiment = El[0]
    tilt_fit(
        expand_fact, GAIN, sigma_readout, outlier_Z,
        experiment, refls, minsnr=args.minsnr, plot=True,
        vmin=args.vmin, vmax=args.vmax)
