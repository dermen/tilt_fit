

import numpy as np
from IPython import embed
import h5py
import pylab as plt
from scipy.spatial import cKDTree
from IPython import embed

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


def tilt_fit(imgs, is_bg_pix, delta_q, photon_gain, sigma_rdout, zinger_zscore,
             exper, predicted_refls, sb_pad=0, filter_boundary_spots=False,
             minsnr=None, mintilt=None, plot=False, verbose=False, is_BAD_pix=None,
             min_strong=None, min_bg=10, min_dist_to_bad_pix=7, **kwargs):

    if is_BAD_pix is None:
        is_BAD_pix = np.zeros(is_bg_pix.shape, np.bool)

    predicted_refls['id'] = flex.int(len(predicted_refls), -1)
    predicted_refls['imageset_id'] = flex.int(len(predicted_refls), 0)
    El = ExperimentList()
    El.append(exper)
    predicted_refls.centroid_px_to_mm(El)
    predicted_refls.map_centroids_to_reciprocal_space(El)
    ss_dim, fs_dim = imgs[0].shape
    n_refl = len(predicted_refls)
    integrations = []
    variances = []
    coeffs = []
    new_shoeboxes = []
    tilt_error = []
    boundary = []
    detdist = exper.detector[0].get_distance()
    pixsize = exper.detector[0].get_pixel_size()[0]
    ave_wave = exper.beam.get_wavelength()

    bad_trees = {}
    unique_panels = set(predicted_refls["panel"])
    for p in unique_panels:
        panel_bad_pix = is_BAD_pix[p]
        ybad, xbad = np.where(is_BAD_pix[0])
        if ybad.size:
            bad_pts = zip(ybad, xbad)
            bad_trees[p] = cKDTree(bad_pts)
        else:
            bad_trees[p] = None

    sel = []

    for i_ref in range(len(predicted_refls)):
        ref = predicted_refls[i_ref]
        i_com, j_com, _ = ref['xyzobs.px.value']

        # which detector panel am I on ?
        i_panel = ref['panel']

        if bad_trees[i_panel] is not None:
            if bad_trees[i_panel].query_ball_point((i_com, j_com), r=min_dist_to_bad_pix):
                sel.append(False)
                integrations.append(None)
                variances.append(None)
                coeffs.append(None)
                new_shoeboxes.append(None)
                tilt_error.append(None)
                boundary.append(None)
                continue

        i1_a, i2_a, j1_a, j2_a, _, _ = ref['bbox']  # bbox of prediction

        i1_ = max(i1_a, 0)
        i2_ = min(i2_a, fs_dim-1)
        j1_ = max(j1_a, 0)
        j2_ = min(j2_a, ss_dim-1)

        # get the number of pixels spanning the box in pixels
        Qmag = 2*np.pi*np.linalg.norm(ref['rlp'])  # magnitude momentum transfer of the RLP in physicist convention
        rad1 = (detdist/pixsize) * np.tan(2*np.arcsin((Qmag-delta_q*.5)*ave_wave/4/np.pi))
        rad2 = (detdist/pixsize) * np.tan(2*np.arcsin((Qmag+delta_q*.5)*ave_wave/4/np.pi))
        bbox_extent = (rad2-rad1) / np.sqrt(2)   # rad2 - rad1 is the diagonal across the bbox
        i_com = i_com - 0.5
        j_com = j_com - 0.5
        i_low = int(i_com - bbox_extent/2.)
        i_high = int(i_com + bbox_extent/2.)
        j_low = int(j_com - bbox_extent/2.)
        j_high = int(j_com + bbox_extent/2.)

        i1_orig = max(i_low, 0)
        i2_orig = min(i_high, fs_dim-1)
        j1_orig = max(j_low, 0)
        j2_orig = min(j_high, ss_dim-1)

        i_low = i_low - sb_pad
        i_high = i_high + sb_pad
        j_low = j_low - sb_pad
        j_high = j_high + sb_pad

        i1 = max(i_low, 0)
        i2 = min(i_high, fs_dim-1)
        j1 = max(j_low, 0)
        j2 = min(j_high, ss_dim-1)

        i1_p = i1_orig - i1
        i2_p = i1_p + i2_orig-i1_orig
        j1_p = j1_orig - j1
        j2_p = j1_p + j2_orig-j1_orig

        if i1 == 0 or i2 == fs_dim or j1 == 0 or j2 == ss_dim:
            boundary.append(True)
            if filter_boundary_spots:
                sel.append(False)
                integrations.append(None)
                variances.append(None)
                coeffs.append(None)
                new_shoeboxes.append(None)
                tilt_error.append(None)
                continue
        else:
            boundary.append(False)

        # get the iamge and mask
        shoebox_img = imgs[i_panel][j1:j2, i1:i2] / photon_gain  # NOTE: gain is imortant here!
        dials_mask = np.zeros(shoebox_img.shape).astype(np.int32)

        # initially all pixels are valid
        dials_mask += MaskCode.Valid
        shoebox_mask = is_bg_pix[i_panel][j1:j2, i1:i2]

        badpix_mask = is_BAD_pix[i_panel, j1:j2, i1:i2]

        dials_mask[shoebox_mask] = dials_mask[shoebox_mask] + MaskCode.Background

        new_shoebox = Shoebox((i1_orig, i2_orig, j1_orig, j2_orig, 0, 1))
        new_shoebox.allocate()
        new_shoebox.data = flex.float(np.ascontiguousarray(shoebox_img[None, j1_p:j2_p, i1_p: i2_p]))
        #new_shoebox.data = flex.float(shoebox_img[None,])

        # get coordinates arrays of the image
        Y, X = np.indices(shoebox_img.shape)

        # determine if any more outliers are present in background pixels
        img1d = shoebox_img.ravel()
        mask1d = shoebox_mask.ravel()  # mask specifies which pixels are bg
        # out1d specifies which bg pixels are outliers (zingers)
        out1d = np.zeros(mask1d.shape, bool)
        out1d[mask1d] = is_outlier(img1d[mask1d].ravel(), zinger_zscore)
        out2d = out1d.reshape(shoebox_img.shape)

        # combine bad2d with badpix mask
        out2d = np.logical_or(out2d, badpix_mask)

        # these are points we fit to: both zingers and original mask
        fit_sel = np.logical_and(~out2d, shoebox_mask)  # fit plane to these points, no outliers, no masked

        if np.sum(fit_sel) < min_bg:
            integrations.append(None)
            variances.append(None)
            coeffs.append(None)
            new_shoeboxes.append(None)
            tilt_error.append(None)
            sel.append(False)
            continue

        # update the dials mask...
        dials_mask[fit_sel] = dials_mask[fit_sel] + MaskCode.BackgroundUsed

        # fast scan pixels, slow scan pixels, pixel values (corrected for gain)
        fast, slow, rho_bg = X[fit_sel], Y[fit_sel], shoebox_img[fit_sel]

        # do the fit of the background plane
        A = np.array([fast, slow, np.ones_like(fast)]).T
        # weights matrix:
        W = np.diag(1 / (sigma_rdout ** 2 + rho_bg))
        AWA = np.dot(A.T, np.dot(W, A))
        try:
            AWA_inv = np.linalg.inv(AWA)
        except np.linalg.LinAlgError:
            print ( "WARNING: Fit did not work.. investigate reflection")
            print (ref)
            integrations.append(None)
            variances.append(None)
            coeffs.append(None)
            new_shoeboxes.append(None)
            tilt_error.append(None)
            sel.append(False)
            continue


        AtW = np.dot(A.T, W)
        a, b, c = np.dot(np.dot(AWA_inv, AtW), rho_bg)
        coeffs.append((a, b, c))

        # fit of the tilt plane background
        X1d = np.ravel(X)
        Y1d = np.ravel(Y)
        background = (X1d * a + Y1d * b + c).reshape(shoebox_img.shape)
        new_shoebox.background = flex.float(np.ascontiguousarray(background[None, j1_p: j2_p, i1_p:i2_p]))

        # vector of residuals
        r = rho_bg - np.dot(A, (a, b, c))
        Nbg = len(rho_bg)
        Nparam = 3
        r_fact = np.dot(r.T, np.dot(W, r)) / (Nbg - Nparam)
        var_covar = AWA_inv * r_fact
        abc_var = var_covar[0][0], var_covar[1][1], var_covar[2][2]

        # place the strong spot mask in the expanded shoebox
        peak_mask = ref['shoebox'].mask.as_numpy_array()[0] == MaskCode.Valid + MaskCode.Foreground
        peak_mask_valid = peak_mask[j1_-j1_a:- j1_a + j2_, i1_-i1_a:-i1_a + i2_]
        peak_mask_expanded = np.zeros_like(shoebox_mask)

        # overlap region
        i1_o = max(i1_, i1)
        i2_o = min(i2_, i2)
        j1_o = max(j1_, j1)
        j2_o = min(j2_, j2)

        pk_mask_istart = i1_o - i1_
        pk_mask_jstart = j1_o - j1_
        pk_mask_istop = peak_mask_valid.shape[1] - (i2_ - i2_o)
        pk_mask_jstop = peak_mask_valid.shape[0] - (j2_ - j2_o)
        peak_mask_overlap = peak_mask_valid[pk_mask_jstart: pk_mask_jstop, pk_mask_istart: pk_mask_istop]

        pk_mask_exp_i1 = i1_o - i1
        pk_mask_exp_j1 = j1_o - j1
        pk_mask_exp_i2 = peak_mask_expanded.shape[1] - (i2 - i2_o)
        pk_mask_exp_j2 = peak_mask_expanded.shape[0] - (j2 - j2_o)
        peak_mask_expanded[pk_mask_exp_j1: pk_mask_exp_j2, pk_mask_exp_i1: pk_mask_exp_i2] = peak_mask_overlap

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
        tilt_error.append(var_a_term + var_b_term + var_c_term)

        # total variance of the spot
        var_Isum = np.sum(var_rho_peak) + var_a_term + var_b_term + var_c_term

        integrations.append(Isum)
        variances.append(var_Isum)
        new_shoebox.mask = flex.int(np.ascontiguousarray(dials_mask[None, j1_p:j2_p, i1_p:i2_p]))
        new_shoeboxes.append(new_shoebox)
        sel.append(True)

        if i_ref % 50 == 0 and verbose:
            print("Integrated refls %d / %d" % (i_ref+1, n_refl))


    #if filter_boundary_spots:
    #    sel = flex.bool([I is not None for I in integrations])
    boundary = np.array(boundary)[sel].astype(bool)
    integrations = np.array([I for I in integrations if I is not None])
    variances = np.array([v for v in variances if v is not None])
    coeffs = np.array([c for c in coeffs if c is not None])
    tilt_error = np.array([te for te in tilt_error if te is not None])

    #boundary = np.zeros(tilt_error.shape).astype(np.bool)

    predicted_refls = predicted_refls.select(flex.bool(sel))

    predicted_refls['resolution'] = flex.double( 1/ np.linalg.norm(predicted_refls['rlp'], axis=1))

    predicted_refls['boundary'] = flex.bool(boundary)
    predicted_refls["intensity.sum.value.Leslie99"] = flex.double(integrations)
    predicted_refls["intensity.sum.variance.Leslie99"] = flex.double(variances)
    predicted_refls['shoebox'] = flex.shoebox([sb for sb in new_shoeboxes if sb is not None])
    idx_assign = assign_indices.AssignIndicesGlobal(tolerance=0.333)
    idx_assign(predicted_refls, El)

    if plot:
        assert minsnr is not None
        assert mintilt is not None
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

        r_color = 'Limegreen'
        rej_color = 'Darkorange'
        rej_patches = []
        r_patches = []
        snr = integrations / np.sqrt(variances)
        for i_r in range(len(predicted_refls)):

            ref = predicted_refls[i_r]

            node = exper.detector[ref['panel']]
            pixsize = node.get_pixel_size()[0]
            xpix, ypix, _ = ref['xyzobs.px.value']
            xpix = xpix - 0.5
            ypix = ypix - 0.5
            xlab, ylab, _ = np.array(node.get_pixel_lab_coord((xpix, ypix)))/pixsize

            # FIXME: something weird with the display on CSPADs, to do with fast /slow scan vectors?
            i1, i2, j1, j2, _, _ = ref['shoebox'].bbox
            width = i2-i1
            height = j2-j1
            if snr[i_r] >= minsnr and tilt_error[i_r] <= mintilt:
                r_rect = plt.Rectangle(xy=(ylab - height/2., xlab - width/2.), width=width,
                                       height=height, ec=r_color, fc='None')
                r_patches.append(r_rect)

            else:
                rect = plt.Rectangle(xy=(ylab-height/2., xlab-width/2.), width=width,
                                 height=height, ec=rej_color, fc='None')
                rej_patches.append(rect)

        plt.clf()
        plt.imshow(out[0], extent=(out[2][0], out[2][-1], out[1][-1], out[1][0]), **kwargs)
        coll = plt.mpl.collections.PatchCollection(rej_patches, match_original=True)
        r_coll = plt.mpl.collections.PatchCollection(r_patches, match_original=True)
        plt.gca().add_collection(coll)
        plt.gca().add_collection(r_coll)
        plt.suptitle("Green: SNR >= %.2f, Orange:SNR < %.2f" % (minsnr, minsnr))
        plt.show()

    return predicted_refls, coeffs, tilt_error, integrations, variances


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser("tilt filt")
    parser.add_argument("--filterboundaryspots", action='store_true')
    parser.add_argument("--minsnr", default=1.1, type=float)
    parser.add_argument("--mintilt", default=250, type=float)
    parser.add_argument("--vmax", default=None, type=float)
    parser.add_argument("--vmin", default=None, type=float)
    args = parser.parse_args()
    GAIN = 28
    sigma_readout = 3
    outlier_Z = np.inf
    data = h5py.File("img_and_spotmask.h5py", "r")
    is_bg_pix = data["is_background_pixel"][()]  # boolean tells if pixel is bg
    imgs = data["panel_imgs"][()]
    El = ExperimentListFactory.from_json_file('idx-dalinar_rank0_data0_fluence0.h5_refined.expt', check_format=False)
    refls = flex.reflection_table.from_file("R")

    res = tilt_fit(
        imgs=imgs, is_bg_pix=is_bg_pix,
        delta_q=0.07, photon_gain=GAIN, sigma_rdout=sigma_readout, zinger_zscore=outlier_Z,
        exper=El[0], predicted_refls=refls, sb_pad=5, filter_boundary_spots=args.filterboundaryspots,
        minsnr=args.minsnr, mintilt=args.mintilt, plot=True, vmin=args.vmin, vmax=args.vmax)
