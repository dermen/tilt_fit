

import numpy as np
from IPython import embed
import h5py
import pylab as plt
from scipy.spatial import cKDTree

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


class TiltPlanes:

    def __init__(self, panel_imgs, panel_bg_masks, panel_badpix_masks=None):   #, photon_gain, sigma_rdout, zinger_zscore):
        self.panel_imgs = panel_imgs
        self.sdim, self.fdim = self.panel_imgs[0].shape

        self.panel_bg_masks = panel_bg_masks
        self.panel_badpix_masks = panel_badpix_masks
        if self.panel_badpix_masks is None:
            self.panel_badpix_masks = np.zeros(self.panel_imgs.shape, dtype=bool)

        self._bad_pixel_proximity_tester = None  # tests whether the reflection is close to a bad pixel
        #self._make_quick_bad_pixel_proximity_checker()


        self.min_background_pix_for_fit = 10
        self.detdist_mm = None
        self.pixsize_mm = None
        self.ave_wavelength_A = None
        self.i_com=self.j_com=self.pid = None
        self.adu_per_photon = 1
        self.delta_Q = 0.06  # inverse Angstrom
        self.sigma_rdout = 3  # readout noise of the pixel
        self.zinger_zscore = 5
        self.tilt_dips_below_zero = False  # during integration we check if tilt plane dips below 0 and set this flag
        self.min_dist_to_bad_pix = 7


    def make_quick_bad_pixel_proximity_checker(self, predicted_refls):
        self._bad_pixel_proximity_tester = {}
        unique_panels = set(predicted_refls["panel"])
        for p in unique_panels:
            ybad, xbad = np.where(self.panel_badpix_masks[0])
            if ybad.size:
                bad_pts = list(zip(ybad, xbad))
                self._bad_pixel_proximity_tester[p] = cKDTree(bad_pts)
            else:
                self._bad_pixel_proximity_tester[p] = None

    @property
    def sigma_rdout(self):
        """readout noise in ADU"""
        return self._sigma_rdout

    @sigma_rdout.setter
    def sigma_rdout(self, val):
        self._sigma_rdout = val

    @property
    def adu_per_photon(self):
        """readout noise in ADU"""
        return self._adu_per_photon

    @adu_per_photon.setter
    def adu_per_photon(self, val):
        self._adu_per_photon = val

    @property
    def zinger_zscore(self):
        """MAD Zscore for a pixel to be flagged as a zinger outlier"""
        return self._zinger_zscore

    @zinger_zscore.setter
    def zinger_zscore(self, val):
        self._zinger_zscore = val

    @property
    def min_dist_to_bad_pix(self):
        """if a reflection is this many pixels from a pixel in the bad pixel mask, then it wont be integrated"""
        return self._min_dist_to_bad_pix

    @min_dist_to_bad_pix.setter
    def min_dist_to_bad_pix(self, val):
        self._min_dist_to_bad_pix = val

    @property
    def pixsize_mm(self):
        return self._pixsize_mm

    @pixsize_mm.setter
    def pixsize_mm(self, val):
        self._pixsize_mm = val

    @property
    def detdist_mm(self):
        return self._detdist_mm

    @detdist_mm.setter
    def detdist_mm(self, val):
        self._detdist_mm = val

    @property
    def delta_Q(self):
        return self._delta_Q

    @delta_Q.setter
    def delta_Q(self, val):
        self._delta_Q = val

    @property
    def ave_wavelength_A(self):
        return self._ave_wavelength_A

    @ave_wavelength_A.setter
    def ave_wavelength_A(self, val):
        self._ave_wavelength_A = val

    @property
    def i_com(self):
        """fast-scan direction of center of mass for shoebox being integrated"""
        return self._i_com

    @i_com.setter
    def i_com(self, val):
        self._i_com = val

    @property
    def j_com(self):
        """slow-scan direction of center of mass for shoebox being integrated"""
        return self._j_com

    @j_com.setter
    def j_com(self, val):
        self._j_com = val

    @property
    def pid(self):
        """panel id for teh shoebox being integrated"""
        return self._pid

    @pid.setter
    def pid(self, val):
        self._pid = val

    @property
    def min_background_pix_for_fit(self):
        """if shoebox has less than this many background pixels, it will be skipped and flagged as invalid"""
        return self._min_background_pix_for_fit

    @min_background_pix_for_fit.setter
    def min_background_pix_for_fit(self, val):
        self._min_background_pix_for_fit = val

    def _refl_is_close_to_bad_pixel(self):
        panel_tester = self._bad_pixel_proximity_tester[self.pid]
        is_close = False
        if panel_tester is not None:
            refl_com = self.i_com, self.j_com
            if panel_tester.query_ball_point(refl_com, r=self.min_dist_to_bad_pix):
                is_close = True
        return is_close

    def check_if_refls_are_formatted_for_this_class(self, predicted_refls):
        x1, x2, y1, y2, _, _ = zip(*predicted_refls["bbox"])
        if min(x1) < 0 or min(y1) < 0 or max(x2) > self.fdim or max(y2) > self.sdim:
            raise ValueError("The reflection bbox outside of panel")
        for key in ["panel", "rlp", "xyzobs.px.value", "bbox"]:
            if key not in predicted_refls:
                raise KeyError("expects `%s` column to be in the reflections table" % key)

    def _get_width_of_shoebox(self, rlp):
        Qmag = 2 * np.pi * np.linalg.norm(rlp)  # magnitude momentum transfer of the RLP in physicist convention
        rad1 = (self.detdist_mm / self.pixsize_mm) * np.tan(2 * np.arcsin((Qmag - self.delta_Q * .5) * self.ave_wavelength_A / 4 / np.pi))
        rad2 = (self.detdist_mm / self.pixsize_mm) * np.tan(2 * np.arcsin((Qmag + self.delta_Q * .5) * self.ave_wavelength_A / 4 / np.pi))
        bbox_extent = (rad2 - rad1) / np.sqrt(2)  # rad2 - rad1 is the diagonal across the bbox
        return bbox_extent

    def determine_shoebox_ROI(self):
        width = self._get_width_of_shoebox(self.refl["rlp"])
        wby2 = int(round(width/2.))
        i1, i2, j1, j2 = self.i_com - wby2, self.i_com + wby2, self.j_com - wby2, self.j_com + wby2
        i1 = max(int(i1), 0)
        i2 = min(int(i2), self.fdim - 1)
        j1 = max(int(j1), 0)
        j2 = min(int(j2), self.sdim - 1)
        return i1, i2, j1, j2

    def _search_for_zingers_in_background_pixels(self):
        # determine if any more outliers are present in background pixels
        img1d = self.shoebox_img.ravel()
        is_bg_1d = self.is_background_pixel.ravel()  # mask specifies which pixels are bg

        # out1d specifies which bg pixels are outliers (zingers)
        out1d = np.zeros(img1d.shape, bool)
        out1d[is_bg_1d] = is_outlier(img1d[is_bg_1d].ravel(), self.zinger_zscore)

        # out2d is out1d in its 2D shape
        out2d = out1d.reshape(self.shoebox_img.shape)
        return out2d

    def fit_background_pixels_to_plane(self):
        # fast scan pixels, slow scan pixels, pixel values (corrected for gain)
        fast, slow, rho_bg = self.X[self.fit_sel], self.Y[self.fit_sel], self.shoebox_img[self.fit_sel]

        # do the fit of the background plane
        A = np.array([fast, slow, np.ones_like(fast)]).T
        # weights matrix:
        W = np.diag(1 / (self.sigma_rdout ** 2 + rho_bg))
        AWA = np.dot(A.T, np.dot(W, A))
        try:
            AWA_inv = np.linalg.inv(AWA)
        except np.linalg.LinAlgError:
            print("WARNING: Fit did not work.. investigate reflection")
            print(self.refl)
            return None
        AtW = np.dot(A.T, W)
        t1, t2, t3 = np.dot(np.dot(AWA_inv, AtW), rho_bg)

        # get the covariance of the tilt coefficients:
        # vector of residuals
        r = rho_bg - np.dot(A, (t1, t2, t3))
        Nbg = len(rho_bg)
        r_fact = np.dot(r.T, np.dot(W, r)) / (Nbg - 3)  # 3 parameters fit
        var_covar = AWA_inv * r_fact  # TODO: check for correlations in the off diagonal elems

        return (t1, t2, t3), var_covar

    def _integrate(self):
        is_bragg_peak = np.logical_not(self.is_background_pixel)

        p = self.X[is_bragg_peak]  # fast scan coords
        q = self.Y[is_bragg_peak]  # slow scan coords
        rho_peak = self.shoebox_img[is_bragg_peak]  # pixel values

        t1, t2, t3 = self.coefs
        Isum = np.sum(rho_peak - t1*p - t2*q - t3)  # summed spot intensity

        var_rho_peak = (self.sigma_rdout/self.adu_per_photon) ** 2 + rho_peak  # include readout noise in the variance
        Ns = len(rho_peak)  # number of integrated peak pixels

        # variance propagated from tilt plane constants
        var_a_term = self.variance_matrix[0, 0] * ((np.sum(p)) ** 2)
        var_b_term = self.variance_matrix[1, 1] * ((np.sum(q)) ** 2)
        var_c_term = self.variance_matrix[2, 2] * (Ns ** 2)

        # total variance of the spot
        var_Isum = np.sum(var_rho_peak) + var_a_term + var_b_term + var_c_term

        return Isum, var_Isum

    def _check_if_tilt_dips_below_zero(self):
        tX, tY, tZ = self.coefs
        tilt_plane = self.X*tX + self.Y*tY + tZ
        if np.min(tilt_plane) < 0:
            self.tilt_dips_below_zero = True
        else:
            self.tilt_dips_below_zero = False

    def integrate_shoebox(self, refl):
        if self.pixsize_mm is None:
            print("Set pixel size first")
            return
        if self.detdist_mm is None:
            print("Set detdist fist")
            return
        if self.ave_wavelength_A is None:
            print("set wavelen first")
            return

        self.refl = refl

        # get the center of mass of the reflection
        self.i_com, self.j_com, _ = map(lambda x: x-0.5, refl['xyzobs.px.value'])

        # which detector panel am I on ?
        self.pid = refl['panel']

        if self._refl_is_close_to_bad_pixel():
            print("Close to bad pix!")
            return None

        #integration_window = refl["bbox"]
        shoebox_roi = self.determine_shoebox_ROI()
        print(shoebox_roi[1]-shoebox_roi[0], shoebox_roi[3] - shoebox_roi[2])
        sX = slice(shoebox_roi[0], shoebox_roi[1], 1)
        sY = slice(shoebox_roi[2], shoebox_roi[3], 1)

        # get the  ROI image in photon units, as well as the
        self.shoebox_img = self.panel_imgs[self.pid][sY, sX] / self.adu_per_photon
        self.is_background_pixel = self.panel_bg_masks[self.pid][sY, sX]
        is_bad_pixel = self.panel_badpix_masks[self.pid][sY, sX]

        # get coordinates arrays of the image
        self.Y, self.X = np.indices(self.shoebox_img.shape)
        #self.Y, self.X = np.meshgrid(range(shoebox_roi[0], shoebox_roi[1]), range(shoebox_roi[2], shoebox_roi[3]))

        is_zinger = self._search_for_zingers_in_background_pixels()

        # combine zingers with badpix mask
        not_for_fitting = np.logical_or(is_zinger, is_bad_pixel)
        is_for_fitting = np.logical_not(not_for_fitting)

        # these are points we fit to: both zingers and original mask
        self.fit_sel = np.logical_and(is_for_fitting, self.is_background_pixel)

        nfitpix = np.sum(self.fit_sel)
        if nfitpix < self.min_background_pix_for_fit:
            print("Not enough pixels for fitting (%d background pix)" % nfitpix)
            return None


        # fit of the tilt plane background
        self.coefs, self.variance_matrix = self.fit_background_pixels_to_plane()

        Isum, varIsum = self._integrate()

        self._check_if_tilt_dips_below_zero()

        # returns 5 objects:
        #  -the shoebox ROI (fast1,fast2,slow1,slow2)
        #  -the tilt coefficients (t1,t2,t3) where t1 is for fast coord, t2 is for slow coord, t3 is height)
        #  -the variance matrix from the least squares determination of the coefficients
        #  -the integrated I (summed intensity of Bragg peak)
        #  -the variance of integrated I
        return shoebox_roi, self.coefs, self.variance_matrix, Isum, varIsum, self.tilt_dips_below_zero

    @staticmethod
    def prep_relfs_for_tiltalization(predicted_refls, exper):
        predicted_refls['id'] = flex.int(len(predicted_refls), -1)
        predicted_refls['imageset_id'] = flex.int(len(predicted_refls), 0)
        El = ExperimentList()
        El.append(exper)
        predicted_refls.centroid_px_to_mm(El)
        predicted_refls.map_centroids_to_reciprocal_space(El)
        idx_assign = assign_indices.AssignIndicesGlobal(tolerance=0.333)
        idx_assign(predicted_refls, El)
        return predicted_refls


#def tilt_fit(imgs, is_bg_pix, delta_q, photon_gain, sigma_rdout, zinger_zscore,
#             exper, predicted_refls, sb_pad=0, filter_boundary_spots=False,
#             minsnr=None, mintilt=None, plot=False, verbose=False, is_BAD_pix=None,
#             min_strong=None, min_bg=10, min_dist_to_bad_pix=7, **kwargs):
#
#    if is_BAD_pix is None:
#        is_BAD_pix = np.zeros(np.array(is_bg_pix).shape, np.bool)
#
#    predicted_refls['id'] = flex.int(len(predicted_refls), -1)
#    predicted_refls['imageset_id'] = flex.int(len(predicted_refls), 0)
#    El = ExperimentList()
#    El.append(exper)
#    predicted_refls.centroid_px_to_mm(El)
#    predicted_refls.map_centroids_to_reciprocal_space(El)
#    ss_dim, fs_dim = imgs[0].shape
#    n_refl = len(predicted_refls)
#    integrations = []
#    variances = []
#    coeffs = []
#    new_shoeboxes = []
#    tilt_error = []
#    boundary = []
#    detdist = exper.detector[0].get_distance()
#    pixsize = exper.detector[0].get_pixel_size()[0]
#    ave_wave = exper.beam.get_wavelength()
#
#    bad_trees = {}
#    unique_panels = set(predicted_refls["panel"])
#    for p in unique_panels:
#        panel_bad_pix = is_BAD_pix[p]
#        ybad, xbad = np.where(is_BAD_pix[0])
#        if ybad.size:
#            bad_pts = zip(ybad, xbad)
#            bad_trees[p] = cKDTree(bad_pts)
#        else:
#            bad_trees[p] = None
#
#    sel = []
#    for i_ref in range(len(predicted_refls)):
#        ref = predicted_refls[i_ref]
#        i_com, j_com, _ = ref['xyzobs.px.value']
#
#        # which detector panel am I on ?
#        i_panel = ref['panel']
#
#        if bad_trees[i_panel] is not None:
#            if bad_trees[i_panel].query_ball_point((i_com, j_com), r=min_dist_to_bad_pix):
#                sel.append(False)
#                integrations.append(None)
#                variances.append(None)
#                coeffs.append(None)
#                new_shoeboxes.append(None)
#                tilt_error.append(None)
#                boundary.append(None)
#                continue
#
#        i1_a, i2_a, j1_a, j2_a, _, _ = ref['bbox']  # bbox of prediction
#
#        i1_ = max(i1_a, 0)
#        i2_ = min(i2_a, fs_dim-1)
#        j1_ = max(j1_a, 0)
#        j2_ = min(j2_a, ss_dim-1)
#
#        # get the number of pixels spanning the box in pixels
#        Qmag = 2*np.pi*np.linalg.norm(ref['rlp'])  # magnitude momentum transfer of the RLP in physicist convention
#        rad1 = (detdist/pixsize) * np.tan(2*np.arcsin((Qmag-delta_q*.5)*ave_wave/4/np.pi))
#        rad2 = (detdist/pixsize) * np.tan(2*np.arcsin((Qmag+delta_q*.5)*ave_wave/4/np.pi))
#        bbox_extent = (rad2-rad1) / np.sqrt(2)   # rad2 - rad1 is the diagonal across the bbox
#        i_com = i_com - 0.5
#        j_com = j_com - 0.5
#        i_low = int(i_com - bbox_extent/2.)
#        i_high = int(i_com + bbox_extent/2.)
#        j_low = int(j_com - bbox_extent/2.)
#        j_high = int(j_com + bbox_extent/2.)
#
#        i1_orig = max(i_low, 0)
#        i2_orig = min(i_high, fs_dim-1)
#        j1_orig = max(j_low, 0)
#        j2_orig = min(j_high, ss_dim-1)
#
#        i_low = i_low - sb_pad
#        i_high = i_high + sb_pad
#        j_low = j_low - sb_pad
#        j_high = j_high + sb_pad
#
#        i1 = max(i_low, 0)
#        i2 = min(i_high, fs_dim-1)
#        j1 = max(j_low, 0)
#        j2 = min(j_high, ss_dim-1)
#
#        i1_p = i1_orig - i1
#        i2_p = i1_p + i2_orig-i1_orig
#        j1_p = j1_orig - j1
#        j2_p = j1_p + j2_orig-j1_orig
#
#        if i1 == 0 or i2 == fs_dim or j1 == 0 or j2 == ss_dim:
#            boundary.append(True)
#            if filter_boundary_spots:
#                sel.append(False)
#                integrations.append(None)
#                variances.append(None)
#                coeffs.append(None)
#                new_shoeboxes.append(None)
#                tilt_error.append(None)
#                continue
#        else:
#            boundary.append(False)
#
#        # get the iamge and mask
#        shoebox_img = imgs[i_panel][j1:j2, i1:i2] / photon_gain  # NOTE: gain is imortant here!
#        dials_mask = np.zeros(shoebox_img.shape).astype(np.int32)
#
#        # initially all pixels are valid
#        dials_mask += MaskCode.Valid
#        shoebox_mask = is_bg_pix[i_panel][j1:j2, i1:i2]
#
#
#        dials_mask[shoebox_mask] = dials_mask[shoebox_mask] + MaskCode.Background
#
#        new_shoebox = Shoebox((i1_orig, i2_orig, j1_orig, j2_orig, 0, 1))
#        new_shoebox.allocate()
#        new_shoebox.data = flex.float(np.ascontiguousarray(shoebox_img[None, j1_p:j2_p, i1_p: i2_p]))
#        #new_shoebox.data = flex.float(shoebox_img[None,])
#
#        # get coordinates arrays of the image
#        Y, X = np.indices(shoebox_img.shape)
#
#        # determine if any more outliers are present in background pixels
#        img1d = shoebox_img.ravel()
#        mask1d = shoebox_mask.ravel()  # mask specifies which pixels are bg
#        # out1d specifies which bg pixels are outliers (zingers)
#        out1d = np.zeros(mask1d.shape, bool)
#        out1d[mask1d] = is_outlier(img1d[mask1d].ravel(), zinger_zscore)
#        out2d = out1d.reshape(shoebox_img.shape)
#
#        # combine bad2d with badpix mask
#        out2d = np.logical_or(out2d, badpix_mask)
#
#        # these are points we fit to: both zingers and original mask
#        fit_sel = np.logical_and(~out2d, shoebox_mask)  # fit plane to these points, no outliers, no masked
#
#        if np.sum(fit_sel) < min_bg:
#            integrations.append(None)
#            variances.append(None)
#            coeffs.append(None)
#            new_shoeboxes.append(None)
#            tilt_error.append(None)
#            sel.append(False)
#            continue
#
#        # update the dials mask...
#        dials_mask[fit_sel] = dials_mask[fit_sel] + MaskCode.BackgroundUsed
#
#        # fast scan pixels, slow scan pixels, pixel values (corrected for gain)
#        fast, slow, rho_bg = X[fit_sel], Y[fit_sel], shoebox_img[fit_sel]
#
#        # do the fit of the background plane
#        A = np.array([fast, slow, np.ones_like(fast)]).T
#        # weights matrix:
#        W = np.diag(1 / (sigma_rdout ** 2 + rho_bg))
#        AWA = np.dot(A.T, np.dot(W, A))
#        try:
#            AWA_inv = np.linalg.inv(AWA)
#        except np.linalg.LinAlgError:
#            print ("WARNING: Fit did not work.. investigate reflection")
#            print (ref)
#            integrations.append(None)
#            variances.append(None)
#            coeffs.append(None)
#            new_shoeboxes.append(None)
#            tilt_error.append(None)
#            sel.append(False)
#            continue
#
#
#        AtW = np.dot(A.T, W)
#        a, b, c = np.dot(np.dot(AWA_inv, AtW), rho_bg)
#        coeffs.append((a, b, c))
#
#        # fit of the tilt plane background
#        X1d = np.ravel(X)
#        Y1d = np.ravel(Y)
#        background = (X1d * a + Y1d * b + c).reshape(shoebox_img.shape)
#        new_shoebox.background = flex.float(np.ascontiguousarray(background[None, j1_p: j2_p, i1_p:i2_p]))
#
#        # vector of residuals
#        r = rho_bg - np.dot(A, (a, b, c))
#        Nbg = len(rho_bg)
#        Nparam = 3
#        r_fact = np.dot(r.T, np.dot(W, r)) / (Nbg - Nparam)
#        var_covar = AWA_inv * r_fact
#        abc_var = var_covar[0][0], var_covar[1][1], var_covar[2][2]
#
#        # place the strong spot mask in the expanded shoebox
#        peak_mask = ref['shoebox'].mask.as_numpy_array()[0] == MaskCode.Valid + MaskCode.Foreground
#        peak_mask_valid = peak_mask[j1_-j1_a:- j1_a + j2_, i1_-i1_a:-i1_a + i2_]
#        peak_mask_expanded = np.zeros_like(shoebox_mask)
#
#        # overlap region
#        i1_o = max(i1_, i1)
#        i2_o = min(i2_, i2)
#        j1_o = max(j1_, j1)
#        j2_o = min(j2_, j2)
#
#        pk_mask_istart = i1_o - i1_
#        pk_mask_jstart = j1_o - j1_
#        pk_mask_istop = peak_mask_valid.shape[1] - (i2_ - i2_o)
#        pk_mask_jstop = peak_mask_valid.shape[0] - (j2_ - j2_o)
#        peak_mask_overlap = peak_mask_valid[pk_mask_jstart: pk_mask_jstop, pk_mask_istart: pk_mask_istop]
#
#        pk_mask_exp_i1 = i1_o - i1
#        pk_mask_exp_j1 = j1_o - j1
#        pk_mask_exp_i2 = peak_mask_expanded.shape[1] - (i2 - i2_o)
#        pk_mask_exp_j2 = peak_mask_expanded.shape[0] - (j2 - j2_o)
#        peak_mask_expanded[pk_mask_exp_j1: pk_mask_exp_j2, pk_mask_exp_i1: pk_mask_exp_i2] = peak_mask_overlap
#
#        # update the dials mask
#        dials_mask[peak_mask_expanded] = dials_mask[peak_mask_expanded] + MaskCode.Foreground
#
#        p = X[peak_mask_expanded]  # fast scan coords
#        q = Y[peak_mask_expanded]  # slow scan coords
#        rho_peak = shoebox_img[peak_mask_expanded]  # pixel values
#
#        Isum = np.sum(rho_peak - a*p - b*q - c)  # summed spot intensity
#
#        var_rho_peak = sigma_rdout ** 2 + rho_peak  # include readout noise in the variance
#        Ns = len(rho_peak)  # number of integrated peak pixels
#
#        # variance propagated from tilt plane constants
#        var_a_term = abc_var[0] * ((np.sum(p))**2)
#        var_b_term = abc_var[1] * ((np.sum(q))**2)
#        var_c_term = abc_var[2] * (Ns**2)
#        tilt_error.append(var_a_term + var_b_term + var_c_term)
#
#        # total variance of the spot
#        var_Isum = np.sum(var_rho_peak) + var_a_term + var_b_term + var_c_term
#
#        integrations.append(Isum)
#        variances.append(var_Isum)
#        new_shoebox.mask = flex.int(np.ascontiguousarray(dials_mask[None, j1_p:j2_p, i1_p:i2_p]))
#        new_shoeboxes.append(new_shoebox)
#        sel.append(True)
#
#        if i_ref % 50 == 0 and verbose:
#            print("Integrated refls %d / %d" % (i_ref+1, n_refl))
#
#
#    #if filter_boundary_spots:
#    #    sel = flex.bool([I is not None for I in integrations])
#    boundary = np.array(boundary)[sel].astype(bool)
#    integrations = np.array([I for I in integrations if I is not None])
#    variances = np.array([v for v in variances if v is not None])
#    coeffs = np.array([c for c in coeffs if c is not None])
#    tilt_error = np.array([te for te in tilt_error if te is not None])
#
#    #boundary = np.zeros(tilt_error.shape).astype(np.bool)
#
#    predicted_refls = predicted_refls.select(flex.bool(sel))
#
#    predicted_refls['resolution'] = flex.double( 1/ np.linalg.norm(predicted_refls['rlp'], axis=1))
#    predicted_refls['boundary'] = flex.bool(boundary)
#    predicted_refls["intensity.sum.value.Leslie99"] = flex.double(integrations)
#    predicted_refls["intensity.sum.variance.Leslie99"] = flex.double(variances)
#    predicted_refls['shoebox'] = flex.shoebox([sb for sb in new_shoeboxes if sb is not None])
#    idx_assign = assign_indices.AssignIndicesGlobal(tolerance=0.333)
#    idx_assign(predicted_refls, El)
#



if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser("tilt filt")
    parser.add_argument("--filterboundaryspots", action='store_true')
    parser.add_argument("--minsnr", default=1.1, type=float)
    parser.add_argument("--mintilt", default=10000, type=float)
    parser.add_argument("--vmax", default=250, type=float)
    parser.add_argument("--vmin", default=None, type=float)
    parser.add_argument("--pause", default=0.3, type=float)
    parser.add_argument("--plotassembled", action="store_true")
    args = parser.parse_args()
    GAIN = 28
    sigma_readout = 3
    outlier_Z = np.inf
    data = h5py.File("img_and_spotmask.h5py", "r")
    is_bg_pix = data["is_background_pixel"][()]  # boolean tells if pixel is bg
    imgs = data["panel_imgs"][()]
    El = ExperimentListFactory.from_json_file('idx-dalinar_rank0_data0_fluence0.h5_refined.expt', check_format=False)
    refls = flex.reflection_table.from_file("R")

    refls = TiltPlanes.prep_relfs_for_tiltalization(refls, exper=El[0])

    tiltnation = TiltPlanes(panel_imgs=imgs, panel_bg_masks=is_bg_pix, panel_badpix_masks=None)
    tiltnation.check_if_refls_are_formatted_for_this_class(refls)
    tiltnation.make_quick_bad_pixel_proximity_checker(refls)
    tiltnation.sigma_rdout = sigma_readout
    tiltnation.adu_per_photon = GAIN
    tiltnation.delta_Q = 0.06
    tiltnation.zinger_zscore = outlier_Z
    tiltnation.pixsize_mm = El.detectors()[0][0].get_pixel_size()[0]
    tiltnation.ave_wavelength_A = El.beams()[0].get_wavelength()
    tiltnation.detdist_mm = El.detectors()[0][0].get_distance()
    tiltnation.min_background_pix_for_fit = 10
    tiltnation.min_dist_to_bad_pix = 7

    Nrefl = len(refls)
    all_residual = []
    mins = []
    integrations = []
    variances = []
    tilt_error =[]
    from dials.array_family import flex
    sel = []
    import pandas
    panels = []
    bboxes = []
    xyzobs =[]
    dips_below_zero = []

    for i_r in range(Nrefl):
        ref = refls[i_r]
        result = tiltnation.integrate_shoebox(ref)
        if result is None:
            ref["snr"] = -999 #np.inf
            ref["tilt_error"] = 999 #np.inf
            ref["shoebox_roi"] = 0,1,0,1,0,1
            below_zero_flag = None
            continue
        shoebox_roi, coefs, variance_matrix, Isum, varIsum, below_zero_flag = result
        dips_below_zero.append(below_zero_flag)
        bboxes.append(list(shoebox_roi))
        integrations.append(Isum)
        variances.append(varIsum)
        tilt_error.append(np.diag(variance_matrix).sum())

        x1, x2, y1, y2 = shoebox_roi
        Y, X = np.indices((y2-y1, x2-x1)) #range(x1, x2), range(y1, y2))
        tX, tY, tHeight = coefs

        tiltplane = tX*X + tY*Y + tHeight
        min_in_tilt = np.min(tiltplane)
        mins.append(min_in_tilt)

        pid = ref["panel"]
        pan_img = imgs[pid][y1:y2, x1:x2]/GAIN

        bgmask = is_bg_pix[pid][y1:y2, x1:x2]
        residual = pan_img[bgmask] - tiltplane[bgmask]

        if min_in_tilt < 0:
            print("residual with negative tilt =%f" % np.mean(residual))
            tX = tY = 0
            tHeight = np.median(pan_img)
            tiltplane = tX*X + tY*Y + tHeight
            residual = pan_img[bgmask] - tiltplane[bgmask]
            print("residual with median only =%f" % np.mean(residual))
            print("")

        all_residual.append(np.mean(residual))
        ref["snr"] = np.nan_to_num(Isum / varIsum)
        ref["tilt_error"] = np.diag(variance_matrix).sum()
        ref["shoebox_roi"] = shoebox_roi

        panels.append(ref["panel"])
        xyzobs.append(list(ref["xyzobs.px.value"]))

    snr = np.array(integrations) / np.sqrt(variances)
    data = {"panel": panels, "shoebox_roi": bboxes, "integration": integrations, "variance": variances,
            "tilt_errors": tilt_error, "xyzobs.px.value": xyzobs, "snr": snr, "dips_below_zero": dips_below_zero}
    df = pandas.DataFrame(data)

    if not args.plotassembled:
        #from cxid9114.prediction import prediction_utils
        import pylab as plt
        #refls_predict_bypanel = prediction_utils.refls_by_panelname(Rnew)
        pause = args.pause
        plt.figure(1)
        ax = plt.gca()
        refls_bypanel = df.groupby("panel")
        for panel_id in df.panel.unique():
            if args.pause == -1:
                plt.figure(1)
                ax = plt.gca()
            panel_img = imgs[panel_id]
            #panel_img = is_bg_pix[panel_id]
            m = panel_img.mean()
            s = panel_img.std()
            vmax = m + 4 * s
            vmin = m - s
            ax.clear()
            im = ax.imshow(panel_img, vmax=vmax, vmin=vmin)

            df_p = refls_bypanel.get_group(panel_id)
            for i_ref in range(len(df_p)):
                #ref = redict_bypanel[panel_id][i_ref]
                i1, i2, j1, j2 = df_p.iloc[i_ref]['shoebox_roi']
                snr = df_p.iloc[i_ref]["snr"]
                if snr >= args.minsnr:
                    color = "Limegreen"
                else:
                    color = "Darkorange"
                rect = plt.Rectangle(xy=(i1, j1), width=i2 - i1, height=j2 - j1, fc='none', ec=color)
                plt.gca().add_patch(rect)
                xx, yy, _ = df_p.iloc[i_ref]["xyzobs.px.value"]
                plt.plot([xx-0.5], [yy-0.5], 'rx')
                #mask = ref['shoebox'].mask.as_numpy_array()[0]
                #int_mask[j1:j2, i1:i2] = np.logical_or(mask == 5, int_mask[j1:j2, i1:i2])
                #bg_mask[j1:j2, i1:i2] = np.logical_or(mask == 19, bg_mask[j1:j2, i1:i2])
            if pause == -1:
                plt.show()
            else:
                plt.draw()
                plt.pause(pause)
    else:
        assert args.minsnr is not None
        assert args.mintilt is not None
        all_xx = []
        all_yy = []
        for i_panel in range(len(imgs)):
            node = El[0].detector[i_panel]
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
        snr = np.nan_to_num(snr)
        for i_r in range(len(df)):

            ref = refls[i_r]
            panel = int( df.iloc[i_r]["panel"])
            node = El[0].detector[panel]
            pixsize = node.get_pixel_size()[0]
            xpix, ypix, _ = df.iloc[i_r]["xyzobs.px.value"]
            xpix = xpix - 0.5
            ypix = ypix - 0.5
            xlab, ylab, _ = np.array(node.get_pixel_lab_coord((xpix, ypix)))/pixsize

            # FIXME: something weird with the display on CSPADs, to do with fast /slow scan vectors?
            i1, i2, j1, j2 = df.iloc[i_r]['shoebox_roi']
            width = i2-i1
            height = j2-j1
            if snr[i_r] >= args.minsnr and tilt_error[i_r] <= args.mintilt:
                r_rect = plt.Rectangle(xy=(ylab - height/2., xlab - width/2.), width=width,
                                       height=height, ec=r_color, fc='None')
                r_patches.append(r_rect)

            else:
                rect = plt.Rectangle(xy=(ylab-height/2., xlab-width/2.), width=width,
                                 height=height, ec=rej_color, fc='None')
                rej_patches.append(rect)

        plt.clf()
        plt.imshow(out[0], extent=(out[2][0], out[2][-1], out[1][-1], out[1][0]), vmin=args.vmin, vmax=args.vmax)
        coll = plt.mpl.collections.PatchCollection(rej_patches, match_original=True)
        r_coll = plt.mpl.collections.PatchCollection(r_patches, match_original=True)
        plt.gca().add_collection(coll)
        plt.gca().add_collection(r_coll)
        plt.suptitle("Green: SNR >= %.2f, Orange:SNR < %.2f" % (args.minsnr, args.minsnr))
        plt.show()


    #    delta_q=0.07, photon_gain=GAIN, sigma_rdout=sigma_readout, zinger_zscore=outlier_Z,
    #    exper=El[0], predicted_refls=refls, sb_pad=5, filter_boundary_spots=args.filterboundaryspots,
    #    minsnr=args.minsnr, mintilt=args.mintilt, plot=True, vmin=args.vmin, vmax=args.vmax)
