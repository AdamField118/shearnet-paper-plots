"""Presentation changes preserve the measured arrays and uncertainty artists."""
import unittest
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from paper_colors import COLORS, recolor_artists, color_psf_maps

class PaletteTests(unittest.TestCase):
    def tearDown(self): plt.close('all')

    def test_leakage_colors_preserve_values_errors_and_alpha(self):
        fig,ax=plt.subplots()
        x=np.arange(3); y=np.array([.02,.03,-.01]); errors=np.array([.01,.02,.005])
        line,caps,bars=ax.errorbar(x,y,yerr=errors,fmt='o',color='magenta',capsize=3,alpha=.4,label='ShearNet')
        ax.legend()
        before=[a.copy() for a in bars[0].get_segments()]
        recolor_artists(fig,{'magenta':COLORS['shearnet']})
        np.testing.assert_array_equal(line.get_ydata(),y)
        for a,b in zip(before,bars[0].get_segments()):np.testing.assert_array_equal(a,b)
        np.testing.assert_allclose(to_rgba(line.get_color())[:3],to_rgba(COLORS['shearnet'])[:3])
        self.assertEqual(line.get_alpha(),.4)
        np.testing.assert_allclose(bars[0].get_colors()[0,:3],to_rgba(COLORS['shearnet'])[:3])

    def test_psf_colors_preserve_maps_and_center_zero(self):
        fig,axes=plt.subplots(1,3); vals=np.array([[-.06,0],[.03,.06]])
        ims=[]
        for i,ax in enumerate(axes):
            data=vals if i<2 else np.abs(vals)+.09
            im=ax.imshow(data,vmin=-.06 if i<2 else .09,vmax=.06 if i<2 else .18)
            fig.colorbar(im,ax=ax);ims.append(im)
        before=[im.get_array().copy() for im in ims]
        color_psf_maps([axes])
        for im,data in zip(ims,before):np.testing.assert_array_equal(im.get_array(),data)
        self.assertEqual(ims[0].norm(0),.5)
        self.assertEqual(ims[1].norm(0),.5)
        self.assertEqual(ims[2].get_clim(),(.09,.18))
        self.assertEqual(ims[2].cmap.name,'cividis')

class PrintTests(unittest.TestCase):
    def test_signed_scale_has_no_grayscale_reversal(self):
        from paper_colors import PSF_ELLIPTICITY_CMAP, srgb_to_linear
        rgb=PSF_ELLIPTICITY_CMAP(np.linspace(0,1,256))[:,:3]
        luminance=srgb_to_linear(rgb)@np.array([.2126,.7152,.0722])
        self.assertTrue(np.all(np.diff(luminance)>0))
        self.assertLess(luminance[0],.05)
        self.assertGreater(luminance[-1],.75)

    def test_methods_and_components_remain_distinct_without_hue(self):
        from paper_colors import RESPONSE_STYLES, HATCHES, srgb_to_linear
        self.assertEqual(len(set(x[0] for x in RESPONSE_STYLES.values())),4)
        self.assertEqual(len(set(x[1] for x in RESPONSE_STYLES.values())),4)
        self.assertNotEqual(HATCHES['shearnet'],HATCHES['ngmix'])
        rgb=np.array([to_rgba(COLORS[e])[:3] for e in ['shearnet','ngmix']])
        lum=srgb_to_linear(rgb)@np.array([.2126,.7152,.0722])
        self.assertGreater((lum[1]+.05)/(lum[0]+.05),3)

if __name__=='__main__':unittest.main()
