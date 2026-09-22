"""Reproduce the supplied paper-5 figures with the shared palette, without FITS.

Only PDF color instructions, response line styles and embedded image palettes
change. Plot geometry, labels, fitted coefficients and image index arrays are
preserved. The PSF map palettes match the original Matplotlib LUT exactly;
colorbar palettes use its rounded form. Reject any unexpected input.
"""
from pathlib import Path
import argparse, hashlib, re, sys, zipfile
import fitz
import numpy as np
import matplotlib
from matplotlib.colors import to_rgb
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'results'))
from paper_colors import COLORS, PSF_ELLIPTICITY_CMAP, PSF_SIZE_CMAP

HASHES = {'psf_properties': 'bb228491f40108b5aadb49bba11c7ab73cda2ea72835407f25a9e9ef79a665fb', 'response_vs_snr': '59e0c9c1c1402887229f81b98316ddfcf203fd39c99ccb28ba40f7fc9d91a529', 'psf_leakage': '5169d7a83f91500b26d8c1dc5a066af5d3211ca4dee2d8254f0283988aed2ef1', 'unit_test_bias': '38d3d3f00181efc1e3a9cfcd657b90095d7405cec39e77846ce18550ae3dc68a'}
NUMBER = r'[-+]?(?:\d*\.\d+|\d+)(?:[Ee][-+]?\d+)?'
RGB = re.compile(rf'({NUMBER})\s+({NUMBER})\s+({NUMBER})\s+(rg|RG)\b')
DASH = re.compile(rf'\[[^\]]*\]\s*{NUMBER}\s+d\b')


def vectors(doc, mapping, response=False):
    lookup={tuple(np.round(np.array(to_rgb(k))*255).astype(int)):(v,c) for k,(v,c) in mapping.items()}
    page=doc[0]; source=page.read_contents().decode('latin1')
    if response:
        # The duplicate right-hand legend obscures a response curve. The left
        # legend and caption define estimator colors and component line styles.
        anchor='1.4 w [ 1.4 2.31 ] 0 d /DeviceRGB cs\n744.57452 140.102344 m'
        assert source.count(anchor)==1
        source=source[:source.index(anchor)]+'Q\n'
    # Dash instructions usually precede the RGB operator. Track graphics state
    # per q/Q block, then change only the response-curve/legend dash pattern.
    chunks=re.split(r'(?<!\S)([qQ])(?!\S)', source)
    for i,chunk in enumerate(chunks):
        matches=list(RGB.finditer(chunk));components=[]
        for m in matches:
            key=tuple(round(float(m[j])*255) for j in (1,2,3))
            if key in lookup: components.append(lookup[key][1])
        if components and components[0] is not None and len(set(components))==1:
            c=components[0]
            chunk=DASH.sub({'s11':'[] 0 d','s22':'[8.14 3.52] 0 d','n11':'[8 3 1.5 3] 0 d','n22':'[1.5 3] 0 d','s':'[] 0 d','n':'[8 4] 0 d'}[c],chunk)
        def repl(m):
            key=tuple(round(float(m[j])*255) for j in (1,2,3))
            if key not in lookup:return m[0]
            return ' '.join(f'{v:.10f}' for v in to_rgb(lookup[key][0]))+' '+m[4]
        chunks[i]=RGB.sub(repl,chunk)
    assert len(page.get_contents()) == 1
    doc.update_stream(page.get_contents()[0],''.join(chunks).encode('latin1'))


def psf_palettes(doc):
    for xref, *_ in doc[0].get_images():
        cs=doc.xref_get_key(xref,'ColorSpace')[1]
        match=re.search(r'<([A-Fa-f0-9]+)>',cs)
        if not match:raise ValueError('expected indexed PSF image')
        colors=np.frombuffer(bytes.fromhex(match[1]),dtype=np.uint8).reshape(-1,3)
        # Identify the old LUT by exact match, accepting either PDF rounding rule.
        candidates=[]
        for old,new in [('RdBu_r',PSF_ELLIPTICITY_CMAP),('viridis',PSF_SIZE_CMAP)]:
            rgb=matplotlib.colormaps[old](np.linspace(0,1,256))[:,:3]*255
            for lut in (rgb.astype('uint8'),np.round(rgb).astype('uint8')):
                dist=np.sum((colors[:,None,:].astype(float)-lut[None,:,:])**2,axis=-1)
                inds=dist.argmin(axis=1)
                candidates.append((float(dist.min(axis=1).max()),new,inds))
        error,new,inds=min(candidates,key=lambda c:c[0])
        if error!=0:raise ValueError(f'unknown PSF palette: {error}')
        replacement=((matplotlib.colormaps[new] if isinstance(new,str) else new)(np.linspace(0,1,256))[:,:3]*255).astype('uint8')[inds]
        doc.xref_set_key(xref,'ColorSpace',cs[:match.start(1)]+replacement.tobytes().hex()+cs[match.end(1):])


def draw_marker(page, center, marker, color, radius=2.1):
    x,y=center
    if marker=='o':page.draw_circle(center,radius,color=color,fill=(1,1,1),width=.8)
    else:
        page.draw_circle(center,radius+0.5,color=None,fill=(1,1,1))
        if marker=='s':pts=[(x-radius,y-radius),(x+radius,y-radius),(x+radius,y+radius),(x-radius,y+radius)]
        elif marker=='^':pts=[(x,y-radius-0.5),(x+radius,y+radius),(x-radius,y+radius)]
        else:pts=[(x,y-radius-.4),(x+radius+.4,y),(x,y+radius+.4),(x-radius-.4,y)]
        page.draw_polyline(pts,color=color,fill=(1,1,1),closePath=True,width=.8)


def response_markers_and_legend(doc,original):
    page=doc[0]
    source_colors={'#2ca02c':('shearnet','o'),'#ff7f0e':('shearnet','s'),
                   '#3B4CC0':('ngmix','^'),'#B40426':('ngmix','D')}
    for d in original[0].get_drawings():
        rect=d['rect']
        if abs(rect.x0+2-754.57452)<.02:continue
        if d['fill']!=(1.,1.,1.) or not d['color'] or abs(rect.width-4)>.02 or abs(rect.height-4)>.02:continue
        for old,(est,marker) in source_colors.items():
            if np.allclose(d['color'],to_rgb(old),atol=1e-6):
                draw_marker(page,(rect.x0+2,rect.y0+2),marker,to_rgb(COLORS[est]))
    # Replace the old same-marker legend in the data-free lower-right area.
    page.draw_rect(fitz.Rect(187,127,306,226),color=None,fill=(1,1,1))
    rows=[('unit response','k','[1 2] 0',None),
          ('ShearNet 11','shearnet','[] 0','o'),('ShearNet 22','shearnet','[7 3] 0','s'),
          ('ngmix 11','ngmix','[7 3 1 3] 0','^'),('ngmix 22','ngmix','[1 3] 0','D')]
    for i,(label,est,dash,marker) in enumerate(rows):
        y=139+i*18
        col=(0,0,0) if est=='k' else to_rgb(COLORS[est])
        page.draw_line((193,y),(220,y),color=col,dashes=dash,width=1.5)
        if marker:draw_marker(page,(207,y),marker,col)
        page.insert_text((226,y+3),label,fontsize=10,fontname='tiro',color=(0,0,0))


def hatch_bars(doc):
    page=doc[0]; ochre=to_rgb(COLORS['ngmix'])
    rects=[d['rect'] for d in page.get_drawings() if d['fill'] and np.allclose(d['fill'],ochre,atol=1e-6)]
    for r in rects:
        page.draw_rect(r,color=(0,0,0),width=.5)
        # Slope-one lines clipped analytically to the exact bar rectangle.
        for c in np.arange(r.y0-r.x1,r.y1-r.x0+1,6):
            lo=max(r.x0,r.y0-c);hi=min(r.x1,r.y1-c)
            if hi>lo:page.draw_line((lo,lo+c),(hi,hi+c),color=(0,0,0),width=.5)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--paper-zip',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.paper_zip) as z:
        for name,digest in HASHES.items():
            raw=z.read('figures/'+name+'.pdf')
            if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('unexpected source '+name)
            doc=fitz.open(stream=raw,filetype='pdf')
            if name=='psf_properties':
                psf_palettes(doc)
                page=doc[0]
                stream=page.read_contents().decode('latin1')
                stream=re.sub(r'BT.*?ET',lambda m: '' if '(Obs)' in m[0] and 'SFEx' in m[0] else m[0],stream,flags=re.S)
                assert len(page.get_contents())==1
                doc.update_stream(page.get_contents()[0],stream.encode('latin1'))
            elif name=='response_vs_snr':
                vectors(doc,{'#2ca02c':(COLORS['shearnet'],'s11'),'#ff7f0e':(COLORS['shearnet'],'s22'),
                             '#3B4CC0':(COLORS['ngmix'],'n11'),'#B40426':(COLORS['ngmix'],'n22')},True)
            elif name=='psf_leakage':vectors(doc,{'magenta':(COLORS['shearnet'],'s'),'teal':(COLORS['ngmix'],'n')})
            if name=='response_vs_snr':
                response_markers_and_legend(doc,fitz.open(stream=raw,filetype='pdf'))
            if name=='unit_test_bias':
                vectors(doc,{'#0072B2':(COLORS['shearnet'],None),'#D55E00':(COLORS['ngmix'],None)})
                hatch_bars(doc)
            doc.save(args.out/(name+'.pdf'),garbage=4,deflate=True,no_new_id=True)
            print('wrote',args.out/(name+'.pdf'))

if __name__=='__main__':main()
