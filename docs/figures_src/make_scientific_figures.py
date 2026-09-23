"""Build the physics-reviewed figure collection from formulas and recorded runs.

Run through tools/build_figures.py. Diagrams are native TikZ; numerical figures
retain Python sources and source-data CSVs. Original experimental assets remain
unchanged. All lengths below are final printed dimensions.
"""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle, Rectangle
from scipy.optimize import brentq, curve_fit
from scipy.signal import fftconvolve
import style as s
from figure_data import airy, AIRY_ZERO, gaussian_pixels, two_emitter_example
from project_paths import PAPER, HAS_MANUSCRIPT

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / 'figures'
DATA = HERE / 'data'
MANIFEST = []


def record(name, kind, caption, scope='analytical or synthetic teaching example'):
    MANIFEST.append(dict(name=name, category=kind, caption=caption, scope=scope))


def save(fig, name, caption, kind='theory', scope='analytical or synthetic teaching example'):
    s.save(fig, str(FIG/kind), name)
    record(name, kind, caption, scope)


def csv_data(name, columns, values):
    DATA.mkdir(exist_ok=True)
    with (DATA/(name+'.csv')).open('w', newline='', encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(columns); w.writerows(np.asarray(values))


def panels(n, height=2.7, fraction=1.):
    fig, ax = plt.subplots(1, n, figsize=s.figsize(fraction,height), squeeze=False)
    fig.subplots_adjust(left=.17 if n==1 else .085, right=.96 if n==1 else .98, bottom=.23, top=.82, wspace=.43)
    for a in ax[0]: a.set_anchor('N')
    if n > 1:
        for a,l in zip(ax[0], 'abcdef'): s.panel_label(a,l)
    return fig, ax[0]


def img(ax, data, title, vmax=None, ticks=False):
    h,w=data.shape
    im=ax.imshow(data, origin='lower', extent=(-.5,w-.5,-.5,h-.5),
                 cmap='gray', interpolation='nearest', aspect='equal', vmin=0, vmax=vmax)
    ax.set_title(title, pad=7)
    if not ticks: ax.set_xticks([]); ax.set_yticks([])
    return im


def T1():
    fig,ax=panels(2,2.9)
    x=np.linspace(-1.3,1.3,501); X,Y=np.meshgrid(x,x); A=airy(np.hypot(X,Y))
    im=ax[0].imshow(A,extent=(-1.3,1.3,-1.3,1.3),origin='lower',cmap='viridis',
                    norm=LogNorm(vmin=1e-4,vmax=1),interpolation='nearest')
    ax[0].set(xlabel=r'$x/(\lambda/\mathrm{NA})$',ylabel=r'$y/(\lambda/\mathrm{NA})$',title='Circular-pupil PSF')
    cb=fig.colorbar(im,ax=ax[0],fraction=.045,pad=.035);cb.set_label(r'$I/I(0)$ (log scale)')
    r=np.linspace(0,1.4,650); half=brentq(lambda r:airy(r)-.5,.1,.4)
    sigma=half/np.sqrt(2*np.log(2)); G=np.exp(-r*r/(2*sigma*sigma))
    ax[1].plot(r,airy(r),color=s.COL_RECON,label='Airy intensity')
    ax[1].plot(r,G,color=s.COL_ACCENT,ls='--',label='Gaussian (same FWHM)')
    ax[1].set(xlabel=r'$r/(\lambda/\mathrm{NA})$',ylabel='Peak-normalized intensity',title='Central-lobe approximation',xlim=(0,1.4),ylim=(-.025,1.05))
    ax[1].legend(fontsize=8.5,loc='upper right')
    csv_data('airy_gaussian',['r_lambda_over_NA','Airy','Gaussian'],np.c_[r,airy(r),G])
    save(fig,'T1_airy_gaussian','Ideal scalar, incoherent circular-pupil PSF. The image uses a labelled logarithmic colour scale to reveal rings; the profile is linear. The Gaussian matches the Airy FWHM only, not the side lobes or total energy.')


def T2():
    fig,ax=panels(3,2.6)
    x=np.linspace(-1.4,1.4,1601)
    for a,d,label in zip(ax,[1.6*AIRY_ZERO,AIRY_ZERO,.55*AIRY_ZERO],['Above Rayleigh','At Rayleigh','Below Rayleigh']):
        p=airy(x-d/2);q=airy(x+d/2);den=(p+q).max()
        a.plot(x,(p+q)/den,color=s.COL_AXIS,label='Incoherent sum')
        a.plot(x,p/den,color=s.COL_RECON,ls='--',lw=.8,label='Individual PSFs')
        a.plot(x,q/den,color=s.COL_RECON,ls='--',lw=.8)
        a.set(title=label+'\n'+r'$d=%.3f\,\lambda/\mathrm{NA}$'%d,xlabel=r'$x/(\lambda/\mathrm{NA})$',xlim=(-1.4,1.4),ylim=(0,1.08))
        a.set_xticks([-1,0,1])
        csv_data('rayleigh_'+label.split()[0].lower(),['x_lambda_over_NA','PSF_1','PSF_2','sum'],np.c_[x,p/den,q/den,(p+q)/den])
    ax[0].set_ylabel('Common-normalized intensity')
    fig.subplots_adjust(bottom=.30)
    fig.legend(*ax[2].get_legend_handles_labels(),loc='lower center',ncol=2,fontsize=8.5,bbox_to_anchor=(.5,.005))
    save(fig,'T2_rayleigh','Two equal-brightness incoherent point sources through an ideal circular pupil. Each panel uses one normalization for the sum and both component PSFs, so the components add exactly. Rayleigh separation is 0.610 lambda/NA; it is a conventional two-point criterion, not a universal localization limit.')


def T3():
    n=161; y,x=np.indices((n,n));c=(n-1)/2
    obj=np.zeros((n,n));obj[80,73]=obj[80,87]=1
    h=np.exp(-((x-c)**2+(y-c)**2)/(2*8**2));h/=h.sum()
    image=fftconvolve(obj,h,mode='same')
    fig,ax=panels(3,2.65)
    for a,z,t in zip(ax,[obj,h,image],['Two point emitters','Unit-sum Gaussian PSF','Expected optical image']):
        img(a,z,t);a.set_xlim(45,115);a.set_ylim(45,115)
    ax[0].scatter([73,87],[80,80],s=24,facecolors='none',edgecolors=s.COL_ACCENT)
    fig.text(.5,.05,r'$g=f*h$; camera pixels integrate $g$ over their area. Images scaled separately for visibility.',ha='center',fontsize=8.5)
    save(fig,'T3_image_formation','Linear shift-invariant, incoherent image formation: two equal point sources convolved with a normalized Gaussian PSF. Pixel integration is a subsequent detector operation. Panel brightness is independently scaled; the circles identify source positions.')


def T4():
    n=100;y,x=np.indices((n,n));ang=np.linspace(0,2*np.pi,40,endpoint=False)
    points=np.c_[49.5+28*np.cos(ang),49.5+28*np.sin(ang)]
    allon=sum(gaussian_pixels((n,n),*p,4,1000) for p in points)
    selected=[0,8,16,24,32];sparse=sum(gaussian_pixels((n,n),*points[i],4,1000) for i in selected)
    fig,ax=panels(2,2.8,.9);vmax=max(allon.max(),sparse.max())
    for a,z,t in zip(ax,[allon,sparse],['All sources emitting','Five active sources']):img(a,z,t,vmax)
    fig.text(.5,.06,'Same source positions, PSF and intensity scale; only the active subset changes.',ha='center',fontsize=8.5)
    save(fig,'T4_blinking','Temporal sparsity: the same emitter arrangement with all 40 sources active or five well-separated sources active. These are noise-free expectations with one shared intensity scale; sparse activation enables individual localization.')


def T5():
    from classroomstorm_core.reconstruction import reconstruct_frames
    rng=np.random.default_rng(110);shape=(25,25);truth=np.array([12.3,11.8])
    expected=3+gaussian_pixels(shape,*truth,2.3,2000)
    frames=rng.poisson(expected,size=(800,*shape))
    result=reconstruct_frames(frames,blinker_mode='single',threshold_quantile=.98,background_mode='frame_median')
    errors=np.array([p['x_px']-truth[0] for p in result.localizations])
    fig=plt.figure(figsize=s.figsize(1,3.9));gs=fig.add_gridspec(2,4,left=.10,right=.98,bottom=.15,top=.9,hspace=.75)
    for i in range(4):
        a=fig.add_subplot(gs[0,i]);img(a,frames[i],f'Frame {i+1}',frames[:4].max());s.panel_label(a,'abcd'[i])
    a=fig.add_subplot(gs[1,:]);s.panel_label(a,'e');a.hist(errors,bins=28,color=s.COL_RECON,edgecolor='white',lw=.4)
    a.axvline(0,color=s.COL_TRUTH,ls='--',label='Zero error');a.axvline(errors.mean(),color=s.COL_ACCENT,label='Mean error')
    a.set(xlabel=r'$\hat{x}-x_0$ (camera pixels)',ylabel='Frames',title='Repeated position estimates')
    a.legend(fontsize=8.5);fig.text(.5,.015,f'800 frames; bias = {errors.mean():.3f} px; sample s.d. = {errors.std(ddof=1):.3f} px',ha='center',fontsize=8.5)
    csv_data('centroid_repeatability',['x_error_px'],errors[:,None])
    save(fig,'T5_poisson_noise','Pixel-integrated Gaussian signal with 2000 expected photons, sigma=2.3 pixels, and Poisson background mean 3 counts/pixel. The V1.2 single-emitter centroid uses q=0.98 and frame-median subtraction. The histogram measures repeated-estimate spread and bias, not per-event uncertainty.')


def T6():
    fig,ax=panels(1,3.05,.7);fig.subplots_adjust(bottom=.30);a=ax[0];N=np.logspace(2,5,300);vals=[]
    for b,col,ls in zip([0,2,8],[s.COL_AXIS,s.COL_RECON,s.COL_ACCENT],['-','--',':']):
        se=s.thompson_se(N,1.5,1,b);vals.append(se);a.loglog(N,se,color=col,ls=ls,label=rf'$b_{{\rm rms}}={b}$')
    a.set(xlabel='Expected detected signal photons $N$',ylabel='Predicted position s.d. (pixels)',title='Theoretical localization precision')
    a.legend(fontsize=8.5);fig.text(.5,.02,r'$s=1.5$ px; pixel pitch $a=1$ px.'+'\n'+r'$b_{\rm rms}$: RMS background noise (photon-equivalent).',ha='center',fontsize=8.5)
    csv_data('thompson_reference',['N','b_rms_0','b_rms_2','b_rms_8'],np.c_[N,*vals])
    save(fig,'T6_precision_vs_photons','Thompson-Larson-Webb reference approximation, not uncertainty reported by the centroid software. PSF sigma=1.5 camera pixels, pixel pitch a=1 in the same length units. b_rms is background RMS noise, not mean background. Position s.d. scales as N^(-1/2) in the signal-limited regime and N^(-1) in the background-limited regime.')


def T7():
    rng=np.random.default_rng(5);shape=(27,27);truth=(14.4,12.6)
    frame=rng.poisson(4+gaussian_pixels(shape,*truth,2.4,4200));q=.98
    x,y=s.centroid_threshold(frame,q);mask=frame>=np.quantile(frame,q)
    fig,ax=panels(1,3.05,.7);fig.subplots_adjust(left=.14,right=.80);a=ax[0];im=img(a,frame,'Thresholded intensity centroid',ticks=True)
    a.contour(mask,levels=[.5],colors=[s.COL_ACCENT],linewidths=.8)
    a.plot(*truth,marker='+',color=s.COL_TRUTH,ms=8,mew=1.3,label='True position')
    a.plot(x,y,marker='o',mfc='none',color=s.COL_RECON,ms=7,label='Estimated position')
    a.set(xlabel='$x$ (camera pixels)',ylabel='$y$ (camera pixels)');a.legend(fontsize=8.5,loc='lower left',frameon=True,facecolor='white',framealpha=1)
    cb=fig.colorbar(im,ax=a,fraction=.045,pad=.04);cb.set_label('Simulated detected counts')
    save(fig,'T7_centroid','A synthetic pixel-integrated Gaussian spot with Poisson noise. The orange contour marks the q=0.98 threshold mask; the plus and circle distinguish truth and the application centroid. Subpixel coordinates do not imply zero error. This example uses no background correction.')


def T8():
    rng=np.random.default_rng(8);fig,ax=plt.subplots(2,2,figsize=s.figsize(.72,3.9))
    fig.subplots_adjust(left=.08,right=.98,bottom=.17,top=.85,hspace=.65,wspace=.25)
    for a,sd,bias,label in zip(ax.ravel(),[.09,.09,.29,.29],[(0,0),(.48,.30),(0,0),(.48,.30)],'abcd'):
        z=rng.normal(size=(60,2));z-=z.mean(axis=0);z=z*sd+np.array(bias)
        for r in [.3,.6,.9]:a.add_patch(Circle((0,0),r,fill=False,color='.85',lw=.6))
        a.scatter(z[:,0],z[:,1],s=10,color=s.COL_RECON,alpha=.8)
        a.plot(0,0,'+',color=s.COL_TRUTH,ms=10,mew=1.4);a.plot(*z.mean(axis=0),'x',color=s.COL_ACCENT,ms=6)
        a.set(xlim=(-1.2,1.2),ylim=(-1.2,1.2),aspect='equal',xticks=[],yticks=[],title=('Small' if sd<.2 else 'Large')+' spread\n'+('Small' if bias==(0,0) else 'Large')+' bias')
        for sp in a.spines.values():sp.set_visible(False)
        s.panel_label(a,label,x=0,y=1.26)
    fig.text(.5,.025,'+ True position    Orange cross: mean estimate\nDots: repeated estimates',ha='center',fontsize=8.5)
    save(fig,'T8_precision_accuracy','Illustrative samples separate repeatability (spread) from systematic offset (bias). The mean is explicitly controlled. Accuracy depends on both bias and random error; small spread alone is not accuracy.')


def T9():
    rng=np.random.default_rng(2);fig,ax=panels(2,2.65,.86);ang=np.linspace(0,2*np.pi,400)
    for a,n,t in zip(ax,[16,320],['Sparse spatial sampling','Denser spatial sampling']):
        a.plot(np.cos(ang),np.sin(ang),color='.55',ls='--',lw=.8,label='True structure')
        theta=rng.uniform(0,2*np.pi,n);rr=1+rng.normal(0,.035,n)
        a.scatter(rr*np.cos(theta),rr*np.sin(theta),s=9,color=s.COL_RECON,label='Sampled positions')
        a.set(xlim=(-1.2,1.2),ylim=(-1.2,1.2),aspect='equal',xticks=[],yticks=[],title=f'{t}\n{n} positions')
        for sp in a.spines.values():sp.set_visible(False)
    fig.text(.5,.045,'Same positional noise; different spatial coverage.\nRepeated detections can revisit the same site.',ha='center',fontsize=8.5)
    save(fig,'T9_nyquist','Synthetic ring sampled at 16 or 320 independently drawn positions with the same radial noise. Sampling spacing and localization precision are different quantities. No universal resolution threshold or adequate sample count is inferred from this illustration.')


def result_figures():
    ex=two_emitter_example();r=ex['result'];p=ex['parameters'];DATA.mkdir(exist_ok=True)
    (DATA/'two_emitter_run.json').write_text(json.dumps(p,indent=2)+'\n')
    csv_data('two_emitter_truth',['frame','emitter_id','x_px','y_px'],np.c_[np.arange(len(ex['ids'])),ex['ids'],ex['truth']])
    csv_data('two_emitter_localizations',['frame','x_px','y_px','x_error_px','y_error_px'],np.c_[[v['frame'] for v in r.localizations],ex['estimates'],ex['errors']])
    np.savez_compressed(DATA/'two_emitter_images.npz',raw_mean=r.raw_mean,reconstruction=r.superres,first_frame=ex['frames'][0])
    raw=np.maximum(r.raw_mean-p['mean_background_counts_per_pixel'],0);rec=r.superres
    # Integrate over y so both profiles represent the same spatial operation.
    rp=raw.sum(axis=0);sp=rec.sum(axis=0);rp/=rp.max();sp/=sp.max();x=np.arange(raw.shape[1])
    fig,ax=panels(3,2.8)
    for a,z,t in zip(ax[:2],[raw/raw.max(),rec/rec.max()],['Mean signal image','Rendered localizations']):
        img(a,z,t,vmax=1);a.set_xlim(18,61);a.set_ylim(18,61);a.set_xlabel('Same field of view')
    ax[1].scatter(ex['centers'][:,0],ex['centers'][:,1],s=30,marker='+',color=s.COL_ACCENT)
    ax[2].plot(x,rp,color=s.COL_RAW,ls='--',label='Mean signal')
    ax[2].plot(x,sp,color=s.COL_RECON,label='Rendered positions')
    for xx in ex['centers'][:,0]:ax[2].axvline(xx,color=s.COL_TRUTH,ls=':',lw=.7)
    ax[2].set(xlim=(18,61),ylim=(0,1.12),xlabel='$x$ (camera pixels)',ylabel='Unit-peak projected signal',title='Column-sum profiles')
    fig.legend(*ax[2].get_legend_handles_labels(),loc='upper center',ncol=2,fontsize=8.5,bbox_to_anchor=(.5,1.0))
    fig.text(.5,.025,'Synthetic sparse blinking; same camera grid. Render width is a display parameter.',ha='center',fontsize=8.5)
    cap=f'Actual V1.2 centroid reconstruction of {p["frames"]} synthetic pixel-integrated Poisson frames. Exactly one of two sources is active per frame; separation 8 pixels and optical Gaussian sigma 6 pixels. q=0.995 with frame-median subtraction. The mean image subtracts the known simulated background for display. Both images and column-sum profiles are independently peak-normalized. Green guides and orange crosses mark truth. Gaussian rendering sigma=1.5 pixels does not measure resolution. Source data and run parameters accompany the figure.'
    save(fig,'T11_raw_vs_reconstruction',cap,scope='synthetic input processed by maintained V1.2 reconstruct_frames')
    # Help: one line selected at an actual integer camera row, no fabricated narrower PSF.
    fig,ax=panels(3,2.8);row=40
    for a,z,t in zip(ax[:2],[raw/raw.max(),rec/rec.max()],['Mean signal','Rendered positions']):
        img(a,z,t,vmax=1);a.axhline(row,color=s.COL_ACCENT,ls='--',lw=.8);a.set_xlim(18,61);a.set_ylim(18,61)
    for z,col,ls,label in [(raw,s.COL_RAW,'--','Mean signal'),(rec,s.COL_RECON,'-','Rendered positions')]:
        v=z[row];ax[2].plot(x,v/v.max(),color=col,ls=ls,label=label)
    ax[2].set(xlim=(18,61),ylim=(0,1.12),xlabel='$x$ (camera pixels)',ylabel='Unit-peak line signal',title=f'Line at camera row {row}')
    fig.legend(*ax[2].get_legend_handles_labels(),loc='upper center',ncol=2,fontsize=8.5,bbox_to_anchor=(.5,1.0))
    save(fig,'H8_linked_profile','Linked row profiles from the recorded two-source synthetic reconstruction. Dashed horizontal lines identify camera row 40; each profile is peak-normalized. A profile describes the displayed images and is not an independent resolution measurement.',kind='help',scope='recorded two-source synthetic application run')
    return ex


def help_figures(ex):
    fig,ax=panels(2,2.8,.8);ang=np.linspace(0,2*np.pi,18,endpoint=False);points=np.c_[49.5+30*np.cos(ang),49.5+30*np.sin(ang)]
    mask=np.zeros((100,100));yy,xx=np.indices(mask.shape)
    for x,y in points:mask[(xx-x)**2+(yy-y)**2<2.3**2]=1
    img(ax[0],mask,'Eligible pattern regions')
    ax[1].scatter(points[:,0],points[:,1],s=16,color=s.COL_RECON);ax[1].set(xlim=(-.5,99.5),ylim=(-.5,99.5),aspect='equal',xticks=[],yticks=[],title='Illustrative source positions')
    save(fig,'H4_pattern_preview','A schematic source pattern and one illustrative set of source positions. The mask defines eligible regions; it is not a measured optical image or a guarantee of one emitter per bright connected region.',kind='help')
    fig,ax=panels(1,3,.7);fig.subplots_adjust(left=.14,right=.80);im=img(ax[0],ex['frames'][0],'One simulated camera frame',ticks=True)
    ax[0].set(xlabel='$x$ (camera pixels)',ylabel='$y$ (camera pixels)');cb=fig.colorbar(im,ax=ax[0],fraction=.045);cb.set_label('Simulated detected counts')
    save(fig,'H5_blinking_frame','One raw synthetic frame from the recorded two-source run. Only one emitter is active in this frame; signal and background are Poisson-distributed.',kind='help')


def model_mismatch():
    x=np.linspace(-7,7,700);fig,ax=plt.subplots(2,3,figsize=s.figsize(1,4.0),sharex=True)
    fig.subplots_adjust(left=.12,right=.98,bottom=.20,top=.86,wspace=.38,hspace=.78)
    def g(x,A,c,w):return A*np.exp(-.5*((x-c)/w)**2)
    for k,d in enumerate([1.,2.,4.]):
        y=g(x,1,-d/2,1)+g(x,1,d/2,1);fit,_=curve_fit(g,x,y,p0=[y.max(),0,1+d/3],bounds=([0,-.001,.1],[3,.001,8]));f=g(x,*fit)
        ax[0,k].plot(x,y,color=s.COL_AXIS,label='Two sources');ax[0,k].plot(x,f,color=s.COL_ACCENT,ls='--',label='One-Gaussian fit')
        ax[0,k].set_title(r'$d=%g\sigma$'%d);ax[1,k].plot(x,y-f,color=s.COL_RECON);ax[1,k].axhline(0,color='.7',lw=.6)
        ax[1,k].set(xlabel=r'$x/\sigma$',title=f'Residual (RMS {np.sqrt(np.mean((y-f)**2)):.3f})')
        ax[0,k].set_ylim(0,2.1);ax[1,k].set_ylim(-.6,.6)
        s.panel_label(ax[0,k],'abc'[k],x=-.12,y=1.12)
        s.panel_label(ax[1,k],'def'[k],x=-.12,y=1.12)
        csv_data(f'overlap_d{d:g}',['x_over_sigma','two_source_signal','single_gaussian_fit','residual'],np.c_[x,y,f,y-f])
    ax[0,0].set_ylabel('Intensity (a.u.)');ax[1,0].set_ylabel('Data minus fit')
    fig.legend(*ax[0,2].get_legend_handles_labels(),loc='lower center',ncol=2,fontsize=8.5,bbox_to_anchor=(.5,.005))
    s.save(fig,str(PAPER),'P3_model_mismatch')
    record('P3_model_mismatch','manuscript','Analytical two-Gaussian signals fitted by a constrained symmetric single Gaussian. Residuals reveal model mismatch as separation increases. This illustrative fit is not an implemented ClassRoomSTORM fitting stage, and no automatic acceptance criterion is claimed.')


def validation_figures(ex):
    from classroomstorm_core.reconstruction import render_localization_image
    fig, ax = panels(2, 2.9)
    for i, (col, ls) in enumerate([(s.COL_RECON, '-'), (s.COL_ACCENT, '--')]):
        selected = ex['estimates'][ex['ids'] == i]
        ax[0].hist(selected[:, 0], bins=np.linspace(33, 46, 65), histtype='step',
                   color=col, ls=ls, lw=1.2, label=f'Source {i+1}')
        ax[0].axvline(ex['centers'][i, 0], color=col, ls=':', lw=.8)
        errors = ex['errors'][ex['ids'] == i]
        ax[1].scatter(errors[:, 0], errors[:, 1], s=5, alpha=.45, color=col,
                      marker='o' if i == 0 else '^')
    ax[0].set(xlabel=r'$\hat{x}$ (camera pixels)', ylabel='Localizations', title='Position estimates and truth')
    ax[0].legend(fontsize=8.5)
    ax[1].axvline(0, color='.7', lw=.6); ax[1].axhline(0, color='.7', lw=.6)
    ax[1].set(xlabel=r'$\hat{x}-x_0$ (pixels)', ylabel=r'$\hat{y}-y_0$ (pixels)',
              title='Errors relative to known truth', aspect='equal')
    p = ex['parameters']
    fig.text(.5,.025, f'Known separation: 8 px; recovered: {p["estimated_separation_px"]:.3f} px; 2D RMSE: {p["rmse_position_px"]:.3f} px', ha='center', fontsize=8.5)
    s.save(fig, str(PAPER), 'P10_position_validation')
    record('P10_position_validation', 'manuscript',
           'Position estimates from the recorded 600-frame two-source synthetic run. Source membership comes from recorded simulation truth, not an inferred clustering result. Vertical dotted lines mark true x coordinates. The error scatter and two-dimensional RMSE assess position accuracy independently of the rendering kernel.',
           scope='recorded synthetic application run with known truth')
    fig, ax = panels(3, 2.8)
    x = np.arange(80)
    profiles = []
    for a, sigma in zip(ax, [.75, 1.5, 3.]):
        z = render_localization_image(ex['result'].localizations, (80, 80), sigma_px=sigma)
        profile = z.sum(axis=0); profile /= profile.max(); profiles.append(profile)
        a.plot(x, profile, color=s.COL_RECON)
        for xx in ex['centers'][:,0]: a.axvline(xx, color=s.COL_TRUTH, ls=':', lw=.7)
        a.set(xlim=(25,54), ylim=(0,1.1), xlabel='$x$ (camera pixels)', title=rf'Render $\sigma_r={sigma:g}$ px')
    ax[0].set_ylabel('Unit-peak column sum')
    fig.text(.5, .025, 'Identical estimated positions; only the display kernel changes.', ha='center', fontsize=8.5)
    csv_data('rendering_sensitivity', ['x_px','sigma_0p75','sigma_1p5','sigma_3'], np.c_[x,*profiles])
    s.save(fig, str(PAPER), 'P11_rendering_sensitivity')
    record('P11_rendering_sensitivity', 'manuscript',
           'The same localization list rendered by the application with Gaussian sigma 0.75, 1.5 or 3 camera pixels. Column-sum profiles are independently peak-normalized. Narrower display kernels change apparent peak widths and valley depths without improving the underlying position estimates or establishing structural resolution.',
           scope='controlled rendering variation on one recorded synthetic localization list')


def main():
    s.setup();DATA.mkdir(exist_ok=True)
    if HAS_MANUSCRIPT:
        PAPER.mkdir(parents=True,exist_ok=True)
    for fn in [T1,T2,T3,T4,T5,T6,T7,T8,T9]:fn()
    ex=result_figures();help_figures(ex)
    if HAS_MANUSCRIPT:
        model_mismatch();validation_figures(ex)
    (HERE/'figure_manifest.json').write_text(json.dumps(MANIFEST,indent=2)+'\n')
    print('Physics figures:',len(MANIFEST),'; synthetic run:',ex['parameters'])


if __name__=='__main__':main()
