'''
	plots isometric contours of a given coding approach with respect to the ideal setting where you have the full histogram available
'''
## Standard Library Imports
import os
import sys
sys.path.append('./tof-lib')

## Library Imports
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import open3d as o3d
from IPython.core import debugger
breakpoint = debugger.set_trace

## Local Imports
from research_utils import plot_utils, np_utils, io_ops
from eval_coding_gauss_mu_est import compose_fname
from scan_data_scripts.scan_data_utils import get_hist_img_fname, get_nt, get_unimodal_nt

def get_rec_algo_id(coding_id, account_irf=True):
	rec_algo_id = 'ncc'
	if('Gated' == coding_id): rec_algo_id =  'linear'
	elif('GatedZNCC' == coding_id): rec_algo_id =  'zncc'
	elif('Timestamp' == coding_id): rec_algo_id =  'matchfilt'
	elif('Identity' == coding_id): rec_algo_id =  'matchfilt'
	if(account_irf): rec_algo_id+='-irf'
	return rec_algo_id

def compose_fname(coding_id, n_codes, account_irf=True):
	rec_algo_id = get_rec_algo_id(coding_id, account_irf)
	if(coding_id == 'GatedZNCC'):
		return 'Gated_ncodes-{}_rec-{}.npz'.format(n_codes, rec_algo_id)
	else:
		return '{}_ncodes-{}_rec-{}.npz'.format(coding_id, n_codes, rec_algo_id)

def get_zmap_lims(scene_id):
	if(scene_id == '20190207_face_scanning_low_mu/free'): return (0.42, 0.63)
	elif(scene_id == '20190207_face_scanning_low_mu/ground_truth'): return (0.27, 0.51)
	else: return (0.1, 0.6)

def get_depth_lims(scene_id):
	if(scene_id == '20190207_face_scanning_low_mu/free'): return (0.29*1000, 0.5*1000)
	elif(scene_id == '20190207_face_scanning_low_mu/ground_truth'): return (0.12*1000, 0.36*1000)
	elif(scene_id == '20190209_deer_high_mu/free'): return (0.1*1000, 0.4*1000)
	else: return (0.1*1000, 0.6*1000)

def get_view_params(scene_id):
	(front, lookat, up, zoom) = (None,None,None,None)
	if(scene_id == '20190207_face_scanning_low_mu/free'): 
		front = [ 0.53934536951615974, -0.045708523079230172, -0.84084320970047255 ]
		lookat = [ 0.045659274947734085, 0.017748703115832466, 0.50725641455970027 ]
		up = [ -0.019677888208964399, 0.99756895718723848, -0.066850253342833732 ]
		zoom = 0.22999999999999982
	elif(scene_id == '20190207_face_scanning_low_mu/ground_truth'):
		# front = [ 0.70890177059360016, -0.04361026557816064, -0.7039576865021514 ]
		# lookat = [ 0.0052418660497504507, 0.013488347372243336, 0.43816082053425337 ]
		# up = [-0.071426187799642604, 0.9885165492929614, -0.13316655537463773 ]
		# zoom = 0.55
		front = [ 0.45720051231524506, -0.042279971471530994, -0.88835808970878627 ]
		lookat = [ 0.0052418660497504507, 0.013488347372243336, 0.43816082053425337 ]
		up = [ -0.029043899836118436, 0.99762677531826971, -0.06242811105888374 ]
		zoom = 0.87000000000000033
	return (front, lookat, up, zoom)

def vis_and_save_pcd(point_count_obj, out_fpath, front, lookat, up, zoom ):
	vis = o3d.visualization.Visualizer()
	vis.create_window(width=500, height=900, left=10, top=10)
	vis.add_geometry(point_count_obj)
	vc = vis.get_view_control()
	vc.set_front(front) 
	vc.set_lookat(lookat) 
	vc.set_up(up) 
	vc.set_zoom(zoom) 
	vis.update_geometry(point_count_obj)
	vis.poll_events()
	vis.update_renderer()
	vis.capture_screen_image(out_fpath)
	vis.destroy_window()

def vis_pcd_interactive(point_count_obj, out_fpath, front, lookat, up, zoom ):
	vis = o3d.visualization.Visualizer()
	vis.create_window(width=500, height=900, left=10, top=10)
	vis.add_geometry(point_count_obj)
	vc = vis.get_view_control()
	vc.set_front(front) 
	vc.set_lookat(lookat) 
	vc.set_up(up) 
	vc.set_zoom(zoom) 
	vis.run()

def save_xyz_pc(xyz_arr, dirpath, fname):
	pcd_obj = o3d.geometry.PointCloud()
	pcd_obj.points = o3d.utility.Vector3dVector(xyz_arr)
	o3d.io.write_point_cloud(os.path.join(dirpath, fname+'.xyz'), pcd_obj)

if __name__=='__main__':
	io_dirpaths = io_ops.load_json('./io_dirpaths.json')
	results_data_base_dirpath = io_dirpaths['results_data']
	scan_params = io_ops.load_json('./scan_data_scripts/scan_params.json')
	base_dirpath = scan_params["scan_data_base_dirpath"]
	plot_params = io_ops.load_json('./plotting_scripts/plot_params.json')
	out_base_dirpath = io_dirpaths['paper_results_dirpath']
	if(plot_params['dark_mode']):
		plt.style.use('dark_background')
		out_base_dirpath += '_dark'
	else:
		plt.rcParams["font.family"] = "Times New Roman"

	## Set parameters
	is_unimodal = True
	hist_tbin_factor = 1
	# scene_id = '20190209_deer_high_mu/free'
	scene_id = '20190207_face_scanning_low_mu/free'
	# scene_id = '20190207_face_scanning_low_mu/ground_truth'
	
	## Get scan params
	nr = scan_params['scene_params'][scene_id]['n_rows_fullres']
	nc = scan_params['scene_params'][scene_id]['n_cols_fullres']
	min_tbin_size = scan_params['min_tbin_size'] # Bin size in ps
	hist_tbin_size = min_tbin_size*hist_tbin_factor # increase size of time bin to make histogramming faster
	hist_img_tau = scan_params['hist_preprocessing_params']['hist_end_time'] - scan_params['hist_preprocessing_params']['hist_start_time']
	nt = get_nt(hist_img_tau, hist_tbin_size)
	unimodal_nt = get_unimodal_nt(nt, scan_params['irf_params']['pulse_len'], hist_tbin_size)
	unimodal_hist_img_tau = unimodal_nt*hist_tbin_size
	if(is_unimodal):
		nt = unimodal_nt
		hist_img_tau = unimodal_hist_img_tau
	hist_img_fname = get_hist_img_fname(nr, nc, hist_tbin_size, hist_img_tau, is_unimodal=is_unimodal)

	##
	data_dirpath = os.path.join(results_data_base_dirpath, 'scan_data_results', scene_id, hist_img_fname.split('.npy')[0])
	out_dirpath = os.path.join(out_base_dirpath, 'scan_data_results', scene_id, hist_img_fname.split('.npy')[0])
	os.makedirs(out_dirpath, exist_ok=True)

	## Get colors
	(min_plot_zmap, max_plot_zmap) = get_zmap_lims(scene_id)
	(min_plot_depth, max_plot_depth) = get_depth_lims(scene_id)
	(min_plot_err, max_plot_err) = (0, 40)
	cmap_norm = mpl.colors.Normalize(vmin=min_plot_zmap, vmax=max_plot_zmap)

	## Get visualization params
	(front, lookat, up, zoom) = get_view_params(scene_id)

	## Load estimated histogram stats (nphotons, sbr, etc)
	hist_stats_data = np.load(os.path.join(data_dirpath,'hist_stats.npz'))
	photon_count_img = hist_stats_data['nphotons']
	plt.clf()
	img = plt.imshow(photon_count_img)
	plot_utils.remove_ticks()
	plot_utils.save_currfig(dirpath = out_dirpath, filename = 'PhotonCounts', file_ext = 'svg')
	plot_utils.set_cbar(img, cbar_orientation='vertical', fontsize=18)
	plot_utils.save_currfig(dirpath = out_dirpath, filename = 'PhotonCounts-cbar', file_ext = 'svg')

	## Load Full-res histogram results
	frh_fname_base = compose_fname('Identity', nt, account_irf=True).split('.npz')[0]
	frh_results = np.load(os.path.join(data_dirpath, frh_fname_base+'.npz'))
	frh_decoded_xyz = frh_results['decoded_xyz']
	frh_decoded_zmap = frh_results['decoded_zmap']
	frh_decoded_depths = frh_results['decoded_depths']*1000
	frh_decoded_depth_errs = frh_results['depth_errs']
	frh_medfilt_decoded_xyz = frh_results['medfilt_decoded_xyz']
	frh_medfilt_decoded_depths = frh_results['medfilt_decoded_depths']*1000
	gt_decoded_xyz = frh_results['gt_xyz']
	validsignal_mask = np.logical_not(np.isnan(frh_decoded_depth_errs))
	frh_pcd = o3d.geometry.PointCloud()
	frh_pcd.points = o3d.utility.Vector3dVector(frh_decoded_xyz)
	o3d.io.write_point_cloud(os.path.join(data_dirpath, frh_fname_base+'.xyz'), frh_pcd)
	frh_cmap_colors = plt.get_cmap()(cmap_norm(frh_decoded_xyz[:,-1]))[:,0:3]
	frh_pcd.colors = o3d.utility.Vector3dVector(frh_cmap_colors)
	frh_out_fpath = os.path.join(out_dirpath, "PointCloud_"+frh_fname_base+'.png')
	
	## Save Identity and Ground truth point clouds
	save_xyz_pc(frh_decoded_xyz, dirpath=data_dirpath, fname=frh_fname_base)
	save_xyz_pc(frh_medfilt_decoded_xyz, dirpath=data_dirpath, fname='MedFilt-'+frh_fname_base)
	save_xyz_pc(gt_decoded_xyz, dirpath=data_dirpath, fname='GT-'+frh_fname_base)
	
	# vis_and_save_pcd(frh_pcd, out_fpath=frh_out_fpath, front=front, lookat=lookat, up=up, zoom=zoom)
	# ## Save point cloud
	# o3d.io.write_point_cloud(os.path.join(data_dirpath, "PointCloud_"+frh_fname_base+'.xyz'), frh_pcd)
	## Plot depth and depth errors
	plt.clf()
	img = plt.imshow(frh_decoded_depths, vmin=min_plot_depth, vmax=max_plot_depth)
	plot_utils.remove_ticks()
	plot_utils.save_currfig(dirpath = out_dirpath, filename = 'Depthmap_'+frh_fname_base, file_ext = 'svg')
	plot_utils.set_cbar(img, cbar_orientation='vertical', fontsize=18)
	plot_utils.save_currfig(dirpath = out_dirpath, filename = 'Depthmap-cbar_'+frh_fname_base, file_ext = 'svg')
	plt.clf()
	img = plt.imshow(frh_decoded_depth_errs, vmin=min_plot_err, vmax=max_plot_err)
	plot_utils.remove_ticks()
	plot_utils.save_currfig(dirpath = out_dirpath, filename = 'DepthErrors_'+frh_fname_base, file_ext = 'svg')
	plot_utils.set_cbar(img, cbar_orientation='vertical', fontsize=18)
	plot_utils.save_currfig(dirpath = out_dirpath, filename = 'DepthErrors-cbar_'+frh_fname_base, file_ext = 'svg')

	## plot depths to get the colormap
	# plt.clf()
	# img = plt.imshow(frh_decoded_zmap, vmin=min_plot_zmap, vmax=max_plot_zmap)

	## Set coding schemes we want to plot
	account_irf = True
	n_codes = 8
	coding_ids = ['TruncatedFourier', 'PSeriesFourier', 'PSeriesGray', 'Gated', 'GatedZNCC']
	# coding_ids = ['PSeriesGray']
	# coding_ids = ['TruncatedFourier']

	# idx = [0]
	idx = np.arange(0,len(coding_ids))

	for i in idx:
		## Load data
		fname = compose_fname(coding_ids[i], n_codes, account_irf=account_irf)
		fname_base = fname.split('.npz')[0]
		results = np.load(os.path.join(data_dirpath, fname), allow_pickle=True)
		out_fpath = os.path.join(out_dirpath, fname.replace('.npz', '.png'))
		out_pcd_fpath = os.path.join(data_dirpath, fname.replace('.npz', '.xyz'))
		decoded_xyz = results['decoded_xyz']
		decoded_zmap = results['decoded_zmap']
		decoded_depths = results['decoded_depths']*1000
		decoded_depth_errs = results['depth_errs']
		decoded_depths[np.logical_not(validsignal_mask)] = np.nan
		medfilt_decoded_xyz = results['medfilt_decoded_xyz']
		medfilt_decoded_depths = results['medfilt_decoded_depths']*1000
		medfilt_decoded_depths[np.logical_not(validsignal_mask)] = np.nan
		masked_depth_error_metrics = results['masked_depth_error_metrics'].item()
		save_xyz_pc(decoded_xyz, dirpath=data_dirpath, fname=fname_base)
		save_xyz_pc(medfilt_decoded_xyz, dirpath=data_dirpath, fname='MedFilt-'+fname_base)
		print(fname+":")
		np_utils.print_error_metrics(masked_depth_error_metrics, '    ')
		## Plot depth and depth errors
		plt.clf()
		img = plt.imshow(decoded_depths, vmin=min_plot_depth, vmax=max_plot_depth)
		plot_utils.remove_ticks()
		plot_utils.save_currfig(dirpath = out_dirpath, filename = 'Depthmap_'+fname_base, file_ext = 'svg')
		plot_utils.set_cbar(img, cbar_orientation='vertical', fontsize=20)
		plot_utils.save_currfig(dirpath = out_dirpath, filename = 'Depthmap-cbar_'+fname_base, file_ext = 'svg')

		plt.clf()
		img = plt.imshow(medfilt_decoded_depths, vmin=min_plot_depth, vmax=max_plot_depth)
		plot_utils.remove_ticks()
		plot_utils.save_currfig(dirpath = out_dirpath, filename = 'MedFilt-Depthmap_'+fname_base, file_ext = 'svg')
		plot_utils.set_cbar(img, cbar_orientation='vertical', fontsize=20)
		plot_utils.save_currfig(dirpath = out_dirpath, filename = 'MedFilt-Depthmap-cbar_'+fname_base, file_ext = 'svg')

		plt.clf()
		img = plt.imshow(decoded_depth_errs, vmin=min_plot_err, vmax=max_plot_err)
		plot_utils.remove_ticks()
		plot_utils.save_currfig(dirpath = out_dirpath, filename = 'DepthErrors_'+fname_base, file_ext = 'svg')
		plot_utils.set_cbar(img, cbar_orientation='vertical', fontsize=20)
		plot_utils.save_currfig(dirpath = out_dirpath, filename = 'DepthErrors-cbar_'+fname_base, file_ext = 'svg')


		# ## Get colors
		# cmap_colors = plt.get_cmap()(cmap_norm(decoded_xyz[:,-1]))[:,0:3]
		# ## Create and save point cloud
		# pcd = o3d.geometry.PointCloud()
		# pcd.points = o3d.utility.Vector3dVector(decoded_xyz)
		# pcd.colors = o3d.utility.Vector3dVector(cmap_colors)
		# o3d.io.write_point_cloud(out_pcd_fpath, pcd)
		
		## Visualize Point Cloud and save
		# vis_and_save_pcd(pcd, out_fpath=out_fpath, front=front, lookat=lookat, up=up, zoom=zoom)
		# vis_pcd_interactive(cl, out_fpath=out_fpath, front=front, lookat=lookat, up=up, zoom=zoom)
		# vis.run()
		
		## Pre-process point cloud
		# cl, ind = pcd.remove_radius_outlier(nb_points=16, radius=0.01)
		# cl.estimate_normals()
		# # radii = [0.02, 0.04, 0.08, 0.1]
		# # rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
		# # 	cl, o3d.utility.DoubleVector(radii))
		# rec_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cl, depth=20)
		# # o3d.visualization.draw_geometries([cl, rec_mesh])
		# # o3d.visualization.draw_geometries([rec_mesh], mesh_show_back_face=True)




