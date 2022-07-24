clear;
% close all;
color_list = get(gca,'ColorOrder');


dirpath = '../../../results/sptof_sim_mae_results';
filename1 = 'mae_ntbins-256_drange-10.0_pwidth-1.00_nfreqs-16';
filename2 = 'mae_ntbins-512_drange-10.0_pwidth-1.00_nfreqs-16';
filename3 = 'mae_ntbins-64_drange-10.0_pwidth-1.00_nfreqs-16';
filename4 = 'mae_ntbins-64_drange-10.0_pwidth-0.25_nfreqs-16';
% filename3 = 'mae_ntbins-128_drange-10.0_pwidth-1.00_nfreqs-16';
% filename4 = 'mae_ntbins-128_drange-10.0_pwidth-0.50_nfreqs-16';
% filename4 = 'mae_ntbins-128_drange-10.0_pwidth-0.25_nfreqs-16';
filepath1 = [dirpath, '/', filename1, '.mat'];
filepath2 = [dirpath, '/', filename2, '.mat'];
filepath3 = [dirpath, '/', filename3, '.mat'];
filepath4 = [dirpath, '/', filename4, '.mat'];

 
results_dict1 = load(filepath1);
results_dict2 = load(filepath2);
results_dict3 = load(filepath3);
results_dict4 = load(filepath4);


nfreqs1 = results_dict1.n_freqs;
nfreqs2 = results_dict2.n_freqs;
nfreqs3 = results_dict3.n_freqs;
nfreqs4 = results_dict4.n_freqs;
ntbins1 = results_dict1.n_tbins;
ntbins2 = results_dict2.n_tbins;
ntbins3 = results_dict3.n_tbins;
ntbins4 = results_dict4.n_tbins;
tres1 = round(results_dict1.time_res*1e12);
tres2 = round(results_dict2.time_res*1e12);
tres3 = round(results_dict3.time_res*1e12);
tres4 = round(results_dict4.time_res*1e12);
pwidth1 = round(tres1*results_dict1.pulse_width_factor);
pwidth2 = round(tres2*results_dict2.pulse_width_factor);
pwidth3 = round(tres3*results_dict3.pulse_width_factor);
pwidth4 = round(tres4*results_dict4.pulse_width_factor);



quantization_error1 = floor(0.5*results_dict1.tbin_depth_res*1000);
quantization_error2 = floor(0.5*results_dict2.tbin_depth_res*1000);
quantization_error3 = floor(0.5*results_dict3.tbin_depth_res*1000);
quantization_error4 = floor(0.5*results_dict4.tbin_depth_res*1000);

max_depth = results_dict1.max_depth;
photon_levels = results_dict1.photon_levels;
sbr_levels = results_dict1.sbr_levels;
start_photon_level_idx = 10;
photon_levels = photon_levels(start_photon_level_idx:end);
start_sbr_level_idx = 3;
sbr_levels = sbr_levels(start_sbr_level_idx:end);
log_photon_levels = log10(photon_levels);
log_sbr_levels = log10(sbr_levels);
[X,Y] = meshgrid(log_sbr_levels, log_photon_levels);


mle_mae_all1 = results_dict1.mle_mae_all(start_photon_level_idx:end, start_sbr_level_idx:end);
ift_mp_mae_all1 = results_dict1.ift_mp_mae_all(start_photon_level_idx:end, start_sbr_level_idx:end);
mle_mae_all2 = results_dict2.mle_mae_all(start_photon_level_idx:end, start_sbr_level_idx:end);
ift_mp_mae_all2 = results_dict2.ift_mp_mae_all(start_photon_level_idx:end, start_sbr_level_idx:end);
mle_mae_all3 = results_dict3.mle_mae_all(start_photon_level_idx:end, start_sbr_level_idx:end);
ift_mp_mae_all3 = results_dict3.ift_mp_mae_all(start_photon_level_idx:end, start_sbr_level_idx:end);
mle_mae_all4 = results_dict4.mle_mae_all(start_photon_level_idx:end, start_sbr_level_idx:end);
ift_mp_mae_all4 = results_dict4.ift_mp_mae_all(start_photon_level_idx:end, start_sbr_level_idx:end);

quantization_limit1 = ones(size(ift_mp_mae_all1))*quantization_error1;
quantization_limit2 = ones(size(ift_mp_mae_all2))*quantization_error2;
quantization_limit3 = ones(size(ift_mp_mae_all3))*quantization_error3;
quantization_limit4 = ones(size(ift_mp_mae_all4))*quantization_error4;


clf;

% Set up the figure properties
fig = gcf;
% Set up the axes properties
ax = gca;
ax.FontName = 'LaTeX';
ax.Box = 'off';
ax.LineWidth = 2;

surf(X, Y, mle_mae_all1*1000, 'FaceColor', color_list(1,:), 'FaceAlpha', 0.7);
hold on;
surf(X, Y, mle_mae_all3*1000, 'FaceColor', color_list(2,:), 'FaceAlpha', 0.7);
surf(X, Y, mle_mae_all4*1000, 'FaceColor', color_list(3,:), 'FaceAlpha', 0.7);
surf(X, Y, ift_mp_mae_all1*1000, 'FaceColor', color_list(4,:), 'FaceAlpha', 0.7);
surf(X, Y, quantization_limit1, 'FaceColor', color_list(5,:), 'FaceAlpha', 1);
surf(X, Y, quantization_limit3, 'FaceColor', color_list(6,:), 'FaceAlpha', 1);
legend(...
    "MLE, tres="+tres1+"ps, ntbins="+ntbins1+" pulse width="+pwidth1+"ps", ...
    "MLE, tres="+tres3+"ps, ntbins="+ntbins3+" pulse width="+pwidth3+"ps", ...
    "MLE, tres="+tres4+"ps, ntbins="+ntbins4+" pulse width="+pwidth4+"ps", ...
    "Fourier, tres="+tres1+"ps, nfreq="+nfreqs1+" pulse width="+pwidth1+"ps", ...
    "Quantization Limit="+quantization_error1+"mm, tres="+tres1+"ps", ...
    "Quantization Limit="+quantization_error3+"mm, tres="+tres3+"ps", ...
    'Location', 'North','FontSize',14);
grid on;
view(45,10);
ylabel('Log Photon Counts','FontSize',14);
xlabel('Log SBR Levels','FontSize',14);
zlabel('MAE (mm)','FontSize',14);
zlim([0,175])
title("MAE Over "+max_depth+" Meter Depth Range",'FontSize',16)
set(gcf, 'Position',  [-800, 300, 700, 500])


clf;
surf(X, Y, mle_mae_all2*1000, 'FaceColor', color_list(1,:), 'FaceAlpha', 0.7);
hold on;
surf(X, Y, mle_mae_all1*1000, 'FaceColor', color_list(2,:), 'FaceAlpha', 0.7);
surf(X, Y, ift_mp_mae_all2*1000, 'FaceColor', color_list(3,:), 'FaceAlpha', 0.7);
surf(X, Y, quantization_limit1, 'FaceColor', color_list(4,:), 'FaceAlpha', 0.5);
surf(X, Y, quantization_limit2, 'FaceColor', color_list(5,:), 'FaceAlpha', 0.5);
legend( ...
    "MLE, tres="+tres2+"ps, ntbins="+ntbins2+" pulse width="+pwidth2+"ps", ...
    "MLE, tres="+tres1+"ps, ntbins="+ntbins1+" pulse width="+pwidth1+"ps", ...
    "Fourier, tres="+tres2+"ps, nfreq="+nfreqs2+" pulse width="+pwidth2+"ps", ...
    "Quantization Limit="+quantization_error1+"mm, tres="+tres1+"ps", ...
    "Quantization Limit="+quantization_error2+"mm, tres="+tres2+"ps", ...
    'Location', 'North','FontSize',14);
grid on;
view(45,10);
ylabel('Log Photon Counts','FontSize',14);
xlabel('Log SBR Levels','FontSize',14);
zlabel('MAE (mm)','FontSize',14);
zlim([0,80])







