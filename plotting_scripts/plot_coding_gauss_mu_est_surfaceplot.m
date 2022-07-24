% This script loads the results from the coding_gauss_mu_est for multiple
% coding approaches at a fixed ncodes, and plots a surface plot of their
% performance across SBR and nphotons levels

clear;
color_list = get(gca,'ColorOrder');


%%% Get dirpath where results data is in
io_dirpaths = loadjson('../io_dirpaths.json');
base_dirpath = fullfile('../', io_dirpaths.results_data);
out_dirpath = fullfile('../', io_dirpaths.paper_results_dirpath, '/MAESimulations');

%%% Set params and generate the dirpath where the data is stored
max_rel_error = 0.1;
n_tbins = 1024;
absmin_logsbr = -2; 
absmax_logsbr = 1;
absmin_lognphotons = 0; 
absmax_lognphotons = 2;
rel_dirpath_fmt = 'final_coding_gauss_mu_est/ntbins-%d_logsbr-%0.1f-%0.1f_lognp-%0.1f-%0.1f';
rel_dirpath = sprintf(rel_dirpath_fmt, n_tbins, absmin_logsbr, absmax_logsbr, absmin_lognphotons, absmax_lognphotons);
data_dirpath = fullfile(base_dirpath, rel_dirpath);

%%% Set params for which to plot
is_high_flux = true;
is_high_sbr = false;
is_photonstarved = absmin_lognphotons < 2;
pw_factor_shared = 10;
is_narrow_pulse = pw_factor_shared <= 1;


% ncodes = 16;
% coding_ids = {'TruncatedFourier', 'PSeriesFourier', 'Timestamp'};
% rec_algo_ids = {'ncc-irf', 'ncc-irf', 'matchfilt-irf'};    

% ncodes = 32;
% coding_ids = {'PSeriesGray','TruncatedFourier', 'PSeriesFourier' 'Gated', 'GatedWide', 'Timestamp'};
% rec_algo_ids = {'ncc-irf', 'ncc-irf', 'ncc-irf', 'linear-irf', 'linear-irf', 'matchfilt-irf'};    

ncodes = 64;
coding_ids = {'TruncatedFourier', 'PSeriesFourier' 'Gated', 'GatedWide', 'Timestamp'};
rec_algo_ids = {'ncc-irf', 'ncc-irf', 'linear-irf', 'linear-irf', 'matchfilt-irf'};    

% coding_ids = {'TruncatedFourier','WalshHadamard', 'Timestamp'};
% rec_algo_ids = {'ncc-irf','ncc-irf', 'matchfilt-irf'};    
% 
% coding_ids = {'TruncatedFourier','WalshHadamard', 'GatedFourier-F-1'};
% rec_algo_ids = {'ncc-irf','ncc-irf', 'ncc-irf'};    

%%% Set coding IDS and their params
n_coding_schemes = numel(coding_ids);
fname_fmt = '%s_ncodes-%d_rec-%s_pw-%0.1f.mat';

coding_scheme_ids = {};
for i = 1:n_coding_schemes
    coding_scheme_ids{i} = sprintf('%s-%s-K-%d', coding_ids{i}, rec_algo_ids{i}, ncodes); 
end

%%% Plot only SBR and nphoton values within this range
[min_plot_sbr, max_plot_sbr, min_plot_nphotons, max_plot_nphotons] = get_plot_snr_settings(is_high_flux, is_high_sbr, is_photonstarved, ncodes);


out_fname_fmt = '%s-surfplt_ncodes-%d_sbr-%0.2f-%0.2f_np-%d-%d.svg';
out_fname_mae = sprintf(out_fname_fmt, 'MAE', ncodes, min_plot_sbr, max_plot_sbr, min_plot_nphotons, max_plot_nphotons);
out_fname_medae = sprintf(out_fname_fmt, 'MEDAE', ncodes, min_plot_sbr, max_plot_sbr, min_plot_nphotons, max_plot_nphotons); 

%%% Load data and plot MAE
close all;
for i = 1:n_coding_schemes
    %%% Load data
    if(strcmp(coding_ids{i},'GatedWide'))
        fname = sprintf(fname_fmt, 'Gated', ncodes, rec_algo_ids{i}, double(n_tbins/ncodes));
    else
        fname = sprintf(fname_fmt, coding_ids{i}, ncodes, rec_algo_ids{i}, pw_factor_shared);        
    end
    fpath = fullfile(data_dirpath, fname);
    data = load(fpath);
    rel_errs = 100*(data.metric_mae / n_tbins);
    plotmaesurf(data, min_plot_sbr, max_plot_sbr, min_plot_nphotons, max_plot_nphotons, color_list(i,:));
end
%%% Plot the results for full histogram
fname = sprintf(fname_fmt, 'Identity', n_tbins, 'matchfilt-irf', pw_factor_shared);
fpath = fullfile(data_dirpath, fname);
data = load(fpath);
plotmaesurf(data, min_plot_sbr, max_plot_sbr, min_plot_nphotons, max_plot_nphotons, color_list(i+1,:));
set_zlim(is_high_flux, is_high_sbr, is_narrow_pulse, ncodes);
setplotparams()

% set(gcf, 'Color', 'None');
% set(gca, 'Color', 'None');
tightlayout();
saveas(gca,fullfile(out_dirpath, out_fname_mae),'svg');
% export_fig(fullfile(out_dirpath, out_fname_mae), '-svg');
% plot2svg(fullfile(out_dirpath, out_fname_mae));

% exportgraphics(gca,fullfile(out_dirpath, out_fname_mae),'BackgroundColor','none','ContentType','vector')

% Set Plot parameters
legend(coding_scheme_ids, 'Location', 'best', 'FontSize', 13, 'NumColumns', 2);
title('REL MEAN ABS ERR')
ylabel('Log Photon Counts');
xlabel('Log SBR Levels');
zlabel('Relative Error (%)');



% %%% Load data and plot MEDAE
% figure;
% for i = 1:n_coding_schemes
%     %%% Load data
%     if(strcmp(coding_ids{i},'GatedWide'))
%         fname = sprintf(fname_fmt, 'Gated', ncodes, rec_algo_ids{i}, double(n_tbins/ncodes));
%     else
%         fname = sprintf(fname_fmt, coding_ids{i}, ncodes, rec_algo_ids{i}, pw_factor_shared);        
%     end
%     fpath = fullfile(data_dirpath, fname);
%     data = load(fpath);
%     plotmedaesurf(data, min_plot_sbr, max_plot_sbr, min_plot_nphotons, max_plot_nphotons, color_list(i,:));
% end
% %%% Plot the results for full histogram
% fname = sprintf(fname_fmt, 'Identity', n_tbins, 'matchfilt-irf', 1.0);
% fpath = fullfile(data_dirpath, fname);
% data = load(fpath);
% plotmedaesurf(data, min_plot_sbr, max_plot_sbr, min_plot_nphotons, max_plot_nphotons, color_list(i+1,:));
% setplotparams()
% set_zlim(is_high_flux, is_high_sbr, is_narrow_pulse, ncodes);
% 
% % set(gcf, 'Color', 'None');
% % set(gca, 'Color', 'None');
% tightlayout();
% saveas(gca,fullfile(out_dirpath, out_fname_medae),'svg');
% % exportgraphics(gca,fullfile(out_dirpath, out_fname_medae),'BackgroundColor','none','ContentType','vector')
% 
% % Set Plot parameters
% legend(coding_scheme_ids, 'Location', 'best', 'FontSize', 13, 'NumColumns', 2);
% title('REL MEDIAN ABS ERR')
% ylabel('Log Photon Counts');
% xlabel('Log SBR Levels');
% zlabel('Relative Error (%)');



%%%%%% Functions
function [json_struct] = loadjson(fpath)
    fid = fopen(fpath); 
    raw = fread(fid,inf); 
    str = char(raw'); 
    fclose(fid); 
    json_struct = jsondecode(str);
end

function [] = setplotparams()
    % Set up the figure properties
    fig = gcf;
    % Set up the axes properties
    ax = gca;
    ax.FontName = 'LaTeX';
    ax.Box = 'off';
    ax.LineWidth = 2;
    ax.FontSize = 18;
    set(gca, 'YScale', 'log')
    set(gca, 'XScale', 'log')
%     set(gca, 'ZScale', 'log')
    %%% Finalize plot
    grid on;
    view(50,20);
    set(gcf, 'Position',  [-800, 300, 700, 500])
end

function [] = setextraplotparams()

end

function [] = plotsurf(data, min_plot_sbr, max_plot_sbr, min_plot_nphotons, max_plot_nphotons, color )
    %%% Only plot points withing the logsbr
    X_sbr_levels= data.X_sbr_levels;
    Y_nphotons_levels = data.Y_nphotons_levels;
    rel_mae = 100*data.metric_mae/ double(data.n_tbins);
    mask1 = (X_sbr_levels>= min_plot_sbr) & (X_sbr_levels<= max_plot_sbr);
    mask2 = (Y_nphotons_levels >= min_plot_nphotons) & (Y_nphotons_levels <= max_plot_nphotons);
    mask = not(mask1 & mask2);
    %%% Plot - Set the values we don't want to include as NaN
    X_sbr_levels(mask) = NaN;
    Y_nphotons_levels(mask) = NaN;
    rel_mae(mask) = NaN;
    surf(X_sbr_levels, Y_nphotons_levels, rel_mae,... 
        'FaceColor', color, 'FaceAlpha', 0.7);
    hold on;
end

function [] = plotmaesurf(data, min_plot_sbr, max_plot_sbr, min_plot_nphotons, max_plot_nphotons, color )
    %%% Only plot points withing the logsbr
    X_sbr_levels= data.X_sbr_levels;
    Y_nphotons_levels = data.Y_nphotons_levels;
    rel_mae = 100*data.metric_mae/ double(data.n_tbins);
    mask1 = (X_sbr_levels>= min_plot_sbr) & (X_sbr_levels<= max_plot_sbr);
    mask2 = (Y_nphotons_levels >= min_plot_nphotons) & (Y_nphotons_levels <= max_plot_nphotons);
    mask = not(mask1 & mask2);
    %%% Plot - Set the values we don't want to include as NaN
    X_sbr_levels(mask) = NaN;
    Y_nphotons_levels(mask) = NaN;
    rel_mae(mask) = NaN;
    surf(X_sbr_levels, Y_nphotons_levels, rel_mae,... 
        'FaceColor', color, 'FaceAlpha', 0.7);
    hold on;
end

function [] = plotmedaesurf(data, min_plot_sbr, max_plot_sbr, min_plot_nphotons, max_plot_nphotons, color )
    %%% Only plot points withing the logsbr
    X_sbr_levels= data.X_sbr_levels;
    Y_nphotons_levels = data.Y_nphotons_levels;
    rel_medae = 100*data.metric_medae/ double(data.n_tbins);
    mask1 = (X_sbr_levels>= min_plot_sbr) & (X_sbr_levels<= max_plot_sbr);
    mask2 = (Y_nphotons_levels >= min_plot_nphotons) & (Y_nphotons_levels <= max_plot_nphotons);
    mask = not(mask1 & mask2);
    %%% Plot - Set the values we don't want to include as NaN
    X_sbr_levels(mask) = NaN;
    Y_nphotons_levels(mask) = NaN;
    rel_medae(mask) = NaN;
    surf(X_sbr_levels, Y_nphotons_levels, rel_medae,... 
        'FaceColor', color, 'FaceAlpha', 0.7);
    hold on;
end

function [min_plot_sbr, max_plot_sbr, min_plot_nphotons, max_plot_nphotons] = get_plot_snr_settings(is_high_flux, is_high_sbr, is_photonstarved,ncodes)



    if(is_photonstarved)
        if(ncodes <= 20)
            min_plot_nphotons = 10;
            max_plot_nphotons = 20;
        elseif(ncodes <= 50)
            min_plot_nphotons = 20;
            max_plot_nphotons = 40;
        else
            min_plot_nphotons = 40;
            max_plot_nphotons = 80;
        end
        if(is_high_sbr)
            min_plot_sbr = 1.0;
            max_plot_sbr = 10.0;
        else
            min_plot_sbr = 0.1;
            max_plot_sbr = 1.0;        
        end
    else
        if(is_high_flux)
            min_plot_nphotons = 500;
            max_plot_nphotons = 10000;
        else
            min_plot_nphotons = 100;
            max_plot_nphotons = 1000;        
        end
        if(is_high_sbr)
            % Used for main paper
            min_plot_sbr = 0.05;
            max_plot_sbr = 1.0;
%             % Used for supplement to cover wider range
%             min_plot_sbr = 0.08;
%             max_plot_sbr = 1.0;
        else
            min_plot_sbr = 0.01;
            max_plot_sbr = 0.08;        
        end
    end

    
end

function [] = set_zlim(is_high_flux, is_high_sbr, is_narrow_pulse, ncodes)
    if((is_high_flux && is_high_sbr && is_narrow_pulse) && (ncodes==8)) 
        % Used for main paper
%         zlim([0,5])
        zlim([0,4])
    elseif((is_high_flux && is_high_sbr && is_narrow_pulse) && (ncodes==16))
        zlim([0,2])
    elseif((is_high_flux && is_high_sbr && is_narrow_pulse) && (ncodes==32))
        zlim([0,1])
    elseif((is_high_flux && is_high_sbr && is_narrow_pulse) && (ncodes==64))
        zlim([0,0.5])
    elseif((is_high_flux && is_high_sbr && is_narrow_pulse) && (ncodes==128))
        zlim([0,0.5])
    end
end

function [] = tightlayout()
ax = gca;
outerpos = ax.OuterPosition;
ti = ax.TightInset; 
left = outerpos(1) + ti(1);
bottom = outerpos(2) + ti(2);
ax_width = outerpos(3) - ti(1) - ti(3);
ax_height = outerpos(4) - ti(2) - ti(4);
ax.Position = [left bottom ax_width ax_height];
end


