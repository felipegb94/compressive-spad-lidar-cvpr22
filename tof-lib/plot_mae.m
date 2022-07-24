% Plot all json
clear
% close all
color_list = get(gca,'ColorOrder');

coding_ids = {'Fourier', 'HamK4', 'TruncatedFourier'};
coding_ids = {'TruncatedFourier', 'TruncatedFourier', 'TruncatedFourier'};
rec_algo_ids = {'ifft', 'pizarenko', 'zncc' };
coding_ids = {'TruncatedFourier', 'OptC'};
rec_algo_ids = {'ifft', 'zncc' };

pw_factors= {'1.0', '1.0', '1.0' };

n_coding_ids = numel(coding_ids);


clf;
% Set up the figure properties
fig = gcf;
% Set up the axes properties
ax = gca;
ax.FontName = 'LaTeX';
ax.Box = 'off';
ax.LineWidth = 2;
legend_strs = {};

for i = 1:n_coding_ids
    
    coding_scheme = [coding_ids{i}, '-', rec_algo_ids{i}, '-pw-', pw_factors{i}];
    legend_strs{i} = coding_scheme;
    filename = ['mae_', coding_scheme, '.json']
    
    results_dict = load_json(filename);
    
    disp(results_dict.Z)
    
    surf(log10(results_dict.X), log10(results_dict.Y), results_dict.Z, 'FaceColor', color_list(i,:), 'FaceAlpha', 0.7);
    hold on;
    
end

xlabel(['Log ', results_dict.X_label],'FontSize',14)
ylabel(['Log ', results_dict.Y_label],'FontSize',14)
zlabel(['', results_dict.Z_label],'FontSize',14)
legend(legend_strs, 'Location', 'North','FontSize',14)

% zlim([0,175])
view(60,10);
set(gcf, 'Position',  [-800, 300, 700, 500])