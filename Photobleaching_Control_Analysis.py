# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 12:31:42 2024

@author: nka205
"""

import os
from tirf_tools import io, corrections, PeakFitter, integrator, colocalizer
from tirf_tools import hmm_tools as ht
import tqdm
from tirf_tools  import hmm_plots as hp
from matplotlib import pyplot as plt
import napari
import numpy as np

#%% load data

data = io.load_image()

#%% correct data

corrections.correct_data(data, 2, 422)

#%%

corrections.drift_correct(data,key='corr_data')

#%% make projection

data['proj'] = np.mean(data['corr_data'], axis = 0)

#%% find peaks

peaks = PeakFitter.peak_finder(data['proj'], roi=[10,10,512,512],
                       threshold_rel= 0.075,
                       min_dist = 2)

#%%

v = napari.Viewer()

#%%

v.add_image(data['corr_data'], name = data['filename'], colormap = 'green')
# v.add_image(data[1]['drift_corr'], name = data[1]['filename'], colormap = 'green')

#%%

v.add_image(data['proj'], name = 'Peaksmean', colormap = 'green')

#%%
v.add_points(peaks, name = 'Peaks', face_color = 'transparent', edge_color = 'magenta')

#%%
inner_radius = 2
outer_radius = 3
data['integration'] = integrator.integrate_trajectories(data['corr_data'], peaks, inner_radius, outer_radius)

#data[1]['integration'] = integrator.integrate_trajectories(data[1]['corr_data'], np.array(coloc_peaks[['1_x','1_y']]), inner_radius, outer_radius)

#%% hmm analysis Part1: calibration and filtering

time_conversion = 0.4
#calibrate frames to seconds
ht.calibrate_dataset(data['integration'], 
                     time_conversion, 
                     traj_column = 'trajectory', 
                     time_column = 'slice')



#%%

# filter out garbage such as big bright aggregates, as well as trajectories with 
# very negatve intensities
intensity_threshold = 5
filtered, fig_filter = ht.filter_garbage(data['integration'], 
                                  intensity_column = 'Intensity',
                                  traj_column = 'trajectory', 
                                  threshold =intensity_threshold , 
                                  plot = True)
ax = fig_filter.get_axes()
ax[0].ticklabel_format(axis="x", style="sci", scilimits=(0,0))
ax[1].ticklabel_format(axis="x", style="sci", scilimits=(0,0))
title = ax[1].title.get_text()
new_title = title + '\n theshold: mean + {}*std'.format(intensity_threshold)
ax[1].set_title(new_title)

#%%
# hmm analysis Part2: determine an initial guess and do two rounds of fitting

#find initial guess for markov model by gaussian fit
popt,init_fig = ht.find_initial_guess(filtered, 
                             intensity_column = 'Intensity', 
                             showFig = False, 
                             return_fig=True,
                             figName = 'initial guess', 
                             p0 = [0,100000,50000,50000,0.1,1])
init_fig.show()

#%%
print('initializing HMM')

SNR_threshold = 0.0 #distance between on and off distributions in units of standard deviation
min_events = 0      #minimum number of binding events

model = ht.init_hmm(n_components=2,init_params= "",initial_guess = None)

fig_train1, score = ht.train_hmm(filtered,model, fraction = 1.0, traj_column = 'trajectory' ,intensity_column='Intensity')
#fit the data with the trained model    
ht.fit_dataset(filtered,model,intensity_column = 'Intensity')
#this maps the states (i.e. 0 or 1) to the most likely intensity
ht.map_states_dataset(filtered,model)
# filter by SNR ish and number of binding events
refiltered,bad = ht.filter_SNR(filtered,SNR_threshold, return_bad=True, intensity_column = 'Intensity')
refiltered = ht.filter_inactive(refiltered, threshold = min_events) #threshold is number of binding events
# now there should be a lot less garbage so update, i.e. retrain the hmm
figure_hmm, score = ht.train_hmm(refiltered,model, 
                                 fraction = 1.0, 
                                 traj_column = 'trajectory',
                                 intensity_column='Intensity')
#finally fit the data with the retrained model
ht.fit_dataset(refiltered,model,
               intensity_column = 'Intensity')
ht.map_states_dataset(refiltered,model)

#make a figure
ax = figure_hmm.get_axes()
ax[0].set_title('Intensity distribution for all trajectories n={}, '.format(len(refiltered['trajectory'].unique())))
ax[0].set_xlabel('Intensity (arb units)', size = 15)
ax[0].set_ylabel('Probability density', size = 15)

title = ax[0].title.get_text()
new_title = title + '\n SNR threshold:{}, min binding events: {}'.format(SNR_threshold,min_events)
ax[0].set_title(new_title)
figure_hmm.show()


#%% get pulse statistics

pulses, off = ht.get_pulses_dataset(refiltered)
allDwells, meanDwells = ht.get_dwelltimes_dataset_pd(refiltered, method = 'mean')
pulses['dur'] = pulses['dur']*time_conversion
pulses = ht.filter_by_length(pulses)
fig_dwelltimes = hp.dwelltime_distributions(pulses, off, bins = 'fd')

#%%

from scipy import stats

# bleaching = pulses[pulses['start_frame']==0]

fit = stats.expon.fit(pulses['dur'],floc=0)

#%%


newpath = data['folder']+'/' + data['filename']+'_all_trajectory/' 
if not os.path.exists(newpath):
    print(newpath)
    os.makedirs(newpath)



#%%
for name, group in tqdm.tqdm(refiltered.groupby('trajectory')):
    fig, ax = plt.subplots()
    ax.plot(group['seconds'],group['Intensity'])
    ax.plot(group['seconds'],group['mapped_fit'])
    # ax.plot(group['seconds'],group['mapped_fit'])
    fig.savefig(newpath+'trajectory_'+str(name)+'.png')
    plt.close(fig)

#%%

max_time = 623 * time_conversion  # Convert 250 frames into seconds
x = np.arange(0, 250, 5)
# x = np.arange(0, 600, 0.1)
plt.figure()
plt.title('Pol Lifetime')
plt.hist(pulses['dur'], 
         edgecolor = 'black',
         bins = 'fd', 
         density = True)
plt.plot(x, stats.expon.pdf(x, *fit),
         label='exponential fit t={:.2f}'.format(fit[1]))
plt.xlabel('Time (s)')
plt.ylabel('Probability density')
plt.legend()
plt.show()

#%% Export lifetimes to Excel
import pandas as pd

# Prepare the DataFrame for export
export_data = pulses[['trajectory', 'dur']].copy()
export_data.rename(columns={'trajectory': 'Trajectory', 'dur': 'Lifetime (s)'}, inplace=True)

# Define the output path
output_file = os.path.join(data['folder'], f"{data['filename']}_trajectories_lifetimes.xlsx")

# Save to Excel
export_data.to_excel(output_file, index=False)

print(f"Excel file created: {output_file}")

#%% Export mean lifetimes per trajectory to Excel
import pandas as pd

# Calculate mean lifetime for each trajectory
mean_lifetimes = pulses.groupby('trajectory')['dur'].mean().reset_index()
mean_lifetimes.rename(columns={'trajectory': 'Trajectory', 'dur': 'Mean Lifetime (s)'}, inplace=True)

# Define the output path
output_file = os.path.join(data['folder'], f"{data['filename']}_mean_trajectories_lifetimes.xlsx")

# Save to Excel
mean_lifetimes.to_excel(output_file, index=False)

print(f"Excel file with mean lifetimes created: {output_file}")

num_trajectories = mean_lifetimes.shape[0]
print(f"Number of trajectories in the dataset: {num_trajectories}")

#%%

# Define the number of bins
num_bins = 20  # Adjust this number as needed for more bins

# Plot the histogram with increased bins
plt.figure()
plt.title('Pol Lifetime')
plt.hist(
    pulses['dur'], 
    bins=num_bins,  # Use the specified number of bins
    edgecolor='black', 
    density=True, 
    color='gray')
plt.xlabel('Time (s)')
plt.ylabel('Probability density')
plt.show()

# Save the figure as an SVG
output_path = os.path.join(data['folder'], f"{data['filename']}_pol_lifetime_histogram.svg")
plt.savefig(output_path, format='svg')

plt.show()

print(f"Histogram saved as SVG: {output_path}")

#%%

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

# Define the number of bins
num_bins = 20  # Adjust this number as needed for more bins

# Plot the histogram with increased bins
plt.figure()

# Calculate histogram data
counts, bin_edges = np.histogram(pulses['dur'], bins=num_bins, density=True)
bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2  # Calculate the center of each bin

# Plot the histogram
plt.hist(pulses['dur'], bins=num_bins, edgecolor='black', density=True, color='gray', alpha=0.7)

# Define a and b for the exponential fit (from MATLAB)
a = 110
b = -0.02541

# Generate the fitted exponential curve using the defined parameters
x_fit = np.linspace(min(pulses['dur']), max(pulses['dur']), 1000)  # X-values for the fit curve
y_fit = a * np.exp(b * x_fit)  # Exponential model with a and b

# Plot the exponential fit on top of the histogram
plt.plot(x_fit, y_fit, 'r-', label=f'Exponential fit\n a={a}, b={b}')

plt.xlabel('Time (s)')
plt.ylabel('Probability density')
plt.title('Pol Lifetime with Exponential Fit')
plt.legend()

# Save the figure as an SVG
output_path = os.path.join(data['folder'], f"{data['filename']}_pol_lifetime_histogram_with_fit.svg")
plt.savefig(output_path, format='svg')

# Show the plot
plt.show()

print(f"Histogram and exponential fit saved as SVG: {output_path}")


#%%

avgTraj = data['integration'].groupby('slice')['Intensity'].mean()
plt.figure()
plt.plot(avgTraj)
plt.title('Average Trajectory')
plt.xlabel('Time (s)')
plt.ylabel('Intensity')
plt.show()

#%%

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import stats
# from scipy.optimize import curve_fit  # Import curve_fit from scipy.optimize
# from scipy.stats import poisson

# # Count binding events per trajectory
# binding_events = pulses.groupby('trajectory')['dur'].count()

# # Get the counts of binding events
# counts = binding_events.value_counts().sort_index()

# # Create a range for the x-axis based on observed counts
# x = counts.index.values

# # Histogram of binding events
# hist, bin_edges = np.histogram(binding_events, bins=np.arange(0.5, 15.5, 1), density=True)

# # Calculate the bin centers
# bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

# # Fit a Poisson distribution to the histogram data
# def poisson_fit_func(x, lambda_poisson):
#     return poisson.pmf(x, lambda_poisson)

# # Use curve_fit to estimate the best fit for λ
# popt, pcov = curve_fit(poisson_fit_func, bin_centers, hist, p0=[1])  # Use curve_fit from scipy.optimize
# lambda_poisson_fit = popt[0]

# # Calculate the standard deviation
# std_dev = np.sqrt(lambda_poisson_fit)

# # Create a range for x based on observed counts
# x_range = np.arange(0, counts.index.max() + 1)

# # Calculate the Poisson PMF using the fitted λ
# poisson_fit = poisson.pmf(x_range, lambda_poisson_fit)

# # Plot the histogram of binding events
# plt.figure()
# plt.title('Number of Pol Binding Events on NC Template')
# plt.hist(binding_events[binding_events >= 1],
#          edgecolor='black',
#          bins=np.arange(0.5, 15.5, 1), 
#          density=True,
#          alpha=1,
#          align='mid',
#          color='orange')

# # Set x-ticks with a specified range to ensure visibility
# plt.xticks(np.arange(0, 16, 1))  # Adjusted to cover from 0 to 15
# plt.xlabel('Binding events')
# plt.ylabel('Probability density')

# # Overlay the fitted Poisson PMF
# plt.plot(x_range, poisson_fit, 'o-', color='red')

# # Display the fitted λ and standard deviation on the graph
# plt.text(10, 0.15, f'λ = {lambda_poisson_fit:.2f} ± {std_dev:.2f}', fontsize=12, color='black', 
#           bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.5'))

# plt.show()

# print("Binding events:", binding_events)
# print("Fitted λ (Poisson):", lambda_poisson_fit)
# print("Standard Deviation (Poisson):", std_dev)

