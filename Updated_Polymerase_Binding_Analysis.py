# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
from tirf_tools import io, corrections, PeakFitter, integrator, colocalizer
from tirf_tools import hmm_tools as ht
import tqdm
from tirf_tools  import hmm_plots as hp
from matplotlib import pyplot as plt
import napari
import numpy as np

#%% load datasets

data = io.load_image()

#%% correct datasets

corrections.correct_data(data, 2, 422)

#%% drift correct datasets

corrections.drift_correct(data,key='corr_data')

#%% make projection of DNA dataset

data[0]['proj'] = np.mean(data[0]['corr_data'], axis = 0)

#%% make projection of pol dataset

#data[1]['proj'] = np.mean(data[1]['corr_data'], axis = 0)

# Make projection of pol dataset using only the first 800 frames

num_frames = 800  # Customize this value as needed
data[1]['proj'] = np.mean(data[1]['corr_data'][:num_frames], axis=0)

#%% find dna peaks

peaksDNA = PeakFitter.peak_finder(data[0]['proj'], roi=[10,10,512,512],
                       threshold_rel= 0.05,
                      min_dist = 2)

#%% find pol peaks

peaksPol = PeakFitter.peak_finder(data[1]['proj'], roi=[10,10,512,512],
                       threshold_rel= 0.028,
                       min_dist = 2)

#%%

coloc_peaks = colocalizer.coloc_n_peaks([peaksDNA,peaksPol], threshold=3)

#%%

v = napari.Viewer()

#%%

v.add_image(data[0]['corr_data'], name = data[0]['filename'], colormap = 'green')
v.add_image(data[1]['corr_data'], name = data[1]['filename'], colormap = 'magenta')
# v.add_image(data[1]['drift_corr'], name = data[1]['filename'], colormap = 'green')

#%%

v.add_image(data[0]['proj'], name = 'DNA mean', colormap = 'green')
v.add_image(data[1]['proj'], name = 'Pol mean', colormap = 'magenta')

#%%

v.add_points(peaksDNA, name = 'DNA peaks', face_color = 'transparent', edge_color = 'green')
v.add_points(peaksPol, name = 'Pol peaks', face_color = 'transparent', edge_color = 'magenta')

#%%

v.add_points(coloc_peaks[['0_x','0_y']], name = 'coloc peaks', face_color = 'transparent', edge_color = 'yellow')

#%%

inner_radius = 2
outer_radius = 3
#data[1]['integration'] = integrator.integrate_trajectories(data[1]['corr_data'], peaksPol, inner_radius, outer_radius)

data[1]['integration'] = integrator.integrate_trajectories(data[1]['corr_data'], np.array(coloc_peaks[['1_x','1_y']]), inner_radius, outer_radius)

#%% hmm analysis Part1: calibration and filtering

time_conversion = 0.4
#calibrate frames to seconds
ht.calibrate_dataset(data[1]['integration'], 
                     time_conversion, 
                     traj_column = 'trajectory', 
                     time_column = 'slice')

#%% visualizing the intensity distrubition

plt.figure()
plt.hist(data[1]['integration']['Intensity'], bins='fd', edgecolor='black')
plt.title('Intensity Distribution of Integrated Spots')
plt.xlabel('Intensity (arb units)')
plt.ylabel('Frequency')
plt.show()

#%%

# filter out garbage such as big bright aggregates    , as well as trajectories with 
# very negatve intensities

# intensity_threshold = 1000000
# filtered, fig_filter = ht.filter_garbage(data[1]['integration'], 
#                                   intensity_column = 'Intensity',
#                                   traj_column = 'trajectory', 
#                                   threshold =intensity_threshold , 
#                                   plot = True)
# ax = fig_filter.get_axes()
# ax[0].ticklabel_format(axis="x", style="sci", scilimits=(0,0))
# ax[1].ticklabel_format(axis="x", style="sci", scilimits=(0,0))
# title = ax[1].title.get_text()
# new_title = title + '\n theshold: mean + {}*std'.format(intensity_threshold)
# ax[1].set_title(new_title)
# print(f"Number of spots after intensity filtering: {len(filtered['trajectory'].unique())}")

# don't filter anything

filtered = data[1]['integration']
num_spots = len(filtered['trajectory'].unique())
print(f"Number of spots left: {num_spots}")

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

num_trajectories = len(filtered['trajectory'].unique())

print(f"Number of trajectories left: {num_trajectories}")
#%%

print('initializing HMM')

SNR_threshold = 0.0 #distance between on and off distributions in units of standard deviation
min_events = 1      #minimum number of binding events

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

num_trajectories_after_binding_event_filter = len(refiltered['trajectory'].unique())
print(f"Number of trajectories after filtering by binding events: {num_trajectories_after_binding_event_filter}")

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

newpath = data[1]['folder']+'/' + data[1]['filename']+'_all_trajectory/' 
if not os.path.exists(newpath):
    print(newpath)
    os.makedirs(newpath)

#%%

plot_color = 'blue'  # Change this to your desired plot color
fit_color = 'magenta'    # Change this to your desired fit color

for name, group in tqdm.tqdm(refiltered.groupby('trajectory')):
    fig, ax = plt.subplots()
    
    # Plot the intensity and the HMM fit with custom colors
    ax.plot(group['seconds'], group['Intensity'], color=plot_color, label='Intensity')
    ax.plot(group['seconds'], group['mapped_fit'], color=fit_color, label='HMM Fit')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Intensity')
    ax.legend()

    # Save the plot as SVG
    fig.savefig(newpath + 'trajectory_' + str(name) + '.svg', format='svg')

    plt.close(fig)
    
    
#%%

from scipy.stats import expon

x = np.arange(0,100,0.1)

plt.figure()
plt.title('NC Pol Lifetime')
plt.hist(pulses['dur'], 
         edgecolor = 'black',
         bins = 'fd', 
         density = True,
         alpha=1,
         color='orange')

#plt.plot(x,stats.expon.pdf(x,*fit),
         #label='Pol lifetime={:.2f}'.format(fit[1]))

fit = expon.fit(pulses['dur'], floc=0)
x = np.linspace(0, 100, 100)
plt.plot(x, expon.pdf(x, *fit), color='red', label=f'Pol δ on NC Template T={fit[1]:.2f}', lw=2)

plt.xlabel('Time (s)')
plt.ylabel('Probability density')
plt.legend()
plt.show()
# plt.savefig(newpath+'figure.svg')

#%% average trajectory

avgTraj = data[1]['integration'].groupby('slice')['Intensity'].mean()
plt.figure()
plt.plot(avgTraj, color='royalblue')
plt.title('Average Trajectory on GQ Template')
plt.xlabel('Time (s)')
plt.ylabel('Intensity')
plt.show()

#%% binding events histogram

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit  # Import curve_fit from scipy.optimize
from scipy.stats import poisson

# Count binding events per trajectory
binding_events = pulses.groupby('trajectory')['dur'].count()

# Get the counts of binding events
counts = binding_events.value_counts().sort_index()

# Create a range for the x-axis based on observed counts
x = counts.index.values

# Histogram of binding events
hist, bin_edges = np.histogram(binding_events, bins=np.arange(0.5, 15.5, 1), density=True)

# Calculate the bin centers
bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

# Fit a Poisson distribution to the histogram data
def poisson_fit_func(x, lambda_poisson):
    return poisson.pmf(x, lambda_poisson)

# Use curve_fit to estimate the best fit for λ
popt, pcov = curve_fit(poisson_fit_func, bin_centers, hist, p0=[1])  # Use curve_fit from scipy.optimize
lambda_poisson_fit = popt[0]

# Calculate the standard deviation
std_dev = np.sqrt(lambda_poisson_fit)

# Create a range for x based on observed counts
x_range = np.arange(0, counts.index.max() + 1)

# Calculate the Poisson PMF using the fitted λ
poisson_fit = poisson.pmf(x_range, lambda_poisson_fit)

# Plot the histogram of binding events
plt.figure()
plt.title('Number of Pol Binding Events on NC Template')
plt.hist(binding_events[binding_events >= 1],
         edgecolor='black',
         bins=np.arange(0.5, 15.5, 1), 
         density=True,
         alpha=1,
         align='mid',
         color='orange')

# Set x-ticks with a specified range to ensure visibility
plt.xticks(np.arange(0, 16, 1))  # Adjusted to cover from 0 to 15
plt.xlabel('Binding events')
plt.ylabel('Probability density')

# Overlay the fitted Poisson PMF
plt.plot(x_range, poisson_fit, 'o-', color='red')

# Display the fitted λ and standard deviation on the graph
plt.text(10, 0.15, f'λ = {lambda_poisson_fit:.2f} ± {std_dev:.2f}', fontsize=12, color='black', 
          bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.5'))

plt.show()

print("Binding events:", binding_events)
print("Fitted λ (Poisson):", lambda_poisson_fit)
print("Standard Deviation (Poisson):", std_dev)


#%% all lifetimes and binding events

import pandas as pd

# Create a DataFrame for lifetimes
lifetime_df = pd.DataFrame(pulses[['trajectory', 'dur']])
lifetime_df.rename(columns={'dur': 'lifetime'}, inplace=True)

# Create a DataFrame for the number of binding events per trajectory
binding_events_df = pd.DataFrame(binding_events).reset_index()
binding_events_df.columns = ['trajectory', 'binding_events']

# Merge the two DataFrames on the 'trajectory' column
merged_df = pd.merge(lifetime_df, binding_events_df, on='trajectory', how='left')

# Export to Excel
output_path = os.path.join(newpath, 'NC_Lifetime&binding_events.xlsx')
merged_df.to_excel(output_path, index=False)

print(f"Data exported to {output_path}")

#%% mean lifetimes and binding events

import pandas as pd

# Calculate the mean lifetime for each trajectory
lifetime_df = pulses.groupby('trajectory')['dur'].mean().reset_index()
lifetime_df.rename(columns={'dur': 'mean_lifetime'}, inplace=True)

# Create DataFrame for the number of binding events per trajectory
binding_events_df = pd.DataFrame(binding_events).reset_index()
binding_events_df.columns = ['trajectory', 'binding_events']

# Merge the two DataFrames on 'trajectory'
merged_df = pd.merge(lifetime_df, binding_events_df, on='trajectory', how='left')

# Export to Excel
output_path = os.path.join(newpath, 'Mean_NC_lifetime&binding_events.xlsx')
merged_df.to_excel(output_path, index=False)

print(f"Data exported to {output_path}")

#%% 

import matplotlib.pyplot as plt
from scipy.stats import expon

# Compute the mean lifetime for each trajectory (already computed as `lifetime_df`)
lifetime_df = pulses.groupby('trajectory')['dur'].mean().reset_index()
lifetime_df.rename(columns={'dur': 'mean_lifetime'}, inplace=True)

# Plotting the histogram of mean lifetimes
plt.figure()
plt.title('NC Pol Mean Lifetime Histogram')
plt.hist(lifetime_df['mean_lifetime'], 
         edgecolor='black', 
         bins='fd', 
         density=True, 
         alpha=1, 
         color='orange')

# Fit an exponential distribution to the mean lifetime data
fit = expon.fit(lifetime_df['mean_lifetime'], floc=0)
x = np.linspace(0, 100, 100)
plt.plot(x, expon.pdf(x, *fit), color='red', label=f'Pol δ on NC lifetime fit: {fit[1]:.2f}', lw=2)

# Adding labels and legend
plt.xlabel('Time (s)')
plt.ylabel('Probability density')
plt.legend()
plt.show()